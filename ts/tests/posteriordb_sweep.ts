// How much of posteriordb this runtime can take, and what stops the rest.
//
//   git clone --depth 1 https://github.com/stan-dev/posteriordb
//   make posteriordb PDB=../posteriordb
//
// posteriordb is stan-dev's collection of real posterior inference problems —
// a Stan model, its data, and for some of them a reference posterior from a
// long run. It is the honest answer to "how much of Stan is this subset",
// because the models in it are ones people actually wrote.
//
// Each posterior is loaded in a subprocess: a model that exhausts memory or
// never finishes should cost one row, not the run.

import { execFile } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { readFile, readdir, writeFile } from "node:fs/promises";
import { resolve, join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const run = promisify(execFile);
const here = dirname(fileURLToPath(import.meta.url));

// --- the subprocess half: load one posterior, report how far it got ---------
if (process.argv[2] === "--one") {
  const [stanPath, dataPath] = process.argv.slice(3);
  const out: Record<string, unknown> = { stage: "init" };
  try {
    const { default: init, StanModel } = await import("../index.js");
    await init({ module_or_path: await readFile(resolve(here, "..", "pkg", "stanwasm_bg.wasm")) });

    out.stage = "load";
    const m = new StanModel(await readFile(stanPath, "utf8"), await readFile(dataPath, "utf8"));
    out.nParams = m.n_params;

    out.stage = "log_prob";
    const r = m.logProbGrad(new Float64Array(m.n_params).fill(0.1));
    out.finite = [...r].every(Number.isFinite);

    out.stage = "compile";
    out.wasmBytes = m.compileToWasm().length;
    out.stage = "ok";
  } catch (e) {
    out.error = String((e as Error).message ?? e).replace(/\s+/g, " ").slice(0, 200);
  }
  process.stdout.write(JSON.stringify(out));
  process.exit(0);
}

// --- the driver ------------------------------------------------------------
const pdb = resolve(process.cwd(), process.argv[2] ?? "../posteriordb");
const db = join(pdb, "posterior_database");
if (!existsSync(join(db, "posteriors"))) {
  console.error(`no posteriordb at ${pdb}; pass its path or set PDB=`);
  process.exit(2);
}
const cache = join(pdb, ".unzipped");
mkdirSync(cache, { recursive: true });

type Row = {
  posterior: string; model: string; ref: boolean;
  stage: string; error?: string; nParams?: number; finite?: boolean; wasmBytes?: number;
};

const names = (await readdir(join(db, "posteriors"))).filter((f) => f.endsWith(".json"));
const rows: Row[] = [];
for (const [i, f] of names.entries()) {
  const meta = JSON.parse(await readFile(join(db, "posteriors", f), "utf8"));
  const data = join(cache, `${meta.data_name}.json`);
  if (!existsSync(data)) {
    await run("unzip", ["-o", "-q", join(db, "data", "data", `${meta.data_name}.json.zip`),
                        "-d", cache]);
  }
  let r: Partial<Row>;
  try {
    const { stdout } = await run(
      process.execPath,
      ["--experimental-strip-types", fileURLToPath(import.meta.url), "--one",
       join(db, "models", "stan", `${meta.model_name}.stan`), data],
      { timeout: 120_000, maxBuffer: 1 << 24 },
    );
    r = JSON.parse(stdout);
  } catch (e) {
    const err = e as Error & { killed?: boolean };
    r = { stage: err.killed ? "timeout" : "crash",
          error: String(err.message).replace(/\s+/g, " ").slice(0, 200) };
  }
  rows.push({ posterior: meta.name, model: meta.model_name,
              ref: !!meta.reference_posterior_name, ...r } as Row);
  process.stderr.write(`\r${i + 1}/${names.length} ${meta.name.slice(0, 46).padEnd(46)}`);
}
process.stderr.write("\n");

// The error text is what the runtime chose to say; group by what it blames.
const kind = (r: Row) => {
  if (r.stage === "ok") return "ok";
  const e = (r.error ?? "").replace(/^Error:\s*/, "");
  for (const [re, name] of [
    [/expected RBrack, got Comma/, "index or declare with more than one dimension"],
    [/unrecognized character `'`/, "transpose"],
    [/unexpected token in expression: LBrace/, "an array literal — `{1, 2, 3}`"],
    [/unknown distribution/, "a distribution"],
    [/unknown function/, "a function"],
    [/no unconstrained dimensions|UnsupportedConstraint/, "a constraint transform"],
    [/^parse:/, "other syntax"],
  ] as const) {
    if (re.test(e)) return name;
  }
  return r.stage === "timeout" ? "timeout" : "other";
};

const byKind = new Map<string, Row[]>();
for (const r of rows) byKind.set(kind(r), [...(byKind.get(kind(r)) ?? []), r]);
const ok = byKind.get("ok") ?? [];

await writeFile(join(here, "..", "..", "target", "posteriordb-sweep.json"),
                JSON.stringify(rows, null, 1));

console.log(`\n${ok.length}/${rows.length} posteriors load, evaluate a gradient, and compile\n`);
console.log("what stops it".padEnd(44) + "count  share");
console.log("-".repeat(58));
for (const [k, v] of [...byKind].sort((a, b) => b[1].length - a[1].length)) {
  console.log(`${k.padEnd(44)}${String(v.length).padStart(5)}  ${(100 * v.length / rows.length).toFixed(0).padStart(4)}%`);
}
const ref = rows.filter((r) => r.ref);
console.log(`\nof the ${ref.length} with a reference posterior, ${ref.filter((r) => r.stage === "ok").length} are usable`);

const nonFinite = ok.filter((r) => !r.finite);
if (nonFinite.length) {
  console.log(`\nload but the log density is not finite at 0.1: ${nonFinite.map((r) => r.posterior).join(", ")}`);
}

for (const [k, v] of byKind) {
  if (k === "ok") continue;
  const seen = new Map<string, number>();
  for (const r of v) {
    const key = (r.error ?? "").slice(0, 88);
    seen.set(key, (seen.get(key) ?? 0) + 1);
  }
  console.log(`\n[${k}] ${v.length}`);
  for (const [msg, n] of [...seen].sort((a, b) => b[1] - a[1]).slice(0, 10)) {
    console.log(`  ${String(n).padStart(3)}x ${msg}`);
  }
}
