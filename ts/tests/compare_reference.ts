// Does sampling here land on the posterior the reference says it should?
// `posteriordb_sweep.ts` answers "does it load"; this answers what comes after,
// since agreeing on the gradient does not say the draws land in the same place.
//
//   git clone --depth 1 https://github.com/stan-dev/posteriordb
//   make compare-reference PDB=../posteriordb
//
// Per parameter, `|our mean - reference mean| / reference sd`, in units of the
// posterior's own spread. A gap is Monte Carlo error until shown otherwise, so
// read it against the draw count: what sat past 0.2 sd at 300 draws fell to
// 0.02-0.04 at 4000. `eight_schools_centered` is geometry, not this runtime —
// the non-centred form of the same model comes to 0.02 sd against its 0.13.
//
// Each posterior runs in a subprocess: one that will not finish costs a row.

import { execFile } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { readFile, readdir, writeFile } from "node:fs/promises";
import { resolve, join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const run = promisify(execFile);
const here = dirname(fileURLToPath(import.meta.url));

const WARMUP = Number(process.env.WARMUP ?? 1000);
const DRAWS = Number(process.env.DRAWS ?? 1000);
const SEED = BigInt(process.env.SEED ?? 42);
/** Substring filter, for re-running one posterior with more draws. */
const ONLY = process.env.ONLY ?? null;

/** Mean and sd of a flat array. */
function moments(xs: number[]): [number, number] {
  const n = xs.length;
  const mean = xs.reduce((a, b) => a + b, 0) / n;
  const varr = xs.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1);
  return [mean, Math.sqrt(varr)];
}

// --- the subprocess half: sample one posterior, report per-parameter means --
if (process.argv[2] === "--one") {
  const [stanPath, dataPath] = process.argv.slice(3);
  const out: Record<string, unknown> = { stage: "init" };
  try {
    const { default: init, StanModel } = await import("../index.js");
    await init({ module_or_path: await readFile(resolve(here, "..", "pkg", "stanwasm_bg.wasm")) });

    out.stage = "load";
    const m = new StanModel(await readFile(stanPath, "utf8"), await readFile(dataPath, "utf8"));

    out.stage = "sample";
    // A start where some parameter has no slope is refused; that is the model.
    const start = m.randomInit(SEED);
    const raw = m.sample(start, WARMUP, DRAWS, SEED);
    const n = m.n_params;

    out.stage = "constrain";
    const names = m.paramNames();
    const cols: number[][] = names.map(() => []);
    for (let d = WARMUP; d < WARMUP + DRAWS; d++) {
      const c = m.constrainDraw(raw.slice(d * n, (d + 1) * n));
      for (let j = 0; j < names.length; j++) cols[j].push(c[j]);
    }
    out.names = names;
    out.mean = cols.map((c) => moments(c)[0]);
    out.sd = cols.map((c) => moments(c)[1]);
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
const refCache = join(pdb, ".refdraws");
mkdirSync(cache, { recursive: true });
mkdirSync(refCache, { recursive: true });

/** Reference draws: 10 chains, each a map of parameter name to its draws. */
async function referenceMoments(name: string) {
  const path = join(refCache, `${name}.json`);
  if (!existsSync(path)) {
    await run("unzip", ["-o", "-q",
      join(db, "reference_posteriors", "draws", "draws", `${name}.json.zip`), "-d", refCache]);
  }
  const chains = JSON.parse(await readFile(path, "utf8")) as Record<string, number[]>[];
  const out = new Map<string, [number, number]>();
  for (const p of Object.keys(chains[0])) {
    out.set(p, moments(chains.flatMap((c) => c[p])));
  }
  return out;
}

type Row = {
  posterior: string;
  stage: string;
  error?: string;
  /** Largest |our mean - reference mean| / reference sd, and where. */
  worst?: number;
  worstAt?: string;
  checked?: number;
  /** Parameters the reference has that we did not produce, or vice versa. */
  unmatched?: number;
  seconds?: number;
};

const names = (await readdir(join(db, "posteriors"))).filter((f) => f.endsWith(".json"));
const rows: Row[] = [];
for (const f of names) {
  const meta = JSON.parse(await readFile(join(db, "posteriors", f), "utf8"));
  if (!meta.reference_posterior_name) continue;
  if (ONLY && !meta.name.includes(ONLY)) continue;
  const i = rows.length + 1;
  process.stderr.write(`\r${i} ${meta.name.slice(0, 46).padEnd(46)}`);

  const data = join(cache, `${meta.data_name}.json`);
  if (!existsSync(data)) {
    await run("unzip", ["-o", "-q",
      join(db, "data", "data", `${meta.data_name}.json.zip`), "-d", cache]);
  }

  const t0 = Date.now();
  let r: Partial<Row>;
  try {
    const { stdout } = await run(
      process.execPath,
      ["--experimental-strip-types", fileURLToPath(import.meta.url), "--one",
        join(db, "models", "stan", `${meta.model_name}.stan`), data],
      { timeout: 600_000, maxBuffer: 1 << 26 },
    );
    const got = JSON.parse(stdout) as {
      stage: string; error?: string; names?: string[]; mean?: number[]; sd?: number[];
    };
    if (got.stage !== "ok") {
      r = { stage: got.stage, error: got.error };
    } else {
      const ref = await referenceMoments(meta.reference_posterior_name);
      let worst = 0, worstAt = "", checked = 0, unmatched = 0;
      for (const [j, p] of got.names!.entries()) {
        const have = ref.get(p);
        if (!have) { unmatched++; continue; }
        const [rm, rsd] = have;
        // A parameter pinned to a constant has no spread to scale by.
        if (!(rsd > 0)) continue;
        const gap = Math.abs(got.mean![j] - rm) / rsd;
        checked++;
        if (gap > worst) { worst = gap; worstAt = p; }
      }
      unmatched += [...ref.keys()].filter((p) => !got.names!.includes(p)).length;
      r = { stage: checked ? "ok" : "no-overlap", worst, worstAt, checked, unmatched };
    }
  } catch (e) {
    const err = e as Error & { killed?: boolean };
    r = { stage: err.killed ? "timeout" : "crash",
          error: String(err.message).replace(/\s+/g, " ").slice(0, 160) };
  }
  rows.push({ posterior: meta.name, seconds: (Date.now() - t0) / 1000, ...r } as Row);
}
process.stderr.write("\n");

await writeFile(join(here, "..", "..", "target", "reference-comparison.json"),
                JSON.stringify(rows, null, 1));

const ran = rows.filter((r) => r.stage === "ok");
const band = (r: Row) => (r.worst! < 0.2 ? "within 0.2 sd" : r.worst! < 0.5 ? "0.2 - 0.5 sd" : "past 0.5 sd");
const bands = new Map<string, Row[]>();
for (const r of ran) bands.set(band(r), [...(bands.get(band(r)) ?? []), r]);

console.log(`\n${ran.length}/${rows.length} posteriors with a reference posterior sampled`);
console.log(`${WARMUP} warmup + ${DRAWS} draws, seed ${SEED}; gap is |mean - reference| / reference sd\n`);
for (const k of ["within 0.2 sd", "0.2 - 0.5 sd", "past 0.5 sd"]) {
  const v = bands.get(k) ?? [];
  console.log(`${k.padEnd(20)}${String(v.length).padStart(4)}`);
}

const bad = ran.filter((r) => r.worst! >= 0.2).sort((a, b) => b.worst! - a.worst!);
if (bad.length) {
  console.log("\nfurthest from the reference:");
  for (const r of bad.slice(0, 15)) {
    console.log(`  ${r.worst!.toFixed(2).padStart(6)} sd  ${r.posterior} (${r.worstAt})`);
  }
}

const missed = rows.filter((r) => r.stage !== "ok");
if (missed.length) {
  console.log("\ndid not produce a comparison:");
  for (const r of missed) console.log(`  ${r.posterior}: ${r.stage} ${r.error ?? ""}`.trimEnd());
}

const noOverlap = ran.filter((r) => r.unmatched);
if (noOverlap.length) {
  console.log(`\n${noOverlap.length} posteriors had parameters on only one side (naming, or transformed parameters the reference omits)`);
}
