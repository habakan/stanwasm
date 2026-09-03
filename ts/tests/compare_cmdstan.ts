// Check this implementation's log density and gradients against CmdStan, the
// reference implementation, and time both.
//
//   node --experimental-strip-types tests/bench_gradients.ts 5000 --emit ../target/bench
//   CMDSTAN=~/cmdstan node --experimental-strip-types tests/compare_cmdstan.ts ../target/bench
//
// Compiles each emitted model with CmdStan if it is not built yet, runs its
// `log_prob` method at the same two unconstrained points, and compares.
//
// The two log densities differ by a constant: Stan's `~` drops the normalising
// terms that do not depend on the parameters, and this runtime keeps them.
// That is why two points are checked — a constant offset is the expected
// difference, an offset that moves with the point is a bug.
//
// Timing is static HMC, whose leapfrog count per draw is fixed and where every
// leapfrog is one gradient. A thousand leapfrogs per draw amortises away the
// CSV row each draw writes.

import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { readdir } from "node:fs/promises";
import { resolve, join } from "node:path";
import { homedir } from "node:os";

const bench = resolve(process.cwd(), process.argv[2] ?? "../target/bench");
const cmdstan = resolve((process.env.CMDSTAN ?? "~/cmdstan").replace(/^~/, homedir()));
if (!existsSync(join(cmdstan, "makefile"))) {
  console.error(`no CmdStan at ${cmdstan}; set CMDSTAN=<path to a built CmdStan>`);
  process.exit(2);
}

const STEPSIZE = 0.001;
const INT_TIME = 1;
const DRAWS = 20;
const TARGET_SECONDS = 1.5;

const run = (bin: string, args: string[], cwd?: string) =>
  execFileSync(bin, args, { cwd, encoding: "utf8", maxBuffer: 1 << 28 });

/// Rows of a CmdStan CSV, with its `#` comment lines dropped.
function readCsv(path: string): Record<string, number>[] {
  const lines = readFileSync(path, "utf8").split("\n").filter((l) => l && !l.startsWith("#"));
  const head = lines[0].split(",");
  return lines.slice(1).map((l) => Object.fromEntries(
    l.split(",").map((v, i) => [head[i], Number(v)]),
  ));
}

function logProb(model: string, name: string, tag: string) {
  const out = join(bench, `${name}.lp${tag}.csv`);
  run(model, [
    "log_prob", `unconstrained_params=${join(bench, `${name}.params${tag}.json`)}`,
    "data", `file=${join(bench, `${name}.data.json`)}`,
    "output", `file=${out}`, "sig_figs=17",
  ]);
  const row = readCsv(out)[0];
  const { lp__, ...rest } = row;
  return { lp: lp__, grads: Object.values(rest) };
}

function timeOnce(model: string, name: string, draws: number) {
  const out = join(bench, `${name}.sample.csv`);
  run(model, [
    "sample", "num_warmup=0", `num_samples=${draws}`,
    "algorithm=hmc", "engine=static", `int_time=${INT_TIME}`, `stepsize=${STEPSIZE}`,
    "metric=unit_e", "adapt", "engaged=0",
    "data", `file=${join(bench, `${name}.data.json`)}`,
    "init=0", "output", `file=${out}`, "refresh=0",
  ]);
  const text = readFileSync(out, "utf8");
  const secs = Number(/([\d.]+) seconds \(Sampling\)/.exec(text)![1]);
  const leapfrogs = readCsv(out)
    .reduce((a, r) => a + Math.round(r["int_time__"] / r["stepsize__"]), 0);
  return { secs, leapfrogs };
}

function timeGradient(model: string, name: string) {
  let { secs, leapfrogs } = timeOnce(model, name, DRAWS);
  const scale = Math.max(1, Math.min(200, Math.round(TARGET_SECONDS / Math.max(secs, 0.01))));
  if (scale > 1) ({ secs, leapfrogs } = timeOnce(model, name, DRAWS * scale));
  return (secs * 1e6) / leapfrogs;
}

const ours = JSON.parse(readFileSync(join(bench, "stanwasm.json"), "utf8")) as
  Record<string, { aot: number }>;
const names = (await readdir(bench)).filter((f) => f.endsWith(".stan")).map((f) => f.slice(0, -5));

console.log(`\n${names.length} models, µs per gradient\n`);
console.log("model         | grad match | stanwasm µs | cmdstan µs |  ratio");
console.log("-".repeat(66));
let worst = 0;
let wins = 0;
for (const name of names.sort()) {
  const model = join(bench, name);
  if (!existsSync(model)) run("make", [model], cmdstan);
  let rel = 0;
  const offsets: number[] = [];
  for (const tag of ["", "2"]) {
    const ref = JSON.parse(readFileSync(join(bench, `${name}.ref${tag}.json`), "utf8"));
    const got = logProb(model, name, tag);
    offsets.push(ref.lp - got.lp);
    ref.grads.forEach((a: number, i: number) => {
      rel = Math.max(rel, Math.abs(a - got.grads[i]) / (1 + Math.abs(got.grads[i])));
    });
  }
  const drift = Math.abs(offsets[0] - offsets[1]) / (1 + Math.abs(offsets[0]));
  const us = timeGradient(model, name);
  worst = Math.max(worst, rel);
  if (us > ours[name].aot) wins += 1;
  console.log(
    [
      name.padEnd(13),
      rel.toExponential(2).padStart(10),
      ours[name].aot.toFixed(2).padStart(11),
      us.toFixed(2).padStart(10),
      (us / ours[name].aot).toFixed(2).padStart(6) + "x",
    ].join(" | ") + (drift < 1e-9 ? "" : `  LOG DENSITY OFFSET MOVED BY ${drift.toExponential(2)}`),
  );
}
console.log(`\nfaster on ${wins}/${names.length}`);
console.log(`worst relative gradient difference: ${worst.toExponential(2)}`);
