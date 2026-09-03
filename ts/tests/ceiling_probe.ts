// Times the hand-written variants from `cargo run -p stanwasm-codegen
// --example ceiling_probe` against the real AOT module, on the matrix-product
// model whose gradient the scalar tape records one node at a time.
//
//   cargo run -p stanwasm-codegen --example ceiling_probe -- 5000 4 target/ceiling
//   cd ts && node --experimental-strip-types tests/ceiling_probe.ts ../target/ceiling
//
// Every variant's log_prob and gradients are checked against the runtime's own
// before it is timed: a variant that computes something else is not a ceiling.

import { readFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import init, { StanModel } from "../index.js";

const here = dirname(fileURLToPath(import.meta.url));
await init({ module_or_path: await readFile(resolve(here, "..", "pkg", "stanwasm_bg.wasm")) });

const dir = resolve(process.cwd(), process.argv[2] ?? "../target/ceiling");
const meta = JSON.parse(await readFile(resolve(dir, "meta.json"), "utf8"));
const { n: N, k: K } = meta;

const SRC = `data { int<lower=0> N; int<lower=0> K; matrix[N,K] X; vector[N] y; }
parameters { vector[K] beta; real<lower=0> sigma; }
model {
  beta ~ normal(0, 1);
  sigma ~ exponential(1);
  y ~ normal(X * beta, sigma);
}`;

let seed = 12345;
const rnd = () => {
  seed = (seed * 1664525 + 1013904223) >>> 0;
  return (seed / 4294967296) * 2 - 1;
};
const truth = [0.5, -0.3, 0.8, 0.1];
const X: number[][] = [], y: number[] = [];
for (let n = 0; n < N; n++) {
  const row: number[] = [];
  let mu = 0;
  for (let k = 0; k < K; k++) { const v = rnd(); row.push(v); mu += v * truth[k % 4]; }
  X.push(row);
  y.push(mu + rnd() * 0.2);
}

const m = new StanModel(SRC, JSON.stringify({ N, K, X, y }));
const nP = m.n_params;
const params = new Float64Array(nP).fill(0.1);
params[nP - 1] = -0.5;
const ref = m.logProbGrad(params);

// One memory for every hand-written variant: they agree on where the data is,
// and each rewrites its own scratch on every call.
const mem = new WebAssembly.Memory({ initial: Math.ceil(meta.end / 65536) + 2 });
const view = new Float64Array(mem.buffer);
view.set(params, 0);
for (let k = 0; k < K; k++) for (let n = 0; n < N; n++) view[meta.x_col / 8 + k * N + n] = X[n][k];
for (let n = 0; n < N; n++) for (let k = 0; k < K; k++) view[meta.x_row / 8 + n * K + k] = X[n][k];
view.set(y, meta.y / 8);

const mathImports = { exp: Math.exp, log: Math.log };
const agrees = (lp: number, g: number[]) =>
  Math.abs(lp - ref[0]) < 1e-6 * (1 + Math.abs(ref[0])) &&
  g.every((v, i) => Math.abs(v - ref[i + 1]) < 1e-6 * (1 + Math.abs(ref[i + 1])));

type Cand = { name: string; run: () => void; ok: boolean };
const cands: Cand[] = [];

// The real AOT module, in its own memory with its own scratch.
const bytes = m.compileToWasm();
const scratch = m.aotScratchInit();
const aotMem = new WebAssembly.Memory({
  initial: Math.ceil((nP * 16 + scratch.length * 8) / 65536) + 4,
});
const aot = await WebAssembly.instantiate(bytes, { stan: { memory: aotMem }, Math: mathImports });
const aotView = new Float64Array(aotMem.buffer);
aotView.set(scratch, nP * 2);
aotView.set(params, 0);
const aotLpg = aot.instance.exports.log_prob_grad as
  (p: number, g: number, n: number, s: number) => number;
{
  const lp = aotLpg(0, nP * 8, nP, nP * 16);
  cands.push({
    name: "real AOT",
    run: () => { aotLpg(0, nP * 8, nP, nP * 16); },
    ok: agrees(lp, [...aotView.subarray(nP, 2 * nP)]),
  });
}

for (const name of Object.keys(meta.variants)) {
  const mod = await WebAssembly.instantiate(
    await readFile(resolve(dir, `${name}.wasm`)),
    { stan: { memory: mem }, Math: mathImports },
  );
  const lpg = mod.instance.exports.log_prob_grad as
    (p: number, g: number, n: number, s: number) => number;
  const lp = lpg(0, nP * 8, nP, meta.scratch);
  cands.push({
    name,
    run: () => { lpg(0, nP * 8, nP, meta.scratch); },
    ok: agrees(lp, [...view.subarray(nP, 2 * nP)]),
  });
}

// Interleaved: every variant is measured once per round, in a rotating order,
// so a drift in machine state hits all of them rather than whichever ran last.
const ITERS = 500, ROUNDS = Number(process.env.ROUNDS ?? 100);
const best = new Map(cands.map((c) => [c.name, Infinity]));
for (const c of cands) for (let i = 0; i < 3000; i++) c.run();
for (let r = 0; r < ROUNDS; r++) {
  for (let i = 0; i < cands.length; i++) {
    const c = cands[(i + r) % cands.length];
    const t = performance.now();
    for (let j = 0; j < ITERS; j++) c.run();
    best.set(c.name, Math.min(best.get(c.name)!, ((performance.now() - t) * 1000) / ITERS));
  }
}

const aotUs = best.get("real AOT")!;
console.log(`\nmatrix model, N=${N} K=${K}, ${process.version}, min of ${ROUNDS} interleaved rounds\n`);
console.log("variant       |    µs | vs AOT | gradients");
console.log("-".repeat(48));
for (const c of cands) {
  const us = best.get(c.name)!;
  console.log(
    `${c.name.padEnd(13)} | ${us.toFixed(2).padStart(5)} | ` +
    `${(aotUs / us).toFixed(2).padStart(5)}x | ${c.ok ? "agree" : "MISMATCH"}`,
  );
}
if (cands.some((c) => !c.ok)) process.exit(1);
