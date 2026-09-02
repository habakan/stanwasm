// Scaling probe (not part of the committed suite): per-gradient cost of the
// replay and AOT paths as N grows. Gradient granularity, not sampling wall
// clock — the two paths can take different NUTS trajectories.
import { readFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import init, { StanModel } from "../index.js";

const here = dirname(fileURLToPath(import.meta.url));
await init({ module_or_path: await readFile(resolve(here, "..", "pkg", "stanwasm_bg.wasm")) });

const lgamma = (x: number): number => {
  let z = x, r = 0;
  while (z < 10) { r -= Math.log(z); z += 1; }
  const zi = 1 / z, zi2 = zi * zi;
  return r + (z - 0.5) * Math.log(z) - z + 0.5 * Math.log(2 * Math.PI)
       + zi * (1 / 12 + zi2 * (-1 / 360 + zi2 / 1260));
};
const digamma = (x: number): number => {
  let xx = x, r = 0;
  while (xx < 6) { r -= 1 / xx; xx += 1; }
  const inv = 1 / xx, inv2 = inv * inv;
  return r + Math.log(xx) - 0.5 * inv - inv2 * (1/12 - inv2 * (1/120 - inv2/252));
};
const phi = (x: number): number => {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const poly = t*(0.319381530 + t*(-0.356563782 + t*(1.781477937 + t*(-1.821255978 + t*1.330274429))));
  const cdf = 1 - (1/Math.sqrt(2*Math.PI)) * Math.exp(-0.5*x*x) * poly;
  return x >= 0 ? cdf : 1 - cdf;
};
const mathImports = { exp: Math.exp, log: Math.log, sin: Math.sin, cos: Math.cos,
                      pow: Math.pow, lgamma, digamma, phi };

const SRC = `data { int<lower=0> N; vector[N] x; vector[N] y; }
parameters { real alpha; real beta; real<lower=0> sigma; }
model {
  alpha ~ normal(0, 10); beta ~ normal(0, 10); sigma ~ exponential(1);
  y ~ normal(alpha + beta * x, sigma);
}`;

const mk = (N: number) => JSON.stringify({
  N,
  x: Array.from({ length: N }, (_, i) => -1.5 + (i * 3) / N),
  y: Array.from({ length: N }, (_, i) => -1.3 + (i * 3.6) / N),
});

const time = (f: () => void, iters: number) => {
  for (let i = 0; i < Math.min(200, iters); i++) f();      // warm the JIT
  const t = performance.now();
  for (let i = 0; i < iters; i++) f();
  return ((performance.now() - t) * 1000) / iters;          // µs/call
};

console.log(
  ["     N", "  wasm KB", " codegen ms", "  inst ms", " replay µs", "    AOT µs", " speedup"]
    .join(" |"));
console.log("-".repeat(76));

for (const N of [10, 100, 500, 1000, 2000, 5000]) {
  const data = mk(N);
  const params = new Float64Array([0.1, 0.9, 0.2]);
  const iters = N >= 2000 ? 300 : 3000;

  const m = new StanModel(SRC, data);
  const nP = m.n_params;
  const replayUs = time(() => { m.logProbGrad(params); }, iters);

  let t = performance.now();
  const bytes = m.compileToWasm();
  const codegenMs = performance.now() - t;

  // params, grads, then the module's primal/adjoint scratch (two f64 per tape
  // node; the tape is roughly 12 nodes per data point here).
  const pages = Math.ceil((2 * (16 * N + 512) * 8) / 65536) + 2;
  const mem = new WebAssembly.Memory({ initial: pages });
  t = performance.now();
  const aot = await WebAssembly.instantiate(bytes, {
    stan: { memory: mem }, Math: mathImports,
  });
  const instMs = performance.now() - t;

  const view = new Float64Array(mem.buffer);
  view.set(params, 0);
  const scratch = m.aotScratchInit();
  view.set(scratch, nP * 2);
  const lpg = aot.instance.exports.log_prob_grad as
    (p: number, g: number, n: number, s: number) => number;
  const gPtr = nP * 8;
  const sPtr = nP * 16;
  const aotUs = time(() => { lpg(0, gPtr, nP, sPtr); }, iters);

  console.log([
    String(N).padStart(6), (bytes.length / 1024).toFixed(1).padStart(9),
    codegenMs.toFixed(1).padStart(11), instMs.toFixed(1).padStart(9),
    replayUs.toFixed(2).padStart(10), aotUs.toFixed(2).padStart(10),
    (replayUs / aotUs).toFixed(2).padStart(7) + "x",
  ].join(" |"));
}
