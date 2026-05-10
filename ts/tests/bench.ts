// Node.js benchmark — runs the Rust wasm32 build under V8 to give a
// like-for-like comparison with MoonBit `tests/results/benchmark_nutsrs_aot.json`
// (which is also Node.js + V8).
//
// Run: cd ts && npm run bench

import { readFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import init, { StanModel } from "../index.ts";

const here = dirname(fileURLToPath(import.meta.url));
const wasmPath = resolve(here, "..", "pkg", "stan_wasm_api_bg.wasm");
const wasmBytes = await readFile(wasmPath);
await init({ module_or_path: wasmBytes });

interface Case {
  name: string;
  src: string;
  data: Record<string, unknown>;
  init: number[];
}

const CASES: Case[] = [
  {
    name: "linear_regression",
    src: `data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}
parameters { real alpha; real beta; real<lower=0> sigma; }
model {
  alpha ~ normal(0, 10);
  beta  ~ normal(0, 10);
  sigma ~ exponential(1);
  y ~ normal(alpha + beta * x, sigma);
}`,
    data: {
      N: 30,
      x: Array.from({ length: 30 }, (_, i) => -1.5 + i * 0.1),
      y: Array.from({ length: 30 }, (_, i) => -1.3 + i * 0.18),
    },
    init: [0, 1, 0],
  },
  {
    name: "poisson_regression",
    src: `data { int<lower=0> N; vector[N] x; array[N] int y; }
parameters { real alpha; real beta; }
model {
  alpha ~ normal(0, 5);
  beta  ~ normal(0, 1);
  for (i in 1:N) y[i] ~ poisson(exp(alpha + beta * x[i]));
}`,
    data: { N: 5, x: [0,1,2,3,4], y: [1,2,5,12,30] },
    init: [0, 1],
  },
  {
    name: "eight_schools_ncp",
    src: `data {
  int<lower=0> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}
parameters {
  real mu;
  real<lower=0> tau;
  vector[J] theta_tilde;
}
transformed parameters {
  vector[J] theta = mu + tau * theta_tilde;
}
model {
  mu ~ normal(0, 5);
  tau ~ half_normal(5);
  theta_tilde ~ normal(0, 1);
  y ~ normal(theta, sigma);
}`,
    data: {
      J: 8,
      y: [28, 8, -3, 7, -1, 1, 18, 12],
      sigma: [15, 10, 16, 11, 9, 11, 10, 18],
    },
    init: Array(10).fill(0.1),
  },
];

const N_ITERS = 10_000;
const N_WARMUP = 1000;
const N_DRAWS = 1000;

console.log(`${"case".padEnd(22)} | ${"lpg µs".padStart(8)} | ${"sample ms".padStart(10)}`);
console.log("-".repeat(46));

for (const c of CASES) {
  const m = new StanModel(c.src, JSON.stringify(c.data));
  const init = new Float64Array(c.init);

  // warmup
  for (let i = 0; i < N_ITERS / 10; i++) m.logProbGrad(init);
  const t0 = performance.now();
  for (let i = 0; i < N_ITERS; i++) m.logProbGrad(init);
  const lpgUs = ((performance.now() - t0) * 1000) / N_ITERS;

  // re-create model since logProbGrad doesn't consume it but we want a clean state
  const m2 = new StanModel(c.src, JSON.stringify(c.data));
  const ts0 = performance.now();
  m2.sample(init, N_WARMUP, N_DRAWS, 42n);
  const sampleMs = performance.now() - ts0;

  console.log(`${c.name.padEnd(22)} | ${lpgUs.toFixed(2).padStart(8)} | ${sampleMs.toFixed(1).padStart(10)}`);
}

console.log();
console.log(`lpg µs    = StanModel.logProbGrad average over ${N_ITERS} iters`);
console.log(`sample ms = full NUTS, n_warmup=${N_WARMUP}, n_draws=${N_DRAWS}`);
console.log(`runtime   = Node.js V8, wasm32 build`);
