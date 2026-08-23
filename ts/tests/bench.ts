// Node.js benchmark — runs the Rust wasm32 build under V8.
//
// Times two sampling paths:
//   - StanModel.sample (tape replay; in-wasm)
//   - StanModel.sampleViaAot (AOT model wasm bound via setAotExports)

import { readFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import init, {
  StanModel,
  setAotExports,
  clearAotExports,
  sharedMemory,
} from "../index.js";

const here = dirname(fileURLToPath(import.meta.url));
const wasmPath = resolve(here, "..", "pkg", "stan_wasm_api_bg.wasm");
await init({ module_or_path: await readFile(wasmPath) });

// Math shims for the AOT module's imports.
const lgamma = (x: number): number => {
  let z = x, r = 0;
  while (z < 10) { r -= Math.log(z); z += 1; }
  const zi = 1 / z, zi2 = zi * zi;
  return r + (z - 0.5) * Math.log(z) - z + 0.5 * Math.log(2 * Math.PI) + zi * (1 / 12 + zi2 * (-1 / 360 + zi2 / 1260));
};
const digamma = (x: number): number => {
  let xx = x, r = 0;
  while (xx < 6) { r -= 1 / xx; xx += 1; }
  const inv = 1 / xx, inv2 = inv * inv;
  return r + Math.log(xx) - 0.5 * inv - inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 / 252));
};
const phi = (x: number): number => {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  const cdf = 1 - (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * x * x) * poly;
  return x >= 0 ? cdf : 1 - cdf;
};
const mathImports = { exp: Math.exp, log: Math.log, sin: Math.sin, cos: Math.cos, pow: Math.pow, lgamma, digamma, phi };

interface Case {
  name: string;
  src: string;
  data: Record<string, unknown>;
  init: number[];
}

const CASES: Case[] = [
  {
    name: "linear_regression",
    src: `data { int<lower=0> N; vector[N] x; vector[N] y; }
parameters { real alpha; real beta; real<lower=0> sigma; }
model {
  alpha ~ normal(0, 10); beta ~ normal(0, 10); sigma ~ exponential(1);
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
  alpha ~ normal(0, 5); beta ~ normal(0, 1);
  for (i in 1:N) y[i] ~ poisson(exp(alpha + beta * x[i]));
}`,
    data: { N: 5, x: [0,1,2,3,4], y: [1,2,5,12,30] },
    init: [0, 1],
  },
  {
    name: "eight_schools_ncp",
    src: `data { int<lower=0> J; vector[J] y; vector<lower=0>[J] sigma; }
parameters { real mu; real<lower=0> tau; vector[J] theta_tilde; }
transformed parameters { vector[J] theta = mu + tau * theta_tilde; }
model {
  mu ~ normal(0, 5); tau ~ half_normal(5); theta_tilde ~ normal(0, 1);
  y ~ normal(theta, sigma);
}`,
    data: { J: 8, y: [28, 8, -3, 7, -1, 1, 18, 12], sigma: [15, 10, 16, 11, 9, 11, 10, 18] },
    init: Array(10).fill(0.1),
  },
];

const N_WARMUP = 1000;
const N_DRAWS = 1000;

console.log(`${"case".padEnd(22)} | ${"replay ms".padStart(10)} | ${"AOT ms".padStart(8)} | ${"speedup".padStart(8)}`);
console.log("-".repeat(58));

for (const c of CASES) {
  // Replay path
  const m1 = new StanModel(c.src, JSON.stringify(c.data));
  const init = new Float64Array(c.init);
  const t0 = performance.now();
  m1.sample(init, N_WARMUP, N_DRAWS, 42n);
  const replayMs = performance.now() - t0;

  // AOT path
  const m2 = new StanModel(c.src, JSON.stringify(c.data));
  const aot = await WebAssembly.instantiate(m2.compileToWasm(), {
    stan: { memory: sharedMemory() as WebAssembly.Memory },
    Math: mathImports,
  });
  setAotExports(aot.instance.exports);
  const t1 = performance.now();
  m2.sampleViaAot(init, N_WARMUP, N_DRAWS, 42n);
  const aotMs = performance.now() - t1;
  clearAotExports();

  const speedup = replayMs / aotMs;
  console.log(
    `${c.name.padEnd(22)} | ${replayMs.toFixed(1).padStart(10)} | ${aotMs.toFixed(1).padStart(8)} | ${speedup.toFixed(2).padStart(7)}x`,
  );
}

console.log();
console.log(`replay = StanModel.sample (in-wasm tape replay)`);
console.log(`AOT    = StanModel.sampleViaAot (AOT model wasm via setAotExports)`);
console.log(`runtime = Node.js V8, wasm32, n_warmup=${N_WARMUP}, n_draws=${N_DRAWS}`);
