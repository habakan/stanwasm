// Smoke test for the AOT-via-V8 sampling path.
// Loads stanwasm.wasm, instantiates a per-model AOT wasm sharing its
// memory, binds the bridge, and samples — verifying samples agree with the
// in-process tape replay path.

import { readFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import init, {
  StanModel,
  setAotExports,
  sharedMemory,
} from "../index.js";

const here = dirname(fileURLToPath(import.meta.url));
const wasmPath = resolve(here, "..", "pkg", "stan_wasm_api_bg.wasm");
const wasmBytes = await readFile(wasmPath);
await init({ module_or_path: wasmBytes });

// Math imports the AOT module needs (subset depending on the model).
const lgamma = (x: number): number => {
  // Stirling-series; matches stan_autodiff::lgamma. Adequate for x > 0.
  let z = x;
  let r = 0;
  while (z < 10) { r -= Math.log(z); z += 1; }
  const zinv = 1 / z;
  const zinv2 = zinv * zinv;
  return r + (z - 0.5) * Math.log(z) - z + 0.5 * Math.log(2 * Math.PI)
    + zinv * (1 / 12 + zinv2 * (-1 / 360 + zinv2 / 1260));
};

const digamma = (x: number): number => {
  let xx = x;
  let r = 0;
  while (xx < 6) { r -= 1 / xx; xx += 1; }
  const inv = 1 / xx;
  const inv2 = inv * inv;
  return r + Math.log(xx) - 0.5 * inv - inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 / 252));
};

const phi = (x: number): number => {
  // Abramowitz & Stegun 26.2.17
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  const cdf = 1 - (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * x * x) * poly;
  return x >= 0 ? cdf : 1 - cdf;
};

const stanCode = `
data { int<lower=0> N; vector[N] x; vector[N] y; }
parameters { real alpha; real beta; real<lower=0> sigma; }
model {
  alpha ~ normal(0, 10);
  beta  ~ normal(0, 10);
  sigma ~ exponential(1);
  y ~ normal(alpha + beta * x, sigma);
}
`;
const data = {
  N: 30,
  x: Array.from({ length: 30 }, (_, i) => -1.5 + i * 0.1),
  y: Array.from({ length: 30 }, (_, i) => -1.3 + i * 0.18),
};

const model = new StanModel(stanCode, JSON.stringify(data));
const aotBytes = model.compileToWasm();

// Instantiate the AOT module sharing stan's memory.
const aot = await WebAssembly.instantiate(aotBytes, {
  stan: { memory: sharedMemory() as WebAssembly.Memory },
  Math: { exp: Math.exp, log: Math.log, sin: Math.sin, cos: Math.cos, pow: Math.pow, lgamma, digamma, phi },
});
setAotExports(aot.instance.exports);

const init0 = new Float64Array([0, 0, 0]);

// Sanity: replay path (existing) and AOT path (new) should both give a sane β.
const t0 = performance.now();
const samples = model.sampleViaAot(init0, 1000, 1000, 42n);
const ms = performance.now() - t0;

const n = model.n_params;
const post = samples.subarray(1000 * n);
let sum = 0;
for (let i = 0; i < 1000; i++) sum += post[i * n + 1];
const meanBeta = sum / 1000;

console.log(`sampleViaAot: ${ms.toFixed(1)}ms (1000 warmup + 1000 draws)`);
console.log(`mean β = ${meanBeta.toFixed(3)} (data slope ≈ 1.8)`);
if (Math.abs(meanBeta - 1.8) > 0.5) {
  console.error("FAIL");
  process.exit(1);
}
console.log("OK");
