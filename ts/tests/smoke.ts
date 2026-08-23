// Node.js integration smoke test for stanwasm.
//
// Builds via `npm run build:wasm` first (or run from the repo root). The
// wasm-bindgen output uses `target = "web"`, so we read the .wasm file
// manually and pass it to the init() function.
//
// Run: cd ts && npm install && npm test

import { readFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import init, { StanModel, version } from "../index.ts";

const here = dirname(fileURLToPath(import.meta.url));
const wasmPath = resolve(here, "..", "pkg", "stan_wasm_api_bg.wasm");

const stanCode = `
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
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

const wasmBytes = await readFile(wasmPath);
await init({ module_or_path: wasmBytes });

console.log(`stanwasm v${version()}`);
const model = new StanModel(stanCode, JSON.stringify(data));
console.log(`n_params = ${model.n_params}`);
console.log(`param_names = ${model.paramNames().join(", ")}`);

const lpAndGrad = model.logProbGrad(new Float64Array([0.0, 1.0, 0.0]));
console.log(`logp(α=0,β=1,log_σ=0) = ${lpAndGrad[0].toFixed(4)}`);
console.log(`grad = [${[...lpAndGrad.slice(1)].map(g => g.toFixed(4)).join(", ")}]`);

const t0 = performance.now();
const samples = model.sample(new Float64Array([0, 0, 0]), 1000, 1000, 42n);
const elapsed = performance.now() - t0;

const nParams = model.n_params;
const postWarmup = samples.slice(1000 * nParams);
let sumBeta = 0;
for (let i = 0; i < 1000; i++) sumBeta += postWarmup[i * nParams + 1];
const meanBeta = sumBeta / 1000;

console.log(`sample() = ${elapsed.toFixed(1)}ms (1000 warmup + 1000 draws)`);
console.log(`mean β   = ${meanBeta.toFixed(3)}  (data slope ≈ 1.8)`);

if (Math.abs(meanBeta - 1.8) > 0.5) {
  console.error(`FAIL: posterior mean of β too far from data slope`);
  process.exit(1);
}

// --- generated quantities -------------------------------------------------

const gqCode = `
data {
  int<lower=0> N;
  vector[N] x;
  array[N] real y;
}
parameters {
  real mu;
  real<lower=0> sigma;
}
model {
  mu    ~ normal(0, 5);
  sigma ~ exponential(1);
  for (i in 1:N) {
    y[i] ~ normal(mu, sigma);
  }
}
generated quantities {
  real y_ln  = lognormal_rng(mu, sigma);
  real y_exp = exponential_rng(1.0);
  real y_unif = uniform_rng(0.0, 1.0);
  real y_gam = gamma_rng(2.0, 1.0);
}
`;
const gqData = { N: 2, x: [0.0, 1.0], y: [0.0, 1.0] };
const gqModel = new StanModel(gqCode, JSON.stringify(gqData));
console.log(`genQuantityNames = ${gqModel.genQuantityNames().join(", ")}`);

const gqDraws = gqModel.sample(new Float64Array([0, 0]), 50, 20, 42n);
const gqN = gqModel.n_params;
const gqPostWarmup = gqDraws.slice(50 * gqN);

const constrained = gqModel.constrainDraw(gqPostWarmup.slice(0, gqN));
console.log(`constrainDraw(first draw) = [${[...constrained].map(v => v.toFixed(3)).join(", ")}]`);
if (!(constrained[1] > 0)) {
  console.error(`FAIL: constrained sigma must be positive, got ${constrained[1]}`);
  process.exit(1);
}

const gq = gqModel.generatedQuantities(gqPostWarmup, 20, 123n);
for (let i = 0; i < 20; i++) {
  const [yLn, yExp, yUnif, yGam] = gq.slice(i * 4, i * 4 + 4);
  if (!(yLn > 0) || !(yExp >= 0) || !(yUnif >= 0 && yUnif <= 1) || !(yGam >= 0)) {
    console.error(`FAIL: generated quantities out of support at draw ${i}: ${[yLn, yExp, yUnif, yGam]}`);
    process.exit(1);
  }
}
console.log(`generatedQuantities OK (${20} draws)`);

console.log("OK");
