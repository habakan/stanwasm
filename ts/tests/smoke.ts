// Node.js integration smoke test. Run `cd ts && npm install && npm test`; the
// wasm-bindgen output targets "web", so the .wasm is read and passed to init().

import { readFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import init, { StanModel, version } from "../index.js";

const here = dirname(fileURLToPath(import.meta.url));
const wasmPath = resolve(here, "..", "pkg", "stanwasm_bg.wasm");

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

// The path a user takes with a posterior fitted elsewhere: constrained draws
// in, unconstrained back, generated quantities from those.
const freed = gqModel.unconstrainDraw(constrained);
const again = gqModel.constrainDraw(freed);
for (let i = 0; i < constrained.length; i++) {
  if (Math.abs(again[i] - constrained[i]) > 1e-9) {
    console.error(`FAIL: unconstrainDraw round trip moved slot ${i}: ${constrained[i]} -> ${again[i]}`);
    process.exit(1);
  }
}
// A model whose computation depends on the parameters: it has no recorded tape,
// so it loads, refuses `sample`, and samples through `sampleFresh` instead.
const odeCode = `
functions {
  array[] real f(real t, array[] real y, array[] real theta,
                 array[] real x_r, array[] int x_i) {
    return { -theta[1] * y[1] };
  }
}
parameters { real<lower=0> k; }
model {
  array[1] real y0 = { 1.0 };
  array[1] real ts = { 1.0 };
  array[1, 1] real y = integrate_ode_rk45(f, y0, 0.0, ts, { k },
                                          rep_array(0.0, 0), rep_array(0, 0));
  k ~ lognormal(0, 1);
  target += 100 * y[1, 1];
}`;
const odeModel = new StanModel(odeCode, "{}");
try {
  odeModel.sample(new Float64Array([0.0]), 10, 10, 1n);
  console.error("FAIL: sample() should refuse a model with no recorded tape");
  process.exit(1);
} catch (e) {
  const msg = String((e as Error).message ?? e);
  if (!msg.includes("sampleFresh")) {
    console.error(`FAIL: the refusal should name sampleFresh, got: ${msg}`);
    process.exit(1);
  }
}
const odeDraws = odeModel.sampleFresh(new Float64Array([0.0]), 200, 200, 7n);
const odeN = odeModel.n_params;
let kMean = 0;
for (let i = 200; i < 400; i++) kMean += Math.exp(odeDraws[i * odeN]);
kMean /= 200;
// target += 100 * exp(-k) pushes k down against a lognormal(0,1) prior.
if (!(kMean > 0 && kMean < 1)) {
  console.error(`FAIL: sampleFresh gave a mean k of ${kMean}`);
  process.exit(1);
}
console.log(`sampleFresh runs a model with no recorded tape (mean k = ${kMean.toFixed(3)})`);

console.log(`unconstrainDraw round-trips ${gqModel.constrainedParamNames().join(", ")}`);

const gq = gqModel.generatedQuantities(gqPostWarmup, 20, 123n);
for (let i = 0; i < 20; i++) {
  const [yLn, yExp, yUnif, yGam] = gq.slice(i * 4, i * 4 + 4);
  if (!(yLn > 0) || !(yExp >= 0) || !(yUnif >= 0 && yUnif <= 1) || !(yGam >= 0)) {
    console.error(`FAIL: generated quantities out of support at draw ${i}: ${[yLn, yExp, yUnif, yGam]}`);
    process.exit(1);
  }
}
console.log(`generatedQuantities OK (${20} draws)`);

// nuts-rs asserts on a zero-length warmup schedule, and an assertion inside wasm
// reaches the caller as an untraceable trap.
for (const [what, run] of [
  ["sample", (m: StanModel, i: Float64Array) => m.sample(i, 0, 10, 7n)],
  ["sampleViaAot", (m: StanModel, i: Float64Array) => m.sampleViaAot(i, 0, 10, 7n)],
  ["startStepSampling", (m: StanModel, i: Float64Array) => m.startStepSampling(i, 0, 10, 7n)],
] as const) {
  let message = "";
  try {
    const m = new StanModel(stanCode, JSON.stringify(data));
    run(m, new Float64Array(m.n_params).fill(0.1));
  } catch (e) {
    message = String(e);
  }
  if (!message.includes("num_warmup must be at least 1")) {
    console.error(`FAIL: ${what} with no warmup gave ${message || "no error"}`);
    process.exit(1);
  }
}
console.log("zero warmup rejected on every entry point");

// A refused start has to name the parameters, and `randomInit` has to find one.
{
  const src = `parameters { real a; real b; }
               model { a ~ normal(0, 1); target += a * b; }`;
  const m = new StanModel(src, "{}");
  let message = "";
  try {
    m.sample(new Float64Array([0, 0.1]), 5, 5, 1n);
  } catch (e) {
    message = String(e);
  }
  if (!message.includes("does not move with") || !message.includes("b")) {
    console.error(`FAIL: a zero-gradient start gave ${message || "no error"}`);
    process.exit(1);
  }
  m.sample(m.randomInit(1n), 10, 10, 1n);
  console.log("a refused start names its parameters; randomInit finds one");
}

console.log("OK");
