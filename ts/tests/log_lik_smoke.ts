// The pointwise log-likelihood, both ways out: traced fresh, and read from a
// compiled module. They have to agree, because what `az.loo` reads is whichever
// one the page could afford.

import { readFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import init, { StanModel, setAotExports, sharedMemory } from "../index.js";

const here = dirname(fileURLToPath(import.meta.url));
await init({ module_or_path: await readFile(resolve(here, "..", "pkg", "stanwasm_bg.wasm")) });

const fail = (m: string): never => { console.error(`FAIL: ${m}`); process.exit(1); };

const N = 30;
const data = {
  N,
  x: Array.from({ length: N }, (_, i) => -1.5 + i * 0.1),
  y: Array.from({ length: N }, (_, i) => -1.3 + i * 0.18),
};
const src = `
data { int<lower=0> N; vector[N] x; vector[N] y; }
parameters { real alpha; real beta; real<lower=0> sigma; }
model {
  alpha ~ normal(0, 10);
  beta  ~ normal(0, 10);
  sigma ~ exponential(1);
  y ~ normal(alpha + beta * x, sigma);
}
`;

const model = new StanModel(src, JSON.stringify(data));
if (model.logLikCount !== N) fail(`logLikCount ${model.logLikCount}, not ${N}`);

// Compiled without the terms, the module has none and says so.
await bind(model.compileToWasm());
let refused = "";
try { model.logLikViaAot(new Float64Array([0.1, 0.1, 0.1])); } catch (e) { refused = String(e); }
if (!refused.includes("compileToWasm")) fail(`the refusal does not say what to do: ${refused}`);

// Compiled with them, the module answers what a fresh trace does.
await bind(model.compileToWasm("auto", true));
const at = new Float64Array([0.3, 1.7, -0.4]);
const traced = model.logLik(at);
const aot = model.logLikViaAot(at);
if (aot.length !== N) fail(`logLikViaAot returned ${aot.length} terms`);
aot.forEach((v, i) => {
  if (Math.abs(v - traced[i]) > 1e-12 * Math.max(1, Math.abs(traced[i]))) {
    fail(`term ${i}: module ${v}, trace ${traced[i]}`);
  }
});

// The terms have to be the likelihood: the density at this point, less the three
// priors and the log-sigma Jacobian, is their sum.
const [alpha, beta, logSigma] = Array.from(at);
const sigma = Math.exp(logSigma);
const normal = (v: number, s: number) => -0.5 * (v / s) ** 2 - Math.log(s) - 0.5 * Math.log(2 * Math.PI);
const priors = normal(alpha, 10) + normal(beta, 10) + Math.log(1) - sigma + logSigma;
const lp = model.logProbGrad(at)[0];
const sum = aot.reduce((a, b) => a + b, 0);
if (Math.abs(sum + priors - lp) > 1e-9 * Math.max(1, Math.abs(lp))) {
  fail(`terms sum to ${sum}, and lp - priors is ${lp - priors}`);
}

// What a page does with it: one row per draw, which is the `log_lik` group
// ArviZ and loo read.
const draws = model.sampleViaAot(new Float64Array([0, 0, 0]), 200, 200, 42n);
const n = model.n_params;
const rows = Array.from({ length: 200 }, (_, i) =>
  model.logLikViaAot(draws.subarray((200 + i) * n, (200 + i + 1) * n)));
if (rows.some((r) => r.length !== N || r.some((v) => !Number.isFinite(v)))) {
  fail("a draw's log-likelihood row is not finite");
}

// A likelihood written as a sum has no terms to name, and says so rather than
// reporting a wrong shape.
const summed = new StanModel(
  `data { int<lower=0> N; vector[N] y; }
   parameters { real mu; }
   model { mu ~ normal(0, 10); target += normal_lpdf(y | mu, 1.0); }`,
  JSON.stringify({ N: 4, y: [0.1, -0.3, 1.2, 0.7] }),
);
if (summed.logLikCount !== 0) fail("a `target +=` likelihood named terms");

console.log(`log_lik: ${N} terms per draw, 200 rows, module and trace agree`);

async function bind(bytes: Uint8Array) {
  const aot = await WebAssembly.instantiate(bytes, {
    tapewasm: { memory: sharedMemory() as WebAssembly.Memory },
    Math: { exp: Math.exp, log: Math.log, sin: Math.sin, cos: Math.cos, pow: Math.pow,
            tan: Math.tan, asin: Math.asin, acos: Math.acos, atan: Math.atan,
            lgamma: () => NaN, digamma: () => NaN, phi: () => NaN },
  });
  setAotExports(aot.instance.exports);
}
