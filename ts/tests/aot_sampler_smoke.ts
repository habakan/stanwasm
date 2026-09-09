// Smoke test for `AotSampler`: sample a module compiled ahead of time, with no
// StanModel behind it. Here the module still comes from one — that is how the
// draws get something to be compared against — but nothing the sampler is
// handed comes from the Stan side except the numbers a compiler would record
// beside the module.

import { readFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import init, {
  AotSampler,
  StanModel,
  setAotExports,
  clearAotExports,
  sharedMemory,
} from "../index.js";

const here = dirname(fileURLToPath(import.meta.url));
const wasmBytes = await readFile(resolve(here, "..", "pkg", "stanwasm_bg.wasm"));
await init({ module_or_path: wasmBytes });

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

// ---- what a build step would produce and a page would ship -----------------
const model = new StanModel(stanCode, JSON.stringify(data));
const moduleBytes = model.compileToWasm();
const artifact = {
  nParams: model.n_params,
  scratchInit: model.aotScratchInit(),
  paramNames: model.paramNames(),
};

const hostImports = {
  stan: { memory: sharedMemory() as WebAssembly.Memory },
  Math: {
    exp: Math.exp, log: Math.log, sin: Math.sin, cos: Math.cos, pow: Math.pow,
    tan: Math.tan, asin: Math.asin, acos: Math.acos, atan: Math.atan,
    // Not reached by this model; present so a missing import is not what fails.
    lgamma: () => NaN, digamma: () => NaN, phi: () => NaN,
  },
};

const aot = await WebAssembly.instantiate(moduleBytes, hostImports);
setAotExports(aot.instance.exports);

// The id is read off the module itself, the way a page with only the artifact
// would have to.
const layoutId = (aot.instance.exports as Record<string, WebAssembly.Global>)
  .stanwasm_layout_id.value as number;

const sampler = new AotSampler(
  artifact.nParams,
  artifact.scratchInit,
  layoutId,
  artifact.paramNames,
);

const init0 = new Float64Array([0, 0, 0]);
const warmup = 500;
const draws = 500;

// ---- the draws must be the ones sampleViaAot would have produced -----------
const mine = sampler.sample(init0, warmup, draws, 42n);
const theirs = model.sampleViaAot(init0, warmup, draws, 42n);

if (mine.length !== theirs.length) {
  console.error(`FAIL: ${mine.length} draws vs ${theirs.length}`);
  process.exit(1);
}
let worst = 0;
for (let i = 0; i < mine.length; i++) {
  worst = Math.max(worst, Math.abs(mine[i] - theirs[i]));
}
if (worst !== 0) {
  console.error(`FAIL: AotSampler and sampleViaAot differ by ${worst}`);
  process.exit(1);
}
console.log(`AotSampler reproduces sampleViaAot exactly (${draws} draws, same seed)`);

// ---- the gradient entry point agrees with the model's ----------------------
const at2 = new Float64Array([0.3, 1.1, -0.2]);
const a = sampler.logProbGrad(at2);
const b = model.logProbGrad(at2);
for (let i = 0; i < a.length; i++) {
  if (Math.abs(a[i] - b[i]) > 1e-12) {
    console.error(`FAIL: logProbGrad[${i}] ${a[i]} vs ${b[i]}`);
    process.exit(1);
  }
}
console.log("logProbGrad agrees with the model it was compiled from");

// ---- the binding is checked, the same way sampleViaAot checks it -----------
const other = new StanModel(
  `data { int<lower=0> N; vector[N] y; }
   parameters { real mu; }
   model { mu ~ normal(0, 1); y ~ normal(mu, 1); }`,
  JSON.stringify({ N: 4, y: [0.1, 0.2, 0.3, 0.4] }),
);
const otherAot = await WebAssembly.instantiate(other.compileToWasm(), hostImports);
setAotExports(otherAot.instance.exports);

let refusal = "";
try {
  sampler.sample(init0, 10, 10, 42n);
} catch (e) {
  refusal = String(e);
}
if (!refusal.includes("compiled for a different model")) {
  console.error(`FAIL: a mismatched binding gave ${refusal || "no error"}`);
  process.exit(1);
}

setAotExports(aot.instance.exports);
sampler.sample(init0, 10, 10, 42n);

clearAotExports();
let unbound = "";
try {
  sampler.logProbGrad(at2);
} catch (e) {
  unbound = String(e);
}
if (!unbound.includes("no AOT module is bound")) {
  console.error(`FAIL: with nothing bound, got ${unbound || "no error"}`);
  process.exit(1);
}
console.log("a mismatched or missing binding is refused, and re-binding restores it");

// ---- the constructor refuses a buffer that cannot belong to the module -----
setAotExports(aot.instance.exports);
let tooSmall = "";
try {
  new AotSampler(3, new Float64Array([0, 0]), layoutId, []);
} catch (e) {
  tooSmall = String(e);
}
if (!tooSmall.includes("too few")) {
  console.error(`FAIL: a short scratch buffer gave ${tooSmall || "no error"}`);
  process.exit(1);
}

let wrongNames = "";
try {
  new AotSampler(3, artifact.scratchInit, layoutId, ["only", "two"]);
} catch (e) {
  wrongNames = String(e);
}
if (!wrongNames.includes("param_names has 2 entries")) {
  console.error(`FAIL: mismatched names gave ${wrongNames || "no error"}`);
  process.exit(1);
}
console.log("a scratch buffer or name list that cannot belong to the module is refused");

// ---- a module carrying no ABI version is refused before anything else ------
// Stripping the export is the only way to produce one here: every module this
// build emits carries the number. A module from an older release would arrive
// the same way.
const stripped = new Uint8Array(moduleBytes);
const marker = new TextEncoder().encode("stanwasm_abi_version");
let at = -1;
outer: for (let i = 0; i + marker.length <= stripped.length; i++) {
  for (let j = 0; j < marker.length; j++) {
    if (stripped[i + j] !== marker[j]) continue outer;
  }
  at = i;
  break;
}
if (at < 0) {
  console.error("FAIL: the emitted module exports no stanwasm_abi_version");
  process.exit(1);
}
stripped[at] = "x".charCodeAt(0); // rename the export; the global stays

const strippedInstance = await WebAssembly.instantiate(stripped, hostImports);
setAotExports(strippedInstance.instance.exports);
let noAbi = "";
try {
  sampler.logProbGrad(at2);
} catch (e) {
  noAbi = String(e);
}
if (!noAbi.includes("no version at all")) {
  console.error(`FAIL: a module with no ABI version gave ${noAbi || "no error"}`);
  process.exit(1);
}
console.log("a module that does not name an ABI this runtime knows is refused");

console.log("OK");
