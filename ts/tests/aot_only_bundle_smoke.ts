// The bundle without the Stan front end, doing the job it exists for: sample a
// module that was compiled somewhere else. Two bundles are loaded here — the
// full one only to play the part of the build step that produced the artifact,
// and the small one to do what a page would do with it.

import { readFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import initFull, { StanModel, setAotExports as setFull, sharedMemory as memFull }
  from "../index.js";
import initAot, {
  AotSampler,
  setAotExports as setAot,
  sharedMemory as memAot,
} from "../pkg-aot/stanwasm.js";

const here = dirname(fileURLToPath(import.meta.url));
const full = resolve(here, "..", "pkg", "stanwasm_bg.wasm");
const aotOnly = resolve(here, "..", "pkg-aot", "stanwasm_bg.wasm");

await initFull({ module_or_path: await readFile(full) });
await initAot({ module_or_path: await readFile(aotOnly) });

const fullBytes = (await readFile(full)).length;
const aotBytes = (await readFile(aotOnly)).length;
console.log(`bundle: ${aotBytes} bytes without the Stan front end, ${fullBytes} with`);
if (aotBytes >= fullBytes / 2) {
  console.error("FAIL: the AOT-only bundle is not meaningfully smaller");
  process.exit(1);
}

// ---- the build step ---------------------------------------------------------
const model = new StanModel(
  `data { int<lower=0> N; vector[N] x; vector[N] y; }
   parameters { real alpha; real beta; real<lower=0> sigma; }
   model {
     alpha ~ normal(0, 10); beta ~ normal(0, 10); sigma ~ exponential(1);
     y ~ normal(alpha + beta * x, sigma);
   }`,
  JSON.stringify({
    N: 30,
    x: Array.from({ length: 30 }, (_, i) => -1.5 + i * 0.1),
    y: Array.from({ length: 30 }, (_, i) => -1.3 + i * 0.18),
  }),
);
const artifact = {
  module: model.compileToWasm(),
  nParams: model.n_params,
  scratchInit: model.aotScratchInit(),
  paramNames: model.paramNames(),
};

const mathImports = {
  exp: Math.exp, log: Math.log, sin: Math.sin, cos: Math.cos, pow: Math.pow,
  tan: Math.tan, asin: Math.asin, acos: Math.acos, atan: Math.atan,
  lgamma: () => NaN, digamma: () => NaN, phi: () => NaN,
};

// ---- what the page does, with the small bundle only -------------------------
const pageInstance = await WebAssembly.instantiate(artifact.module, {
  stan: { memory: memAot() as WebAssembly.Memory },
  Math: mathImports,
});
setAot(pageInstance.instance.exports);

const layoutId = (pageInstance.instance.exports as Record<string, WebAssembly.Global>)
  .stanwasm_layout_id.value as number;
const sampler = new AotSampler(
  artifact.nParams,
  artifact.scratchInit,
  layoutId,
  artifact.paramNames,
);

const init0 = new Float64Array([0, 0, 0]);
const mine = sampler.sample(init0, 500, 500, 42n);

// ---- and it has to be the same run the full bundle would have made ----------
const buildInstance = await WebAssembly.instantiate(artifact.module, {
  stan: { memory: memFull() as WebAssembly.Memory },
  Math: mathImports,
});
setFull(buildInstance.instance.exports);
const theirs = model.sampleViaAot(init0, 500, 500, 42n);

let worst = 0;
for (let i = 0; i < mine.length; i++) {
  worst = Math.max(worst, Math.abs(mine[i] - theirs[i]));
}
if (mine.length !== theirs.length || worst !== 0) {
  console.error(`FAIL: the two bundles disagree by ${worst} over ${mine.length} values`);
  process.exit(1);
}

const n = artifact.nParams;
let sum = 0;
for (let i = 500; i < 1000; i++) sum += mine[i * n + 1];
console.log(`mean β = ${(sum / 500).toFixed(3)} (data slope ≈ 1.8)`);
console.log("the Stan-free bundle reproduces the full one exactly");
console.log("OK");
