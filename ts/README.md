# stanwasm

[![npm](https://img.shields.io/npm/v/stanwasm?logo=npm&color=cb3837)](https://www.npmjs.com/package/stanwasm)
[![crates.io](https://img.shields.io/crates/v/stanwasm?logo=rust&color=e43717)](https://crates.io/crates/stanwasm)
[![bundle](https://img.shields.io/badge/wasm-514%20KB%20%7C%20192%20KB%20gzip-654ff0?logo=webassembly&logoColor=white)](https://github.com/habakan/stanwasm/blob/main/docs/en/BENCHMARKS.md)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/habakan/stanwasm/blob/main/LICENSE)

Stan probabilistic models compiled and sampled **entirely in the browser**.
Pure Rust compiled to WebAssembly, a single ~514 KB bundle (~192 KB gzipped),
with [nuts-rs](https://github.com/pymc-devs/nuts-rs) embedded as the sampler.
No server, no cmdstan, no round trip.

> **Status: alpha.** Pre-1.0, the API may change, and Stan language coverage is
> a documented subset. Not a replacement for
> [cmdstan](https://github.com/stan-dev/cmdstan) or
> [Stan Playground](https://github.com/flatironinstitute/stan-playground) —
> this is for browser-embedded use cases where those do not fit.

## Browser support

Chrome, Edge, Firefox, Safari and Node.js, plus every browser on iOS and
iPadOS. Verified under Playwright's three engines: Chromium 151, Firefox 153
and WebKit 26.5 all instantiate the module and sample.

Safari failed outright before 0.1.1 — WebKit rejects a module containing
[relaxed SIMD](https://github.com/WebAssembly/relaxed-simd) opcodes at
validation time, and `pulp` emitted them. This package ships the prebuilt wasm
with that resolved, so nothing is required of you; see
[the README](https://github.com/habakan/stanwasm#browser-support) for the
details and the one caveat that applies to the Rust crate.

## Install

```bash
npm install stanwasm
```

## Quick start

```js
import init, { StanModel, version } from "stanwasm";

// Loads the wasm. Bundlers resolve it from the package; pass
// `{ module_or_path: url }` to point somewhere else.
await init();

const model = new StanModel(`
  data { int<lower=0> N; vector[N] x; vector[N] y; }
  parameters { real alpha; real beta; real<lower=0> sigma; }
  model {
    alpha ~ normal(0, 10);
    beta  ~ normal(0, 10);
    sigma ~ exponential(1);
    y ~ normal(alpha + beta * x, sigma);
  }
`, JSON.stringify({ N: 30, x: [...], y: [...] }));

// (warmup + draws) * n_params, row-major, warmup first.
const draws = model.sample(new Float64Array(model.n_params), 1000, 1000, 42n);
```

`sample` blocks while it runs — put it in a Web Worker if the tab has to stay
responsive.

## API

| | |
|---|---|
| `init(opts?)` | Loads the wasm. Await before anything else. |
| `version()` | Crate version string. |
| `new StanModel(code, dataJson)` | Parses, type-checks and compiles a model. Throws with a located message on invalid Stan. |
| `.n_params`, `.paramNames()` | Unconstrained dimension, and names in order. |
| `.logProbGrad(params)` | `[logp, ...gradient]` in one `Float64Array`. |
| `.sample(init, warmup, draws, seed)` | NUTS. `seed` is a `BigInt`. |
| `.startStepSampling(init, warmup, draws, seed)`, `.stepDraw()`, `.finishStepSampling()` | One draw at a time, for live visualisation. |
| `.constrainDraw(draw)` | Unconstrained draw back to the model's own scale. |
| `.genQuantityNames()`, `.generatedQuantities(draws, nDraws, seed)` | `generated quantities` block. |
| `.compileToWasm()`, `.sampleViaAot(...)` | Ahead-of-time compile this model to its own wasm module and sample through it — 2 to 12x faster per gradient across the repository's fifteen benchmark models, and faster than CmdStan's native gradient on twelve of them. The emitted module uses fixed-width SIMD (Safari 16.4+). Needs `setAotExports` wiring; see the repository. |

Types ship with the package; the entry point is plain `.js` with a `.d.ts`
alongside, so plain-JavaScript and bundler consumers both work with no
TypeScript step.

## Node

The bundle is built with wasm-bindgen's `web` target, so under Node you read
the file and hand it over:

```js
import init, { StanModel } from "stanwasm";
import { readFile } from "node:fs/promises";

await init({ module_or_path: await readFile("node_modules/stanwasm/pkg/stanwasm_bg.wasm") });
```

## More

Runnable demos, the supported Stan subset, benchmarks and the architecture
write-up all live in the repository:
**<https://github.com/habakan/stanwasm>**

Licensed under Apache-2.0.
