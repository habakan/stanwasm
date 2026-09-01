# stanwasm

[![npm](https://img.shields.io/npm/v/stanwasm?logo=npm&color=cb3837)](https://www.npmjs.com/package/stanwasm)
[![crates.io](https://img.shields.io/crates/v/stanwasm?logo=rust&color=e43717)](https://crates.io/crates/stanwasm)
[![bundle](https://img.shields.io/badge/wasm-482%20KB%20%7C%20180%20KB%20gzip-654ff0?logo=webassembly&logoColor=white)](docs/en/BENCHMARKS.md)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

> **Status: alpha** — usable but pre-1.0, API may change, Stan language coverage is a subset (see below). Not a replacement for [cmdstan](https://github.com/stan-dev/cmdstan) or [Stan Playground](https://github.com/flatironinstitute/stan-playground); intended for browser-embedded use cases where those don't fit.

Stan probabilistic models compiled and sampled entirely inside the browser. Pure Rust, single `~482 KB` wasm bundle (`~180 KB` gzipped), embedded [`nuts-rs`](https://github.com/pymc-devs/nuts-rs) sampler, zero backend required.

![stanwasm examples gallery demo](examples/gallery/demo.gif)

**[Try it in your browser](https://habakan.github.io/stanwasm/)** — the gallery
below, deployed from `main`. Nothing to install, and no server does any of the
sampling.

## Browser support

Chrome, Edge, Firefox, Safari, and Node.js, plus every browser on iOS and
iPadOS. Verified by loading the gallery under Playwright's three engines:
Chromium 151, Firefox 153 and WebKit 26.5 all instantiate the module and
sample, and the posterior means agree across them.

> **Safari support depends on a dependency fix that has not shipped yet.** The
> npm package is unaffected — it carries the prebuilt wasm, so
> `npm install stanwasm` gets it. If you depend on the `stanwasm` **crate** from
> crates.io and build the wasm yourself, you need the same patch in your own
> workspace (below) until the fix is released.

WebKit rejects a module containing
[relaxed SIMD](https://github.com/WebAssembly/relaxed-simd) opcodes at
validation time. Runtime dispatch does not help: wasm validates a whole module
up front, so instructions that are never reached still fail it — the module
never instantiates and nothing on the page works. The opcodes come from
`nuts-rs` → `faer` → `pulp`, which attaches
`#[target_feature(enable = "relaxed-simd")]` to its wasm kernels.

pulp made this optional in [pulp#30](https://github.com/sarah-quinones/pulp/pull/30)
(a `relaxed-simd` feature, on by default), released in 0.22.3, and faer already
opts out. nuts-rs is the last crate in the graph pulling pulp's default
features, and Cargo cannot subtract a transitive default feature, so this
workspace pins a fork until
[nuts-rs#76](https://github.com/pymc-devs/nuts-rs/pull/76) is merged and
released:

```toml
[patch.crates-io]
nuts-rs = { git = "https://github.com/habakan/nuts-rs", rev = "61f261b26815be8cb21d5eef0840ba9f869d3af4" }
```

Dropping relaxed SIMD costs nothing measurable at these parameter dimensions:
-4.3% and 0.0% on two models (1000 warmup + 1000 draws, median of 7 runs,
Chromium). The bundle grows about 2 KB.

## Quick start (browser / Node.js)

```bash
npm install stanwasm
```

The entry point is plain `.js` with a `.d.ts` alongside, so bundlers and
plain-JS consumers work without a TypeScript step. The wasm ships inside the
package — nothing is fetched at install time and nothing is compiled on a
server.

To build from a checkout instead, `make smoke` produces the bundle in `ts/pkg/`
and exercises it in Node.

```ts
import init, { StanModel } from "stanwasm";

await init();

const stanCode = `...`;
const data = { N: 30, x: [...], y: [...] };
const model = new StanModel(stanCode, JSON.stringify(data));

console.log(`n_params = ${model.n_params}`);
console.log(`names    = ${model.paramNames().join(", ")}`);

// Single-call gradient
const lpAndGrad = model.logProbGrad(new Float64Array([0, 1, 0]));
//   [logp, dα, dβ, dlog_σ]

// Full NUTS sampling
const samples = model.sample(
  new Float64Array([0, 0, 0]),
  /*nWarmup*/ 1000,
  /*nDraws*/  1000,
  /*seed*/    42n,
);
// samples is Float64Array, row-major shape (nWarmup + nDraws) × n_params

// `sample()`/`sampleViaAot` return unconstrained draws; get constrained
// parameter values (e.g. sigma on its natural, not log, scale) per draw:
const constrained = model.constrainDraw(samples.slice(0, model.n_params));

// If the model has a `generated quantities` block, evaluate it over a batch
// of draws (one shared, seeded RNG stream across the whole batch):
console.log(model.genQuantityNames().join(", "));
const gq = model.generatedQuantities(samples, /*nDraws*/ 1000 + 1000, /*seed*/ 7n);

// Want to watch the sampler work rather than just get the finished draws?
// startStepSampling/stepDraw keep the NUTS chain's state alive between
// calls, so you can drive it one draw at a time (e.g. one per animation
// frame) instead of blocking on the whole run:
model.startStepSampling(new Float64Array([0, 0, 0]), 500, 500, 42n);
const draw = model.stepDraw();
// [alpha, beta, log_sigma, tuning(0/1), diverging(0/1), step_size, num_steps]
// step_size/num_steps are nuts-rs's own live adaptation state, not values
// this crate computes.
```

### Demos

**[habakan.github.io/stanwasm](https://habakan.github.io/stanwasm/)** — deployed
from `main` on every change to the crates, `ts/`, or the app itself. Source:
[`examples/gallery`](examples/gallery). One app, tabbed:

- **MCMC Visualizer** — NUTS and Random-Walk Metropolis step the same hard posterior (Neal's funnel) side by side, one real draw per animation frame, via the step-by-step sampling API (`startStepSampling`/`stepDraw`) — not a replay of a finished chain.
- **Live Regression** — drag a data point and watch a robust (Student-t) and a conjugate (normal) regression refit **live**, every animation frame, diverging on the outlier — no closed form for the former, no server round trip for either.
- **Hierarchical Shrinkage** — six marketing campaigns' observed A/B test lift (three well-powered, three small-sample pilots) fit with a partial-pooling model (the classic "eight schools" structure). Drag one's observed value and watch a flashy small-sample number get pulled toward the population estimate live, by an amount the posterior derives rather than a hand-tuned rule.
- **Wasm Sandbox** — a fuller API tour: CSV upload, editable Stan source, multiple presets, posterior summary table.

## Stan language coverage

A subset — enough for linear, logistic, Poisson and negative-binomial
regression, hierarchical models, and Cholesky-parameterised multivariate ones.
Anything outside it is a clean load-time or evaluation error, never a model
that silently samples something else.

[`ROADMAP.md`](ROADMAP.md) has the full table of what is supported and what is
not, the four behavioural caveats a table cannot carry, and the remaining gaps
ordered by effort.

## Architecture

[`ARCHITECTURE.md`](ARCHITECTURE.md) is the internals tour: the data flow from
Stan source to samples, the seven-crate workspace layout, the autodiff tape
design, the AOT codegen ABI, what differs between the native and wasm builds,
and why wasm32 rather than wasm-gc. Performance numbers live in
[`docs/en/BENCHMARKS.md`](docs/en/BENCHMARKS.md). Documentation is organized by
language under `docs/en/` and `docs/ja/`.

## Native development

Requires Rust 1.88+ (the workspace MSRV; `nuts-rs` needs edition 2024).

```bash
cargo build --release
cargo test                    # ~76 tests across all crates
cargo run --release -p stanwasm-cli -- bench all
```

## Security

See [`SECURITY.md`](SECURITY.md) for how to report a vulnerability and what is
in scope. In short: a malicious Stan model crashing its own page is expected
(the sandbox contains it), a sandbox escape or a supply-chain problem is not.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). The project is in alpha and maintained on evenings and weekends, so issue triage and PR reviews happen in batches. Distribution / constraint additions and example PRs are especially welcome.

For the broader Stan ecosystem (cmdstan, stanc3, official interfaces), see [stan-dev](https://github.com/stan-dev). For the official browser playground, see [Stan Playground](https://github.com/flatironinstitute/stan-playground).

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
