# stan-wasm-rs

> **Status: alpha** — usable but pre-1.0, API may change, Stan language coverage is a subset (see below). Not a replacement for [cmdstan](https://github.com/stan-dev/cmdstan) or [Stan Playground](https://github.com/flatironinstitute/stan-playground); intended for browser-embedded use cases where those don't fit.

Stan probabilistic models compiled and sampled entirely inside the browser. Pure Rust, single `~415 KB` wasm bundle, embedded [`nuts-rs`](https://github.com/pymc-devs/nuts-rs) sampler, zero backend required.

## When to use this (and when not)

| Need | Use this | Use cmdstan / Stan Playground instead |
|---|---|---|
| Full Stan language coverage | | ✓ |
| Run in a browser with no server | ✓ | |
| Embed a Bayesian model into a web app (npm package) | ✓ | |
| Data must not leave the user's device | ✓ | |
| Offline / air-gapped environment | ✓ | |
| Research / publication-grade workflows | | ✓ |
| `functions { ... }` block, full multivariate suite | | ✓ |

[Stan Playground](https://stan-playground.flatironinstitute.org) is the official Stan-recommended browser interface; it uses a compile server and supports the full Stan language. stan-wasm-rs is **complementary**: smaller surface, no server, embeddable.

## Validated end-to-end

Linear regression posterior recovers the true slope to ±0.3 in 1000 post-warmup draws (seed=42). AOT codegen output agrees with the AST oracle to 1e-12 on log_prob and gradients across all covered distributions.

## Stan language coverage

Distributions covered:
- Continuous: `normal`, `std_normal`, `exponential`, `half_normal`, `cauchy`, `student_t`, `lognormal`, `gamma`, `beta`
- Discrete: `bernoulli`, `bernoulli_logit`, `poisson`, `neg_binomial_2`
- Multivariate: `multi_normal_cholesky`, `lkj_corr_cholesky`, `dirichlet`

Constraint transforms:
- Scalar: `lower`, `upper`, `lower_upper`
- Vector: same with element-wise broadcast
- Vector shape: `simplex`, `ordered`, `positive_ordered`
- Matrix: `cholesky_factor_corr`

Blocks: `data`, `parameters`, `transformed parameters`, `model`, `generated quantities`, plus `for`/`while` loops, `if`/`else` (including parameter-dependent conditions), `break`/`continue`, comparison/logical operators, sampling statements (`y ~ dist(...)`), and `target += expr`.

`generated quantities` supports RNG draws for every covered distribution above (`normal_rng`, `exponential_rng`, `gamma_rng`, `dirichlet_rng`, `multi_normal_cholesky_rng`, etc.) plus `uniform_rng`. It runs natively (tape-replay) inside the same wasm bundle — there is no separate AOT-compiled path for it (see [Architecture](#architecture)).

**Not yet supported**: `multi_normal` (full covariance), `multinomial`, `categorical`, `cov_matrix`, `cholesky_factor_cov`, `corr_matrix`, `unit_vector`, `lkj_corr_cholesky_rng`, `functions { ... }` block (user-defined functions).

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for an internals tour, [`docs/en/MIGRATION.md`](docs/en/MIGRATION.md) for the per-phase plan, [`docs/en/BENCHMARKS.md`](docs/en/BENCHMARKS.md) for performance numbers, and [`docs/ja/MOONBIT_VS_RUST.md`](docs/ja/MOONBIT_VS_RUST.md) (Japanese, [English summary](docs/en/MOONBIT_VS_RUST.md)) for a tech-note on the rewrite history. Documentation is organized by language under `docs/en/` and `docs/ja/`.

## Architecture

```
Stan source + JSON data
       │
       ▼  (one-time)
stan-parser ─► AST ─► stan-runtime (trace forward pass on autodiff tape)
                                      │
                                      ▼
                          ┌──── tape replay (sampling)
                          │              │
                          │              ▼
                          │         nuts-rs (embedded in same wasm)
                          │              │
                          │              ▼
                          │           samples
                          │
                          └──── stan-codegen (wasm-encoder)  ──►  AOT model wasm bytes
                                                                  (Web Worker handoff)
```

A single `stan_wasm_api_bg.wasm` is shipped to the browser (replaces the previous `wasm_api.wasm` + `nuts_rs.wasm` pair). Native builds expose the AST evaluation interpreter for golden-value testing only.

## Workspace layout

| Crate | Target | Role |
|---|---|---|
| `stan-ast` | lib | AST types, shared definitions |
| `stan-parser` | lib | Lexer, parser, error reporting |
| `stan-autodiff` | lib | Reverse-mode tape (flat array) |
| `stan-runtime` | lib | Distributions, constraints, AST evaluator (reference oracle) |
| `stan-codegen` | lib | AOT compilation: tape → wasm bytes (via `wasm-encoder`) |
| `stan-wasm-api` | cdylib (wasm32) | wasm-bindgen public API; embeds nuts-rs |
| `stan-cli` | bin (native) | Local development, golden-value tests, benchmarks |

## Quick start (browser / Node.js)

```bash
# Build the wasm bundle
./scripts/build-wasm.sh

# Run smoke test
cd ts
node --experimental-strip-types tests/smoke.ts
```

```ts
import init, { StanModel } from "stan-wasm-rs";

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
```

### Demos

- [`examples/live_regression`](examples/live_regression) — drag a data point and watch a robust (Student-t) and a conjugate (normal) regression refit **live**, every animation frame, diverging on the outlier — no closed form for the former, no server round trip for either.
- [`examples/get_started`](examples/get_started) — a fuller API tour: CSV upload, editable Stan source, multiple presets, posterior summary table.

## Native development

```bash
cargo build --release
cargo test                    # ~30 tests across all crates
cargo run --release -p stan-cli -- bench all
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). The project is in alpha and maintained on evenings and weekends, so issue triage and PR reviews happen in batches. Distribution / constraint additions and example PRs are especially welcome.

For the broader Stan ecosystem (cmdstan, stanc3, official interfaces), see [stan-dev](https://github.com/stan-dev). For the official browser playground, see [Stan Playground](https://github.com/flatironinstitute/stan-playground).

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
