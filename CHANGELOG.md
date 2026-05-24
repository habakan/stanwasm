# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`
- Status banner and Stan ecosystem positioning in README

## [0.1.0] — TBD (alpha)

Initial alpha release covering enough Stan to sample linear regression,
logistic regression, Poisson regression, eight schools (non-centered),
and multivariate-LKJ-style models end-to-end in the browser.

### Architecture

- Seven Rust crates: `stan-ast`, `stan-parser`, `stan-autodiff`,
  `stan-runtime`, `stan-codegen`, `stan-wasm-api`, `stan-cli`
- Single wasm bundle (~365 KB after `wasm-opt`) shipping the parser,
  AOT codegen, tape replay, and embedded `nuts-rs` sampler
- AOT model wasm imports memory from the host wasm-bindgen bundle —
  zero-copy bridge between sampler and per-model log_prob_grad
- AOT path emits wasm binary directly via `wasm-encoder`; no browser
  `wabt` dependency

### Stan language coverage

- Distributions: `normal`, `std_normal`, `exponential`, `half_normal`,
  `cauchy`, `student_t`, `lognormal`, `gamma`, `beta`, `bernoulli`,
  `bernoulli_logit`, `poisson`, `neg_binomial_2`,
  `multi_normal_cholesky`, `lkj_corr_cholesky`, `dirichlet`
- Constraints (scalar and vector): `lower`, `upper`, `lower_upper`
- Higher-order constraints: `simplex`, `ordered`, `positive_ordered`,
  `cholesky_factor_corr`
- Blocks: `data`, `parameters`, `transformed parameters`, `model`
- Control flow: `for` loops (parameter-independent bounds)
- Sampling statements (`y ~ dist(...)`), `target += expr`, local declarations

### Not yet supported

- `functions { ... }` block (user-defined functions)
- `generated quantities { ... }` block
- `multi_normal` (full covariance), `multinomial`, `categorical`,
  `cov_matrix`, `cholesky_factor_cov`, `corr_matrix`, `unit_vector`
- Parameter-dependent control flow (`if (alpha > 0) { ... }`)
- Stan profiling / `print()` / `reject()`
- Pathfinder, ADVI, or fixed_param samplers (NUTS only by design)

### Performance (Apple Silicon, Node.js V8, n_warmup=1000, n_draws=1000)

| model | replay | AOT |
|---|---:|---:|
| poisson_regression (2 params) | 10 ms | 5 ms |
| eight_schools_ncp (10 params) | 16 ms | 6 ms |

Comparable to the MoonBit-based predecessor `stan-wasm` and the `nuts-rs`
direct-call benchmark. See `docs/BENCHMARKS.md`.

### Validation

34 tests across the workspace, including:
- Per-distribution finite-difference gradient checks
- AOT-output-vs-AST-oracle log_prob/grad agreement to 1e-12
- End-to-end posterior recovery (linear regression slope β within ±0.3
  of truth on N=30 synthetic data, seed=42)
- `wasmparser` validation confirming the artifact uses no wasm-gc opcodes

[Unreleased]: https://github.com/habakan/stan-wasm-rs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/habakan/stan-wasm-rs/releases/tag/v0.1.0
