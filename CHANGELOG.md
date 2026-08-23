# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Typos and unsupported constructs that used to silently contribute nothing
  to the log density (or a stale value) instead of failing now raise a clean
  error at model construction or evaluation time: an unrecognized
  distribution/function name, an unknown top-level block name, a bare
  expression statement (e.g. `print(...)`/`reject(...)`, not yet
  supported), an `_rng` call outside `generated quantities`, invalid `_rng`
  parameters (e.g. a negative `gamma_rng` shape), an unrecognized source
  character, and — the sharpest one — an `if`/`while` in `model`/
  `transformed parameters` whose condition depends on a sampled parameter
  (NUTS traces that block once and replays the same graph per draw, so a
  parameter-dependent branch can't be honored there; `generated quantities`
  is unaffected, since it re-evaluates natively every draw).
- `exp`/`log`/`abs`/`lgamma`/`Phi` (and `sqrt`, `pow`'s general path) now
  broadcast element-wise over vectors, matching Stan's vectorized math
  functions (e.g. `vector[N] y = exp(x);`) — previously these panicked on
  a vector argument, crashing the wasm instance.
- Assigning through an indexed or otherwise non-`name` target (e.g.
  `arr[i] = expr;`) is now a clean error instead of silently discarding the
  assignment; implementing it for real is tracked in `ROADMAP.md`.
- A panic hook (`console_error_panic_hook`) is installed on wasm module
  init, so the remaining internal-invariant panics surface a real message
  in the console instead of an opaque `RuntimeError: unreachable`.
- An out-of-bounds index (e.g. `v[5]` on a length-2 vector) is now a clean
  error instead of silently reading `0`.
- `lkj_corr_cholesky_lpdf` computed the wrong density: it applied the
  `(2η-2)` LKJ term as a single factor multiplied across the whole
  Cholesky-Jacobian-weighted sum, rather than adding it to each row's own
  weight before that row's `log(L_kk)` term — for K=2 this made the
  density exactly 0 for any input. Fixed and covered by a regression test
  against the K=2 closed-form density.
- `sample()` and `startStepSampling()` left the model permanently unable
  to sample or evaluate `logProbGrad` after a nuts-rs init failure (e.g. a
  rejected `init` with zero gradient) — the internal `Compiled` was taken
  out before nuts-rs ran and only restored on success. Now restored
  regardless of outcome.
- `-a^2` parsed as `(-a)^2`, flipping the sign of a common prior/likelihood
  idiom, and `^` was left-associative (`2^3^2` gave 64 instead of 512). `^`
  now binds tighter than unary minus and associates right, as in Stan.
- `int / int` did real division, so `N / 2` with `int N = 3` gave `1.5`
  instead of `1`. Integer literals are a distinct token and AST node now,
  and `Env` records which bindings are int-typed (`int` data, `array[...]
  int`, loop counters, `int` locals), so `/` truncates exactly where Stan
  says it does.
- `array[N] real<lower=0>` (and every other array-of-constrained-element
  declaration) was sampled with no transform and no Jacobian, accepting
  negative values as if they were positive. Array element constraints are
  applied recursively now.
- `cov_matrix`, `corr_matrix`, `cholesky_factor_cov` and `unit_vector`
  parsed and then sampled unconstrained with a zero Jacobian, despite the
  README listing them as unsupported. They are now
  `EvalError::UnsupportedConstraint`.
- A `matrix[R, C]` parameter arrived as one flat vector, so `M[i, j]` read
  the wrong element. It is reshaped into rows.
- Element-wise arithmetic `zip`ped its operands, so `vector[3] + vector[2]`
  quietly returned a length-2 vector and `X * beta` (matrix × vector)
  returned the matrix with each row scaled by one element of `beta`.
  Operand shapes are checked; a matrix product is a clean "not implemented"
  error rather than a plausible wrong number.
- A range index past the end (`y[2:5]` on a length-3 vector) silently
  returned a short vector instead of a bounds error.
- The `data` block was never checked against the supplied JSON. A missing
  field, a wrong length or shape, a non-integral value for an `int`, and a
  violated `<lower=...>`/`<upper=...>` bound (`{"N": -5}` for
  `int<lower=0> N`) all loaded and sampled. `Model::parse_and_load`
  validates every declaration once, up front.
- `a ~ normal(0);` indexed past the argument list and panicked — a wasm
  trap, which kills the module instance and forces a page reload in the
  browser. Distribution arity is checked before dispatch, and the
  multivariate forms that used to fall back to a zero log-density
  contribution on an unexpected shape now report the shape they wanted.
- A vectorized distribution argument shorter than the variate
  (`y ~ normal(mu, s)` with `mu` shorter than `y`) panicked; a longer one
  had its tail silently ignored. Both are errors.
- `Val::to_f64`/`to_tape` panicked on a container, so comparing two vectors
  with `==` or feeding a matrix product to a scalar lpdf trapped the wasm
  instance. They return `EvalError` instead.
- `stan-codegen` emitted AOT modules with two wasm locals per tape node and
  no ceiling, so a model whose trace exceeds ~25,000 nodes (roughly
  `N ≈ 2,000` for a vectorized regression) produced a module the browser
  rejects with `CompileError: local count too large`. `compile()` now
  returns `CodegenError::TooManyLocals` up front, and the limit is
  documented in `ARCHITECTURE.md` and `docs/en/BENCHMARKS.md`.
- `num_warmup + num_draws` was summed as `u32` before widening to `u64`, so
  a large pair wrapped to a different run length.

### Changed

- The npm entry point is `ts/index.js` with a hand-written `ts/index.d.ts`,
  and `package.json` declares `types`/`exports`. It was `index.ts`, which
  broke plain-JS consumers, bundlers, and Node without
  `--experimental-strip-types`.
- Declared MSRV: Rust 1.88 (`rust-version` in the workspace `Cargo.toml`),
  set by `nuts-rs` 0.18's edition-2024 + let-chains usage and verified with
  `cargo +1.88 test --workspace`. `CONTRIBUTING.md` previously said 1.80,
  which does not build.

### Removed

- `docs/internal/DELIVERY.md` (maintainer-only release-planning notes) is no
  longer tracked; `docs/internal/` is gitignored.

## [0.1.0] — 2026-08-23 (alpha)

Initial alpha release: enough Stan to sample linear regression, logistic
regression, Poisson regression, eight schools (non-centered), and
multivariate-LKJ-style models end-to-end in the browser — plus a
`generated quantities` block, step-by-step sampling for live
visualization, and a tabbed examples gallery.

### Added

- `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`
- Status banner and Stan ecosystem positioning in README
- `generated quantities { ... }` block, evaluated natively (tape-replay,
  inside the same wasm bundle) after sampling: `_rng` draws for every
  covered distribution (`normal_rng`, `exponential_rng`, `gamma_rng`,
  `dirichlet_rng`, `multi_normal_cholesky_rng`, etc.) plus `uniform_rng`.
  New `StanModel` methods: `genQuantityNames()`, `generatedQuantities()`,
  `constrainDraw()` (constrained parameter values for one unconstrained
  draw — previously `sample()`'s output had no way to recover these).
- Real evaluation of `if`/`else`, `while`, `break`/`continue`, and
  comparison/logical operators (`== != < > <= >= && ||`) in `model`,
  `transformed parameters`, and `generated quantities` — these previously
  parsed but silently no-oped.
- Step-by-step NUTS sampling: `StanModel.startStepSampling()` +
  `.stepDraw()` advance the sampler one draw at a time, keeping its state
  alive between calls (`.finishStepSampling()` to stop early), instead of
  `sample()`'s "run the whole chain, return at the end" model — for driving
  a live visualization of sampling in progress rather than replaying an
  already-finished chain. `.stepDraw()` also returns nuts-rs's own
  `step_size`/`num_steps` for that draw (its live dual-averaging adaptation
  and trajectory-length search), not values this crate computes.
- `examples/gallery`: a tabbed demo app — MCMC Visualizer (NUTS vs
  Random-Walk Metropolis racing live on Neal's funnel via the step-by-step
  API, adjustable chain count, a "fog of war" coverage veil, and a
  per-chain live wasm log), Live Regression (drag-to-refit robust vs
  conjugate regression), Hierarchical Shrinkage (partial-pooling shrinkage
  on marketing-campaign A/B test data), and Wasm Sandbox (a fuller,
  IDE-style API tour: CSV upload, editable Stan source, multiple presets,
  posterior summary table).
- `examples/gallery`: graphical-model diagrams (node-and-plate plots next
  to each Stan code block) parsed directly from the Stan source rather
  than hand-drawn per tab, including a live one in Wasm Sandbox that
  follows the editor. Distribution formulas render via MathJax, served
  from a locally-copied bundle rather than a CDN.

### Architecture

- Seven Rust crates: `stan-ast`, `stan-parser`, `stan-autodiff`,
  `stan-runtime`, `stan-codegen`, `stan-wasm-api`, `stan-cli`
- Single wasm bundle (~431 KB after `wasm-opt`, including `rand`/
  `rand_distr` for `generated quantities` RNG support) shipping the
  parser, AOT codegen, tape replay, and embedded `nuts-rs` sampler
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
- Blocks: `data`, `parameters`, `transformed parameters`, `model`,
  `generated quantities`
- Control flow: `for`/`while` loops, `if`/`else`, `break`/`continue`,
  comparison/logical operators (`== != < > <= >= && ||`). A
  parameter-dependent `if`/`while` condition works in `generated quantities`
  but is a compile-time error in `model`/`transformed parameters` (see
  `[Unreleased]`).
- Sampling statements (`y ~ dist(...)`), `target += expr`, local declarations
- `generated quantities` supports RNG draws for every covered distribution
  above (`normal_rng`, `exponential_rng`, `gamma_rng`, `dirichlet_rng`,
  `multi_normal_cholesky_rng`, etc.) plus `uniform_rng`. It runs natively
  (tape-replay) inside the same wasm bundle — there is no separate
  AOT-compiled path for it.

### Not yet supported

- `functions { ... }` block (user-defined functions)
- `multi_normal` (full covariance), `multinomial`, `categorical`,
  `cov_matrix`, `cholesky_factor_cov`, `corr_matrix`, `unit_vector`,
  `lkj_corr_cholesky_rng`
- Stan profiling / `print()` / `reject()`
- Pathfinder, ADVI, or fixed_param samplers (NUTS only by design)

### Performance (Apple Silicon, Node.js V8, n_warmup=1000, n_draws=1000)

| model | replay | AOT |
|---|---:|---:|
| poisson_regression (2 params) | 10 ms | 5 ms |
| eight_schools_ncp (10 params) | 16 ms | 6 ms |

Comparable to the `nuts-rs` direct-call benchmark. See `docs/en/BENCHMARKS.md`.

### Validation

~45 tests across the workspace, including:
- Per-distribution finite-difference gradient checks
- AOT-output-vs-AST-oracle log_prob/grad agreement to 1e-12
- End-to-end posterior recovery (linear regression slope β within ±0.3
  of truth on N=30 synthetic data, seed=42)
- `wasmparser` validation confirming the artifact uses no wasm-gc opcodes
- `generated quantities` RNG output checked against each distribution's
  support/range (property-based, not exact-value, since it's random)
- Step-by-step sampling (`startStepSampling`/`stepDraw`) matches full
  `sample()` behavior and correctly restores `logProbGrad`/`sample`
  afterward

[Unreleased]: https://github.com/habakan/stanwasm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/habakan/stanwasm/releases/tag/v0.1.0
