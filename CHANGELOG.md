# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Matrix products.** `*` now dispatches on operand shape, so `X * beta` and `A * B`
  work in model code instead of erroring; anything else stays element-wise. Verified
  against the hand-expanded loop form, which produces identical estimates, and the AOT
  path compiles it. Two vectors still multiply element-wise, where Stan would reject
  the expression — see `ROADMAP.md`.

## [0.1.1] — 2026-09-02 (npm only)

Published to npm as `stanwasm@0.1.1`. The crates are still at 0.1.0: the
Safari fix below lives in a `[patch]`, which Cargo does not carry into a
published crate, so a crates.io release would not have carried it. The next
crates.io release waits for nuts-rs to ship the fix and drops the patch.


### Added

- **stanwasm now runs on Safari and on iOS/iPadOS.** WebKit rejects a module
  containing relaxed SIMD opcodes at validation time, so the bundle previously
  failed to instantiate at all on those platforms. The opcodes came from
  `nuts-rs` → `faer` → `pulp`. pulp made them optional in
  [pulp#30](https://github.com/sarah-quinones/pulp/pull/30) (a `relaxed-simd`
  feature, on by default) and released it in 0.22.3; faer already opts out.
  nuts-rs was the last crate in the graph pulling pulp's default features, and
  Cargo cannot subtract a transitive default feature.
  [nuts-rs#76](https://github.com/pymc-devs/nuts-rs/pull/76) fixed that upstream
  and was merged; the workspace points at the merge commit until it reaches
  crates.io. The pin covers workspace builds and therefore the npm
  package, which ships the prebuilt wasm; the `stanwasm` crate published to
  crates.io still resolves plain nuts-rs, because Cargo does not carry
  `[patch]` into a published crate.

  Verified under Playwright: Chromium 151, Firefox 153 and WebKit 26.5 all
  instantiate and sample, with posterior means agreeing across engines, and the
  gallery renders in WebKit at an iPhone viewport. Dropping relaxed SIMD costs
  nothing measurable at these parameter dimensions (-4.3% and 0.0% on two
  models, 1000 warmup + 1000 draws, median of 7 runs, Chromium); the bundle
  grows about 2 KB.

- Three previously-TODO distributions: `multi_normal` (full covariance,
  Cholesky-decomposed internally and routed through the existing
  `multi_normal_cholesky` math), `multinomial`, and `categorical`. See
  `stanwasm-runtime/src/distributions.rs`.

### Fixed

- The gallery no longer hangs on "Loading WebAssembly bundle…" when the module
  fails to instantiate. `init()` had no `.catch()`, so a rejected promise left
  the loading state up forever with the error swallowed — which is exactly how
  the relaxed-SIMD failure presented on iOS: a permanent spinner and no clue
  why. The error is now shown, with a specific explanation when it is the
  relaxed-SIMD rejection.

### Documentation

- README and the npm README now describe Safari and iOS/iPadOS as supported,
  with the one caveat that applies to the crates.io crate. An earlier revision
  of this section documented the platform as unsupported; that was accurate
  when written and is superseded by the fix above.

## [0.1.0] — 2026-08-28 (alpha)

Initial alpha release: enough Stan to sample linear regression, logistic
regression, Poisson regression, eight schools (non-centered), and
multivariate-LKJ-style models end-to-end in the browser — plus a
`generated quantities` block, step-by-step sampling for live
visualization, and a tabbed examples gallery.

The *Fixed*, *Changed* and *Security* sections below record hardening done
before this first public tag — three rounds of pre-release correctness
review — not regressions from an earlier published version. There is no
earlier published version.

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

- The examples gallery is deployed to GitHub Pages from `main`
  ([habakan.github.io/stanwasm](https://habakan.github.io/stanwasm/)), so the
  demos can be tried without a Rust, wasm-pack and Node toolchain. Rebuilt on
  any change to the crates as well as to the app — the wasm the page loads is
  built from the Rust sources.
- Every publishable crate now sets `keywords`, `categories` and `readme`, and
  each has its own `README.md`; the npm package has one too. Without them a
  crates.io or npm page renders as a title and nothing else.

### Changed

- `wasm-encoder` 0.220 -> 0.258, and the `wasmparser` dev-dependency with it.
  `Instruction::F64Const` takes an `Ieee64` rather than an `f64` now, which is
  the whole of the break — the AOT codegen is otherwise unchanged, and
  `aot_vs_oracle` still matches the native oracle on every model it covers. The
  bundle grows 477 KB -> 494 KB raw, but only 184.0 KB -> 184.8 KB gzipped:
  `wasm-encoder` runs *inside* the browser bundle, so its own size is part of
  the payload.
- `wasmi` 0.32 -> 1.1, the AOT reference interpreter behind `aot_vs_oracle`
  and `stanwasm-cli bench`. Two breaks, both mechanical: `MemoryType::new`
  returns a `MemoryType` rather than a `Result`, and `Linker::instantiate(..)
  .start(..)` is now a single `instantiate_and_start`. Neither crate reaches
  the browser bundle — `stanwasm-cli` is `publish = false`, and
  `stanwasm-codegen` takes wasmi as a dev-dependency only.
- `chacha20` 0.10.0 -> 0.10.2 and `spin` 0.9.8 -> 0.9.9 in `Cargo.lock`; both
  earlier versions were yanked. Lockfile only, no manifest change.
- Every crate is renamed to a `stanwasm` prefix: `stanwasm-ast`,
  `stanwasm-parser`, `stanwasm-autodiff`, `stanwasm-runtime`,
  `stanwasm-codegen`, `stanwasm-cli`, and `stan-wasm-api` becomes plain
  `stanwasm` — the same name as the npm package. crates.io is a flat namespace
  and never frees a name once taken, so shipping `stan-parser` and
  `stan-runtime` would have claimed generic Stan names that read as belonging
  to Stan itself. The wasm-bindgen output moves with it:
  `stan_wasm_api_bg.wasm` is now `stanwasm_bg.wasm`. Nothing was published
  under the old names, so no import path in the wild breaks.
- The five crates below `stanwasm` now say in their `description` that they
  are internal and carry no API stability guarantee. They reach crates.io only
  because cargo requires a dependency to be on the registry before its
  dependent can be.
- `make package` now also asserts the npm tarball carries the wasm, not just
  the licence. It was the one manual `npm pack --dry-run` step in
  RELEASING.md, and npm is the artifact most people actually install.
- Build commands moved into a `Makefile`; `scripts/build-wasm.sh` is gone.
  `make wasm` replaces it, and `make` on its own lists every target. Building
  from source is otherwise unchanged — the underlying `wasm-pack` invocation
  is the same one.

- The npm entry point is `ts/index.js` with a hand-written `ts/index.d.ts`,
  and `package.json` declares `types`/`exports`. It was `index.ts`, which
  broke plain-JS consumers, bundlers, and Node without
  `--experimental-strip-types`.
- Bundle size: `~466 KB` after `wasm-opt -Oz` (`~180 KB` gzipped), up from
  `~431 KB` — the cost of the validation and the error messages above.
  README, CITATION.cff and ARCHITECTURE.md all still said 431 KB.
- Declared MSRV: Rust 1.88 (`rust-version` in the workspace `Cargo.toml`),
  set by `nuts-rs` 0.18's edition-2024 + let-chains usage and verified with
  `cargo +1.88 test --workspace`. `CONTRIBUTING.md` previously said 1.80,
  which does not build.

### Fixed

- Published artifacts now carry the Apache-2.0 licence text. `cargo package`
  only collects files inside a crate's own directory and `npm pack` behaves
  the same way, so the `LICENSE` at the repo root reached neither the six
  `.crate` files nor the npm tarball — every artifact declared a licence it
  did not include. `make package` now fails if any of them is missing.
- The gallery no longer ships the wasm twice. Passing an explicit URL out of
  `public/` did not stop Vite emitting an asset for wasm-bindgen's own
  `new URL("stanwasm_bg.wasm", import.meta.url)`, so the built site
  carried both copies — 955 KB where 477 KB is used, and the copy actually
  fetched had no content hash. The default resolution is used now.
- The pre-publish packaging check now runs `cargo package --workspace` instead
  of one `cargo package -p` per crate. The per-crate form cannot pass before
  the first crates.io release, because each manifest resolves its siblings
  from the registry rather than from `path` — so the check that exists to
  catch a path dependency missing its `version` requirement failed for an
  unrelated reason at exactly the moment it was needed.

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
- `stanwasm-codegen` emitted AOT modules with two wasm locals per tape node and
  no ceiling, so a model whose trace exceeds ~25,000 nodes (roughly
  `N ≈ 2,000` for a vectorized regression) produced a module the browser
  rejects with `CompileError: local count too large`. `compile()` now
  returns `CodegenError::TooManyLocals` up front, and the limit is
  documented in `ARCHITECTURE.md` and `docs/en/BENCHMARKS.md`.
- `num_warmup + num_draws` was summed as `u32` before widening to `u64`, so
  a large pair wrapped to a different run length.
- The operand-shape check compared matrices by row count alone, so
  `matrix[2,3] + matrix[2,4]` passed and the wider operand's columns were
  zipped away. `Shape::Matrix` carries the column count now, and a ragged
  container never compares equal.
- `int` parameters, unsupported constraint types, and bad size expressions
  now name the offending declaration and say what to do
  (`parameter \`k\` is declared \`int\`. Stan parameters must be continuous…`)
  instead of reporting an internal constraint-table miss.
- `array[N] vector[K] y; y ~ multi_normal_cholesky(mu, L);` reported a size
  mismatch between the N array rows and the K-long `mu`. It now says the
  multivariate form isn't vectorized here and gives the loop form.
- `logProbGrad`/`sample` during a step-sampling session reported
  `internal: compiled missing`. They now say the session holds the compiled
  model and point at `finishStepSampling()`.
- The lexer's unknown-character error cast a byte to `char`, so a non-ASCII
  character was reported as mojibake. It decodes the character.

### Security

- CI pins every third-party action by commit SHA instead of by mutable tag, and
  declares `permissions: contents: read`. A tag can be re-pointed by whoever
  controls the action repository, which would have run arbitrary code with this
  workflow's token.
- `examples/gallery` moves to Vite 7 / `@vitejs/plugin-react` 5, clearing the
  esbuild dev-server advisory (GHSA-67mh-4wv8-2f99) and the Vite 5 path
  traversal reports. Dev-server-only and dev-dependency-only, so nothing
  shipped was affected. `npm audit` is clean; the built gallery renders and
  samples unchanged.
- Added `SECURITY.md` (reporting channel and threat model) and
  `.github/dependabot.yml` (Cargo, npm, GitHub Actions).

### Architecture

- Seven Rust crates: `stanwasm-ast`, `stanwasm-parser`, `stanwasm-autodiff`,
  `stanwasm-runtime`, `stanwasm-codegen`, `stanwasm`, `stanwasm-cli`
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

[0.1.0]: https://github.com/habakan/stanwasm/releases/tag/v0.1.0
