# Roadmap

Informal notes on scope and remaining gaps — not a commitment or a schedule.
See [`CHANGELOG.md`](CHANGELOG.md) for what has already shipped.

## Positioning

This project is **not** an attempt to replace cmdstan or Stan Playground, and
Stan-power-user adoption isn't the target — the language subset makes that a
losing comparison. The realistic use case is narrower: an npm-embeddable,
server-free Stan engine for things like interactive teaching content,
explorable blog-post demos, and privacy-constrained client-side analytics
(data never leaves the browser). Feature priority below should track that —
depth of embeddability and honesty about the subset matter more than chasing
full language coverage for its own sake.

## Remaining language gaps, roughly ordered by effort

### `functions { ... }` (user-defined functions) — the hard one

- Parser/AST already capture this block (`StanProgram.functions`); nothing
  downstream consumes it yet.
- The engine's whole model is "trace once, fully unroll" (both the AST-eval
  tape-replay path and AOT codegen). As long as recursion isn't supported,
  a function call can just be inlined at its call site during tracing —
  this fits the existing design, no architectural rework needed. Stan
  models essentially never rely on recursion, so that restriction should be
  fine in practice.
- The real work is the type system: binding `real/vector/matrix/int` (+
  array/size) arguments, return values, and function-local scoping, in
  **both** `stan-runtime::eval` (interpreter) and the AOT tape-unrolling
  path in `stan-codegen`.
- Rough size: comparable to or a bit larger than the `generated quantities`
  work (spans parser → runtime → codegen → wasm-api).

### Missing distributions & constraint transforms — incremental, not hard

- Distributions: `multi_normal` (full covariance, not just Cholesky),
  `multinomial`, `categorical`
- Constraint/shape transforms: `cov_matrix`, `cholesky_factor_cov`,
  `corr_matrix`, `unit_vector`
- `lkj_corr_cholesky_rng` (needs the onion-method sampling algorithm —
  self-contained, known algorithm, just not implemented yet)
- Each of these follows the same pattern as the ~15 distributions already
  implemented in `stan-runtime/src/distributions.rs`: add the log-pdf/pmf
  (and Jacobian, if it's a constrained type) using the existing tape/matrix
  ops. Roughly half a day to a day each, no structural blocker.

### `print()` / `reject()` / profiling — small-to-medium plumbing

- Now that `if`/`while` evaluate for real, these are plausible to wire up.
- Caveat: the AOT path traces once and reuses a fixed graph, so per-draw
  varying `print()` output only really makes sense on the tape-replay path,
  not AOT — that mismatch needs a decision, not just an implementation.

### Pathfinder / ADVI / fixed_param samplers — categorically bigger

- This isn't "add a feature" — it's a new inference algorithm on par with
  the original NUTS integration. Not comparable in scope to the items
  above; would be its own initiative if ever pursued.

## Correctness follow-ups from the pre-launch review

A pre-launch review (external, via another agent) surfaced several
correctness bugs. The "silently wrong instead of erroring" family (unknown
names, unsupported statements, parameter-dependent branching, panics on
vectorized math, invalid RNG params) is fixed — see `[Unreleased]` in
`CHANGELOG.md`. Still open:

- **Indexed assignment is a no-op.** `mu[i] = expr;` doesn't write into the
  vector — it now errors cleanly (`EvalError::UnsupportedAssignmentTarget`)
  instead of silently discarding the write, but the feature itself isn't
  implemented. Needs an lvalue-resolution helper that walks `Expr::Index`
  chains down to the root `Env` binding and writes back through them.
- **`lkj_corr_cholesky_lpdf` is wrong.** For K=2 the implementation reduces
  to exactly 0 for any input — the `(K-1-k)` per-row weight is applied as a
  single `(2η−2)` factor multiplied across the whole sum, rather than added
  to each row's own weight before that row's `log(L_kk)` term. Needs a
  rederivation against Stan's actual formula, not just a K=2 patch.
- **Constraint Jacobians silently pass through for unhandled types.**
  `constraints.rs`'s fallback arm returns a zero Jacobian for any
  `StanType` not explicitly matched (e.g. nested `array[N] real<lower=..>`
  in some shapes). Needs auditing which type/constraint combinations
  actually reach that fallback and either implementing them or rejecting
  them explicitly.
- **`Val::Vec` is used for both math vectors and matrices, and `*` is
  always element-wise.** Writing `X * beta` (matrix-vector product, a
  standard Stan idiom) with the generic `*` operator does not do a real
  matrix multiply — internal distributions that need real matrix ops
  (`multi_normal_cholesky`, etc.) route around this via dedicated helpers
  in `matrix.rs`, but user-written model code has no way to ask for real
  matrix multiplication. Needs a type-aware dispatch (or a distinct
  operator) once matrices are meant to support general linear algebra
  syntax, not just Cholesky-factor plumbing.
- **A failed `sample()`/`sampleViaAot()`/`startStepSampling()` call leaves
  the model unusable.** `self.compiled` is taken out before nuts-rs
  initializes and is only restored after a successful run; an init failure
  (e.g. a bad `init` position) returns early without restoring it, so
  `logProbGrad`/`sample` on the same `StanModel` instance then fail with
  "internal: compiled missing" — a clean error now, not a crash, but the
  instance shouldn't need to be discarded over one bad call. Needs a
  restore-on-error path (or restoring before rather than after the nuts-rs
  call, since re-tracing is cheap).
- **`sampleViaAot` doesn't check the AOT module's `n_params` against the
  current model's.** Passing an AOT module compiled for a different model
  shape reads/writes past the shared buffer. Needs a dimension check before
  the first `aot_logp` call.
- Smaller items not yet triaged: parser operator-precedence edge cases
  (`^` associativity, unary-minus precedence interacting with it), the
  `logProbGrad`/`sample` API mixing `n_params` (snake_case) with everything
  else camelCase, and `num_warmup`/`num_draws` not being validated against
  negative/huge values before allocating.
