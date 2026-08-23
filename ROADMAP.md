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
correctness bugs, verified independently and fixed across three rounds — see
`[Unreleased]` in `CHANGELOG.md` for the full list. The last round closed the
"silently wrong instead of erroring" family: `^` precedence/associativity,
integer division, array element constraints, unhandled constraint types,
matrix parameter shape, mismatched operand lengths, and `data`-block
validation. Still open:

- **Indexed assignment is a no-op.** `mu[i] = expr;` doesn't write into the
  vector — it errors cleanly (`EvalError::UnsupportedAssignmentTarget`)
  instead of silently discarding the write, but the feature itself isn't
  implemented. Needs an lvalue-resolution helper that walks `Expr::Index`
  chains down to the root `Env` binding and writes back through them.
  This is the main thing blocking the standard posterior-predictive idiom
  in `generated quantities`.

- **Scalar `_rng` functions don't vectorize.** `normal_rng(mu, sigma)` with a
  vector `mu` is an error rather than a vector of draws. Together with the
  item above, `vector[N] y_rep = normal_rng(mu, sigma);` — the reason most
  people write a `generated quantities` block at all — doesn't work. Both
  are small next to the payoff: broadcasting in `rng::dispatch` plus lvalue
  resolution in `eval::eval_stmt`.

- **`Val::Vec` is used for both math vectors and matrices, and `*` is always
  element-wise.** `X * beta` (matrix-vector product, a standard Stan idiom)
  is now a clean `ShapeMismatch` error rather than a wrong answer, but real
  matrix multiplication still isn't available to user-written model code —
  the internal distributions that need it (`multi_normal_cholesky`, etc.)
  route around it via dedicated helpers in `matrix.rs`. Needs a type-aware
  dispatch (or a distinct operator) before matrices support general linear
  algebra syntax.

- **`transformed data { ... }` is folded into `model`.** Its statements are
  appended to the model block, so they re-run on every trace instead of once
  at load, and the variables they define aren't visible from `generated
  quantities` (`undefined variable`). Needs its own `StanProgram` field,
  evaluated once into `data_env` during `Model::parse_and_load`.

- **`sampleViaAot` doesn't check the AOT module's `n_params` against the
  current model's.** Passing an AOT module compiled for a different model
  shape reads/writes past the shared buffer. Needs a dimension check before
  the first `aot_logp` call.

- **The AOT path has a hard size ceiling.** Two wasm locals per tape node
  against V8's 50,000-local limit caps it at ~25,000 tape nodes (`N ≈ 2,000`
  for a vectorized regression). `compile()` reports this cleanly now and
  callers can fall back to `sample()`, but lifting it means spilling
  intermediates to linear memory instead of locals, or splitting the
  emitted function.

- **Loop-form model building is slow.** `for (i in 1:N) y[i] ~ ...` clones
  the whole vector per element (`Val::Vec` + `.get(n).cloned()`), so tracing
  is O(N²): ~1.6 s at N = 20,000 where the vectorized form takes ~5 ms.
  Needs indexing that borrows instead of cloning.

- Smaller items not yet triaged: the `logProbGrad`/`sample` API mixes
  `n_params` (snake_case) with otherwise-camelCase names; `UnknownChar`
  renders a non-ASCII byte as mojibake because it casts a byte to `char`;
  and calling `logProbGrad`/`sample` mid-step-sampling reports
  `internal: compiled missing` instead of pointing at `finishStepSampling`.
