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
