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

## What works today

A subset. Everything in the TODO column is a clean load-time or evaluation
error — never a model that silently samples something else. The gaps are
worked through below, ordered by effort.

| | Supported | TODO |
|---|---|---|
| Continuous | `normal`, `std_normal`, `exponential`, `half_normal`*, `cauchy`, `student_t`, `lognormal`, `gamma`, `beta` | |
| Discrete | `bernoulli`, `bernoulli_logit`, `poisson`, `neg_binomial_2`, `categorical` | |
| Multivariate | `multi_normal_cholesky`, `multi_normal` (full covariance), `lkj_corr_cholesky`, `dirichlet`, `multinomial` | |
| Scalar constraints | `lower`, `upper`, `lower_upper` — element-wise on vectors | |
| Vector shape | `simplex`, `ordered`, `positive_ordered`, `unit_vector` | |
| Matrix constraints | `cholesky_factor_corr`, `cholesky_factor_cov`, `cov_matrix`, `corr_matrix` | |
| Blocks | `data`, `parameters`, `transformed parameters`, `model`, `generated quantities`, `functions` | |
| Statements | `for`/`while`, `if`/`else`, `break`/`continue`, `y ~ dist(...)`, `target += expr`, indexed assignment, ternary `?:` | |
| Operators | arithmetic, comparison, logical, `^`, matrix product (`X * beta`, `A * B`), element-wise `.*` `./` `.^` | |
| `_rng` | scalar draws for every distribution above, vectorized over container arguments, plus `uniform_rng`, `dirichlet_rng`, `multi_normal_cholesky_rng` | `lkj_corr_cholesky_rng` |
| Math functions | `log`, `exp`, `sqrt`, `abs`, `pow`, `square`, `lgamma`, `logit`, `inv_logit`, `tanh`, `Phi`, `sum`, `mean`, `segment`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `dot_product`, `size`, `num_elements`, `rows`, `cols`, `rep_vector`, `rep_matrix` | `log10`, `norm` |

\* `half_normal` is not a Stan distribution — stanc rejects it. Write
`real<lower=0> tau; tau ~ normal(0, s);` for a model that also runs under Stan.

Four caveats the table can't carry:

- **Branches on a sampled parameter.** `if`/`while` conditions that depend on a
  parameter work in `generated quantities` (re-evaluated natively per draw) but
  are a compile-time error in `model`/`transformed parameters` — NUTS traces
  that block once and replays the same graph for every draw.
- **Posterior prediction works both ways.** The loop form and the vectorized
  `normal_rng(rep_vector(mu, N), sigma)` both do. As in Stan, a scalar `_rng` given
  only scalars returns a scalar — it is the container argument that vectorizes.
- **The AOT path covers fewer ops than the interpreter.** `sample()` (tape
  replay) evaluates everything the runtime produces; `compile()`/`sampleViaAot`
  have no emitter for `tan`/`asin`/`acos`/`atan` (hence `atan2`), `erf`/`erfc`
  or `digamma`, and report `CodegenError::UnsupportedOp` rather than emitting a
  module that traps. Adding them needs new math imports on the host side.
- **`transformed data` parses but does not memoize.** Its statements fold into
  `model`, so they re-run every trace and its variables are invisible from
  `generated quantities`.
- **Stan's static typing is honored where it changes results.** `int / int` is
  integer division (`N / 2` with `N = 3` is `1`), and `^` binds tighter than
  unary minus and associates right (`-a^2` is `-(a^2)`, `2^3^2` is `512`).

`X * beta` with data on the left is the form to write — it records one node
per row rather than one per multiply. An array of vectors on the left of a
multivariate density is a load-time error pointing at the loop form, which is
what that case wants.

The `data` block is checked against the supplied JSON when the model loads — a
missing field, a wrong length, a non-integral `int`, or a violated
`<lower=...>`/`<upper=...>` bound is an error rather than a model that samples
the wrong thing.

## Remaining language gaps, roughly ordered by effort

### `functions { ... }` — what is and is not supported

Calls are inlined at the call site while tracing, which fits the "trace once, fully
unroll" design and means the AOT path gets them for free — it consumes the tape, not
the AST. Supported: scalar, `vector` and `matrix` arguments (unsized in the signature,
as Stan writes them), local variables, one function calling another, and gradients
flowing through the call.

Not supported, each a clean error rather than a wrong answer:

- **Recursion**, which Stan allows. Inlining a recursive call expands forever, so
  self- and mutual recursion are rejected with `EvalError::RecursiveCall`.
- **`void` functions** and the **`data` qualifier** on arguments, both parse errors.
- The **`_lp` / `_rng` / `_lupdf` suffix rules**. A user function ending in `_rng` is
  not given the RNG, and one ending in `_lp` has no access to the accumulator, so
  neither does what its name promises in Stan.

### Missing constraint transforms — incremental, not hard

- `multi_normal` (full covariance), `multinomial` and `categorical` are done —
  see `stanwasm-runtime/src/distributions.rs`.
- `lkj_corr_cholesky_rng` (needs the onion-method sampling algorithm —
  self-contained, known algorithm, just not implemented yet)
- Each of these follows the same pattern as the distributions already
  implemented in `stanwasm-runtime/src/distributions.rs`: add the log-pdf/pmf
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

## What the tape records, and what it costs

Every tape node holds one `f64`, and for a while that meant a matrix-vector
product was recorded one multiply at a time: `y ~ normal(X * beta, sigma)` over
`matrix[N,K]` took `2K` nodes per row. It no longer does. Two opcodes now stand
for a whole run:

- a **contraction**, for `X * beta` with data on the left — one node per row
  instead of `2K`
- a **reduction**, for the total a vectorised statement accumulates — one node
  instead of a chain of adds, and summed in the same order the chain was, so
  no log density changes value

Neither widens the tape row. The run's shape sits in a table beside the node
and the node holds a handle, so a model with no contraction in it pays nothing.

The emitter re-rolls the repeated part of a vectorised statement into a wasm
loop, lays that loop's values out so consecutive repeats are adjacent, and runs
**two repeats at a time as `f64x2`** where every slot it touches moves by one or
not at all. A density that calls a host math import per observation — anything
built on `exp` or `log` — stays scalar, because those have no lane-wise form.

Per gradient at N=5000, Node 22 on Apple arm64, against the same commit's
predecessor:

| | before | after |
|---|---:|---:|
| `y ~ normal(X * beta, sigma)`, K=4 | 52.6 µs | **17.4 µs** |
| `y ~ normal(alpha + beta * x, sigma)` | 40.8 | **21.0** |
| hierarchical `mu[g[i]]` | 33.6 | **25.6** |
| `y ~ bernoulli_logit(...)` | 171.1 | **163.6** |

The trace for the first went from 70,040 nodes to 30,040, and the scratch
buffer with it.

**The emitted module now uses the fixed-width SIMD proposal.** Every engine
that ships WebAssembly today has it — Safari since 16.4 — but an embedder that
disables the proposal will reject the module.

### Checked against CmdStan

`make compare-cmdstan` compiles the models in `ts/tests/bench_models.ts` with a
CmdStan of your choosing, evaluates both at the same point in the unconstrained
space, and compares. Across twelve models the gradients agree to **3e-13**
relative at worst, and the log densities differ by exactly the normalising
constants Stan's `~` drops — checked at two points, since a constant offset is
the expected difference and one that moves with the point is a bug.

That harness also times both — CmdStan under static HMC, whose leapfrog count
per draw is fixed and where every leapfrog is one gradient. The AOT gradient is
faster than CmdStan's on ten of the twelve, which is the trace-once design
paying off: Stan builds a tape on every evaluation, this builds one at compile
time and emits code with no allocation and no dispatch. The two it loses are `bernoulli_logit` and `student_t`, where
stan-math has an analytic special case and this project expands the density
into elementary operations; each of those spends most of its time in one
transcendental call per observation.

### What is still open

- A distribution whose density is built from `log(1 + exp(x))` pays two
  transcendental calls per observation where one op with an analytic
  derivative would do. That is the gap on the two models above.
- A hand-written module of the same shape reaches 8.9 µs on the matrix model
  against the emitter's 17.4, by fusing the mean into the density loop. Merging
  those two loops was measured on its own and was a *loss*, so the remaining
  distance is not simply available.

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

- **`vector * vector` is element-wise, where Stan rejects it.** `*` now dispatches
  on shape, so matrix-vector and matrix-matrix products work, but two vectors still
  multiply element-wise instead of erroring the way Stan does. Stan spells that `.*`,
  which is unimplemented, so nothing yet distinguishes the two.

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
