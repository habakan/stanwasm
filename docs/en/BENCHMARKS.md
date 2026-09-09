# Benchmarks

Two harnesses available:

| Harness | What it measures | Run |
|---------|------------------|-----|
| native Rust release | Tape replay & nuts-rs without any wasm overhead | `make bench-native` |
| Node.js V8 + wasm32 | Same code path users see in the browser | `make bench` |

The Node bench compares **two sampling paths** end-to-end:

- `sample` — tape replay inside the same wasm bundle. Walks the recorded autodiff tape per leapfrog step.
- `sampleViaAot` — calls into a per-model AOT-compiled wasm bound via `setAotExports`. The AOT module shares stanwasm's linear memory (zero-copy), and V8 JITs its fully-unrolled forward+backward pass.

## Apple Silicon, 2026-05-10 (older, whole-run timings)

### Node.js V8 + wasm32 (the path users see)

| case                 | replay ms | **AOT ms** | speedup |
|----------------------|----------:|-----------:|--------:|
| poisson_regression (2 p)   |       9.9 |    **5.2** |   1.91× |
| eight_schools_ncp (10 p)   |      16.0 |    **5.5** |   2.90× |
| linear_regression (3 p)†   |     248.8 |   **44.3** |   5.62× |

† `linear_regression` posterior is sharp (σ ≈ 0.1 on N=30 synthetic data); NUTS adapts to a small step and takes many leapfrog steps. Not representative.

### Native Rust release

| case                 | AST µs | replay µs | AOT (wasmi) µs | sample ms |
|----------------------|-------:|----------:|---------------:|----------:|
| linear_regression    |   ~6.0 |      1.55 |           1.71 |      12.5 |
| poisson_regression   |   ~4.5 |      0.37 |           0.58 |       8.1 |
| eight_schools_ncp    |   ~6.4 |      0.84 |           0.91 |      13.5 |

`AOT (wasmi)` is meaningful only as an internal sanity check — `wasmi` is an interpreter, not a JIT, and is ~10× slower than V8 on the AOT path.

## Per-gradient cost, both paths — 2026-09-06

`make bench-gradients` (N=5000, Node 22, min of 30 interleaved rounds). This is
the number the README quotes, and `grads` is the check that the two paths agree
on the gradient itself rather than only on how long they take.

| model | params | replay µs | AOT µs | speedup | grads |
|---|---:|---:|---:|---:|---|
| linreg | 3 | 163.81 | 20.79 | 7.88x | agree |
| logistic | 2 | 212.58 | 131.69 | 1.61x | agree |
| poisson | 2 | 374.44 | 124.96 | 3.00x | agree |
| neg_binomial | 3 | 355.33 | 162.93 | 2.18x | agree |
| student_t | 4 | 293.23 | 108.09 | 2.71x | agree |
| gather | 9 | 210.30 | 48.64 | 4.32x | agree |
| two_level | 12 | 169.51 | 39.56 | 4.28x | agree |
| matrix_k4 | 5 | 169.22 | 17.23 | 9.82x | agree |
| matrix_k16 | 17 | 231.58 | 38.82 | 5.97x | agree |
| mvn_cholesky | 11 | 63.54 | 9.66 | 6.58x | agree |
| eight_schools | 10 | 0.97 | 0.08 | 12.21x | agree |
| binomial | 3 | 492.33 | 334.04 | 1.47x | agree |
| count_mix | 7 | 192.99 | 89.35 | 2.16x | agree |
| cov_builders | 11 | 58.54 | 17.05 | 3.43x | agree |
| glm | 6 | 425.14 | 170.13 | 2.50x | agree |
| funnel | 10 | 0.61 | 0.08 | 7.27x | agree |

**1.5x to 12x across the sixteen.** The narrowest margins are the ones that
spend their time in a host transcendental rather than in dispatch: `binomial`
at 1.47x and `logistic` at 1.61x are one `log1p_exp` per observation, and
unrolling the loop does not make the call cheaper. The widest are the ones
where dispatch was the cost — `eight_schools` and `matrix_k4` do little
arithmetic per node.

The `sample ms` numbers below are older (2026-05-10) and cover three cases
rather than sixteen; they are kept because they measure a whole sampling run
rather than one gradient.

## Reproducing

```bash
# Native
make bench-native

# Node.js (requires wasm-pack on PATH; rebuilds ts/pkg/ if it is stale)
make bench
```

## Architecture details

`sample` (replay) lives entirely inside `stanwasm_bg.wasm`. nuts-rs gets its log_prob_grad from `Compiled::log_prob_grad`, which dispatches on `Op` per tape node. V8 JIT-compiles this dispatch loop, but cannot inline across the dispatch.

`sampleViaAot` lives across two wasm modules:

```
┌──────────────────────────────┐    shared linear memory    ┌──────────────────────────┐
│  stanwasm_bg.wasm       │ ◄────────────────────────► │  AOT model wasm          │
│                              │                             │                          │
│  - parser, codegen           │                             │  - imports "stan.memory" │
│  - nuts-rs sampler driver    │                             │  - exports log_prob_grad │
│  - shims math fns to JS      │                             │    (fully unrolled)      │
│                              │                             │                          │
│  imports aot_logp via JS     │ ──── one JS shim call ────► │  V8 JITs the unrolled    │
│  bridge (set_aot_exports)    │                             │  forward + backward      │
└──────────────────────────────┘                             └──────────────────────────┘
```

The JS bridge between the two modules is a 5-line snippet (`crates/stanwasm/js/aot_bridge.js`) that V8 inlines aggressively after warmup.

## Caveats

- Each row is one run of 2,000 NUTS draws. Variance ±5–10 % between runs. `linear_regression` is the most variable (sharp posterior).
- `sample ms` includes warmup adaptation. For longer chains the per-draw rate stabilizes lower.
- Math import shims (lgamma, digamma, phi) are JS-side polynomial approximations; precision matches `tapewasm_autodiff` Rust functions.
- AOT path requires the host to provide math imports; missing imports throw at instantiate time.
- The AOT path has a size ceiling: the emitted function needs two wasm locals per tape node and V8 caps a function at 50,000 locals, so a fully-unrolled trace above ~25,000 nodes cannot be compiled. For a vectorized linear regression that is roughly `N ≈ 2,000`. `compile()` reports this as `CodegenError::TooManyLocals`; use `sample()` (tape replay), which has no such limit. See [ARCHITECTURE.md](../../ARCHITECTURE.md#size-limit-on-the-aot-path).
