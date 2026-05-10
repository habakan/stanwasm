# Benchmarks

Reproduce locally with:

```bash
cargo run --release -p stan-cli -- bench all
```

The CLI times four execution paths per Stan model:

| Path        | What it measures |
|-------------|------------------|
| `AST µs`    | `Model::log_prob_grad` — fresh AST trace per call. Reference oracle, expected to be slow. |
| `replay µs` | `Compiled::log_prob_grad` — recorded-tape replay. Used by `StanModel::sample`. |
| `AOT µs`    | `stan-codegen` output run inside `wasmi`. Mirrors browser AOT path. |
| `sample ms` | `StanModel::sample` end-to-end (n_warmup=1000, n_draws=1000, seed=42). |

`µs/call` numbers are means over 10,000 iterations; `sample ms` is wall-clock for a single sampling run.

## Results — Apple Silicon, native release build, 2026-05-10

| case                 | AST µs | replay µs | AOT µs | sample ms |
|----------------------|-------:|----------:|-------:|----------:|
| linear_regression    |   ~6.0 |      1.55 |   1.71 |      12.5 |
| poisson_regression   |   ~4.5 |      0.37 |   0.58 |       8.1 |
| eight_schools_ncp    |   ~6.4 |      0.84 |   0.91 |      13.5 |

### Observations

1. **Replay is 4–12× faster than fresh AST trace.** Recording the tape once and re-evaluating its op array avoids `Val` allocation and AST dispatch on every call. This is the path the in-process `StanModel::sample` uses.

2. **AOT-via-wasmi is roughly the same speed as native replay.** wasmi is an interpreter, so it pays per-instruction dispatch overhead that wipes out the structural advantage of fully-unrolled wasm. In V8 (browser, Node.js) the AOT path will JIT to native and beat replay; in `wasmi` it does not.

3. **Per-call lpg cost is sub-microsecond on the simple models** — the bulk of `sample ms` is NUTS leapfrog overhead, not the user's model.

## Comparison with MoonBit reference

The MoonBit ancestor (`stan-wasm`) runs its WAT AOT path under Node.js (V8 JIT). Numbers from `stan-wasm/tests/results/benchmark_nutsrs_aot.json`:

| model                 | MoonBit WAT AOT walltime | Rust `sample ms` (native) |
|-----------------------|-------------------------:|---------------------------:|
| logistic_regression (2 params)    | 6.0 ms |                       — |
| poisson_regression (2 params)     |    —   |                  8.1 ms |
| eight_schools (10 params)         | 5.5 ms |                 13.5 ms |
| gaussian_mixture                  | 37.4 ms |                      — |

The 2-parameter case is within ~35 % of the MoonBit number despite running through tape replay rather than V8-JIT'd unrolled wasm. The 10-parameter `eight_schools_ncp` is roughly 2.5× slower than MoonBit's WAT AOT — closing this gap requires either:

- **(near-term)** wiring `StanModel::sample` to call AOT model wasm inside the same wasm bundle (the codegen output already exists; just not invoked from `sample`), so V8 can JIT the unrolled forward+backward pass;
- **(longer-term)** SIMD or vectorized leapfrog inside `nuts-rs`.

## Caveats

- `AST µs` numbers are noisy (4–7 µs range across runs). The other paths are stable to within ~5 %.
- Running on an idle laptop. Background load skews `sample ms` more than the per-call numbers.
- These are **native Rust** numbers. Browser delivery is `wasm32` inside V8 — expect roughly the same magnitude on the AOT path, slightly slower (~1.2–1.5×) on tape replay.
- `sample ms` includes warmup adaptation, which dominates for small `n_draws`. For larger draw counts the pure-sampling rate is a more honest comparison metric, and we should switch to `ESS/sec` once the ESS estimator from MoonBit is ported.

## Next steps for fair MoonBit comparison

Run the Rust `stan-wasm-api.wasm` bundle inside Node.js with the same harness MoonBit uses, so the V8 JIT applies to both. Tracked under Phase 7 (cutover).
