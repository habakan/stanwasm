# Benchmarks

Two harnesses available:

| Harness | What it measures | Run |
|---------|------------------|-----|
| native Rust release | Tape replay & nuts-rs without any wasm overhead | `cargo run --release -p stan-cli -- bench all` |
| Node.js V8 + wasm32 | Same code path users see in the browser | `cd ts && npm run build:wasm && npm run bench` |

The Node bench compares **two sampling paths** end-to-end:

- `sample` — tape replay inside the same wasm bundle. Walks the recorded autodiff tape per leapfrog step.
- `sampleViaAot` — calls into a per-model AOT-compiled wasm bound via `setAotExports`. The AOT module shares stan-wasm-rs's linear memory (zero-copy), and V8 JITs its fully-unrolled forward+backward pass.

## Apple Silicon, 2026-05-10

### Node.js V8 + wasm32 (the path users see)

| case                 | replay ms | **AOT ms** | speedup | MoonBit ref |
|----------------------|----------:|-----------:|--------:|------------:|
| poisson_regression (2 p)   |       9.9 |    **5.2** |   1.91× | 6.0 ms (logistic, 2 p) |
| eight_schools_ncp (10 p)   |      16.0 |    **5.5** |   2.90× |     5.5 ms |
| linear_regression (3 p)†   |     248.8 |   **44.3** |   5.62× |          — |

† `linear_regression` posterior is sharp (σ ≈ 0.1 on N=30 synthetic data); NUTS adapts to a small step and takes many leapfrog steps. Not representative.

The AOT path matches or beats the MoonBit reference (`stan-wasm/tests/results/benchmark_nutsrs_aot.json`, also Node.js V8). Same algorithm in both — `nuts-rs` Rust crate inside wasm calling into a fully-unrolled AOT model wasm. The Rust unification's only architectural difference is **shared linear memory**, which removes the inter-wasm `memcpy` MoonBit's bridge had to do per logp call.

### Native Rust release

| case                 | AST µs | replay µs | AOT (wasmi) µs | sample ms |
|----------------------|-------:|----------:|---------------:|----------:|
| linear_regression    |   ~6.0 |      1.55 |           1.71 |      12.5 |
| poisson_regression   |   ~4.5 |      0.37 |           0.58 |       8.1 |
| eight_schools_ncp    |   ~6.4 |      0.84 |           0.91 |      13.5 |

`AOT (wasmi)` is meaningful only as an internal sanity check — `wasmi` is an interpreter, not a JIT, and is ~10× slower than V8 on the AOT path.

## Reproducing

```bash
# Native
cargo run --release -p stan-cli -- bench all

# Node.js (requires wasm-pack on PATH)
./scripts/build-wasm.sh
cd ts && node --experimental-strip-types tests/bench.ts
```

## Architecture details

`sample` (replay) lives entirely inside `stan_wasm_api_bg.wasm`. nuts-rs gets its log_prob_grad from `Compiled::log_prob_grad`, which dispatches on `Op` per tape node. V8 JIT-compiles this dispatch loop, but cannot inline across the dispatch.

`sampleViaAot` lives across two wasm modules:

```
┌──────────────────────────────┐    shared linear memory    ┌──────────────────────────┐
│  stan_wasm_api_bg.wasm       │ ◄────────────────────────► │  AOT model wasm          │
│                              │                             │                          │
│  - parser, codegen           │                             │  - imports "stan.memory" │
│  - nuts-rs sampler driver    │                             │  - exports log_prob_grad │
│  - shims math fns to JS      │                             │    (fully unrolled)      │
│                              │                             │                          │
│  imports aot_logp via JS     │ ──── one JS shim call ────► │  V8 JITs the unrolled    │
│  bridge (set_aot_exports)    │                             │  forward + backward      │
└──────────────────────────────┘                             └──────────────────────────┘
```

The JS bridge between the two modules is a 5-line snippet (`crates/stan-wasm-api/js/aot_bridge.js`) that V8 inlines aggressively after warmup.

## Caveats

- Each row is one run of 2,000 NUTS draws. Variance ±5–10 % between runs. `linear_regression` is the most variable (sharp posterior).
- `sample ms` includes warmup adaptation. For longer chains the per-draw rate stabilizes lower.
- Math import shims (lgamma, digamma, phi) are JS-side polynomial approximations; precision matches `stan_autodiff` Rust functions.
- AOT path requires the host to provide math imports; missing imports throw at instantiate time.
