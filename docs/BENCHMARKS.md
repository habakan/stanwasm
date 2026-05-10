# Benchmarks

Two harnesses available:

| Harness | What it measures | Run |
|---------|------------------|-----|
| native Rust release | Tape replay & nuts-rs without any wasm overhead | `cargo run --release -p stan-cli -- bench all` |
| Node.js V8 + wasm32 | Same code path users see in the browser | `cd ts && npm run build:wasm && npm run bench` |

The native CLI also times the AOT model wasm via `wasmi`; the Node bench times it via V8 (vastly faster).

## Apple Silicon, 2026-05-10

### Native Rust (release)

| case                 | AST µs | replay µs | AOT (wasmi) µs | sample ms |
|----------------------|-------:|----------:|---------------:|----------:|
| linear_regression    |   ~6.0 |      1.55 |           1.71 |      12.5 |
| poisson_regression   |   ~4.5 |      0.37 |           0.58 |       8.1 |
| eight_schools_ncp    |   ~6.4 |      0.84 |           0.91 |      13.5 |

### Node.js V8 (wasm32 + wasm-pack)

| case                 | lpg µs | sample ms |
|----------------------|-------:|----------:|
| linear_regression    |   2.01 |     247.0 |
| poisson_regression   |   0.84 |      10.1 |
| eight_schools_ncp    |   1.12 |      15.9 |

`linear_regression` is anomalously slow because the synthetic data (σ ≈ 0.1) creates a sharp posterior; NUTS adapts to a small step size and takes many leapfrog steps per draw. Not a representative model — listed for completeness.

## Comparison with MoonBit reference

`stan-wasm/tests/results/benchmark_nutsrs_aot.json` (Node.js V8, MoonBit WAT AOT path, originally measured during stan-wasm development):

| model                         | MoonBit | stan-wasm-rs (V8) |
|-------------------------------|--------:|-------------------:|
| logistic_regression (2 p)     |   6.0 ms |  10.1 ms (poisson — similar size) |
| eight_schools (10 p)          |   5.5 ms |  15.9 ms |
| gaussian_mixture              |  37.4 ms | not yet supported (multivariate) |

The 2-parameter case is ~1.7× slower than MoonBit; the 10-parameter case ~2.9× slower. The per-call lpg cost is comparable (Rust ~1 µs in V8), so the gap comes from one of:

- different leapfrog step counts per draw (mass matrix / step size adaptation diverges with the same `nuts-rs` algorithm because input precision differs);
- reference numbers measured on different hardware / older `nuts-rs` (the stan-wasm JSON has no machine metadata);
- the AOT path in MoonBit is fully unrolled wasm that V8 JITs aggressively, vs Rust replay which dispatches on `Op` per node.

Closing that gap is tracked as Phase 7 work — wire `StanModel::sample` to evaluate `log_prob_grad` through the AOT-compiled model wasm (not tape replay) so V8 can JIT the unrolled forward+backward pass directly. The codegen output already exists; just not invoked from `sample`.

## Reproducing

```bash
# Native
cargo run --release -p stan-cli -- bench all

# Node.js (requires wasm-pack on PATH)
cd ts
../scripts/build-wasm.sh   # or: wasm-pack build ../crates/stan-wasm-api --target web --out-dir pkg --release
node --experimental-strip-types tests/bench.ts
```

## Caveats

- Native AOT µs measures `wasmi` (interpreter); browser path is V8 (JIT) and meaningfully faster.
- Each row is one run of 10,000 lpg iterations / 2,000 NUTS draws. Variance ±5 % between runs.
- `sample ms` includes warmup adaptation. For longer chains the per-draw rate stabilizes.
- Browser delivery uses `wasm-pack` build (`web` target) → `Float64Array` round-trip via wasm-bindgen.
