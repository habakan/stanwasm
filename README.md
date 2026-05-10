# stan-wasm-rs

Stan inference engine for WebAssembly, written in Rust. Successor to [stan-wasm](../stan-wasm) (MoonBit-based); unifies the compiler, runtime, and NUTS sampler in a single Rust workspace and a single `.wasm` artifact.

## Status

Phases 0–5 complete. Sampling validated end-to-end: linear regression posterior recovers the true slope to ±0.3 in 1000 draws.

Distributions covered: `normal`, `std_normal`, `exponential`, `half_normal`, `cauchy`, `student_t`, `lognormal`, `gamma`, `beta`, `bernoulli`, `bernoulli_logit`, `poisson`, `neg_binomial_2`. Constraint transforms: `lower`, `upper`, `lower_upper` on scalars and vectors. Multivariate distributions and `simplex` / `ordered` / `cholesky_factor_*` constraints are not yet ported.

See `docs/MIGRATION.md` for the per-phase plan and `docs/BENCHMARKS.md` for performance numbers.

## Architecture

```
Stan source + JSON data
       │
       ▼  (one-time)
stan-parser ─► AST ─► stan-runtime (trace forward pass on autodiff tape)
                                      │
                                      ▼
                          ┌──── tape replay (sampling)
                          │              │
                          │              ▼
                          │         nuts-rs (embedded in same wasm)
                          │              │
                          │              ▼
                          │           samples
                          │
                          └──── stan-codegen (wasm-encoder)  ──►  AOT model wasm bytes
                                                                  (Web Worker handoff)
```

A single `stan_wasm_api_bg.wasm` is shipped to the browser (replaces the previous `wasm_api.wasm` + `nuts_rs.wasm` pair). Native builds expose the AST evaluation interpreter for golden-value testing only.

## Workspace layout

| Crate | Target | Role |
|---|---|---|
| `stan-ast` | lib | AST types, shared definitions |
| `stan-parser` | lib | Lexer, parser, error reporting |
| `stan-autodiff` | lib | Reverse-mode tape (flat array) |
| `stan-runtime` | lib | Distributions, constraints, AST evaluator (reference oracle) |
| `stan-codegen` | lib | AOT compilation: tape → wasm bytes (via `wasm-encoder`) |
| `stan-wasm-api` | cdylib (wasm32) | wasm-bindgen public API; embeds nuts-rs |
| `stan-cli` | bin (native) | Local development, golden-value tests, benchmarks |

## Quick start (browser / Node.js)

```bash
# Build the wasm bundle
./scripts/build-wasm.sh

# Run smoke test
cd ts
node --experimental-strip-types tests/smoke.ts
```

```ts
import init, { StanModel } from "stan-wasm-rs";

await init();

const stanCode = `...`;
const data = { N: 30, x: [...], y: [...] };
const model = new StanModel(stanCode, JSON.stringify(data));

console.log(`n_params = ${model.n_params}`);
console.log(`names    = ${model.paramNames().join(", ")}`);

// Single-call gradient
const lpAndGrad = model.logProbGrad(new Float64Array([0, 1, 0]));
//   [logp, dα, dβ, dlog_σ]

// Full NUTS sampling
const samples = model.sample(
  new Float64Array([0, 0, 0]),
  /*nWarmup*/ 1000,
  /*nDraws*/  1000,
  /*seed*/    42n,
);
// samples is Float64Array, row-major shape (nWarmup + nDraws) × n_params
```

## Native development

```bash
cargo build --release
cargo test                    # ~30 tests across all crates
cargo run --release -p stan-cli -- bench all
```

## License

Apache-2.0
