# stan-wasm-rs

Stan inference engine for WebAssembly, written in Rust. Successor to [stan-wasm](../stan-wasm) (MoonBit-based); unifies the compiler and NUTS sampler in a single Rust workspace and a single `.wasm` artifact.

## Status

Pre-alpha. Phase 0 scaffolding only — no functional code yet. See `docs/MIGRATION.md` for the porting plan.

## Architecture

```
Stan source + data
       │
       ▼  (one-time)
stan-parser ─► AST ─► stan-runtime (trace forward pass on autodiff tape)
                                    │
                                    ▼
                           stan-codegen (wasm-encoder)
                                    │
                                    ▼
                       model.wasm (model-specific log_prob_grad)
                                    │
                                    ▼  (sampling loop)
                           nuts-rs (embedded in same wasm)
                                    │
                                    ▼
                                 samples
```

A single `stan_wasm_api.wasm` is shipped to the browser (replaces the previous `wasm_api.wasm` + `nuts_rs.wasm` pair). Native builds expose the AST evaluation interpreter for golden-value testing only.

## Workspace layout

| Crate | Target | Role |
|---|---|---|
| `stan-ast` | lib | AST types, shared definitions |
| `stan-parser` | lib | Lexer, parser, error reporting |
| `stan-autodiff` | lib | Reverse-mode tape (flat array) |
| `stan-runtime` | lib | Distributions, constraints, AST evaluator (reference oracle) |
| `stan-codegen` | lib | AOT compilation: tape → wasm bytes |
| `stan-wasm-api` | cdylib (wasm32) | wasm-bindgen public API; embeds nuts-rs |
| `stan-cli` | bin (native) | Local development & golden-value tests |

## Build

```bash
# Native (tests, CLI)
cargo build --release
cargo test

# wasm
./scripts/build-wasm.sh
```

## License

Apache-2.0
