# Migration plan: stan-wasm (MoonBit) → stan-wasm-rs (Rust)

## Decisions locked in

| Item | Choice |
|---|---|
| Repository name | `stan-wasm-rs` |
| JS bindings | `wasm-bindgen` |
| TS NUTS sampler | dropped — nuts-rs only |
| AST evaluation interpreter | kept in `stan-runtime` as native-only oracle (not exported to wasm) |
| Single wasm output | yes — nuts-rs embedded as Cargo dep |
| WAT generation | replaced by `wasm-encoder` (direct binary, no wabt JS dep) |
| Stan parser | hand-written recursive descent |
| `compiler/moonbit-nuts/` | dropped (functionally redundant with nuts-rs) |

## What ships in wasm

- Stan parser, AST, data binding
- Constrained transforms + Jacobian
- Parameter name introspection
- Autodiff tape (one-shot trace at compile time)
- AOT codegen (writes model wasm bytes via `wasm-encoder`)
- nuts-rs sampler (embedded)

## What is native-only (not in wasm)

- AST evaluation (`stan-runtime::eval`) — used as golden oracle in `cargo test`
- `stan-cli` binary

## Phases

| # | Scope | Validation |
|---|---|---|
| 0 | Cargo workspace, crate stubs, hello-world wasm | wasm-bindgen TS demo loads & runs |
| 1 | Lexer + parser + AST + serde | AST JSON matches MoonBit version on existing test models |
| 2 | Autodiff tape | Per-op gradient diff vs MoonBit < 1e-12 |
| 3 | Distributions + constraints (`bytecode.mbt`) | logp/grad diff vs MoonBit < 1e-12 |
| 4 | AOT codegen via wasm-encoder | Per-model AOT logp matches interpreter logp |
| 5 | nuts-rs integration, single wasm | Sampling matches existing nuts-rs path |
| 6 | TS API rewiring (`api/stan.ts` → `ts/`) | `examples/get_started` runs on new wasm |
| 7 | Examples, benchmarks, ARCHITECTURE.md update | ESS/sec table refreshed |

## Reference oracle protocol

While the Rust port is in progress, MoonBit's `wasm_api.wasm` is the authoritative implementation. Each phase's tests compare Rust output to MoonBit output; divergence blocks the phase.

After Phase 7 cutover, MoonBit is retired.
