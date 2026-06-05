# MoonBit vs Rust — building the same Stan inference engine twice

> English translation not yet written. Please see the Japanese original at [docs/ja/MOONBIT_VS_RUST.md](../ja/MOONBIT_VS_RUST.md).
>
> A condensed English version is planned as an outreach article (see `docs/ja/DELIVERY.md`). Contributions welcome.

## TL;DR (English summary)

After implementing a browser-side Stan inference engine in MoonBit (using its wasm-gc backend), we re-implemented the same project in Rust and benchmarked both side by side under V8.

**Headline finding**: the wasm-gc advantage that motivated the original MoonBit choice does **not** apply to the Stan sampling hot path. The hot path is the AOT-compiled per-model wasm — emitted as plain wasm32 in both implementations — and `nuts-rs` is the same Rust crate in both. Performance is therefore at parity (±10% noise), not a structural win for either language.

The Rust port still made sense for non-performance reasons:
- Single Rust ecosystem (no JS bridge between two wasm modules)
- Shared linear memory between the host wasm and the per-model AOT wasm (eliminates inter-module memcpy that MoonBit needed)
- Direct dependency on `nuts-rs` via `Cargo.toml`
- 30% smaller total bundle (365 KB single wasm vs 523 KB across two MoonBit wasms)

See the Japanese original for the full architecture comparison, evaluation tables, and `wasmparser` verification that neither artifact uses wasm-gc opcodes.
