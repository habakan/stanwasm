// Public TS facade for stan-wasm-rs. Phase 0: re-export wasm-bindgen output as-is.
// Subsequent phases will add `StanWasm` class, parallel sampling, etc.

export { default as init, greet, version } from "./pkg/stan_wasm_api";
