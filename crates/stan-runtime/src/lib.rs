//! Stan runtime: distributions, constraint transforms, AST evaluator.
//!
//! The AST evaluator (`eval`) is exposed for native-only golden-value testing;
//! it is intentionally NOT re-exported by `stan-wasm-api` and so does not
//! ship in the wasm bundle. Production wasm uses AOT-compiled model wasm.

#![forbid(unsafe_code)]

pub mod distributions;
pub mod constraints;
pub mod eval;
