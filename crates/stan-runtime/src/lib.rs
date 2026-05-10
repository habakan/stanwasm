//! Stan runtime: AST evaluator + distributions + constraint transforms.
//!
//! This crate is the **golden oracle**. It walks the parsed AST and traces
//! tape ops to compute log_prob and gradients. Used by:
//! - `cargo test` for verifying AOT codegen output
//! - `stan-codegen` to perform the one-shot forward trace that produces
//!   the wasm-emit input
//!
//! Intentionally NOT re-exported by `stan-wasm-api`: production wasm
//! ships only the AOT path.

#![forbid(unsafe_code)]

mod compiled;
mod constraints;
mod distributions;
mod env;
mod eval;
mod matrix;
mod model;
mod ops;
mod value;

#[cfg(feature = "json")]
mod data_json;

pub use compiled::Compiled;
pub use env::Env;
pub use model::{Model, ModelError};
pub use value::Val;

#[cfg(feature = "json")]
pub use data_json::data_from_json;
