//! Stan runtime: AST evaluator + distributions + constraint transforms.
//!
//! This crate is the **golden oracle**. It walks the parsed AST and traces
//! tape ops to compute log_prob and gradients. Used by:
//! - `cargo test` for verifying AOT codegen output
//! - `stanwasm-codegen` to perform the one-shot forward trace that produces
//!   the wasm-emit input
//!
//! `stanwasm` does not re-export this crate's types, but it does embed
//! the evaluator: `StanModel::sample` traces the AST and replays the recorded
//! tape inside the shipped wasm, and `generated quantities` is evaluated here
//! natively per draw. The AOT path is the optional fast lane on top
//! (`sampleViaAot`), not a replacement.

#![forbid(unsafe_code)]

mod compiled;
mod constraints;
mod distributions;
mod env;
mod error;
mod eval;
mod matrix;
mod model;
mod ops;
mod rng;
mod value;

#[cfg(feature = "json")]
mod data_json;

pub use compiled::Compiled;
pub use env::Env;
pub use error::EvalError;
pub use model::{Model, ModelError};
pub use value::Val;

#[cfg(feature = "json")]
pub use data_json::data_from_json;
