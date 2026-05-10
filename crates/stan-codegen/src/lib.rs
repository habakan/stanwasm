//! AOT compilation: trace the model on the autodiff tape, emit a
//! model-specific wasm module that computes log_prob and gradients in one call.
//!
//! Replaces `compiler/stan/codegen.mbt` (which emitted WAT text). This
//! crate emits wasm binary directly via `wasm-encoder`, removing the
//! browser-side `wabt` dependency.

#![forbid(unsafe_code)]

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CodegenError {
    #[error("not yet implemented")]
    NotImplemented,
}

/// Returns the bytes of a model-specific wasm module. Phase 4.
pub fn compile(_program: &stan_ast::StanProgram) -> Result<Vec<u8>, CodegenError> {
    Err(CodegenError::NotImplemented)
}
