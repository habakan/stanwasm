//! Trace a Stan model on the autodiff tape, then hand the tape to the emitter.
//!
//! The emitter is [`tapewasm_codegen`], which reads a tape and nothing else.
//! What is left here is the half that knows what a model is: running the
//! evaluator forward to record one, and reporting an evaluation failure apart
//! from a failure to emit.
//!
//! What the emitter exports is re-exported unchanged, so a caller reaching for
//! [`compile_tape`] or [`Reroll`] need not depend on both crates.

#![forbid(unsafe_code)]

use stanwasm_runtime::Model;
use tapewasm_autodiff::Tape;
use thiserror::Error;

pub use tapewasm_codegen::{compile_tape, tape_text, Compiled, Reroll, ABI_VERSION};

#[derive(Debug, Error)]
pub enum CodegenError {
    #[error(transparent)]
    Emit(#[from] tapewasm_codegen::CodegenError),
    #[error(transparent)]
    Eval(#[from] stanwasm_runtime::EvalError),
    #[error("internal: {0}")]
    Internal(String),
}

/// Trace `model` on a fresh tape at `dummy_params` (0.1 throughout; eight_schools
/// needs non-zero seeds), then emit a model-specific wasm module.
pub fn compile(model: &Model, dummy_params: &[f64]) -> Result<Compiled, CodegenError> {
    compile_with(model, dummy_params, Reroll::default())
}

/// [`compile`], choosing when to re-roll vectorised statements.
pub fn compile_with(
    model: &Model,
    dummy_params: &[f64],
    reroll: Reroll,
) -> Result<Compiled, CodegenError> {
    if dummy_params.len() != model.n_params() {
        return Err(CodegenError::Internal(format!(
            "dummy_params len {} != model n_params {}",
            dummy_params.len(),
            model.n_params()
        )));
    }
    let mut tape = Tape::new();
    let leaves: Vec<u32> = dummy_params.iter().map(|p| tape.new_var(*p)).collect();
    let root = model.trace_forward(&mut tape, &leaves, true)?;
    Ok(compile_tape(&tape, dummy_params.len(), root, reroll)?)
}
