//! Stan lexer + parser. Hand-written recursive descent.
//! Phase 1 will replace this stub with the full implementation.

#![forbid(unsafe_code)]

use stan_ast::Program;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("not yet implemented")]
    NotImplemented,
}

pub fn parse(_src: &str) -> Result<Program, ParseError> {
    Err(ParseError::NotImplemented)
}
