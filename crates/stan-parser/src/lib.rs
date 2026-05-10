//! Stan lexer + parser. Hand-written recursive descent.
//!
//! Public entry: `parse(src) -> StanProgram`.
//! Re-exports `Token` and `tokenize` for diagnostics and golden-value tests.

#![forbid(unsafe_code)]

pub mod lexer;
pub mod parser;
pub mod token;

pub use lexer::tokenize;
pub use parser::{parse, ParseError, Parser};
pub use token::Token;
