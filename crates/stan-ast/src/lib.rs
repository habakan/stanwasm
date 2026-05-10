//! Stan AST types. Shared between parser, runtime, and codegen.
//!
//! Phase 1 will populate this with the full AST mirroring `compiler/stan/ast.mbt`.

#![forbid(unsafe_code)]

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub data: Vec<DataDecl>,
    pub parameters: Vec<ParamDecl>,
    pub transformed_parameters: Vec<Stmt>,
    pub model: Vec<Stmt>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct DataDecl {
    pub name: String,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct ParamDecl {
    pub name: String,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Placeholder,
}
