//! Stan AST types shared across the workspace.

#![forbid(unsafe_code)]

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum Constraint {
    None,
    Lower(Expr),
    Upper(Expr),
    LowerUpper(Expr, Expr),
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum StanType {
    Real(Constraint),
    Int(Constraint),
    /// vector[size] with optional element constraint
    Vector(Expr, Constraint),
    /// matrix[rows, cols]
    Matrix(Expr, Expr),
    Simplex(Expr),
    Ordered(Expr),
    /// array[size] of element_type
    Array(Expr, Box<StanType>),
    CholeskyFactorCorr(Expr),
    CholeskyFactorCov(Expr),
    CovMatrix(Expr),
    CorrMatrix(Expr),
    PositiveOrdered(Expr),
    UnitVector(Expr),
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Num(f64),
    /// Integer literal (`3`, not `3.0`). Kept distinct from `Num` because Stan
    /// is statically typed and `/` on two ints is *integer* division: `3 / 2`
    /// is `1`, while `3.0 / 2` is `1.5`.
    IntNum(i64),
    Var(String),
    /// op, left, right
    BinOp(String, Box<Expr>, Box<Expr>),
    /// op, operand
    UnOp(String, Box<Expr>),
    /// arr[idx]
    Index(Box<Expr>, Box<Expr>),
    /// func(args)
    Call(String, Vec<Expr>),
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct VarDecl {
    pub typ: StanType,
    pub name: String,
    pub init: Option<Expr>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    /// for (var in lo:hi) body
    For(String, Expr, Expr, Vec<Stmt>),
    While(Expr, Vec<Stmt>),
    /// if (cond) then else
    If(Expr, Vec<Stmt>, Vec<Stmt>),
    /// lhs ~ dist(args)
    Sample(Expr, String, Vec<Expr>),
    /// target += expr
    TargetIncr(Expr),
    /// lhs = rhs
    Assign(Expr, Expr),
    /// lhs += rhs
    IncrAssign(Expr, Expr),
    /// type name = init?;
    LocalDecl(StanType, String, Option<Expr>),
    /// return expr;
    Return(Expr),
    Break,
    Continue,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct FuncDef {
    pub params: Vec<(StanType, String)>,
    pub body: Vec<Stmt>,
    pub ret_expr: Expr,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StanProgram {
    pub functions: Vec<(String, FuncDef)>,
    pub data: Vec<VarDecl>,
    pub parameters: Vec<VarDecl>,
    pub transformed_params: Vec<VarDecl>,
    pub transformed_stmts: Vec<Stmt>,
    pub model: Vec<Stmt>,
    pub gen_quantities: Vec<Stmt>,
}
