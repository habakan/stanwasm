//! Value type for the Stan AST evaluator.
//!
//! Four kinds:
//! - `Num` — plain constant (data, loop counter)
//! - `Tape(u32)` — autodiff tape index (involves parameters)
//! - `Vec` — vector, array or matrix (matrix = vec of rows)
//! - `Row` — a row vector, which only `'` produces and only `*` reads

use crate::error::EvalError;
use stanwasm_autodiff::Tape;

#[derive(Debug, Clone)]
pub enum Val {
    Num(f64),
    Tape(u32),
    Vec(Vec<Val>),
    Row(Vec<Val>),
}

impl Val {
    /// Coerce to a plain f64. A `Vec` means a container where a scalar was required,
    /// which hand-written Stan can reach — an `EvalError`, not a wasm-trapping panic.
    pub fn to_f64(&self, tape: &Tape) -> Result<f64, EvalError> {
        match self {
            Val::Num(v) => Ok(*v),
            Val::Tape(i) => Ok(tape.value(*i)),
            Val::Vec(_) | Val::Row(_) => Err(EvalError::NotAScalar),
        }
    }

    /// Lift to a tape node (creating a leaf if needed).
    pub fn to_tape(&self, tape: &mut Tape) -> Result<u32, EvalError> {
        match self {
            Val::Num(v) => Ok(tape.new_var(*v)),
            Val::Tape(i) => Ok(*i),
            Val::Vec(_) | Val::Row(_) => Err(EvalError::NotAScalar),
        }
    }

    pub fn to_i32(&self, tape: &Tape) -> Result<i32, EvalError> {
        Ok(self.to_f64(tape)? as i32)
    }

    /// Elements of a container, whichever orientation it carries. Most of the
    /// runtime works on the elements and never asks which way the value points.
    pub fn elems(&self) -> Option<&[Val]> {
        match self {
            Val::Vec(xs) | Val::Row(xs) => Some(xs),
            Val::Num(_) | Val::Tape(_) => None,
        }
    }

    /// Rebuild a container of this one's orientation from new elements, so a
    /// row stays a row through arithmetic.
    pub fn like(&self, xs: Vec<Val>) -> Val {
        match self {
            Val::Row(_) => Val::Row(xs),
            _ => Val::Vec(xs),
        }
    }

    /// Shape of this value, for operand checking in `eval`.
    pub fn shape(&self) -> Shape {
        match self {
            Val::Num(_) | Val::Tape(_) => Shape::Scalar,
            Val::Row(xs) => Shape::RowVector(xs.len()),
            Val::Vec(xs) => {
                if !xs.iter().any(|x| x.elems().is_some()) {
                    return Shape::Vector(xs.len());
                }
                // Rows must be equal-length containers to be a matrix. `cols: None`
                // marks a ragged value, which never compares equal to anything.
                let mut cols = None;
                for x in xs {
                    match (x.elems(), cols) {
                        (Some(r), None) => cols = Some(r.len()),
                        (Some(r), Some(c)) if r.len() == c => {}
                        _ => return Shape::Matrix(xs.len(), None),
                    }
                }
                Shape::Matrix(xs.len(), cols)
            }
        }
    }
}

/// Coarse shape classification — enough to reject the operand combinations
/// this runtime would otherwise answer wrongly (see `eval::check_binop_shapes`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    Scalar,
    Vector(usize),
    RowVector(usize),
    /// A vec-of-rows: `(rows, cols)`. `cols` is `None` when the rows are not
    /// all containers of one common length.
    Matrix(usize, Option<usize>),
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Shape::Scalar => write!(f, "scalar"),
            Shape::Vector(n) => write!(f, "vector[{n}]"),
            Shape::RowVector(n) => write!(f, "row_vector[{n}]"),
            Shape::Matrix(r, Some(c)) => write!(f, "{r}x{c} matrix"),
            Shape::Matrix(r, None) => write!(f, "ragged {r}-row container"),
        }
    }
}
