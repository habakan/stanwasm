//! Value type for the Stan AST evaluator.
//!
//! Three kinds:
//! - `Num` — plain constant (data, loop counter)
//! - `Tape(u32)` — autodiff tape index (involves parameters)
//! - `Vec` — vector or matrix (matrix = vec of row vecs)

use crate::error::EvalError;
use stan_autodiff::Tape;

#[derive(Debug, Clone)]
pub enum Val {
    Num(f64),
    Tape(u32),
    Vec(Vec<Val>),
}

impl Val {
    /// Coerce to a plain f64 (Tape values read their primal off the tape).
    ///
    /// A `Vec` here means the model used a container where a scalar was
    /// required (`if (x == y)` on two vectors, a matrix product fed to a
    /// scalar lpdf, ...). That is user-reachable from hand-written Stan, so it
    /// is an `EvalError` rather than the panic it used to be — a panic
    /// compiles to a wasm trap, which kills the whole module instance and
    /// forces a page reload in the browser.
    pub fn to_f64(&self, tape: &Tape) -> Result<f64, EvalError> {
        match self {
            Val::Num(v) => Ok(*v),
            Val::Tape(i) => Ok(tape.value(*i)),
            Val::Vec(_) => Err(EvalError::NotAScalar),
        }
    }

    /// Lift to a tape node (creating a leaf if needed).
    pub fn to_tape(&self, tape: &mut Tape) -> Result<u32, EvalError> {
        match self {
            Val::Num(v) => Ok(tape.new_var(*v)),
            Val::Tape(i) => Ok(*i),
            Val::Vec(_) => Err(EvalError::NotAScalar),
        }
    }

    pub fn to_i32(&self, tape: &Tape) -> Result<i32, EvalError> {
        Ok(self.to_f64(tape)? as i32)
    }

    /// Shape of this value, for operand checking in `eval`.
    pub fn shape(&self) -> Shape {
        match self {
            Val::Num(_) | Val::Tape(_) => Shape::Scalar,
            Val::Vec(xs) => {
                if xs.iter().any(|x| matches!(x, Val::Vec(_))) {
                    Shape::Matrix(xs.len())
                } else {
                    Shape::Vector(xs.len())
                }
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
    /// A vec-of-rows; the payload is the row count.
    Matrix(usize),
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Shape::Scalar => write!(f, "scalar"),
            Shape::Vector(n) => write!(f, "vector[{n}]"),
            Shape::Matrix(r) => write!(f, "matrix with {r} rows"),
        }
    }
}
