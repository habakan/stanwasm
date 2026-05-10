//! Value type for the Stan AST evaluator.
//!
//! Three kinds:
//! - `Num` — plain constant (data, loop counter)
//! - `Tape(u32)` — autodiff tape index (involves parameters)
//! - `Vec` — vector or matrix (matrix = vec of row vecs)

use stan_autodiff::Tape;

#[derive(Debug, Clone)]
pub enum Val {
    Num(f64),
    Tape(u32),
    Vec(Vec<Val>),
}

impl Val {
    /// Coerce to a plain f64 (Tape values read their primal off the tape).
    pub fn to_f64(&self, tape: &Tape) -> f64 {
        match self {
            Val::Num(v) => *v,
            Val::Tape(i) => tape.value(*i),
            Val::Vec(_) => panic!("Val::to_f64 called on Vec"),
        }
    }

    /// Lift to a tape node (creating a leaf if needed).
    pub fn to_tape(&self, tape: &mut Tape) -> u32 {
        match self {
            Val::Num(v) => tape.new_var(*v),
            Val::Tape(i) => *i,
            Val::Vec(_) => panic!("Val::to_tape called on Vec"),
        }
    }

    pub fn to_i32(&self, tape: &Tape) -> i32 {
        self.to_f64(tape) as i32
    }
}
