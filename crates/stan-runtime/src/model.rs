//! High-level Model: load Stan AST + data, evaluate log_prob and gradients.

use std::cell::RefCell;
use std::rc::Rc;

use crate::constraints::{constrain, param_dims};
use crate::env::Env;
use crate::eval::{eval_expr, eval_stmt};
use crate::ops::v_add;
use crate::value::Val;
use rand::rngs::ChaCha8Rng;
use stan_ast::{StanProgram, StanType, Stmt};
use stan_autodiff::Tape;
use thiserror::Error;

/// Flatten a `Val` (scalar / vector / matrix-as-vec-of-rows) into `out`,
/// reading tape-node primals through `tape`. Matches the flattening order
/// used by `param_names`/`gen_quantity_names`.
fn flatten_val(tape: &Tape, v: &Val, out: &mut Vec<f64>) {
    match v {
        Val::Vec(xs) => {
            for x in xs {
                flatten_val(tape, x, out);
            }
        }
        other => out.push(other.to_f64(tape)),
    }
}

/// Push flattened names for one declared variable, matching Stan's naming
/// convention: scalar → `name`; vector(N) → `name[1]`..`name[N]`; matrix →
/// `name[i,j]`. Shared by `param_names` and `gen_quantity_names`.
fn push_names_for(out: &mut Vec<String>, name: &str, typ: &StanType, env: &Env) {
    match typ {
        StanType::Matrix(r_e, c_e) => {
            let rows = eval_int(r_e, env);
            let cols = eval_int(c_e, env);
            for i in 1..=rows {
                for j in 1..=cols {
                    out.push(format!("{name}[{i},{j}]"));
                }
            }
        }
        other => {
            let k = param_dims(other, env);
            if k <= 1 {
                out.push(name.to_string());
            } else {
                for i in 1..=k {
                    out.push(format!("{name}[{i}]"));
                }
            }
        }
    }
}

fn eval_int(expr: &stan_ast::Expr, env: &Env) -> usize {
    let mut t = Tape::new();
    let v = eval_expr(&mut t, expr, env);
    match v {
        Val::Num(x) => x as usize,
        _ => 0,
    }
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("parse: {0}")]
    Parse(#[from] stan_parser::ParseError),
}

pub struct Model {
    pub prog: StanProgram,
    pub data_env: Env,
    pub n_params: usize,
}

impl Model {
    pub fn new(prog: StanProgram, data_env: Env) -> Self {
        let n_params: usize = prog
            .parameters
            .iter()
            .map(|d| param_dims(&d.typ, &data_env))
            .sum();
        Self {
            prog,
            data_env,
            n_params,
        }
    }

    pub fn parse_and_load(stan_src: &str, data_env: Env) -> Result<Self, ModelError> {
        let prog = stan_parser::parse(stan_src)?;
        Ok(Self::new(prog, data_env))
    }

    pub fn n_params(&self) -> usize {
        self.n_params
    }

    /// Constrained parameter names matching `constrained_params` order.
    /// Scalar → `name`; vector(N) → `name[1]`..`name[N]`; matrix → `name[i,j]`.
    /// Includes both `parameters` and `transformed parameters`.
    pub fn param_names(&self) -> Vec<String> {
        let mut out = Vec::new();
        let env = &self.data_env;
        for d in &self.prog.parameters {
            push_names_for(&mut out, &d.name, &d.typ, env);
        }
        for d in &self.prog.transformed_params {
            push_names_for(&mut out, &d.name, &d.typ, env);
        }
        out
    }

    /// Names of the top-level `generated quantities` declarations, in
    /// declaration order and flattened the same way as `param_names`.
    pub fn gen_quantity_names(&self) -> Vec<String> {
        let mut out = Vec::new();
        for stmt in &self.prog.gen_quantities {
            if let Stmt::LocalDecl(typ, name, _) = stmt {
                push_names_for(&mut out, name, typ, &self.data_env);
            }
        }
        out
    }

    /// Build an `Env` with data bound and `parameters`/`transformed parameters`
    /// evaluated from `unconstrained` (constraint transforms applied). Shared
    /// by `constrained_draw` and `generated_quantities`, which — unlike
    /// `trace_forward` — don't need the constraint Jacobian (no log_prob is
    /// being computed).
    fn build_env(&self, tape: &mut Tape, leaves: &[u32]) -> Env {
        let mut env = self.data_env.clone();
        let mut leaf_idx = 0usize;
        for decl in &self.prog.parameters {
            let k = param_dims(&decl.typ, &self.data_env);
            let raw: Vec<Val> = (0..k).map(|i| Val::Tape(leaves[leaf_idx + i])).collect();
            leaf_idx += k;
            let (constrained, _log_jac) = constrain(tape, &decl.typ, &raw, &self.data_env);
            env.set(&decl.name, constrained);
        }
        for decl in &self.prog.transformed_params {
            let init = match &decl.init {
                Some(e) => eval_expr(tape, e, &env),
                None => Val::Num(0.0),
            };
            env.set(&decl.name, init);
        }
        for stmt in &self.prog.transformed_stmts {
            eval_stmt(tape, stmt, &mut env).into_val();
        }
        env
    }

    /// Constrained values of `parameters` + `transformed parameters` for one
    /// unconstrained draw, flattened in the same order as `param_names()`.
    /// No gradient is computed — the tape here is a disposable value scratchpad.
    pub fn constrained_draw(&self, unconstrained: &[f64]) -> Vec<f64> {
        let mut tape = Tape::new();
        let leaves: Vec<u32> = unconstrained.iter().map(|p| tape.new_var(*p)).collect();
        let env = self.build_env(&mut tape, &leaves);
        let mut out = Vec::new();
        for decl in &self.prog.parameters {
            let v = env
                .get(&decl.name)
                .unwrap_or_else(|| panic!("internal: missing parameter {}", decl.name));
            flatten_val(&tape, v, &mut out);
        }
        for decl in &self.prog.transformed_params {
            let v = env
                .get(&decl.name)
                .unwrap_or_else(|| panic!("internal: missing transformed parameter {}", decl.name));
            flatten_val(&tape, v, &mut out);
        }
        out
    }

    /// Evaluate the `generated quantities` block for one unconstrained draw.
    /// `rng` is shared (via `Rc<RefCell<_>>`) across draws so the RNG stream
    /// advances across a batch instead of resetting every call. Returns the
    /// top-level declared values flattened in `gen_quantity_names()` order.
    pub fn generated_quantities(
        &self,
        unconstrained: &[f64],
        rng: Rc<RefCell<ChaCha8Rng>>,
    ) -> Vec<f64> {
        let mut tape = Tape::new();
        let leaves: Vec<u32> = unconstrained.iter().map(|p| tape.new_var(*p)).collect();
        let mut env = self.build_env(&mut tape, &leaves);
        env.set_rng(rng);
        for stmt in &self.prog.gen_quantities {
            eval_stmt(&mut tape, stmt, &mut env).into_val();
        }
        let mut out = Vec::new();
        for stmt in &self.prog.gen_quantities {
            if let Stmt::LocalDecl(_typ, name, _) = stmt {
                let v = env
                    .get(name)
                    .unwrap_or_else(|| panic!("internal: missing generated quantity {name}"));
                flatten_val(&tape, v, &mut out);
            }
        }
        out
    }

    /// Compute log_prob and gradient at the given unconstrained parameters.
    pub fn log_prob_grad(&self, params: &[f64]) -> (f64, Vec<f64>) {
        let mut tape = Tape::new();
        let leaves: Vec<u32> = params.iter().map(|p| tape.new_var(*p)).collect();
        let root = self.trace_forward(&mut tape, &leaves);
        tape.backward(root);
        let lp = tape.value(root);
        let grads: Vec<f64> = leaves.iter().map(|i| tape.grad_at(*i)).collect();
        (lp, grads)
    }

    /// One-shot forward trace on the supplied tape; returns the root tape index
    /// so codegen can walk the recorded ops. Caller controls tape lifetime.
    pub fn trace_forward(&self, tape: &mut Tape, leaves: &[u32]) -> u32 {
        let mut env = self.data_env.clone();

        // Apply constraint transforms; accumulate Jacobian into lp.
        let mut leaf_idx = 0usize;
        let mut lp: Val = Val::Num(0.0);
        for decl in &self.prog.parameters {
            let k = param_dims(&decl.typ, &self.data_env);
            let raw: Vec<Val> = (0..k).map(|i| Val::Tape(leaves[leaf_idx + i])).collect();
            leaf_idx += k;
            let (constrained, log_jac) = constrain(tape, &decl.typ, &raw, &self.data_env);
            env.set(&decl.name, constrained);
            lp = v_add(tape, &lp, &log_jac);
        }

        // transformed_params are declared then the transformed_stmts run.
        for decl in &self.prog.transformed_params {
            let init = match &decl.init {
                Some(e) => crate::eval::eval_expr(tape, e, &env),
                None => Val::Num(0.0),
            };
            env.set(&decl.name, init);
        }
        for stmt in &self.prog.transformed_stmts {
            let r = eval_stmt(tape, stmt, &mut env).into_val();
            lp = v_add(tape, &lp, &r);
        }

        // model block: each statement may add to log_prob.
        for stmt in &self.prog.model {
            let r = eval_stmt(tape, stmt, &mut env).into_val();
            lp = v_add(tape, &lp, &r);
        }

        lp.to_tape(tape)
    }
}
