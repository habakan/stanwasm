//! High-level Model: load Stan AST + data, evaluate log_prob and gradients.

use crate::constraints::{constrain, param_dims};
use crate::env::Env;
use crate::eval::{eval_expr, eval_stmt};
use crate::ops::v_add;
use crate::value::Val;
use stan_ast::StanProgram;
use stan_autodiff::Tape;
use thiserror::Error;

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
        use stan_ast::StanType;
        let mut out = Vec::new();
        let env = &self.data_env;
        let push_for = |out: &mut Vec<String>, name: &str, typ: &StanType| match typ {
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
                let k = crate::constraints::param_dims(other, env);
                if k <= 1 {
                    out.push(name.to_string());
                } else {
                    for i in 1..=k {
                        out.push(format!("{name}[{i}]"));
                    }
                }
            }
        };
        for d in &self.prog.parameters {
            push_for(&mut out, &d.name, &d.typ);
        }
        for d in &self.prog.transformed_params {
            push_for(&mut out, &d.name, &d.typ);
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
            let r = eval_stmt(tape, stmt, &mut env);
            lp = v_add(tape, &lp, &r);
        }

        // model block: each statement may add to log_prob.
        for stmt in &self.prog.model {
            let r = eval_stmt(tape, stmt, &mut env);
            lp = v_add(tape, &lp, &r);
        }

        lp.to_tape(tape)
    }
}
