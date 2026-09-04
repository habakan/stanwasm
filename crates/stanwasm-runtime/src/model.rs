//! High-level Model: load Stan AST + data, evaluate log_prob and gradients.

use std::cell::RefCell;
use std::rc::Rc;

use crate::constraints::{constrain, param_dims};
use crate::env::Env;
use crate::error::EvalError;
use crate::eval::{eval_expr, eval_stmt, stan_type_is_int};
use crate::ops::v_add;
use crate::value::Val;
use rand::rngs::ChaCha8Rng;
use stanwasm_ast::{Constraint, StanProgram, StanType, Stmt};
use stanwasm_autodiff::Tape;
use thiserror::Error;

/// Flatten a `Val` into `out`, reading primals through `tape`. Matches the
/// order used by `param_names`/`gen_quantity_names`.
fn flatten_val(tape: &Tape, v: &Val, out: &mut Vec<f64>) -> Result<(), EvalError> {
    match v {
        Val::Vec(xs) | Val::Row(xs) => {
            for x in xs {
                flatten_val(tape, x, out)?;
            }
        }
        other => out.push(other.to_f64(tape)?),
    }
    Ok(())
}

/// Push flattened names for one variable: scalar → `name`; vector(N) →
/// `name[1]`..`name[N]`; matrix → `name[i,j]`. An array contributes its own
/// index in front of its element's, so `array[2] simplex[3]` is `p[1,1]`
/// through `p[2,3]` — the shape `param_dims` cannot describe, since it counts
/// a simplex's *unconstrained* entries and there is one fewer of those.
fn push_names_for(out: &mut Vec<String>, name: &str, typ: &StanType, env: &Env) {
    push_named(out, name, &mut Vec::new(), typ, env);
}

fn emit_name(out: &mut Vec<String>, name: &str, idx: &[usize]) {
    if idx.is_empty() {
        out.push(name.to_string());
        return;
    }
    let mut s = String::from(name);
    s.push('[');
    for (n, i) in idx.iter().enumerate() {
        if n > 0 {
            s.push(',');
        }
        s.push_str(&i.to_string());
    }
    s.push(']');
    out.push(s);
}

fn push_named(out: &mut Vec<String>, name: &str, idx: &mut Vec<usize>, typ: &StanType, env: &Env) {
    let grid = |out: &mut Vec<String>, idx: &mut Vec<usize>, rows: usize, cols: usize| {
        for i in 1..=rows {
            for j in 1..=cols {
                idx.push(i);
                idx.push(j);
                emit_name(out, name, idx);
                idx.pop();
                idx.pop();
            }
        }
    };
    match typ {
        StanType::Array(size_e, elem) => {
            for i in 1..=eval_int(size_e, env) {
                idx.push(i);
                push_named(out, name, idx, elem, env);
                idx.pop();
            }
        }
        StanType::Matrix(r_e, c_e, _) => grid(out, idx, eval_int(r_e, env), eval_int(c_e, env)),
        // These constrain to a K×K matrix, while `param_dims` counts the smaller
        // unconstrained vector — naming from that would leave the labels short and
        // misaligned against `constrained_draw`.
        StanType::CholeskyFactorCorr(k_e)
        | StanType::CholeskyFactorCov(k_e)
        | StanType::CovMatrix(k_e)
        | StanType::CorrMatrix(k_e) => {
            let k = eval_int(k_e, env);
            grid(out, idx, k, k);
        }
        // Likewise: K-1 unconstrained, K constrained.
        StanType::Simplex(k_e) => {
            for i in 1..=eval_int(k_e, env) {
                idx.push(i);
                emit_name(out, name, idx);
                idx.pop();
            }
        }
        other => {
            let k = param_dims(other, env);
            if k <= 1 {
                emit_name(out, name, idx);
            } else {
                for i in 1..=k {
                    idx.push(i);
                    emit_name(out, name, idx);
                    idx.pop();
                }
            }
        }
    }
}

/// Display names only; falls back to 0 on an evaluation error, since a wrong
/// label is cosmetic rather than a wrong posterior.
fn eval_int(expr: &stanwasm_ast::Expr, env: &Env) -> usize {
    let mut t = Tape::new();
    match eval_expr(&mut t, expr, env) {
        Ok(Val::Num(x)) => x as usize,
        _ => 0,
    }
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("parse: {0}")]
    Parse(#[from] stanwasm_parser::ParseError),
    #[error("data block: {0}")]
    Data(#[from] DataMismatch),
    #[error("transformed data: {0}")]
    TransformedData(#[from] EvalError),
}

/// A `data { ... }` declaration the supplied JSON doesn't satisfy. Each of these
/// used to be accepted silently and give a wrong answer, so they are checked at load.
#[derive(Debug, Error)]
pub enum DataMismatch {
    #[error("`{name}` is declared in the data block but missing from the data")]
    Missing { name: String },
    #[error("`{name}`: expected {expected}, got {got}")]
    Shape {
        name: String,
        expected: String,
        got: String,
    },
    #[error("`{name}`{at} is declared `int` but the value {value} is not a whole number")]
    NotInteger {
        name: String,
        at: String,
        value: f64,
    },
    #[error("`{name}`{at}: value {value} violates the declared bound {bound}")]
    Constraint {
        name: String,
        at: String,
        value: f64,
        bound: String,
    },
    #[error("`{name}`: size expression does not evaluate to a positive integer from earlier data")]
    BadSize { name: String },
}

/// Concrete expected shape of a data declaration, with all size expressions
/// resolved against the data bound so far.
#[derive(Debug, Clone, PartialEq)]
enum Shaped {
    Scalar,
    Vector(usize),
    Matrix(usize, usize),
    Array(usize, Box<Shaped>),
}

impl std::fmt::Display for Shaped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Shaped::Scalar => write!(f, "a scalar"),
            Shaped::Vector(n) => write!(f, "a length-{n} vector"),
            Shaped::Matrix(r, c) => write!(f, "a {r}x{c} matrix"),
            Shaped::Array(n, e) => write!(f, "an array of {n} x ({e})"),
        }
    }
}

fn describe(v: &Val) -> String {
    match v.elems() {
        Some(xs) => match xs.first().and_then(Val::elems) {
            Some(row) => format!("a {}x{} nested array", xs.len(), row.len()),
            None => format!("a length-{} array", xs.len()),
        },
        None => "a scalar".to_string(),
    }
}

/// Resolve a declared type's sizes against the data bound so far. Sizes may
/// reference earlier declarations (`int N; vector[N] x;`), hence declaration order.
fn shape_of(name: &str, typ: &StanType, env: &Env) -> Result<Shaped, DataMismatch> {
    let size = |e: &stanwasm_ast::Expr| -> Result<usize, DataMismatch> {
        let mut t = Tape::new();
        match eval_expr(&mut t, e, env) {
            Ok(Val::Num(v)) if v >= 0.0 && v.fract() == 0.0 => Ok(v as usize),
            _ => Err(DataMismatch::BadSize {
                name: name.to_string(),
            }),
        }
    };
    Ok(match typ {
        StanType::Real(_) | StanType::Int(_) => Shaped::Scalar,
        StanType::Vector(n, _)
        | StanType::Simplex(n)
        | StanType::Ordered(n)
        | StanType::PositiveOrdered(n)
        | StanType::UnitVector(n)
        | StanType::RowVector(n, _) => Shaped::Vector(size(n)?),
        StanType::Matrix(r, c, _) => Shaped::Matrix(size(r)?, size(c)?),
        StanType::CholeskyFactorCorr(k)
        | StanType::CholeskyFactorCov(k)
        | StanType::CovMatrix(k)
        | StanType::CorrMatrix(k) => {
            let kk = size(k)?;
            Shaped::Matrix(kk, kk)
        }
        StanType::Array(n, elem) => Shaped::Array(size(n)?, Box::new(shape_of(name, elem, env)?)),
    })
}

/// Element constraint carried by a declared type, if any.
fn elem_constraint(typ: &StanType) -> &Constraint {
    match typ {
        StanType::Real(c) | StanType::Int(c) | StanType::Vector(_, c) => c,
        StanType::RowVector(_, c) => c,
        StanType::Matrix(_, _, c) => c,
        StanType::Array(_, elem) => elem_constraint(elem),
        _ => &Constraint::None,
    }
}

/// Walk a supplied value against the resolved shape, checking lengths and —
/// at the leaves — integrality and declared bounds.
fn check_value(
    name: &str,
    at: &str,
    val: &Val,
    shape: &Shaped,
    is_int: bool,
    bounds: (Option<f64>, Option<f64>),
) -> Result<(), DataMismatch> {
    let mismatch = || DataMismatch::Shape {
        name: name.to_string(),
        expected: shape.to_string(),
        got: describe(val),
    };
    match shape {
        Shaped::Scalar => {
            let Val::Num(x) = val else {
                return Err(mismatch());
            };
            if is_int && x.fract() != 0.0 {
                return Err(DataMismatch::NotInteger {
                    name: name.to_string(),
                    at: at.to_string(),
                    value: *x,
                });
            }
            if let Some(lo) = bounds.0 {
                if *x < lo {
                    return Err(DataMismatch::Constraint {
                        name: name.to_string(),
                        at: at.to_string(),
                        value: *x,
                        bound: format!("lower={lo}"),
                    });
                }
            }
            if let Some(hi) = bounds.1 {
                if *x > hi {
                    return Err(DataMismatch::Constraint {
                        name: name.to_string(),
                        at: at.to_string(),
                        value: *x,
                        bound: format!("upper={hi}"),
                    });
                }
            }
            Ok(())
        }
        Shaped::Vector(n) => {
            let Some(xs) = val.elems() else {
                return Err(mismatch());
            };
            if xs.len() != *n {
                return Err(mismatch());
            }
            for (i, x) in xs.iter().enumerate() {
                let at = format!("{at}[{}]", i + 1);
                check_value(name, &at, x, &Shaped::Scalar, is_int, bounds)?;
            }
            Ok(())
        }
        Shaped::Matrix(r, c) => {
            let Some(rows) = val.elems() else {
                return Err(mismatch());
            };
            if rows.len() != *r {
                return Err(mismatch());
            }
            for (i, row) in rows.iter().enumerate() {
                let at = format!("{at}[{}]", i + 1);
                check_value(name, &at, row, &Shaped::Vector(*c), is_int, bounds)?;
            }
            Ok(())
        }
        Shaped::Array(n, elem) => {
            let Some(xs) = val.elems() else {
                return Err(mismatch());
            };
            if xs.len() != *n {
                return Err(mismatch());
            }
            for (i, x) in xs.iter().enumerate() {
                let at = format!("{at}[{}]", i + 1);
                check_value(name, &at, x, elem, is_int, bounds)?;
            }
            Ok(())
        }
    }
}

/// Check every `data` declaration against the supplied values, and record
/// which bindings are int-typed (Stan's `/` is integer division on two ints).
fn validate_data(prog: &StanProgram, env: &mut Env) -> Result<(), DataMismatch> {
    for decl in &prog.data {
        if env.get(&decl.name).is_none() {
            return Err(DataMismatch::Missing {
                name: decl.name.clone(),
            });
        }
        let shape = shape_of(&decl.name, &decl.typ, env)?;
        let is_int = stan_type_is_int(&decl.typ);
        let bounds = {
            let mut t = Tape::new();
            let mut resolve = |e: &stanwasm_ast::Expr| match eval_expr(&mut t, e, env) {
                Ok(Val::Num(v)) => Some(v),
                _ => None,
            };
            match elem_constraint(&decl.typ) {
                Constraint::None => (None, None),
                Constraint::Lower(lo) => (resolve(lo), None),
                Constraint::Upper(hi) => (None, resolve(hi)),
                Constraint::LowerUpper(lo, hi) => (resolve(lo), resolve(hi)),
            }
        };
        // Borrowed and retagged in place. A `data` block holding an MNIST-sized
        // matrix is gigabytes as `Val`, and a copy to check it and another to
        // orient it were two more.
        let val = env
            .get(&decl.name)
            .expect("checked above that the binding is there");
        check_value(&decl.name, "", val, &shape, is_int, bounds)?;
        if is_int {
            env.mark_int(&decl.name);
        } else if let Some(slot) = env.get_mut(&decl.name) {
            orient_rows(&decl.typ, slot);
        }
    }
    Ok(())
}

/// A declared matrix's rows are row vectors; an `array[N] vector[K]`'s are
/// columns. The JSON is the same nested array either way, so the declaration is
/// the only thing that can say which.
fn orient_rows(typ: &StanType, val: &mut Val) {
    match typ {
        StanType::Matrix(..)
        | StanType::CholeskyFactorCorr(_)
        | StanType::CholeskyFactorCov(_)
        | StanType::CovMatrix(_)
        | StanType::CorrMatrix(_) => {
            if let Val::Vec(rows) = val {
                for r in rows.iter_mut() {
                    if let Val::Vec(cells) = r {
                        *r = Val::Row(std::mem::take(cells));
                    }
                }
            }
        }
        StanType::Array(_, elem) => {
            if let Val::Vec(xs) = val {
                for x in xs.iter_mut() {
                    orient_rows(elem, x);
                }
            }
        }
        _ => {}
    }
}

pub struct Model {
    pub prog: StanProgram,
    pub data_env: Env,
    pub n_params: usize,
}

impl Model {
    pub fn new(prog: StanProgram, data_env: Env) -> Result<Self, EvalError> {
        // Held on the data env so every later scope inherits it by clone.
        let mut data_env = data_env;
        if !prog.functions.is_empty() {
            data_env.set_funcs(std::rc::Rc::new(prog.functions.clone()));
        }
        // `transformed data` sees only data, so it is evaluated once here and its
        // results join the data env: parameter sizes may depend on them.
        let mut tape = Tape::new();
        for stmt in &prog.transformed_data {
            eval_stmt(&mut tape, stmt, &mut data_env)?.into_val();
        }
        data_env.freeze(&tape);
        let n_params: usize = prog
            .parameters
            .iter()
            .map(|d| param_dims(&d.typ, &data_env))
            .sum();
        Ok(Self {
            prog,
            data_env,
            n_params,
        })
    }

    pub fn parse_and_load(stan_src: &str, mut data_env: Env) -> Result<Self, ModelError> {
        let prog = stanwasm_parser::parse(stan_src)?;
        validate_data(&prog, &mut data_env)?;
        Ok(Self::new(prog, data_env)?)
    }

    pub fn n_params(&self) -> usize {
        self.n_params
    }

    /// Constrained parameter names in `constrained_params` order, covering both
    /// `parameters` and `transformed parameters`.
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

    /// One name per *unconstrained* slot, which is what a gradient is indexed
    /// by. A constrained declaration has fewer of these than it has entries —
    /// a `simplex[K]` is `K - 1` — so `param_names` cannot be used for it.
    pub fn unconstrained_param_names(&self) -> Vec<String> {
        let mut out = Vec::new();
        for d in &self.prog.parameters {
            let k = param_dims(&d.typ, &self.data_env);
            if k == 1 {
                out.push(d.name.clone());
            } else {
                out.extend((1..=k).map(|i| format!("{}[{i}]", d.name)));
            }
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

    /// `Env` with data bound and parameters evaluated from `unconstrained`. Shared
    /// by `constrained_draw`/`generated_quantities`, which need no Jacobian.
    fn build_env(&self, tape: &mut Tape, leaves: &[u32]) -> Result<Env, EvalError> {
        let mut env = self.data_env.clone();
        let mut leaf_idx = 0usize;
        for decl in &self.prog.parameters {
            let k = param_dims(&decl.typ, &self.data_env);
            let raw: Vec<Val> = (0..k).map(|i| Val::Tape(leaves[leaf_idx + i])).collect();
            leaf_idx += k;
            let (constrained, _log_jac) =
                constrain(tape, &decl.name, &decl.typ, &raw, &self.data_env)?;
            env.set(&decl.name, constrained);
        }
        for decl in &self.prog.transformed_params {
            let init = match &decl.init {
                Some(e) => eval_expr(tape, e, &env)?,
                // An uninitialised `vector[W] k;` has to arrive shaped, or the
                // element assignments that follow have nothing to write into.
                None => crate::eval::default_for_type(tape, &decl.typ, &env)?,
            };
            env.set(&decl.name, init);
        }
        for stmt in &self.prog.transformed_stmts {
            eval_stmt(tape, stmt, &mut env)?.into_val();
        }
        Ok(env)
    }

    /// Constrained `parameters` + `transformed parameters` for one draw, in
    /// `param_names()` order. No gradient — the tape is a scratchpad.
    pub fn constrained_draw(&self, unconstrained: &[f64]) -> Result<Vec<f64>, EvalError> {
        let mut tape = Tape::new();
        let leaves: Vec<u32> = unconstrained.iter().map(|p| tape.new_var(*p)).collect();
        let env = self.build_env(&mut tape, &leaves)?;
        let mut out = Vec::new();
        for decl in &self.prog.parameters {
            let v = env
                .get(&decl.name)
                .unwrap_or_else(|| panic!("internal: missing parameter {}", decl.name));
            flatten_val(&tape, v, &mut out)?;
        }
        for decl in &self.prog.transformed_params {
            let v = env
                .get(&decl.name)
                .unwrap_or_else(|| panic!("internal: missing transformed parameter {}", decl.name));
            flatten_val(&tape, v, &mut out)?;
        }
        Ok(out)
    }

    /// `generated quantities` for one draw, flattened in `gen_quantity_names()`
    /// order. `rng` is shared across draws so the stream advances over a batch.
    pub fn generated_quantities(
        &self,
        unconstrained: &[f64],
        rng: Rc<RefCell<ChaCha8Rng>>,
    ) -> Result<Vec<f64>, EvalError> {
        let mut tape = Tape::new();
        let leaves: Vec<u32> = unconstrained.iter().map(|p| tape.new_var(*p)).collect();
        let mut env = self.build_env(&mut tape, &leaves)?;
        env.set_rng(rng);
        for stmt in &self.prog.gen_quantities {
            eval_stmt(&mut tape, stmt, &mut env)?.into_val();
        }
        let mut out = Vec::new();
        for stmt in &self.prog.gen_quantities {
            if let Stmt::LocalDecl(typ, name, _) = stmt {
                let v = env
                    .get(name)
                    .unwrap_or_else(|| panic!("internal: missing generated quantity {name}"));
                let before = out.len();
                flatten_val(&tape, v, &mut out)?;

                // `gen_quantity_names` sizes the output from the declaration, so a value
                // of a different length would land as a length-mismatch panic downstream.
                let mut names = Vec::new();
                push_names_for(&mut names, name, typ, &self.data_env);
                if out.len() - before != names.len() {
                    return Err(EvalError::GenQuantityShape {
                        name: name.clone(),
                        expected: names.len(),
                        got: out.len() - before,
                    });
                }
            }
        }
        Ok(out)
    }

    /// log_prob and gradient at the given unconstrained parameters. Traces fresh
    /// every call, so a parameter-dependent `if`/`while` evaluates correctly here.
    pub fn log_prob_grad(&self, params: &[f64]) -> Result<(f64, Vec<f64>), EvalError> {
        let mut tape = Tape::new();
        let leaves: Vec<u32> = params.iter().map(|p| tape.new_var(*p)).collect();
        let root = self.trace_forward(&mut tape, &leaves, false)?;
        tape.backward(root);
        let lp = tape.value(root);
        let grads: Vec<f64> = leaves.iter().map(|i| tape.grad_at(*i)).collect();
        Ok((lp, grads))
    }

    /// One-shot forward trace; returns the root tape index. Set `strict` when this
    /// trace will be replayed, to reject parameter-dependent branches, not freeze them.
    pub fn trace_forward(
        &self,
        tape: &mut Tape,
        leaves: &[u32],
        strict: bool,
    ) -> Result<u32, EvalError> {
        let mut env = self.data_env.clone();
        env.set_strict_no_param_branch(strict);

        // Apply constraint transforms; accumulate Jacobian into lp.
        let mut leaf_idx = 0usize;
        let mut lp: Val = Val::Num(0.0);
        for decl in &self.prog.parameters {
            let k = param_dims(&decl.typ, &self.data_env);
            let raw: Vec<Val> = (0..k).map(|i| Val::Tape(leaves[leaf_idx + i])).collect();
            leaf_idx += k;
            let (constrained, log_jac) =
                constrain(tape, &decl.name, &decl.typ, &raw, &self.data_env)?;
            env.set(&decl.name, constrained);
            lp = v_add(tape, &lp, &log_jac);
        }

        // transformed_params are declared then the transformed_stmts run.
        for decl in &self.prog.transformed_params {
            let init = match &decl.init {
                Some(e) => crate::eval::eval_expr(tape, e, &env)?,
                // An uninitialised `vector[W] k;` has to arrive shaped, or the
                // element assignments that follow have nothing to write into.
                None => crate::eval::default_for_type(tape, &decl.typ, &env)?,
            };
            env.set(&decl.name, init);
        }
        for stmt in &self.prog.transformed_stmts {
            let r = eval_stmt(tape, stmt, &mut env)?.into_val();
            lp = v_add(tape, &lp, &r);
        }

        // model block: each statement may add to log_prob.
        for stmt in &self.prog.model {
            let r = eval_stmt(tape, stmt, &mut env)?.into_val();
            lp = v_add(tape, &lp, &r);
        }

        lp.to_tape(tape)
    }
}
