//! AST evaluator. Walks Stan AST, pushes tape ops, returns Val.

use crate::distributions::{eval_dist, eval_sample_vec};
use crate::env::Env;
use crate::error::EvalError;
use crate::ops::{
    v_abs, v_add, v_div, v_exp, v_inv_logit, v_lgamma, v_log, v_logit, v_mul, v_neg, v_phi, v_pow,
    v_sqrt, v_sub, v_tanh,
};
use crate::value::{Shape, Val};
use stan_ast::{Expr, StanType, Stmt};
use stan_autodiff::Tape;

type Result<T> = std::result::Result<T, EvalError>;

pub fn eval_plain(t: &mut Tape, expr: &Expr, env: &Env) -> Result<Val> {
    eval_expr(t, expr, env)
}

pub fn eval_expr(t: &mut Tape, expr: &Expr, env: &Env) -> Result<Val> {
    match expr {
        Expr::Num(v) => Ok(Val::Num(*v)),
        Expr::IntNum(v) => Ok(Val::Num(*v as f64)),
        Expr::Var(n) => env
            .get(n)
            .cloned()
            .ok_or_else(|| EvalError::UndefinedVariable(n.clone())),
        Expr::BinOp(op, l, r) => {
            let lv = eval_expr(t, l, env)?;
            let rv = eval_expr(t, r, env)?;
            if matches!(
                op.as_str(),
                "==" | "!=" | "<" | ">" | "<=" | ">=" | "&&" | "||"
            ) {
                // Comparison/logical results always collapse to a plain
                // Val::Num (see bool_val below) — checking the *result* for
                // Val::Tape can never catch a parameter-dependent condition,
                // since the taint is lost right here. Check the operands
                // instead, before that collapse happens.
                check_no_param_branch(env, &lv)?;
                check_no_param_branch(env, &rv)?;
            }
            check_binop_shapes(op, &lv, &rv)?;
            Ok(match op.as_str() {
                "+" => v_add(t, &lv, &rv),
                "-" => v_sub(t, &lv, &rv),
                "*" => v_mul(t, &lv, &rv),
                // Stan is statically typed and `int / int` truncates toward
                // zero (`N / 2` with `N = 3` is 1, not 1.5). Int-ness is a
                // property of the *declarations*, not of the runtime value,
                // so it's decided from the expression tree.
                "/" if is_int_expr(l, env) && is_int_expr(r, env) => {
                    let denom = rv.to_f64(t)?;
                    if denom == 0.0 {
                        return Err(EvalError::IntDivisionByZero);
                    }
                    Val::Num((lv.to_f64(t)? / denom).trunc())
                }
                "/" => v_div(t, &lv, &rv),
                "^" => v_pow(t, &lv, &rv),
                "==" => bool_val(lv.to_f64(t)? == rv.to_f64(t)?),
                "!=" => bool_val(lv.to_f64(t)? != rv.to_f64(t)?),
                "<" => bool_val(lv.to_f64(t)? < rv.to_f64(t)?),
                ">" => bool_val(lv.to_f64(t)? > rv.to_f64(t)?),
                "<=" => bool_val(lv.to_f64(t)? <= rv.to_f64(t)?),
                ">=" => bool_val(lv.to_f64(t)? >= rv.to_f64(t)?),
                "&&" => bool_val(lv.to_f64(t)? != 0.0 && rv.to_f64(t)? != 0.0),
                "||" => bool_val(lv.to_f64(t)? != 0.0 || rv.to_f64(t)? != 0.0),
                // Unreachable in practice: the parser only ever produces the
                // operator strings matched above.
                _ => Val::Num(0.0),
            })
        }
        Expr::UnOp(op, e) => {
            let v = eval_expr(t, e, env)?;
            Ok(match op.as_str() {
                "-" => v_neg(t, &v),
                "!" => bool_val(v.to_f64(t)? == 0.0),
                _ => v,
            })
        }
        Expr::Index(arr_e, idx_e) => {
            let one_based = eval_expr(t, idx_e, env)?.to_i32(t)?;
            let idx = one_based - 1;
            let arr = eval_expr(t, arr_e, env)?;
            match arr {
                Val::Vec(xs) => {
                    if idx >= 0 {
                        if let Some(v) = xs.get(idx as usize) {
                            return Ok(v.clone());
                        }
                    }
                    Err(EvalError::IndexOutOfBounds {
                        index: one_based,
                        len: xs.len(),
                    })
                }
                other => Ok(other),
            }
        }
        Expr::Call(name, args) => eval_call(t, name, args, env),
    }
}

fn bool_val(b: bool) -> Val {
    Val::Num(if b { 1.0 } else { 0.0 })
}

/// Whether an expression has an integral Stan type. Only `/` cares (integer
/// division), but the answer has to propagate through the arithmetic that
/// feeds it: `(N + 1) / 2` is still integer division.
fn is_int_expr(e: &Expr, env: &Env) -> bool {
    match e {
        Expr::IntNum(_) => true,
        Expr::Num(_) => false,
        Expr::Var(n) => env.is_int(n),
        // `y[i]` is an int exactly when `y` is an `array[...] int`.
        Expr::Index(base, _) => is_int_expr(base, env),
        Expr::UnOp(op, a) => op == "-" && is_int_expr(a, env),
        Expr::BinOp(op, l, r) => match op.as_str() {
            // `^` yields a real in Stan even for int operands.
            "+" | "-" | "*" | "/" => is_int_expr(l, env) && is_int_expr(r, env),
            // Comparisons and logical ops are int-valued (0/1) in Stan.
            "==" | "!=" | "<" | ">" | "<=" | ">=" | "&&" | "||" => true,
            _ => false,
        },
        Expr::Call(..) => false,
    }
}

/// Whether a declared type binds an integral value (`int`, `array[...] int`).
pub fn stan_type_is_int(typ: &StanType) -> bool {
    match typ {
        StanType::Int(_) => true,
        StanType::Array(_, elem) => stan_type_is_int(elem),
        _ => false,
    }
}

/// Reject operand shapes this runtime would otherwise answer *wrongly*.
///
/// `ops::v_*` broadcast element-wise and `zip` silently truncates, so without
/// this check `vector[3] + vector[2]` quietly returns a length-2 vector and
/// `X * beta` (matrix × vector) quietly returns the matrix with each row
/// scaled by one element of `beta` — neither is what Stan computes.
fn check_binop_shapes(op: &str, lhs: &Val, rhs: &Val) -> Result<()> {
    use Shape::*;
    let (ls, rs) = (lhs.shape(), rhs.shape());
    let ok = match (ls, rs) {
        (Scalar, Scalar) => true,
        // scalar ⊙ container broadcasts element-wise, in both directions.
        (Scalar, _) | (_, Scalar) => true,
        (Vector(a), Vector(b)) => a == b,
        (Matrix(a), Matrix(b)) => a == b,
        // Matrix × vector / vector × matrix is linear algebra, not
        // element-wise broadcast — not implemented (see `EvalError::NotAScalar`).
        (Matrix(_), Vector(_)) | (Vector(_), Matrix(_)) => false,
    };
    if ok {
        return Ok(());
    }
    Err(EvalError::ShapeMismatch {
        op: op.to_string(),
        lhs: ls.to_string(),
        rhs: rs.to_string(),
    })
}

fn eval_call(t: &mut Tape, name: &str, args: &[Expr], env: &Env) -> Result<Val> {
    let evaled: Vec<Val> = args
        .iter()
        .map(|a| eval_expr(t, a, env))
        .collect::<Result<_>>()?;
    Ok(match (name, evaled.as_slice()) {
        ("log", [a]) => v_log(t, a),
        ("exp", [a]) => v_exp(t, a),
        ("sqrt", [a]) => v_sqrt(t, a),
        ("abs", [a]) | ("fabs", [a]) => v_abs(t, a),
        ("lgamma", [a]) => v_lgamma(t, a),
        ("inv_logit", [a]) | ("logistic", [a]) => v_inv_logit(t, a),
        ("logit", [a]) => v_logit(t, a),
        ("tanh", [a]) => v_tanh(t, a),
        ("Phi", [a]) | ("std_normal_cdf", [a]) => v_phi(t, a),
        ("pow", [a, b]) => v_pow(t, a, b),
        ("square", [a]) => v_mul(t, a, a),
        ("sum", [Val::Vec(xs)]) => {
            let mut acc = Val::Num(0.0);
            for x in xs {
                acc = v_add(t, &acc, x);
            }
            acc
        }
        ("mean", [Val::Vec(xs)]) => {
            let n = xs.len() as f64;
            let mut acc = Val::Num(0.0);
            for x in xs {
                acc = v_add(t, &acc, x);
            }
            v_div(t, &acc, &Val::Num(n))
        }
        ("segment", [Val::Vec(xs), start_v, len_v]) => {
            let start_1b = start_v.to_i32(t)?;
            let len = len_v.to_i32(t)?;
            // `skip`/`take` would silently return a short vector for an
            // out-of-range slice; a range index is a bounds error in Stan.
            if start_1b < 1 || len < 0 || (start_1b - 1 + len) as usize > xs.len() {
                return Err(EvalError::IndexOutOfBounds {
                    index: start_1b + len - 1,
                    len: xs.len(),
                });
            }
            let start = (start_1b - 1) as usize;
            Val::Vec(xs[start..start + len as usize].to_vec())
        }
        // distribution _lpdf / _lpmf forms used as expressions
        (n, args) if n.ends_with("_lpdf") || n.ends_with("_lpmf") => {
            let base = &n[..n.len() - 5];
            if args.is_empty() {
                Val::Num(0.0)
            } else {
                let x = &args[0];
                let rest: Vec<Val> = args[1..].to_vec();
                match x {
                    Val::Vec(xs) => eval_sample_vec(t, base, xs, &rest)?,
                    _ => eval_dist(t, base, x, &rest)?,
                }
            }
        }
        // RNG forms, valid only in generated quantities (env carries an rng).
        (n, args) if n.ends_with("_rng") => {
            let base = &n[..n.len() - 4];
            crate::rng::dispatch(t, base, args, env)?
        }
        _ => return Err(EvalError::UnknownFunction(name.to_string())),
    })
}

/// Result of evaluating a statement: either a plain log-prob contribution, or
/// a loop-control signal (each still carrying the log-prob accumulated up to
/// the point of exit, e.g. from statements executed before a `break`).
pub enum Flow {
    Val(Val),
    Break(Val),
    Continue(Val),
}

impl Flow {
    pub fn into_val(self) -> Val {
        match self {
            Flow::Val(v) | Flow::Break(v) | Flow::Continue(v) => v,
        }
    }
}

/// Evaluate a statement list as a scoped block: locals declared inside are
/// visible only for the duration of the block, and a `break`/`continue`
/// short-circuits the remaining statements while propagating the signal (and
/// the log-prob accumulated so far) to the caller.
fn eval_block(t: &mut Tape, stmts: &[Stmt], env: &mut Env) -> Result<Flow> {
    let saved = env.len();
    let mut acc = Val::Num(0.0);
    let mut result = None;
    for s in stmts {
        match eval_stmt(t, s, env)? {
            Flow::Val(v) => acc = v_add(t, &acc, &v),
            Flow::Break(v) => {
                acc = v_add(t, &acc, &v);
                result = Some(Flow::Break(acc.clone()));
                break;
            }
            Flow::Continue(v) => {
                acc = v_add(t, &acc, &v);
                result = Some(Flow::Continue(acc.clone()));
                break;
            }
        }
    }
    env.truncate(saved);
    Ok(result.unwrap_or(Flow::Val(acc)))
}

/// Evaluate a statement; returns the increment to log_prob (zero for non-target
/// statements) wrapped in a `Flow` that also carries `break`/`continue` signals.
pub fn eval_stmt(t: &mut Tape, stmt: &Stmt, env: &mut Env) -> Result<Flow> {
    match stmt {
        Stmt::Sample(lhs, dist, args) => {
            let x = eval_expr(t, lhs, env)?;
            let evaled_args: Vec<Val> = args
                .iter()
                .map(|a| eval_expr(t, a, env))
                .collect::<Result<_>>()?;
            let v = match &x {
                Val::Vec(xs) => eval_sample_vec(t, dist, xs, &evaled_args)?,
                _ => eval_dist(t, dist, &x, &evaled_args)?,
            };
            Ok(Flow::Val(v))
        }
        Stmt::TargetIncr(e) => Ok(Flow::Val(eval_expr(t, e, env)?)),
        Stmt::IncrAssign(lhs, rhs) => {
            // For target += rhs (lhs is `target`), already handled above.
            // Generic form: lhs += rhs.
            let Expr::Var(name) = lhs else {
                return Err(EvalError::UnsupportedAssignmentTarget);
            };
            if name == "target" {
                return Ok(Flow::Val(eval_expr(t, rhs, env)?));
            }
            let cur = env.get(name).cloned().unwrap_or(Val::Num(0.0));
            let r = eval_expr(t, rhs, env)?;
            let new_val = v_add(t, &cur, &r);
            env.set(name, new_val);
            Ok(Flow::Val(Val::Num(0.0)))
        }
        Stmt::Assign(lhs, rhs) => {
            let r = eval_expr(t, rhs, env)?;
            let Expr::Var(name) = lhs else {
                return Err(EvalError::UnsupportedAssignmentTarget);
            };
            env.set(name, r);
            Ok(Flow::Val(Val::Num(0.0)))
        }
        Stmt::LocalDecl(typ, name, init) => {
            let v = match init {
                Some(e) => eval_expr(t, e, env)?,
                None => default_for_type(t, typ, env)?,
            };
            if stan_type_is_int(typ) {
                env.set_int_typed(name, v);
            } else {
                env.set(name, v);
            }
            Ok(Flow::Val(Val::Num(0.0)))
        }
        Stmt::For(var, lo_e, hi_e, body) => {
            let lo = eval_expr(t, lo_e, env)?.to_i32(t)?;
            let hi = eval_expr(t, hi_e, env)?.to_i32(t)?;
            let saved_len = env.len();
            let mut acc = Val::Num(0.0);
            for i in lo..=hi {
                // Loop counters are `int` in Stan, so `i / 2` truncates.
                env.set_int_typed(var, Val::Num(i as f64));
                match eval_block(t, body, env)? {
                    Flow::Val(v) | Flow::Continue(v) => acc = v_add(t, &acc, &v),
                    Flow::Break(v) => {
                        acc = v_add(t, &acc, &v);
                        break;
                    }
                }
            }
            env.truncate(saved_len);
            Ok(Flow::Val(acc))
        }
        Stmt::While(cond, body) => {
            const MAX_ITERS: u64 = 1_000_000;
            let mut acc = Val::Num(0.0);
            let mut iters: u64 = 0;
            loop {
                let c = eval_expr(t, cond, env)?;
                check_no_param_branch(env, &c)?;
                if c.to_f64(t)? == 0.0 {
                    break;
                }
                iters += 1;
                if iters > MAX_ITERS {
                    return Err(EvalError::WhileLoopOverflow(MAX_ITERS));
                }
                match eval_block(t, body, env)? {
                    Flow::Val(v) | Flow::Continue(v) => acc = v_add(t, &acc, &v),
                    Flow::Break(v) => {
                        acc = v_add(t, &acc, &v);
                        break;
                    }
                }
            }
            Ok(Flow::Val(acc))
        }
        Stmt::If(cond, then_body, else_body) => {
            let c = eval_expr(t, cond, env)?;
            check_no_param_branch(env, &c)?;
            let body = if c.to_f64(t)? != 0.0 {
                then_body
            } else {
                else_body
            };
            eval_block(t, body, env)
        }
        Stmt::Break => Ok(Flow::Break(Val::Num(0.0))),
        Stmt::Continue => Ok(Flow::Continue(Val::Num(0.0))),
        Stmt::Return(_) => Ok(Flow::Val(Val::Num(0.0))),
    }
}

/// Zero value matching a declaration's shape, used when a local is declared
/// without an initializer (`vector[N] y_rep;`). Sizing it correctly matters
/// for `generated quantities`, where the declared shape decides how many
/// columns the draw matrix gets.
fn default_for_type(t: &mut Tape, typ: &StanType, env: &Env) -> Result<Val> {
    fn size_of(t: &mut Tape, e: &Expr, env: &Env) -> Result<usize> {
        Ok(eval_expr(t, e, env)?.to_i32(t)?.max(0) as usize)
    }
    Ok(match typ {
        StanType::Real(_) | StanType::Int(_) => Val::Num(0.0),
        StanType::Vector(n, _)
        | StanType::Simplex(n)
        | StanType::Ordered(n)
        | StanType::PositiveOrdered(n)
        | StanType::UnitVector(n) => Val::Vec(vec![Val::Num(0.0); size_of(t, n, env)?]),
        StanType::Matrix(r, c) => {
            let (rows, cols) = (size_of(t, r, env)?, size_of(t, c, env)?);
            Val::Vec(vec![Val::Vec(vec![Val::Num(0.0); cols]); rows])
        }
        StanType::CholeskyFactorCorr(k)
        | StanType::CholeskyFactorCov(k)
        | StanType::CovMatrix(k)
        | StanType::CorrMatrix(k) => {
            let kk = size_of(t, k, env)?;
            Val::Vec(vec![Val::Vec(vec![Val::Num(0.0); kk]); kk])
        }
        StanType::Array(n, elem) => {
            let len = size_of(t, n, env)?;
            let proto = default_for_type(t, elem, env)?;
            Val::Vec(vec![proto; len])
        }
    })
}

/// See `Env::strict_no_param_branch` — refuses to freeze a parameter-dependent
/// branch into the one-shot `model`/`transformed parameters` tape.
fn check_no_param_branch(env: &Env, cond: &Val) -> Result<()> {
    if env.strict_no_param_branch() && matches!(cond, Val::Tape(_)) {
        return Err(EvalError::ParamDependentBranch);
    }
    Ok(())
}
