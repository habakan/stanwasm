//! AST evaluator. Walks Stan AST, pushes tape ops, returns Val.

use crate::distributions::{eval_dist, eval_sample_vec};
use crate::env::Env;
use crate::ops::{
    v_abs, v_add, v_div, v_exp, v_inv_logit, v_lgamma, v_log, v_logit, v_mul, v_neg, v_phi, v_pow,
    v_sqrt, v_sub, v_tanh,
};
use crate::value::Val;
use stan_ast::{Expr, Stmt};
use stan_autodiff::Tape;

pub fn eval_plain(t: &mut Tape, expr: &Expr, env: &Env) -> Val {
    eval_expr(t, expr, env)
}

pub fn eval_expr(t: &mut Tape, expr: &Expr, env: &Env) -> Val {
    match expr {
        Expr::Num(v) => Val::Num(*v),
        Expr::Var(n) => env
            .get(n)
            .cloned()
            .unwrap_or_else(|| panic!("undefined variable: {n}")),
        Expr::BinOp(op, l, r) => {
            let lv = eval_expr(t, l, env);
            let rv = eval_expr(t, r, env);
            match op.as_str() {
                "+" => v_add(t, &lv, &rv),
                "-" => v_sub(t, &lv, &rv),
                "*" => v_mul(t, &lv, &rv),
                "/" => v_div(t, &lv, &rv),
                "^" => v_pow(t, &lv, &rv),
                "==" => bool_val(lv.to_f64(t) == rv.to_f64(t)),
                "!=" => bool_val(lv.to_f64(t) != rv.to_f64(t)),
                "<" => bool_val(lv.to_f64(t) < rv.to_f64(t)),
                ">" => bool_val(lv.to_f64(t) > rv.to_f64(t)),
                "<=" => bool_val(lv.to_f64(t) <= rv.to_f64(t)),
                ">=" => bool_val(lv.to_f64(t) >= rv.to_f64(t)),
                "&&" => bool_val(lv.to_f64(t) != 0.0 && rv.to_f64(t) != 0.0),
                "||" => bool_val(lv.to_f64(t) != 0.0 || rv.to_f64(t) != 0.0),
                _ => Val::Num(0.0),
            }
        }
        Expr::UnOp(op, e) => {
            let v = eval_expr(t, e, env);
            match op.as_str() {
                "-" => v_neg(t, &v),
                "!" => bool_val(v.to_f64(t) == 0.0),
                _ => v,
            }
        }
        Expr::Index(arr_e, idx_e) => {
            let idx = eval_expr(t, idx_e, env).to_i32(t) - 1;
            let arr = eval_expr(t, arr_e, env);
            match arr {
                Val::Vec(xs) => xs.get(idx as usize).cloned().unwrap_or(Val::Num(0.0)),
                other => other,
            }
        }
        Expr::Call(name, args) => eval_call(t, name, args, env),
    }
}

fn bool_val(b: bool) -> Val {
    Val::Num(if b { 1.0 } else { 0.0 })
}

fn eval_call(t: &mut Tape, name: &str, args: &[Expr], env: &Env) -> Val {
    let evaled: Vec<Val> = args.iter().map(|a| eval_expr(t, a, env)).collect();
    match (name, evaled.as_slice()) {
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
            let start = start_v.to_i32(t) as usize - 1;
            let len = len_v.to_i32(t) as usize;
            Val::Vec(xs.iter().skip(start).take(len).cloned().collect())
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
                    Val::Vec(xs) => eval_sample_vec(t, base, xs, &rest).unwrap_or(Val::Num(0.0)),
                    _ => eval_dist(t, base, x, &rest).unwrap_or(Val::Num(0.0)),
                }
            }
        }
        // RNG forms, valid only in generated quantities (env carries an rng).
        (n, args) if n.ends_with("_rng") => {
            let base = &n[..n.len() - 4];
            crate::rng::dispatch(t, base, args, env)
        }
        _ => Val::Num(0.0),
    }
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
fn eval_block(t: &mut Tape, stmts: &[Stmt], env: &mut Env) -> Flow {
    let saved = env.len();
    let mut acc = Val::Num(0.0);
    let mut result = None;
    for s in stmts {
        match eval_stmt(t, s, env) {
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
    result.unwrap_or(Flow::Val(acc))
}

/// Evaluate a statement; returns the increment to log_prob (zero for non-target
/// statements) wrapped in a `Flow` that also carries `break`/`continue` signals.
pub fn eval_stmt(t: &mut Tape, stmt: &Stmt, env: &mut Env) -> Flow {
    match stmt {
        Stmt::Sample(lhs, dist, args) => {
            let x = eval_expr(t, lhs, env);
            let evaled_args: Vec<Val> = args.iter().map(|a| eval_expr(t, a, env)).collect();
            let v = match &x {
                Val::Vec(xs) => eval_sample_vec(t, dist, xs, &evaled_args).unwrap_or(Val::Num(0.0)),
                _ => eval_dist(t, dist, &x, &evaled_args).unwrap_or(Val::Num(0.0)),
            };
            Flow::Val(v)
        }
        Stmt::TargetIncr(e) => Flow::Val(eval_expr(t, e, env)),
        Stmt::IncrAssign(lhs, rhs) => {
            // For target += rhs (lhs is `target`), already handled above.
            // Generic form: lhs += rhs. Update env if lhs is a Var.
            if let Expr::Var(name) = lhs {
                if name == "target" {
                    return Flow::Val(eval_expr(t, rhs, env));
                }
                let cur = env.get(name).cloned().unwrap_or(Val::Num(0.0));
                let r = eval_expr(t, rhs, env);
                let new_val = v_add(t, &cur, &r);
                env.set(name, new_val);
            }
            Flow::Val(Val::Num(0.0))
        }
        Stmt::Assign(lhs, rhs) => {
            let r = eval_expr(t, rhs, env);
            if let Expr::Var(name) = lhs {
                env.set(name, r);
            }
            Flow::Val(Val::Num(0.0))
        }
        Stmt::LocalDecl(_typ, name, init) => {
            let v = match init {
                Some(e) => eval_expr(t, e, env),
                None => Val::Num(0.0),
            };
            env.set(name, v);
            Flow::Val(Val::Num(0.0))
        }
        Stmt::For(var, lo_e, hi_e, body) => {
            let lo = eval_expr(t, lo_e, env).to_i32(t);
            let hi = eval_expr(t, hi_e, env).to_i32(t);
            let saved_len = env.len();
            let mut acc = Val::Num(0.0);
            for i in lo..=hi {
                env.set(var, Val::Num(i as f64));
                match eval_block(t, body, env) {
                    Flow::Val(v) | Flow::Continue(v) => acc = v_add(t, &acc, &v),
                    Flow::Break(v) => {
                        acc = v_add(t, &acc, &v);
                        break;
                    }
                }
            }
            env.truncate(saved_len);
            Flow::Val(acc)
        }
        Stmt::While(cond, body) => {
            const MAX_ITERS: u64 = 1_000_000;
            let mut acc = Val::Num(0.0);
            let mut iters: u64 = 0;
            loop {
                let c = eval_expr(t, cond, env);
                if c.to_f64(t) == 0.0 {
                    break;
                }
                iters += 1;
                if iters > MAX_ITERS {
                    panic!("while loop exceeded {MAX_ITERS} iterations — possible infinite loop");
                }
                match eval_block(t, body, env) {
                    Flow::Val(v) | Flow::Continue(v) => acc = v_add(t, &acc, &v),
                    Flow::Break(v) => {
                        acc = v_add(t, &acc, &v);
                        break;
                    }
                }
            }
            Flow::Val(acc)
        }
        Stmt::If(cond, then_body, else_body) => {
            let c = eval_expr(t, cond, env);
            let body = if c.to_f64(t) != 0.0 {
                then_body
            } else {
                else_body
            };
            eval_block(t, body, env)
        }
        Stmt::Break => Flow::Break(Val::Num(0.0)),
        Stmt::Continue => Flow::Continue(Val::Num(0.0)),
        Stmt::Return(_) => Flow::Val(Val::Num(0.0)),
    }
}
