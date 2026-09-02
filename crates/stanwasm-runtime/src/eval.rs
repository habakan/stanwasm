//! AST evaluator. Walks Stan AST, pushes tape ops, returns Val.

use crate::distributions::{eval_dist, eval_sample_vec};
use crate::env::Env;
use crate::error::EvalError;
use crate::matrix;
use crate::ops::{
    v_abs, v_acos, v_add, v_asin, v_atan, v_cos, v_div, v_exp, v_inv_logit, v_lgamma, v_log,
    v_logit, v_mul, v_neg, v_phi, v_pow, v_sin, v_sqrt, v_sub, v_tan, v_tanh,
};
use crate::value::{Shape, Val};
use stanwasm_ast::{Expr, FuncDef, StanType, Stmt};
use stanwasm_autodiff::Tape;
use std::f64::consts::PI;

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
                // Comparison results collapse to `Val::Num`, so checking the result
                // can never catch a parameter-dependent condition. Check the operands.
                check_no_param_branch(env, &lv)?;
                check_no_param_branch(env, &rv)?;
            }
            check_binop_shapes(op, &lv, &rv)?;
            Ok(match op.as_str() {
                "+" => v_add(t, &lv, &rv),
                "-" => v_sub(t, &lv, &rv),
                "*" => mul_or_matmul(t, &lv, &rv)?,
                // Always element-wise, whatever the ranks — that is the whole point
                // of the dotted spelling.
                ".*" => v_mul(t, &lv, &rv),
                "./" => v_div(t, &lv, &rv),
                ".^" => v_pow(t, &lv, &rv),
                // `int / int` truncates toward zero (`N / 2` with `N = 3` is 1).
                // Int-ness is a property of the declarations, so it comes from the tree.
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
        // Both branches would be recorded if this were built from arithmetic, so it
        // evaluates only the taken one — and the condition follows the same
        // parameter-dependence rule as `if`.
        Expr::Ternary(cond_e, then_e, else_e) => {
            let c = eval_expr(t, cond_e, env)?;
            check_no_param_branch(env, &c)?;
            if c.to_f64(t)? != 0.0 {
                eval_expr(t, then_e, env)
            } else {
                eval_expr(t, else_e, env)
            }
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

/// Whether an expression has an integral Stan type. Only `/` cares, but it has to
/// propagate through the arithmetic feeding it: `(N + 1) / 2` is still integer.
fn is_int_expr(e: &Expr, env: &Env) -> bool {
    match e {
        Expr::IntNum(_) => true,
        Expr::Num(_) => false,
        Expr::Var(n) => env.is_int(n),
        // `y[i]` is an int exactly when `y` is an `array[...] int`.
        Expr::Index(base, _) => is_int_expr(base, env),
        // Int only if it is int whichever way the condition goes.
        Expr::Ternary(_, a, b) => is_int_expr(a, env) && is_int_expr(b, env),
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

/// Stan's `*` is a matrix product when the ranks call for one and element-wise
/// otherwise; `check_binop_shapes` has already rejected the mismatched cases.
fn mul_or_matmul(t: &mut Tape, lhs: &Val, rhs: &Val) -> Result<Val> {
    use Shape::*;
    match (lhs.shape(), rhs.shape()) {
        (Matrix(_, Some(_)), Vector(_)) => {
            let (Val::Vec(a), Val::Vec(b)) = (lhs, rhs) else {
                return Err(EvalError::NotAScalar);
            };
            Ok(Val::Vec(matrix::mat_vec_mul(t, a, b)))
        }
        (Matrix(_, Some(_)), Matrix(_, Some(cb))) => {
            let (Val::Vec(a), Val::Vec(b)) = (lhs, rhs) else {
                return Err(EvalError::NotAScalar);
            };
            Ok(Val::Vec(matrix::mat_mat_mul(t, a, b, cb)))
        }
        _ => Ok(v_mul(t, lhs, rhs)),
    }
}

/// Reject operand shapes this runtime would otherwise answer wrongly: `ops::v_*`
/// broadcast and `zip` truncates, so `vector[3] + vector[2]` would quietly work.
fn check_binop_shapes(op: &str, lhs: &Val, rhs: &Val) -> Result<()> {
    use Shape::*;
    let (ls, rs) = (lhs.shape(), rhs.shape());
    let ok = match (ls, rs) {
        (Scalar, Scalar) => true,
        // scalar ⊙ container broadcasts element-wise, in both directions.
        (Scalar, _) | (_, Scalar) => true,
        (Vector(a), Vector(b)) => a == b,
        // `*` is a matrix product, so it needs the inner dimensions to meet; every
        // other operator is element-wise and needs both to agree. A ragged operand
        // (`cols: None`) matches nothing, so it can't be zipped down silently.
        (Matrix(_, Some(ca)), Matrix(rb, Some(cb))) if op == "*" => {
            let _ = cb;
            ca == rb
        }
        (Matrix(ra, Some(ca)), Matrix(rb, Some(cb))) => ra == rb && ca == cb,
        (Matrix(..), Matrix(..)) => false,
        // `*` on these is linear algebra, handled in eval_binop; any other operator
        // would be an element-wise broadcast across mismatched ranks.
        (Matrix(_, Some(ca)), Vector(b)) => op == "*" && ca == b,
        (Matrix(..), Vector(_)) | (Vector(_), Matrix(..)) => false,
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

/// Inlines a user-defined call: binds the arguments into a scope of their own,
/// runs the body, and evaluates the return expression there.
///
/// Stan passes by constant reference, and this scope is discarded afterwards, so an
/// assignment to a parameter inside the body cannot reach the caller either way.
fn eval_user_call(
    t: &mut Tape,
    name: &str,
    def: &FuncDef,
    argv: Vec<Val>,
    env: &Env,
) -> Result<Val> {
    if env.in_call(name) {
        return Err(EvalError::RecursiveCall(name.to_string()));
    }
    if def.params.len() != argv.len() {
        return Err(EvalError::WrongArity {
            name: name.to_string(),
            expected: def.params.len(),
            got: argv.len(),
        });
    }

    // Starts from the caller's env so data and earlier declarations stay visible;
    // Stan scopes function bodies that way too.
    let mut local = env.clone();
    local.enter_call(name);
    for ((typ, pname), v) in def.params.iter().zip(argv) {
        match typ {
            StanType::Int(_) => local.set_int_typed(pname, v),
            _ => local.set(pname, v),
        }
    }
    for stmt in &def.body {
        eval_stmt(t, stmt, &mut local)?.into_val();
    }
    eval_expr(t, &def.ret_expr, &local)
}

/// Writes `val` into `y[i]` / `M[i, j]`, which the parser nests as
/// `Index(Index(M, i), j)`. Walks down to the root binding, collecting the indices,
/// then rebuilds the containers on the way back out — `Val` is a tree of owned
/// vectors, so the write has to replace each level rather than mutate in place.
fn assign_indexed(t: &mut Tape, lhs: &Expr, val: Val, env: &mut Env) -> Result<()> {
    let mut idxs: Vec<usize> = Vec::new();
    let mut cur = lhs;
    let root = loop {
        match cur {
            Expr::Index(base, idx_e) => {
                let one_based = eval_expr(t, idx_e, env)?.to_i32(t)?;
                if one_based < 1 {
                    return Err(EvalError::IndexOutOfBounds {
                        index: one_based,
                        len: 0,
                    });
                }
                idxs.push((one_based - 1) as usize);
                cur = base;
            }
            Expr::Var(name) => break name,
            _ => return Err(EvalError::UnsupportedAssignmentTarget),
        }
    };
    idxs.reverse();

    fn put(container: &mut Val, idxs: &[usize], val: Val) -> Result<()> {
        let Some((&i, rest)) = idxs.split_first() else {
            *container = val;
            return Ok(());
        };
        let Val::Vec(xs) = container else {
            return Err(EvalError::NotAScalar);
        };
        let len = xs.len();
        let slot = xs.get_mut(i).ok_or(EvalError::IndexOutOfBounds {
            index: i as i32 + 1,
            len,
        })?;
        put(slot, rest, val)
    }

    let mut updated = env
        .get(root)
        .cloned()
        .ok_or_else(|| EvalError::UndefinedVariable(root.clone()))?;
    put(&mut updated, &idxs, val)?;
    env.set(root, updated);
    Ok(())
}

fn eval_call(t: &mut Tape, name: &str, args: &[Expr], env: &Env) -> Result<Val> {
    let evaled: Vec<Val> = args
        .iter()
        .map(|a| eval_expr(t, a, env))
        .collect::<Result<_>>()?;
    if let Some(def) = env.func(name) {
        return eval_user_call(t, name, &def, evaled, env);
    }
    Ok(match (name, evaled.as_slice()) {
        ("log", [a]) => v_log(t, a),
        ("exp", [a]) => v_exp(t, a),
        ("sqrt", [a]) => v_sqrt(t, a),
        ("abs", [a]) | ("fabs", [a]) => v_abs(t, a),
        ("lgamma", [a]) => v_lgamma(t, a),
        ("inv_logit", [a]) | ("logistic", [a]) => v_inv_logit(t, a),
        ("logit", [a]) => v_logit(t, a),
        ("tanh", [a]) => v_tanh(t, a),
        ("sin", [a]) => v_sin(t, a),
        ("cos", [a]) => v_cos(t, a),
        ("tan", [a]) => v_tan(t, a),
        ("asin", [a]) => v_asin(t, a),
        ("acos", [a]) => v_acos(t, a),
        ("atan", [a]) => v_atan(t, a),
        // Stan's two-argument arctangent. The tape has no atan2 node, so it is built
        // from atan plus the quadrant correction, which keeps the gradient exact.
        ("atan2", [y, x]) => {
            let q = v_div(t, y, x);
            let base = v_atan(t, &q);
            let (yv, xv) = (y.to_f64(t)?, x.to_f64(t)?);
            if xv >= 0.0 {
                base
            } else {
                let pi = Val::Num(if yv >= 0.0 { PI } else { -PI });
                v_add(t, &base, &pi)
            }
        }
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
        // Sizes are structural, so they read off the container rather than the tape.
        ("size", [Val::Vec(xs)]) | ("num_elements", [Val::Vec(xs)]) | ("rows", [Val::Vec(xs)]) => {
            Val::Num(xs.len() as f64)
        }
        ("size", [_]) | ("num_elements", [_]) => Val::Num(1.0),
        ("cols", [Val::Vec(xs)]) => Val::Num(match xs.first() {
            Some(Val::Vec(row)) => row.len() as f64,
            _ => 1.0,
        }),
        ("rep_vector", [v, n_e]) | ("rep_array", [v, n_e]) => {
            let n = n_e.to_i32(t)?.max(0) as usize;
            Val::Vec(vec![v.clone(); n])
        }
        ("rep_matrix", [v, r_e, c_e]) => {
            let (r, c) = (
                r_e.to_i32(t)?.max(0) as usize,
                c_e.to_i32(t)?.max(0) as usize,
            );
            Val::Vec(vec![Val::Vec(vec![v.clone(); c]); r])
        }
        ("dot_product", [Val::Vec(a), Val::Vec(b)]) => {
            if a.len() != b.len() {
                return Err(EvalError::ShapeMismatch {
                    op: "dot_product".into(),
                    lhs: Shape::Vector(a.len()).to_string(),
                    rhs: Shape::Vector(b.len()).to_string(),
                });
            }
            let mut acc = Val::Num(0.0);
            for (x, y) in a.iter().zip(b) {
                let p = v_mul(t, x, y);
                acc = v_add(t, &acc, &p);
            }
            acc
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

/// Either a log-prob contribution or a loop-control signal, each carrying the
/// log-prob accumulated up to the point of exit.
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

/// Evaluate a statement list as a scoped block: locals stay local, and
/// `break`/`continue` short-circuits while propagating the signal and log-prob.
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
            if let Expr::Index(..) = lhs {
                let cur = eval_expr(t, lhs, env)?;
                let r = eval_expr(t, rhs, env)?;
                let sum = v_add(t, &cur, &r);
                assign_indexed(t, lhs, sum, env)?;
                return Ok(Flow::Val(Val::Num(0.0)));
            }
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
            match lhs {
                Expr::Var(name) => env.set(name, r),
                Expr::Index(..) => assign_indexed(t, lhs, r, env)?,
                _ => return Err(EvalError::UnsupportedAssignmentTarget),
            }
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

/// Zero value matching a declaration's shape, for a local declared without an
/// initializer. The shape decides `generated quantities`' column count.
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
