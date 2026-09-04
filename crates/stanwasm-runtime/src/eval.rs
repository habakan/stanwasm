//! AST evaluator. Walks Stan AST, pushes tape ops, returns Val.

use crate::distributions::{eval_dist, eval_sample_vec};
use crate::env::Env;
use crate::error::EvalError;
use crate::matrix;
use crate::ops::{
    v_abs, v_acos, v_add, v_asin, v_atan, v_cos, v_div, v_exp, v_inv_logit, v_lgamma, v_log,
    v_logit, v_mul, v_neg, v_phi, v_pow, v_sin, v_sqrt, v_sub, v_sum, v_tan, v_tanh,
};
use crate::value::{Shape, Val};
use stanwasm_ast::{Expr, FuncDef, SliceIdx, StanType, Stmt};
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
        // `&&` and `||` short-circuit, which Stan relies on to guard the operand
        // beside them: `i <= n && x[i] > 0` must not index past the end.
        Expr::BinOp(op, l, r) if op == "&&" || op == "||" => {
            let lv = eval_expr(t, l, env)?;
            check_no_param_branch(env, &lv)?;
            let lb = lv.to_f64(t)? != 0.0;
            if lb != (op == "&&") {
                return Ok(bool_val(lb));
            }
            let rv = eval_expr(t, r, env)?;
            check_no_param_branch(env, &rv)?;
            Ok(bool_val(rv.to_f64(t)? != 0.0))
        }
        Expr::BinOp(op, l, r) => {
            let lv = eval_expr(t, l, env)?;
            let rv = eval_expr(t, r, env)?;
            if matches!(op.as_str(), "==" | "!=" | "<" | ">" | "<=" | ">=") {
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
                "'" => transpose(&v),
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
                Val::Vec(xs) | Val::Row(xs) => {
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
        Expr::Slice(base_e, idxs) => {
            let base = eval_expr(t, base_e, env)?;
            let path = resolve_idxs(t, idxs, env)?;
            slice_val(&base, &path)
        }
        Expr::Call(name, args) => eval_call(t, name, args, env),
    }
}

/// A `SliceIdx` with its bounds evaluated. One-based, like Stan's; `None` is
/// the container's own end, which is only known once the walk reaches it.
#[derive(Debug, Clone, Copy)]
enum Path {
    At(i32),
    Range(Option<i32>, Option<i32>),
}

fn resolve_idxs(t: &mut Tape, idxs: &[SliceIdx], env: &Env) -> Result<Vec<Path>> {
    idxs.iter()
        .map(|i| {
            Ok(match i {
                SliceIdx::At(e) => Path::At(eval_expr(t, e, env)?.to_i32(t)?),
                SliceIdx::Range(lo, hi) => Path::Range(
                    lo.as_ref()
                        .map(|e| eval_expr(t, e, env)?.to_i32(t))
                        .transpose()?,
                    hi.as_ref()
                        .map(|e| eval_expr(t, e, env)?.to_i32(t))
                        .transpose()?,
                ),
            })
        })
        .collect()
}

/// The half-open span a range covers, checked against the container it indexes.
/// `hi == lo - 1` is Stan's empty slice and stays legal.
fn range_bounds(lo: Option<i32>, hi: Option<i32>, len: usize) -> Result<(i32, i32)> {
    let lo = lo.unwrap_or(1);
    let hi = hi.unwrap_or(len as i32);
    if lo < 1 || hi > len as i32 || hi < lo - 1 {
        return Err(EvalError::IndexOutOfBounds {
            index: if lo < 1 { lo } else { hi },
            len,
        });
    }
    Ok((lo, hi))
}

fn slice_val(v: &Val, path: &[Path]) -> Result<Val> {
    let Some((head, rest)) = path.split_first() else {
        return Ok(v.clone());
    };
    let Some(xs) = v.elems() else {
        return Err(EvalError::NotAScalar);
    };
    match head {
        Path::At(i) => {
            let at = usize::try_from(i - 1).ok().and_then(|k| xs.get(k)).ok_or(
                EvalError::IndexOutOfBounds {
                    index: *i,
                    len: xs.len(),
                },
            )?;
            slice_val(at, rest)
        }
        Path::Range(lo, hi) => {
            let (lo, hi) = range_bounds(*lo, *hi, xs.len())?;
            let mut out = Vec::with_capacity((hi - lo + 1).max(0) as usize);
            for k in lo..=hi {
                out.push(slice_val(&xs[(k - 1) as usize], rest)?);
            }
            Ok(v.like(out))
        }
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
        Expr::Index(base, _) | Expr::Slice(base, _) => is_int_expr(base, env),
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
/// Orientation only ever changes what `*` does, so everything else — the
/// remaining builtins, and every density — reads the same list of numbers
/// either way. A `Val::Row` left in place would fall through their
/// `Val::Vec` arms unmatched.
fn drop_orientation(args: Vec<Val>) -> Vec<Val> {
    args.into_iter()
        .map(|v| match v {
            Val::Row(xs) => Val::Vec(xs),
            other => other,
        })
        .collect()
}

/// A value seen as rows of elements: a matrix as itself, a column vector as one
/// column, a row vector as one row, a scalar as a single cell.
fn as_rows(v: &Val) -> Vec<Vec<Val>> {
    match v {
        Val::Row(xs) => vec![xs.clone()],
        Val::Vec(xs) if xs.iter().any(|x| x.elems().is_some()) => xs
            .iter()
            .map(|r| {
                r.elems()
                    .map(<[Val]>::to_vec)
                    .unwrap_or_else(|| vec![r.clone()])
            })
            .collect(),
        Val::Vec(xs) => xs.iter().map(|x| vec![x.clone()]).collect(),
        scalar => vec![vec![scalar.clone()]],
    }
}

/// One element of a `[...]` literal as a row.
fn as_rows_flat(v: &Val) -> Vec<Val> {
    v.elems()
        .map(<[Val]>::to_vec)
        .unwrap_or_else(|| vec![v.clone()])
}

/// Rows into a matrix, refusing a ragged one — the shape a `Val` cannot carry.
fn same_width(op: &str, rows: Vec<Vec<Val>>) -> Result<Vec<Val>> {
    let w = rows.first().map_or(0, Vec::len);
    match rows.iter().find(|r| r.len() != w) {
        Some(bad) => Err(EvalError::ShapeMismatch {
            op: op.to_string(),
            lhs: Shape::Vector(w).to_string(),
            rhs: Shape::Vector(bad.len()).to_string(),
        }),
        None => Ok(rows.into_iter().map(Val::Vec).collect()),
    }
}

/// Rows of a matrix as columns. The caller has already established that every
/// element is a container, so a shorter row simply contributes fewer entries.
fn transpose_rows(rows: &[Val]) -> Vec<Val> {
    let cols = rows
        .iter()
        .filter_map(|r| r.elems())
        .map(<[Val]>::len)
        .max()
        .unwrap_or(0);
    (0..cols)
        .map(|j| {
            Val::Vec(
                rows.iter()
                    .filter_map(|r| r.elems().and_then(|xs| xs.get(j)).cloned())
                    .collect(),
            )
        })
        .collect()
}

/// Stan's `'`. A scalar is its own transpose; a vector and a row vector swap;
/// a matrix reflects. The orientation is what makes `x' * y` an inner product.
fn transpose(v: &Val) -> Val {
    match v {
        Val::Num(_) | Val::Tape(_) => v.clone(),
        Val::Row(xs) => Val::Vec(xs.clone()),
        Val::Vec(xs) if xs.iter().any(|x| x.elems().is_some()) => Val::Vec(transpose_rows(xs)),
        Val::Vec(xs) => Val::Row(xs.clone()),
    }
}

/// otherwise; `check_binop_shapes` has already rejected the mismatched cases.
fn mul_or_matmul(t: &mut Tape, lhs: &Val, rhs: &Val) -> Result<Val> {
    use Shape::*;
    match (lhs.shape(), rhs.shape()) {
        // row · column is Stan's inner product, and column · row its outer one.
        (RowVector(_), Vector(_)) => {
            let (Some(a), Some(b)) = (lhs.elems(), rhs.elems()) else {
                return Err(EvalError::NotAScalar);
            };
            let terms: Vec<Val> = a.iter().zip(b).map(|(x, y)| v_mul(t, x, y)).collect();
            Ok(v_sum(t, &terms))
        }
        (Vector(_), RowVector(_)) => {
            let (Some(a), Some(b)) = (lhs.elems(), rhs.elems()) else {
                return Err(EvalError::NotAScalar);
            };
            let mut rows = Vec::with_capacity(a.len());
            for x in a {
                rows.push(Val::Vec(b.iter().map(|y| v_mul(t, x, y)).collect()));
            }
            Ok(Val::Vec(rows))
        }
        // A row vector on the left of a matrix is `(Mᵀ r)ᵀ`.
        (RowVector(_), Matrix(..)) => {
            let (Some(r), Val::Vec(m)) = (lhs.elems(), rhs) else {
                return Err(EvalError::NotAScalar);
            };
            let mt = transpose_rows(m);
            Ok(Val::Row(matrix::mat_vec_mul(t, &mt, r)))
        }
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
        // `*` between two 1-D operands is linear algebra: row·column is the inner
        // product, column·row the outer one. Two of the same orientation is a
        // type error in Stan, and answering it element-wise would be a wrong
        // dot product.
        (RowVector(a), Vector(b)) if op == "*" => a == b,
        (Vector(_), RowVector(_)) if op == "*" => true,
        (Vector(_) | RowVector(_), Vector(_) | RowVector(_)) if op == "*" => false,
        // Every other operator is element-wise, and orientation doesn't change
        // what it computes, so a row and a column of one length are accepted.
        (Vector(a) | RowVector(a), Vector(b) | RowVector(b)) => a == b,
        // The linear-algebra cases, handled in `mul_or_matmul`.
        (RowVector(a), Matrix(rb, Some(_))) => op == "*" && a == rb,
        (Matrix(_, Some(ca)), Matrix(rb, Some(cb))) if op == "*" => {
            let _ = cb;
            ca == rb
        }
        (Matrix(ra, Some(ca)), Matrix(rb, Some(cb))) => ra == rb && ca == cb,
        (Matrix(..), Matrix(..)) => false,
        (Matrix(_, Some(ca)), Vector(b)) => op == "*" && ca == b,
        (Matrix(..), Vector(_) | RowVector(_)) | (Vector(_) | RowVector(_), Matrix(..)) => false,
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

/// `m + log ∑ exp(xᵢ − m)`.
///
/// The identity holds for any `m`, so the shift is chosen once while tracing —
/// at the largest element there — and stays that element for every parameter
/// value afterwards. The result is exact either way; only how much cancellation
/// it survives depends on the choice.
fn log_sum_exp(t: &mut Tape, xs: &[Val]) -> Result<Val> {
    let mut at = 0;
    let mut best = f64::NEG_INFINITY;
    for (i, x) in xs.iter().enumerate() {
        let v = x.to_f64(t)?;
        if v > best {
            best = v;
            at = i;
        }
    }
    let shift = xs.get(at).ok_or(EvalError::NotAScalar)?.clone();
    let mut acc = Val::Num(0.0);
    for x in xs {
        let d = v_sub(t, x, &shift);
        let e = v_exp(t, &d);
        acc = v_add(t, &acc, &e);
    }
    let l = v_log(t, &acc);
    Ok(v_add(t, &shift, &l))
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
    /// The index path from the outermost expression down to the root binding.
    fn walk<'a>(t: &mut Tape, e: &'a Expr, env: &Env, path: &mut Vec<Path>) -> Result<&'a String> {
        match e {
            Expr::Var(name) => Ok(name),
            Expr::Index(base, idx_e) => {
                let root = walk(t, base, env, path)?;
                path.push(Path::At(eval_expr(t, idx_e, env)?.to_i32(t)?));
                Ok(root)
            }
            Expr::Slice(base, idxs) => {
                let root = walk(t, base, env, path)?;
                path.extend(resolve_idxs(t, idxs, env)?);
                Ok(root)
            }
            _ => Err(EvalError::UnsupportedAssignmentTarget),
        }
    }

    fn put(container: &mut Val, path: &[Path], val: &Val) -> Result<()> {
        let Some((head, rest)) = path.split_first() else {
            *container = val.clone();
            return Ok(());
        };
        let (Val::Vec(xs) | Val::Row(xs)) = container else {
            return Err(EvalError::NotAScalar);
        };
        let len = xs.len();
        match head {
            Path::At(i) => {
                let slot = usize::try_from(i - 1)
                    .ok()
                    .and_then(|k| xs.get_mut(k))
                    .ok_or(EvalError::IndexOutOfBounds { index: *i, len })?;
                put(slot, rest, val)
            }
            Path::Range(lo, hi) => {
                let (lo, hi) = range_bounds(*lo, *hi, len)?;
                // A container on the right is written across the span; a scalar
                // is written into every slot of it.
                let src = val.elems();
                if let Some(es) = src {
                    if es.len() as i32 != hi - lo + 1 {
                        return Err(EvalError::ShapeMismatch {
                            op: "=".into(),
                            lhs: Shape::Vector((hi - lo + 1).max(0) as usize).to_string(),
                            rhs: val.shape().to_string(),
                        });
                    }
                }
                for (k, pos) in (lo..=hi).enumerate() {
                    let piece = match src {
                        Some(es) => &es[k],
                        None => val,
                    };
                    put(&mut xs[(pos - 1) as usize], rest, piece)?;
                }
                Ok(())
            }
        }
    }

    let mut path = Vec::new();
    let root = walk(t, lhs, env, &mut path)?.clone();
    let mut updated = env
        .get(&root)
        .cloned()
        .ok_or(EvalError::UndefinedVariable(root.clone()))?;
    put(&mut updated, &path, &val)?;
    env.set(&root, updated);
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
    // These read their operands' orientation — which way a 1-D value lies decides
    // the result's shape — so they are answered before it is dropped below.
    match (name, evaled.as_slice()) {
        ("rep_matrix", [v, n_e]) => {
            let n = n_e.to_i32(t)?.max(0) as usize;
            return Ok(match v {
                Val::Row(xs) => Val::Vec(vec![Val::Vec(xs.clone()); n]),
                Val::Vec(xs) => Val::Vec(xs.iter().map(|x| Val::Vec(vec![x.clone(); n])).collect()),
                scalar => Val::Vec(vec![Val::Vec(vec![scalar.clone(); n]); n]),
            });
        }
        // A matrix or a row vector on either side stacks rows; two columns, or
        // scalars, run together into one column.
        ("append_row", [a, b]) => {
            let lies_across =
                |v: &Val| matches!(v.shape(), Shape::Matrix(..) | Shape::RowVector(_));
            let mut rows = as_rows(a);
            rows.extend(as_rows(b));
            if !lies_across(a) && !lies_across(b) {
                return Ok(Val::Vec(rows.into_iter().flatten().collect()));
            }
            return Ok(Val::Vec(same_width("append_row", rows)?));
        }
        // Mirrors `append_row`: two rows stay a row, anything else is a matrix.
        ("append_col", [a, b]) => {
            let (ra, rb) = (as_rows(a), as_rows(b));
            if ra.len() != rb.len() {
                return Err(EvalError::ShapeMismatch {
                    op: "append_col".into(),
                    lhs: a.shape().to_string(),
                    rhs: b.shape().to_string(),
                });
            }
            let joined: Vec<Vec<Val>> = ra
                .into_iter()
                .zip(rb)
                .map(|(mut x, y)| {
                    x.extend(y);
                    x
                })
                .collect();
            let lies_along = |v: &Val| matches!(v.shape(), Shape::RowVector(_) | Shape::Scalar);
            if lies_along(a) && lies_along(b) {
                return Ok(Val::Row(joined.into_iter().flatten().collect()));
            }
            return Ok(Val::Vec(same_width("append_col", joined)?));
        }
        _ => {}
    }
    let evaled = drop_orientation(evaled);
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
        ("log10", [a]) => {
            let l = v_log(t, a);
            v_div(t, &l, &Val::Num(std::f64::consts::LN_10))
        }
        ("log_sum_exp", [Val::Vec(xs)]) => log_sum_exp(t, xs)?,
        ("log_sum_exp", [a, b]) => log_sum_exp(t, &[a.clone(), b.clone()])?,
        ("log_mix", [theta, a, b]) => {
            let log_theta = v_log(t, theta);
            let one_minus = v_sub(t, &Val::Num(1.0), theta);
            let log_1m = v_log(t, &one_minus);
            let first = v_add(t, &log_theta, a);
            let second = v_add(t, &log_1m, b);
            log_sum_exp(t, &[first, second])?
        }
        // Stan's `sd` is the sample standard deviation: the denominator is
        // `n - 1`, not `n`.
        ("sd", [Val::Vec(xs)]) => {
            if xs.len() < 2 {
                return Err(EvalError::NotAScalar);
            }
            let n = xs.len() as f64;
            let mut acc = Val::Num(0.0);
            for x in xs {
                acc = v_add(t, &acc, x);
            }
            let mean = v_div(t, &acc, &Val::Num(n));
            let mut ss = Val::Num(0.0);
            for x in xs {
                let d = v_sub(t, x, &mean);
                let sq = v_mul(t, &d, &d);
                ss = v_add(t, &ss, &sq);
            }
            let var = v_div(t, &ss, &Val::Num(n - 1.0));
            v_sqrt(t, &var)
        }
        // The decomposition is already here for `multi_normal`; this only
        // makes it reachable from a model.
        ("cholesky_decompose", [Val::Vec(rows)]) => Val::Vec(matrix::cholesky_decompose(t, rows)),
        ("diag_matrix", [Val::Vec(diag)]) => {
            let n = diag.len();
            let mut rows = Vec::with_capacity(n);
            for (i, d) in diag.iter().enumerate() {
                let mut row = vec![Val::Num(0.0); n];
                row[i] = d.clone();
                rows.push(Val::Vec(row));
            }
            Val::Vec(rows)
        }
        ("diag_pre_multiply", [Val::Vec(diag), Val::Vec(rows)]) => {
            if diag.len() != rows.len() {
                return Err(EvalError::ShapeMismatch {
                    op: "diag_pre_multiply".into(),
                    lhs: Shape::Vector(diag.len()).to_string(),
                    rhs: Shape::Vector(rows.len()).to_string(),
                });
            }
            let mut out = Vec::with_capacity(rows.len());
            for (scale, row) in diag.iter().zip(rows) {
                out.push(v_mul(t, row, scale));
            }
            Val::Vec(out)
        }
        // `alpha² exp(-(xᵢ - xⱼ)² / 2ρ²)`. The differences are data, so only the
        // two scale parameters reach the tape; the matrix is still N² nodes.
        ("gp_exp_quad_cov", [Val::Vec(xs), alpha, rho]) => {
            let a2 = v_mul(t, alpha, alpha);
            let r2 = v_mul(t, rho, rho);
            let denom = v_mul(t, &Val::Num(2.0), &r2);
            let coords: Vec<f64> = xs.iter().map(|x| x.to_f64(t)).collect::<Result<_>>()?;
            let mut rows = Vec::with_capacity(coords.len());
            for xi in &coords {
                let mut row = Vec::with_capacity(coords.len());
                for xj in &coords {
                    let d = xi - xj;
                    let q = v_div(t, &Val::Num(-(d * d)), &denom);
                    let e = v_exp(t, &q);
                    row.push(v_mul(t, &a2, &e));
                }
                rows.push(Val::Vec(row));
            }
            Val::Vec(rows)
        }
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
        // Stan's `min`/`max` return the extreme element itself, so the gradient
        // reaches whichever one that is. Which one is chosen while tracing, so a
        // container of parameters follows the same rule as a branch on one.
        ("min", [Val::Vec(xs)]) | ("max", [Val::Vec(xs)]) => {
            let want_max = name == "max";
            let mut best: Option<&Val> = None;
            for x in xs {
                check_no_param_branch(env, x)?;
                let better = match best {
                    Some(b) => (x.to_f64(t)? > b.to_f64(t)?) == want_max,
                    None => true,
                };
                if better {
                    best = Some(x);
                }
            }
            best.ok_or(EvalError::NotAScalar)?.clone()
        }
        ("min", [a, b]) | ("max", [a, b]) => {
            check_no_param_branch(env, a)?;
            check_no_param_branch(env, b)?;
            let take_a = (a.to_f64(t)? > b.to_f64(t)?) == (name == "max");
            if take_a {
                a.clone()
            } else {
                b.clone()
            }
        }
        ("cumulative_sum", [Val::Vec(xs)]) => {
            let mut acc = Val::Num(0.0);
            let mut out = Vec::with_capacity(xs.len());
            for x in xs {
                acc = v_add(t, &acc, x);
                out.push(acc.clone());
            }
            Val::Vec(out)
        }
        ("softmax", [Val::Vec(xs)]) => {
            // exp(xᵢ − max x) / ∑ exp(xⱼ − max x): the shift cancels and keeps
            // the exponentials from overflowing.
            let mut shift = f64::NEG_INFINITY;
            for x in xs {
                shift = shift.max(x.to_f64(t)?);
            }
            let es: Vec<Val> = xs
                .iter()
                .map(|x| {
                    let d = v_sub(t, x, &Val::Num(shift));
                    v_exp(t, &d)
                })
                .collect();
            let total = v_sum(t, &es);
            Val::Vec(es.iter().map(|e| v_div(t, e, &total)).collect())
        }
        ("negative_infinity", []) => Val::Num(f64::NEG_INFINITY),
        ("pi", []) => Val::Num(PI),
        // `tail(v, n)` — the last `n` entries.
        ("tail", [Val::Vec(xs), n_e]) => {
            let n = n_e.to_i32(t)?;
            let start = xs.len() as i32 - n;
            if n < 0 || start < 0 {
                return Err(EvalError::IndexOutOfBounds {
                    index: n,
                    len: xs.len(),
                });
            }
            Val::Vec(xs[start as usize..].to_vec())
        }
        // `to_vector` flattens whatever it is given, in row-major order.
        ("to_vector", [v]) => {
            fn flat(v: &Val, out: &mut Vec<Val>) {
                match v.elems() {
                    Some(xs) => xs.iter().for_each(|x| flat(x, out)),
                    None => out.push(v.clone()),
                }
            }
            let mut out = Vec::new();
            flat(v, &mut out);
            Val::Vec(out)
        }
        // `[a, b, c]`: scalars make a row vector, containers make its rows.
        ("[]", args) if !args.is_empty() => {
            if args.iter().all(|a| a.elems().is_none()) {
                Val::Row(args.to_vec())
            } else {
                Val::Vec(same_width("[]", args.iter().map(as_rows_flat).collect())?)
            }
        }
        // `sub_col(m, i, j, n)` — `n` entries of column `j`, starting at row `i`.
        ("sub_col", [Val::Vec(rows), i_e, j_e, n_e]) => {
            let (i, j, n) = (i_e.to_i32(t)?, j_e.to_i32(t)?, n_e.to_i32(t)?);
            let mut out = Vec::with_capacity(n.max(0) as usize);
            for k in 0..n {
                let row = usize::try_from(i - 1 + k)
                    .ok()
                    .and_then(|r| rows.get(r))
                    .ok_or(EvalError::IndexOutOfBounds {
                        index: i + k,
                        len: rows.len(),
                    })?;
                let cell = row
                    .elems()
                    .and_then(|xs| usize::try_from(j - 1).ok().and_then(|c| xs.get(c)))
                    .ok_or(EvalError::IndexOutOfBounds {
                        index: j,
                        len: row.elems().map_or(1, <[Val]>::len),
                    })?;
                out.push(cell.clone());
            }
            Val::Vec(out)
        }
        // `diag(v) m diag(v)`, which is how a correlation matrix and a vector of
        // scales become a covariance.
        ("quad_form_diag", [Val::Vec(rows), Val::Vec(d)]) => {
            let mut out = Vec::with_capacity(rows.len());
            for (i, row) in rows.iter().enumerate() {
                let Some(cells) = row.elems() else {
                    return Err(EvalError::NotAScalar);
                };
                let mut scaled = Vec::with_capacity(cells.len());
                for (j, c) in cells.iter().enumerate() {
                    let left = v_mul(t, &d[i], c);
                    scaled.push(v_mul(t, &left, &d[j]));
                }
                out.push(Val::Vec(scaled));
            }
            Val::Vec(out)
        }
        // `L Lᵀ` from the lower triangle of `L`, so entry (i, j) sums to
        // `min(i, j)` rather than over the whole row.
        ("multiply_lower_tri_self_transpose", [Val::Vec(rows)]) => {
            let n = rows.len();
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let mut row = Vec::with_capacity(n);
                for j in 0..n {
                    let mut acc = Val::Num(0.0);
                    for k in 0..=i.min(j) {
                        let (Some(li), Some(lj)) = (rows[i].elems(), rows[j].elems()) else {
                            return Err(EvalError::NotAScalar);
                        };
                        let p = v_mul(t, &li[k], &lj[k]);
                        acc = v_add(t, &acc, &p);
                    }
                    row.push(acc);
                }
                out.push(Val::Vec(row));
            }
            Val::Vec(out)
        }
        // Sizes are structural, so they read off the container rather than the tape.
        ("dims", [v]) => {
            let mut out = Vec::new();
            let mut cur = v;
            while let Some(xs) = cur.elems() {
                out.push(Val::Num(xs.len() as f64));
                match xs.first() {
                    Some(first) => cur = first,
                    None => break,
                }
            }
            Val::Vec(out)
        }
        ("size", [Val::Vec(xs)]) | ("num_elements", [Val::Vec(xs)]) | ("rows", [Val::Vec(xs)]) => {
            Val::Num(xs.len() as f64)
        }
        ("size", [_]) | ("num_elements", [_]) => Val::Num(1.0),
        ("cols", [Val::Vec(xs)]) => Val::Num(match xs.first() {
            Some(Val::Vec(row)) => row.len() as f64,
            _ => 1.0,
        }),
        ("rep_row_vector", [v, n_e]) => {
            let n = n_e.to_i32(t)?.max(0) as usize;
            Val::Row(vec![v.clone(); n])
        }
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
        ("dot_self", [Val::Vec(a)]) => {
            let mut acc = Val::Num(0.0);
            for x in a {
                let sq = v_mul(t, x, x);
                acc = v_add(t, &acc, &sq);
            }
            acc
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
                let rest = drop_orientation(args[1..].to_vec());
                match x {
                    Val::Vec(xs) | Val::Row(xs) => eval_sample_vec(t, base, xs, &rest)?,
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
            let evaled_args = drop_orientation(
                args.iter()
                    .map(|a| eval_expr(t, a, env))
                    .collect::<Result<Vec<_>>>()?,
            );
            let v = match &x {
                Val::Vec(xs) | Val::Row(xs) => eval_sample_vec(t, dist, xs, &evaled_args)?,
                _ => eval_dist(t, dist, &x, &evaled_args)?,
            };
            Ok(Flow::Val(v))
        }
        Stmt::TargetIncr(e) => Ok(Flow::Val(eval_expr(t, e, env)?)),
        Stmt::Block(body) => eval_block(t, body, env),
        Stmt::IncrAssign(lhs, rhs) => {
            // For target += rhs (lhs is `target`), already handled above.
            // Generic form: lhs += rhs.
            if let Expr::Index(..) | Expr::Slice(..) = lhs {
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
                Expr::Index(..) | Expr::Slice(..) => assign_indexed(t, lhs, r, env)?,
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
pub fn default_for_type(t: &mut Tape, typ: &StanType, env: &Env) -> Result<Val> {
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
        StanType::RowVector(n, _) => Val::Row(vec![Val::Num(0.0); size_of(t, n, env)?]),
        StanType::Matrix(r, c, _) => {
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
