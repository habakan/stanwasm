//! Constraint transforms with Jacobian.
//!
//! Covered: scalar reals (none, lower, upper, lower_upper), vectors and
//! arrays with element-wise constraints, unconstrained matrices, and the
//! shape transforms `simplex`, `ordered`, `positive_ordered`,
//! `cholesky_factor_corr`, `cholesky_factor_cov`, `cov_matrix` and `unit_vector`.
//!
//! Anything else (`cov_matrix`, `corr_matrix`, `cholesky_factor_cov`,
//! `unit_vector`) is rejected with `EvalError::UnsupportedConstraint` rather
//! than passed through untransformed: an untransformed declaration samples the
//! parameter on the wrong space with no Jacobian, which yields a wrong
//! posterior with no outward sign that anything went wrong.

use crate::env::Env;
use crate::error::EvalError;
use crate::eval::eval_plain;
use crate::matrix;
use crate::ops::{v_add, v_div, v_exp, v_inv_logit, v_log, v_mul, v_sqrt, v_sub, v_tanh};
use crate::value::Val;
use stanwasm_ast::{Constraint, StanType};
use stanwasm_autodiff::Tape;

/// Lower-triangular factor from the raw slice, row-major with each row's diagonal
/// last. Returns the rows, the diagonal's own log Jacobian (Σ of the raw diagonal,
/// since each is exponentiated), and the diagonal entries.
fn tri_from_raw(t: &mut Tape, raw: &[Val], kk: usize) -> (Vec<Val>, Val, Vec<Val>) {
    let mut rows: Vec<Val> = Vec::with_capacity(kk);
    let mut log_jac = Val::Num(0.0);
    let mut diag: Vec<Val> = Vec::with_capacity(kk);
    let mut idx = 0;
    for i in 0..kk {
        let mut row: Vec<Val> = vec![Val::Num(0.0); kk];
        for cell in row.iter_mut().take(i) {
            *cell = raw.get(idx).cloned().unwrap_or(Val::Num(0.0));
            idx += 1;
        }
        let d_raw = raw.get(idx).cloned().unwrap_or(Val::Num(0.0));
        idx += 1;
        log_jac = v_add(t, &log_jac, &d_raw);
        let d = v_exp(t, &d_raw);
        diag.push(d.clone());
        row[i] = d;
        rows.push(Val::Vec(row));
    }
    (rows, log_jac, diag)
}

pub fn param_dims(typ: &StanType, env: &Env) -> usize {
    match typ {
        StanType::Real(_) => 1,
        StanType::Int(_) => 0,
        StanType::Vector(size, _) => eval_plain_int(size, env),
        StanType::Matrix(r, c) => eval_plain_int(r, env) * eval_plain_int(c, env),
        StanType::Simplex(k) => eval_plain_int(k, env).saturating_sub(1),
        StanType::Ordered(k) => eval_plain_int(k, env),
        StanType::Array(size, elem) => eval_plain_int(size, env) * param_dims(elem, env),
        StanType::CholeskyFactorCorr(k) => {
            let kk = eval_plain_int(k, env);
            kk * (kk - 1) / 2
        }
        StanType::CholeskyFactorCov(k) | StanType::CovMatrix(k) => {
            let kk = eval_plain_int(k, env);
            kk * (kk + 1) / 2
        }
        StanType::CorrMatrix(k) => {
            let kk = eval_plain_int(k, env);
            kk * (kk - 1) / 2
        }
        StanType::PositiveOrdered(k) | StanType::UnitVector(k) => eval_plain_int(k, env),
    }
}

/// Sizing only; falls back to 0 on an evaluation error, since a bad size shows
/// up downstream as a wrong param count, not a silently-wrong log density.
fn eval_plain_int(expr: &stanwasm_ast::Expr, env: &Env) -> usize {
    let mut tape = Tape::new();
    match eval_plain(&mut tape, expr, env) {
        Ok(Val::Num(x)) => x as usize,
        _ => 0,
    }
}

/// Apply the constraint transform to unconstrained tape leaves, returning
/// `(constrained_value, log_jacobian)`. `name` only labels errors.
pub fn constrain(
    t: &mut Tape,
    name: &str,
    typ: &StanType,
    raw: &[Val],
    env: &Env,
) -> Result<(Val, Val), EvalError> {
    Ok(match typ {
        StanType::Real(Constraint::None) => (raw[0].clone(), Val::Num(0.0)),
        StanType::Real(Constraint::Lower(lo_e)) => {
            let lo = eval_plain(t, lo_e, env)?;
            let exp_r = v_exp(t, &raw[0]);
            let c = v_add(t, &lo, &exp_r);
            // log|dc/draw| = raw
            (c, raw[0].clone())
        }
        StanType::Real(Constraint::Upper(hi_e)) => {
            let hi = eval_plain(t, hi_e, env)?;
            apply_upper(t, &raw[0], &hi)
        }
        StanType::Real(Constraint::LowerUpper(lo_e, hi_e)) => {
            let lo = eval_plain(t, lo_e, env)?;
            let hi = eval_plain(t, hi_e, env)?;
            apply_lower_upper(t, &raw[0], &lo, &hi)
        }
        StanType::Vector(_, Constraint::None) => (Val::Vec(raw.to_vec()), Val::Num(0.0)),
        StanType::Vector(_, Constraint::Lower(lo_e)) => {
            let lo = eval_plain(t, lo_e, env)?;
            let mut jac = Val::Num(0.0);
            let mut cs = Vec::with_capacity(raw.len());
            for r in raw {
                let exp_r = v_exp(t, r);
                let c = v_add(t, &lo, &exp_r);
                jac = v_add(t, &jac, r);
                cs.push(c);
            }
            (Val::Vec(cs), jac)
        }
        StanType::Vector(_, Constraint::Upper(hi_e)) => {
            let hi = eval_plain(t, hi_e, env)?;
            let mut jac = Val::Num(0.0);
            let mut cs = Vec::with_capacity(raw.len());
            for r in raw {
                let (c, j) = apply_upper(t, r, &hi);
                jac = v_add(t, &jac, &j);
                cs.push(c);
            }
            (Val::Vec(cs), jac)
        }
        StanType::Vector(_, Constraint::LowerUpper(lo_e, hi_e)) => {
            let lo = eval_plain(t, lo_e, env)?;
            let hi = eval_plain(t, hi_e, env)?;
            let mut jac = Val::Num(0.0);
            let mut cs = Vec::with_capacity(raw.len());
            for r in raw {
                let (c, j) = apply_lower_upper(t, r, &lo, &hi);
                jac = v_add(t, &jac, &j);
                cs.push(c);
            }
            (Val::Vec(cs), jac)
        }
        // simplex[K]: K-1 raw → K-dim simplex. Stick-breaking with the (K-1-i)
        // shift, so the zero vector maps to the uniform simplex (1/K, ..., 1/K).
        StanType::Simplex(_) => {
            let k = raw.len() + 1;
            let mut theta = vec![Val::Num(0.0); k];
            let mut log_jac = Val::Num(0.0);
            let mut stick = Val::Num(1.0);
            for i in 0..(k - 1) {
                let shift = ((k - 1 - i) as f64).ln();
                let adj = v_sub(t, &raw[i], &Val::Num(shift));
                let z = v_inv_logit(t, &adj);
                theta[i] = v_mul(t, &stick, &z);
                let one_z = v_sub(t, &Val::Num(1.0), &z);
                let log_stick = v_log(t, &stick);
                let log_z = v_log(t, &z);
                let log_oz = v_log(t, &one_z);
                let term1 = v_add(t, &log_z, &log_oz);
                let term2 = v_add(t, &log_stick, &term1);
                log_jac = v_add(t, &log_jac, &term2);
                stick = v_mul(t, &stick, &one_z);
            }
            theta[k - 1] = stick;
            (Val::Vec(theta), log_jac)
        }
        // ordered[K]: K unconstrained → K reals with μ₀ < μ₁ < … < μ_{K-1}
        // μ₀ = raw[0]; μᵢ = μ_{i-1} + exp(raw[i]); log_jac = ∑_{i≥1} raw[i]
        StanType::Ordered(_) => {
            let k = raw.len();
            let mut mu = vec![Val::Num(0.0); k];
            let mut log_jac = Val::Num(0.0);
            mu[0] = raw[0].clone();
            for i in 1..k {
                let e = v_exp(t, &raw[i]);
                mu[i] = v_add(t, &mu[i - 1], &e);
                log_jac = v_add(t, &log_jac, &raw[i]);
            }
            (Val::Vec(mu), log_jac)
        }
        // positive_ordered[K]: y₀ = exp(raw[0]); yᵢ = y_{i-1} + exp(raw[i])
        StanType::PositiveOrdered(_) => {
            let k = raw.len();
            let mut cs = vec![Val::Num(0.0); k];
            let mut log_jac = Val::Num(0.0);
            let mut prev = Val::Num(0.0);
            for i in 0..k {
                let e = v_exp(t, &raw[i]);
                prev = v_add(t, &prev, &e);
                cs[i] = prev.clone();
                log_jac = v_add(t, &log_jac, &raw[i]);
            }
            (Val::Vec(cs), log_jac)
        }
        // cholesky_factor_corr[K]: K*(K-1)/2 raw → lower-triangular L. Spherical,
        // per row: z = tanh(raw); L[i][j] = z·√rem; jac += ½log rem + log(1−z²); rem *= 1−z².
        StanType::CholeskyFactorCorr(k_e) => {
            let kk = match eval_plain(t, k_e, env)? {
                Val::Num(v) => v as usize,
                _ => 0,
            };
            let mut mat: Vec<Val> = Vec::with_capacity(kk);
            let mut log_jac = Val::Num(0.0);
            let mut idx = 0;
            for i in 0..kk {
                let mut row: Vec<Val> = vec![Val::Num(0.0); kk];
                if i == 0 {
                    row[0] = Val::Num(1.0);
                } else {
                    let mut rem = Val::Num(1.0);
                    #[allow(clippy::needless_range_loop)]
                    for j in 0..i {
                        let z = v_tanh(t, &raw[idx]);
                        idx += 1;
                        let z2 = v_mul(t, &z, &z);
                        let log_rem = v_log(t, &rem);
                        let half_log_rem = v_mul(t, &Val::Num(0.5), &log_rem);
                        let one_minus_z2 = v_sub(t, &Val::Num(1.0), &z2);
                        let log_1mz2 = v_log(t, &one_minus_z2);
                        let term = v_add(t, &half_log_rem, &log_1mz2);
                        log_jac = v_add(t, &log_jac, &term);
                        let sqrt_rem = v_sqrt(t, &rem);
                        row[j] = v_mul(t, &z, &sqrt_rem);
                        rem = v_mul(t, &rem, &one_minus_z2);
                    }
                    row[i] = v_sqrt(t, &rem);
                }
                mat.push(Val::Vec(row));
            }
            (Val::Vec(mat), log_jac)
        }
        // array[N] T — constrain each element and sum the Jacobians. Without this,
        // `array[N] real<lower=0> s;` passed through untransformed and unjacobianed.
        StanType::Array(_, elem) => {
            // `param_dims` is 0 for an element type with no unconstrained
            // representation — `int` (rejected below) or a zero-sized vector.
            if matches!(**elem, StanType::Int(_)) {
                return Err(EvalError::IntParameter(name.to_string()));
            }
            let chunk = param_dims(elem, env);
            if chunk == 0 {
                return Err(EvalError::BadParameterDeclaration {
                    name: name.to_string(),
                    detail: format!(
                        "`array[...] {}` has no unconstrained dimensions — \
                         check that the element size is a positive data value",
                        type_name(elem)
                    ),
                });
            }
            let mut out = Vec::with_capacity(raw.len() / chunk);
            let mut log_jac = Val::Num(0.0);
            for part in raw.chunks(chunk) {
                let (c, j) = constrain(t, name, elem, part, env)?;
                log_jac = v_add(t, &log_jac, &j);
                out.push(c);
            }
            (Val::Vec(out), log_jac)
        }
        // matrix[R, C] — unconstrained, but reshaped into rows so indexing and the
        // matrix-shaped distributions see structure rather than one flat vector.
        StanType::Matrix(r_e, c_e) => {
            let cols = match eval_plain(t, c_e, env)? {
                Val::Num(v) => v as usize,
                other => return Err(bad_size(name, &other)),
            };
            let rows = match eval_plain(t, r_e, env)? {
                Val::Num(v) => v as usize,
                other => return Err(bad_size(name, &other)),
            };
            if cols == 0 || rows * cols != raw.len() {
                return Err(EvalError::BadParameterDeclaration {
                    name: name.to_string(),
                    detail: format!(
                        "`matrix[{rows}, {cols}]` — both sizes must be \
                         positive data values"
                    ),
                });
            }
            let mat = raw.chunks(cols).map(|row| Val::Vec(row.to_vec())).collect();
            (Val::Vec(mat), Val::Num(0.0))
        }
        // cholesky_factor_cov[K]: K(K+1)/2 raw → lower-triangular L with a positive
        // diagonal. L[m][m] = exp(raw), L[m][n<m] = raw; log_jac = Σ raw diagonal.
        StanType::CholeskyFactorCov(k_e) => {
            let kk = eval_plain_int(k_e, env);
            let (mat, log_jac, _) = tri_from_raw(t, raw, kk);
            (Val::Vec(mat), log_jac)
        }
        // cov_matrix[K]: the same L, then Σ = L Lᵀ. `K log 2 + Σ_k (K − k + 1) log L_kk`
        // is the *whole* Jacobian, exp of the diagonal included, so `diag_jac` is not
        // added on top of it the way cholesky_factor_cov does.
        StanType::CovMatrix(k_e) => {
            let kk = eval_plain_int(k_e, env);
            let (l_rows, _diag_jac, diag) = tri_from_raw(t, raw, kk);
            let mut log_jac = Val::Num((kk as f64) * std::f64::consts::LN_2);
            for (i, d) in diag.iter().enumerate() {
                let power = (kk - i) as f64 + 1.0;
                let log_d = v_log(t, d);
                let term = v_mul(t, &Val::Num(power), &log_d);
                log_jac = v_add(t, &log_jac, &term);
            }
            let sigma = matrix::mat_mat_mul_transpose_rhs(t, &l_rows, kk);
            (Val::Vec(sigma), log_jac)
        }
        // unit_vector[K]: x = y/‖y‖. The Jacobian is singular, and Stan compensates
        // with the standard-normal kernel −½ yᵀy, which also pins the radius.
        StanType::UnitVector(k_e) => {
            let kk = eval_plain_int(k_e, env);
            let ys: Vec<Val> = raw.iter().take(kk).cloned().collect();
            let ss = matrix::vec_dot_self(t, &ys);
            let norm = v_sqrt(t, &ss);
            let unit = ys.iter().map(|y| v_div(t, y, &norm)).collect();
            let log_jac = v_mul(t, &Val::Num(-0.5), &ss);
            (Val::Vec(unit), log_jac)
        }
        StanType::Int(_) => return Err(EvalError::IntParameter(name.to_string())),
        other => {
            return Err(EvalError::UnsupportedConstraint {
                name: name.to_string(),
                typ: type_name(other),
            })
        }
    })
}

/// Human-readable Stan type name, for `UnsupportedConstraint` messages.
fn type_name(typ: &StanType) -> String {
    match typ {
        StanType::Real(_) => "real".into(),
        StanType::Int(_) => "int".into(),
        StanType::Vector(..) => "vector".into(),
        StanType::Matrix(..) => "matrix".into(),
        StanType::Simplex(_) => "simplex".into(),
        StanType::Ordered(_) => "ordered".into(),
        StanType::PositiveOrdered(_) => "positive_ordered".into(),
        StanType::Array(_, elem) => format!("array[...] {}", type_name(elem)),
        StanType::CholeskyFactorCorr(_) => "cholesky_factor_corr".into(),
        StanType::CholeskyFactorCov(_) => "cholesky_factor_cov".into(),
        StanType::CovMatrix(_) => "cov_matrix".into(),
        StanType::CorrMatrix(_) => "corr_matrix".into(),
        StanType::UnitVector(_) => "unit_vector".into(),
    }
}

fn bad_size(name: &str, got: &Val) -> EvalError {
    EvalError::BadParameterDeclaration {
        name: name.to_string(),
        detail: format!(
            "a size expression must evaluate to a plain data integer, got {}",
            got.shape()
        ),
    }
}

fn apply_upper(t: &mut Tape, raw: &Val, upper: &Val) -> (Val, Val) {
    // c = upper - exp(raw); log_jac = raw
    let exp_r = v_exp(t, raw);
    let c = v_sub(t, upper, &exp_r);
    (c, raw.clone())
}

fn apply_lower_upper(t: &mut Tape, raw: &Val, lower: &Val, upper: &Val) -> (Val, Val) {
    // p = inv_logit(raw); c = lower + (upper - lower) * p
    // log_jac = log(upper - lower) + log(p) + log(1 - p)
    let range = v_sub(t, upper, lower);
    let p = v_inv_logit(t, raw);
    let scaled = v_mul(t, &range, &p);
    let c = v_add(t, lower, &scaled);
    let log_range = v_log(t, &range);
    let log_p = v_log(t, &p);
    let one_minus_p = v_sub(t, &Val::Num(1.0), &p);
    let log_1mp = v_log(t, &one_minus_p);
    let log_p_plus_1mp = v_add(t, &log_p, &log_1mp);
    let log_jac = v_add(t, &log_range, &log_p_plus_1mp);
    (c, log_jac)
}
