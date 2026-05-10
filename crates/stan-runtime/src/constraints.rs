//! Constraint transforms with Jacobian. Mirrors `interp.mbt::constrain`.
//!
//! Phase 3 covers the scalar real cases (none, lower, upper, lower_upper)
//! plus vectors with element-wise constraints. Multivariate transforms
//! (simplex, ordered, cholesky_factor_*) are not yet ported.

use crate::env::Env;
use crate::eval::eval_plain;
use crate::ops::{v_add, v_exp, v_inv_logit, v_log, v_mul, v_sqrt, v_sub, v_tanh};
use crate::value::Val;
use stan_ast::{Constraint, StanType};
use stan_autodiff::Tape;

pub fn param_dims(typ: &StanType, env: &Env) -> usize {
    match typ {
        StanType::Real(_) => 1,
        StanType::Int => 0,
        StanType::Vector(size, _) => eval_plain_int(size, env),
        StanType::Matrix(r, c) => eval_plain_int(r, env) * eval_plain_int(c, env),
        StanType::Simplex(k) => eval_plain_int(k, env).saturating_sub(1),
        StanType::Ordered(k) => eval_plain_int(k, env),
        StanType::Array(size, elem) => eval_plain_int(size, env) * param_dims(elem, env),
        StanType::CholeskyFactorCorr(k) => {
            let kk = eval_plain_int(k, env);
            kk * (kk - 1) / 2
        }
        StanType::CholeskyFactorCov(k)
        | StanType::CovMatrix(k) => {
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

fn eval_plain_int(expr: &stan_ast::Expr, env: &Env) -> usize {
    let mut tape = Tape::new();
    let v = eval_plain(&mut tape, expr, env);
    match v {
        Val::Num(x) => x as usize,
        _ => 0,
    }
}

/// Apply the constraint transform to a slice of unconstrained tape leaves.
/// Returns `(constrained_value, log_jacobian)`.
pub fn constrain(
    t: &mut Tape,
    typ: &StanType,
    raw: &[Val],
    env: &Env,
) -> (Val, Val) {
    match typ {
        StanType::Real(Constraint::None) => (raw[0].clone(), Val::Num(0.0)),
        StanType::Real(Constraint::Lower(lo_e)) => {
            let lo = eval_plain(t, lo_e, env);
            let exp_r = v_exp(t, &raw[0]);
            let c = v_add(t, &lo, &exp_r);
            // log|dc/draw| = raw
            (c, raw[0].clone())
        }
        StanType::Real(Constraint::Upper(hi_e)) => {
            let hi = eval_plain(t, hi_e, env);
            let (c, j) = apply_upper(t, &raw[0], &hi);
            (c, j)
        }
        StanType::Real(Constraint::LowerUpper(lo_e, hi_e)) => {
            let lo = eval_plain(t, lo_e, env);
            let hi = eval_plain(t, hi_e, env);
            apply_lower_upper(t, &raw[0], &lo, &hi)
        }
        StanType::Vector(_, Constraint::None) => {
            (Val::Vec(raw.to_vec()), Val::Num(0.0))
        }
        StanType::Vector(_, Constraint::Lower(lo_e)) => {
            let lo = eval_plain(t, lo_e, env);
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
            let hi = eval_plain(t, hi_e, env);
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
            let lo = eval_plain(t, lo_e, env);
            let hi = eval_plain(t, hi_e, env);
            let mut jac = Val::Num(0.0);
            let mut cs = Vec::with_capacity(raw.len());
            for r in raw {
                let (c, j) = apply_lower_upper(t, r, &lo, &hi);
                jac = v_add(t, &jac, &j);
                cs.push(c);
            }
            (Val::Vec(cs), jac)
        }
        // cholesky_factor_corr[K]:
        //   K*(K-1)/2 unconstrained → K×K lower-triangular L (Cholesky of corr matrix).
        //   Spherical parameterization: row i, col j < i:
        //     z = tanh(raw[idx])
        //     L[i][j] = z * sqrt(rem)
        //     log_jac += 0.5*log(rem) + log(1 - z²)
        //     rem *= (1 - z²)
        //   L[i][i] = sqrt(rem). Row 0: L[0][0] = 1.
        StanType::CholeskyFactorCorr(k_e) => {
            let kk = eval_plain(t, k_e, env);
            let kk = match kk {
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
        // Phase 3 fallback: pass-through. Higher-order constraints port later.
        _ => (Val::Vec(raw.to_vec()), Val::Num(0.0)),
    }
}

fn apply_upper(t: &mut Tape, raw: &Val, upper: &Val) -> (Val, Val) {
    // c = upper - exp(raw); log_jac = raw
    let exp_r = v_exp(t, raw);
    let c = v_sub(t, upper, &exp_r);
    (c, raw.clone())
}

fn apply_lower_upper(t: &mut Tape, raw: &Val, lower: &Val, upper: &Val) -> (Val, Val) {
    // p = inv_logit(raw)
    // c = lower + (upper - lower) * p
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
