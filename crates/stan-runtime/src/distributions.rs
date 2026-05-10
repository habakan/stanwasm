//! Distribution log-pdfs / log-pmfs. Mirrors `interp.mbt`.
//!
//! Subset implemented in Phase 3: scalar continuous and discrete distributions
//! that cover the linear_regression / eight_schools / logistic_regression /
//! poisson_regression / gamma_regression / lognormal_regression test models.
//! Multivariate (multi_normal, lkj_*) deferred.
//!
//! Note: every intermediate is bound to a `let` because Rust's borrow checker
//! cannot prove that two nested `v_add(t, ...)` calls don't alias the tape
//! borrow. Equivalent to MoonBit's evaluation order, just more verbose.

use crate::matrix::{mat_mdiv_ltri_low, vec_dot_self};
use crate::ops::{
    v_add, v_div, v_exp, v_lgamma, v_log, v_mul, v_neg, v_sub,
};
use crate::value::Val;
use stan_autodiff::Tape;

const LOG_SQRT_2PI: f64 = 0.918_938_533_204_672_8;
const LN_2: f64 = std::f64::consts::LN_2;
const LN_PI: f64 = 1.144_729_885_849_400_2;

pub fn normal_lpdf(t: &mut Tape, x: &Val, mu: &Val, sigma: &Val) -> Val {
    // -log(sqrt(2π)) - log(σ) - 0.5 ((x-μ)/σ)²
    let diff = v_sub(t, x, mu);
    let z = v_div(t, &diff, sigma);
    let z2 = v_mul(t, &z, &z);
    let half_z2 = v_mul(t, &Val::Num(-0.5), &z2);
    let log_sigma = v_log(t, sigma);
    let log_sqrt2pi_plus_logsigma = v_add(t, &Val::Num(LOG_SQRT_2PI), &log_sigma);
    let prefix = v_neg(t, &log_sqrt2pi_plus_logsigma);
    v_add(t, &prefix, &half_z2)
}

pub fn exponential_lpdf(t: &mut Tape, x: &Val, lambda: &Val) -> Val {
    let log_l = v_log(t, lambda);
    let lx = v_mul(t, lambda, x);
    v_sub(t, &log_l, &lx)
}

pub fn half_normal_lpdf(t: &mut Tape, x: &Val, sigma: &Val) -> Val {
    let n = normal_lpdf(t, x, &Val::Num(0.0), sigma);
    v_add(t, &n, &Val::Num(LN_2))
}

pub fn cauchy_lpdf(t: &mut Tape, x: &Val, mu: &Val, sigma: &Val) -> Val {
    let diff = v_sub(t, x, mu);
    let z = v_div(t, &diff, sigma);
    let z2 = v_mul(t, &z, &z);
    let one_plus_z2 = v_add(t, &Val::Num(1.0), &z2);
    let log_term = v_log(t, &one_plus_z2);
    let log_sigma = v_log(t, sigma);
    let lnpi_plus_logsigma = v_add(t, &Val::Num(LN_PI), &log_sigma);
    let prefix = v_neg(t, &lnpi_plus_logsigma);
    v_sub(t, &prefix, &log_term)
}

pub fn student_t_lpdf(t: &mut Tape, x: &Val, nu: &Val, mu: &Val, sigma: &Val) -> Val {
    let nu1 = v_add(t, nu, &Val::Num(1.0));
    let half_nu1 = v_div(t, &nu1, &Val::Num(2.0));
    let half_nu = v_div(t, nu, &Val::Num(2.0));
    let lg_nu1 = v_lgamma(t, &half_nu1);
    let lg_nu = v_lgamma(t, &half_nu);
    let lg_diff = v_sub(t, &lg_nu1, &lg_nu);
    let log_nu = v_log(t, nu);
    let log_nu_pi = v_add(t, &log_nu, &Val::Num(LN_PI));
    let half_log_nu_pi = v_mul(t, &Val::Num(0.5), &log_nu_pi);
    let log_sigma = v_log(t, sigma);
    let denom = v_add(t, &half_log_nu_pi, &log_sigma);
    let prefix = v_sub(t, &lg_diff, &denom);
    let diff = v_sub(t, x, mu);
    let z = v_div(t, &diff, sigma);
    let z2 = v_mul(t, &z, &z);
    let z2_over_nu = v_div(t, &z2, nu);
    let one_plus = v_add(t, &Val::Num(1.0), &z2_over_nu);
    let log_one_plus = v_log(t, &one_plus);
    let tail = v_mul(t, &half_nu1, &log_one_plus);
    v_sub(t, &prefix, &tail)
}

pub fn lognormal_lpdf(t: &mut Tape, x: &Val, mu: &Val, sigma: &Val) -> Val {
    let log_x = v_log(t, x);
    let n = normal_lpdf(t, &log_x, mu, sigma);
    v_sub(t, &n, &log_x)
}

pub fn gamma_lpdf(t: &mut Tape, x: &Val, alpha: &Val, beta: &Val) -> Val {
    let log_b = v_log(t, beta);
    let a_log_b = v_mul(t, alpha, &log_b);
    let log_x = v_log(t, x);
    let a_minus = v_sub(t, alpha, &Val::Num(1.0));
    let am_logx = v_mul(t, &a_minus, &log_x);
    let term1 = v_add(t, &a_log_b, &am_logx);
    let lg_a = v_lgamma(t, alpha);
    let term1_minus_lga = v_sub(t, &term1, &lg_a);
    let bx = v_mul(t, beta, x);
    let neg_bx = v_neg(t, &bx);
    v_add(t, &term1_minus_lga, &neg_bx)
}

pub fn beta_lpdf(t: &mut Tape, x: &Val, a: &Val, b: &Val) -> Val {
    let a_plus_b = v_add(t, a, b);
    let lg_ab = v_lgamma(t, &a_plus_b);
    let lg_a = v_lgamma(t, a);
    let lg_b = v_lgamma(t, b);
    let lg_ab_minus_a = v_sub(t, &lg_ab, &lg_a);
    let prefix = v_sub(t, &lg_ab_minus_a, &lg_b);
    let log_x = v_log(t, x);
    let am1 = v_sub(t, a, &Val::Num(1.0));
    let am1_logx = v_mul(t, &am1, &log_x);
    let one_minus_x = v_sub(t, &Val::Num(1.0), x);
    let log_1mx = v_log(t, &one_minus_x);
    let bm1 = v_sub(t, b, &Val::Num(1.0));
    let bm1_log1mx = v_mul(t, &bm1, &log_1mx);
    let prefix_plus_a = v_add(t, &prefix, &am1_logx);
    v_add(t, &prefix_plus_a, &bm1_log1mx)
}

pub fn bernoulli_lpmf(t: &mut Tape, y: &Val, theta: &Val) -> Val {
    let log_th = v_log(t, theta);
    let y_logth = v_mul(t, y, &log_th);
    let one_minus_y = v_sub(t, &Val::Num(1.0), y);
    let one_minus_th = v_sub(t, &Val::Num(1.0), theta);
    let log_1mth = v_log(t, &one_minus_th);
    let r = v_mul(t, &one_minus_y, &log_1mth);
    v_add(t, &y_logth, &r)
}

pub fn bernoulli_logit_lpmf(t: &mut Tape, y: &Val, alpha: &Val) -> Val {
    let y_alpha = v_mul(t, y, alpha);
    let exp_a = v_exp(t, alpha);
    let one_plus = v_add(t, &Val::Num(1.0), &exp_a);
    let log_term = v_log(t, &one_plus);
    v_sub(t, &y_alpha, &log_term)
}

pub fn poisson_lpmf(t: &mut Tape, y: &Val, lambda: &Val) -> Val {
    let log_l = v_log(t, lambda);
    let y_log_l = v_mul(t, y, &log_l);
    let inner = v_sub(t, &y_log_l, lambda);
    let yp1 = v_add(t, y, &Val::Num(1.0));
    let lg = v_lgamma(t, &yp1);
    v_sub(t, &inner, &lg)
}

pub fn neg_binomial_2_lpmf(t: &mut Tape, y: &Val, mu: &Val, phi: &Val) -> Val {
    let yp_phi = v_add(t, y, phi);
    let lg_yphi = v_lgamma(t, &yp_phi);
    let lg_phi = v_lgamma(t, phi);
    let yp1 = v_add(t, y, &Val::Num(1.0));
    let lg_yp1 = v_lgamma(t, &yp1);
    let lg_diff = v_sub(t, &lg_yphi, &lg_phi);
    let combo = v_sub(t, &lg_diff, &lg_yp1);
    let phi_mu = v_add(t, phi, mu);
    let log_phi_mu = v_log(t, &phi_mu);
    let log_phi = v_log(t, phi);
    let log_mu = v_log(t, mu);
    let phi_diff = v_sub(t, &log_phi, &log_phi_mu);
    let phi_term = v_mul(t, phi, &phi_diff);
    let mu_diff = v_sub(t, &log_mu, &log_phi_mu);
    let y_term = v_mul(t, y, &mu_diff);
    let combo_plus_phi = v_add(t, &combo, &phi_term);
    v_add(t, &combo_plus_phi, &y_term)
}

/// `multi_normal_cholesky_lpdf(y | μ, L)` — y, μ are vectors, L is the
/// Cholesky factor of the covariance (lower-triangular K×K).
/// log p = -K/2 · log(2π) − Σ log Lᵢᵢ − 0.5 · ||L⁻¹(y − μ)||²
pub fn multi_normal_cholesky_lpdf(
    t: &mut Tape,
    y: &[Val],
    mu: &[Val],
    l_rows: &[Val],
) -> Val {
    let kk = y.len();
    let mut diff = Vec::with_capacity(kk);
    for i in 0..kk {
        diff.push(v_sub(t, &y[i], &mu[i]));
    }
    let r = mat_mdiv_ltri_low(t, l_rows, &diff);
    let mut sum_log_diag = Val::Num(0.0);
    for (i, row_v) in l_rows.iter().enumerate() {
        if let Val::Vec(row) = row_v {
            if i < row.len() {
                let lr = v_log(t, &row[i]);
                sum_log_diag = v_add(t, &sum_log_diag, &lr);
            }
        }
    }
    let ds = vec_dot_self(t, &r);
    // -K/2 * log(2π) - sum_log_diag - 0.5 * ds
    let prefix = Val::Num(-(kk as f64) * LOG_SQRT_2PI);
    let half_ds = v_mul(t, &Val::Num(0.5), &ds);
    let prefix_minus_diag = v_sub(t, &prefix, &sum_log_diag);
    v_sub(t, &prefix_minus_diag, &half_ds)
}

/// `lkj_corr_cholesky_lpdf(L | η)` — L is the Cholesky factor of a K×K
/// correlation matrix.
/// log p = (2η − 2) · Σ_{k=0..K-1} (K − 1 − k) · log Lₖₖ
pub fn lkj_corr_cholesky_lpdf(t: &mut Tape, l_rows: &[Val], eta: &Val) -> Val {
    let kk = l_rows.len();
    let two_eta = v_mul(t, &Val::Num(2.0), eta);
    let two_eta_minus_2 = v_sub(t, &two_eta, &Val::Num(2.0));
    let mut weighted_sum = Val::Num(0.0);
    for (k, row_v) in l_rows.iter().enumerate() {
        let wt = (kk - 1 - k) as f64;
        if wt > 0.0 {
            if let Val::Vec(row) = row_v {
                if k < row.len() {
                    let lr = v_log(t, &row[k]);
                    let term = v_mul(t, &Val::Num(wt), &lr);
                    weighted_sum = v_add(t, &weighted_sum, &term);
                }
            }
        }
    }
    v_mul(t, &two_eta_minus_2, &weighted_sum)
}

/// Distributions whose first argument is a *whole* vector / matrix (not a
/// scalar element). For these, sampling a `Val::Vec` does NOT broadcast the
/// scalar lpdf over elements — the entire structure is the observation.
fn is_multivariate(name: &str) -> bool {
    matches!(
        name,
        "multi_normal_cholesky" | "lkj_corr_cholesky" | "dirichlet" | "multinomial"
    )
}

/// Dispatch: dist_name → lpdf/lpmf computation. Returns None if the
/// distribution is not yet supported in the Rust port.
pub fn eval_dist(t: &mut Tape, name: &str, x: &Val, args: &[Val]) -> Option<Val> {
    match name {
        "normal" => Some(normal_lpdf(t, x, &args[0], &args[1])),
        "std_normal" => Some(normal_lpdf(t, x, &Val::Num(0.0), &Val::Num(1.0))),
        "exponential" => Some(exponential_lpdf(t, x, &args[0])),
        "half_normal" => Some(half_normal_lpdf(t, x, &args[0])),
        "cauchy" => Some(cauchy_lpdf(t, x, &args[0], &args[1])),
        "student_t" => Some(student_t_lpdf(t, x, &args[0], &args[1], &args[2])),
        "lognormal" => Some(lognormal_lpdf(t, x, &args[0], &args[1])),
        "gamma" => Some(gamma_lpdf(t, x, &args[0], &args[1])),
        "beta" => Some(beta_lpdf(t, x, &args[0], &args[1])),
        "bernoulli" => Some(bernoulli_lpmf(t, x, &args[0])),
        "bernoulli_logit" => Some(bernoulli_logit_lpmf(t, x, &args[0])),
        "poisson" => Some(poisson_lpmf(t, x, &args[0])),
        "neg_binomial_2" => Some(neg_binomial_2_lpmf(t, x, &args[0], &args[1])),
        "multi_normal_cholesky" => match (x, &args[0], &args[1]) {
            (Val::Vec(y), Val::Vec(mu), Val::Vec(l_rows)) => {
                Some(multi_normal_cholesky_lpdf(t, y, mu, l_rows))
            }
            _ => Some(Val::Num(0.0)),
        },
        "lkj_corr_cholesky" => match x {
            Val::Vec(l_rows) => Some(lkj_corr_cholesky_lpdf(t, l_rows, &args[0])),
            _ => Some(Val::Num(0.0)),
        },
        _ => None,
    }
}

/// Sample statement (`y ~ dist(...)`) on a vector observation. Behaviour:
/// - For multivariate distributions, the whole vector / matrix is the
///   observation and we delegate to `eval_dist` once with `x = Val::Vec(xs)`.
/// - For scalar distributions, sums the scalar lpdf over each element with
///   element-wise argument broadcast.
pub fn eval_sample_vec(t: &mut Tape, name: &str, xs: &[Val], args: &[Val]) -> Option<Val> {
    if is_multivariate(name) {
        return eval_dist(t, name, &Val::Vec(xs.to_vec()), args);
    }
    let mut acc = Val::Num(0.0);
    let mut elem_args: Vec<Val> = Vec::with_capacity(args.len());
    for (i, x) in xs.iter().enumerate() {
        elem_args.clear();
        for a in args {
            elem_args.push(broadcast_elem(a, i));
        }
        let term = eval_dist(t, name, x, &elem_args)?;
        acc = v_add(t, &acc, &term);
    }
    Some(acc)
}

fn broadcast_elem(v: &Val, i: usize) -> Val {
    match v {
        Val::Vec(xs) => xs[i].clone(),
        other => other.clone(),
    }
}
