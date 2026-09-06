//! Distribution log-pdfs / log-pmfs.
//!
//! Covered: the scalar continuous and discrete distributions listed in the
//! README, `categorical`, plus the multivariate `multi_normal_cholesky`,
//! `multi_normal`, `lkj_corr_cholesky`, `dirichlet` and `multinomial`.
//! `arity()` is the authoritative list — anything absent from it is an
//! `UnknownDistribution` error, never a zero contribution to the log density.
//!
//! Note: every intermediate is bound to a `let` because Rust's borrow checker
//! cannot prove that two nested `v_add(t, ...)` calls don't alias the tape
//! borrow.

use crate::error::EvalError;
use crate::matrix::{cholesky_decompose, mat_mdiv_ltri_low, mat_vec_mul, vec_dot_self};
use crate::ops::{v_abs, v_add, v_div, v_exp, v_lgamma, v_log, v_mul, v_neg, v_sub, v_sum};
use crate::value::Val;
use stanwasm_autodiff::Tape;

type Result<T> = std::result::Result<T, EvalError>;

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

/// `-z - log s - 2 log(1 + e^-z)`, with `z = (x - mu) / s`.
pub fn logistic_lpdf(t: &mut Tape, x: &Val, mu: &Val, s: &Val) -> Val {
    let diff = v_sub(t, x, mu);
    let z = v_div(t, &diff, s);
    let neg_z = v_neg(t, &z);
    let tail = log1p_exp(t, &neg_z);
    let two_tail = v_mul(t, &Val::Num(2.0), &tail);
    let log_s = v_log(t, s);
    let prefix = v_sub(t, &neg_z, &log_s);
    v_sub(t, &prefix, &two_tail)
}

/// Weibull: `log(a) - log(s) + (a - 1) log(x/s) - (x/s)^a`. The power is taken
/// through `exp(a log(x/s))`, since the shape is usually a parameter.
pub fn weibull_lpdf(t: &mut Tape, x: &Val, alpha: &Val, sigma: &Val) -> Val {
    let z = v_div(t, x, sigma);
    let log_z = v_log(t, &z);
    let a_minus_1 = v_sub(t, alpha, &Val::Num(1.0));
    let shaped = v_mul(t, &a_minus_1, &log_z);
    let scaled = v_mul(t, alpha, &log_z);
    let z_pow_a = v_exp(t, &scaled);
    let log_a = v_log(t, alpha);
    let log_s = v_log(t, sigma);
    let prefix = v_sub(t, &log_a, &log_s);
    let with_shape = v_add(t, &prefix, &shaped);
    v_sub(t, &with_shape, &z_pow_a)
}

/// Laplace: `-log(2 s) - |x - mu| / s`.
pub fn double_exponential_lpdf(t: &mut Tape, x: &Val, mu: &Val, s: &Val) -> Val {
    let diff = v_sub(t, x, mu);
    let dist = v_abs(t, &diff);
    let scaled = v_div(t, &dist, s);
    let log_s = v_log(t, s);
    let norm = v_add(t, &Val::Num(LN_2), &log_s);
    let neg_norm = v_neg(t, &norm);
    v_sub(t, &neg_norm, &scaled)
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

/// `log theta` or `log(1 - theta)` — the term `y` selects, rather than the sum
/// of both weighted by `y` and `1 - y`. At theta 0 or 1 the unselected term is
/// `0 * log 0`, which is NaN where the density is perfectly finite. `y` is an
/// observation, so choosing on it is not a branch on a parameter.
pub fn bernoulli_lpmf(t: &mut Tape, y: &Val, theta: &Val) -> Val {
    if y.to_f64(t).unwrap_or(f64::NAN) != 0.0 {
        v_log(t, theta)
    } else {
        let one_minus_th = v_sub(t, &Val::Num(1.0), theta);
        v_log(t, &one_minus_th)
    }
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

/// `log C(n, y)`. Constant in the parameters, but `~` here keeps its
/// normalising terms, so it is recorded like anything else.
fn log_binomial_coeff(t: &mut Tape, y: &Val, n: &Val) -> Val {
    let np1 = v_add(t, n, &Val::Num(1.0));
    let yp1 = v_add(t, y, &Val::Num(1.0));
    let n_y_p1 = v_sub(t, &np1, y);
    let all = v_lgamma(t, &np1);
    let chosen = v_lgamma(t, &yp1);
    let rest = v_lgamma(t, &n_y_p1);
    let d = v_sub(t, &all, &chosen);
    v_sub(t, &d, &rest)
}

/// `log(1 + exp(x))`, which the logit-scale densities need.
fn log1p_exp(t: &mut Tape, x: &Val) -> Val {
    let e = v_exp(t, x);
    let s = v_add(t, &Val::Num(1.0), &e);
    v_log(t, &s)
}

/// A zero count drops its term rather than weighting it: `0 * log 0` is NaN at
/// theta 0 or 1, where the density itself is finite. Both counts are
/// observations, so skipping on them is not a branch on a parameter.
pub fn binomial_lpmf(t: &mut Tape, y: &Val, n: &Val, theta: &Val) -> Val {
    let mut acc = log_binomial_coeff(t, y, n);
    let n_y = v_sub(t, n, y);
    if y.to_f64(t).unwrap_or(f64::NAN) != 0.0 {
        let log_theta = v_log(t, theta);
        let hit = v_mul(t, y, &log_theta);
        acc = v_add(t, &acc, &hit);
    }
    if n_y.to_f64(t).unwrap_or(f64::NAN) != 0.0 {
        let one_minus = v_sub(t, &Val::Num(1.0), theta);
        let log_1m = v_log(t, &one_minus);
        let miss = v_mul(t, &n_y, &log_1m);
        acc = v_add(t, &acc, &miss);
    }
    acc
}

/// `log C + y log inv_logit(a) + (n - y) log(1 - inv_logit(a))`, written with
/// `log1p_exp` so neither logarithm is taken of something near zero.
pub fn binomial_logit_lpmf(t: &mut Tape, y: &Val, n: &Val, alpha: &Val) -> Val {
    let coeff = log_binomial_coeff(t, y, n);
    let neg_alpha = v_neg(t, alpha);
    let hit = log1p_exp(t, &neg_alpha);
    let miss = log1p_exp(t, alpha);
    let n_y = v_sub(t, n, y);
    let a = v_mul(t, y, &hit);
    let b = v_mul(t, &n_y, &miss);
    let s = v_sub(t, &coeff, &a);
    v_sub(t, &s, &b)
}

pub fn poisson_log_lpmf(t: &mut Tape, y: &Val, alpha: &Val) -> Val {
    let y_alpha = v_mul(t, y, alpha);
    let rate = v_exp(t, alpha);
    let inner = v_sub(t, &y_alpha, &rate);
    let yp1 = v_add(t, y, &Val::Num(1.0));
    let lg = v_lgamma(t, &yp1);
    v_sub(t, &inner, &lg)
}

pub fn inv_gamma_lpdf(t: &mut Tape, y: &Val, alpha: &Val, beta: &Val) -> Val {
    let log_beta = v_log(t, beta);
    let a_log_b = v_mul(t, alpha, &log_beta);
    let lg = v_lgamma(t, alpha);
    let ap1 = v_add(t, alpha, &Val::Num(1.0));
    let log_y = v_log(t, y);
    let tail = v_mul(t, &ap1, &log_y);
    let over = v_div(t, beta, y);
    let s = v_sub(t, &a_log_b, &lg);
    let s = v_sub(t, &s, &tail);
    v_sub(t, &s, &over)
}

/// `-log(b - a)`.
///
/// Outside `[a, b]` Stan's is `-inf`, and this cannot be: the support test is a
/// branch on the variate, and a graph traced once has no form for one. A model
/// that writes `y ~ uniform(a, b)` declares `real<lower=a, upper=b> y`, and it
/// is that transform, not this, which keeps the variate inside.
pub fn uniform_lpdf(t: &mut Tape, _y: &Val, a: &Val, b: &Val) -> Val {
    let width = v_sub(t, b, a);
    let log_width = v_log(t, &width);
    v_neg(t, &log_width)
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

/// `multi_normal_cholesky_lpdf(y | μ, L)`, L lower-triangular K×K:
/// log p = -K/2·log(2π) − Σ log Lᵢᵢ − 0.5·||L⁻¹(y − μ)||²
pub fn multi_normal_cholesky_lpdf(t: &mut Tape, y: &[Val], mu: &[Val], l_rows: &[Val]) -> Val {
    let kk = y.len();
    let mut diff = Vec::with_capacity(kk);
    for i in 0..kk {
        diff.push(v_sub(t, &y[i], &mu[i]));
    }
    let r = mat_mdiv_ltri_low(t, l_rows, &diff);
    let mut sum_log_diag = Val::Num(0.0);
    for (i, row_v) in l_rows.iter().enumerate() {
        if let Some(row) = row_v.elems() {
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

/// `log Γ_K(a)`, the multivariate gamma: `K(K-1)/4 · log π + Σ_j log Γ(a + (1-j)/2)`.
fn log_multigamma(t: &mut Tape, a: &Val, k: usize) -> Val {
    let mut acc = Val::Num((k * (k - 1)) as f64 / 4.0 * std::f64::consts::PI.ln());
    for j in 1..=k {
        let shifted = v_add(t, a, &Val::Num((1.0 - j as f64) / 2.0));
        let lg = v_lgamma(t, &shifted);
        acc = v_add(t, &acc, &lg);
    }
    acc
}

/// `log |M|` for a positive-definite `M` given its Cholesky factor: twice the
/// sum of the factor's log diagonal.
fn log_det_from_chol(t: &mut Tape, l_rows: &[Val]) -> Val {
    let mut acc = Val::Num(0.0);
    for (i, row) in l_rows.iter().enumerate() {
        if let Some(cells) = row.elems() {
            if let Some(d) = cells.get(i) {
                let ld = v_log(t, d);
                acc = v_add(t, &acc, &ld);
            }
        }
    }
    v_mul(t, &Val::Num(2.0), &acc)
}

/// `tr(A⁻¹ B)` where `la` is the Cholesky factor of A and `lb` that of B.
/// Writing B as `Lb Lbᵀ` turns the trace into `‖La⁻¹ Lb‖²`, so it is a
/// triangular solve per column rather than an inverse.
fn trace_solve(t: &mut Tape, la_rows: &[Val], lb_rows: &[Val], k: usize) -> Val {
    let mut acc = Val::Num(0.0);
    for j in 0..k {
        let col: Vec<Val> = (0..k)
            .map(|i| {
                lb_rows[i]
                    .elems()
                    .and_then(|r| r.get(j))
                    .cloned()
                    .unwrap_or(Val::Num(0.0))
            })
            .collect();
        let x = mat_mdiv_ltri_low(t, la_rows, &col);
        let s = vec_dot_self(t, &x);
        acc = v_add(t, &acc, &s);
    }
    acc
}

/// `wishart_lpdf(W | nu, S)`, both K×K covariance matrices:
/// `(nu−K−1)/2·log|W| − tr(S⁻¹W)/2 − nu·K/2·log2 − nu/2·log|S| − log Γ_K(nu/2)`.
pub fn wishart_lpdf(t: &mut Tape, w_rows: &[Val], nu: &Val, s_rows: &[Val]) -> Val {
    let k = w_rows.len();
    let lw = cholesky_decompose(t, w_rows);
    let ls = cholesky_decompose(t, s_rows);
    let log_det_w = log_det_from_chol(t, &lw);
    let log_det_s = log_det_from_chol(t, &ls);
    let tr = trace_solve(t, &ls, &lw, k);

    let half_nu = v_mul(t, &Val::Num(0.5), nu);
    let a = v_sub(t, nu, &Val::Num((k + 1) as f64));
    let a = v_mul(t, &Val::Num(0.5), &a);
    let term_w = v_mul(t, &a, &log_det_w);
    let term_tr = v_mul(t, &Val::Num(-0.5), &tr);
    let term_2 = v_mul(t, &half_nu, &Val::Num(-(k as f64) * std::f64::consts::LN_2));
    let term_s = v_mul(t, &half_nu, &log_det_s);
    let lg = log_multigamma(t, &half_nu, k);

    let acc = v_add(t, &term_w, &term_tr);
    let acc = v_add(t, &acc, &term_2);
    let acc = v_sub(t, &acc, &term_s);
    v_sub(t, &acc, &lg)
}

/// `inv_wishart_lpdf(W | nu, S)`:
/// `nu/2·log|S| − (nu+K+1)/2·log|W| − tr(S W⁻¹)/2 − nu·K/2·log2 − log Γ_K(nu/2)`.
pub fn inv_wishart_lpdf(t: &mut Tape, w_rows: &[Val], nu: &Val, s_rows: &[Val]) -> Val {
    let k = w_rows.len();
    let lw = cholesky_decompose(t, w_rows);
    let ls = cholesky_decompose(t, s_rows);
    let log_det_w = log_det_from_chol(t, &lw);
    let log_det_s = log_det_from_chol(t, &ls);
    // tr(S W⁻¹) is the same shape with the roles of the two factors swapped.
    let tr = trace_solve(t, &lw, &ls, k);

    let half_nu = v_mul(t, &Val::Num(0.5), nu);
    let b = v_add(t, nu, &Val::Num((k + 1) as f64));
    let b = v_mul(t, &Val::Num(-0.5), &b);
    let term_w = v_mul(t, &b, &log_det_w);
    let term_s = v_mul(t, &half_nu, &log_det_s);
    let term_tr = v_mul(t, &Val::Num(-0.5), &tr);
    let term_2 = v_mul(t, &half_nu, &Val::Num(-(k as f64) * std::f64::consts::LN_2));
    let lg = log_multigamma(t, &half_nu, k);

    let acc = v_add(t, &term_s, &term_w);
    let acc = v_add(t, &acc, &term_tr);
    let acc = v_add(t, &acc, &term_2);
    v_sub(t, &acc, &lg)
}

/// `multinomial_lpmf(y | θ)`, y an integer count array, θ a simplex of length K:
/// log p = lgamma(N+1) − Σ lgamma(yᵢ+1) + Σ yᵢ log θᵢ, N = Σ yᵢ.
pub fn multinomial_lpmf(t: &mut Tape, y: &[Val], theta: &[Val]) -> Val {
    let mut sum_y = Val::Num(0.0);
    let mut sum_lg_yp1 = Val::Num(0.0);
    for yi in y {
        sum_y = v_add(t, &sum_y, yi);
        let yp1 = v_add(t, yi, &Val::Num(1.0));
        let lg = v_lgamma(t, &yp1);
        sum_lg_yp1 = v_add(t, &sum_lg_yp1, &lg);
    }
    let np1 = v_add(t, &sum_y, &Val::Num(1.0));
    let lg_np1 = v_lgamma(t, &np1);
    let mut lp = v_sub(t, &lg_np1, &sum_lg_yp1);
    let k = y.len().min(theta.len());
    for i in 0..k {
        let log_th = v_log(t, &theta[i]);
        let term = v_mul(t, &y[i], &log_th);
        lp = v_add(t, &lp, &term);
    }
    lp
}

/// `categorical_lpmf(y | θ)`, y a 1-indexed label: log p = log θ_y. Vectorized, θ is
/// shared across every element rather than per-observation — see `eval_sample_vec`.
pub fn categorical_lpmf(t: &mut Tape, y: &Val, theta: &[Val]) -> Result<Val> {
    let label = y.to_i32(t)?;
    let zero_based = label - 1;
    if zero_based < 0 || zero_based as usize >= theta.len() {
        return Err(EvalError::IndexOutOfBounds {
            index: label,
            len: theta.len(),
        });
    }
    Ok(v_log(t, &theta[zero_based as usize]))
}

/// `categorical_logit_lpmf(y | β)` — `β_y - log ∑ exp β`.
pub fn categorical_logit_lpmf(t: &mut Tape, y: &Val, beta: &[Val]) -> Result<Val> {
    let label = y.to_i32(t)?;
    let zero_based = label - 1;
    if zero_based < 0 || zero_based as usize >= beta.len() {
        return Err(EvalError::IndexOutOfBounds {
            index: label,
            len: beta.len(),
        });
    }
    let shift = beta.iter().try_fold(f64::NEG_INFINITY, |m, b| {
        Ok::<f64, EvalError>(m.max(b.to_f64(t)?))
    })?;
    let mut total = Val::Num(0.0);
    for b in beta {
        let d = v_sub(t, b, &Val::Num(shift));
        let e = v_exp(t, &d);
        total = v_add(t, &total, &e);
    }
    let log_total = v_log(t, &total);
    let norm = v_add(t, &log_total, &Val::Num(shift));
    Ok(v_sub(t, &beta[zero_based as usize], &norm))
}

/// `dirichlet_lpdf(θ | α)` — both vectors of length K.
/// log p = lgamma(∑αᵢ) − ∑ lgamma(αᵢ) + ∑ (αᵢ − 1) log θᵢ
pub fn dirichlet_lpdf(t: &mut Tape, theta: &[Val], alpha: &[Val]) -> Val {
    let mut sum_alpha = Val::Num(0.0);
    for a in alpha {
        sum_alpha = v_add(t, &sum_alpha, a);
    }
    let mut lp = v_lgamma(t, &sum_alpha);
    let k = theta.len().min(alpha.len());
    for i in 0..k {
        let lg = v_lgamma(t, &alpha[i]);
        lp = v_sub(t, &lp, &lg);
        let am1 = v_sub(t, &alpha[i], &Val::Num(1.0));
        let log_th = v_log(t, &theta[i]);
        let term = v_mul(t, &am1, &log_th);
        lp = v_add(t, &lp, &term);
    }
    lp
}

/// `lkj_corr_cholesky_lpdf(L | η)`: `Σₖ [(K−1−k) + (2η−2)]·log Lₖₖ + log c_K(η)`,
/// exponents summed on a shared base (a `(2η−2)` factor over the sum is 0 at K=2).
///
/// Carries the same constant as the matrix form, for the same reason: it is a
/// function of `η`, so leaving it out is only safe while `η` is data.
pub fn lkj_corr_cholesky_lpdf(t: &mut Tape, l_rows: &[Val], eta: &Val) -> Val {
    let kk = l_rows.len();
    let two_eta = v_mul(t, &Val::Num(2.0), eta);
    let two_eta_minus_2 = v_sub(t, &two_eta, &Val::Num(2.0));
    let mut lp = Val::Num(0.0);
    for (k, row_v) in l_rows.iter().enumerate() {
        let base_wt = (kk - 1 - k) as f64;
        if let Some(row) = row_v.elems() {
            if k < row.len() {
                let weight = v_add(t, &Val::Num(base_wt), &two_eta_minus_2);
                let lr = v_log(t, &row[k]);
                let term = v_mul(t, &weight, &lr);
                lp = v_add(t, &lp, &term);
            }
        }
    }
    let c = lkj_log_constant(t, eta, kk);
    v_add(t, &lp, &c)
}

/// LKJ's normalising constant over K×K correlation matrices:
/// `(K−1)·lgamma(η + (K−1)/2) − Σ_{k=1..K−1} [ ½k·log π + lgamma(η + (K−1−k)/2) ]`.
///
/// Kept rather than dropped, because it depends on `η`. Dropping it is exact
/// while `η` is data and silently wrong the moment it is a parameter — the
/// posterior for `η` then comes out of the wrong density with no sign that
/// anything happened. Checked against a reference implementation at K=2 and
/// K=3, where solving for the constant from its log density agrees exactly.
fn lkj_log_constant(t: &mut Tape, eta: &Val, k: usize) -> Val {
    let km1 = (k - 1) as f64;
    let shifted = v_add(t, eta, &Val::Num(km1 / 2.0));
    let lg = v_lgamma(t, &shifted);
    let mut acc = v_mul(t, &Val::Num(km1), &lg);
    for j in 1..k {
        let e = v_add(t, eta, &Val::Num((km1 - j as f64) / 2.0));
        let lg = v_lgamma(t, &e);
        let term = v_add(t, &Val::Num(0.5 * j as f64 * std::f64::consts::PI.ln()), &lg);
        acc = v_sub(t, &acc, &term);
    }
    acc
}

/// `lkj_corr_lpdf(R | η)`: `(η−1)·log|R| + log c_K(η)`, the same density as
/// `lkj_corr_cholesky` seen on the matrix rather than on its factor.
pub fn lkj_corr_lpdf(t: &mut Tape, r_rows: &[Val], eta: &Val) -> Val {
    let l = cholesky_decompose(t, r_rows);
    let log_det = log_det_from_chol(t, &l);
    let eta_minus_1 = v_sub(t, eta, &Val::Num(1.0));
    let kernel = v_mul(t, &eta_minus_1, &log_det);
    let c = lkj_log_constant(t, eta, r_rows.len());
    v_add(t, &kernel, &c)
}

/// Distributions whose first argument is a whole vector / matrix. Sampling a
/// `Val::Vec` does not broadcast the scalar lpdf — the structure is the observation.
fn is_multivariate(name: &str) -> bool {
    matches!(
        name,
        "bernoulli_logit_glm"
            | "normal_id_glm"
            | "multi_normal_cholesky"
            | "multi_normal"
            | "wishart"
            | "inv_wishart"
            | "lkj_corr_cholesky"
            | "lkj_corr"
            | "dirichlet"
            | "multinomial"
    )
}

/// Argument count after the variate, checked before dispatch so `a ~ normal(0);`
/// is a clean error rather than an out-of-bounds panic.
fn arity(name: &str) -> Option<usize> {
    Some(match name {
        "std_normal" => 0,
        "exponential" | "half_normal" | "bernoulli" | "bernoulli_logit" | "poisson"
        | "poisson_log" | "dirichlet" | "lkj_corr_cholesky" | "lkj_corr" | "multinomial"
        | "categorical"
        | "categorical_logit" => 1,
        "normal"
        | "cauchy"
        | "lognormal"
        | "gamma"
        | "beta"
        | "neg_binomial_2"
        | "binomial"
        | "binomial_logit"
        | "inv_gamma"
        | "uniform"
        | "multi_normal_cholesky"
        | "logistic"
        | "weibull"
        | "double_exponential"
        | "multi_normal"
        | "wishart"
        | "inv_wishart" => 2,
        "student_t" | "bernoulli_logit_glm" => 3,
        "normal_id_glm" => 4,
        _ => return None,
    })
}

/// The elements of a container argument, or the type error naming what it wanted.
fn val_elems<'a>(name: &str, v: &'a Val) -> Result<&'a [Val]> {
    v.elems()
        .ok_or_else(|| wrong_type(name, "a coefficient vector beta", v))
}

fn wrong_type(name: &str, expected: &str, got: &Val) -> EvalError {
    EvalError::DistributionArgType {
        name: name.to_string(),
        expected: expected.to_string(),
        got: got.shape().to_string(),
    }
}

/// Dispatch: dist_name → lpdf/lpmf computation.
pub fn eval_dist(t: &mut Tape, name: &str, x: &Val, args: &[Val]) -> Result<Val> {
    let expected = arity(name).ok_or_else(|| EvalError::UnknownDistribution(name.to_string()))?;
    if args.len() != expected {
        return Err(EvalError::DistributionArity {
            name: name.to_string(),
            expected,
            got: args.len(),
        });
    }
    Ok(match name {
        "normal" => normal_lpdf(t, x, &args[0], &args[1]),
        "std_normal" => normal_lpdf(t, x, &Val::Num(0.0), &Val::Num(1.0)),
        "exponential" => exponential_lpdf(t, x, &args[0]),
        "half_normal" => half_normal_lpdf(t, x, &args[0]),
        "cauchy" => cauchy_lpdf(t, x, &args[0], &args[1]),
        "student_t" => student_t_lpdf(t, x, &args[0], &args[1], &args[2]),
        "lognormal" => lognormal_lpdf(t, x, &args[0], &args[1]),
        "logistic" => logistic_lpdf(t, x, &args[0], &args[1]),
        "weibull" => weibull_lpdf(t, x, &args[0], &args[1]),
        "double_exponential" => double_exponential_lpdf(t, x, &args[0], &args[1]),
        "gamma" => gamma_lpdf(t, x, &args[0], &args[1]),
        "beta" => beta_lpdf(t, x, &args[0], &args[1]),
        "bernoulli" => bernoulli_lpmf(t, x, &args[0]),
        "bernoulli_logit" => bernoulli_logit_lpmf(t, x, &args[0]),
        "poisson" => poisson_lpmf(t, x, &args[0]),
        "poisson_log" => poisson_log_lpmf(t, x, &args[0]),
        "binomial" => binomial_lpmf(t, x, &args[0], &args[1]),
        "binomial_logit" => binomial_logit_lpmf(t, x, &args[0], &args[1]),
        "inv_gamma" => inv_gamma_lpdf(t, x, &args[0], &args[1]),
        "uniform" => uniform_lpdf(t, x, &args[0], &args[1]),
        "neg_binomial_2" => neg_binomial_2_lpmf(t, x, &args[0], &args[1]),
        "categorical" => match &args[0] {
            Val::Vec(theta) => categorical_lpmf(t, x, theta)?,
            _ => return Err(wrong_type(name, "a simplex vector theta", &args[0])),
        },
        // `beta_y - log_sum_exp(beta)`, which is `categorical` on `softmax(beta)`
        // without forming the simplex.
        "categorical_logit" => match &args[0] {
            Val::Vec(beta) => categorical_logit_lpmf(t, x, beta)?,
            _ => return Err(wrong_type(name, "a vector of log odds beta", &args[0])),
        },
        // The GLM forms are `dist(alpha + x * beta, ...)`, but with `x * beta`
        // recorded as one contraction per row rather than a chain per element.
        "bernoulli_logit_glm" | "normal_id_glm" => match (x, &args[0]) {
            (Val::Vec(y), Val::Vec(rows)) => {
                if rows.len() != y.len() {
                    return Err(EvalError::DistributionArgLength {
                        name: name.to_string(),
                        arg_len: rows.len(),
                        var_len: y.len(),
                    });
                }
                let xb = mat_vec_mul(t, rows, val_elems(name, &args[2])?);
                let eta = v_add(t, &args[1], &Val::Vec(xb));
                let (base, rest) = match name {
                    "normal_id_glm" => ("normal", vec![eta, args[3].clone()]),
                    _ => ("bernoulli_logit", vec![eta]),
                };
                return eval_sample_vec(t, base, y, &rest);
            }
            _ => return Err(wrong_type(name, "a data matrix x", &args[0])),
        },
        // `array[N] vector[K] y` is N observations sharing one covariance, which
        // Stan sums the density over; an unrecognised shape used to contribute 0.
        // Both take a K×K covariance as the variate, so the structure is the
        // observation rather than K of them.
        "wishart" | "inv_wishart" => match (x, &args[0], &args[1]) {
            (Val::Vec(w), nu, Val::Vec(sc))
                // Square and the same size, checked on the rows too — a length-K
                // vector and a K x K matrix both have K entries at the top.
                if !w.is_empty()
                    && w.len() == sc.len()
                    && w.iter().chain(sc).all(|r| r.elems().is_some_and(|e| e.len() == w.len())) =>
            {
                if name == "wishart" {
                    wishart_lpdf(t, w, nu, sc)
                } else {
                    inv_wishart_lpdf(t, w, nu, sc)
                }
            }
            _ => {
                return Err(wrong_type(
                    name,
                    "a K x K covariance variate, degrees of freedom, and a K x K scale",
                    x,
                ))
            }
        },
        "multi_normal_cholesky" | "multi_normal" => match (x, &args[0], &args[1]) {
            (Val::Vec(y), Val::Vec(mu), Val::Vec(rows)) => {
                let owned;
                let l_rows: &[Val] = if name == "multi_normal" {
                    owned = cholesky_decompose(t, rows);
                    &owned
                } else {
                    rows
                };
                let obs: Vec<&[Val]> = match y.first().and_then(Val::elems) {
                    Some(_) => y
                        .iter()
                        .map(|r| r.elems().unwrap_or(std::slice::from_ref(r)))
                        .collect(),
                    None => vec![y.as_slice()],
                };
                let k = mu.len();
                if l_rows.len() != k || obs.iter().any(|o| o.len() != k) {
                    return Err(wrong_type(
                        name,
                        &format!("a variate and a covariance sized to mu (length {k})"),
                        &args[0],
                    ));
                }
                let mut acc = Val::Num(0.0);
                for o in obs {
                    let lp = multi_normal_cholesky_lpdf(t, o, mu, l_rows);
                    acc = v_add(t, &acc, &lp);
                }
                acc
            }
            _ => {
                return Err(wrong_type(
                    name,
                    "a vector variate, a vector mu and a covariance (Sigma, or its \
                     Cholesky factor L)",
                    x,
                ))
            }
        },
        "multinomial" => match (x, &args[0]) {
            (Val::Vec(y), Val::Vec(theta)) => {
                if y.len() != theta.len() {
                    return Err(EvalError::DistributionArgLength {
                        name: name.to_string(),
                        arg_len: theta.len(),
                        var_len: y.len(),
                    });
                }
                multinomial_lpmf(t, y, theta)
            }
            _ => {
                return Err(wrong_type(
                    name,
                    "an integer count array and a simplex theta",
                    x,
                ))
            }
        },
        "lkj_corr_cholesky" => match x {
            Val::Vec(l_rows) => lkj_corr_cholesky_lpdf(t, l_rows, &args[0]),
            _ => return Err(wrong_type(name, "a cholesky_factor_corr variate", x)),
        },
        "lkj_corr" => match x {
            Val::Vec(rows)
                if !rows.is_empty()
                    && rows.iter().all(|r| r.elems().is_some_and(|e| e.len() == rows.len())) =>
            {
                lkj_corr_lpdf(t, rows, &args[0])
            }
            _ => return Err(wrong_type(name, "a K x K correlation matrix variate", x)),
        },
        "dirichlet" => match (x, &args[0]) {
            (Val::Vec(theta), Val::Vec(alpha)) => {
                if theta.len() != alpha.len() {
                    return Err(EvalError::DistributionArgLength {
                        name: name.to_string(),
                        arg_len: alpha.len(),
                        var_len: theta.len(),
                    });
                }
                dirichlet_lpdf(t, theta, alpha)
            }
            _ => return Err(wrong_type(name, "a simplex variate and a vector alpha", x)),
        },
        // `arity` already rejected anything not listed above.
        _ => unreachable!("arity() and eval_dist() must cover the same names"),
    })
}

/// `y ~ dist(...)` on a vector observation: multivariate distributions take the
/// whole structure; scalar ones sum element-wise with argument broadcast.
pub fn eval_sample_vec(t: &mut Tape, name: &str, xs: &[Val], args: &[Val]) -> Result<Val> {
    if is_multivariate(name) {
        return eval_dist(t, name, &Val::Vec(xs.to_vec()), args);
    }
    // Before the length check below, which would otherwise blame the arguments
    // of a distribution that does not exist here at all.
    if arity(name).is_none() {
        return Err(EvalError::UnknownDistribution(name.to_string()));
    }
    // `categorical`'s theta, and its logit form's beta, are shared by every element
    // of the variate rather than being per-observation, so they skip the broadcast.
    if name == "categorical" || name == "categorical_logit" {
        let terms: Vec<Val> = xs
            .iter()
            .map(|x| eval_dist(t, name, x, args))
            .collect::<Result<_>>()?;
        return Ok(v_sum(t, &terms));
    }
    // Vectorized arguments must line up element-wise. Indexing without this check
    // panicked on a short argument and ignored the tail of a long one.
    for a in args {
        if let Val::Vec(av) = a {
            if av.len() != xs.len() {
                return Err(EvalError::DistributionArgLength {
                    name: name.to_string(),
                    arg_len: av.len(),
                    var_len: xs.len(),
                });
            }
        }
    }
    let mut terms: Vec<Val> = Vec::with_capacity(xs.len());
    let mut elem_args: Vec<Val> = Vec::with_capacity(args.len());
    for (i, x) in xs.iter().enumerate() {
        elem_args.clear();
        for a in args {
            elem_args.push(broadcast_elem(a, i));
        }
        terms.push(eval_dist(t, name, x, &elem_args)?);
    }
    Ok(v_sum(t, &terms))
}

fn broadcast_elem(v: &Val, i: usize) -> Val {
    match v {
        // Length already verified by the caller.
        Val::Vec(xs) => xs[i].clone(),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stanwasm_autodiff::lgamma;

    /// K=2 has one free parameter ρ, so the kernel is `(2η-2)·log(L[1][1])`
    /// exactly, and the constant at K=2 is `lgamma(η+½) − lgamma(η) − ½log π`.
    /// Regression test for the structural bug where the kernel always
    /// evaluated to 0, and for the constant that used to be missing.
    #[test]
    fn lkj_corr_cholesky_k2_matches_analytic_formula() {
        let mut t = Tape::new();
        for &rho in &[0.0_f64, 0.3, -0.6, 0.9] {
            for &eta in &[1.0_f64, 2.0, 0.5, 3.5] {
                let l11 = (1.0 - rho * rho).sqrt();
                let l_rows = vec![
                    Val::Vec(vec![Val::Num(1.0), Val::Num(0.0)]),
                    Val::Vec(vec![Val::Num(rho), Val::Num(l11)]),
                ];
                let lp = lkj_corr_cholesky_lpdf(&mut t, &l_rows, &Val::Num(eta));
                let expected = (2.0 * eta - 2.0) * l11.ln()
                    + lgamma(eta + 0.5)
                    - lgamma(eta)
                    - 0.5 * std::f64::consts::PI.ln();
                let got = lp.to_f64(&t).unwrap();
                assert!(
                    (got - expected).abs() < 1e-9,
                    "rho={rho}, eta={eta}: got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn categorical_lpmf_matches_analytic_formula() {
        let mut t = Tape::new();
        let theta = vec![Val::Num(0.2), Val::Num(0.5), Val::Num(0.3)];

        let lp = categorical_lpmf(&mut t, &Val::Num(2.0), &theta).unwrap();
        let got = lp.to_f64(&t).unwrap();
        assert!((got - 0.5_f64.ln()).abs() < 1e-9, "got {got}");

        assert!(categorical_lpmf(&mut t, &Val::Num(0.0), &theta).is_err());
        assert!(categorical_lpmf(&mut t, &Val::Num(4.0), &theta).is_err());
    }

    #[test]
    fn multinomial_lpmf_matches_analytic_formula() {
        let mut t = Tape::new();
        let y = [Val::Num(1.0), Val::Num(2.0), Val::Num(3.0)];
        let theta = [Val::Num(0.2), Val::Num(0.3), Val::Num(0.5)];

        let lp = multinomial_lpmf(&mut t, &y, &theta);
        let got = lp.to_f64(&t).unwrap();

        fn fact(n: u64) -> f64 {
            (1..=n).product::<u64>().max(1) as f64
        }
        let log_coeff = (fact(6) / (fact(1) * fact(2) * fact(3))).ln();
        let expected = log_coeff + 1.0 * 0.2_f64.ln() + 2.0 * 0.3_f64.ln() + 3.0 * 0.5_f64.ln();
        assert!(
            (got - expected).abs() < 1e-9,
            "got {got}, expected {expected}"
        );
    }

    /// `multi_normal` decomposes Σ on the way in; check that path against a
    /// hand-written `L` on `Σ = [[4,2],[2,3]]`, `L = [[2,0],[1,√2]]`.
    #[test]
    fn multi_normal_matches_manual_cholesky_factor() {
        let y = [Val::Num(1.0), Val::Num(2.0)];
        let mu = [Val::Num(0.0), Val::Num(0.0)];

        let mut t = Tape::new();
        let sigma_rows = [
            Val::Vec(vec![Val::Num(4.0), Val::Num(2.0)]),
            Val::Vec(vec![Val::Num(2.0), Val::Num(3.0)]),
        ];
        let l = cholesky_decompose(&mut t, &sigma_rows);
        let got = multi_normal_cholesky_lpdf(&mut t, &y, &mu, &l)
            .to_f64(&t)
            .unwrap();

        let mut t2 = Tape::new();
        let l_rows = [
            Val::Vec(vec![Val::Num(2.0), Val::Num(0.0)]),
            Val::Vec(vec![Val::Num(1.0), Val::Num(2.0_f64.sqrt())]),
        ];
        let expected = multi_normal_cholesky_lpdf(&mut t2, &y, &mu, &l_rows)
            .to_f64(&t2)
            .unwrap();

        assert!(
            (got - expected).abs() < 1e-9,
            "got {got}, expected {expected}"
        );
    }
}
