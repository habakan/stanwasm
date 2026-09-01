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
use crate::matrix::{cholesky_decompose, mat_mdiv_ltri_low, vec_dot_self};
use crate::ops::{v_add, v_div, v_exp, v_lgamma, v_log, v_mul, v_neg, v_sub};
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
pub fn multi_normal_cholesky_lpdf(t: &mut Tape, y: &[Val], mu: &[Val], l_rows: &[Val]) -> Val {
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

/// `multi_normal_lpdf(y | μ, Σ)` — y, μ are vectors, Σ is the full K×K
/// covariance matrix (as opposed to `multi_normal_cholesky`, which takes its
/// Cholesky factor directly). Cholesky-decomposes Σ and reuses the same
/// math as `multi_normal_cholesky_lpdf`.
pub fn multi_normal_lpdf(t: &mut Tape, y: &[Val], mu: &[Val], sigma_rows: &[Val]) -> Val {
    let l_rows = cholesky_decompose(t, sigma_rows);
    multi_normal_cholesky_lpdf(t, y, mu, &l_rows)
}

/// `multinomial_lpmf(y | θ)` — y is an integer count array of length K,
/// θ a simplex of length K. log p = lgamma(N+1) − Σ lgamma(yᵢ+1) + Σ yᵢ log θᵢ,
/// where N = Σ yᵢ.
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

/// `categorical_lpmf(y | θ)` — y is a 1-indexed category label, θ a simplex
/// of length K. log p = log θ_y. Unlike the other distributions here, θ is a
/// vector argument shared across every element when vectorized (`y ~
/// categorical(theta)` over an `array[N] int y`), not a per-observation
/// value — see the special case in `eval_sample_vec`.
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

/// `lkj_corr_cholesky_lpdf(L | η)` — L is the Cholesky factor of a K×K
/// correlation matrix.
///
/// log p = Σ_{k=0..K-1} [(K − 1 − k) + (2η − 2)] · log Lₖₖ
///
/// The `(K-1-k)` term is the Jacobian of Σ=LLᵀ restricted to the
/// correlation-Cholesky manifold (why `lkj_corr_cholesky` is a distinct
/// distribution from `lkj_corr`, not just a change of variables); `(2η-2)`
/// is the LKJ density's own `det(Σ)^(η-1)` term, since det(Σ) = Π Lₖₖ².
/// These combine additively per row (same base, summed exponents), NOT as
/// a single `(2η-2)` factor applied to the whole weighted sum — that former
/// structure is wrong: for K=2 it always evaluates to exactly 0, since the
/// only row with a free diagonal (row 1) has `(K-1-k) = 0`, so multiplying
/// by `(2η-2)` afterward can't undo that. Row 0's diagonal is always the
/// fixed constant 1 (`log(1) = 0`), so including it in the sum is harmless
/// regardless of its weight. Omits the η,K-only normalizing constant
/// (doesn't affect gradients w.r.t. `L`; would matter only if `eta` itself
/// were a sampled parameter).
pub fn lkj_corr_cholesky_lpdf(t: &mut Tape, l_rows: &[Val], eta: &Val) -> Val {
    let kk = l_rows.len();
    let two_eta = v_mul(t, &Val::Num(2.0), eta);
    let two_eta_minus_2 = v_sub(t, &two_eta, &Val::Num(2.0));
    let mut lp = Val::Num(0.0);
    for (k, row_v) in l_rows.iter().enumerate() {
        let base_wt = (kk - 1 - k) as f64;
        if let Val::Vec(row) = row_v {
            if k < row.len() {
                let weight = v_add(t, &Val::Num(base_wt), &two_eta_minus_2);
                let lr = v_log(t, &row[k]);
                let term = v_mul(t, &weight, &lr);
                lp = v_add(t, &lp, &term);
            }
        }
    }
    lp
}

/// Distributions whose first argument is a *whole* vector / matrix (not a
/// scalar element). For these, sampling a `Val::Vec` does NOT broadcast the
/// scalar lpdf over elements — the entire structure is the observation.
fn is_multivariate(name: &str) -> bool {
    matches!(
        name,
        "multi_normal_cholesky"
            | "multi_normal"
            | "lkj_corr_cholesky"
            | "dirichlet"
            | "multinomial"
    )
}

/// How many arguments each supported distribution takes after the variate.
/// Checked before dispatch so `a ~ normal(0);` is a clean error instead of the
/// out-of-bounds `args[1]` panic (a wasm trap in the browser) it used to be.
fn arity(name: &str) -> Option<usize> {
    Some(match name {
        "std_normal" => 0,
        "exponential" | "half_normal" | "bernoulli" | "bernoulli_logit" | "poisson"
        | "dirichlet" | "lkj_corr_cholesky" | "multinomial" | "categorical" => 1,
        "normal"
        | "cauchy"
        | "lognormal"
        | "gamma"
        | "beta"
        | "neg_binomial_2"
        | "multi_normal_cholesky"
        | "multi_normal" => 2,
        "student_t" => 3,
        _ => return None,
    })
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
        "gamma" => gamma_lpdf(t, x, &args[0], &args[1]),
        "beta" => beta_lpdf(t, x, &args[0], &args[1]),
        "bernoulli" => bernoulli_lpmf(t, x, &args[0]),
        "bernoulli_logit" => bernoulli_logit_lpmf(t, x, &args[0]),
        "poisson" => poisson_lpmf(t, x, &args[0]),
        "neg_binomial_2" => neg_binomial_2_lpmf(t, x, &args[0], &args[1]),
        "categorical" => match &args[0] {
            Val::Vec(theta) => categorical_lpmf(t, x, theta)?,
            _ => return Err(wrong_type(name, "a simplex vector theta", &args[0])),
        },
        // The multivariate forms below used to fall back to `Val::Num(0.0)` on
        // a shape they didn't recognize, i.e. contribute nothing to the log
        // density and silently return a wrong posterior. They are errors now.
        "multi_normal_cholesky" => match (x, &args[0], &args[1]) {
            (Val::Vec(y), Val::Vec(mu), Val::Vec(l_rows)) => {
                // `array[N] vector[K] y; y ~ multi_normal_cholesky(mu, L);` is
                // legal Stan but arrives here as one N-row container, so it
                // would otherwise be reported as a confusing size mismatch
                // between the N rows and the K-long mu.
                if matches!(y.first(), Some(Val::Vec(_))) {
                    return Err(EvalError::MultivariateNotVectorized {
                        name: name.to_string(),
                        got: x.shape().to_string(),
                    });
                }
                if y.len() != mu.len() || l_rows.len() != y.len() {
                    return Err(wrong_type(
                        name,
                        &format!("mu and L sized to match the variate (length {})", y.len()),
                        &args[0],
                    ));
                }
                multi_normal_cholesky_lpdf(t, y, mu, l_rows)
            }
            _ => {
                return Err(wrong_type(
                    name,
                    "a vector variate, a vector mu and a Cholesky factor L",
                    x,
                ))
            }
        },
        "multi_normal" => match (x, &args[0], &args[1]) {
            (Val::Vec(y), Val::Vec(mu), Val::Vec(sigma_rows)) => {
                // Same `array[N] vector[K] y` footgun as `multi_normal_cholesky`.
                if matches!(y.first(), Some(Val::Vec(_))) {
                    return Err(EvalError::MultivariateNotVectorized {
                        name: name.to_string(),
                        got: x.shape().to_string(),
                    });
                }
                if y.len() != mu.len() || sigma_rows.len() != y.len() {
                    return Err(wrong_type(
                        name,
                        &format!(
                            "mu and Sigma sized to match the variate (length {})",
                            y.len()
                        ),
                        &args[0],
                    ));
                }
                multi_normal_lpdf(t, y, mu, sigma_rows)
            }
            _ => {
                return Err(wrong_type(
                    name,
                    "a vector variate, a vector mu and a covariance matrix Sigma",
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

/// Sample statement (`y ~ dist(...)`) on a vector observation. Behaviour:
/// - For multivariate distributions, the whole vector / matrix is the
///   observation and we delegate to `eval_dist` once with `x = Val::Vec(xs)`.
/// - For scalar distributions, sums the scalar lpdf over each element with
///   element-wise argument broadcast.
pub fn eval_sample_vec(t: &mut Tape, name: &str, xs: &[Val], args: &[Val]) -> Result<Val> {
    if is_multivariate(name) {
        return eval_dist(t, name, &Val::Vec(xs.to_vec()), args);
    }
    // `categorical`'s vector argument (theta) is the simplex shared by every
    // element of the variate, not a per-observation value like every other
    // vectorized distribution's arguments — so it skips the element-wise
    // broadcast below and is passed through unchanged to each call.
    if name == "categorical" {
        let mut acc = Val::Num(0.0);
        for x in xs {
            let term = eval_dist(t, name, x, args)?;
            acc = v_add(t, &acc, &term);
        }
        return Ok(acc);
    }
    // Every vectorized argument must line up with the variate element-wise.
    // Indexing without this check panicked (wasm trap) on a short argument and
    // silently ignored the tail of a long one.
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
    Ok(acc)
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

    /// For K=2, a correlation matrix has one free parameter ρ; its Cholesky
    /// factor is `L = [[1, 0], [ρ, sqrt(1-ρ²)]]`. The LKJ(η) density is
    /// `p(Σ|η) ∝ det(Σ)^(η-1) = (1-ρ²)^(η-1) = L[1][1]^(2η-2)`, so
    /// `log p(L|η) = (2η-2)·log(L[1][1])` exactly (K=2 has no Cholesky
    /// Jacobian contribution beyond this). Regression test for the
    /// structural bug where this always evaluated to exactly 0.
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
                let expected = (2.0 * eta - 2.0) * l11.ln();
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

    /// `multi_normal_lpdf` Cholesky-decomposes Σ internally; check it agrees
    /// with `multi_normal_cholesky_lpdf` fed the hand-computed factor
    /// `Σ = [[4,2],[2,3]] = L Lᵀ`, `L = [[2,0],[1,√2]]`.
    #[test]
    fn multi_normal_matches_manual_cholesky_factor() {
        let y = [Val::Num(1.0), Val::Num(2.0)];
        let mu = [Val::Num(0.0), Val::Num(0.0)];

        let mut t = Tape::new();
        let sigma_rows = [
            Val::Vec(vec![Val::Num(4.0), Val::Num(2.0)]),
            Val::Vec(vec![Val::Num(2.0), Val::Num(3.0)]),
        ];
        let got = multi_normal_lpdf(&mut t, &y, &mu, &sigma_rows)
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
