//! `_rng` sampling functions, valid only inside `generated quantities`.
//!
//! Parameterizations mirror the corresponding `_lpdf`/`_lpmf` in
//! `distributions.rs` (e.g. `gamma(shape, rate)`, matching Stan).

use crate::env::Env;
use crate::error::EvalError;
use crate::value::Val;
use rand::Rng;
use rand_distr::{
    Bernoulli, Beta, Cauchy, Distribution, Exp, Gamma, LogNormal, Normal, Poisson, StudentT,
    Uniform,
};
use stan_autodiff::Tape;

type Result<T> = std::result::Result<T, EvalError>;

/// Build a distribution or turn its rejection (e.g. non-positive scale) into
/// a clean `EvalError` instead of the `.unwrap()` panic this used to be —
/// invalid RNG parameters are an easy mistake in hand-edited Stan source
/// (e.g. the Wasm Sandbox tab), not an internal bug.
fn invalid(call: &str, e: impl std::fmt::Display) -> EvalError {
    EvalError::InvalidRngParams(format!("{call}: {e}"))
}

pub fn normal_rng(rng: &mut impl Rng, mu: f64, sigma: f64) -> Result<f64> {
    let d = Normal::new(0.0, 1.0).map_err(|e| invalid("normal_rng", e))?;
    Ok(mu + sigma * d.sample(rng))
}

pub fn exponential_rng(rng: &mut impl Rng, lambda: f64) -> Result<f64> {
    let d = Exp::new(lambda).map_err(|e| invalid("exponential_rng", e))?;
    Ok(d.sample(rng))
}

pub fn half_normal_rng(rng: &mut impl Rng, sigma: f64) -> Result<f64> {
    Ok(normal_rng(rng, 0.0, sigma)?.abs())
}

pub fn cauchy_rng(rng: &mut impl Rng, mu: f64, sigma: f64) -> Result<f64> {
    let d = Cauchy::new(mu, sigma).map_err(|e| invalid("cauchy_rng", e))?;
    Ok(d.sample(rng))
}

pub fn student_t_rng(rng: &mut impl Rng, nu: f64, mu: f64, sigma: f64) -> Result<f64> {
    let d = StudentT::new(nu).map_err(|e| invalid("student_t_rng", e))?;
    Ok(mu + sigma * d.sample(rng))
}

pub fn lognormal_rng(rng: &mut impl Rng, mu: f64, sigma: f64) -> Result<f64> {
    let d = LogNormal::new(mu, sigma).map_err(|e| invalid("lognormal_rng", e))?;
    Ok(d.sample(rng))
}

/// Stan's `gamma(alpha, beta)` is shape/rate; `rand_distr::Gamma` is shape/scale.
pub fn gamma_rng(rng: &mut impl Rng, alpha: f64, beta: f64) -> Result<f64> {
    let d = Gamma::new(alpha, 1.0 / beta).map_err(|e| invalid("gamma_rng", e))?;
    Ok(d.sample(rng))
}

pub fn beta_rng(rng: &mut impl Rng, a: f64, b: f64) -> Result<f64> {
    let d = Beta::new(a, b).map_err(|e| invalid("beta_rng", e))?;
    Ok(d.sample(rng))
}

pub fn uniform_rng(rng: &mut impl Rng, lo: f64, hi: f64) -> Result<f64> {
    let d = Uniform::new(lo, hi).map_err(|e| invalid("uniform_rng", e))?;
    Ok(d.sample(rng))
}

pub fn bernoulli_rng(rng: &mut impl Rng, theta: f64) -> Result<f64> {
    let d = Bernoulli::new(theta).map_err(|e| invalid("bernoulli_rng", e))?;
    Ok(if d.sample(rng) { 1.0 } else { 0.0 })
}

pub fn bernoulli_logit_rng(rng: &mut impl Rng, alpha: f64) -> Result<f64> {
    bernoulli_rng(rng, 1.0 / (1.0 + (-alpha).exp()))
}

pub fn poisson_rng(rng: &mut impl Rng, lambda: f64) -> Result<f64> {
    let d = Poisson::new(lambda).map_err(|e| invalid("poisson_rng", e))?;
    Ok(d.sample(rng))
}

/// Gamma-Poisson mixture: `neg_binomial_2(mu, phi)` has mean `mu` and
/// variance `mu + mu^2/phi`.
pub fn neg_binomial_2_rng(rng: &mut impl Rng, mu: f64, phi: f64) -> Result<f64> {
    let g = Gamma::new(phi, mu / phi).map_err(|e| invalid("neg_binomial_2_rng", e))?;
    let lambda = g.sample(rng);
    let p = Poisson::new(lambda).map_err(|e| invalid("neg_binomial_2_rng", e))?;
    Ok(p.sample(rng))
}

/// Independent `Gamma(alpha_i, 1)` draws, normalized.
pub fn dirichlet_rng(rng: &mut impl Rng, alpha: &[f64]) -> Result<Vec<f64>> {
    let draws: Vec<f64> = alpha
        .iter()
        .map(|&a| {
            Gamma::new(a, 1.0)
                .map_err(|e| invalid("dirichlet_rng", e))
                .map(|d| d.sample(rng))
        })
        .collect::<Result<_>>()?;
    let sum: f64 = draws.iter().sum();
    Ok(draws.iter().map(|x| x / sum).collect())
}

/// `mu + L * z`, `z ~ iid N(0, 1)`. `l` is the lower-triangular Cholesky
/// factor as a vec of rows (matches how `multi_normal_cholesky_lpdf` in
/// `distributions.rs` receives it).
pub fn multi_normal_cholesky_rng(
    rng: &mut impl Rng,
    mu: &[f64],
    l: &[Vec<f64>],
) -> Result<Vec<f64>> {
    let k = mu.len();
    let z: Vec<f64> = (0..k)
        .map(|_| normal_rng(rng, 0.0, 1.0))
        .collect::<Result<_>>()?;
    Ok((0..k)
        .map(|i| {
            let mut s = mu[i];
            for (j, zj) in z.iter().enumerate().take(i + 1) {
                s += l[i][j] * zj;
            }
            s
        })
        .collect())
}

/// Dispatch a `<base>_rng(args...)` call. `env` must carry an RNG (set only
/// while evaluating `generated quantities`); calling this outside that
/// context, or with an unknown name / wrong arity, is a clean error rather
/// than a panic — both are easy mistakes to make from hand-edited Stan
/// source (e.g. the Wasm Sandbox tab).
pub fn dispatch(t: &Tape, base: &str, args: &[Val], env: &Env) -> Result<Val> {
    let rng_handle = env
        .rng()
        .ok_or_else(|| EvalError::RngOutsideGeneratedQuantities(base.to_string()))?;
    let mut rng = rng_handle.borrow_mut();
    Ok(match (base, args) {
        ("normal", [mu, sigma]) => Val::Num(normal_rng(&mut *rng, mu.to_f64(t), sigma.to_f64(t))?),
        ("std_normal", []) => Val::Num(normal_rng(&mut *rng, 0.0, 1.0)?),
        ("exponential", [lambda]) => Val::Num(exponential_rng(&mut *rng, lambda.to_f64(t))?),
        ("half_normal", [sigma]) => Val::Num(half_normal_rng(&mut *rng, sigma.to_f64(t))?),
        ("cauchy", [mu, sigma]) => Val::Num(cauchy_rng(&mut *rng, mu.to_f64(t), sigma.to_f64(t))?),
        ("student_t", [nu, mu, sigma]) => Val::Num(student_t_rng(
            &mut *rng,
            nu.to_f64(t),
            mu.to_f64(t),
            sigma.to_f64(t),
        )?),
        ("lognormal", [mu, sigma]) => {
            Val::Num(lognormal_rng(&mut *rng, mu.to_f64(t), sigma.to_f64(t))?)
        }
        ("gamma", [alpha, beta]) => {
            Val::Num(gamma_rng(&mut *rng, alpha.to_f64(t), beta.to_f64(t))?)
        }
        ("beta", [a, b]) => Val::Num(beta_rng(&mut *rng, a.to_f64(t), b.to_f64(t))?),
        ("uniform", [lo, hi]) => Val::Num(uniform_rng(&mut *rng, lo.to_f64(t), hi.to_f64(t))?),
        ("bernoulli", [theta]) => Val::Num(bernoulli_rng(&mut *rng, theta.to_f64(t))?),
        ("bernoulli_logit", [alpha]) => Val::Num(bernoulli_logit_rng(&mut *rng, alpha.to_f64(t))?),
        ("poisson", [lambda]) => Val::Num(poisson_rng(&mut *rng, lambda.to_f64(t))?),
        ("neg_binomial_2", [mu, phi]) => {
            Val::Num(neg_binomial_2_rng(&mut *rng, mu.to_f64(t), phi.to_f64(t))?)
        }
        ("dirichlet", [Val::Vec(alpha)]) => {
            let a: Vec<f64> = alpha.iter().map(|v| v.to_f64(t)).collect();
            Val::Vec(
                dirichlet_rng(&mut *rng, &a)?
                    .into_iter()
                    .map(Val::Num)
                    .collect(),
            )
        }
        ("multi_normal_cholesky", [Val::Vec(mu), Val::Vec(l_rows)]) => {
            let mu: Vec<f64> = mu.iter().map(|v| v.to_f64(t)).collect();
            let l: Vec<Vec<f64>> = l_rows
                .iter()
                .map(|row| match row {
                    Val::Vec(r) => r.iter().map(|v| v.to_f64(t)).collect(),
                    _ => Vec::new(),
                })
                .collect();
            Val::Vec(
                multi_normal_cholesky_rng(&mut *rng, &mu, &l)?
                    .into_iter()
                    .map(Val::Num)
                    .collect(),
            )
        }
        _ => return Err(EvalError::UnknownRng(base.to_string())),
    })
}
