//! End-to-end log_prob and gradient tests against hand-computed values.

use stan_runtime::{Env, Model};

const LINEAR_REGRESSION: &str = r#"
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 10);
  beta  ~ normal(0, 10);
  sigma ~ exponential(1);
  y ~ normal(alpha + beta * x, sigma);
}
"#;

fn close(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps
}

#[test]
fn linear_regression_logp_at_known_point() {
    // Tiny dataset: N=2, x=[0,1], y=[0,1]
    let mut data = Env::new();
    data.set_scalar("N", 2.0);
    data.set_vector("x", &[0.0, 1.0]);
    data.set_vector("y", &[0.0, 1.0]);

    let model = Model::parse_and_load(LINEAR_REGRESSION, data).unwrap();
    assert_eq!(model.n_params(), 3);

    // Unconstrained params: alpha=0, beta=1, log_sigma=0 (so sigma=1).
    let params = vec![0.0, 1.0, 0.0];
    let (lp, grads) = model.log_prob_grad(&params);

    // Hand-computed components:
    //   prior: normal_lpdf(0; 0, 10)
    //          = -log(sqrt(2π)) - log(10) = -0.918938... - 2.302585... = -3.221523...
    //   prior: normal_lpdf(1; 0, 10)
    //          = -log(sqrt(2π)) - log(10) - 0.5 * (1/10)² = -3.221523 - 0.005 = -3.226523
    //   prior: exponential_lpdf(1; 1) = log(1) - 1 = -1
    //   likelihood: y[0]=0 ~ N(0, 1) = -log(sqrt(2π)) = -0.918938
    //               y[1]=1 ~ N(1, 1) = -log(sqrt(2π)) = -0.918938
    //   jacobian: log|d sigma/d log_sigma| = log_sigma = 0
    let log_sqrt_2pi = 0.918_938_533_204_672_8;
    let expected_lp = -log_sqrt_2pi - 10f64.ln() // prior alpha
        + (-log_sqrt_2pi - 10f64.ln() - 0.5 * (1.0 / 10.0_f64).powi(2)) // prior beta
        + (0f64.ln_1p() * 0.0 - 1.0) // exponential_lpdf at sigma=1, lambda=1: log(1) - 1*1
        + (-log_sqrt_2pi) // y[0]
        + (-log_sqrt_2pi) // y[1]
        + 0.0; // jacobian

    assert!(
        close(lp, expected_lp, 1e-9),
        "lp mismatch: got {lp}, expected {expected_lp}"
    );
    assert_eq!(grads.len(), 3, "grad length");

    // Sanity: gradient w.r.t. alpha at (0, 1, 1) with x=[0,1], y=[0,1]:
    //   d/d alpha [normal(α; 0, 10)] = -α/100 = 0
    //   d/d alpha sum normal(y_i; α + β x_i, σ) = sum (y_i - α - β x_i)/σ²
    //                                            = (0-0-0) + (1-0-1) = 0
    //   total: 0
    assert!(close(grads[0], 0.0, 1e-9), "d/dalpha = {}", grads[0]);

    // d/d beta:
    //   prior: -β/100 = -0.01
    //   likelihood: sum (y - α - β x) * x / σ² = 0*0 + (1-0-1)*1 = 0
    //   total: -0.01
    assert!(close(grads[1], -0.01, 1e-9), "d/dbeta = {}", grads[1]);
}

const EIGHT_SCHOOLS_NCP: &str = r#"
data {
  int<lower=0> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}
parameters {
  real mu;
  real<lower=0> tau;
  vector[J] theta_tilde;
}
transformed parameters {
  vector[J] theta = mu + tau * theta_tilde;
}
model {
  mu ~ normal(0, 5);
  tau ~ half_normal(5);
  theta_tilde ~ normal(0, 1);
  y ~ normal(theta, sigma);
}
"#;

#[test]
fn eight_schools_logp_finite_at_origin() {
    // Smoke test: parser handles transformed parameters, half_normal, vector arithmetic.
    let mut data = Env::new();
    data.set_scalar("J", 8.0);
    data.set_vector("y", &[28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]);
    data.set_vector("sigma", &[15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]);

    let model = Model::parse_and_load(EIGHT_SCHOOLS_NCP, data).unwrap();
    // Params: mu (1) + log_tau (1) + theta_tilde (8) = 10
    assert_eq!(model.n_params(), 10);

    // Initial point: all 0.1 (avoids tau=0 zero-grad pathology)
    let params = vec![0.1; 10];
    let (lp, grads) = model.log_prob_grad(&params);
    assert!(lp.is_finite(), "lp = {lp}");
    assert!(grads.iter().all(|g| g.is_finite()), "grads = {grads:?}");
    assert_eq!(grads.len(), 10);
}

const POISSON_REGRESSION: &str = r#"
data {
  int<lower=0> N;
  vector[N] x;
  array[N] int y;
}
parameters {
  real alpha;
  real beta;
}
model {
  alpha ~ normal(0, 5);
  beta  ~ normal(0, 1);
  for (i in 1:N) y[i] ~ poisson(exp(alpha + beta * x[i]));
}
"#;

#[test]
fn poisson_regression_logp_finite() {
    let mut data = Env::new();
    data.set_scalar("N", 5.0);
    data.set_vector("x", &[0.0, 1.0, 2.0, 3.0, 4.0]);
    data.set_vector("y", &[1.0, 2.0, 5.0, 12.0, 30.0]);

    let model = Model::parse_and_load(POISSON_REGRESSION, data).unwrap();
    assert_eq!(model.n_params(), 2);

    let params = vec![0.0, 1.0];
    let (lp, grads) = model.log_prob_grad(&params);
    assert!(lp.is_finite(), "lp = {lp}");
    assert_eq!(grads.len(), 2);
    assert!(grads.iter().all(|g| g.is_finite()));
}

const MULTIVARIATE_LKJ: &str = r#"
data {
  int<lower=1> K;
  vector[K] y;
}
parameters {
  vector[K] mu;
  cholesky_factor_corr[K] L;
}
model {
  mu ~ normal(0, 5);
  L  ~ lkj_corr_cholesky(2.0);
  y  ~ multi_normal_cholesky(mu, L);
}
"#;

#[test]
fn multivariate_lkj_sampling_reasonable() {
    let mut data = Env::new();
    data.set_scalar("K", 2.0);
    data.set_vector("y", &[1.0, 2.0]);
    let model = Model::parse_and_load(MULTIVARIATE_LKJ, data).unwrap();
    // params: mu (K=2) + cholesky_factor_corr[K] raw (K*(K-1)/2 = 1) = 3
    assert_eq!(model.n_params(), 3);

    // Smoke: lp + grads finite at small init
    let init = vec![0.1; 3];
    let (lp, grads) = model.log_prob_grad(&init);
    assert!(lp.is_finite(), "lp = {lp}");
    assert!(grads.iter().all(|g| g.is_finite()), "grads = {grads:?}");

    // Numerical gradient check
    let h = 1e-5;
    for i in 0..3 {
        let mut p_plus = init.clone();
        let mut p_minus = init.clone();
        p_plus[i] += h;
        p_minus[i] -= h;
        let (lp_plus, _) = model.log_prob_grad(&p_plus);
        let (lp_minus, _) = model.log_prob_grad(&p_minus);
        let fd = (lp_plus - lp_minus) / (2.0 * h);
        assert!(
            (fd - grads[i]).abs() < 1e-4,
            "param[{i}]: autodiff={}, finite-diff={}",
            grads[i],
            fd
        );
    }
}

const SIMPLEX_DIRICHLET: &str = r#"
data {
  int<lower=2> K;
  array[K] int<lower=0> y;
  vector[K] alpha;
}
parameters {
  simplex[K] theta;
}
model {
  for (k in 1:K) {
    target += (alpha[k] - 1) * log(theta[k]);
  }
  for (k in 1:K) {
    target += y[k] * log(theta[k]);
  }
}
"#;

#[test]
fn simplex_dirichlet_finite_diff() {
    let mut data = Env::new();
    data.set_scalar("K", 3.0);
    data.set_vector("y", &[3.0, 5.0, 2.0]);
    data.set_vector("alpha", &[2.0, 1.0, 3.0]);
    let model = Model::parse_and_load(SIMPLEX_DIRICHLET, data).unwrap();
    // simplex[K=3] uses K-1 = 2 unconstrained params
    assert_eq!(model.n_params(), 2);

    let init = vec![0.2, -0.3];
    let (lp, grads) = model.log_prob_grad(&init);
    assert!(lp.is_finite(), "lp = {lp}");

    let h = 1e-5;
    for i in 0..2 {
        let mut p_plus = init.clone();
        let mut p_minus = init.clone();
        p_plus[i] += h;
        p_minus[i] -= h;
        let (lp_plus, _) = model.log_prob_grad(&p_plus);
        let (lp_minus, _) = model.log_prob_grad(&p_minus);
        let fd = (lp_plus - lp_minus) / (2.0 * h);
        assert!(
            (fd - grads[i]).abs() < 1e-4,
            "param[{i}]: autodiff={}, finite-diff={}",
            grads[i],
            fd
        );
    }
}

const ORDERED_MEANS: &str = r#"
data { int<lower=1> K; vector[K] y; }
parameters { ordered[K] mu; real<lower=0> sigma; }
model {
  mu ~ normal(0, 10);
  sigma ~ exponential(1);
  for (i in 1:K) y[i] ~ normal(mu[i], sigma);
}
"#;

#[test]
fn ordered_means_finite_diff() {
    let mut data = Env::new();
    data.set_scalar("K", 3.0);
    data.set_vector("y", &[1.0, 2.5, 4.0]);
    let model = Model::parse_and_load(ORDERED_MEANS, data).unwrap();
    // ordered[K=3] uses K=3 raw params + sigma = 4
    assert_eq!(model.n_params(), 4);

    let init = vec![0.1, -0.2, 0.3, 0.0];
    let (lp, grads) = model.log_prob_grad(&init);
    assert!(lp.is_finite(), "lp = {lp}");

    let h = 1e-5;
    for i in 0..4 {
        let mut p_plus = init.clone();
        let mut p_minus = init.clone();
        p_plus[i] += h;
        p_minus[i] -= h;
        let (lp_plus, _) = model.log_prob_grad(&p_plus);
        let (lp_minus, _) = model.log_prob_grad(&p_minus);
        let fd = (lp_plus - lp_minus) / (2.0 * h);
        assert!(
            (fd - grads[i]).abs() < 1e-4,
            "param[{i}]: autodiff={}, finite-diff={}",
            grads[i],
            fd
        );
    }
}

#[test]
fn finite_difference_check_linear_regression() {
    // Numerical gradient check vs analytical autodiff.
    let mut data = Env::new();
    data.set_scalar("N", 3.0);
    data.set_vector("x", &[1.0, 2.0, 3.0]);
    data.set_vector("y", &[1.5, 3.1, 4.9]);
    let model = Model::parse_and_load(LINEAR_REGRESSION, data).unwrap();

    let params = vec![0.5, 1.5, -0.2]; // alpha, beta, log_sigma
    let (_lp0, grads) = model.log_prob_grad(&params);

    let h = 1e-5;
    for i in 0..params.len() {
        let mut p_plus = params.clone();
        let mut p_minus = params.clone();
        p_plus[i] += h;
        p_minus[i] -= h;
        let (lp_plus, _) = model.log_prob_grad(&p_plus);
        let (lp_minus, _) = model.log_prob_grad(&p_minus);
        let fd = (lp_plus - lp_minus) / (2.0 * h);
        assert!(
            (fd - grads[i]).abs() < 1e-5,
            "param[{i}]: autodiff={}, finite-diff={}",
            grads[i],
            fd
        );
    }
}
