//! End-to-end log_prob and gradient tests against hand-computed values.

use stanwasm_runtime::{Env, Model, Val};

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
    let (lp, grads) = model.log_prob_grad(&params).unwrap();

    // Hand-computed: priors -3.221523 (alpha), -3.226523 (beta), -1 (sigma);
    // likelihood -0.918938 twice; jacobian log|d sigma/d log_sigma| = log_sigma = 0.
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

    // d/d alpha at (0, 1, 1): prior -alpha/100 = 0, likelihood
    // sum (y_i - alpha - beta x_i)/sigma² = (0-0-0) + (1-0-1) = 0. Total 0.
    assert!(close(grads[0], 0.0, 1e-9), "d/dalpha = {}", grads[0]);

    // d/d beta: prior -beta/100 = -0.01, likelihood
    // sum (y - alpha - beta x)·x/sigma² = 0*0 + (1-0-1)*1 = 0. Total -0.01.
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
    let (lp, grads) = model.log_prob_grad(&params).unwrap();
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
    let (lp, grads) = model.log_prob_grad(&params).unwrap();
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
    let (lp, grads) = model.log_prob_grad(&init).unwrap();
    assert!(lp.is_finite(), "lp = {lp}");
    assert!(grads.iter().all(|g| g.is_finite()), "grads = {grads:?}");

    // Numerical gradient check
    let h = 1e-5;
    for i in 0..3 {
        let mut p_plus = init.clone();
        let mut p_minus = init.clone();
        p_plus[i] += h;
        p_minus[i] -= h;
        let (lp_plus, _) = model.log_prob_grad(&p_plus).unwrap();
        let (lp_minus, _) = model.log_prob_grad(&p_minus).unwrap();
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
    let (lp, grads) = model.log_prob_grad(&init).unwrap();
    assert!(lp.is_finite(), "lp = {lp}");

    let h = 1e-5;
    for i in 0..2 {
        let mut p_plus = init.clone();
        let mut p_minus = init.clone();
        p_plus[i] += h;
        p_minus[i] -= h;
        let (lp_plus, _) = model.log_prob_grad(&p_plus).unwrap();
        let (lp_minus, _) = model.log_prob_grad(&p_minus).unwrap();
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
    let (lp, grads) = model.log_prob_grad(&init).unwrap();
    assert!(lp.is_finite(), "lp = {lp}");

    let h = 1e-5;
    for i in 0..4 {
        let mut p_plus = init.clone();
        let mut p_minus = init.clone();
        p_plus[i] += h;
        p_minus[i] -= h;
        let (lp_plus, _) = model.log_prob_grad(&p_plus).unwrap();
        let (lp_minus, _) = model.log_prob_grad(&p_minus).unwrap();
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
    let (_lp0, grads) = model.log_prob_grad(&params).unwrap();

    let h = 1e-5;
    for i in 0..params.len() {
        let mut p_plus = params.clone();
        let mut p_minus = params.clone();
        p_plus[i] += h;
        p_minus[i] -= h;
        let (lp_plus, _) = model.log_prob_grad(&p_plus).unwrap();
        let (lp_minus, _) = model.log_prob_grad(&p_minus).unwrap();
        let fd = (lp_plus - lp_minus) / (2.0 * h);
        assert!(
            (fd - grads[i]).abs() < 1e-5,
            "param[{i}]: autodiff={}, finite-diff={}",
            grads[i],
            fd
        );
    }
}

#[test]
fn trig_functions_evaluate_and_differentiate() {
    // The tape has carried Sin/Cos/Tan/Asin/Acos/Atan since the autodiff crate was
    // written; until now nothing in `eval_call` could reach them.
    let src = r#"
data { int<lower=0> N; vector[N] t; vector[N] y; }
parameters { real amp; real phase; }
model { for (n in 1:N) y[n] ~ normal(amp * sin(t[n] + phase), 1.0); }
"#;
    let mut data = Env::new();
    data.set_scalar("N", 3.0);
    data.set_vector("t", &[0.0, 0.5, 1.0]);
    data.set_vector("y", &[0.0, 0.4794255386, 0.8414709848]);
    let model = Model::parse_and_load(src, data).unwrap();

    // amp=1, phase=0 reproduces y exactly, so every residual is 0 and the log
    // density is the three normal constants.
    let (lp, grad) = model.log_prob_grad(&[1.0, 0.0]).unwrap();
    assert!((lp - 3.0 * -0.9189385332).abs() < 1e-9, "lp = {lp}");
    assert!(grad.iter().all(|g| g.abs() < 1e-9), "grad = {grad:?}");
}

#[test]
fn atan2_gives_the_quadrant_corrected_angle() {
    let src = r#"
data { real mx; real my; }
parameters { real hd; }
model { atan2(my, mx) ~ normal(hd, 1.0); }
"#;
    // Second quadrant: atan alone would report -pi/4, atan2 reports 3pi/4.
    let mut data = Env::new();
    data.set_scalar("mx", -1.0);
    data.set_scalar("my", 1.0);
    let model = Model::parse_and_load(src, data).unwrap();

    let expected = std::f64::consts::FRAC_PI_2 + std::f64::consts::FRAC_PI_4;
    let (_, grad) = model.log_prob_grad(&[expected]).unwrap();
    assert!(
        grad[0].abs() < 1e-9,
        "hd at the true angle should be a mode: {grad:?}"
    );
}

#[test]
fn matrix_times_vector_is_the_matrix_product() {
    // `X * beta` used to be a clean ShapeMismatch, so the standard regression idiom
    // had to be written as a loop.
    let src = r#"
data { matrix[2,2] X; vector[2] y; }
parameters { vector[2] b; }
model { y ~ normal(X * b, 1.0); }
"#;
    let mut data = Env::new();
    data.set(
        "X",
        Val::Vec(vec![
            Val::Vec(vec![Val::Num(1.0), Val::Num(2.0)]),
            Val::Vec(vec![Val::Num(3.0), Val::Num(4.0)]),
        ]),
    );
    data.set_vector("y", &[1.0, 2.0]);
    let model = Model::parse_and_load(src, data).unwrap();

    // b = (1, 0) gives X*b = (1, 3), so the residuals are (0, -1).
    let (lp, grad) = model.log_prob_grad(&[1.0, 0.0]).unwrap();
    assert!(close(lp, -0.5 - 2.0 * 0.9189385332, 1e-9), "lp = {lp}");
    assert!(
        close(grad[0], -3.0, 1e-9) && close(grad[1], -4.0, 1e-9),
        "grad = {grad:?}"
    );
}

/// `bernoulli(1)` and `binomial(n | n, 1)` are certainties, not NaN: the term
/// the observation doesn't select used to be `0 * log 0`.
#[test]
fn a_degenerate_bernoulli_is_finite() {
    let mut d = Env::new();
    d.set_scalar("y", 1.0);
    let src = "data { int y; } parameters { real a; } model { y ~ bernoulli(1.0); }";
    let (lp, g) = Model::parse_and_load(src, d.clone())
        .unwrap()
        .log_prob_grad(&[0.0])
        .unwrap();
    assert_eq!(lp, 0.0);
    assert_eq!(g, vec![0.0]);

    let src0 = "data { int y; } parameters { real a; } model { y ~ bernoulli(0.0); }";
    let lp0 = Model::parse_and_load(src0, d)
        .unwrap()
        .log_prob_grad(&[0.0])
        .unwrap()
        .0;
    assert_eq!(lp0, f64::NEG_INFINITY);
}

#[test]
fn a_degenerate_binomial_is_finite() {
    let src = "parameters { real a; } model { 3 ~ binomial(3, 1.0); }";
    let (lp, g) = Model::parse_and_load(src, Env::new())
        .unwrap()
        .log_prob_grad(&[0.0])
        .unwrap();
    assert!((lp - 0.0).abs() < 1e-12, "{lp}");
    assert_eq!(g, vec![0.0]);
}

/// `bernoulli_logit_glm(x, alpha, beta)` is `bernoulli_logit(alpha + x * beta)`,
/// and has to agree with it term for term.
#[test]
fn the_glm_form_matches_the_long_form() {
    let mut d = Env::new();
    d.set_scalar("N", 3.0);
    d.set(
        "x",
        Val::Vec(
            [[1.0, 0.5], [-1.0, 2.0], [0.25, -0.75]]
                .iter()
                .map(|r| Val::Vec(r.iter().map(|v| Val::Num(*v)).collect()))
                .collect(),
        ),
    );
    d.set(
        "y",
        Val::Vec([1.0, 0.0, 1.0].iter().map(|v| Val::Num(*v)).collect()),
    );
    let head = "data { int<lower=0> N; matrix[N, 2] x; array[N] int<lower=0,upper=1> y; }
                parameters { real alpha; vector[2] beta; }
                model { ";
    let at = [0.3, -0.4, 0.9];
    let glm = Model::parse_and_load(
        &format!("{head} y ~ bernoulli_logit_glm(x, alpha, beta); }}"),
        d.clone(),
    )
    .unwrap()
    .log_prob_grad(&at)
    .unwrap();
    let long = Model::parse_and_load(
        &format!("{head} y ~ bernoulli_logit(alpha + x * beta); }}"),
        d,
    )
    .unwrap()
    .log_prob_grad(&at)
    .unwrap();
    assert!((glm.0 - long.0).abs() < 1e-12, "{glm:?} vs {long:?}");
    for (g, l) in glm.1.iter().zip(&long.1) {
        assert!((g - l).abs() < 1e-12, "{glm:?} vs {long:?}");
    }
}

/// `normal_id_glm(x, alpha, beta, sigma)` is `normal(alpha + x * beta, sigma)`,
/// and has to agree with it term for term.
#[test]
fn the_normal_glm_form_matches_the_long_form() {
    let mut d = Env::new();
    d.set_scalar("N", 3.0);
    d.set(
        "x",
        Val::Vec(
            [[1.0, 0.5], [-1.0, 2.0], [0.25, -0.75]]
                .iter()
                .map(|r| Val::Row(r.iter().map(|v| Val::Num(*v)).collect()))
                .collect(),
        ),
    );
    d.set_vector("y", &[0.3, -1.2, 0.8]);
    let head = "data { int<lower=0> N; matrix[N, 2] x; vector[N] y; }
                parameters { real alpha; vector[2] beta; real<lower=0> sigma; }
                model { ";
    let at = [0.3, -0.4, 0.9, 0.2];
    let glm = Model::parse_and_load(
        &format!("{head} y ~ normal_id_glm(x, alpha, beta, sigma); }}"),
        d.clone(),
    )
    .unwrap()
    .log_prob_grad(&at)
    .unwrap();
    let long = Model::parse_and_load(
        &format!("{head} y ~ normal(alpha + x * beta, sigma); }}"),
        d,
    )
    .unwrap()
    .log_prob_grad(&at)
    .unwrap();
    assert!((glm.0 - long.0).abs() < 1e-12, "{glm:?} vs {long:?}");
    for (g, l) in glm.1.iter().zip(&long.1) {
        assert!((g - l).abs() < 1e-12, "{glm:?} vs {long:?}");
    }
}

/// An unknown distribution on a container variate used to be reported as an
/// argument-length mismatch, which blames arguments it never had.
#[test]
fn an_unknown_distribution_on_a_vector_says_so() {
    let mut d = Env::new();
    d.set_vector("y", &[1.0, 2.0]);
    let src = "data { vector[2] y; } parameters { real a; } model { y ~ not_a_dist(a, 1); }";
    let msg = match Model::parse_and_load(src, d).unwrap().log_prob_grad(&[0.5]) {
        Err(e) => e.to_string(),
        Ok(v) => panic!("expected an error, got {v:?}"),
    };
    assert!(msg.contains("not_a_dist"), "{msg}");
    assert!(!msg.contains("length"), "{msg}");
}

/// `categorical_logit(beta)` is `categorical(softmax(beta))`, and `beta` is
/// shared by every element of the variate rather than broadcast across it.
#[test]
fn categorical_logit_matches_the_simplex_form() {
    let mut d = Env::new();
    d.set(
        "y",
        Val::Vec([1.0, 3.0, 2.0].iter().map(|v| Val::Num(*v)).collect()),
    );
    let head = "data { array[3] int<lower=1,upper=3> y; } parameters { vector[3] b; } model { ";
    let at = [0.4, -1.1, 0.7];
    let logit = Model::parse_and_load(&format!("{head} y ~ categorical_logit(b); }}"), d.clone())
        .unwrap()
        .log_prob_grad(&at)
        .unwrap();
    let simplex = Model::parse_and_load(&format!("{head} y ~ categorical(softmax(b)); }}"), d)
        .unwrap()
        .log_prob_grad(&at)
        .unwrap();
    assert!(
        (logit.0 - simplex.0).abs() < 1e-12,
        "{logit:?} vs {simplex:?}"
    );
    for (a, b) in logit.1.iter().zip(&simplex.1) {
        assert!((a - b).abs() < 1e-12, "{logit:?} vs {simplex:?}");
    }
}

/// An underflowed intermediate used to poison the whole gradient. `x^n` with
/// `n < 1` — `sqrt` among them — has an infinite slope at zero, and that
/// infinity meeting an adjoint gives NaN. The reference implementation drops
/// the term at a zero base; so does this one.
#[test]
fn a_root_of_an_underflowed_value_does_not_poison_the_gradient() {
    // exp(-1000) is exactly zero in f64, so sqrt sees a zero base
    let src = "parameters { real a; } model { target += sqrt(exp(-1000 * (a * a + 1))) + a; }";
    let (lp, g) = Model::parse_and_load(src, Env::new())
        .unwrap()
        .log_prob_grad(&[0.0])
        .unwrap();
    assert_eq!(lp, 0.0);
    assert_eq!(g, vec![1.0]);

    // and a base that is merely small still carries its slope
    let small = "parameters { real a; } model { target += sqrt(exp(-40 * (a * a + 1))); }";
    let (_, g) = Model::parse_and_load(small, Env::new())
        .unwrap()
        .log_prob_grad(&[0.5])
        .unwrap();
    assert!(g[0] < 0.0 && g[0].is_finite(), "{g:?}");
}
