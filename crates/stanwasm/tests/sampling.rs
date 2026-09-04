//! End-to-end: parse + trace + nuts-rs sample inside a single Rust crate.
//! Validates that `StanModel::sample` produces reasonable posterior samples
//! for linear regression by checking that the mean of the post-warmup draws
//! lies near the (data-implied) maximum-likelihood point.

use stanwasm::{init_gradient_check, StanModel};

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

const DATA: &str = r#"{
  "N": 30,
  "x": [-1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6,
        -0.5, -0.4, -0.3, -0.2, -0.1,  0.0,  0.1,  0.2,  0.3,  0.4,
         0.5,  0.6,  0.7,  0.8,  0.9,  1.0,  1.1,  1.2,  1.3,  1.4],
  "y": [-1.3, -1.0, -0.7, -0.5, -0.4, -0.1,  0.0,  0.2,  0.4,  0.5,
         0.7,  0.8,  1.0,  1.2,  1.3,  1.5,  1.7,  1.8,  2.0,  2.2,
         2.4,  2.5,  2.7,  2.9,  3.1,  3.3,  3.4,  3.6,  3.8,  4.0]
}"#;

#[test]
fn linear_regression_n_params_and_names() {
    let model = StanModel::new(LINEAR_REGRESSION, DATA).unwrap();
    assert_eq!(model.n_params(), 3);
    assert_eq!(
        model.param_names(),
        vec!["alpha".to_string(), "beta".to_string(), "sigma".to_string()]
    );
}

#[test]
fn linear_regression_log_prob_grad_finite() {
    let mut model = StanModel::new(LINEAR_REGRESSION, DATA).unwrap();
    let out = model.log_prob_grad(&[0.0, 1.0, 0.0]).unwrap();
    assert_eq!(out.len(), 4); // [lp, dα, dβ, dσ]
    assert!(out.iter().all(|v| v.is_finite()), "{out:?}");
}

#[test]
fn linear_regression_sample_recovers_slope() {
    // y ≈ 1.5 + 1.5*x with σ ≈ 0.1 → posterior mean of β should be near 1.5
    let mut model = StanModel::new(LINEAR_REGRESSION, DATA).unwrap();
    let init = vec![0.0, 0.0, 0.0]; // α, β, log_σ
    let n_warmup = 200;
    let n_draws = 400;
    let n = model.n_params();
    let samples = model.sample(&init, n_warmup, n_draws, 42).unwrap();
    assert_eq!(samples.len(), n * (n_warmup + n_draws) as usize);

    // Take post-warmup draws only
    let post_warmup_offset = n * n_warmup as usize;
    let draws = &samples[post_warmup_offset..];
    let mut sum_alpha = 0.0;
    let mut sum_beta = 0.0;
    for chunk in draws.chunks(n) {
        sum_alpha += chunk[0];
        sum_beta += chunk[1];
    }
    let mean_alpha = sum_alpha / n_draws as f64;
    let mean_beta = sum_beta / n_draws as f64;

    // True slope is 1.5; intercept is ~0.45 (line through points). NUTS should
    // get within 0.5 of the truth on this short run.
    assert!(
        (mean_beta - 1.5).abs() < 0.3,
        "mean β = {mean_beta}, expected near 1.5"
    );
    assert!(mean_alpha.is_finite() && mean_alpha.abs() < 2.0);
}

#[test]
fn step_sampling_recovers_slope_and_restores_model() {
    let mut model = StanModel::new(LINEAR_REGRESSION, DATA).unwrap();
    let init = vec![0.0, 0.0, 0.0];
    let n_warmup: u32 = 200;
    let n_draws: u32 = 400;
    let n = model.n_params();

    model
        .start_step_sampling(&init, n_warmup, n_draws, 42)
        .unwrap();

    let mut sum_alpha = 0.0;
    let mut sum_beta = 0.0;
    for i in 0..(n_warmup + n_draws) {
        let out = model.step_draw().unwrap();
        // n_params position values + tuning flag + diverging flag + step_size + num_steps.
        assert_eq!(out.len(), n + 4);
        assert!(out[..n].iter().all(|v| v.is_finite()));
        let tuning = out[n] != 0.0;
        assert_eq!(
            tuning,
            i < n_warmup,
            "tuning flag should match warmup phase at draw {i}"
        );
        assert!(out[n + 2] > 0.0, "step_size should be positive");
        assert!(out[n + 3] >= 1.0, "num_steps should be at least 1");
        if !tuning {
            sum_alpha += out[0];
            sum_beta += out[1];
        }
    }
    let mean_alpha = sum_alpha / n_draws as f64;
    let mean_beta = sum_beta / n_draws as f64;
    assert!(
        (mean_beta - 1.5).abs() < 0.3,
        "mean β = {mean_beta}, expected near 1.5"
    );
    assert!(mean_alpha.is_finite() && mean_alpha.abs() < 2.0);

    // Exhausting stepDraw() should have auto-restored logProbGrad/sample.
    let out = model.log_prob_grad(&init).unwrap();
    assert!(out.iter().all(|v| v.is_finite()));
    let samples = model.sample(&init, 10, 10, 7).unwrap();
    assert_eq!(samples.len(), n * 20);
}

#[test]
fn finish_step_sampling_is_safe_to_call_when_idle() {
    let mut model = StanModel::new(LINEAR_REGRESSION, DATA).unwrap();
    // No startStepSampling call yet — must not panic.
    model.finish_step_sampling();
    // Still usable afterward.
    let out = model.log_prob_grad(&[0.0, 0.0, 0.0]).unwrap();
    assert!(out.iter().all(|v| v.is_finite()));
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

const MULTIVARIATE_DATA: &str = r#"{
  "K": 2,
  "y": [1.0, 2.0]
}"#;

#[test]
fn multivariate_lkj_compiles_and_samples() {
    let mut model = StanModel::new(MULTIVARIATE_LKJ, MULTIVARIATE_DATA).unwrap();
    assert_eq!(model.n_params(), 3);

    let init = vec![0.1; 3];
    let n_warmup = 200;
    let n_draws = 400;
    let samples = model.sample(&init, n_warmup, n_draws, 42).unwrap();
    assert_eq!(samples.len(), 3 * (n_warmup + n_draws) as usize);
    assert!(samples.iter().all(|s| s.is_finite()));
}

const MULTI_NORMAL_FULL_COV: &str = r#"
data {
  int<lower=1> K;
  vector[K] y;
  matrix[K, K] Sigma;
}
parameters {
  vector[K] mu;
}
model {
  mu ~ normal(0, 5);
  y  ~ multi_normal(mu, Sigma);
}
"#;

const MULTI_NORMAL_FULL_COV_DATA: &str = r#"{
  "K": 2,
  "y": [1.0, 2.0],
  "Sigma": [[4.0, 2.0], [2.0, 3.0]]
}"#;

#[test]
fn multi_normal_full_cov_compiles_and_samples() {
    let mut model = StanModel::new(MULTI_NORMAL_FULL_COV, MULTI_NORMAL_FULL_COV_DATA).unwrap();
    assert_eq!(model.n_params(), 2);

    let init = vec![0.1; 2];
    let n_warmup = 200;
    let n_draws = 400;
    let samples = model.sample(&init, n_warmup, n_draws, 42).unwrap();
    assert_eq!(samples.len(), 2 * (n_warmup + n_draws) as usize);
    assert!(samples.iter().all(|s| s.is_finite()));
}

const CATEGORICAL_MODEL: &str = r#"
data {
  int<lower=1> K;
  int<lower=1> y;
}
parameters {
  simplex[K] theta;
}
model {
  y ~ categorical(theta);
}
"#;

const CATEGORICAL_DATA: &str = r#"{
  "K": 3,
  "y": 2
}"#;

#[test]
fn categorical_compiles_and_samples() {
    let mut model = StanModel::new(CATEGORICAL_MODEL, CATEGORICAL_DATA).unwrap();
    assert_eq!(model.n_params(), 2);

    let init = vec![0.1; 2];
    let n_warmup = 200;
    let n_draws = 400;
    let samples = model.sample(&init, n_warmup, n_draws, 42).unwrap();
    assert_eq!(samples.len(), 2 * (n_warmup + n_draws) as usize);
    assert!(samples.iter().all(|s| s.is_finite()));
}

const MULTINOMIAL_MODEL: &str = r#"
data {
  int<lower=1> K;
  array[K] int<lower=0> y;
}
parameters {
  simplex[K] theta;
}
model {
  y ~ multinomial(theta);
}
"#;

const MULTINOMIAL_DATA: &str = r#"{
  "K": 3,
  "y": [3, 5, 2]
}"#;

#[test]
fn multinomial_compiles_and_samples() {
    let mut model = StanModel::new(MULTINOMIAL_MODEL, MULTINOMIAL_DATA).unwrap();
    assert_eq!(model.n_params(), 2);

    let init = vec![0.1; 2];
    let n_warmup = 200;
    let n_draws = 400;
    let samples = model.sample(&init, n_warmup, n_draws, 42).unwrap();
    assert_eq!(samples.len(), 2 * (n_warmup + n_draws) as usize);
    assert!(samples.iter().all(|s| s.is_finite()));
}

const GQ_RNG_MODEL: &str = r#"
data {
  int<lower=0> N;
  vector[N] x;
  array[N] real y;
}
parameters {
  real mu;
  real<lower=0> sigma;
}
model {
  mu    ~ normal(0, 5);
  sigma ~ exponential(1);
  for (i in 1:N) {
    y[i] ~ normal(mu, sigma);
  }
}
generated quantities {
  real y_ln  = lognormal_rng(mu, sigma);
  real y_exp = exponential_rng(1.0);
  real y_unif = uniform_rng(0.0, 1.0);
  real y_gam = gamma_rng(2.0, 1.0);
}
"#;

const GQ_RNG_DATA: &str = r#"{
  "N": 2,
  "x": [0.0, 1.0],
  "y": [0.0, 1.0]
}"#;

#[test]
fn generated_quantities_end_to_end() {
    let mut model = StanModel::new(GQ_RNG_MODEL, GQ_RNG_DATA).unwrap();
    assert_eq!(
        model.gen_quantity_names(),
        vec!["y_ln", "y_exp", "y_unif", "y_gam"]
    );

    let n = model.n_params();
    let n_warmup = 50;
    let n_draws = 20;
    let draws = model.sample(&[0.0, 0.0], n_warmup, n_draws, 42).unwrap();
    let post_warmup = &draws[n * n_warmup as usize..];
    assert_eq!(post_warmup.len(), n * n_draws as usize);

    let constrained = model.constrain_draw(&post_warmup[0..n]).unwrap();
    assert_eq!(constrained.len(), 2);
    assert!(
        constrained[1] > 0.0,
        "sigma must be positive: {constrained:?}"
    );

    let gq = model
        .generated_quantities(post_warmup, n_draws, 123)
        .unwrap();
    assert_eq!(gq.len(), 4 * n_draws as usize);
    for row in gq.chunks(4) {
        let [y_ln, y_exp, y_unif, y_gam] = [row[0], row[1], row[2], row[3]];
        assert!(y_ln > 0.0);
        assert!(y_exp >= 0.0);
        assert!((0.0..=1.0).contains(&y_unif));
        assert!(y_gam >= 0.0);
    }
}

#[test]
fn compile_to_wasm_returns_valid_module() {
    let mut model = StanModel::new(LINEAR_REGRESSION, DATA).unwrap();
    let wasm = model.compile_to_wasm(None).unwrap();
    assert!(
        wasm.starts_with(&[0x00, 0x61, 0x73, 0x6d]),
        "wasm magic missing"
    );
    let result = wasmparser::Validator::new().validate_all(&wasm);
    assert!(result.is_ok(), "wasm did not validate: {:?}", result.err());
}

/// The sampler refuses a starting point whose gradient has a zero component —
/// nuts-rs says only "Invalid initial point" — so the message has to name the
/// parameters that make it one.
#[test]
fn a_starting_point_the_sampler_refuses_names_the_parameters() {
    let names: Vec<String> = ["a", "unused", "b"].iter().map(|s| s.to_string()).collect();
    let msg = init_gradient_check(&names, -3.0, &[1.0, 0.0, 2.0]).unwrap_err();
    assert!(msg.contains("(unused)"), "{msg}");
    assert!(msg.contains("1 of the 3 parameters"), "{msg}");

    assert!(init_gradient_check(&names, -3.0, &[1.0, 2.0, 3.0]).is_ok());
    let nan = init_gradient_check(&names, f64::NAN, &[1.0, 2.0, 3.0]).unwrap_err();
    assert!(nan.contains("log density is NaN"), "{nan}");
}

/// `b` enters only through `a`, so its gradient is zero wherever `a` is — and
/// zero is where the obvious starting point puts it. `randomInit` is what finds
/// a point the sampler accepts. (The refusal itself is a `JsError`, which
/// cannot be built off wasm, so the smoke test covers that half.)
#[test]
fn random_init_finds_a_startable_point() {
    let src = "parameters { real a; real b; }
               model { a ~ normal(0, 1); target += a * b; }";
    let mut model = StanModel::new(src, "{}").unwrap();
    let at = model.random_init(1).unwrap();
    assert_eq!(at.len(), 2);
    assert!(at.iter().all(|v| (-2.0..=2.0).contains(v)), "{at:?}");
    assert!(at[0] != 0.0, "a must be somewhere b's gradient is not zero");
    model.sample(&at, 10, 10, 1).unwrap();
}
