//! End-to-end: parse + trace + nuts-rs sample inside a single Rust crate.
//! Validates that `StanModel::sample` produces reasonable posterior samples
//! for linear regression by checking that the mean of the post-warmup draws
//! lies near the (data-implied) maximum-likelihood point.

use stan_wasm_api::StanModel;

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
fn compile_to_wasm_returns_valid_module() {
    let model = StanModel::new(LINEAR_REGRESSION, DATA).unwrap();
    let wasm = model.compile_to_wasm().unwrap();
    assert!(wasm.starts_with(&[0x00, 0x61, 0x73, 0x6d]), "wasm magic missing");
    let result = wasmparser::Validator::new().validate_all(&wasm);
    assert!(result.is_ok(), "wasm did not validate: {:?}", result.err());
}
