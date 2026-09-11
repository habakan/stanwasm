//! The gallery's Wasm Sandbox presets, kept building against the runtime.
//!
//! These are the models a first-time visitor actually runs, and the data-block
//! validation added alongside them is strict enough that a preset drifting out
//! of sync with its data would now fail to load rather than quietly sample
//! something else.

use stanwasm_runtime::{data_from_json, Model};

fn check(name: &str, src: &str, data: &str, init: &[f64]) {
    let env = data_from_json(data).unwrap_or_else(|e| panic!("{name}: data: {e}"));
    let model = Model::parse_and_load(src, env).unwrap_or_else(|e| panic!("{name}: load: {e}"));
    assert_eq!(model.n_params(), init.len(), "{name}: n_params");
    let (lp, grad) = model
        .log_prob_grad(init)
        .unwrap_or_else(|e| panic!("{name}: log_prob_grad: {e}"));
    assert!(lp.is_finite(), "{name}: lp = {lp}");
    assert!(
        grad.iter().all(|g| g.is_finite()),
        "{name}: grad = {grad:?}"
    );
}

#[test]
fn linear_regression_preset() {
    let x: Vec<String> = (0..30)
        .map(|i| format!("{}", -1.5 + i as f64 * 0.1))
        .collect();
    // The preset's y in examples/gallery/src/models.ts.
    let y = "-2.15,-2.34,-2.01,-1.67,-1.22,-1.52,-1.46,-0.8,-0.48,-0.25,\
             -0.45,-0.4,-0.38,0.37,-0.15,-0.15,0.21,0.85,0.94,0.76,\
             0.6,1.74,1.72,1.35,2.44,2.82,2.69,2.91,3.23,3.04";
    check(
        "linear_regression",
        r#"
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
}"#,
        &format!(r#"{{"N":30,"x":[{}],"y":[{y}]}}"#, x.join(",")),
        &[0.0, 1.0, 0.0],
    );
}

#[test]
fn poisson_regression_preset() {
    check(
        "poisson_regression",
        r#"
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
}"#,
        r#"{"N":5,"x":[0,1,2,3,4],"y":[1,2,5,12,30]}"#,
        &[0.0, 1.0],
    );
}

#[test]
fn eight_schools_preset() {
    check(
        "eight_schools",
        r#"
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
}"#,
        r#"{"J":8,"y":[28,8,-3,7,-1,1,18,12],"sigma":[15,10,16,11,9,11,10,18]}"#,
        &[0.1; 10],
    );
}

#[test]
fn funnel_preset_from_the_mcmc_visualizer() {
    // `exp(y / 2)` — `y` is a real parameter, so this must stay real division.
    check(
        "funnel",
        r#"
parameters {
  real y;
  real x;
}
model {
  y ~ normal(0, 3);
  x ~ normal(0, exp(y / 2));
}"#,
        "{}",
        &[0.5, 0.2],
    );
}
