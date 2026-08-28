//! Tests for the `generated quantities` block, `_rng` functions, and the
//! control-flow evaluator (`if`/`while`/`break`/`continue`, comparisons).

use std::cell::RefCell;
use std::rc::Rc;

use rand::{rngs::ChaCha8Rng, SeedableRng};
use stanwasm_runtime::{Env, Model};

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

fn gq_rng_model() -> Model {
    let mut data = Env::new();
    data.set_scalar("N", 2.0);
    data.set_vector("x", &[0.0, 1.0]);
    data.set_vector("y", &[0.0, 1.0]);
    Model::parse_and_load(GQ_RNG_MODEL, data).unwrap()
}

#[test]
fn gen_quantity_names_match_declaration_order() {
    let model = gq_rng_model();
    assert_eq!(
        model.gen_quantity_names(),
        vec!["y_ln", "y_exp", "y_unif", "y_gam"]
    );
}

#[test]
fn generated_quantities_respect_distribution_support() {
    let model = gq_rng_model();
    // mu=0.5, log_sigma=0 (sigma=1) unconstrained.
    let unconstrained = vec![0.5, 0.0];
    let rng = Rc::new(RefCell::new(ChaCha8Rng::seed_from_u64(42)));

    for _ in 0..20 {
        let gq = model
            .generated_quantities(&unconstrained, rng.clone())
            .unwrap();
        assert_eq!(gq.len(), 4);
        let [y_ln, y_exp, y_unif, y_gam] = [gq[0], gq[1], gq[2], gq[3]];
        assert!(y_ln > 0.0, "lognormal_rng must be positive, got {y_ln}");
        assert!(
            y_exp >= 0.0,
            "exponential_rng must be non-negative, got {y_exp}"
        );
        assert!(
            (0.0..=1.0).contains(&y_unif),
            "uniform_rng out of [0,1]: {y_unif}"
        );
        assert!(y_gam >= 0.0, "gamma_rng must be non-negative, got {y_gam}");
    }
}

#[test]
fn generated_quantities_rng_stream_advances_across_draws() {
    let model = gq_rng_model();
    let unconstrained = vec![0.5, 0.0];
    let rng = Rc::new(RefCell::new(ChaCha8Rng::seed_from_u64(7)));

    let first = model
        .generated_quantities(&unconstrained, rng.clone())
        .unwrap();
    let second = model
        .generated_quantities(&unconstrained, rng.clone())
        .unwrap();
    assert_ne!(first, second, "shared rng must not repeat the same draw");
}

#[test]
fn constrained_draw_applies_lower_bound_transform() {
    let model = gq_rng_model();
    // mu=0.5 (unconstrained == constrained, no bound), log_sigma=0 -> sigma=exp(0)=1.
    let constrained = model.constrained_draw(&[0.5, 0.0]).unwrap();
    assert_eq!(constrained.len(), 2);
    assert!((constrained[0] - 0.5).abs() < 1e-12);
    assert!((constrained[1] - 1.0).abs() < 1e-12);
}

fn logp_of(src: &str, params: &[f64]) -> f64 {
    let model = Model::parse_and_load(src, Env::new()).unwrap();
    let (lp, _grads) = model.log_prob_grad(params).unwrap();
    lp
}

#[test]
fn if_else_branches_on_comparison() {
    const SRC: &str = r#"
parameters {
  real x;
}
model {
  if (x > 0) {
    target += 100;
  } else {
    target += 200;
  }
}
"#;
    assert!((logp_of(SRC, &[1.0]) - 100.0).abs() < 1e-12);
    assert!((logp_of(SRC, &[-1.0]) - 200.0).abs() < 1e-12);
}

#[test]
fn while_loop_with_break_stops_early() {
    const SRC: &str = r#"
parameters {
  real x;
}
model {
  real i;
  i = 0;
  while (1) {
    i = i + 1;
    if (i >= 3) {
      break;
    }
  }
  target += i;
}
"#;
    assert!((logp_of(SRC, &[0.0]) - 3.0).abs() < 1e-12);
}

#[test]
fn while_loop_with_continue_skips_body_tail() {
    const SRC: &str = r#"
parameters {
  real x;
}
model {
  real acc;
  real i;
  acc = 0;
  i = 0;
  while (i < 5) {
    i = i + 1;
    if (i == 3) {
      continue;
    }
    acc = acc + 1;
  }
  target += acc;
}
"#;
    // i = 1,2,3,4,5; acc increments for every i except i==3 -> acc = 4.
    assert!((logp_of(SRC, &[0.0]) - 4.0).abs() < 1e-12);
}
