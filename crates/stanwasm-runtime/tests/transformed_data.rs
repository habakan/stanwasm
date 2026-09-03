//! `transformed data` is evaluated once against the data env, so what it
//! declares is a constant visible to every block that follows.

use std::cell::RefCell;
use std::rc::Rc;

use rand::{rngs::ChaCha8Rng, SeedableRng};
use stanwasm_runtime::{Env, Model};

fn data() -> Env {
    let mut d = Env::new();
    d.set_scalar("n", 3.0);
    d.set_vector("year", &[1.0, 2.0, 3.0]);
    d
}

#[test]
fn declaration_and_assignment_are_separate_statements() {
    let src = r#"
data { int<lower=0> n; vector[n] year; }
transformed data {
  vector[n] year_squared;
  year_squared = square(year);
}
parameters { real a; }
model { target += a * sum(year_squared); }
"#;
    let (lp, grad) = Model::parse_and_load(src, data())
        .unwrap()
        .log_prob_grad(&[1.0])
        .unwrap();
    assert!((lp - 14.0).abs() < 1e-12);
    assert!((grad[0] - 14.0).abs() < 1e-12);
}

#[test]
fn transformed_parameters_see_it() {
    let src = r#"
data { int<lower=0> n; vector[n] year; }
transformed data { vector[n] year_squared = square(year); }
parameters { real a; }
transformed parameters { vector[n] mu = a * year_squared; }
model { target += sum(mu); }
"#;
    let (lp, _) = Model::parse_and_load(src, data())
        .unwrap()
        .log_prob_grad(&[1.0])
        .unwrap();
    assert!((lp - 14.0).abs() < 1e-12);
}

#[test]
fn generated_quantities_see_it() {
    let src = r#"
data { int<lower=0> n; vector[n] year; }
transformed data { real total = sum(year); }
parameters { real a; }
model { a ~ normal(0, 1); }
generated quantities { real scaled = a * total; }
"#;
    let model = Model::parse_and_load(src, data()).unwrap();
    let rng = Rc::new(RefCell::new(ChaCha8Rng::seed_from_u64(0)));
    let gq = model.generated_quantities(&[2.0], rng).unwrap();
    assert!((gq[0] - 12.0).abs() < 1e-12);
}

#[test]
fn a_parameter_may_be_sized_by_it() {
    let src = r#"
data { int<lower=0> n; }
transformed data { int k = n - 1; }
parameters { vector[k] b; }
model { b ~ normal(0, 1); }
"#;
    let mut d = Env::new();
    d.set_scalar("n", 4.0);
    assert_eq!(Model::parse_and_load(src, d).unwrap().n_params(), 3);
}

#[test]
fn it_is_evaluated_once_and_stays_constant() {
    let src = r#"
data { int<lower=0> n; vector[n] year; }
transformed data { vector[n] w = exp(year); }
parameters { real a; }
model { target += a * sum(w); }
"#;
    let model = Model::parse_and_load(src, data()).unwrap();
    let first = model.log_prob_grad(&[1.0]).unwrap();
    let second = model.log_prob_grad(&[1.0]).unwrap();
    assert_eq!(first, second);
}

#[test]
fn a_failing_statement_is_reported_at_load() {
    let src = r#"
data { int<lower=0> n; }
transformed data { real bad = missing_variable; }
parameters { real a; }
model { a ~ normal(0, 1); }
"#;
    let mut d = Env::new();
    d.set_scalar("n", 1.0);
    let err = match Model::parse_and_load(src, d) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected the undefined variable to be reported"),
    };
    assert!(err.contains("transformed data"), "{err}");
}
