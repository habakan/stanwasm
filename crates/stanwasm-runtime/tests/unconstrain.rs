//! `unconstrain_draw` against `constrained_draw`. Round-tripping is the whole
//! claim: a draw fitted elsewhere has to land on the unconstrained point that
//! produced it, or the density evaluated from it is a different model's.

use stanwasm_runtime::{Env, Model};

fn model(decl: &str) -> Model {
    let src = format!("parameters {{ {decl} }}\nmodel {{ }}");
    Model::parse_and_load(&src, Env::new()).unwrap()
}

/// `constrained_draw` also emits `transformed parameters`; with none declared
/// its output is exactly what `unconstrain_draw` reads back.
fn round_trip(decl: &str, raw: &[f64]) {
    let m = model(decl);
    let constrained = m.constrained_draw(raw).unwrap();
    assert_eq!(
        constrained.len(),
        m.constrained_param_names().len(),
        "{decl}: names and values disagree"
    );
    let back = m.unconstrain_draw(&constrained).unwrap();
    assert_eq!(back.len(), raw.len(), "{decl}: wrong unconstrained length");
    for (i, (b, r)) in back.iter().zip(raw).enumerate() {
        assert!(
            (b - r).abs() < 1e-9,
            "{decl}: slot {i} came back {b}, not {r}"
        );
    }
}

#[test]
fn scalar_bounds_round_trip() {
    round_trip("real a;", &[-1.3]);
    round_trip("real<lower=0> s;", &[0.7]);
    round_trip("real<upper=4> u;", &[-0.4]);
    round_trip("real<lower=-1, upper=2> b;", &[1.1]);
}

#[test]
fn containers_round_trip() {
    round_trip("vector[3] v;", &[0.1, -0.2, 0.3]);
    round_trip("vector<lower=0>[3] v;", &[0.1, -0.2, 0.3]);
    round_trip("row_vector<upper=1>[2] r;", &[0.4, -0.9]);
    round_trip("matrix<lower=0, upper=5>[2, 3] m;", &[0.1, 0.2, 0.3, -0.1, -0.2, 0.4]);
    round_trip("array[2] real<lower=0> a;", &[0.5, -0.5]);
    round_trip("array[2] vector[2] a;", &[0.1, 0.2, 0.3, 0.4]);
}

#[test]
fn shape_transforms_round_trip() {
    round_trip("simplex[4] p;", &[0.3, -0.6, 1.2]);
    round_trip("ordered[3] o;", &[-1.0, 0.2, 0.7]);
    round_trip("positive_ordered[3] o;", &[-1.0, 0.2, 0.7]);
    round_trip("cholesky_factor_corr[3] L;", &[0.4, -0.3, 0.9]);
    round_trip("corr_matrix[3] C;", &[0.4, -0.3, 0.9]);
    round_trip("cholesky_factor_cov[3] L;", &[0.4, -0.3, 0.2, 0.9, -0.1, 0.3]);
    round_trip("cov_matrix[3] S;", &[0.4, -0.3, 0.2, 0.9, -0.1, 0.3]);
    round_trip("array[2] simplex[3] p;", &[0.3, -0.6, 0.1, 0.8]);
}

/// A unit vector's radius is not recoverable, so the round trip is through the
/// constrained value rather than back to the same raw point.
#[test]
fn a_unit_vector_round_trips_through_its_constrained_value() {
    let m = model("unit_vector[3] u;");
    let c = m.constrained_draw(&[1.0, -2.0, 2.0]).unwrap();
    let again = m.constrained_draw(&m.unconstrain_draw(&c).unwrap()).unwrap();
    for (a, b) in c.iter().zip(&again) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }
}

/// The log density has to agree too, which is what a user actually does with
/// the result — a round trip that lands elsewhere on the manifold would not.
#[test]
fn the_log_density_survives_a_round_trip() {
    let src = "
        data { int N; vector[N] y; }
        parameters { real mu; real<lower=0> sigma; simplex[3] w; }
        model { y ~ normal(mu, sigma); target += sum(w); }
    ";
    let mut env = Env::new();
    env.set_scalar("N", 4.0);
    env.set_vector("y", &[1.0, 2.0, 3.0, 4.5]);
    let m = Model::parse_and_load(src, env).unwrap();

    let raw = [0.4, -0.2, 0.6, 1.1];
    let (lp, _) = m.log_prob_grad(&raw).unwrap();
    let back = m.unconstrain_draw(&m.constrained_draw(&raw).unwrap()).unwrap();
    let (lp2, _) = m.log_prob_grad(&back).unwrap();
    assert!((lp - lp2).abs() < 1e-9, "{lp} != {lp2}");
}

#[test]
fn a_draw_of_the_wrong_length_is_refused() {
    let m = model("real<lower=0> s; vector[2] v;");
    let err = m.unconstrain_draw(&[1.0, 2.0]).unwrap_err().to_string();
    assert!(err.contains("needs 2 more"), "{err}");
    let err = m.unconstrain_draw(&[1.0, 2.0, 3.0, 4.0]).unwrap_err().to_string();
    assert!(err.contains("expected 3"), "{err}");
}

#[test]
fn a_value_off_its_support_is_named() {
    let m = model("real<lower=0> s;");
    let err = m.unconstrain_draw(&[-1.0]).unwrap_err().to_string();
    assert!(err.contains('s') && err.contains("outside"), "{err}");
}

/// A bound that names an earlier parameter has to be resolved against that
/// parameter's value, in both directions.
#[test]
fn a_parameter_dependent_bound_round_trips() {
    round_trip(
        "real<lower=0, upper=1> a; real<lower=0, upper=(1 - a)> b;",
        &[0.3, -0.8],
    );
    let m = model("real<lower=0, upper=1> a; real<lower=0, upper=(1 - a)> b;");
    let c = m.constrained_draw(&[0.3, -0.8]).unwrap();
    assert!(c[1] < 1.0 - c[0], "b = {} is not under 1 - a = {}", c[1], 1.0 - c[0]);
}
