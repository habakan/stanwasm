//! `student_t_lccdf(y | nu, mu, sigma)` — `log P(T > y)`. The value goes
//! through a regularized incomplete beta, so it is checked against the closed
//! forms the Student-t CDF has at one, two and three degrees of freedom rather
//! than against itself.

use std::f64::consts::PI;

use stanwasm_runtime::{Env, Model};

fn lccdf(y: f64, nu: f64, mu: f64, sigma: f64) -> f64 {
    let src = format!(
        "parameters {{ real a; }} model {{ target += student_t_lccdf({y} | {nu}, {mu}, {sigma}); }}"
    );
    Model::parse_and_load(&src, Env::new())
        .unwrap()
        .log_prob_grad(&[0.0])
        .unwrap()
        .0
}

/// `1 - F(t)` for a standard Student-t, from the elementary form each of these
/// degrees of freedom happens to have.
fn tail(t: f64, nu: u32) -> f64 {
    let cdf = match nu {
        1 => 0.5 + t.atan() / PI,
        2 => 0.5 + t / (2.0 * (2.0 + t * t).sqrt()),
        3 => {
            let u = t / 3.0_f64.sqrt();
            0.5 + (u / (1.0 + u * u) + u.atan()) / PI
        }
        _ => unreachable!("no closed form kept for {nu} degrees of freedom"),
    };
    1.0 - cdf
}

#[test]
fn it_matches_the_closed_forms() {
    for nu in [1u32, 2, 3] {
        for t in [-4.0, -1.5, -0.25, 0.0, 0.25, 1.5, 4.0, 12.0] {
            let got = lccdf(t, nu as f64, 0.0, 1.0);
            let want = tail(t, nu).ln();
            assert!(
                (got - want).abs() < 1e-12 * want.abs().max(1.0),
                "nu={nu} t={t}: {got} vs {want}"
            );
        }
    }
}

#[test]
fn location_and_scale_shift_and_stretch_it() {
    // P(T > y) with mu, sigma is P(T > (y - mu) / sigma) standardised
    for (y, mu, sigma) in [(3.0, 1.0, 2.0), (-5.0, 2.0, 0.5), (0.0, 0.0, 10.0)] {
        let got = lccdf(y, 3.0, mu, sigma);
        let want = tail((y - mu) / sigma, 3).ln();
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }
}

/// The half-Student-t normaliser every model in the sweep asks for.
#[test]
fn the_median_tail_is_a_half() {
    assert!((lccdf(0.0, 3.0, 0.0, 10.0) - 0.5_f64.ln()).abs() < 1e-15);
}

/// `d/dy log(1 - F(y)) = -f(y) / (1 - F(y))`, checked against a difference of
/// the value function rather than against the formula it was derived from.
#[test]
fn the_gradient_matches_a_finite_difference() {
    let src = "parameters { real a; } model { target += student_t_lccdf(a | 4, 1, 2); }";
    let model = Model::parse_and_load(src, Env::new()).unwrap();
    for at in [-2.0, 0.0, 0.7, 3.0] {
        let (_, g) = model.log_prob_grad(&[at]).unwrap();
        let h = 1e-6;
        let up = model.log_prob_grad(&[at + h]).unwrap().0;
        let down = model.log_prob_grad(&[at - h]).unwrap().0;
        let fd = (up - down) / (2.0 * h);
        assert!(
            (g[0] - fd).abs() < 1e-6 * fd.abs().max(1.0),
            "at {at}: {} vs {fd}",
            g[0]
        );
    }
}

#[test]
fn degrees_of_freedom_may_not_depend_on_a_parameter() {
    let src = "parameters { real a; } model { target += student_t_lccdf(0 | a, 0, 1); }";
    let msg = match Model::parse_and_load(src, Env::new())
        .unwrap()
        .log_prob_grad(&[3.0])
    {
        Err(e) => e.to_string(),
        Ok(v) => panic!("expected an error, got {v:?}"),
    };
    assert!(msg.contains("degrees of freedom"), "{msg}");
}
