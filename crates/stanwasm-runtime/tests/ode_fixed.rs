//! `ode_rk4_fixed`, against a system whose solution is known in closed form.
//!
//! The point of a fixed step count is that the recorded graph does not depend on
//! the parameters, so the checks are: the value is right to the accuracy the
//! step count buys, the error falls like h⁴, and the gradient through the
//! solver is the derivative of the closed form rather than of the discretisation.

use stanwasm_runtime::{Env, Model};

/// dy/dt = -k y, y(0) = 1, so y(t) = exp(-k t).
fn decay(n_steps: u32) -> String {
    format!(
        "functions {{
           array[] real f(real t, array[] real y, array[] real theta,
                          array[] real x_r, array[] int x_i) {{
             return {{ -theta[1] * y[1] }};
           }}
         }}
         parameters {{ real k; }}
         model {{
           array[1] real y0 = {{ 1.0 }};
           array[2] real ts = {{ 0.5, 1.0 }};
           array[2, 1] real y = ode_rk4_fixed(f, y0, 0.0, ts, {{ k }},
                                              rep_array(0.0, 0), rep_array(0, 0),
                                              {n_steps});
           target += y[2, 1];
         }}"
    )
}

fn lp_grad(src: &str, k: f64) -> (f64, f64) {
    let (v, g) = Model::parse_and_load(src, Env::new())
        .unwrap()
        .log_prob_grad(&[k])
        .unwrap();
    (v, g[0])
}

#[test]
fn a_linear_decay_matches_its_closed_form() {
    let k = 0.7_f64;
    let (v, _) = lp_grad(&decay(64), k);
    let want = (-k).exp(); // y(1) = exp(-k)
    assert!((v - want).abs() < 1e-9, "got {v}, want {want}");
}

/// RK4 is fourth order, so halving the step should cut the error about 16x.
/// Anything far off that means the stages are wired wrong, not merely coarse.
#[test]
fn the_error_falls_like_the_fourth_power_of_the_step() {
    let k = 0.7_f64;
    let want = (-k).exp();
    let err = |n| (lp_grad(&decay(n), k).0 - want).abs();
    let (coarse, fine) = (err(2), err(4));
    let ratio = coarse / fine;
    assert!(coarse > fine, "refining did not help: {coarse} then {fine}");
    assert!(
        (8.0..32.0).contains(&ratio),
        "halving the step changed the error {ratio:.1}x, not about 16x"
    );
}

/// d/dk exp(-k) = -exp(-k). The gradient has to come through the solver.
#[test]
fn the_gradient_runs_through_the_solver() {
    let k = 0.7_f64;
    let (_, g) = lp_grad(&decay(64), k);
    let want = -(-k).exp();
    assert!((g - want).abs() < 1e-8, "got {g}, want {want}");
}

/// The step count is data, so the recorded graph is the same whatever the
/// parameter is — which is the whole reason this function exists.
#[test]
fn the_graph_does_not_depend_on_the_parameter() {
    use stanwasm_autodiff::Tape;
    let m = Model::parse_and_load(&decay(8), Env::new()).unwrap();
    let trace = |k: f64| {
        let mut tape = Tape::new();
        let leaves: Vec<u32> = vec![tape.new_var(k)];
        m.trace_forward(&mut tape, &leaves, true).unwrap();
        tape.ops().to_vec()
    };
    assert_eq!(trace(0.2), trace(5.0), "the recorded ops differ by parameter");
}

#[test]
fn a_bad_step_count_and_a_missing_system_are_named() {
    let e = Model::parse_and_load(&decay(0), Env::new())
        .unwrap()
        .log_prob_grad(&[0.7])
        .unwrap_err()
        .to_string();
    assert!(e.contains("n_steps must be at least 1"), "{e}");

    let src = decay(4).replace("ode_rk4_fixed(f,", "ode_rk4_fixed(nope,");
    let e = Model::parse_and_load(&src, Env::new())
        .unwrap()
        .log_prob_grad(&[0.7])
        .unwrap_err()
        .to_string();
    assert!(e.contains("nope"), "{e}");
}

/// The adaptive integrators still refuse, and say why.
#[test]
fn the_adaptive_integrators_still_refuse() {
    let src = decay(4).replace("ode_rk4_fixed(f,", "integrate_ode_rk45(f,");
    let e = Model::parse_and_load(&src, Env::new())
        .unwrap()
        .log_prob_grad(&[0.7])
        .unwrap_err()
        .to_string();
    assert!(e.contains("integrate_ode_rk45"), "{e}");
}
