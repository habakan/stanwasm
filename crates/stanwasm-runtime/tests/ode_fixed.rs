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

/// The integrator that is still not implemented refuses everywhere, including
/// on the fresh trace that `integrate_ode_rk45` now runs on.
#[test]
fn the_implicit_integrator_still_refuses() {
    let src = decay(4).replace("ode_rk4_fixed(f,", "integrate_ode_bdf(f,");
    let e = Model::parse_and_load(&src, Env::new())
        .unwrap()
        .log_prob_grad(&[0.7])
        .unwrap_err()
        .to_string();
    assert!(e.contains("integrate_ode_bdf"), "{e}");
}

// ---- the adaptive integrator, which only a fresh trace can run --------------

fn adaptive(tol: &str) -> String {
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
           array[2, 1] real y = integrate_ode_rk45(f, y0, 0.0, ts, {{ k }},
                                                   rep_array(0.0, 0), rep_array(0, 0){tol});
           target += y[2, 1];
         }}"
    )
}

#[test]
fn the_adaptive_integrator_matches_its_closed_form() {
    let k = 0.7_f64;
    let (v, g) = lp_grad(&adaptive(", 1e-10, 1e-10, 100000"), k);
    let want = (-k).exp();
    assert!((v - want).abs() < 1e-9, "got {v}, want {want}");
    assert!((g + want).abs() < 1e-8, "gradient {g}, want {}", -want);
}

/// A looser tolerance should cost accuracy — otherwise the control is not
/// reading the error estimate at all.
#[test]
fn the_tolerance_actually_controls_the_step() {
    let k = 0.7_f64;
    let want = (-k).exp();
    let err = |tol: &str| (lp_grad(&adaptive(tol), k).0 - want).abs();
    let loose = err(", 1e-3, 1e-3, 100000");
    let tight = err(", 1e-11, 1e-11, 100000");
    assert!(tight < loose, "tightening did nothing: {loose} then {tight}");
}

/// The step count is chosen from the values, so the recorded graph differs by
/// parameter. That is exactly why the replay path refuses it.
#[test]
fn the_adaptive_graph_moves_with_the_parameter_and_replay_refuses() {
    use stanwasm_autodiff::Tape;
    let src = adaptive(", 1e-8, 1e-8, 100000");
    let m = Model::parse_and_load(&src, Env::new()).unwrap();
    let ops = |k: f64, strict: bool| {
        let mut tape = Tape::new();
        let leaves = vec![tape.new_var(k)];
        m.trace_forward(&mut tape, &leaves, strict).map(|_| tape.len())
    };
    let a = ops(0.2, false).unwrap();
    let b = ops(9.0, false).unwrap();
    assert_ne!(a, b, "the adaptive solver took the same number of steps");

    let refused = ops(0.2, true).unwrap_err().to_string();
    assert!(refused.contains("integrate_ode_rk45"), "{refused}");
}
