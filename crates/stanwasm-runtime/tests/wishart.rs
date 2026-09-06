//! `wishart` and `inv_wishart`, against the closed forms they reduce to.
//!
//! Checked against the shapes where the answer is known rather than against
//! themselves: a 1×1 Wishart is a gamma, a 1×1 inverse Wishart is an inverse
//! gamma, and the two are each other's change of variables. A density that is
//! only self-consistent samples a wrong posterior without saying so.

use stanwasm_runtime::{Env, Model, Val};

/// `target += <dist>(W | nu, S)` at a fixed W, so log_prob is that density.
fn lp(dist: &str, w: &[[f64; 2]], nu: f64, s: &[[f64; 2]], k: usize) -> f64 {
    let mat = |m: &[[f64; 2]]| {
        Val::Vec(
            (0..k)
                .map(|i| Val::Row((0..k).map(|j| Val::Num(m[i][j])).collect()))
                .collect(),
        )
    };
    let src = format!(
        "data {{ matrix[{k},{k}] W; matrix[{k},{k}] S; }}
         parameters {{ real z; }}
         model {{ target += {dist}_lpdf(W | {nu}, S) + 0 * z; }}"
    );
    let mut env = Env::new();
    env.set("W", mat(w));
    env.set("S", mat(s));
    Model::parse_and_load(&src, env)
        .unwrap()
        .log_prob_grad(&[0.0])
        .unwrap()
        .0
}

/// Stirling with the recurrence pushed far enough that the series, not this,
/// is what the comparisons below rest on.
fn lgamma(x: f64) -> f64 {
    let mut a = x;
    let mut acc = 0.0;
    while a < 20.0 {
        acc -= a.ln();
        a += 1.0;
    }
    let inv = 1.0 / a;
    let inv2 = inv * inv;
    acc + (a - 0.5) * a.ln() - a
        + 0.5 * (2.0 * std::f64::consts::PI).ln()
        + inv
            * (1.0 / 12.0
                - inv2 * (1.0 / 360.0 - inv2 * (1.0 / 1260.0 - inv2 / 1680.0)))
}

/// A 1×1 Wishart(nu, s) is Gamma(shape = nu/2, rate = 1/(2s)).
#[test]
fn a_one_by_one_wishart_is_a_gamma() {
    for (w, nu, s) in [(2.0, 3.0, 1.0), (0.4, 5.0, 2.5), (7.1, 4.0, 0.3)] {
        let got = lp("wishart", &[[w, 0.0], [0.0, 0.0]], nu, &[[s, 0.0], [0.0, 0.0]], 1);
        let (shape, rate) = (nu / 2.0, 1.0 / (2.0 * s));
        let want = shape * rate.ln() + (shape - 1.0) * w.ln() - rate * w - lgamma(shape);
        assert!(
            (got - want).abs() < 1e-10,
            "wishart({w} | {nu}, {s}) = {got}, gamma says {want}"
        );
    }
}

/// A 1×1 inverse Wishart(nu, s) is InvGamma(shape = nu/2, scale = s/2).
#[test]
fn a_one_by_one_inv_wishart_is_an_inverse_gamma() {
    for (w, nu, s) in [(2.0, 3.0, 1.0), (0.4, 5.0, 2.5), (7.1, 4.0, 0.3)] {
        let got = lp("inv_wishart", &[[w, 0.0], [0.0, 0.0]], nu, &[[s, 0.0], [0.0, 0.0]], 1);
        let (shape, scale) = (nu / 2.0, s / 2.0);
        let want = shape * scale.ln() - (shape + 1.0) * w.ln() - scale / w - lgamma(shape);
        assert!(
            (got - want).abs() < 1e-10,
            "inv_wishart({w} | {nu}, {s}) = {got}, inv-gamma says {want}"
        );
    }
}

/// The 2×2 density written out longhand, which exercises the determinant, the
/// trace and the multivariate gamma together rather than one at a time.
#[test]
fn a_two_by_two_wishart_matches_the_written_out_density() {
    let w = [[2.0, 0.5], [0.5, 1.5]];
    let s = [[1.0, 0.2], [0.2, 0.8]];
    let nu = 4.0_f64;
    let got = lp("wishart", &w, nu, &s, 2);

    let det = |m: &[[f64; 2]]| m[0][0] * m[1][1] - m[0][1] * m[1][0];
    // tr(S⁻¹W) for 2×2, S⁻¹ = adj(S)/|S|
    let ds = det(&s);
    let inv_s = [[s[1][1] / ds, -s[0][1] / ds], [-s[1][0] / ds, s[0][0] / ds]];
    let tr = (0..2)
        .map(|i| (0..2).map(|j| inv_s[i][j] * w[j][i]).sum::<f64>())
        .sum::<f64>();
    let log_multigamma = 0.5 * std::f64::consts::PI.ln()
        + lgamma(nu / 2.0)
        + lgamma(nu / 2.0 - 0.5);
    let want = (nu - 3.0) / 2.0 * det(&w).ln() - 0.5 * tr
        - nu * std::f64::consts::LN_2
        - nu / 2.0 * ds.ln()
        - log_multigamma;
    assert!((got - want).abs() < 1e-10, "{got} != {want}");
}

/// `wishart` and `inv_wishart` are the same distribution seen through `W → W⁻¹`,
/// so their densities differ by the Jacobian `|W|^-(K+1)` — 2·log|W| times
/// (K+1)/2 at K = 2. Checking the pair together catches a constant that is
/// wrong in both.
#[test]
fn the_two_are_each_others_change_of_variables() {
    let w = [[2.0, 0.5], [0.5, 1.5]];
    let s = [[1.0, 0.2], [0.2, 0.8]];
    let nu = 5.0_f64;
    let det_w = w[0][0] * w[1][1] - w[0][1] * w[1][0];
    let inv_w = [
        [w[1][1] / det_w, -w[0][1] / det_w],
        [-w[1][0] / det_w, w[0][0] / det_w],
    ];
    // p_IW(W | nu, S) = p_W(W⁻¹ | nu, S⁻¹) · |W|^-(K+1)
    let det_s = s[0][0] * s[1][1] - s[0][1] * s[1][0];
    let inv_s = [
        [s[1][1] / det_s, -s[0][1] / det_s],
        [-s[1][0] / det_s, s[0][0] / det_s],
    ];
    let lhs = lp("inv_wishart", &w, nu, &s, 2);
    let rhs = lp("wishart", &inv_w, nu, &inv_s, 2) - 3.0 * det_w.ln();
    assert!((lhs - rhs).abs() < 1e-10, "{lhs} != {rhs}");
}

/// A variate that is not a covariance is an error rather than a NaN.
#[test]
fn a_variate_of_the_wrong_shape_is_refused() {
    let src = "data { vector[2] v; matrix[2,2] S; } parameters { real z; }
               model { target += wishart_lpdf(v | 4.0, S) + 0 * z; }";
    let mut env = Env::new();
    env.set_vector("v", &[1.0, 2.0]);
    env.set(
        "S",
        Val::Vec(vec![
            Val::Row(vec![Val::Num(1.0), Val::Num(0.0)]),
            Val::Row(vec![Val::Num(0.0), Val::Num(1.0)]),
        ]),
    );
    let e = Model::parse_and_load(src, env)
        .unwrap()
        .log_prob_grad(&[0.0])
        .unwrap_err()
        .to_string();
    assert!(e.contains("wishart"), "{e}");
}
