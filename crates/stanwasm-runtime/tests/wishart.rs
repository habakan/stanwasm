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

// ---- lkj_corr -------------------------------------------------------------

/// `lkj_corr(R | η)` and `lkj_corr_cholesky(chol(R) | η)` describe the same
/// distribution seen on the matrix and on its factor, so they differ by the
/// change of variables `R = L Lᵀ`. Whatever the shared constant is, the two
/// have to move together — which is what this pins.
#[test]
fn lkj_on_the_matrix_and_on_its_factor_agree_up_to_the_change_of_variables() {
    for rho in [-0.6_f64, -0.1, 0.25, 0.8] {
        for eta in [1.0_f64, 2.0, 3.5] {
            let l21 = rho;
            let l22 = (1.0 - rho * rho).sqrt();
            let src = format!(
                "data {{ matrix[2,2] R; matrix[2,2] L; }}
                 parameters {{ real z; }}
                 model {{ target += lkj_corr_lpdf(R | {eta})
                                  - lkj_corr_cholesky_lpdf(L | {eta}) + 0 * z; }}"
            );
            let m = |a: f64, b: f64, c: f64, d: f64| {
                Val::Vec(vec![
                    Val::Row(vec![Val::Num(a), Val::Num(b)]),
                    Val::Row(vec![Val::Num(c), Val::Num(d)]),
                ])
            };
            let mut env = Env::new();
            env.set("R", m(1.0, rho, rho, 1.0));
            env.set("L", m(1.0, 0.0, l21, l22));
            let (got, _) = Model::parse_and_load(&src, env)
                .unwrap()
                .log_prob_grad(&[0.0])
                .unwrap();
            // lkj_corr uses log|R| = 2 log L22; the Cholesky form weights the
            // diagonal by (K-1-k) + 2η-2, which at K=2 is (2η-2) log L22 for k=1.
            let want = (eta - 1.0) * 2.0 * l22.ln() - (2.0 * eta - 2.0) * l22.ln();
            assert!(
                (got - want).abs() < 1e-12,
                "rho={rho} eta={eta}: {got} != {want}"
            );
        }
    }
}

/// At η = 1 the density is flat over correlation matrices, so every R gives the
/// same value — the check that the exponent sits on `log|R|` rather than on
/// something that happens to agree at one point. The shared value is the
/// normalising constant, not zero.
#[test]
fn lkj_corr_is_flat_at_eta_one() {
    let at = |rho: f64| {
        let src = "data { matrix[2,2] R; } parameters { real z; }
                   model { target += lkj_corr_lpdf(R | 1.0) + 0 * z; }";
        let mut env = Env::new();
        env.set(
            "R",
            Val::Vec(vec![
                Val::Row(vec![Val::Num(1.0), Val::Num(rho)]),
                Val::Row(vec![Val::Num(rho), Val::Num(1.0)]),
            ]),
        );
        Model::parse_and_load(src, env)
            .unwrap()
            .log_prob_grad(&[0.0])
            .unwrap()
            .0
    };
    let first = at(-0.9);
    for rho in [-0.3_f64, 0.0, 0.5, 0.95] {
        assert!(
            (at(rho) - first).abs() < 1e-12,
            "at rho={rho} it is {}, at -0.9 it was {first}",
            at(rho)
        );
    }
}

#[test]
fn lkj_corr_refuses_a_variate_that_is_not_a_matrix() {
    let src = "data { vector[2] v; } parameters { real z; }
               model { target += lkj_corr_lpdf(v | 2.0) + 0 * z; }";
    let mut env = Env::new();
    env.set_vector("v", &[1.0, 0.5]);
    let e = Model::parse_and_load(src, env)
        .unwrap()
        .log_prob_grad(&[0.0])
        .unwrap_err()
        .to_string();
    assert!(e.contains("lkj_corr"), "{e}");
}

/// Values from CmdStan 2.39.0's `log_prob` method, at `R` and `L` fixed as
/// data and `eta` a parameter — the case where the normalising constant is not
/// a constant, and where dropping it used to make the density wrong by an
/// amount that moved with the point.
#[test]
fn lkj_matches_a_reference_implementation_with_eta_a_parameter() {
    let m = |rows: Vec<Vec<f64>>| {
        Val::Vec(
            rows.into_iter()
                .map(|r| Val::Row(r.into_iter().map(Val::Num).collect()))
                .collect(),
        )
    };
    let run = |src: &str, name: &str, mat: Val, u: f64| {
        let mut env = Env::new();
        env.set(name, mat);
        Model::parse_and_load(src, env)
            .unwrap()
            .log_prob_grad(&[u])
            .unwrap()
            .0
    };

    let r2 = m(vec![vec![1.0, 0.4], vec![0.4, 1.0]]);
    let src2 = "data { matrix[2,2] R; } parameters { real<lower=1> eta; }
                model { R ~ lkj_corr(eta); }";
    for (u, want) in [
        (-0.5, -1.017745071486),
        (0.0, -0.462035459597),
        (0.5, 0.080290215176),
        (1.0, 0.576805805269),
    ] {
        let got = run(src2, "R", r2.clone(), u);
        assert!((got - want).abs() < 1e-10, "K=2 at u={u}: {got} != {want}");
    }

    let r3 = m(vec![
        vec![1.0, 0.3, -0.2],
        vec![0.3, 1.0, 0.15],
        vec![-0.2, 0.15, 1.0],
    ]);
    let src3 = "data { matrix[3,3] R; } parameters { real<lower=1> eta; }
                model { R ~ lkj_corr(eta); }";
    for (u, want) in [
        (-0.5, -1.542691498779),
        (0.0, -0.802415507479),
        (0.7, 0.302541164563),
    ] {
        let got = run(src3, "R", r3.clone(), u);
        assert!((got - want).abs() < 1e-10, "K=3 at u={u}: {got} != {want}");
    }

    // The Cholesky form carries the same constant, checked at the factor of the
    // same K=3 matrix.
    let l3 = m(vec![
        vec![1.0, 0.0, 0.0],
        vec![0.3, 0.953_939_201_416_945_6, 0.0],
        vec![-0.2, 0.220_139_816_015_670_9, 0.954_745_233_631_397_3],
    ]);
    let srcl = "data { matrix[3,3] L; } parameters { real<lower=1> eta; }
                model { L ~ lkj_corr_cholesky(eta); }";
    for (u, want) in [
        (-0.5, -1.589846838514),
        (0.0, -0.849570847214),
        (0.7, 0.255385824828),
    ] {
        let got = run(srcl, "L", l3.clone(), u);
        assert!((got - want).abs() < 1e-9, "chol K=3 at u={u}: {got} != {want}");
    }
}
