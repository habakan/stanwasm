//! `cov_matrix`, `cholesky_factor_cov` and `unit_vector`. The log Jacobian is checked
//! against the formula in the Stan reference manual, not just against itself: a wrong
//! constant is consistent with its own gradient and still samples the wrong posterior.

use stanwasm_runtime::{Env, Model};

/// With no sampling statement, log_prob is exactly the constraint's log Jacobian.
fn jacobian_only(decl: &str, params: &[f64]) -> (f64, Vec<f64>) {
    let src = format!("parameters {{ {decl} }}\nmodel {{ }}");
    Model::parse_and_load(&src, Env::new())
        .unwrap()
        .log_prob_grad(params)
        .unwrap()
}

fn finite_diff_matches(decl: &str, params: &[f64]) {
    let src = format!("parameters {{ {decl} }}\nmodel {{ }}");
    let model = Model::parse_and_load(&src, Env::new()).unwrap();
    let (_, grads) = model.log_prob_grad(params).unwrap();
    let h = 1e-5;
    for i in 0..params.len() {
        let (mut hi, mut lo) = (params.to_vec(), params.to_vec());
        hi[i] += h;
        lo[i] -= h;
        let fd =
            (model.log_prob_grad(&hi).unwrap().0 - model.log_prob_grad(&lo).unwrap().0) / (2.0 * h);
        assert!(
            (fd - grads[i]).abs() < 1e-4,
            "{decl} param[{i}]: ad={}, fd={fd}",
            grads[i]
        );
    }
}

#[test]
fn cholesky_factor_cov_jacobian_is_the_sum_of_the_raw_diagonal() {
    // L(m,m) = exp(y_mm), so log|J| = Σ y_nn. Raw order is row-major with each row's
    // diagonal last: [d0, o10, d1].
    let (d0, o10, d1) = (0.3, -0.7, 0.2);
    let (lp, _) = jacobian_only("cholesky_factor_cov[2] L;", &[d0, o10, d1]);
    assert!(
        (lp - (d0 + d1)).abs() < 1e-12,
        "lp = {lp}, expected {}",
        d0 + d1
    );
}

#[test]
fn cov_matrix_jacobian_matches_the_reference_formula() {
    // K log 2 + Σ_{k=1..K} (K − k + 2) log L_kk, and log L_kk is the raw diagonal.
    // This is the whole Jacobian: adding the exp() term again would give 4·d0 + 3·d1.
    let (d0, o10, d1) = (0.3, -0.7, 0.2);
    let expected = 2.0 * std::f64::consts::LN_2 + 3.0 * d0 + 2.0 * d1;
    let (lp, _) = jacobian_only("cov_matrix[2] S;", &[d0, o10, d1]);
    assert!(
        (lp - expected).abs() < 1e-12,
        "lp = {lp}, expected {expected}"
    );
}

#[test]
fn unit_vector_jacobian_is_the_standard_normal_kernel() {
    // x = y/‖y‖ has a singular Jacobian; Stan adds −½ yᵀy, which also pins the radius.
    let y = [0.6, -0.8, 0.5];
    let expected = -0.5 * y.iter().map(|v| v * v).sum::<f64>();
    let (lp, _) = jacobian_only("unit_vector[3] u;", &y);
    assert!(
        (lp - expected).abs() < 1e-12,
        "lp = {lp}, expected {expected}"
    );
}

#[test]
fn gradients_agree_with_finite_differences() {
    finite_diff_matches("cholesky_factor_cov[2] L;", &[0.3, -0.7, 0.2]);
    finite_diff_matches("cov_matrix[2] S;", &[0.3, -0.7, 0.2]);
    finite_diff_matches("unit_vector[3] u;", &[0.6, -0.8, 0.5]);
}

#[test]
fn cov_matrix_is_the_product_of_its_cholesky_factor() {
    // Σ = L Lᵀ with L = [[e^0.3, 0], [-0.7, e^0.2]], checked through the constrained
    // draw rather than by reading the transform's own output back.
    let src = "parameters { cov_matrix[2] S; }\nmodel { }";
    let model = Model::parse_and_load(src, Env::new()).unwrap();
    let s = model.constrained_draw(&[0.3, -0.7, 0.2]).unwrap();
    let (l00, l10, l11) = (0.3_f64.exp(), -0.7_f64, 0.2_f64.exp());
    let expect = [l00 * l00, l00 * l10, l10 * l00, l10 * l10 + l11 * l11];
    for (got, want) in s.iter().zip(expect) {
        assert!((got - want).abs() < 1e-12, "got {s:?}, want {expect:?}");
    }
}

#[test]
fn unit_vector_has_length_one() {
    let src = "parameters { unit_vector[3] u; }\nmodel { }";
    let model = Model::parse_and_load(src, Env::new()).unwrap();
    let u = model.constrained_draw(&[0.6, -0.8, 0.5]).unwrap();
    let norm: f64 = u.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-12, "‖u‖ = {norm}, u = {u:?}");
}

#[test]
fn parameter_names_line_up_with_the_constrained_draw() {
    // `param_names` used to be sized from the *unconstrained* dimension, so a
    // cov_matrix[2] got three labels for four values and every name after it in the
    // model was reported against its neighbour's number.
    for (decl, raw) in [
        ("cov_matrix[2] S;", vec![0.3, -0.7, 0.2]),
        ("cholesky_factor_cov[2] L;", vec![0.3, -0.7, 0.2]),
        ("cholesky_factor_corr[2] C;", vec![0.4]),
        ("simplex[3] p;", vec![0.1, -0.2]),
        ("unit_vector[3] u;", vec![0.6, -0.8, 0.5]),
    ] {
        let src = format!("parameters {{ {decl} }}\nmodel {{ }}");
        let model = Model::parse_and_load(&src, Env::new()).unwrap();
        let drawn = model.constrained_draw(&raw).unwrap();
        assert_eq!(
            model.param_names().len(),
            drawn.len(),
            "{decl}: {:?} vs {} values",
            model.param_names(),
            drawn.len()
        );
    }
}

#[test]
fn corr_matrix_shares_the_cholesky_factor_corr_jacobian() {
    // x = L Lᵀ contributes nothing further: Stan's `read_corr_matrix` adds only what
    // `read_corr_L` already did, which is not true of cov_matrix's L Lᵀ.
    let raw = [0.4, -0.2, 0.7];
    let (corr, _) = jacobian_only("corr_matrix[3] R;", &raw);
    let (chol, _) = jacobian_only("cholesky_factor_corr[3] L;", &raw);
    assert!((corr - chol).abs() < 1e-12, "corr={corr}, chol={chol}");
}

#[test]
fn corr_matrix_has_a_unit_diagonal_and_is_symmetric() {
    let src = "parameters { corr_matrix[3] R; }\nmodel { }";
    let model = Model::parse_and_load(src, Env::new()).unwrap();
    let r = model.constrained_draw(&[0.4, -0.2, 0.7]).unwrap();
    assert_eq!(r.len(), 9, "{r:?}");
    for i in 0..3 {
        assert!(
            (r[i * 3 + i] - 1.0).abs() < 1e-12,
            "diagonal {i} = {}",
            r[i * 3 + i]
        );
        for j in 0..3 {
            assert!(
                (r[i * 3 + j] - r[j * 3 + i]).abs() < 1e-12,
                "not symmetric: {r:?}"
            );
            assert!(r[i * 3 + j].abs() <= 1.0 + 1e-12, "|r| > 1: {r:?}");
        }
    }
}

#[test]
fn corr_matrix_gradients_agree_with_finite_differences() {
    finite_diff_matches("corr_matrix[3] R;", &[0.4, -0.2, 0.7]);
}

/// A bound on a matrix declaration transforms every entry, the way a bound on
/// a vector does.
#[test]
fn a_matrix_parameter_can_carry_an_element_bound() {
    let src = "parameters { matrix<lower=0>[2, 2] M; } model { target += sum(M[1]) + sum(M[2]); }";
    let m = Model::parse_and_load(src, Env::new()).unwrap();
    assert_eq!(m.n_params(), 4);
    let (lp, _) = m.log_prob_grad(&[0.0, 0.0, 0.0, 0.0]).unwrap();
    // every entry is exp(0) = 1, and the Jacobian contributes ∑ raw = 0
    assert!((lp - 4.0).abs() < 1e-12, "{lp}");

    let draw = m.constrained_draw(&[0.0, 0.0, 0.0, 0.0]).unwrap();
    assert_eq!(draw, vec![1.0; 4]);
}
