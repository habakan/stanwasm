//! Ternary `?:`, elementwise operators, and the container utilities. Each closed a
//! gap found while writing a Maxwell-constrained magnetic-field GP.

use stanwasm_runtime::{Env, Model, Val};

fn value_of(body: &str, data: Env) -> f64 {
    let src = format!("parameters {{ real a; }}\nmodel {{ {body} }}");
    Model::parse_and_load(&src, data)
        .unwrap()
        .log_prob_grad(&[0.0])
        .unwrap()
        .0
}

fn err_of(body: &str, data: Env) -> String {
    let src = format!("parameters {{ real a; }}\nmodel {{ {body} }}");
    match Model::parse_and_load(&src, data).map_err(|e| e.to_string()) {
        Err(e) => e,
        Ok(m) => m
            .log_prob_grad(&[0.0])
            .map(|_| String::new())
            .unwrap_err()
            .to_string(),
    }
}

fn vec_data(name: &str, xs: &[f64]) -> Env {
    let mut d = Env::new();
    d.set_vector(name, xs);
    d
}

#[test]
fn ternary_picks_a_branch() {
    assert!((value_of("target += 1 > 0 ? 7 : 9;", Env::new()) - 7.0).abs() < 1e-12);
    assert!((value_of("target += 1 < 0 ? 7 : 9;", Env::new()) - 9.0).abs() < 1e-12);
}

#[test]
fn ternary_is_right_associative_and_binds_loosest() {
    // `1 + 1 ? ...` must read as `(1 + 1) ? ...`, and the nested one groups right.
    assert!((value_of("target += 1 + 1 ? 5 : 6;", Env::new()) - 5.0).abs() < 1e-12);
    let v = value_of("target += 0 ? 1 : 0 ? 2 : 3;", Env::new());
    assert!((v - 3.0).abs() < 1e-12, "{v}");
}

#[test]
fn ternary_evaluates_only_the_taken_branch() {
    // The other branch indexes out of bounds, so evaluating it would error.
    let v = value_of(
        "vector[2] y; y[1] = 4; y[2] = 5; target += 1 ? y[1] : y[9];",
        Env::new(),
    );
    assert!((v - 4.0).abs() < 1e-12, "{v}");
}

#[test]
fn elementwise_operators_work_on_vectors() {
    let mut d = vec_data("x", &[1.0, 2.0, 4.0]);
    d.set_vector("z", &[2.0, 4.0, 8.0]);
    assert!((value_of("target += sum(x .* z);", d.clone()) - 42.0).abs() < 1e-12);
    assert!((value_of("target += sum(z ./ x);", d.clone()) - 6.0).abs() < 1e-12);
    assert!((value_of("target += sum(x .^ 2);", d) - 21.0).abs() < 1e-12);
}

#[test]
fn elementwise_still_rejects_mismatched_lengths() {
    let mut d = vec_data("x", &[1.0, 2.0, 3.0]);
    d.set_vector("z", &[1.0, 2.0]);
    assert!(err_of("target += sum(x .* z);", d).contains("shape mismatch"));
}

#[test]
fn size_and_rep_vector() {
    let d = vec_data("x", &[1.0, 2.0, 3.0]);
    assert!((value_of("target += size(x);", d.clone()) - 3.0).abs() < 1e-12);
    assert!((value_of("target += num_elements(x);", d) - 3.0).abs() < 1e-12);
    assert!((value_of("target += sum(rep_vector(2.5, 4));", Env::new()) - 10.0).abs() < 1e-12);
}

#[test]
fn rows_cols_and_rep_matrix() {
    let mut d = Env::new();
    d.set(
        "M",
        Val::Vec(vec![
            Val::Vec(vec![Val::Num(1.0), Val::Num(2.0), Val::Num(3.0)]),
            Val::Vec(vec![Val::Num(4.0), Val::Num(5.0), Val::Num(6.0)]),
        ]),
    );
    assert!((value_of("target += rows(M);", d.clone()) - 2.0).abs() < 1e-12);
    assert!((value_of("target += cols(M);", d) - 3.0).abs() < 1e-12);
    let v = value_of(
        "matrix[2,3] R = rep_matrix(1.5, 2, 3); target += sum(R[1]) + sum(R[2]);",
        Env::new(),
    );
    assert!((v - 9.0).abs() < 1e-12, "{v}");
}

#[test]
fn dot_product_sums_the_products() {
    let mut d = vec_data("x", &[1.0, 2.0, 3.0]);
    d.set_vector("z", &[4.0, 5.0, 6.0]);
    assert!((value_of("target += dot_product(x, z);", d.clone()) - 32.0).abs() < 1e-12);
    d.set_vector("z", &[4.0, 5.0]);
    assert!(err_of("target += dot_product(x, z);", d).contains("shape mismatch"));
}

/// Stan scopes a local declaration by wrapping it in a block, which is how a
/// forward-filtering loop keeps its working array out of the enclosing block.
#[test]
fn a_bare_block_scopes_what_it_declares() {
    let body = "{ real tmp = 3.0; target += tmp; } target += 10.0;";
    assert!((value_of(body, Env::new()) - 13.0).abs() < 1e-12);

    // And what it declared is gone afterwards.
    let leaked = "{ real tmp = 3.0; target += tmp; } target += tmp;";
    assert!(
        err_of(leaked, Env::new()).contains("tmp"),
        "{}",
        err_of(leaked, Env::new())
    );
}

#[test]
fn diag_matrix_puts_the_vector_on_the_diagonal() {
    let mut d = Env::new();
    d.set_vector("v", &[2.0, 5.0]);
    // Row 1 is [2, 0] and row 2 is [0, 5], so this sums to 7 and the corners to 0.
    let body = "target += sum(diag_matrix(v)[1]) + sum(diag_matrix(v)[2]);";
    assert!((value_of(body, d.clone()) - 7.0).abs() < 1e-12);
    assert!(value_of("target += diag_matrix(v)[1, 2];", d).abs() < 1e-12);
}
