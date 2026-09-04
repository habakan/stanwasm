//! Indexing by an array of positions — `phi[node1]`. The result is as long as
//! the index, not as the thing indexed.

use stanwasm_runtime::{Env, Model, Val};

fn ints(xs: &[f64]) -> Val {
    Val::Vec(xs.iter().map(|v| Val::Num(*v)).collect())
}

fn data() -> Env {
    let mut d = Env::new();
    d.set_vector("v", &[10.0, 20.0, 30.0, 40.0]);
    d.set("idx", ints(&[3.0, 1.0, 1.0]));
    d.set(
        "M",
        Val::Vec(
            (0..3)
                .map(|i| Val::Row((0..2).map(|j| Val::Num((10 * i + j) as f64)).collect()))
                .collect(),
        ),
    );
    d
}

const HEAD: &str = "data { vector[4] v; array[3] int idx; matrix[3, 2] M; } \
                    parameters { real a; } model { ";

fn lp(body: &str) -> f64 {
    let src = format!("{HEAD}{body} }}");
    Model::parse_and_load(&src, data())
        .unwrap()
        .log_prob_grad(&[1.0])
        .unwrap()
        .0
}

fn err(body: &str) -> String {
    let src = format!("{HEAD}{body} }}");
    match Model::parse_and_load(&src, data())
        .unwrap()
        .log_prob_grad(&[1.0])
    {
        Err(e) => e.to_string(),
        Ok(v) => panic!("expected an error, got {v:?}"),
    }
}

#[test]
fn a_gather_is_as_long_as_its_index() {
    // v[idx] = [30, 10, 10]
    assert!((lp("target += a * sum(v[idx]);") - 50.0).abs() < 1e-12);
    assert!((lp("target += a * num_elements(v[idx]);") - 3.0).abs() < 1e-12);
}

#[test]
fn a_gather_composes_with_element_wise_operators() {
    // v[idx] .* v[idx] = [900, 100, 100]
    assert!((lp("target += a * sum(v[idx] .* v[idx]);") - 1100.0).abs() < 1e-12);
}

#[test]
fn a_matrix_gathers_whole_rows() {
    // M[idx] takes rows 3, 1, 1; each row keeps its orientation
    assert!((lp("target += a * rows(M[idx]);") - 3.0).abs() < 1e-12);
    assert!((lp("target += a * M[idx][1, 2];") - 21.0).abs() < 1e-12);
}

#[test]
fn a_gather_mixes_with_a_plain_index() {
    // column 2 of rows 3, 1, 1
    assert!((lp("target += a * sum(M[idx, 2]);") - 23.0).abs() < 1e-12);
}

#[test]
fn a_gather_is_an_assignment_target() {
    assert!(
        (lp("vector[4] u = v; u[idx] = rep_vector(0, 3); target += a * sum(u);") - 60.0).abs()
            < 1e-12
    );
    // a scalar fills every gathered position
    assert!((lp("vector[4] u = v; u[idx] = 0; target += a * sum(u);") - 60.0).abs() < 1e-12);
}

#[test]
fn a_gather_past_the_end_is_an_error() {
    assert!(err("target += a * sum(v[idx] + v[5]);").contains("index"));
}

#[test]
fn the_gradient_reaches_every_gathered_position() {
    let src =
        "data { array[3] int idx; } parameters { vector[4] b; } model { target += sum(b[idx]); }";
    let mut d = Env::new();
    d.set("idx", ints(&[3.0, 1.0, 1.0]));
    let g = Model::parse_and_load(src, d)
        .unwrap()
        .log_prob_grad(&[1.0, 2.0, 3.0, 4.0])
        .unwrap()
        .1;
    // b[1] is read twice, b[3] once, b[2] and b[4] not at all
    assert_eq!(g, vec![2.0, 0.0, 1.0, 0.0]);
}

/// Indices inside one bracket compose; a second bracket indexes what the first
/// produced. They differ only once an index is an array.
#[test]
fn a_second_bracket_indexes_the_gathered_result() {
    // M[idx, 2] is column 2 of rows 3, 1, 1 — a length-3 vector
    assert!((lp("target += a * sum(M[idx, 2]);") - 23.0).abs() < 1e-12);
    // M[idx][1, 2] is one entry of the gathered 3x2 matrix
    assert!((lp("target += a * M[idx][1, 2];") - 21.0).abs() < 1e-12);
}
