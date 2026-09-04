//! Range indexing: a range keeps its dimension, an index drops it, and the two
//! mix in one `[...]`. Both reading and assigning walk the same path.

use stanwasm_runtime::{Env, Model, Val};

fn lp(src: &str, data: Env, at: &[f64]) -> f64 {
    Model::parse_and_load(src, data)
        .unwrap()
        .log_prob_grad(at)
        .unwrap()
        .0
}

fn err(src: &str, data: Env, at: &[f64]) -> String {
    match Model::parse_and_load(src, data).unwrap().log_prob_grad(at) {
        Err(e) => e.to_string(),
        Ok(v) => panic!("expected an error, got {v:?}"),
    }
}

/// W = [[1,2,3],[4,5,6],[7,8,9]], v = [1..5]
fn grid() -> Env {
    let mut d = Env::new();
    d.set(
        "W",
        Val::Vec(
            (0..3)
                .map(|i| Val::Vec((1..=3).map(|j| Val::Num((3 * i + j) as f64)).collect()))
                .collect(),
        ),
    );
    d.set_vector("v", &[1.0, 2.0, 3.0, 4.0, 5.0]);
    d
}

const HEAD: &str = "data { matrix[3, 3] W; vector[5] v; } parameters { real a; } model { ";

#[test]
fn a_column_is_a_range_over_the_rows() {
    // W[1:3, 2] = [2, 5, 8]
    let src = format!("{HEAD} target += a * sum(W[1:rows(W), 2]); }}");
    assert!((lp(&src, grid(), &[1.0]) - 15.0).abs() < 1e-12);
}

#[test]
fn a_row_slice_reads_along_one_row() {
    // W[3, 2:3] = [8, 9]
    let src = format!("{HEAD} target += a * sum(W[3, 2:3]); }}");
    assert!((lp(&src, grid(), &[1.0]) - 17.0).abs() < 1e-12);
}

#[test]
fn an_omitted_bound_is_the_containers_own_end() {
    let whole = format!("{HEAD} target += a * sum(W[ : , 1]); }}");
    assert!((lp(&whole, grid(), &[1.0]) - 12.0).abs() < 1e-12);

    let tail = format!("{HEAD} target += a * sum(v[3 : ]); }}");
    assert!((lp(&tail, grid(), &[1.0]) - 12.0).abs() < 1e-12);

    let head = format!("{HEAD} target += a * sum(v[ : 2]); }}");
    assert!((lp(&head, grid(), &[1.0]) - 3.0).abs() < 1e-12);
}

#[test]
fn a_range_keeps_its_dimension_and_an_index_drops_it() {
    // W[2:3, 1:2] is a 2x2 block; its [2,1] is W[3,1] = 7
    let src = format!("{HEAD} target += a * W[2:3, 1:2][2, 1]; }}");
    assert!((lp(&src, grid(), &[1.0]) - 7.0).abs() < 1e-12);
}

#[test]
fn a_slice_is_an_assignment_target() {
    let src = format!(
        "{HEAD} matrix[3, 3] Z = W; Z[1:2, 3] = rep_vector(0, 2); target += a * sum(Z[ : , 3]); }}"
    );
    // only W[3,3] = 9 survives
    assert!((lp(&src, grid(), &[1.0]) - 9.0).abs() < 1e-12);
}

#[test]
fn a_scalar_fills_the_whole_span() {
    let src = format!("{HEAD} vector[5] u = v; u[2:4] = 0; target += a * sum(u); }}");
    assert!((lp(&src, grid(), &[1.0]) - 6.0).abs() < 1e-12);
}

#[test]
fn a_right_hand_side_of_the_wrong_length_is_an_error() {
    let src = format!("{HEAD} vector[5] u = v; u[2:4] = v; target += a * sum(u); }}");
    let msg = err(&src, grid(), &[1.0]);
    assert!(msg.contains("vector[3]"), "{msg}");
}

#[test]
fn a_range_past_the_end_is_an_error() {
    let src = format!("{HEAD} target += a * sum(v[3:9]); }}");
    let msg = err(&src, grid(), &[1.0]);
    assert!(msg.contains("index"), "{msg}");
}

/// The gradient flows through a slice the same as through a plain index.
#[test]
fn a_slice_of_a_parameter_differentiates() {
    let src = "data { } parameters { vector[4] b; } model { target += sum(b[2:3]); }";
    let g = Model::parse_and_load(src, Env::new())
        .unwrap()
        .log_prob_grad(&[1.0, 2.0, 3.0, 4.0])
        .unwrap()
        .1;
    assert_eq!(g, vec![0.0, 1.0, 1.0, 0.0]);
}
