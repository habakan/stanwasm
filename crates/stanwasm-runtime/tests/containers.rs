//! Building and taking apart containers: the `[...]` literal, `append_row` /
//! `append_col`, and the matrix functions that go with them.

use stanwasm_runtime::{Env, Model, Val};

fn lp(src: &str, data: Env) -> f64 {
    Model::parse_and_load(src, data)
        .unwrap()
        .log_prob_grad(&[1.0])
        .unwrap()
        .0
}

fn err(src: &str, data: Env) -> String {
    match Model::parse_and_load(src, data)
        .unwrap()
        .log_prob_grad(&[1.0])
    {
        Err(e) => e.to_string(),
        Ok(v) => panic!("expected an error, got {v:?}"),
    }
}

fn matrix(rows: &[&[f64]]) -> Val {
    Val::Vec(
        rows.iter()
            .map(|r| Val::Vec(r.iter().map(|v| Val::Num(*v)).collect()))
            .collect(),
    )
}

fn data() -> Env {
    let mut d = Env::new();
    d.set_vector("u", &[1.0, 2.0, 3.0]);
    d.set_vector("v", &[4.0, 5.0]);
    d.set("M", matrix(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]));
    d.set("L", matrix(&[&[2.0, 0.0], &[3.0, 4.0]]));
    d
}

const HEAD: &str = "data { vector[3] u; vector[2] v; matrix[3, 2] M; matrix[2, 2] L; } \
     parameters { real a; } model { ";

#[test]
fn a_literal_of_scalars_is_a_row_vector() {
    // [1, 2, 3] * u is the inner product
    let src = format!("{HEAD} target += a * ([1, 2, 3] * u); }}");
    assert!((lp(&src, data()) - 14.0).abs() < 1e-12);
}

#[test]
fn a_literal_of_rows_is_a_matrix() {
    // [u', u']' is 3x2, so its [3, 2] is u[3]
    let src = format!("{HEAD} target += a * [u', u']'[3, 2]; }}");
    assert!((lp(&src, data()) - 3.0).abs() < 1e-12);
}

#[test]
fn a_ragged_literal_is_an_error() {
    let src = format!("{HEAD} target += a * rows([u', v']); }}");
    let msg = err(&src, data());
    assert!(
        msg.contains("row_vector") || msg.contains("vector["),
        "{msg}"
    );
}

#[test]
fn append_row_runs_two_columns_together() {
    let src = format!("{HEAD} target += a * sum(append_row(u, v)); }}");
    assert!((lp(&src, data()) - 15.0).abs() < 1e-12);
}

#[test]
fn append_row_stacks_two_rows_into_a_matrix() {
    // [u' ; v'] is ragged, so use two rows of M
    let src = format!("{HEAD} target += a * rows(append_row(M, M)); }}");
    assert!((lp(&src, data()) - 6.0).abs() < 1e-12);
}

#[test]
fn append_col_puts_a_column_beside_a_matrix() {
    let src = format!("{HEAD} target += a * cols(append_col(u, M)); }}");
    assert!((lp(&src, data()) - 3.0).abs() < 1e-12);

    let rows = format!("{HEAD} target += a * rows(append_col(u, M)); }}");
    assert!((lp(&rows, data()) - 3.0).abs() < 1e-12);
}

#[test]
fn append_col_keeps_two_rows_a_row() {
    // a row on the left of a column is still an inner product
    let src = format!("{HEAD} target += a * (append_col([1, 2], [3]) * u); }}");
    assert!((lp(&src, data()) - 14.0).abs() < 1e-12);
}

#[test]
fn sub_col_takes_part_of_one_column() {
    // rows 2..3 of column 2 of M is [4, 6]
    let src = format!("{HEAD} target += a * sum(sub_col(M, 2, 2, 2)); }}");
    assert!((lp(&src, data()) - 10.0).abs() < 1e-12);
}

#[test]
fn sub_col_past_the_end_is_an_error() {
    let src = format!("{HEAD} target += a * sum(sub_col(M, 2, 2, 3)); }}");
    assert!(err(&src, data()).contains("index"));
}

#[test]
fn quad_form_diag_scales_both_ways() {
    // diag(v) L diag(v), so [2,1] is v[2] * 3 * v[1] = 5 * 3 * 4
    let src = format!("{HEAD} target += a * quad_form_diag(L, v)[2, 1]; }}");
    assert!((lp(&src, data()) - 60.0).abs() < 1e-12);
}

#[test]
fn multiply_lower_tri_self_transpose_reads_only_the_lower_triangle() {
    // L = [[2, 0], [3, 4]]; L Lᵀ = [[4, 6], [6, 25]]
    let src = format!("{HEAD} target += a * multiply_lower_tri_self_transpose(L)[2, 2]; }}");
    assert!((lp(&src, data()) - 25.0).abs() < 1e-12);

    // the upper triangle is ignored even when it isn't zero
    let mut d = data();
    d.set("L", matrix(&[&[2.0, 9.0], &[3.0, 4.0]]));
    assert!((lp(&src, d) - 25.0).abs() < 1e-12);
}

#[test]
fn dims_reports_each_dimension() {
    let src = format!("{HEAD} target += a * (dims(M)[1] * 10 + dims(M)[2]); }}");
    assert!((lp(&src, data()) - 32.0).abs() < 1e-12);
}

#[test]
fn negative_infinity_is_what_it_says() {
    let src = format!("{HEAD} target += a * 0 + negative_infinity(); }}");
    assert_eq!(lp(&src, data()), f64::NEG_INFINITY);
}

#[test]
fn tail_takes_the_last_entries() {
    let src = format!("{HEAD} target += a * sum(tail(u, 2)); }}");
    assert!((lp(&src, data()) - 5.0).abs() < 1e-12);
    let past = format!("{HEAD} target += a * sum(tail(u, 4)); }}");
    assert!(err(&past, data()).contains("index"));
}

#[test]
fn to_vector_flattens_in_row_major_order() {
    // M is [[1,2],[3,4],[5,6]], so its second entry is 2
    let src = format!("{HEAD} target += a * to_vector(M)[2]; }}");
    assert!((lp(&src, data()) - 2.0).abs() < 1e-12);
}

#[test]
fn dot_self_is_the_vector_against_itself() {
    let src = format!("{HEAD} target += a * dot_self(u); }}");
    assert!((lp(&src, data()) - 14.0).abs() < 1e-12);
}

#[test]
fn an_array_literal_is_a_container() {
    let src = format!("{HEAD} target += a * sum({{1, 2, 3}}); }}");
    assert!((lp(&src, data()) - 6.0).abs() < 1e-12);

    // and it indexes like one, including as a gather
    let idx = format!("{HEAD} target += a * sum(u[{{3, 1}}]); }}");
    assert!((lp(&idx, data()) - 4.0).abs() < 1e-12);
}
