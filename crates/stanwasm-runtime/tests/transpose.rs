//! `'`. Orientation exists so that `x' * y` is the inner product Stan means
//! and `x * y'` the outer one; every other operation reads the elements.

use stanwasm_runtime::{Env, Model, Val};

fn lp(src: &str, data: Env, at: &[f64]) -> f64 {
    Model::parse_and_load(src, data)
        .unwrap()
        .log_prob_grad(at)
        .unwrap()
        .0
}

fn err(src: &str, data: Env, at: &[f64]) -> String {
    let m = match Model::parse_and_load(src, data) {
        Ok(m) => m,
        Err(e) => return e.to_string(),
    };
    match m.log_prob_grad(at) {
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

fn xy() -> Env {
    let mut d = Env::new();
    d.set_vector("x", &[1.0, 2.0, 3.0]);
    d.set_vector("y", &[4.0, 5.0, 6.0]);
    d
}

#[test]
fn row_times_column_is_the_inner_product() {
    let src = "data { vector[3] x; vector[3] y; }
               parameters { real a; }
               model { target += a * (x' * y); }";
    assert!((lp(src, xy(), &[1.0]) - 32.0).abs() < 1e-12);
}

#[test]
fn column_times_row_is_the_outer_product() {
    let src = "data { vector[3] x; vector[3] y; }
               parameters { real a; }
               model { target += a * (x * y')[3, 2]; }";
    // (x yᵀ)[3,2] = x₃ y₂ = 3 * 5
    assert!((lp(src, xy(), &[1.0]) - 15.0).abs() < 1e-12);
}

#[test]
fn a_double_transpose_is_the_original() {
    let src = "data { vector[3] x; vector[3] y; }
               parameters { real a; }
               model { target += a * sum(x'' .* y); }";
    assert!((lp(src, xy(), &[1.0]) - 32.0).abs() < 1e-12);
}

/// Stan rejects `vector * vector`; answering it element-wise would be a wrong
/// dot product, so it is an error here rather than a silent divergence.
#[test]
fn two_columns_cannot_be_multiplied() {
    let src = "data { vector[3] x; vector[3] y; }
               parameters { real a; }
               model { target += a * sum(x * y); }";
    let msg = err(src, xy(), &[1.0]);
    assert!(msg.contains("vector[3]"), "{msg}");
}

/// Orientation changes nothing element-wise, so a row and a column of one
/// length still subtract — the shape check only guards `*`.
#[test]
fn element_wise_ignores_orientation() {
    let src = "data { vector[3] x; vector[3] y; }
               parameters { real a; }
               model { target += a * sum(y - x'); }";
    assert!((lp(src, xy(), &[1.0]) - 9.0).abs() < 1e-12);
}

#[test]
fn a_matrix_reflects() {
    let mut d = Env::new();
    d.set("M", matrix(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]));
    let src = "data { matrix[2, 3] M; }
               parameters { real a; }
               model { target += a * M'[1, 2]; }";
    assert!((lp(src, d, &[1.0]) - 4.0).abs() < 1e-12);
}

/// `rep_matrix` lays a row along the rows and a column along the columns, so
/// the orientation decides the result's shape.
#[test]
fn rep_matrix_follows_the_orientation() {
    let src = "data { vector[3] x; vector[3] y; }
               parameters { real a; }
               model { target += a * rows(rep_matrix(x', 4)); }";
    assert!((lp(src, xy(), &[1.0]) - 4.0).abs() < 1e-12);

    let src_col = "data { vector[3] x; vector[3] y; }
                   parameters { real a; }
                   model { target += a * rows(rep_matrix(x, 4)); }";
    assert!((lp(src_col, xy(), &[1.0]) - 3.0).abs() < 1e-12);
}

#[test]
fn a_row_on_the_left_of_a_matrix_stays_a_row() {
    let mut d = xy();
    d.set("M", matrix(&[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]]));
    let src = "data { vector[3] x; vector[3] y; matrix[3, 2] M; }
               parameters { real a; }
               model { target += a * sum(x' * M); }";
    // x' M = [1+3, 2+3] = [4, 5]
    assert!((lp(src, d, &[1.0]) - 9.0).abs() < 1e-12);
}

/// The gradient flows through the orientation, not just the value.
#[test]
fn the_inner_product_differentiates() {
    let src = "data { vector[3] x; vector[3] y; }
               parameters { real a; }
               model { target += a * (x' * y); }";
    let g = Model::parse_and_load(src, xy())
        .unwrap()
        .log_prob_grad(&[2.0])
        .unwrap()
        .1;
    assert!((g[0] - 32.0).abs() < 1e-12);
}

/// The inner product against a parameter vector: each element's gradient is the
/// other operand, which a wrong orientation would not produce.
#[test]
fn the_inner_product_differentiates_a_parameter_vector() {
    let src = "data { vector[3] y; }
               parameters { vector[3] b; }
               model { target += b' * y; }";
    let mut d = Env::new();
    d.set_vector("y", &[4.0, 5.0, 6.0]);
    let (lp, g) = Model::parse_and_load(src, d)
        .unwrap()
        .log_prob_grad(&[1.0, 2.0, 3.0])
        .unwrap();
    assert!((lp - 32.0).abs() < 1e-12);
    assert_eq!(g, vec![4.0, 5.0, 6.0]);
}

/// A declared `row_vector` is a row without a `'` having written it, so it
/// multiplies against a column the same way.
#[test]
fn a_declared_row_vector_is_a_row() {
    let src = "data { vector[3] y; }
               parameters { row_vector[3] r; }
               model { target += r * y; }";
    let mut d = Env::new();
    d.set_vector("y", &[4.0, 5.0, 6.0]);
    let (lp, g) = Model::parse_and_load(src, d)
        .unwrap()
        .log_prob_grad(&[1.0, 2.0, 3.0])
        .unwrap();
    assert!((lp - 32.0).abs() < 1e-12);
    assert_eq!(g, vec![4.0, 5.0, 6.0]);
}

#[test]
fn rep_row_vector_makes_a_row() {
    let src = "data { vector[3] x; vector[3] y; }
               parameters { real a; }
               model { target += a * (rep_row_vector(2, 3) * y); }";
    assert!((lp(src, xy(), &[1.0]) - 30.0).abs() < 1e-12);
}
