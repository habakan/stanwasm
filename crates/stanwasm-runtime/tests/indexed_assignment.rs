//! `y[i] = ...` and `M[i, j] = ...`. The parser nests these as `Index(Index(M, i), j)`,
//! and `Val` is a tree of owned vectors, so the write rebuilds each level.

use stanwasm_runtime::{Env, Model, Val};

fn lp(src: &str, data: Env, params: &[f64]) -> Result<f64, String> {
    Model::parse_and_load(src, data)
        .map_err(|e| e.to_string())?
        .log_prob_grad(params)
        .map(|(v, _)| v)
        .map_err(|e| e.to_string())
}

/// `target +=` the result, so the log prob is the value under test.
fn value_of(body: &str, data: Env) -> f64 {
    let src = format!("parameters {{ real a; }}\nmodel {{ {body} }}");
    lp(&src, data, &[0.0]).unwrap()
}

#[test]
fn writes_into_a_vector_element() {
    let v = value_of(
        "vector[3] y; y[1] = 0; y[2] = 5; y[3] = 0; target += sum(y);",
        Env::new(),
    );
    assert!((v - 5.0).abs() < 1e-12, "{v}");
}

#[test]
fn writes_into_a_matrix_cell() {
    let v = value_of(
        "matrix[2,2] M; for (i in 1:2) for (j in 1:2) M[i,j] = 0; M[2,1] = 7; target += M[2,1];",
        Env::new(),
    );
    assert!((v - 7.0).abs() < 1e-12, "{v}");
}

#[test]
fn a_write_leaves_its_neighbours_alone() {
    let v = value_of(
        "vector[4] y; for (i in 1:4) y[i] = 1; y[3] = 10; target += sum(y);",
        Env::new(),
    );
    assert!((v - 13.0).abs() < 1e-12, "1+1+10+1 = 13, got {v}");
}

#[test]
fn compound_assignment_reads_then_writes_the_same_cell() {
    let v = value_of(
        "vector[2] y; y[1] = 2; y[2] = 0; y[1] += 3; target += sum(y);",
        Env::new(),
    );
    assert!((v - 5.0).abs() < 1e-12, "{v}");
}

#[test]
fn a_parameter_written_into_a_cell_keeps_its_gradient() {
    // The tape has to see the value through the container, not a detached copy.
    let src = "parameters { real a; }\n\
               model { vector[2] y; y[1] = a * a; y[2] = 0; target += -sum(y); }";
    let model = Model::parse_and_load(src, Env::new()).unwrap();
    let (v, grad) = model.log_prob_grad(&[3.0]).unwrap();
    assert!((v + 9.0).abs() < 1e-12, "{v}");
    assert!((grad[0] + 6.0).abs() < 1e-12, "{grad:?}");
}

#[test]
fn out_of_bounds_is_an_error_not_a_silent_no_op() {
    let e = lp(
        "parameters { real a; }\nmodel { vector[2] y; y[5] = 1; target += a; }",
        Env::new(),
        &[0.0],
    )
    .unwrap_err();
    assert!(e.contains("out of bounds"), "{e}");
}

#[test]
fn writing_into_data_does_not_disturb_the_caller_of_a_function() {
    // Arguments are bound into a scope that is discarded, so the data stays intact.
    let mut data = Env::new();
    data.set("v", Val::Vec(vec![Val::Num(1.0), Val::Num(2.0)]));
    let src = "functions { real f(vector w) { w[1] = 99; return w[1]; } }\n\
               parameters { real a; }\n\
               model { target += f(v); target += sum(v); }";
    let v = lp(src, data, &[0.0]).unwrap();
    assert!(
        (v - (99.0 + 3.0)).abs() < 1e-12,
        "caller's v was modified: {v}"
    );
}

#[test]
fn an_uninitialised_transformed_parameter_arrives_shaped() {
    // `vector[W] k;` in `transformed parameters` used to bind a scalar zero, so the
    // element assignments below it had nothing to write into and `sum` refused it —
    // while the identical code inside `model` worked.
    let src = "data { int<lower=0> W; }\n\
               parameters { real a; }\n\
               transformed parameters { vector[W] k; for (w in 1:W) k[w] = a * w; }\n\
               model { target += sum(k); }";
    let mut data = Env::new();
    data.set_scalar("W", 3.0);
    let model = Model::parse_and_load(src, data).unwrap();

    // a=1 gives k = (1, 2, 3), summing to 6.
    let (lp, grad) = model.log_prob_grad(&[1.0]).unwrap();
    assert!((lp - 6.0).abs() < 1e-12, "{lp}");
    assert!((grad[0] - 6.0).abs() < 1e-12, "{grad:?}");
}
