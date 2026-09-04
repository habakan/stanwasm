//! Regression tests for inputs that used to be answered *wrongly* or that
//! panicked (a panic compiles to a wasm trap, which kills the module instance
//! and forces a page reload in the browser). Every case here is reachable from
//! hand-written Stan source, e.g. via the gallery's Wasm Sandbox tab.

use stanwasm_autodiff::Tape;
use stanwasm_runtime::{data_from_json, Env, EvalError, Model, Val};

fn load(src: &str, data: &str) -> Result<Model, String> {
    let env = data_from_json(data).map_err(|e| e.to_string())?;
    Model::parse_and_load(src, env).map_err(|e| e.to_string())
}

/// `load` for cases that must fail (`Model` isn't `Debug`, so `unwrap_err`
/// can't be used directly).
fn load_err(src: &str, data: &str) -> String {
    match load(src, data) {
        Ok(_) => panic!("expected the data block to be rejected"),
        Err(e) => e,
    }
}

fn lp(src: &str, data: &str, params: &[f64]) -> Result<(f64, Vec<f64>), String> {
    load(src, data)?
        .log_prob_grad(params)
        .map_err(|e| e.to_string())
}

fn err(src: &str, data: &str, params: &[f64]) -> String {
    match lp(src, data, params) {
        Ok((v, _)) => panic!("expected an error, got lp = {v}"),
        Err(e) => e,
    }
}

// ---- operator semantics ---------------------------------------------------

#[test]
fn unary_minus_binds_looser_than_pow() {
    // `-a^2` is `-(a^2)`. This used to parse as `(-a)^2` and flip the sign.
    let (v, g) = lp(
        "parameters { real a; } model { target += -a^2; }",
        "{}",
        &[2.0],
    )
    .unwrap();
    assert_eq!(v, -4.0);
    assert_eq!(g, vec![-4.0]);
}

#[test]
fn pow_is_right_associative() {
    // 2^(3^2) = 512, not (2^3)^2 = 64.
    let (v, _) = lp(
        "parameters { real a; } model { target += a * 2^3^2; }",
        "{}",
        &[1.0],
    )
    .unwrap();
    assert_eq!(v, 512.0);
}

#[test]
fn int_division_truncates() {
    // Stan is statically typed: `N / 2` with `int N = 3` is 1, not 1.5.
    let (v, _) = lp(
        "data { int N; } parameters { real a; } model { target += a * (N / 2); }",
        r#"{"N":3}"#,
        &[1.0],
    )
    .unwrap();
    assert_eq!(v, 1.0);
}

#[test]
fn real_division_is_unaffected_by_int_division() {
    // Only int/int truncates — a real operand keeps full precision. This is
    // the funnel model from the MCMC Visualizer tab (`exp(y / 2)`).
    let (v, _) = lp(
        "parameters { real a; } model { target += a / 2; }",
        "{}",
        &[3.0],
    )
    .unwrap();
    assert_eq!(v, 1.5);
}

#[test]
fn loop_counter_is_int_typed() {
    let (v, _) = lp(
        "data { int N; } parameters { real a; } model { for (i in 1:N) target += a * (i / 2); }",
        r#"{"N":3}"#,
        &[1.0],
    )
    .unwrap();
    // i/2 for i = 1, 2, 3 → 0 + 1 + 1
    assert_eq!(v, 2.0);
}

#[test]
fn int_division_by_zero_is_an_error() {
    let e = err(
        "data { int N; } parameters { real a; } model { target += a * (2 / N); }",
        r#"{"N":0}"#,
        &[1.0],
    );
    assert!(e.contains("integer division by zero"), "{e}");
}

// ---- constraint transforms ------------------------------------------------

#[test]
fn array_element_constraint_is_applied_with_jacobian() {
    // `array[N] real<lower=0>` used to fall through to a pass-through arm:
    // no exp() transform, no Jacobian, negative values accepted as positive.
    let (v, g) = lp(
        "data { int N; } parameters { array[N] real<lower=0> s; } model { target += s[1] + s[2]; }",
        r#"{"N":2}"#,
        &[-1.0, -2.0],
    )
    .unwrap();
    // s = exp(raw); log|J| = raw₁ + raw₂
    let expected = (-1.0_f64).exp() + (-2.0_f64).exp() - 3.0;
    assert!((v - expected).abs() < 1e-12, "got {v}, want {expected}");
    // d/draw_i of (exp(raw_i) + raw_i)
    assert!((g[0] - ((-1.0_f64).exp() + 1.0)).abs() < 1e-12);
    assert!((g[1] - ((-2.0_f64).exp() + 1.0)).abs() < 1e-12);
}

#[test]
fn matrix_shape_check_compares_columns_not_just_rows() {
    // Both operands have 2 rows, so a row-count-only check would let this
    // through and `zip` would truncate the wider one's columns.
    let e = err(
        "data { matrix[2,3] A; matrix[2,4] B; } parameters { real a; } \
         model { target += a * sum(A[1] + B[1]); }",
        r#"{"A":[[1,2,3],[4,5,6]],"B":[[1,2,3,4],[5,6,7,8]]}"#,
        &[1.0],
    );
    assert!(e.contains("shape mismatch"), "{e}");
}

#[test]
fn int_parameters_are_rejected_with_their_own_message() {
    let e = err(
        "parameters { real a; int k; } model { a ~ normal(0, 1); }",
        "{}",
        &[0.0],
    );
    assert!(
        e.contains("`k` is declared `int`") && e.contains("must be"),
        "{e}"
    );
}

#[test]
fn array_of_vectors_variate_points_at_the_loop_form() {
    // Legal Stan, but not vectorized here. The message must name the loop form;
    // a size mismatch between N rows and a K-long mu sends the reader the wrong way.
    let e = err(
        "data { int N; int K; array[N] vector[K] y; vector[K] mu; } \
         parameters { cholesky_factor_corr[K] L; } \
         model { y ~ multi_normal_cholesky(mu, L); }",
        r#"{"N":3,"K":2,"y":[[1,2],[3,4],[5,6]],"mu":[0,0]}"#,
        &[0.3],
    );
    assert!(
        e.contains("not vectorized here") && e.contains("for (n in 1:N)"),
        "{e}"
    );
}

#[test]
fn matrix_parameter_keeps_its_row_structure() {
    // A `matrix[R, C]` parameter used to arrive as one flat vector, so `M[i,j]`
    // silently read the wrong element.
    let (v, _) = lp(
        "parameters { matrix[2,2] M; } model { target += M[2,1]; }",
        "{}",
        &[1.0, 2.0, 3.0, 4.0],
    )
    .unwrap();
    assert_eq!(v, 3.0);
}

// ---- data block validation ------------------------------------------------

#[test]
fn declared_data_must_be_present() {
    let e = load_err(
        "data { int N; real x; } parameters { real a; } model { a ~ normal(0, 1); }",
        r#"{"N":2}"#,
    );
    assert!(e.contains("`x`") && e.contains("missing"), "{e}");
}

#[test]
fn data_bounds_are_checked() {
    let e = load_err(
        "data { int<lower=0> N; } parameters { real a; } model { a ~ normal(0, 1); }",
        r#"{"N":-5}"#,
    );
    assert!(e.contains("lower=0"), "{e}");
}

#[test]
fn data_lengths_must_match_the_declaration() {
    let e = load_err(
        "data { int N; vector[N] x; } parameters { real a; } model { a ~ normal(0, 1); }",
        r#"{"N":5,"x":[1,2]}"#,
    );
    assert!(e.contains("length-5"), "{e}");
}

#[test]
fn int_data_must_be_whole_numbers() {
    let e = load_err(
        "data { array[2] int y; } parameters { real a; } model { a ~ normal(0, 1); }",
        r#"{"y":[1,2.5]}"#,
    );
    assert!(e.contains("not a whole number"), "{e}");
}

#[test]
fn valid_data_still_loads() {
    let m = load(
        "data { int<lower=0> N; vector[N] x; array[N] int y; } \
         parameters { real a; } model { a ~ normal(0, 1); }",
        r#"{"N":2,"x":[0.5,1.5],"y":[3,4]}"#,
    )
    .unwrap();
    assert_eq!(m.n_params(), 1);
}

// ---- former panics --------------------------------------------------------

#[test]
fn wrong_distribution_arity_is_an_error() {
    let e = err(
        "parameters { real a; } model { a ~ normal(0); }",
        "{}",
        &[0.0],
    );
    assert!(e.contains("normal expects 2"), "{e}");
}

#[test]
fn comparing_containers_is_an_error() {
    let e = err(
        "data { vector[2] x; vector[2] y; } parameters { real a; } \
         model { if (x == y) target += a; }",
        r#"{"x":[1,2],"y":[1,2]}"#,
        &[0.0],
    );
    assert!(e.contains("expected a scalar"), "{e}");
}

#[test]
fn matrix_times_vector_with_a_mismatched_inner_dimension_is_an_error() {
    // The product itself is supported; only dimensions that cannot meet are rejected,
    // and `zip` would otherwise truncate to the shorter operand.
    let e = err(
        "data { matrix[2,3] X; vector[2] y; } parameters { vector[2] b; } \
         model { y ~ normal(X * b, 1); }",
        r#"{"X":[[1,0,1],[0,1,1]],"y":[1,2]}"#,
        &[0.0, 0.0],
    );
    assert!(e.contains("shape mismatch"), "{e}");
}

#[test]
fn mismatched_vector_lengths_are_an_error() {
    // `zip` used to truncate to the shorter operand and return a short vector.
    let e = err(
        "data { vector[3] x; vector[2] z; } parameters { real a; } \
         model { target += a * sum(x + z); }",
        r#"{"x":[1,2,3],"z":[1,2]}"#,
        &[1.0],
    );
    assert!(e.contains("shape mismatch"), "{e}");
}

#[test]
fn vectorized_distribution_argument_must_match_the_variate() {
    let e = err(
        "data { vector[3] y; vector[2] mu; } parameters { real<lower=0> s; } \
         model { y ~ normal(mu, s); }",
        r#"{"y":[1,2,3],"mu":[0,0]}"#,
        &[0.0],
    );
    assert!(e.contains("length 2") && e.contains("length 3"), "{e}");
}

#[test]
fn out_of_range_slice_is_an_error() {
    let e = err(
        "data { vector[3] y; } parameters { real a; } model { target += a * sum(y[2:5]); }",
        r#"{"y":[1,2,3]}"#,
        &[1.0],
    );
    assert!(e.contains("out of bounds"), "{e}");
}

/// An ODE integrator is refused by name, before its first argument — which is
/// the system function, and would otherwise read as an undefined variable.
#[test]
fn an_ode_integrator_says_what_it_is() {
    let src = "functions { array[] real f(real t, array[] real y, array[] real th,
                                          data array[] real x_r, data array[] int x_i) {
                 return y;
               } }
               parameters { real a; }
               model { target += a * integrate_ode_rk45(f, {1.0}, 0, {1.0}, {a}, {1.0}, {1})[1, 1]; }";
    let msg = match Model::parse_and_load(src, data_from_json("{}").unwrap())
        .unwrap()
        .log_prob_grad(&[1.0])
    {
        Err(e) => e.to_string(),
        Ok(v) => panic!("expected an error, got {v:?}"),
    };
    assert!(msg.contains("integrate_ode_rk45"), "{msg}");
    assert!(msg.contains("adaptive"), "{msg}");
}

/// The data block is deserialised straight into `Val`, so the rejections that
/// used to come from a `serde_json::Value` walk have to still happen — and say
/// which field.
#[test]
fn a_data_field_that_is_not_a_number_says_which() {
    for bad in [
        r#"{"x": "hello"}"#,
        r#"{"x": [1, true]}"#,
        r#"{"x": {"a": 1}}"#,
    ] {
        let msg = data_from_json(bad).map(|_| ()).unwrap_err().to_string();
        assert!(msg.contains("\"x\""), "{bad} gave {msg}");
    }
    assert!(data_from_json("[1, 2]").is_err());
    assert!(data_from_json("{ not json").is_err());
}

#[test]
fn nested_arrays_keep_their_shape() {
    let env = data_from_json(r#"{"N": 2, "M": [[1, 2, 3], [4, 5, 6]], "v": [7.5, -8]}"#).unwrap();
    let Some(Val::Vec(rows)) = env.get("M") else {
        panic!("M is not a container");
    };
    assert_eq!(rows.len(), 2);
    let Some(Val::Vec(first)) = rows.first() else {
        panic!("row is not a container");
    };
    assert_eq!(first.len(), 3);
    assert_eq!(env.get("N").unwrap().to_f64(&Tape::new()).unwrap(), 2.0);
}

/// A graph too large for memory is an error, not an allocator abort — on wasm
/// an abort traps, and a trap takes the module instance down with it.
#[test]
fn a_graph_past_the_node_limit_is_reported() {
    let src = "
        data { int N; matrix[N, N] x; }
        parameters { vector[N] b; }
        model { target += sum(x * x * x * b); }
    ";
    let n = 700;
    let row: Vec<Val> = (0..n).map(|_| Val::Num(1.0)).collect();
    let mut env = Env::new();
    env.set("N", Val::Num(n as f64));
    env.set("x", Val::Vec((0..n).map(|_| Val::Row(row.clone())).collect()));
    let model = Model::parse_and_load(src, env).expect("loads");
    let err = model
        .log_prob_grad(&vec![0.1; n])
        .expect_err("700 cubed is past the limit");
    assert!(matches!(err, EvalError::TapeTooLarge(_)), "{err}");
}
