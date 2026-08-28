//! Regression tests for inputs that used to be answered *wrongly* or that
//! panicked (a panic compiles to a wasm trap, which kills the module instance
//! and forces a page reload in the browser). Every case here is reachable from
//! hand-written Stan source, e.g. via the gallery's Wasm Sandbox tab.

use stanwasm_runtime::{data_from_json, Model};

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
fn unsupported_constraint_types_are_rejected() {
    for (decl, k, n_raw) in [
        ("cov_matrix[K] S;", 2, 3),
        ("corr_matrix[K] S;", 2, 1),
        ("cholesky_factor_cov[K] S;", 2, 3),
        ("unit_vector[K] u;", 2, 2),
    ] {
        let src = format!("data {{ int K; }} parameters {{ {decl} }} model {{ target += 0; }}");
        let e = err(&src, &format!(r#"{{"K":{k}}}"#), &vec![0.1; n_raw]);
        assert!(
            e.contains("has no constraint transform in this runtime yet"),
            "{decl}: {e}"
        );
    }
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
fn unsupported_constraint_names_the_parameter() {
    let e = err(
        "data { int K; } parameters { cov_matrix[K] S; } model { target += 0; }",
        r#"{"K":2}"#,
        &[0.1, 0.2, 0.3],
    );
    assert!(e.contains("`S`") && e.contains("`cov_matrix`"), "{e}");
}

#[test]
fn array_of_vectors_variate_points_at_the_loop_form() {
    // Legal Stan, but this runtime doesn't vectorize a multivariate variate.
    // The message has to name the loop form; reporting a size mismatch between
    // the N array rows and the K-long mu sends the reader the wrong way.
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
fn matrix_times_vector_is_an_error_not_a_wrong_answer() {
    let e = err(
        "data { matrix[2,2] X; vector[2] y; } parameters { vector[2] b; } \
         model { y ~ normal(X * b, 1); }",
        r#"{"X":[[1,0],[0,1]],"y":[1,2]}"#,
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
