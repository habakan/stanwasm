//! The `functions` block. Cases follow the rules in the Stan reference manual's
//! user-defined functions chapter, each either working or failing with a message.

use stanwasm_runtime::{Env, Model, Val};

fn load(src: &str, data: Env) -> Result<Model, String> {
    Model::parse_and_load(src, data).map_err(|e| e.to_string())
}

fn lp(src: &str, data: Env, params: &[f64]) -> Result<f64, String> {
    load(src, data)?
        .log_prob_grad(params)
        .map(|(v, _)| v)
        .map_err(|e| e.to_string())
}

fn err(src: &str, data: Env, params: &[f64]) -> String {
    match lp(src, data, params) {
        Ok(v) => panic!("expected an error, got lp = {v}"),
        Err(e) => e,
    }
}

/// `target += f(...)` puts the returned value straight into the log density, so the
/// log prob *is* the function's result and the assertions can be exact.
fn returns(body: &str, call: &str, data: Env) -> f64 {
    let src =
        format!("functions {{ {body} }}\nparameters {{ real a; }}\nmodel {{ target += {call}; }}");
    lp(&src, data, &[0.0]).unwrap()
}

#[test]
fn scalar_argument_and_arithmetic() {
    let v = returns("real sq(real x) { return x * x; }", "sq(3.0)", Env::new());
    assert!((v - 9.0).abs() < 1e-12, "{v}");
}

#[test]
fn local_variables_inside_the_body() {
    let v = returns(
        "real f(real x) { real y = x + 1; real z = y * 2; return z; }",
        "f(4.0)",
        Env::new(),
    );
    assert!((v - 10.0).abs() < 1e-12, "{v}");
}

#[test]
fn vector_argument_is_unsized_in_the_signature() {
    // Stan writes `vector v`, not `vector[N] v` — the argument carries its length.
    let mut data = Env::new();
    data.set_vector("z", &[1.0, 2.0, 3.0]);
    let v = returns("real s(vector v) { return sum(v); }", "s(z)", data);
    assert!((v - 6.0).abs() < 1e-12, "{v}");
}

#[test]
fn matrix_argument_and_product() {
    let mut data = Env::new();
    data.set(
        "M",
        Val::Vec(vec![
            Val::Vec(vec![Val::Num(1.0), Val::Num(2.0)]),
            Val::Vec(vec![Val::Num(3.0), Val::Num(4.0)]),
        ]),
    );
    data.set_vector("w", &[1.0, 1.0]);
    // M * w = (3, 7), summing to 10.
    let v = returns(
        "real q(matrix M, vector w) { return sum(M * w); }",
        "q(M, w)",
        data,
    );
    assert!((v - 10.0).abs() < 1e-12, "{v}");
}

#[test]
fn a_parameter_shadows_an_outer_name_of_the_same_spelling() {
    let mut data = Env::new();
    data.set_scalar("x", 100.0);
    // The body must see the argument, not the data variable also called `x`.
    let v = returns("real f(real x) { return x; }", "f(7.0)", data);
    assert!((v - 7.0).abs() < 1e-12, "{v}");
}

#[test]
fn one_function_may_call_another() {
    let v = returns(
        "real g(real x) { return x + 1; } real f(real x) { return g(x) * 2; }",
        "f(3.0)",
        Env::new(),
    );
    assert!((v - 8.0).abs() < 1e-12, "{v}");
}

#[test]
fn locals_do_not_leak_into_the_caller() {
    let e = err(
        "functions { real f(real x) { real hidden = x; return hidden; } }\n\
         parameters { real a; }\n\
         model { target += f(1.0); target += hidden; }",
        Env::new(),
        &[0.0],
    );
    assert!(e.contains("undefined variable"), "{e}");
}

#[test]
fn assigning_to_an_argument_does_not_reach_the_caller() {
    // Stan passes by constant reference. The scope is discarded either way, so the
    // caller's binding has to be untouched.
    let mut data = Env::new();
    data.set_scalar("x", 5.0);
    let v = returns("real f(real x) { x = 99; return 0.0; }", "f(x) + x", data);
    assert!((v - 5.0).abs() < 1e-12, "caller's x was modified: {v}");
}

#[test]
fn recursion_is_rejected_rather_than_expanded_forever() {
    // Calls are inlined into one recorded graph, so a recursive one would expand
    // until the stack gives out — a wasm trap in the browser.
    let e = err(
        "functions { real fac(real n) { return n * fac(n - 1); } }\n\
         parameters { real a; }\n\
         model { target += fac(3.0); }",
        Env::new(),
        &[0.0],
    );
    assert!(e.contains("calls itself"), "{e}");
}

#[test]
fn mutual_recursion_is_rejected_too() {
    let e = err(
        "functions { real f(real x) { return g(x); } real g(real x) { return f(x); } }\n\
         parameters { real a; }\n\
         model { target += f(1.0); }",
        Env::new(),
        &[0.0],
    );
    assert!(e.contains("calls itself"), "{e}");
}

#[test]
fn wrong_argument_count_is_an_error() {
    let e = err(
        "functions { real add2(real x, real y) { return x + y; } }\n\
         parameters { real a; }\n\
         model { target += add2(1.0); }",
        Env::new(),
        &[0.0],
    );
    assert!(e.contains("takes 2 argument(s), got 1"), "{e}");
}

#[test]
fn a_parameter_flows_through_a_function_with_its_gradient() {
    // The call is inlined onto the tape, so the derivative has to survive it.
    let src = "functions { real sq(real x) { return x * x; } }\n\
               parameters { real a; }\n\
               model { target += -sq(a); }";
    let model = load(src, Env::new()).unwrap();
    let (v, grad) = model.log_prob_grad(&[3.0]).unwrap();
    assert!((v + 9.0).abs() < 1e-12, "{v}");
    assert!(
        (grad[0] + 6.0).abs() < 1e-12,
        "d(-a^2)/da at 3 is -6: {grad:?}"
    );
}
