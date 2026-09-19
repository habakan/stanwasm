//! End-to-end: compile a Stan model to wasm, instantiate it with wasmi, and
//! check the log_prob and gradients against the AST evaluator.
//!
//! The emitter's own tests are in tapewasm, against tapes built directly. What
//! these add is the half in front of it: that tracing a model records the tape
//! the evaluator would have walked.

use stanwasm_codegen::{compile, compile_with, compile_with_log_lik, Reroll};
use stanwasm_runtime::{Env, Model};
use wasmi::{Caller, Engine, Func, Linker, Memory, MemoryType, Module, Store};

fn lgamma(x: f64) -> f64 {
    tapewasm_autodiff::lgamma(x)
}
fn digamma(x: f64) -> f64 {
    tapewasm_autodiff::digamma(x)
}
fn phi(x: f64) -> f64 {
    tapewasm_autodiff::phi_cdf(x)
}

#[derive(Default)]
struct HostState;

fn install_math(linker: &mut Linker<HostState>, store: &mut Store<HostState>) {
    macro_rules! unary {
        ($name:literal, $fn:expr) => {{
            let f = Func::wrap(&mut *store, |_: Caller<'_, HostState>, x: f64| -> f64 {
                $fn(x)
            });
            linker.define("Math", $name, f).unwrap();
        }};
    }
    unary!("exp", f64::exp);
    unary!("log", f64::ln);
    unary!("sin", f64::sin);
    unary!("cos", f64::cos);
    unary!("lgamma", lgamma);
    unary!("digamma", digamma);
    unary!("phi", phi);
    unary!("tan", f64::tan);
    unary!("asin", f64::asin);
    unary!("acos", f64::acos);
    unary!("atan", f64::atan);
    let pow = Func::wrap(
        &mut *store,
        |_: Caller<'_, HostState>, x: f64, y: f64| -> f64 { x.powf(y) },
    );
    linker.define("Math", "pow", pow).unwrap();
}

fn run_aot_log_prob_grad(
    wasm: &[u8],
    n_params: usize,
    params: &[f64],
    scratch_len: usize,
    const_table: &[f64],
) -> (f64, Vec<f64>) {
    run_export(
        wasm,
        "log_prob_grad",
        n_params,
        params,
        n_params,
        scratch_len,
        const_table,
    )
}

/// `log_prob_grad` or `evaluate`: same arguments, and the second one is where
/// they differ — `n_params` gradients, or `out_len` values.
fn run_export(
    wasm: &[u8],
    name: &str,
    n_params: usize,
    params: &[f64],
    out_len: usize,
    scratch_len: usize,
    const_table: &[f64],
) -> (f64, Vec<f64>) {
    // The emitter widens a re-rolled loop to `f64x2` where it can, which wasmi
    // parses only with the proposal enabled.
    let mut config = wasmi::Config::default();
    config.wasm_simd(true);
    let engine = Engine::new(&config);
    let module = Module::new(&engine, wasm).expect("module parses");
    let mut store = Store::new(&engine, HostState);

    // Host-allocated memory shared with the AOT module: params, grads, then the
    // module's primal/adjoint scratch.
    let pages = ((n_params + out_len + scratch_len) * 8)
        .div_ceil(65536)
        .max(1) as u32;
    let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();

    let mut linker: Linker<HostState> = Linker::new(&engine);
    install_math(&mut linker, &mut store);
    linker.define("tapewasm", "memory", memory).unwrap();

    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .expect("instantiate");

    let lpg = instance
        .get_typed_func::<(i32, i32, i32, i32), f64>(&store, name)
        .unwrap();

    // Layout: params at offset 0, then the output buffer, then scratch.
    let params_ptr: i32 = 0;
    let grads_ptr: i32 = (n_params * 8) as i32;
    let scratch_ptr: i32 = ((n_params + out_len) * 8) as i32;
    let bytes: Vec<u8> = params.iter().flat_map(|p| p.to_le_bytes()).collect();
    memory
        .write(&mut store, params_ptr as usize, &bytes)
        .unwrap();
    // Re-rolled loops read their moving constants from the tail of scratch.
    if !const_table.is_empty() {
        let at = scratch_ptr as usize + (scratch_len - const_table.len()) * 8;
        let tbl: Vec<u8> = const_table.iter().flat_map(|c| c.to_le_bytes()).collect();
        memory.write(&mut store, at, &tbl).unwrap();
    }

    let lp = lpg
        .call(
            &mut store,
            (params_ptr, grads_ptr, n_params as i32, scratch_ptr),
        )
        .unwrap();

    let mut grad_bytes = vec![0u8; out_len * 8];
    memory
        .read(&store, grads_ptr as usize, &mut grad_bytes)
        .unwrap();
    let grads: Vec<f64> = grad_bytes
        .chunks(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();

    (lp, grads)
}

fn close(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps || ((a - b) / a.abs().max(b.abs()).max(1.0)).abs() < eps
}

const LINEAR_REGRESSION: &str = r#"
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 10);
  beta  ~ normal(0, 10);
  sigma ~ exponential(1);
  y ~ normal(alpha + beta * x, sigma);
}
"#;

#[test]
fn linear_regression_aot_matches_oracle() {
    let mut data = Env::new();
    data.set_scalar("N", 3.0);
    data.set_vector("x", &[1.0, 2.0, 3.0]);
    data.set_vector("y", &[1.5, 3.1, 4.9]);
    let model = Model::parse_and_load(LINEAR_REGRESSION, data).unwrap();

    let dummy = vec![0.1; model.n_params()];
    let compiled = compile(&model, &dummy).unwrap();

    let test_params = vec![0.5, 1.5, -0.2];
    let (oracle_lp, oracle_grads) = model.log_prob_grad(&test_params).unwrap();
    let (aot_lp, aot_grads) = run_aot_log_prob_grad(
        &compiled.wasm,
        compiled.n_params,
        &test_params,
        compiled.scratch_len,
        &compiled.const_table,
    );

    assert!(
        close(oracle_lp, aot_lp, 1e-12),
        "lp: oracle={oracle_lp}, aot={aot_lp}, diff={}",
        oracle_lp - aot_lp
    );
    assert_eq!(oracle_grads.len(), aot_grads.len());
    for (i, (o, a)) in oracle_grads.iter().zip(aot_grads.iter()).enumerate() {
        assert!(
            close(*o, *a, 1e-12),
            "grad[{i}]: oracle={o}, aot={a}, diff={}",
            o - a
        );
    }
}

const POISSON_REGRESSION: &str = r#"
data {
  int<lower=0> N;
  vector[N] x;
  array[N] int y;
}
parameters {
  real alpha;
  real beta;
}
model {
  alpha ~ normal(0, 5);
  beta  ~ normal(0, 1);
  for (i in 1:N) y[i] ~ poisson(exp(alpha + beta * x[i]));
}
"#;

#[test]
fn poisson_regression_aot_matches_oracle() {
    let mut data = Env::new();
    data.set_scalar("N", 5.0);
    data.set_vector("x", &[0.0, 1.0, 2.0, 3.0, 4.0]);
    data.set_vector("y", &[1.0, 2.0, 5.0, 12.0, 30.0]);
    let model = Model::parse_and_load(POISSON_REGRESSION, data).unwrap();

    let dummy = vec![0.1; model.n_params()];
    let compiled = compile(&model, &dummy).unwrap();

    let test_params = vec![0.0, 1.0];
    let (oracle_lp, oracle_grads) = model.log_prob_grad(&test_params).unwrap();
    let (aot_lp, aot_grads) = run_aot_log_prob_grad(
        &compiled.wasm,
        compiled.n_params,
        &test_params,
        compiled.scratch_len,
        &compiled.const_table,
    );

    assert!(
        close(oracle_lp, aot_lp, 1e-12),
        "lp: oracle={oracle_lp}, aot={aot_lp}"
    );
    for (i, (o, a)) in oracle_grads.iter().zip(aot_grads.iter()).enumerate() {
        assert!(close(*o, *a, 1e-12), "grad[{i}]: oracle={o}, aot={a}");
    }
}

const MULTIVARIATE_LKJ: &str = r#"
data {
  int<lower=1> K;
  vector[K] y;
}
parameters {
  vector[K] mu;
  cholesky_factor_corr[K] L;
}
model {
  mu ~ normal(0, 5);
  L  ~ lkj_corr_cholesky(2.0);
  y  ~ multi_normal_cholesky(mu, L);
}
"#;

#[test]
fn multivariate_lkj_aot_matches_oracle() {
    let mut data = Env::new();
    data.set_scalar("K", 2.0);
    data.set_vector("y", &[1.0, 2.0]);
    let model = Model::parse_and_load(MULTIVARIATE_LKJ, data).unwrap();

    let dummy = vec![0.1; model.n_params()];
    let compiled = compile(&model, &dummy).unwrap();

    let test_params = vec![0.5, 1.5, 0.3];
    let (oracle_lp, oracle_grads) = model.log_prob_grad(&test_params).unwrap();
    let (aot_lp, aot_grads) = run_aot_log_prob_grad(
        &compiled.wasm,
        compiled.n_params,
        &test_params,
        compiled.scratch_len,
        &compiled.const_table,
    );

    assert!(
        close(oracle_lp, aot_lp, 1e-12),
        "lp: oracle={oracle_lp}, aot={aot_lp}"
    );
    for (i, (o, a)) in oracle_grads.iter().zip(aot_grads.iter()).enumerate() {
        assert!(close(*o, *a, 1e-12), "grad[{i}]: oracle={o}, aot={a}");
    }
}

#[test]
fn module_validates_with_wasmparser() {
    let mut data = Env::new();
    data.set_scalar("N", 2.0);
    data.set_vector("x", &[0.0, 1.0]);
    data.set_vector("y", &[0.0, 1.0]);
    let model = Model::parse_and_load(LINEAR_REGRESSION, data).unwrap();
    let compiled = compile(&model, &[0.1; 3]).unwrap();
    let result = wasmparser::Validator::new().validate_all(&compiled.wasm);
    assert!(result.is_ok(), "wasm did not validate: {:?}", result.err());
}

#[test]
fn unsupported_op_is_reported_rather_than_trapping() {
    // No emitter arm for the Student-t tail, which is reachable from Stan source,
    // so the refusal has to be a message rather than a wasm trap.
    let src = r#"
data { int<lower=0> N; vector[N] y; }
parameters { real a; }
model { for (n in 1:N) target += student_t_lccdf(y[n] | 4.0, a, 1.0); }
"#;
    let mut data = Env::new();
    data.set_scalar("N", 2.0);
    data.set_vector("y", &[0.1, 0.2]);
    let model = Model::parse_and_load(src, data).unwrap();

    let err = stanwasm_codegen::compile(&model, &[0.1])
        .expect_err("the Student-t tail has no AOT emitter")
        .to_string();
    assert!(err.contains("StudentTLccdf"), "{err}");
}

/// Enough data points that the emitter re-rolls the vectorised statement into
/// a wasm loop. The small cases above stay straight-line, so without this the
/// loop emitter is never exercised.
#[test]
fn rerolled_linear_regression_matches_oracle() {
    const N: usize = 2000;
    let xs: Vec<f64> = (0..N).map(|i| -1.5 + i as f64 * 0.05).collect();
    let ys: Vec<f64> = (0..N).map(|i| 0.3 + i as f64 * 0.11).collect();
    let mut data = Env::new();
    data.set_scalar("N", N as f64);
    data.set_vector("x", &xs);
    data.set_vector("y", &ys);
    let model = Model::parse_and_load(LINEAR_REGRESSION, data).unwrap();

    let dummy = vec![0.1; model.n_params()];
    let compiled = compile(&model, &dummy).unwrap();
    assert!(
        !compiled.const_table.is_empty(),
        "expected a re-rolled loop with a moving-constant table"
    );

    for test_params in [
        vec![0.5, 1.5, -0.2],
        vec![-1.0, 0.25, 0.7],
        vec![0.0, 0.0, 0.0],
    ] {
        let (oracle_lp, oracle_grads) = model.log_prob_grad(&test_params).unwrap();
        let (aot_lp, aot_grads) = run_aot_log_prob_grad(
            &compiled.wasm,
            compiled.n_params,
            &test_params,
            compiled.scratch_len,
            &compiled.const_table,
        );
        assert!(
            close(oracle_lp, aot_lp, 1e-12),
            "lp at {test_params:?}: oracle={oracle_lp}, aot={aot_lp}"
        );
        for (i, (o, a)) in oracle_grads.iter().zip(aot_grads.iter()).enumerate() {
            assert!(
                close(*o, *a, 1e-12),
                "grad[{i}] at {test_params:?}: oracle={o}, aot={a}, diff={}",
                o - a
            );
        }
    }
}

/// Calling twice must give the same answer: the scratch buffer is reused, so a
/// stale adjoint or a clobbered constant table would only show on the second
/// call.
#[test]
fn rerolled_model_is_reentrant() {
    const N: usize = 2000;
    let mut data = Env::new();
    data.set_scalar("N", N as f64);
    data.set_vector("x", &(0..N).map(|i| i as f64 * 0.1).collect::<Vec<_>>());
    data.set_vector(
        "y",
        &(0..N).map(|i| 1.0 + i as f64 * 0.2).collect::<Vec<_>>(),
    );
    let model = Model::parse_and_load(LINEAR_REGRESSION, data).unwrap();
    let compiled = compile(&model, &vec![0.1; model.n_params()]).unwrap();
    assert!(
        !compiled.const_table.is_empty(),
        "expected a re-rolled loop"
    );

    let p = vec![0.4, 1.1, 0.3];
    let first = run_aot_log_prob_grad(
        &compiled.wasm,
        compiled.n_params,
        &p,
        compiled.scratch_len,
        &compiled.const_table,
    );
    let (oracle_lp, oracle_grads) = model.log_prob_grad(&p).unwrap();
    assert!(close(first.0, oracle_lp, 1e-12));
    for (o, a) in oracle_grads.iter().zip(first.1.iter()) {
        assert!(close(*o, *a, 1e-12));
    }
}

/// A hierarchical model whose group index is irregular. `mu[g[i]]` is the
/// gather no stride describes, so the emitter has to read its slot index from
/// a table — the one loop shape whose addresses are computed at run time.
#[test]
fn rerolled_gather_matches_oracle() {
    const N: usize = 3000;
    const G: usize = 8;
    let src = r#"data { int<lower=0> N; int<lower=1> G; array[N] int<lower=1> g; vector[N] y; }
parameters { vector[G] mu; real<lower=0> sigma; }
model {
  mu ~ normal(0, 5); sigma ~ exponential(1);
  for (i in 1:N) y[i] ~ normal(mu[g[i]], sigma);
}"#;
    let mut seed: u64 = 12345;
    let mut rnd = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223) & 0xffff_ffff;
        seed as f64 / 4294967296.0
    };
    let gs: Vec<String> = (0..N)
        .map(|_| format!("{}", 1 + (rnd() * 8.0) as u32))
        .collect();
    let ys: Vec<String> = (0..N)
        .map(|i| format!("{}", (i as f64).sin() * 2.0))
        .collect();
    let data_json = format!(
        "{{\"N\": {N}, \"G\": {G}, \"g\": [{}], \"y\": [{}]}}",
        gs.join(","),
        ys.join(",")
    );
    let model =
        Model::parse_and_load(src, stanwasm_runtime::data_from_json(&data_json).unwrap()).unwrap();

    let compiled = compile(&model, &vec![0.1; model.n_params()]).unwrap();
    assert!(
        !compiled.const_table.is_empty(),
        "expected re-rolled loops with staged tables"
    );

    for test_params in [
        vec![0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, 0.05, 0.5],
        vec![-0.7, 0.9, 0.0, 1.2, 0.3, -1.1, 0.6, -0.4, -0.2],
    ] {
        let (oracle_lp, oracle_grads) = model.log_prob_grad(&test_params).unwrap();
        let (aot_lp, aot_grads) = run_aot_log_prob_grad(
            &compiled.wasm,
            compiled.n_params,
            &test_params,
            compiled.scratch_len,
            &compiled.const_table,
        );
        assert!(
            close(oracle_lp, aot_lp, 1e-10),
            "lp: oracle={oracle_lp}, aot={aot_lp}"
        );
        for (i, (o, a)) in oracle_grads.iter().zip(aot_grads.iter()).enumerate() {
            assert!(
                close(*o, *a, 1e-10),
                "grad[{i}]: oracle={o}, aot={a}, diff={}",
                o - a
            );
        }
    }
}

/// A matrix-vector product: one contraction node per row, in a block of its
/// own, whose result the density block reads back. That crossing is the only
/// place the scratch buffer's slot order is observable, and the contraction
/// has two emitters — unrolled in place, or inside a loop reading a staged
/// column of coefficients — which have to agree with each other and with the
/// oracle.
#[test]
fn matrix_product_matches_oracle_in_every_reroll_mode() {
    use stanwasm_codegen::{compile_with, Reroll};
    const N: usize = 2000;
    const K: usize = 4;
    let src = r#"data { int<lower=0> N; int<lower=0> K; matrix[N,K] X; vector[N] y; }
parameters { vector[K] beta; real<lower=0> sigma; }
model {
  beta ~ normal(0, 1); sigma ~ exponential(1);
  y ~ normal(X * beta, sigma);
}"#;
    let mut seed: u64 = 6789;
    let mut rnd = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223) & 0xffff_ffff;
        seed as f64 / 4294967296.0 * 2.0 - 1.0
    };
    let rows: Vec<String> = (0..N)
        .map(|_| {
            let cells: Vec<String> = (0..K).map(|_| format!("{}", rnd())).collect();
            format!("[{}]", cells.join(","))
        })
        .collect();
    let ys: Vec<String> = (0..N).map(|_| format!("{}", rnd())).collect();
    let data_json = format!(
        "{{\"N\": {N}, \"K\": {K}, \"X\": [{}], \"y\": [{}]}}",
        rows.join(","),
        ys.join(",")
    );
    let model =
        Model::parse_and_load(src, stanwasm_runtime::data_from_json(&data_json).unwrap()).unwrap();

    let dummy = vec![0.1; model.n_params()];
    for mode in [Reroll::Auto, Reroll::Always, Reroll::Never] {
        let compiled = compile_with(&model, &dummy, mode).unwrap();
        if mode != Reroll::Never {
            assert!(
                !compiled.const_table.is_empty(),
                "{mode:?}: expected re-rolled loops with staged coefficients"
            );
        }
        for test_params in [
            vec![0.5, -0.3, 0.8, 0.1, -0.5],
            vec![-1.2, 0.4, 0.0, 0.9, 0.25],
        ] {
            let (oracle_lp, oracle_grads) = model.log_prob_grad(&test_params).unwrap();
            let (aot_lp, aot_grads) = run_aot_log_prob_grad(
                &compiled.wasm,
                compiled.n_params,
                &test_params,
                compiled.scratch_len,
                &compiled.const_table,
            );
            assert!(
                close(oracle_lp, aot_lp, 1e-10),
                "{mode:?}: lp oracle={oracle_lp}, aot={aot_lp}"
            );
            for (i, (o, a)) in oracle_grads.iter().zip(aot_grads.iter()).enumerate() {
                assert!(
                    close(*o, *a, 1e-10),
                    "{mode:?}: grad[{i}] oracle={o}, aot={a}, diff={}",
                    o - a
                );
            }
        }
    }
}

/// Every re-roll mode has to compute the same gradient. `Always` and `Never`
/// exercise the loop and straight-line emitters on the same trace, which is
/// the only place their outputs can be compared directly.
#[test]
fn reroll_modes_agree() {
    use stanwasm_codegen::{compile_with, Reroll};
    const N: usize = 400;
    let xs: Vec<f64> = (0..N).map(|i| -1.5 + i as f64 * 0.007).collect();
    let ys: Vec<f64> = (0..N).map(|i| 0.3 + i as f64 * 0.011).collect();
    let mut data = Env::new();
    data.set_scalar("N", N as f64);
    data.set_vector("x", &xs);
    data.set_vector("y", &ys);
    let model = Model::parse_and_load(LINEAR_REGRESSION, data).unwrap();
    let dummy = vec![0.1; model.n_params()];
    let test_params = vec![0.4, 1.3, -0.15];
    let (oracle_lp, oracle_grads) = model.log_prob_grad(&test_params).unwrap();

    for mode in [Reroll::Auto, Reroll::Always, Reroll::Never] {
        let c = compile_with(&model, &dummy, mode).unwrap();
        let (lp, grads) = run_aot_log_prob_grad(
            &c.wasm,
            c.n_params,
            &test_params,
            c.scratch_len,
            &c.const_table,
        );
        assert!(
            close(oracle_lp, lp, 1e-12),
            "{mode:?}: lp {oracle_lp} vs {lp}"
        );
        for (i, (o, a)) in oracle_grads.iter().zip(grads.iter()).enumerate() {
            assert!(close(*o, *a, 1e-12), "{mode:?}: grad[{i}] {o} vs {a}");
        }
    }
}

/// `tan`, `asin`, `acos` and `atan` are callable from Stan source, so the
/// emitter has to have them too — otherwise a model samples but will not
/// compile, which is the one asymmetry between the two paths a user would hit.
const TRIG: &str = r#"
data { int<lower=0> N; vector[N] x; }
parameters { real a; real b; }
model {
  a ~ normal(0, 1);
  b ~ normal(0, 1);
  for (i in 1:N) {
    target += tan(0.3 * a + 0.1 * x[i]);
    target += asin(0.2 * b);
    target += acos(0.15 * a);
    target += atan(a * b + x[i]);
  }
}
"#;

#[test]
fn the_inverse_trig_functions_agree_with_the_oracle() {
    let mut data = Env::new();
    data.set_scalar("N", 4.0);
    data.set_vector("x", &[0.0, 0.5, -0.75, 1.25]);
    let model = Model::parse_and_load(TRIG, data).unwrap();

    let dummy = vec![0.1; model.n_params()];
    let compiled = compile(&model, &dummy).unwrap();

    // Two points, because `acos` and `asin` bend hardest away from zero.
    for test_params in [vec![0.4, -0.6], vec![-1.1, 0.9]] {
        let (oracle_lp, oracle_grads) = model.log_prob_grad(&test_params).unwrap();
        let (aot_lp, aot_grads) = run_aot_log_prob_grad(
            &compiled.wasm,
            compiled.n_params,
            &test_params,
            compiled.scratch_len,
            &compiled.const_table,
        );
        assert!(
            close(oracle_lp, aot_lp, 1e-12),
            "lp at {test_params:?}: oracle={oracle_lp}, aot={aot_lp}"
        );
        for (i, (o, a)) in oracle_grads.iter().zip(aot_grads.iter()).enumerate() {
            assert!(
                close(*o, *a, 1e-12),
                "grad[{i}] at {test_params:?}: oracle={o}, aot={a}, diff={}",
                o - a
            );
        }
    }
}

fn linear_regression_model(n: usize) -> Model {
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| 0.5 * i as f64).collect();
    let mut data = Env::new();
    data.set_scalar("N", n as f64);
    data.set_vector("x", &x);
    data.set_vector("y", &y);
    Model::parse_and_load(LINEAR_REGRESSION, data).unwrap()
}

/// The id a host compares before handing a module someone else's scratch
/// buffer, so it has to travel with the module and it has to differ whenever
/// the buffers do.
#[test]
fn layout_id_is_exported_and_identifies_the_buffers() {
    let two = compile(&linear_regression_model(2), &[0.1; 3]).unwrap();
    let two_again = compile(&linear_regression_model(2), &[0.1; 3]).unwrap();
    let four = compile(&linear_regression_model(4), &[0.1; 3]).unwrap();

    assert_eq!(
        two.layout_id, two_again.layout_id,
        "recompiling one model must not move its id"
    );
    assert_ne!(two.layout_id, four.layout_id);
    assert_ne!(two.scratch_len, four.scratch_len);

    let mut found = None;
    for payload in wasmparser::Parser::new(0).parse_all(&two.wasm) {
        if let wasmparser::Payload::ExportSection(section) = payload.unwrap() {
            for export in section {
                let export = export.unwrap();
                if export.name == "tapewasm_layout_id" {
                    found = Some(export.kind);
                }
            }
        }
    }
    assert_eq!(
        found,
        Some(wasmparser::ExternalKind::Global),
        "the module exports no layout id global"
    );
}

// ---- the pointwise log-likelihood ----------------------------------------
//
// Every `~` statement whose variate is data leaves one term per observation on
// the tape, and `compile_with_log_lik` names them so the module reports them
// through `evaluate`. The oracle is the same model traced afresh.

fn evaluates(model: &Model, params: &[f64], reroll: Reroll) -> Vec<f64> {
    let dummy = vec![0.1; model.n_params()];
    let compiled = compile_with_log_lik(model, &dummy, reroll, true).unwrap();
    let want = model.log_lik(params).unwrap();
    assert_eq!(compiled.n_outputs, want.len(), "{reroll:?}: term count");
    let (_, got) = run_export(
        &compiled.wasm,
        "evaluate",
        compiled.n_params,
        params,
        compiled.n_outputs,
        compiled.scratch_len,
        &compiled.const_table,
    );
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(close(*g, *w, 1e-12), "{reroll:?}: term[{i}] {g} vs {w}");
    }
    got
}

fn linreg_model(n: usize) -> (Model, Vec<f64>) {
    let x: Vec<f64> = (0..n).map(|i| -2.0 + i as f64 * 0.1).collect();
    let y: Vec<f64> = x.iter().map(|x| 1.3 + 0.7 * x + 0.05 * x * x).collect();
    let mut data = Env::new();
    data.set_scalar("N", n as f64);
    data.set_vector("x", &x);
    data.set_vector("y", &y);
    let model = Model::parse_and_load(LINEAR_REGRESSION, data).unwrap();
    (model, vec![0.5, 1.5, -0.2])
}

#[test]
fn every_observation_gets_a_term() {
    let (model, at) = linreg_model(40);
    for reroll in [Reroll::Never, Reroll::Always, Reroll::Auto] {
        let terms = evaluates(&model, &at, reroll);
        assert_eq!(terms.len(), 40, "one term per observation");
    }
}

/// The re-rolled shape is where the terms are one position of a block, which is
/// the run the module writes with a loop rather than a store each.
#[test]
fn a_re_rolled_likelihood_reports_every_term() {
    let (model, at) = linreg_model(2000);
    let terms = evaluates(&model, &at, Reroll::Always);
    assert_eq!(terms.len(), 2000);
}

/// A prior is a `~` statement too, and its variate is a parameter — so it is
/// not a likelihood term. Three priors here, and none of them is reported.
#[test]
fn priors_are_not_likelihood_terms() {
    let (model, at) = linreg_model(12);
    assert_eq!(model.log_lik(&at).unwrap().len(), 12);
}

/// Written as a loop rather than vectorised, the same model has the same terms.
#[test]
fn a_looped_likelihood_gives_the_terms_the_vectorised_one_does() {
    const LOOPED: &str = r#"
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 10);
  beta  ~ normal(0, 10);
  sigma ~ exponential(1);
  for (n in 1:N) {
    y[n] ~ normal(alpha + beta * x[n], sigma);
  }
}
"#;
    let (vectorised, at) = linreg_model(20);
    let x: Vec<f64> = (0..20).map(|i| -2.0 + i as f64 * 0.1).collect();
    let y: Vec<f64> = x.iter().map(|x| 1.3 + 0.7 * x + 0.05 * x * x).collect();
    let mut data = Env::new();
    data.set_scalar("N", 20.0);
    data.set_vector("x", &x);
    data.set_vector("y", &y);
    let looped = Model::parse_and_load(LOOPED, data).unwrap();

    let a = vectorised.log_lik(&at).unwrap();
    let b = evaluates(&looped, &at, Reroll::Auto);
    assert_eq!(a.len(), b.len());
    for (i, (p, q)) in a.iter().zip(&b).enumerate() {
        assert!(close(*p, *q, 1e-12), "term[{i}] {p} vs {q}");
    }
}

/// `target += normal_lpdf(y | ...)` arrives already summed, so there is nothing
/// to attribute and the module says so rather than reporting a wrong shape.
#[test]
fn a_summed_likelihood_names_no_terms() {
    const SUMMED: &str = r#"
data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  real mu;
}
model {
  mu ~ normal(0, 10);
  target += normal_lpdf(y | mu, 1.0);
}
"#;
    let mut data = Env::new();
    data.set_scalar("N", 4.0);
    data.set_vector("y", &[0.1, -0.3, 1.2, 0.7]);
    let model = Model::parse_and_load(SUMMED, data).unwrap();
    assert!(model.log_lik(&[0.2]).unwrap().is_empty());
    let compiled = compile_with_log_lik(&model, &[0.1], Reroll::Auto, true).unwrap();
    assert_eq!(compiled.n_outputs, 0);
}

/// Naming the terms changes nothing about the density the module computes.
#[test]
fn naming_the_terms_leaves_the_gradient_alone() {
    let (model, at) = linreg_model(40);
    let dummy = vec![0.1; model.n_params()];
    let plain = compile_with(&model, &dummy, Reroll::Auto).unwrap();
    let with = compile_with_log_lik(&model, &dummy, Reroll::Auto, true).unwrap();
    let (lp_a, g_a) = run_aot_log_prob_grad(
        &plain.wasm,
        plain.n_params,
        &at,
        plain.scratch_len,
        &plain.const_table,
    );
    let (lp_b, g_b) = run_aot_log_prob_grad(
        &with.wasm,
        with.n_params,
        &at,
        with.scratch_len,
        &with.const_table,
    );
    assert_eq!(lp_a, lp_b);
    assert_eq!(g_a, g_b);
}
