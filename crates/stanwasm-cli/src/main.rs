//! Local development CLI. Currently has one subcommand:
//!
//!   stanwasm-cli bench [model]   run a per-path benchmark suite
//!
//! Times each of:
//!   A) Model::log_prob_grad      — fresh AST trace per call (slow oracle)
//!   B) Compiled::log_prob_grad   — recorded-tape replay (fast Rust path)
//!   C) AOT wasm via wasmi        — emitted by stanwasm-codegen
//!   D) StanModel::sample         — full NUTS sampling end-to-end

use std::time::{Duration, Instant};

use stanwasm::StanModel;
use stanwasm_codegen::compile as aot_compile;
use stanwasm_runtime::{data_from_json, Compiled, Model};
use wasmi::{Caller, Engine, Func, Linker, Memory, MemoryType, Module as WasmModule, Store};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("bench") => run_bench(args.get(2).map(String::as_str)),
        Some(other) => {
            eprintln!("unknown subcommand {other:?}");
            eprintln!("usage: stanwasm-cli bench [linear|poisson|all]");
            std::process::exit(2);
        }
        None => {
            println!("stanwasm-cli v{}", env!("CARGO_PKG_VERSION"));
            println!("usage: stanwasm-cli bench [linear|poisson|all]");
            Ok(())
        }
    }
}

// ---- benchmark suite --------------------------------------------------------

const N_WARMUP: u32 = 1000;
const N_DRAWS: u32 = 1000;
const N_LPG_ITERS: usize = 10_000;

struct Case {
    name: &'static str,
    src: &'static str,
    data: &'static str,
    init: &'static [f64],
}

const CASES: &[Case] = &[
    Case {
        name: "linear_regression",
        src: r#"
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}
parameters { real alpha; real beta; real<lower=0> sigma; }
model {
  alpha ~ normal(0, 10);
  beta  ~ normal(0, 10);
  sigma ~ exponential(1);
  y ~ normal(alpha + beta * x, sigma);
}
"#,
        data: r#"{
  "N": 30,
  "x": [-1.5,-1.4,-1.3,-1.2,-1.1,-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4],
  "y": [-1.3,-1.0,-0.7,-0.5,-0.4,-0.1,0.0,0.2,0.4,0.5,0.7,0.8,1.0,1.2,1.3,1.5,1.7,1.8,2.0,2.2,2.4,2.5,2.7,2.9,3.1,3.3,3.4,3.6,3.8,4.0]
}"#,
        init: &[0.0, 1.0, 0.0],
    },
    Case {
        name: "poisson_regression",
        src: r#"
data { int<lower=0> N; vector[N] x; array[N] int y; }
parameters { real alpha; real beta; }
model {
  alpha ~ normal(0, 5);
  beta  ~ normal(0, 1);
  for (i in 1:N) y[i] ~ poisson(exp(alpha + beta * x[i]));
}
"#,
        data: r#"{
  "N": 5, "x": [0,1,2,3,4], "y": [1,2,5,12,30]
}"#,
        init: &[0.0, 1.0],
    },
    Case {
        name: "eight_schools_ncp",
        src: r#"
data {
  int<lower=0> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}
parameters {
  real mu;
  real<lower=0> tau;
  vector[J] theta_tilde;
}
transformed parameters {
  vector[J] theta = mu + tau * theta_tilde;
}
model {
  mu ~ normal(0, 5);
  tau ~ half_normal(5);
  theta_tilde ~ normal(0, 1);
  y ~ normal(theta, sigma);
}
"#,
        data: r#"{
  "J": 8,
  "y": [28, 8, -3, 7, -1, 1, 18, 12],
  "sigma": [15, 10, 16, 11, 9, 11, 10, 18]
}"#,
        init: &[0.1; 10],
    },
];

fn run_bench(filter: Option<&str>) -> anyhow::Result<()> {
    println!(
        "{:<20} | {:>10} | {:>10} | {:>10} | {:>12}",
        "case", "AST µs", "replay µs", "AOT µs", "sample ms"
    );
    println!("{}", "-".repeat(74));

    for case in CASES {
        if let Some(f) = filter {
            if f != "all" && !case.name.starts_with(f) {
                continue;
            }
        }
        let row = bench_case(case)?;
        println!(
            "{:<20} | {:>10.2} | {:>10.2} | {:>10.2} | {:>12.1}",
            case.name, row.ast_us, row.replay_us, row.aot_us, row.sample_ms
        );
    }
    println!();
    println!("Notes:");
    println!("- AST µs    = Model::log_prob_grad (fresh trace per call) — slow oracle");
    println!("- replay µs = Compiled::log_prob_grad (recorded-tape replay) — fast Rust");
    println!("- AOT µs    = stanwasm-codegen output run via wasmi (Rust → wasm32)");
    println!("- sample ms = full NUTS, n_warmup={N_WARMUP}, n_draws={N_DRAWS}, seed=42");
    println!("- µs/call averaged over {N_LPG_ITERS} iterations");
    Ok(())
}

struct Row {
    ast_us: f64,
    replay_us: f64,
    aot_us: f64,
    sample_ms: f64,
}

fn bench_case(case: &Case) -> anyhow::Result<Row> {
    let env = data_from_json(case.data)?;
    let model = Model::parse_and_load(case.src, env.clone())?;
    let n = model.n_params();

    // A) AST trace per call
    let ast_us = mean_us(N_LPG_ITERS, || {
        let _ = model.log_prob_grad(case.init);
    });

    // B) Tape replay
    let mut compiled = Compiled::from(&model, &vec![0.1; n])?;
    let mut grads = vec![0.0; n];
    let replay_us = mean_us(N_LPG_ITERS, || {
        let _ = compiled.log_prob_grad(case.init, &mut grads);
    });

    // C) AOT wasm via wasmi
    let aot = aot_compile(&model, &vec![0.1; n])?;
    let aot_us = bench_aot_via_wasmi(&aot.wasm, n, case.init, aot.scratch_len)?;

    // D) End-to-end sampling
    let mut sm = StanModel::new(case.src, case.data).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let t0 = Instant::now();
    let _samples = sm
        .sample(case.init, N_WARMUP, N_DRAWS, 42)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let sample_ms = t0.elapsed().as_secs_f64() * 1000.0;

    Ok(Row {
        ast_us,
        replay_us,
        aot_us,
        sample_ms,
    })
}

fn mean_us(iters: usize, mut f: impl FnMut()) -> f64 {
    // small warmup
    for _ in 0..iters / 10 {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = t0.elapsed();
    elapsed.as_secs_f64() * 1e6 / iters as f64
}

// ---- wasmi runner -----------------------------------------------------------

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
    unary!("lgamma", stanwasm_autodiff_lgamma);
    unary!("digamma", stanwasm_autodiff_digamma);
    unary!("phi", stanwasm_autodiff_phi);
    let pow = Func::wrap(
        &mut *store,
        |_: Caller<'_, HostState>, x: f64, y: f64| -> f64 { x.powf(y) },
    );
    linker.define("Math", "pow", pow).unwrap();
}

fn stanwasm_autodiff_lgamma(x: f64) -> f64 {
    stanwasm_autodiff::lgamma(x)
}
fn stanwasm_autodiff_digamma(x: f64) -> f64 {
    stanwasm_autodiff::digamma(x)
}
fn stanwasm_autodiff_phi(x: f64) -> f64 {
    stanwasm_autodiff::phi_cdf(x)
}

fn bench_aot_via_wasmi(
    wasm: &[u8],
    n_params: usize,
    params: &[f64],
    scratch_len: usize,
) -> anyhow::Result<f64> {
    let engine = Engine::default();
    let module = WasmModule::new(&engine, wasm)?;
    let mut store = Store::new(&engine, HostState);
    // params + grads + the module's primal/adjoint scratch, rounded up to pages.
    let pages = ((n_params * 2 + scratch_len) * 8).div_ceil(65536).max(1) as u32;
    let memory = Memory::new(&mut store, MemoryType::new(pages, None))
        .map_err(|e| anyhow::anyhow!("memory: {e}"))?;
    let mut linker: Linker<HostState> = Linker::new(&engine);
    install_math(&mut linker, &mut store);
    linker
        .define("stan", "memory", memory)
        .map_err(|e| anyhow::anyhow!("define memory: {e}"))?;
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .map_err(|e| anyhow::anyhow!("instantiate: {e}"))?;

    let lpg = instance.get_typed_func::<(i32, i32, i32, i32), f64>(&store, "log_prob_grad")?;
    let params_ptr: i32 = 0;
    let grads_ptr: i32 = (n_params * 8) as i32;
    let scratch_ptr: i32 = (n_params * 16) as i32;
    let bytes: Vec<u8> = params.iter().flat_map(|p| p.to_le_bytes()).collect();
    memory
        .write(&mut store, params_ptr as usize, &bytes)
        .map_err(|e| anyhow::anyhow!("memory write: {e}"))?;

    for _ in 0..(N_LPG_ITERS / 10) {
        lpg.call(&mut store, (params_ptr, grads_ptr, n_params as i32, scratch_ptr))?;
    }
    let t0 = Instant::now();
    for _ in 0..N_LPG_ITERS {
        lpg.call(&mut store, (params_ptr, grads_ptr, n_params as i32, scratch_ptr))?;
    }
    let elapsed: Duration = t0.elapsed();
    Ok(elapsed.as_secs_f64() * 1e6 / N_LPG_ITERS as f64)
}
