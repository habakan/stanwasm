//! Public wasm-bindgen API for stanwasm.
//!
//! Single-wasm browser API: parse Stan source, trace once, then run the
//! nuts-rs sampler in-process by replaying the recorded autodiff tape.
//! No JS callback into separate AOT model wasm — sampling, log-prob
//! evaluation, and gradients all happen inside this single wasm module.
//!
//! Also exposes `compile_to_wasm` which returns the AOT model wasm bytes
//! (for callers that want to use the AOT module independently, e.g. in
//! a Web Worker or a non-stanwasm runtime).

#![forbid(unsafe_code)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use nuts_rs::{
    sample_sequentially, Chain, CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, HasDims,
    LogpError, Settings,
};
use rand::{rngs::ChaCha8Rng, SeedableRng};
use stanwasm_runtime::{data_from_json, Compiled, EvalError, Model};
use thiserror::Error;
use wasm_bindgen::prelude::*;

#[derive(Debug, Error)]
enum SamplerError {
    #[error("logp returned non-finite value")]
    NonFinite,
}

impl LogpError for SamplerError {
    fn is_recoverable(&self) -> bool {
        true
    }
}

/// nuts-rs adapter that replays the recorded autodiff tape. Owns the
/// `Compiled` for one `sample()` call so `CpuMath` can take it by value.
struct LogpAdapter {
    compiled: Compiled,
}

impl HasDims for LogpAdapter {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        let n = self.compiled.n_params() as u64;
        [
            ("unconstrained_parameter".to_string(), n),
            ("dim".to_string(), n),
        ]
        .into_iter()
        .collect()
    }
}

impl CpuLogpFunc for LogpAdapter {
    type LogpError = SamplerError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.compiled.n_params()
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, SamplerError> {
        let lp = self.compiled.log_prob_grad(position, gradient);
        if lp.is_finite() {
            Ok(lp)
        } else {
            Err(SamplerError::NonFinite)
        }
    }

    fn expand_vector<R>(&mut self, _rng: &mut R, array: &[f64]) -> Result<Vec<f64>, CpuMathError>
    where
        R: rand::Rng + ?Sized,
    {
        Ok(array.to_vec())
    }
}

/// Concrete type nuts-rs returns from `DiagNutsSettings::new_chain`. It owns
/// its RNG, so it survives across wasm-bindgen calls and can be stepped.
type StepChain = <DiagNutsSettings as Settings>::Chain<CpuMath<LogpAdapter>>;

struct StepSampler {
    chain: StepChain,
    total: u32,
    count: u32,
}

// ---- Public wasm-bindgen surface --------------------------------------------

/// One compiled Stan model: the parsed AST plus a pre-traced `Compiled`.
/// Sampling consumes the `Compiled` and rebuilds it from the AST after.
#[wasm_bindgen]
pub struct StanModel {
    model: Model,
    compiled: Option<Compiled>,
    step: Option<StepSampler>,
    /// Scratch slots the last `compileToWasm` output needs. `sampleViaAot`
    /// cannot size the buffer without it.
    aot_scratch_len: Option<usize>,
}

#[wasm_bindgen]
impl StanModel {
    /// Parse `stan_src`, bind `data_json`, trace the model on the autodiff
    /// tape, and return a handle ready for sampling.
    #[wasm_bindgen(constructor)]
    pub fn new(stan_src: &str, data_json: &str) -> Result<StanModel, JsError> {
        let env = data_from_json(data_json).map_err(jserr)?;
        let model = Model::parse_and_load(stan_src, env).map_err(jserr)?;
        let compiled = Some(trace(&model).map_err(jserr)?);
        Ok(StanModel {
            model,
            compiled,
            step: None,
            aot_scratch_len: None,
        })
    }

    /// Number of unconstrained parameters.
    #[wasm_bindgen(getter)]
    pub fn n_params(&self) -> usize {
        self.model.n_params()
    }

    /// Constrained parameter names (parameters then transformed parameters).
    #[wasm_bindgen(js_name = paramNames)]
    pub fn param_names(&self) -> Vec<String> {
        self.model.param_names()
    }

    /// Evaluate log_prob and gradient at `params`. Returns a flat array of
    /// length `n_params + 1`: the log-prob is at index 0, gradients follow.
    #[wasm_bindgen(js_name = logProbGrad)]
    pub fn log_prob_grad(&mut self, params: &[f64]) -> Result<Vec<f64>, JsError> {
        let compiled = self
            .compiled
            .as_mut()
            .ok_or_else(|| compiled_checked_out("logProbGrad"))?;
        let n = compiled.n_params();
        if params.len() != n {
            return Err(JsError::new(&format!(
                "params length {} != n_params {n}",
                params.len()
            )));
        }
        let mut out = vec![0.0_f64; n + 1];
        let (lp_slot, grads_slot) = out.split_at_mut(1);
        lp_slot[0] = compiled.log_prob_grad(params, grads_slot);
        Ok(out)
    }

    /// Run NUTS sampling. Returns a flat row-major buffer of shape
    /// `(num_warmup + num_draws) × n_params`. Tuning draws come first.
    pub fn sample(
        &mut self,
        init: &[f64],
        num_warmup: u32,
        num_draws: u32,
        seed: u64,
    ) -> Result<Vec<f64>, JsError> {
        let n = self.model.n_params();
        if init.len() != n {
            return Err(JsError::new(&format!(
                "init length {} != n_params {n}",
                init.len()
            )));
        }
        // Widen before adding: `u32 + u32` wraps, and a wrapped total
        // silently becomes a different (possibly enormous) run length.
        let total = num_warmup as u64 + num_draws as u64;

        // Take the Compiled out for nuts-rs (CpuMath consumes by value).
        let compiled = self
            .compiled
            .take()
            .ok_or_else(|| compiled_checked_out("sample"))?;

        // Closure so `self.compiled` is restored on every exit path: a
        // rejected `init` used to leave the model unable to sample again.
        let result: Result<Vec<f64>, JsError> = (|| {
            let math = CpuMath::new(LogpAdapter { compiled });
            let settings = DiagNutsSettings {
                num_tune: num_warmup as u64,
                num_draws: num_draws as u64,
                ..Default::default()
            };
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let iter = sample_sequentially(math, settings, init, total, 0, &mut rng)
                .map_err(|e| JsError::new(&format!("nuts-rs init: {e}")))?;

            let mut out = vec![0.0_f64; n * total as usize];
            for (i, draw) in iter.enumerate() {
                let (pos, _progress) =
                    draw.map_err(|e| JsError::new(&format!("nuts-rs draw: {e}")))?;
                out[i * n..(i + 1) * n].copy_from_slice(pos.as_ref());
            }
            Ok(out)
        })();

        // Restore by re-tracing. Cheap relative to the sampling itself.
        self.compiled = Some(trace(&self.model).map_err(jserr)?);
        result
    }

    /// Constrained `parameters` + `transformed parameters` for one
    /// unconstrained draw, flattened in `paramNames()` order.
    #[wasm_bindgen(js_name = constrainDraw)]
    pub fn constrain_draw(&self, unconstrained: &[f64]) -> Result<Vec<f64>, JsError> {
        let n = self.model.n_params();
        if unconstrained.len() != n {
            return Err(JsError::new(&format!(
                "unconstrained length {} != n_params {n}",
                unconstrained.len()
            )));
        }
        self.model.constrained_draw(unconstrained).map_err(jserr)
    }

    /// Names of the top-level `generated quantities` declarations, flattened
    /// the same way as `paramNames()`.
    #[wasm_bindgen(js_name = genQuantityNames)]
    pub fn gen_quantity_names(&self) -> Vec<String> {
        self.model.gen_quantity_names()
    }

    /// `generated quantities` over row-major `(n_draws, n_params)` draws.
    /// Result is row-major in `genQuantityNames()` order; one seeded RNG stream.
    #[wasm_bindgen(js_name = generatedQuantities)]
    pub fn generated_quantities(
        &self,
        draws: &[f64],
        num_draws: u32,
        seed: u64,
    ) -> Result<Vec<f64>, JsError> {
        let n = self.model.n_params();
        let num_draws = num_draws as usize;
        if draws.len() != n * num_draws {
            return Err(JsError::new(&format!(
                "draws length {} != num_draws * n_params ({num_draws} * {n})",
                draws.len()
            )));
        }
        let n_gq = self.model.gen_quantity_names().len();
        let rng = Rc::new(RefCell::new(ChaCha8Rng::seed_from_u64(seed)));
        let mut out = vec![0.0_f64; n_gq * num_draws];
        for i in 0..num_draws {
            let draw = &draws[i * n..(i + 1) * n];
            let gq = self
                .model
                .generated_quantities(draw, rng.clone())
                .map_err(jserr)?;
            out[i * n_gq..(i + 1) * n_gq].copy_from_slice(&gq);
        }
        Ok(out)
    }

    /// Start a NUTS run that `stepDraw()` advances one draw at a time. Consumes
    /// the `Compiled`: call `finishStepSampling()` before `logProbGrad`/`sample`.
    #[wasm_bindgen(js_name = startStepSampling)]
    pub fn start_step_sampling(
        &mut self,
        init: &[f64],
        num_warmup: u32,
        num_draws: u32,
        seed: u64,
    ) -> Result<(), JsError> {
        let n = self.model.n_params();
        if init.len() != n {
            return Err(JsError::new(&format!(
                "init length {} != n_params {n}",
                init.len()
            )));
        }
        let compiled = self
            .compiled
            .take()
            .ok_or_else(|| compiled_checked_out("startStepSampling"))?;
        let math = CpuMath::new(LogpAdapter { compiled });
        let settings = DiagNutsSettings {
            num_tune: num_warmup as u64,
            num_draws: num_draws as u64,
            ..Default::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut chain = settings.new_chain(0, math, &mut rng);
        if let Err(e) = chain.set_position(init) {
            // Same reasoning as `sample()`: restore `compiled` before
            // returning so a rejected `init` doesn't strand the model.
            self.compiled = Some(trace(&self.model).map_err(jserr)?);
            return Err(JsError::new(&format!("nuts-rs init: {e}")));
        }
        self.step = Some(StepSampler {
            chain,
            total: num_warmup.saturating_add(num_draws),
            count: 0,
        });
        Ok(())
    }

    /// Advance one draw: `n_params` positions, tuning and diverging as `1.0`/`0.0`,
    /// then nuts-rs's own `step_size` and `num_steps`. Restores `sample` when done.
    #[wasm_bindgen(js_name = stepDraw)]
    pub fn step_draw(&mut self) -> Result<Vec<f64>, JsError> {
        let done;
        let out = {
            let step = self
                .step
                .as_mut()
                .ok_or_else(|| JsError::new("call startStepSampling first"))?;
            let (pos, progress) = step
                .chain
                .draw()
                .map_err(|e| JsError::new(&format!("nuts-rs draw: {e}")))?;
            step.count += 1;
            done = step.count >= step.total;
            let mut out = pos.into_vec();
            out.push(if progress.tuning { 1.0 } else { 0.0 });
            out.push(if progress.diverging { 1.0 } else { 0.0 });
            out.push(progress.step_size);
            out.push(progress.num_steps as f64);
            out
        };
        if done {
            self.finish_step_sampling();
        }
        Ok(out)
    }

    /// Stop step-sampling (safe after it ended naturally too) and restore
    /// `logProbGrad`/`sample` by re-tracing.
    #[wasm_bindgen(js_name = finishStepSampling)]
    pub fn finish_step_sampling(&mut self) {
        if self.step.take().is_some() {
            // Re-tracing the same model definition that already traced
            // successfully at construction time cannot fail differently.
            self.compiled =
                Some(trace(&self.model).expect("internal: re-trace of a valid model failed"));
        }
    }

    /// AOT-compile this model to a self-contained wasm module. Pass the bytes to
    /// `WebAssembly.instantiate` for an independent log_prob_grad runtime.
    #[wasm_bindgen(js_name = compileToWasm)]
    pub fn compile_to_wasm(&mut self) -> Result<Vec<u8>, JsError> {
        let dummy = vec![0.1_f64; self.model.n_params()];
        let compiled = stanwasm_codegen::compile(&self.model, &dummy).map_err(jserr)?;
        self.aot_scratch_len = Some(compiled.scratch_len);
        Ok(compiled.wasm)
    }
}

fn trace(model: &Model) -> Result<Compiled, EvalError> {
    let dummy = vec![0.1_f64; model.n_params()];
    Compiled::from(model, &dummy)
}

fn jserr<E: std::fmt::Display>(e: E) -> JsError {
    JsError::new(&e.to_string())
}

/// The one reason `self.compiled` is ever `None`: a step-sampling session has
/// it checked out. Say so, instead of reporting an internal invariant.
fn compiled_checked_out(method: &str) -> JsError {
    JsError::new(&format!(
        "{method} is unavailable while a step-sampling session is running — \
         it holds the compiled model. Call finishStepSampling() first (or \
         exhaust stepDraw(), which calls it for you)."
    ))
}

/// Forwards Rust panics to `console.error` with a message and backtrace rather
/// than an opaque `RuntimeError: unreachable`. Diagnostics: the instance still traps.
#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// AOT bridge: `sample_via_aot` swaps tape replay for a host-provided AOT wasm
// sharing this module's linear memory. Bind it via `setAotExports` first.

#[wasm_bindgen(module = "/js/aot_bridge.js")]
extern "C" {
    #[wasm_bindgen(js_name = aot_logp)]
    fn aot_logp(params_ptr: u32, grads_ptr: u32, n_params: u32, scratch_ptr: u32) -> f64;

    #[wasm_bindgen(js_name = set_aot_exports)]
    fn js_set_aot_exports(exports: JsValue);

    #[wasm_bindgen(js_name = clear_aot_exports)]
    fn js_clear_aot_exports();
}

/// Bind a freshly-instantiated AOT model wasm's exports so subsequent
/// `sampleViaAot` calls dispatch through it. Pass `instance.exports`.
#[wasm_bindgen(js_name = setAotExports)]
pub fn set_aot_exports(exports: JsValue) {
    js_set_aot_exports(exports);
}

/// Release the bound AOT exports. The next `sampleViaAot` call will throw.
#[wasm_bindgen(js_name = clearAotExports)]
pub fn clear_aot_exports() {
    js_clear_aot_exports();
}

/// The linear memory backing this module. Pass as the `stan.memory` import when
/// instantiating an AOT model so the two share buffers.
#[wasm_bindgen(js_name = sharedMemory)]
pub fn shared_memory() -> JsValue {
    wasm_bindgen::memory()
}

struct AotLogp {
    n_params: usize,
    /// Persistent scratch buffer for params (params_ptr) inside our memory.
    params_buf: Vec<f64>,
    /// Persistent scratch buffer for grads (grads_ptr) inside our memory.
    grads_buf: Vec<f64>,
    /// Primal and adjoint storage the AOT module works in, two f64 per node.
    scratch_buf: Vec<f64>,
}

impl HasDims for AotLogp {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        let n = self.n_params as u64;
        [
            ("unconstrained_parameter".to_string(), n),
            ("dim".to_string(), n),
        ]
        .into_iter()
        .collect()
    }
}

impl CpuLogpFunc for AotLogp {
    type LogpError = SamplerError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.n_params
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, SamplerError> {
        // Copy position into the persistent params buffer; capture pointers.
        self.params_buf.copy_from_slice(position);
        let params_ptr = self.params_buf.as_ptr() as u32;
        let grads_ptr = self.grads_buf.as_mut_ptr() as u32;
        let scratch_ptr = self.scratch_buf.as_mut_ptr() as u32;
        let lp = aot_logp(params_ptr, grads_ptr, self.n_params as u32, scratch_ptr);
        gradient.copy_from_slice(&self.grads_buf);
        if lp.is_finite() {
            Ok(lp)
        } else {
            Err(SamplerError::NonFinite)
        }
    }

    fn expand_vector<R>(&mut self, _rng: &mut R, array: &[f64]) -> Result<Vec<f64>, CpuMathError>
    where
        R: rand::Rng + ?Sized,
    {
        Ok(array.to_vec())
    }
}

#[wasm_bindgen]
impl StanModel {
    /// `sample` through a `setAotExports`-bound AOT wasm instead of tape replay;
    /// V8 JITs the unrolled pass. Identical samples for a given seed.
    #[wasm_bindgen(js_name = sampleViaAot)]
    pub fn sample_via_aot(
        &mut self,
        init: &[f64],
        num_warmup: u32,
        num_draws: u32,
        seed: u64,
    ) -> Result<Vec<f64>, JsError> {
        let n = self.model.n_params();
        if init.len() != n {
            return Err(JsError::new(&format!(
                "init length {} != n_params {n}",
                init.len()
            )));
        }
        // Widen before adding: `u32 + u32` wraps, and a wrapped total
        // silently becomes a different (possibly enormous) run length.
        let total = num_warmup as u64 + num_draws as u64;

        let scratch_len = self.aot_scratch_len.ok_or_else(|| {
            JsError::new("call compileToWasm() before sampleViaAot(): the AOT \
                          module works in a scratch buffer this model has not sized yet")
        })?;
        let math = CpuMath::new(AotLogp {
            n_params: n,
            params_buf: vec![0.0; n],
            grads_buf: vec![0.0; n],
            scratch_buf: vec![0.0; scratch_len],
        });

        let settings = DiagNutsSettings {
            num_tune: num_warmup as u64,
            num_draws: num_draws as u64,
            ..Default::default()
        };

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let iter = sample_sequentially(math, settings, init, total, 0, &mut rng)
            .map_err(|e| JsError::new(&format!("nuts-rs init: {e}")))?;

        let mut out = vec![0.0_f64; n * total as usize];
        for (i, draw) in iter.enumerate() {
            let (pos, _progress) = draw.map_err(|e| JsError::new(&format!("nuts-rs draw: {e}")))?;
            out[i * n..(i + 1) * n].copy_from_slice(pos.as_ref());
        }
        Ok(out)
    }
}
