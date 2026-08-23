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
use stan_runtime::{data_from_json, Compiled, EvalError, Model};
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

/// nuts-rs adapter that replays the recorded autodiff tape.
/// Owns the `Compiled` for the duration of one `sample()` call so that
/// `CpuMath` can take it by value; subsequent calls re-build via the public
/// API which holds the `Compiled` separately.
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

/// Concrete type nuts-rs returns from `DiagNutsSettings::new_chain`. Its
/// `draw()` method advances the chain by exactly one iteration and is what
/// makes true step-by-step (not precompute-then-replay) sampling possible:
/// unlike `sample_sequentially`'s iterator, this type owns its RNG outright
/// (seeded once from ours at construction — see nuts-rs's `new_chain`), so
/// it has no borrow tying it to a shorter-lived stack frame and can be
/// stored in `StanModel` across separate wasm-bindgen calls.
type StepChain = <DiagNutsSettings as Settings>::Chain<CpuMath<LogpAdapter>>;

struct StepSampler {
    chain: StepChain,
    total: u32,
    count: u32,
}

// ---- Public wasm-bindgen surface --------------------------------------------

/// One compiled Stan model. Holds both the parsed AST (`Model`) and a
/// pre-traced `Compiled` for fast log-prob evaluation. Sampling consumes
/// the `Compiled` and re-builds it from the retained AST afterwards, so
/// the same `StanModel` instance can be sampled repeatedly.
#[wasm_bindgen]
pub struct StanModel {
    model: Model,
    compiled: Option<Compiled>,
    step: Option<StepSampler>,
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
            .ok_or_else(|| JsError::new("internal: compiled missing"))?;
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
        let total = (num_warmup + num_draws) as u64;

        // Take the Compiled out for nuts-rs (CpuMath consumes by value).
        let compiled = self
            .compiled
            .take()
            .ok_or_else(|| JsError::new("internal: compiled missing — call StanModel anew"))?;
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
            let (pos, _progress) = draw.map_err(|e| JsError::new(&format!("nuts-rs draw: {e}")))?;
            out[i * n..(i + 1) * n].copy_from_slice(pos.as_ref());
        }

        // Restore by re-tracing. Cheap relative to the sampling itself.
        self.compiled = Some(trace(&self.model).map_err(jserr)?);
        Ok(out)
    }

    /// Constrained values of `parameters` + `transformed parameters` for one
    /// unconstrained draw (e.g. one row out of `sample()`'s output), flattened
    /// in `paramNames()` order.
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

    /// Evaluate `generated quantities` for a batch of unconstrained draws
    /// (e.g. `sample()`'s output). `draws` is a flat row-major buffer of
    /// shape `(n_draws, n_params)`; the result is `(n_draws, n_gen_quantities)`,
    /// row-major, in `genQuantityNames()` order. A single RNG stream (seeded
    /// by `seed`) is shared across all draws so repeated `_rng` calls don't
    /// repeat the same values draw-to-draw.
    ///
    /// Note: unlike `sampleViaAot`, there is no AOT-compiled counterpart of
    /// this method — `compileToWasm` only exports `log_prob_grad`. Generated
    /// quantities involve RNG and branching that the flat-tape AOT codegen
    /// doesn't model, and (running once per draw rather than once per NUTS
    /// leapfrog step) don't need it for performance.
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

    /// Start a step-by-step NUTS run: unlike `sample()`, which runs the whole
    /// chain inside one wasm call and returns only at the end, this leaves
    /// the sampler's state alive in the `StanModel` instance so `stepDraw()`
    /// can advance it one draw at a time — genuinely watching the sampler
    /// work, not replaying an already-finished chain. Consumes the internal
    /// `Compiled` the same way `sample()` does; call `finishStepSampling()`
    /// (or exhaust `stepDraw()` up to `num_warmup + num_draws` calls, which
    /// does it automatically) before using `logProbGrad`/`sample` again.
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
            .ok_or_else(|| JsError::new("internal: compiled missing — call StanModel anew"))?;
        let math = CpuMath::new(LogpAdapter { compiled });
        let settings = DiagNutsSettings {
            num_tune: num_warmup as u64,
            num_draws: num_draws as u64,
            ..Default::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut chain = settings.new_chain(0, math, &mut rng);
        chain
            .set_position(init)
            .map_err(|e| JsError::new(&format!("nuts-rs init: {e}")))?;
        self.step = Some(StepSampler {
            chain,
            total: num_warmup + num_draws,
            count: 0,
        });
        Ok(())
    }

    /// Advance the step-sampling chain started by `startStepSampling` by
    /// exactly one draw. Returns a flat array: `n_params` position values,
    /// then `1.0`/`0.0` for whether this draw was still in the warmup
    /// (tuning) phase, then `1.0`/`0.0` for whether it diverged, then the
    /// leapfrog `step_size` and `num_steps` nuts-rs actually used for this
    /// draw — these come straight out of nuts-rs's own dual-averaging
    /// adaptation and trajectory-length search, not anything this crate
    /// computes, so they're a way to show the real sampler internals at
    /// work rather than just the resulting draw. Once the requested
    /// `num_warmup + num_draws` draws have all been returned, this
    /// automatically restores `logProbGrad`/`sample` (by re-tracing, same as
    /// `sample()` does) and further calls fail until `startStepSampling` runs
    /// again.
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

    /// Stop step-sampling early (or clean up after it finished naturally —
    /// safe to call either way) and restore `logProbGrad`/`sample` by
    /// re-tracing, same as `sample()` does at the end of a run.
    #[wasm_bindgen(js_name = finishStepSampling)]
    pub fn finish_step_sampling(&mut self) {
        if self.step.take().is_some() {
            // Re-tracing the same model definition that already traced
            // successfully at construction time cannot fail differently.
            self.compiled =
                Some(trace(&self.model).expect("internal: re-trace of a valid model failed"));
        }
    }

    /// AOT-compile this model to a self-contained wasm module. Returns the
    /// wasm bytes (callers can pass these to `WebAssembly.instantiate` to
    /// obtain an independent log_prob_grad runtime — useful for Web Workers
    /// or for inspection).
    #[wasm_bindgen(js_name = compileToWasm)]
    pub fn compile_to_wasm(&self) -> Result<Vec<u8>, JsError> {
        let dummy = vec![0.1_f64; self.model.n_params()];
        let compiled = stan_codegen::compile(&self.model, &dummy).map_err(jserr)?;
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

/// Runs once when the wasm module is instantiated. Forwards Rust panics
/// (Stan-typo'd names and invalid RNG parameters are now clean `JsError`s
/// instead, but a handful of internal-invariant panics remain, e.g. index
/// out of bounds on a malformed AST) to `console.error` with a real message
/// and backtrace, instead of an opaque `RuntimeError: unreachable`. The
/// panicking call still traps the instance — this is diagnostics, not
/// recovery — but it means a bug report can include what actually broke.
#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// ---- AOT bridge --------------------------------------------------------------
//
// `sample_via_aot` runs the same NUTS sampling driver as `sample`, but
// substitutes the in-process tape replay (`Compiled::log_prob_grad`) with a
// call out to a host-provided AOT-compiled model wasm. The AOT module shares
// `stanwasm`'s linear memory (imported, not its own), so handing it a
// (params_ptr, grads_ptr) is zero-copy.
//
// The host is responsible for instantiating the AOT wasm and binding it via
// `setAotExports` before calling `sampleViaAot`. See `js/aot_bridge.js`.

#[wasm_bindgen(module = "/js/aot_bridge.js")]
extern "C" {
    #[wasm_bindgen(js_name = aot_logp)]
    fn aot_logp(params_ptr: u32, grads_ptr: u32, n_params: u32) -> f64;

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

/// Returns the linear memory backing this wasm module. Pass to
/// `WebAssembly.instantiate` as the `stan.memory` import when bringing up an
/// AOT model so the two modules share buffers (zero-copy bridge).
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
        let lp = aot_logp(params_ptr, grads_ptr, self.n_params as u32);
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
    /// Same as `sample`, but evaluates `log_prob_grad` through a
    /// pre-instantiated AOT-compiled model wasm bound via `setAotExports`.
    /// V8 JITs the unrolled forward+backward pass in the AOT module, which
    /// can be substantially faster than the in-process tape replay used by
    /// `sample`. Both produce identical samples for a given seed.
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
        let total = (num_warmup + num_draws) as u64;

        let math = CpuMath::new(AotLogp {
            n_params: n,
            params_buf: vec![0.0; n],
            grads_buf: vec![0.0; n],
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
