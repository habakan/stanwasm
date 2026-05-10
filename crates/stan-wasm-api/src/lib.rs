//! Public wasm-bindgen API for stan-wasm-rs.
//!
//! Single-wasm browser API: parse Stan source, trace once, then run the
//! nuts-rs sampler in-process by replaying the recorded autodiff tape.
//! No JS callback into separate AOT model wasm — sampling, log-prob
//! evaluation, and gradients all happen inside this single wasm module.
//!
//! Also exposes `compile_to_wasm` which returns the AOT model wasm bytes
//! (for callers that want to use the AOT module independently, e.g. in
//! a Web Worker or a non-stan-wasm-rs runtime).

#![forbid(unsafe_code)]

use std::collections::HashMap;

use nuts_rs::{
    sample_sequentially, CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, HasDims, LogpError,
};
use rand::{rngs::ChaCha8Rng, SeedableRng};
use stan_runtime::{data_from_json, Compiled, Model};
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

    fn logp(
        &mut self,
        position: &[f64],
        gradient: &mut [f64],
    ) -> Result<f64, SamplerError> {
        let lp = self.compiled.log_prob_grad(position, gradient);
        if lp.is_finite() {
            Ok(lp)
        } else {
            Err(SamplerError::NonFinite)
        }
    }

    fn expand_vector<R>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> Result<Vec<f64>, CpuMathError>
    where
        R: rand::Rng + ?Sized,
    {
        Ok(array.to_vec())
    }
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
}

#[wasm_bindgen]
impl StanModel {
    /// Parse `stan_src`, bind `data_json`, trace the model on the autodiff
    /// tape, and return a handle ready for sampling.
    #[wasm_bindgen(constructor)]
    pub fn new(stan_src: &str, data_json: &str) -> Result<StanModel, JsError> {
        let env = data_from_json(data_json).map_err(jserr)?;
        let model = Model::parse_and_load(stan_src, env).map_err(jserr)?;
        let compiled = Some(trace(&model));
        Ok(StanModel { model, compiled })
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
        let compiled = self.compiled.take().ok_or_else(|| {
            JsError::new("internal: compiled missing — call StanModel anew")
        })?;
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
        self.compiled = Some(trace(&self.model));
        Ok(out)
    }

    /// AOT-compile this model to a self-contained wasm module. Returns the
    /// wasm bytes (callers can pass these to `WebAssembly.instantiate` to
    /// obtain an independent log_prob_grad runtime — useful for Web Workers
    /// or for inspection).
    #[wasm_bindgen(js_name = compileToWasm)]
    pub fn compile_to_wasm(&self) -> Result<Vec<u8>, JsError> {
        let dummy = vec![0.1_f64; self.model.n_params()];
        let compiled =
            stan_codegen::compile(&self.model, &dummy).map_err(jserr)?;
        Ok(compiled.wasm)
    }
}

fn trace(model: &Model) -> Compiled {
    let dummy = vec![0.1_f64; model.n_params()];
    Compiled::from(model, &dummy)
}

fn jserr<E: std::fmt::Display>(e: E) -> JsError {
    JsError::new(&e.to_string())
}

// Hello-world placeholders kept for build-system smoke tests; they appear in
// the wasm binary only because they are #[wasm_bindgen]-annotated. Cheap to
// keep until Phase 7 cleanup.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
