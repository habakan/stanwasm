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
use rand::{
    distr::{Distribution, Uniform},
    rngs::ChaCha8Rng,
    SeedableRng,
};
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

/// nuts-rs adapter that records a fresh tape for every gradient, instead of
/// replaying one. About six times the cost of replay, which buys back the
/// things a recorded graph cannot follow: a branch on a parameter, a loop whose
/// length one decides, an adaptive solver choosing its own steps.
struct FreshLogp {
    model: Rc<Model>,
}

impl HasDims for FreshLogp {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        let n = self.model.n_params() as u64;
        [
            ("unconstrained_parameter".to_string(), n),
            ("dim".to_string(), n),
        ]
        .into_iter()
        .collect()
    }
}

impl CpuLogpFunc for FreshLogp {
    type LogpError = SamplerError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.model.n_params()
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, SamplerError> {
        // This path exists for models only some points evaluate, so an error here is
        // a rejected proposal, not a fault.
        let lp = self
            .model
            .log_prob_grad_into(position, gradient)
            .map_err(|_| SamplerError::NonFinite)?;
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
    model: Rc<Model>,
    compiled: Option<Compiled>,
    step: Option<StepSampler>,
    /// What the last `compileToWasm` left for `sampleViaAot`. One field rather
    /// than two, so the scratch buffer and the id of the module it belongs to
    /// cannot be set apart from each other.
    aot: Option<AotBuild>,
}

/// The half of a `compileToWasm` result that stays behind on this side.
struct AotBuild {
    /// Initial scratch contents: zeroed primals and adjoints, then the
    /// re-rolled loops' constant table.
    scratch_init: Vec<f64>,
    /// The emitted module's `stanwasm_layout_id` global.
    layout_id: u32,
}

/// nuts-rs asserts that its step-size adaptation has somewhere to run, and an
/// assertion inside wasm is a trap the caller cannot tell apart from any other.
fn no_warmup_check(num_warmup: u32) -> Result<(), JsError> {
    if num_warmup == 0 {
        return Err(JsError::new(
            "num_warmup must be at least 1: the sampler adapts its step size \
             during warmup and has no schedule to do it on with none",
        ));
    }
    Ok(())
}

/// nuts-rs refuses a starting point whose gradient has a zero component — the
/// mass matrix it adapts is scaled by that gradient — and reports only
/// "Invalid initial point", which names neither the rule nor the parameter.
pub fn init_gradient_check(names: &[String], lp: f64, grad: &[f64]) -> Result<(), String> {
    if !lp.is_finite() {
        return Err(format!(
            "log density is {lp} at the starting point; the sampler needs a \
             finite one to begin from"
        ));
    }
    let named = |predicate: fn(f64) -> bool| {
        grad.iter()
            .enumerate()
            .filter(|(_, g)| predicate(**g))
            .map(|(i, _)| names.get(i).map_or("?", String::as_str))
            .collect::<Vec<_>>()
    };
    let nan = named(f64::is_nan);
    if !nan.is_empty() {
        return Err(format!(
            "gradient is NaN for {} of the {} parameters at the starting point \
             ({}); this points to numerical arithmetic rather than a \
             structurally flat model",
            nan.len(),
            grad.len(),
            nan.join(", "),
        ));
    }
    let infinite = named(f64::is_infinite);
    if !infinite.is_empty() {
        return Err(format!(
            "gradient is infinite for {} of the {} parameters at the starting \
             point ({}); this points to numerical arithmetic rather than a \
             structurally flat model",
            infinite.len(),
            grad.len(),
            infinite.join(", "),
        ));
    }
    let bad: Vec<&str> = grad
        .iter()
        .enumerate()
        .filter(|(_, g)| **g == 0.0)
        .map(|(i, _)| names.get(i).map_or("?", String::as_str))
        .collect();
    if bad.is_empty() {
        return Ok(());
    }
    let shown = bad.iter().take(6).copied().collect::<Vec<_>>().join(", ");
    let rest = if bad.len() > 6 {
        format!(" and {} more", bad.len() - 6)
    } else {
        String::new()
    };
    Err(format!(
        "the log density does not move with {} of the {} parameters at the \
         starting point ({shown}{rest}), and the sampler cannot begin from \
         there. `randomInit(seed)` finds one, or drop the parameters the data \
         says nothing about",
        bad.len(),
        grad.len(),
    ))
}

#[wasm_bindgen]
impl StanModel {
    /// Parse `stan_src`, bind `data_json`, trace the model on the autodiff
    /// tape, and return a handle ready for sampling.
    #[wasm_bindgen(constructor)]
    pub fn new(stan_src: &str, data_json: &str) -> Result<StanModel, JsError> {
        let env = data_from_json(data_json).map_err(jserr)?;
        let model = Model::parse_and_load(stan_src, env).map_err(jserr)?;
        // A model whose graph moves with the parameters cannot be recorded once, but
        // still samples by re-recording, so loading keeps going without a `Compiled`.
        let compiled = match trace(&model) {
            Ok(c) => Some(c),
            Err(e) if needs_fresh_trace(&e) => None,
            Err(e) => return Err(jserr(e)),
        };
        Ok(StanModel {
            model: Rc::new(model),
            compiled,
            step: None,
            aot: None,
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
    /// Refuse a starting point the sampler would reject, while the names of
    /// the parameters that make it one are still to hand.
    fn check_start(&mut self, init: &[f64]) -> Result<(), JsError> {
        self.check_start_inner(init).map_err(|e| JsError::new(&e))
    }

    fn check_start_inner(&mut self, init: &[f64]) -> Result<(), String> {
        let names = self.model.unconstrained_param_names();
        let compiled = self
            .compiled
            .as_mut()
            .ok_or_else(|| "sample: the compiled tape is checked out".to_string())?;
        let mut grad = vec![0.0_f64; init.len()];
        let lp = compiled.log_prob_grad(init, &mut grad);
        init_gradient_check(&names, lp, &grad)
    }

    /// A starting point the sampler will accept: uniform on `[-2, 2]` per
    /// unconstrained parameter, the way CmdStan initialises, redrawn until the
    /// gradient there has no zero component. `sample` still takes whatever it
    /// is given — this only removes the guesswork from finding one.
    #[wasm_bindgen(js_name = randomInit)]
    pub fn random_init(&mut self, seed: u64) -> Result<Vec<f64>, JsError> {
        const TRIES: u32 = 100;
        let n = self.model.n_params();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let span = Uniform::new(-2.0_f64, 2.0).map_err(|e| JsError::new(&e.to_string()))?;
        let mut last = String::new();
        for _ in 0..TRIES {
            let at: Vec<f64> = (0..n).map(|_| span.sample(&mut rng)).collect();
            match self.check_start(&at) {
                Ok(()) => return Ok(at),
                Err(_) => last = self.start_problem(&at),
            }
        }
        Err(JsError::new(&format!(
            "no starting point in [-2, 2] worked in {TRIES} tries. The last one \
             failed because {last}"
        )))
    }

    /// Why `check_start` refused, as text, for the message above.
    fn start_problem(&mut self, at: &[f64]) -> String {
        self.check_start_inner(at).err().unwrap_or_default()
    }

    #[wasm_bindgen(js_name = logProbGrad)]
    pub fn log_prob_grad(&mut self, params: &[f64]) -> Result<Vec<f64>, JsError> {
        if self.compiled.is_none() && self.step.is_none() {
            // No recorded tape because the model does not have one shape. Trace
            // it again here, which is what `sampleFresh` does per gradient.
            let n = self.model.n_params();
            if params.len() != n {
                return Err(JsError::new(&format!(
                    "params length {} != n_params {n}",
                    params.len()
                )));
            }
            let mut out = vec![0.0_f64; n + 1];
            let lp = self
                .model
                .log_prob_grad_into(params, &mut out[1..])
                .map_err(jserr)?;
            out[0] = lp;
            return Ok(out);
        }
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
        no_warmup_check(num_warmup)?;
        // Before the start check, which reads a tape this model may not have.
        if self.compiled.is_none() && self.step.is_none() {
            return Err(no_recorded_tape("sample"));
        }
        self.check_start(init)?;
        // Widen before adding: `u32 + u32` wraps, and a wrapped total
        // silently becomes a different (possibly enormous) run length.
        let total = num_warmup as u64 + num_draws as u64;

        // Take the Compiled out for nuts-rs (CpuMath consumes by value).
        let compiled = self
            .compiled
            .take()
            .ok_or_else(|| compiled_checked_out("sample"))?;

        // A closure so `self.compiled` is restored on every exit path.
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

    /// `sample()`, but recording a fresh tape for every gradient rather than
    /// replaying one. About six times slower, and the only path that can run a
    /// model whose computation changes with the parameters — a branch on one, a
    /// loop it sizes, an adaptive ODE solver picking its own steps. Those are
    /// refused on the replay path rather than frozen at the tracing point.
    #[wasm_bindgen(js_name = sampleFresh)]
    pub fn sample_fresh(
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
        no_warmup_check(num_warmup)?;
        let total = num_warmup as u64 + num_draws as u64;

        let math = CpuMath::new(FreshLogp {
            model: Rc::clone(&self.model),
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

    /// Constrained `parameters` + `transformed parameters` for one
    /// unconstrained draw, flattened in `paramNames()` order. Only the first
    /// `constrainedParamNames().length` values can be passed to `unconstrainDraw()`.
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

    /// Names of the `parameters` block alone, which is the order
    /// `unconstrainDraw()` reads. `paramNames()` also covers `transformed
    /// parameters`, and those are derived rather than free.
    #[wasm_bindgen(js_name = constrainedParamNames)]
    pub fn constrained_param_names(&self) -> Vec<String> {
        self.model.constrained_param_names()
    }

    /// Inverse of `constrainDraw()` over the `parameters` block: a draw fitted
    /// elsewhere, in `constrainedParamNames()` order, becomes the unconstrained
    /// vector `logProbGrad()` and `generatedQuantities()` take.
    #[wasm_bindgen(js_name = unconstrainDraw)]
    pub fn unconstrain_draw(&self, constrained: &[f64]) -> Result<Vec<f64>, JsError> {
        self.model.unconstrain_draw(constrained).map_err(jserr)
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
        no_warmup_check(num_warmup)?;
        let compiled = self.compiled.take().ok_or_else(|| {
            if self.step.is_none() {
                no_recorded_tape("startStepSampling")
            } else {
                compiled_checked_out("startStepSampling")
            }
        })?;
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
            // The same definition already traced at construction time.
            self.compiled =
                Some(trace(&self.model).expect("internal: re-trace of a valid model failed"));
        }
    }

    /// AOT-compile this model to a self-contained wasm module. Pass the bytes to
    /// `WebAssembly.instantiate` for an independent log_prob_grad runtime.
    #[wasm_bindgen(js_name = compileToWasm)]
    /// `reroll` selects how vectorised statements are lowered: `"auto"`
    /// (default), `"always"`, or `"never"`. Which is faster is an engine
    /// preference — Safari prefers `"always"`, Chrome and Firefox `"auto"` —
    /// so the page, which knows what it is running on, gets to choose.
    pub fn compile_to_wasm(&mut self, reroll: Option<String>) -> Result<Vec<u8>, JsError> {
        let mode = match reroll.as_deref() {
            None | Some("auto") => stanwasm_codegen::Reroll::Auto,
            Some("always") => stanwasm_codegen::Reroll::Always,
            Some("never") => stanwasm_codegen::Reroll::Never,
            Some(other) => {
                return Err(JsError::new(&format!(
                    "reroll must be \"auto\", \"always\" or \"never\", got {other:?}"
                )))
            }
        };
        let dummy = vec![0.1_f64; self.model.n_params()];
        let compiled = stanwasm_codegen::compile_with(&self.model, &dummy, mode).map_err(|e| {
            // A graph that moves with the parameters would freeze at `dummy`.
            if matches!(&e, stanwasm_codegen::CodegenError::Eval(inner) if needs_fresh_trace(inner))
            {
                no_recorded_tape("compileToWasm")
            } else {
                jserr(e)
            }
        })?;
        let mut scratch = vec![0.0_f64; compiled.scratch_len];
        let at = compiled.scratch_len - compiled.const_table.len();
        scratch[at..].copy_from_slice(&compiled.const_table);
        self.aot = Some(AotBuild {
            scratch_init: scratch,
            layout_id: compiled.layout_id,
        });
        Ok(compiled.wasm)
    }
}

/// Whether a load-time trace failed because the model's computation depends on
/// the parameters, rather than because the model is wrong. Only these fall back
/// to the fresh-trace path; everything else is still a load error.
fn needs_fresh_trace(e: &EvalError) -> bool {
    matches!(
        e,
        EvalError::UnsupportedOdeIntegrator(_) | EvalError::ParamDependentBranch
    )
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

/// The model has no recorded tape because its computation changes with the
/// parameters. Says which method does work rather than only what does not.
fn no_recorded_tape(method: &str) -> JsError {
    JsError::new(&format!(
        "{method} needs a recorded tape, and this model does not have one — its \
         computation depends on the parameters (an adaptive ODE solver, or a \
         branch on a parameter), so a graph recorded once would be wrong \
         everywhere else. logProbGrad() and sampleFresh() work on this model: \
         they re-record per gradient, at about six times the cost of replay."
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

    /// The bound module's `stanwasm_layout_id`, or NaN when nothing is bound.
    #[wasm_bindgen(js_name = aot_layout_id)]
    fn aot_layout_id() -> f64;
}

/// Refuse a binding that belongs to a different compilation.
///
/// `setAotExports` binds one module for the whole page, while the scratch
/// buffer belongs to a single model. Running model B's module against model
/// A's buffer writes at slot offsets A never sized for, so this compares the
/// two ids before the sampler takes the buffer.
fn check_aot_binding(want: u32) -> Result<(), JsError> {
    let bound = aot_layout_id();
    if bound.is_nan() {
        return Err(JsError::new(
            "no AOT module is bound: call setAotExports(instance.exports) with \
             the module this model's compileToWasm() returned. A module built \
             by an older stanwasm exports no layout id and cannot be checked, \
             so it is refused here too.",
        ));
    }
    if bound as u32 != want {
        return Err(JsError::new(
            "the bound AOT module was compiled for a different model. \
             setAotExports() binds one module per page, so re-bind this \
             model's own compileToWasm() output before sampling it — the \
             module reads and writes a scratch buffer laid out for the model \
             it was compiled from.",
        ));
    }
    Ok(())
}

/// Bind a freshly-instantiated AOT model wasm's exports so subsequent
/// `sampleViaAot` calls dispatch through it. Pass `instance.exports`.
///
/// One binding serves the whole page, so a page holding several models has to
/// re-bind before sampling a different one. `sampleViaAot` compares the bound
/// module's `stanwasm_layout_id` against the model it belongs to and refuses
/// the pair rather than running it against the wrong scratch buffer.
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
    /// Initial contents of the scratch buffer the AOT module works in: zeroed
    /// primals and adjoints, then the constants its re-rolled loops read.
    /// Callers driving `log_prob_grad` themselves must stage this once;
    /// `sampleViaAot` does it internally.
    #[wasm_bindgen(js_name = aotScratchInit)]
    pub fn aot_scratch_init(&self) -> Result<Vec<f64>, JsError> {
        self.aot
            .as_ref()
            .map(|a| a.scratch_init.clone())
            .ok_or_else(|| {
                JsError::new("call compileToWasm() first: the scratch layout comes from it")
            })
    }

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
        no_warmup_check(num_warmup)?;
        self.check_start(init)?;
        // Widen before adding: `u32 + u32` wraps, and a wrapped total
        // silently becomes a different (possibly enormous) run length.
        let total = num_warmup as u64 + num_draws as u64;

        let built = self.aot.as_ref().ok_or_else(|| {
            JsError::new(
                "call compileToWasm() before sampleViaAot(): the AOT \
                          module works in a scratch buffer this model has not built yet",
            )
        })?;
        check_aot_binding(built.layout_id)?;
        let scratch_buf = built.scratch_init.clone();
        let math = CpuMath::new(AotLogp {
            n_params: n,
            params_buf: vec![0.0; n],
            grads_buf: vec![0.0; n],
            scratch_buf,
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

/// A sampler over a module compiled somewhere else.
///
/// [`StanModel`] reaches the AOT path through a parsed model. This reaches it
/// through the module alone, for a caller that compiled one ahead of time —
/// the parser, the evaluator and the constraint transforms are not on this
/// path. Bind the module with `setAotExports` first, exactly as for
/// `sampleViaAot`.
#[wasm_bindgen]
pub struct AotSampler {
    n_params: usize,
    scratch_init: Vec<f64>,
    layout_id: u32,
    param_names: Vec<String>,
}

#[wasm_bindgen]
impl AotSampler {
    /// `scratch_init` is the buffer the module works in — zeroed primals and
    /// adjoints followed by the re-rolled loops' constant table, which is what
    /// `aotScratchInit` returns and what a compiler should record beside the
    /// module. `layout_id` is the module's `stanwasm_layout_id` global, checked
    /// against the bound exports before each run so a module and a scratch
    /// buffer built for different models cannot be used together.
    ///
    /// `param_names` only names a parameter in a rejected starting point; pass
    /// an empty array to go without.
    #[wasm_bindgen(constructor)]
    pub fn new(
        n_params: usize,
        scratch_init: Vec<f64>,
        layout_id: u32,
        param_names: Vec<String>,
    ) -> Result<AotSampler, JsError> {
        if n_params == 0 {
            return Err(JsError::new("n_params must be at least 1"));
        }
        if scratch_init.len() < 2 * n_params {
            return Err(JsError::new(&format!(
                "scratch_init has {} slots, too few for {n_params} parameters",
                scratch_init.len()
            )));
        }
        if !param_names.is_empty() && param_names.len() != n_params {
            return Err(JsError::new(&format!(
                "param_names has {} entries but n_params is {n_params}",
                param_names.len()
            )));
        }
        Ok(Self {
            n_params,
            scratch_init,
            layout_id,
            param_names,
        })
    }

    #[wasm_bindgen(getter, js_name = nParams)]
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    fn logp_fn(&self) -> AotLogp {
        AotLogp {
            n_params: self.n_params,
            params_buf: vec![0.0; self.n_params],
            grads_buf: vec![0.0; self.n_params],
            scratch_buf: self.scratch_init.clone(),
        }
    }

    /// `[log_prob, d/dparam...]`, the shape [`StanModel::log_prob_grad`] uses.
    #[wasm_bindgen(js_name = logProbGrad)]
    pub fn log_prob_grad(&self, params: &[f64]) -> Result<Vec<f64>, JsError> {
        if params.len() != self.n_params {
            return Err(JsError::new(&format!(
                "params length {} != n_params {}",
                params.len(),
                self.n_params
            )));
        }
        check_aot_binding(self.layout_id)?;
        let mut out = vec![0.0_f64; self.n_params + 1];
        let lp = self
            .logp_fn()
            .logp(params, &mut out[1..])
            .map_err(|e| JsError::new(&format!("{e}")))?;
        out[0] = lp;
        Ok(out)
    }

    /// `num_warmup + num_draws` draws, row-major, `n_params` wide.
    pub fn sample(
        &self,
        init: &[f64],
        num_warmup: u32,
        num_draws: u32,
        seed: u64,
    ) -> Result<Vec<f64>, JsError> {
        let n = self.n_params;
        if init.len() != n {
            return Err(JsError::new(&format!(
                "init length {} != n_params {n}",
                init.len()
            )));
        }
        no_warmup_check(num_warmup)?;
        check_aot_binding(self.layout_id)?;

        let mut grad = vec![0.0_f64; n];
        let lp = self
            .logp_fn()
            .logp(init, &mut grad)
            .map_err(|e| JsError::new(&format!("{e}")))?;
        init_gradient_check(&self.param_names, lp, &grad).map_err(|e| JsError::new(&e))?;

        // Widen before adding: `u32 + u32` wraps, and a wrapped total
        // silently becomes a different (possibly enormous) run length.
        let total = num_warmup as u64 + num_draws as u64;
        let math = CpuMath::new(self.logp_fn());
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
