//! Compiled (pre-traced) model. Holds a recorded autodiff tape so subsequent
//! `log_prob_grad` calls replay the static computation graph without re-walking
//! the AST. Used as the inner fast path for the nuts-rs sampling loop.

use crate::error::EvalError;
use crate::model::Model;
use stan_autodiff::Tape;

pub struct Compiled {
    /// Recorded forward-pass tape (frozen at construction).
    tape: Tape,
    /// Tape index of the final log_prob node.
    root: u32,
    /// Number of unconstrained parameters (size of the leaf prefix).
    n_params: usize,
    /// Constrained parameter names in the order produced by `Model::param_names`.
    param_names: Vec<String>,
}

impl Compiled {
    /// Trace `model` once at `dummy_params` to populate the tape.
    ///
    /// Caller is responsible for ensuring the model has no parameter-dependent
    /// control flow — the recorded tape is reused for every subsequent call.
    pub fn from(model: &Model, dummy_params: &[f64]) -> Result<Self, EvalError> {
        assert_eq!(dummy_params.len(), model.n_params());
        let mut tape = Tape::new();
        let leaves: Vec<u32> = dummy_params.iter().map(|p| tape.new_var(*p)).collect();
        let root = model.trace_forward(&mut tape, &leaves, true)?;
        Ok(Self {
            tape,
            root,
            n_params: model.n_params(),
            param_names: model.param_names(),
        })
    }

    pub fn n_params(&self) -> usize {
        self.n_params
    }

    pub fn param_names(&self) -> &[String] {
        &self.param_names
    }

    /// Replay the recorded tape with new parameter values; returns log_prob and
    /// fills `grads_out` with the gradient w.r.t. the unconstrained parameters.
    pub fn log_prob_grad(&mut self, params: &[f64], grads_out: &mut [f64]) -> f64 {
        debug_assert_eq!(params.len(), self.n_params);
        debug_assert_eq!(grads_out.len(), self.n_params);
        self.tape.forward_replay(params);
        self.tape.reset_grads();
        self.tape.backward(self.root);
        for (i, g) in grads_out.iter_mut().enumerate() {
            *g = self.tape.grad_at(i as u32);
        }
        self.tape.value(self.root)
    }

    /// Allocate-and-return convenience form, mostly for tests.
    pub fn log_prob_grad_alloc(&mut self, params: &[f64]) -> (f64, Vec<f64>) {
        let mut grads = vec![0.0; self.n_params];
        let lp = self.log_prob_grad(params, &mut grads);
        (lp, grads)
    }
}
