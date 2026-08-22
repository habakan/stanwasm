//! Variable environment. Most-recent binding wins (for nested scopes).

use std::cell::RefCell;
use std::rc::Rc;

use crate::value::Val;
use rand::rngs::ChaCha8Rng;

#[derive(Debug, Default, Clone)]
pub struct Env {
    vars: Vec<(String, Val)>,
    /// Shared RNG handle for `_rng` calls (only set while evaluating
    /// `generated quantities`; `Rc` sharing means a clone of `Env` still
    /// advances the same underlying stream).
    rng: Option<Rc<RefCell<ChaCha8Rng>>>,
}

impl Env {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, name: &str, val: Val) {
        for slot in self.vars.iter_mut() {
            if slot.0 == name {
                slot.1 = val;
                return;
            }
        }
        self.vars.push((name.to_string(), val));
    }

    /// Push a new binding without overwriting earlier ones — caller can
    /// later `pop_to(saved_len)` to restore the scope.
    pub fn push(&mut self, name: &str, val: Val) {
        self.vars.push((name.to_string(), val));
    }

    pub fn get(&self, name: &str) -> Option<&Val> {
        self.vars
            .iter()
            .rev()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v)
    }

    pub fn len(&self) -> usize {
        self.vars.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vars.is_empty()
    }

    pub fn truncate(&mut self, len: usize) {
        self.vars.truncate(len);
    }

    pub fn set_scalar(&mut self, name: &str, v: f64) {
        self.set(name, Val::Num(v));
    }

    pub fn set_vector(&mut self, name: &str, xs: &[f64]) {
        self.set(name, Val::Vec(xs.iter().map(|x| Val::Num(*x)).collect()));
    }

    pub fn set_rng(&mut self, rng: Rc<RefCell<ChaCha8Rng>>) {
        self.rng = Some(rng);
    }

    pub fn rng(&self) -> Option<Rc<RefCell<ChaCha8Rng>>> {
        self.rng.clone()
    }
}
