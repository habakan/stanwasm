//! Variable environment. Most-recent binding wins (for nested scopes).

use crate::value::Val;

#[derive(Debug, Default, Clone)]
pub struct Env {
    vars: Vec<(String, Val)>,
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
}
