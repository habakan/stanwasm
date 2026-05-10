//! Reverse-mode autodiff tape (flat-array, no GC allocations).
//! Phase 2 will port `compiler/autodiff/autodiff.mbt`.

#![forbid(unsafe_code)]

#[derive(Debug, Default)]
pub struct Tape {
    _val: Vec<f64>,
    _grad: Vec<f64>,
}

impl Tape {
    pub fn new() -> Self {
        Self::default()
    }
}
