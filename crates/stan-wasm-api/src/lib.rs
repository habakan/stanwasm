//! Public wasm-bindgen API. Single wasm artifact for the browser.
//!
//! Phase 0: ships only `greet()` to verify the build pipeline.
//! Subsequent phases add `compile()`, `sample()`, `param_names()`, etc.

#![forbid(unsafe_code)]

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn greet(name: &str) -> String {
    format!("hello from stan-wasm-rs, {name}")
}

#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
