//! JSON → `Env` parser for Stan data inputs. Accepts:
//!   {"N": 10, "x": [1, 2, 3], "y": [...]}  // scalars and 1-D arrays
//!
//! Nested arrays (matrices) are flattened in row-major order as `Val::Vec`s
//! of `Val::Vec`s. Booleans and strings are rejected.

use crate::env::Env;
use crate::value::Val;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DataError {
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("top-level data must be a JSON object")]
    NotObject,
    #[error("field {0:?}: unsupported value type (expected number or array)")]
    UnsupportedValue(String),
}

pub fn data_from_json(s: &str) -> Result<Env, DataError> {
    let v: serde_json::Value = serde_json::from_str(s)?;
    let map = v.as_object().ok_or(DataError::NotObject)?;
    let mut env = Env::new();
    for (k, val) in map {
        let parsed = json_to_val(val).ok_or_else(|| DataError::UnsupportedValue(k.clone()))?;
        env.set(k, parsed);
    }
    Ok(env)
}

fn json_to_val(v: &serde_json::Value) -> Option<Val> {
    match v {
        serde_json::Value::Number(n) => n.as_f64().map(Val::Num),
        serde_json::Value::Array(xs) => {
            let mut out = Vec::with_capacity(xs.len());
            for x in xs {
                out.push(json_to_val(x)?);
            }
            Some(Val::Vec(out))
        }
        _ => None,
    }
}
