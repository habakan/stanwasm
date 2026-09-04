//! JSON → `Env` parser for Stan data inputs. Accepts:
//!   {"N": 10, "x": [1, 2, 3], "y": [...]}  // scalars and 1-D arrays
//!
//! Nested arrays (matrices) are flattened in row-major order as `Val::Vec`s
//! of `Val::Vec`s. Booleans and strings are rejected.
//!
//! Deserialised straight into `Val` rather than through `serde_json::Value`.
//! A data block is the largest thing this runtime holds — an MNIST-sized
//! matrix is a gigabyte of `Val` — and the intermediate tree doubled that.
//! Every allocation on the way goes through `try_reserve`, because on wasm a
//! failed one aborts, and an abort is a trap that takes the module instance
//! down rather than returning an error anyone can act on.

use std::fmt;

use crate::env::Env;
use crate::value::Val;
use serde::de::{self, Deserializer, SeqAccess, Visitor};
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
    let mut de = serde_json::Deserializer::from_str(s);
    let fields = de.deserialize_map(TopVisitor)?;
    let mut env = Env::new();
    for (name, val) in fields {
        env.set(&name, val);
    }
    Ok(env)
}

/// The top level is an object of names to values; anything else is rejected
/// here rather than after a whole document has been built.
struct TopVisitor;

impl<'de> Visitor<'de> for TopVisitor {
    type Value = Vec<(String, Val)>;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("an object of data fields")
    }

    fn visit_map<A: de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut out = Vec::new();
        while let Some(name) = map.next_key::<String>()? {
            let val = map.next_value_seed(ValSeed { field: &name })?;
            out.push((name, val));
        }
        Ok(out)
    }
}

/// Carries the field name so a rejection can say which one it was.
struct ValSeed<'a> {
    field: &'a str,
}

impl<'de> de::DeserializeSeed<'de> for ValSeed<'_> {
    type Value = Val;

    fn deserialize<D: Deserializer<'de>>(self, de: D) -> Result<Val, D::Error> {
        de.deserialize_any(ValVisitor { field: self.field })
    }
}

struct ValVisitor<'a> {
    field: &'a str,
}

impl<'de> Visitor<'de> for ValVisitor<'_> {
    type Value = Val;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "field {:?}: a number or an array of them", self.field)
    }

    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Val, E> {
        Ok(Val::Num(v))
    }

    fn visit_i64<E: de::Error>(self, v: i64) -> Result<Val, E> {
        Ok(Val::Num(v as f64))
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> Result<Val, E> {
        Ok(Val::Num(v as f64))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Val, A::Error> {
        let mut out: Vec<Val> = Vec::new();
        let too_big = || {
            de::Error::custom(format!(
                "field {:?} is larger than the memory available. A value costs \
                 {} bytes here, and on wasm the address space stops at 4 GB",
                self.field,
                std::mem::size_of::<Val>()
            ))
        };
        if let Some(n) = seq.size_hint() {
            out.try_reserve_exact(n).map_err(|_| too_big())?;
        }
        while let Some(v) = seq.next_element_seed(ValSeed { field: self.field })? {
            // `push` on a full vector reallocates, and that is the allocation
            // that would abort. Where there is room this is one comparison.
            out.try_reserve(1).map_err(|_| too_big())?;
            out.push(v);
        }
        Ok(Val::Vec(out))
    }
}
