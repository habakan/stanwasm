//! Matrix / vector helpers used by multivariate distributions and constraint
//! transforms. A matrix is represented as `Val::Vec` of `Val::Vec` rows.

use crate::ops::{v_add, v_div, v_mul, v_sub};
use crate::value::Val;
use stanwasm_autodiff::Tape;

/// Sum of squares: ∑ vᵢ²
pub fn vec_dot_self(t: &mut Tape, v: &[Val]) -> Val {
    let mut acc = Val::Num(0.0);
    for vi in v {
        let sq = v_mul(t, vi, vi);
        acc = v_add(t, &acc, &sq);
    }
    acc
}

/// Forward substitution: solve `L x = b` where `l_rows` is lower-triangular
/// (each row is `Val::Vec`; only entries [0..=i] of row i are read).
/// Returns x as a Vec of scalars.
pub fn mat_mdiv_ltri_low(t: &mut Tape, l_rows: &[Val], b: &[Val]) -> Vec<Val> {
    let n = b.len();
    let mut x = Vec::with_capacity(n);
    for i in 0..n {
        let mut s = b[i].clone();
        if let Val::Vec(row) = &l_rows[i] {
            for j in 0..i {
                let prod = v_mul(t, &row[j], &x[j]);
                s = v_sub(t, &s, &prod);
            }
            let diag = if i < row.len() {
                row[i].clone()
            } else {
                Val::Num(1.0)
            };
            x.push(v_div(t, &s, &diag));
        } else {
            x.push(s);
        }
    }
    x
}
