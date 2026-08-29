//! Matrix / vector helpers used by multivariate distributions and constraint
//! transforms. A matrix is represented as `Val::Vec` of `Val::Vec` rows.

use crate::ops::{v_add, v_div, v_mul, v_sqrt, v_sub};
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

/// Cholesky decomposition of a symmetric positive-definite matrix (`sigma_rows`,
/// a vec of rows): returns the lower-triangular `L` with `Σ = L Lᵀ`, in the
/// same rows-of-`Val::Vec` shape used elsewhere. Used to turn a full-covariance
/// `multi_normal` into the same math as `multi_normal_cholesky`, since
/// `log det Σ = 2 Σ log Lᵢᵢ`.
pub fn cholesky_decompose(t: &mut Tape, sigma_rows: &[Val]) -> Vec<Val> {
    let n = sigma_rows.len();
    let mut l: Vec<Vec<Val>> = vec![Vec::new(); n];
    for i in 0..n {
        let row_i: Vec<Val> = match &sigma_rows[i] {
            Val::Vec(r) => r.clone(),
            other => vec![other.clone()],
        };
        for j in 0..=i {
            let mut sum = row_i.get(j).cloned().unwrap_or(Val::Num(0.0));
            // Both `l[i]` and `l[j]` are indexed by `k`, so this isn't a
            // single-container iteration clippy's needless_range_loop lint
            // has in mind.
            #[allow(clippy::needless_range_loop)]
            for k in 0..j {
                let prod = v_mul(t, &l[i][k], &l[j][k]);
                sum = v_sub(t, &sum, &prod);
            }
            if i == j {
                l[i].push(v_sqrt(t, &sum));
            } else {
                let ljj = l[j][j].clone();
                l[i].push(v_div(t, &sum, &ljj));
            }
        }
    }
    l.into_iter().map(Val::Vec).collect()
}
