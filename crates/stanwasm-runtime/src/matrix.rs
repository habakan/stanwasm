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

/// Forward substitution: solve `L x = b`, `l_rows` lower-triangular (only
/// entries [0..=i] of row i are read). Returns x as a Vec of scalars.
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

/// An evenly spaced run of tape values, which is what a contraction node can
/// walk: its first index and the distance between elements.
fn tape_run(xs: &[Val]) -> Option<(u32, u32)> {
    let (&Val::Tape(first), &Val::Tape(second)) = (xs.first()?, xs.get(1)?) else {
        return None;
    };
    let stride = second.checked_sub(first).filter(|s| *s > 0)?;
    xs.iter()
        .enumerate()
        .all(|(k, x)| matches!(x, Val::Tape(i) if *i == first + k as u32 * stride))
        .then_some((first, stride))
}

/// Cholesky decomposition of a symmetric positive-definite matrix: `Σ = L Lᵀ`,
/// rows of `Val::Vec`. Lets full-covariance `multi_normal` reuse the Cholesky math.
/// `A * b` where `a_rows` is a vec of row containers. Returns the length-rows vector.
pub fn mat_vec_mul(t: &mut Tape, a_rows: &[Val], b: &[Val]) -> Vec<Val> {
    // Data on the left against parameters on the right — `X * beta`, the shape
    // regression is written in — is one contraction node per row instead of
    // the `2K` multiplies and adds the chain would record.
    if let Some((base, stride)) = tape_run(b) {
        let all_data = a_rows.iter().all(|row| {
            matches!(row, Val::Vec(cells)
                if cells.len() == b.len() && cells.iter().all(|c| matches!(c, Val::Num(_))))
        });
        if all_data {
            let mut out = Vec::with_capacity(a_rows.len());
            let mut coeffs = Vec::with_capacity(b.len());
            for row in a_rows {
                let Val::Vec(cells) = row else {
                    unreachable!("checked above")
                };
                coeffs.clear();
                coeffs.extend(cells.iter().map(|c| match c {
                    Val::Num(v) => *v,
                    _ => unreachable!("checked above"),
                }));
                out.push(Val::Tape(t.dot_c(base, stride, &coeffs)));
            }
            return out;
        }
    }
    a_rows
        .iter()
        .map(|row| {
            let cells = match row {
                Val::Vec(xs) => xs.as_slice(),
                _ => std::slice::from_ref(row),
            };
            let mut acc = Val::Num(0.0);
            for (x, y) in cells.iter().zip(b) {
                let p = v_mul(t, x, y);
                acc = v_add(t, &acc, &p);
            }
            acc
        })
        .collect()
}

/// `A * B`, both vecs of row containers. `b_rows` is indexed by row, so the inner
/// loop walks `b_rows[k][j]` rather than a transposed copy.
pub fn mat_mat_mul(t: &mut Tape, a_rows: &[Val], b_rows: &[Val], cols: usize) -> Vec<Val> {
    let cell = |t: &mut Tape, row: &Val, j: usize| -> Val {
        let cells = match row {
            Val::Vec(xs) => xs.clone(),
            _ => vec![row.clone()],
        };
        let mut acc = Val::Num(0.0);
        for (k, x) in cells.iter().enumerate() {
            let Some(brow) = b_rows.get(k) else { break };
            let bv = match brow {
                Val::Vec(xs) => xs.get(j).cloned(),
                other => Some(other.clone()),
            };
            let Some(bv) = bv else { break };
            let p = v_mul(t, x, &bv);
            acc = v_add(t, &acc, &p);
        }
        acc
    };
    a_rows
        .iter()
        .map(|row| Val::Vec((0..cols).map(|j| cell(t, row, j)).collect()))
        .collect()
}

/// `L Lᵀ` for a lower-triangular `l_rows`. Only the entries up to each row's
/// diagonal are read, so the zeros above it never enter the sum.
pub fn mat_mat_mul_transpose_rhs(t: &mut Tape, l_rows: &[Val], k: usize) -> Vec<Val> {
    let cell = |t: &mut Tape, i: usize, j: usize| -> Val {
        let (Some(Val::Vec(ri)), Some(Val::Vec(rj))) = (l_rows.get(i), l_rows.get(j)) else {
            return Val::Num(0.0);
        };
        let mut acc = Val::Num(0.0);
        for m in 0..=i.min(j) {
            let (Some(a), Some(b)) = (ri.get(m), rj.get(m)) else {
                break;
            };
            let p = v_mul(t, a, b);
            acc = v_add(t, &acc, &p);
        }
        acc
    };
    (0..k)
        .map(|i| Val::Vec((0..k).map(|j| cell(t, i, j)).collect()))
        .collect()
}

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
            // `l[i]` and `l[j]` are both indexed by `k`, so this is not the
            // single-container iteration needless_range_loop has in mind.
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
