//! Arithmetic and transcendental operations on `Val`.
//!
//! Each function takes a `&mut Tape` to push autodiff nodes when needed.
//! Plain-number paths short-circuit without touching the tape. Unary math
//! functions (`exp`, `log`, `abs`, `lgamma`, `Phi`) broadcast element-wise
//! over `Val::Vec` — Stan's built-in math functions are vectorized over
//! containers (e.g. `vector[N] y = exp(x);` is standard Stan).

use crate::value::Val;
use stanwasm_autodiff::{
    lgamma as lgamma_double, phi_cdf as phi_cdf_double, student_t_lccdf as student_t_lccdf_double,
    Tape,
};

// For Vec broadcasts we collect into Vec<Val> via a for-loop so the
// recursive call has a clear borrow lifetime on the tape.

fn map_pair(
    t: &mut Tape,
    xs: &[Val],
    ys: &[Val],
    op: fn(&mut Tape, &Val, &Val) -> Val,
) -> Vec<Val> {
    xs.iter().zip(ys.iter()).map(|(x, y)| op(t, x, y)).collect()
}

fn map_left(t: &mut Tape, xs: &[Val], rhs: &Val, op: fn(&mut Tape, &Val, &Val) -> Val) -> Vec<Val> {
    xs.iter().map(|x| op(t, x, rhs)).collect()
}

fn map_right(
    t: &mut Tape,
    lhs: &Val,
    ys: &[Val],
    op: fn(&mut Tape, &Val, &Val) -> Val,
) -> Vec<Val> {
    ys.iter().map(|y| op(t, lhs, y)).collect()
}

/// Element-wise broadcast for the container cases. The result takes the
/// container operand's orientation, so a row stays a row.
fn broadcast(t: &mut Tape, a: &Val, b: &Val, op: fn(&mut Tape, &Val, &Val) -> Val) -> Val {
    match (a.elems(), b.elems()) {
        (Some(xs), Some(ys)) => a.like(map_pair(t, xs, ys, op)),
        (Some(xs), _) => a.like(map_left(t, xs, b, op)),
        (_, Some(ys)) => b.like(map_right(t, a, ys, op)),
        _ => unreachable!("every caller matches the two-scalar pairs first"),
    }
}

pub fn v_add(t: &mut Tape, a: &Val, b: &Val) -> Val {
    match (a, b) {
        (Val::Num(x), Val::Num(y)) => Val::Num(x + y),
        (Val::Tape(i), Val::Num(y)) => Val::Tape(t.add_c(*i, *y)),
        (Val::Num(x), Val::Tape(j)) => Val::Tape(t.add_c(*j, *x)),
        (Val::Tape(i), Val::Tape(j)) => Val::Tape(t.add(*i, *j)),
        _ => broadcast(t, a, b, v_add),
    }
}

pub fn v_sub(t: &mut Tape, a: &Val, b: &Val) -> Val {
    match (a, b) {
        (Val::Num(x), Val::Num(y)) => Val::Num(x - y),
        (Val::Tape(i), Val::Num(y)) => Val::Tape(t.sub_c(*i, *y)),
        (Val::Num(x), Val::Tape(j)) => Val::Tape(t.rsub_c(*j, *x)),
        (Val::Tape(i), Val::Tape(j)) => Val::Tape(t.sub(*i, *j)),
        _ => broadcast(t, a, b, v_sub),
    }
}

pub fn v_mul(t: &mut Tape, a: &Val, b: &Val) -> Val {
    match (a, b) {
        (Val::Num(x), Val::Num(y)) => Val::Num(x * y),
        (Val::Tape(i), Val::Num(y)) => Val::Tape(t.mul_c(*i, *y)),
        (Val::Num(x), Val::Tape(j)) => Val::Tape(t.mul_c(*j, *x)),
        (Val::Tape(i), Val::Tape(j)) => Val::Tape(t.mul(*i, *j)),
        _ => broadcast(t, a, b, v_mul),
    }
}

pub fn v_div(t: &mut Tape, a: &Val, b: &Val) -> Val {
    match (a, b) {
        (Val::Num(x), Val::Num(y)) => Val::Num(x / y),
        (Val::Tape(i), Val::Num(y)) => Val::Tape(t.div_c(*i, *y)),
        (Val::Num(x), Val::Tape(j)) => Val::Tape(t.rdiv_c(*j, *x)),
        (Val::Tape(i), Val::Tape(j)) => Val::Tape(t.div(*i, *j)),
        _ => broadcast(t, a, b, v_div),
    }
}

/// The longest suffix of `terms` that is an evenly spaced run of tape values:
/// its length, first tape index, and the spacing between elements.
fn run_suffix(terms: &[Val]) -> Option<(usize, u32, u32)> {
    let idx: Vec<Option<u32>> = terms
        .iter()
        .map(|v| match v {
            Val::Tape(i) => Some(*i),
            _ => None,
        })
        .collect();
    let n = idx.len();
    let (Some(a), Some(b)) = (*idx.get(n.checked_sub(2)?)?, idx[n - 1]) else {
        return None;
    };
    let stride = b.checked_sub(a).filter(|s| *s > 0)?;
    let mut len = 2;
    while len < n {
        match (idx[n - 1 - len], idx[n - len]) {
            (Some(u), Some(w)) if u + stride == w => len += 1,
            _ => break,
        }
    }
    Some((len, idx[n - len].expect("in the run"), stride))
}

/// Sum a vectorised statement's per-element terms.
///
/// An evenly spaced run of tape values becomes one reduction node. The chain of
/// adds it replaces carries a value from each element to the next, which is the
/// one shape a re-rolled loop cannot run two repeats of at a time. Common
/// subexpressions leave the first element's nodes unlike the rest, so whatever
/// prefix falls outside the run keeps the chain and the run is added onto it —
/// the order the chain itself had.
pub fn v_sum(t: &mut Tape, terms: &[Val]) -> Val {
    let head = match run_suffix(terms) {
        // A run needs something ahead of it to be added to, and below a few
        // elements the chain is no worse.
        Some((len, _, _)) if len >= 4 => terms.len() - len.min(terms.len() - 1),
        _ => terms.len(),
    };
    let mut acc = Val::Num(0.0);
    for x in &terms[..head] {
        acc = v_add(t, &acc, x);
    }
    if let (Val::Tape(seed), Some((_, base, stride))) = (&acc, run_suffix(&terms[head..])) {
        return Val::Tape(t.sum_run(*seed, base, stride, (terms.len() - head) as u32));
    }
    for x in &terms[head..] {
        acc = v_add(t, &acc, x);
    }
    acc
}

pub fn v_neg(t: &mut Tape, a: &Val) -> Val {
    match a {
        Val::Num(x) => Val::Num(-x),
        Val::Tape(i) => Val::Tape(t.neg(*i)),
        Val::Vec(xs) | Val::Row(xs) => {
            let mut out = Vec::with_capacity(xs.len());
            for x in xs {
                out.push(v_neg(t, x));
            }
            a.like(out)
        }
    }
}

fn map_unary(t: &mut Tape, xs: &[Val], op: fn(&mut Tape, &Val) -> Val) -> Vec<Val> {
    xs.iter().map(|x| op(t, x)).collect()
}

pub fn v_abs(t: &mut Tape, a: &Val) -> Val {
    match a {
        Val::Num(x) => Val::Num(x.abs()),
        Val::Tape(i) => Val::Tape(t.abs(*i)),
        Val::Vec(xs) | Val::Row(xs) => a.like(map_unary(t, xs, v_abs)),
    }
}

pub fn v_exp(t: &mut Tape, a: &Val) -> Val {
    match a {
        Val::Num(x) => Val::Num(x.exp()),
        Val::Tape(i) => Val::Tape(t.exp(*i)),
        Val::Vec(xs) | Val::Row(xs) => a.like(map_unary(t, xs, v_exp)),
    }
}

pub fn v_log(t: &mut Tape, a: &Val) -> Val {
    match a {
        Val::Num(x) => Val::Num(x.ln()),
        Val::Tape(i) => Val::Tape(t.log(*i)),
        Val::Vec(xs) | Val::Row(xs) => a.like(map_unary(t, xs, v_log)),
    }
}

macro_rules! v_unary {
    ($name:ident, $prim:ident, $tape:ident) => {
        pub fn $name(t: &mut Tape, a: &Val) -> Val {
            match a {
                Val::Num(x) => Val::Num(x.$prim()),
                Val::Tape(i) => Val::Tape(t.$tape(*i)),
                Val::Vec(xs) | Val::Row(xs) => a.like(map_unary(t, xs, $name)),
            }
        }
    };
}

v_unary!(v_sin, sin, sin);
v_unary!(v_cos, cos, cos);
v_unary!(v_tan, tan, tan);
v_unary!(v_asin, asin, asin);
v_unary!(v_acos, acos, acos);
v_unary!(v_atan, atan, atan);

pub fn v_sqrt(t: &mut Tape, a: &Val) -> Val {
    v_pow(t, a, &Val::Num(0.5))
}

pub fn v_pow(t: &mut Tape, base: &Val, exp: &Val) -> Val {
    match (base, exp) {
        (Val::Num(x), Val::Num(n)) => Val::Num(x.powf(*n)),
        (Val::Tape(i), Val::Num(n)) => Val::Tape(t.pow(*i, *n)),
        _ => {
            let lb = v_log(t, base);
            let m = v_mul(t, exp, &lb);
            v_exp(t, &m)
        }
    }
}

pub fn v_lgamma(t: &mut Tape, a: &Val) -> Val {
    match a {
        Val::Num(x) => Val::Num(lgamma_double(*x)),
        Val::Tape(i) => Val::Tape(t.lgamma(*i)),
        Val::Vec(xs) | Val::Row(xs) => a.like(map_unary(t, xs, v_lgamma)),
    }
}

pub fn v_phi(t: &mut Tape, a: &Val) -> Val {
    match a {
        Val::Num(x) => Val::Num(phi_cdf_double(*x)),
        Val::Tape(i) => Val::Tape(t.phi(*i)),
        Val::Vec(xs) | Val::Row(xs) => a.like(map_unary(t, xs, v_phi)),
    }
}

/// `log P(T > t)` at `nu` degrees of freedom. `nu` is a plain number: the
/// value needs an incomplete beta, whose derivative in its own parameter has
/// no closed form here.
pub fn v_student_t_lccdf(t: &mut Tape, a: &Val, nu: f64) -> Val {
    match a {
        Val::Num(x) => Val::Num(student_t_lccdf_double(*x, nu)),
        Val::Tape(i) => Val::Tape(t.student_t_lccdf(*i, nu)),
        Val::Vec(xs) | Val::Row(xs) => {
            let mut out = Vec::with_capacity(xs.len());
            for x in xs {
                out.push(v_student_t_lccdf(t, x, nu));
            }
            a.like(out)
        }
    }
}

pub fn v_inv_logit(t: &mut Tape, a: &Val) -> Val {
    // 1 / (1 + exp(-a))
    let na = v_neg(t, a);
    let e = v_exp(t, &na);
    let s = v_add(t, &Val::Num(1.0), &e);
    v_div(t, &Val::Num(1.0), &s)
}

pub fn v_logit(t: &mut Tape, a: &Val) -> Val {
    // log(a / (1 - a))
    let one_minus = v_sub(t, &Val::Num(1.0), a);
    let r = v_div(t, a, &one_minus);
    v_log(t, &r)
}

pub fn v_tanh(t: &mut Tape, a: &Val) -> Val {
    // 2 * inv_logit(2*a) - 1
    let two_a = v_mul(t, &Val::Num(2.0), a);
    let il = v_inv_logit(t, &two_a);
    let two_il = v_mul(t, &Val::Num(2.0), &il);
    v_sub(t, &two_il, &Val::Num(1.0))
}
