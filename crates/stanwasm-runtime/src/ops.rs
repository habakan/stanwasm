//! Arithmetic and transcendental operations on `Val`.
//!
//! Each function takes a `&mut Tape` to push autodiff nodes when needed.
//! Plain-number paths short-circuit without touching the tape. Unary math
//! functions (`exp`, `log`, `abs`, `lgamma`, `Phi`) broadcast element-wise
//! over `Val::Vec` — Stan's built-in math functions are vectorized over
//! containers (e.g. `vector[N] y = exp(x);` is standard Stan).

use crate::value::Val;
use stanwasm_autodiff::{lgamma as lgamma_double, phi_cdf as phi_cdf_double, Tape};

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

pub fn v_add(t: &mut Tape, a: &Val, b: &Val) -> Val {
    match (a, b) {
        (Val::Num(x), Val::Num(y)) => Val::Num(x + y),
        (Val::Tape(i), Val::Num(y)) => Val::Tape(t.add_c(*i, *y)),
        (Val::Num(x), Val::Tape(j)) => Val::Tape(t.add_c(*j, *x)),
        (Val::Tape(i), Val::Tape(j)) => Val::Tape(t.add(*i, *j)),
        (Val::Vec(xs), Val::Vec(ys)) => Val::Vec(map_pair(t, xs, ys, v_add)),
        (Val::Vec(xs), other) => Val::Vec(map_left(t, xs, other, v_add)),
        (other, Val::Vec(ys)) => Val::Vec(map_right(t, other, ys, v_add)),
    }
}

pub fn v_sub(t: &mut Tape, a: &Val, b: &Val) -> Val {
    match (a, b) {
        (Val::Num(x), Val::Num(y)) => Val::Num(x - y),
        (Val::Tape(i), Val::Num(y)) => Val::Tape(t.sub_c(*i, *y)),
        (Val::Num(x), Val::Tape(j)) => Val::Tape(t.rsub_c(*j, *x)),
        (Val::Tape(i), Val::Tape(j)) => Val::Tape(t.sub(*i, *j)),
        (Val::Vec(xs), Val::Vec(ys)) => Val::Vec(map_pair(t, xs, ys, v_sub)),
        (Val::Vec(xs), other) => Val::Vec(map_left(t, xs, other, v_sub)),
        (other, Val::Vec(ys)) => Val::Vec(map_right(t, other, ys, v_sub)),
    }
}

pub fn v_mul(t: &mut Tape, a: &Val, b: &Val) -> Val {
    match (a, b) {
        (Val::Num(x), Val::Num(y)) => Val::Num(x * y),
        (Val::Tape(i), Val::Num(y)) => Val::Tape(t.mul_c(*i, *y)),
        (Val::Num(x), Val::Tape(j)) => Val::Tape(t.mul_c(*j, *x)),
        (Val::Tape(i), Val::Tape(j)) => Val::Tape(t.mul(*i, *j)),
        (Val::Vec(xs), Val::Vec(ys)) => Val::Vec(map_pair(t, xs, ys, v_mul)),
        (Val::Vec(xs), other) => Val::Vec(map_left(t, xs, other, v_mul)),
        (other, Val::Vec(ys)) => Val::Vec(map_right(t, other, ys, v_mul)),
    }
}

pub fn v_div(t: &mut Tape, a: &Val, b: &Val) -> Val {
    match (a, b) {
        (Val::Num(x), Val::Num(y)) => Val::Num(x / y),
        (Val::Tape(i), Val::Num(y)) => Val::Tape(t.div_c(*i, *y)),
        (Val::Num(x), Val::Tape(j)) => Val::Tape(t.rdiv_c(*j, *x)),
        (Val::Tape(i), Val::Tape(j)) => Val::Tape(t.div(*i, *j)),
        (Val::Vec(xs), Val::Vec(ys)) => Val::Vec(map_pair(t, xs, ys, v_div)),
        (Val::Vec(xs), other) => Val::Vec(map_left(t, xs, other, v_div)),
        (other, Val::Vec(ys)) => Val::Vec(map_right(t, other, ys, v_div)),
    }
}

pub fn v_neg(t: &mut Tape, a: &Val) -> Val {
    match a {
        Val::Num(x) => Val::Num(-x),
        Val::Tape(i) => Val::Tape(t.neg(*i)),
        Val::Vec(xs) => {
            let mut out = Vec::with_capacity(xs.len());
            for x in xs {
                out.push(v_neg(t, x));
            }
            Val::Vec(out)
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
        Val::Vec(xs) => Val::Vec(map_unary(t, xs, v_abs)),
    }
}

pub fn v_exp(t: &mut Tape, a: &Val) -> Val {
    match a {
        Val::Num(x) => Val::Num(x.exp()),
        Val::Tape(i) => Val::Tape(t.exp(*i)),
        Val::Vec(xs) => Val::Vec(map_unary(t, xs, v_exp)),
    }
}

pub fn v_log(t: &mut Tape, a: &Val) -> Val {
    match a {
        Val::Num(x) => Val::Num(x.ln()),
        Val::Tape(i) => Val::Tape(t.log(*i)),
        Val::Vec(xs) => Val::Vec(map_unary(t, xs, v_log)),
    }
}

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
        Val::Vec(xs) => Val::Vec(map_unary(t, xs, v_lgamma)),
    }
}

pub fn v_phi(t: &mut Tape, a: &Val) -> Val {
    match a {
        Val::Num(x) => Val::Num(phi_cdf_double(*x)),
        Val::Tape(i) => Val::Tape(t.phi(*i)),
        Val::Vec(xs) => Val::Vec(map_unary(t, xs, v_phi)),
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
