//! Reading and writing one element of a container must not cost the whole
//! container. Both used to: indexing evaluated its base, and evaluating a
//! variable copies its binding, so a loop over a matrix's elements was
//! quadratic. Four posteriordb models did not finish tracing in two minutes.
//!
//! The bounds below are wall clock, so they are set two orders of magnitude
//! above what the linear version takes — enough to catch the quadratic return
//! on any machine, not to measure anything.

use std::time::Instant;

use stanwasm_runtime::{Env, Model, Val};

const N: usize = 4000;

fn timed(src: &str) -> f64 {
    let mut d = Env::new();
    d.set_scalar("N", N as f64);
    d.set(
        "M",
        Val::Vec(
            (0..N)
                .map(|i| Val::Vec((0..8).map(|j| Val::Num((i % 7 + j) as f64)).collect()))
                .collect(),
        ),
    );
    let model = Model::parse_and_load(src, d).unwrap();
    let started = Instant::now();
    let (lp, _) = model.log_prob_grad(&[1.0]).unwrap();
    assert!(lp.is_finite(), "{lp}");
    started.elapsed().as_secs_f64()
}

#[test]
fn reading_an_element_does_not_cost_the_container() {
    let secs = timed(
        "data { int<lower=0> N; matrix[N, 8] M; }
         parameters { real a; }
         model { for (i in 1:N) { for (j in 1:8) { target += a * M[i, j]; } } }",
    );
    assert!(secs < 5.0, "{N} rows read in {secs:.1}s");
}

#[test]
fn writing_an_element_does_not_rebuild_the_container() {
    let secs = timed(
        "data { int<lower=0> N; matrix[N, 8] M; }
         parameters { real a; }
         model {
           vector[N] acc = rep_vector(0, N);
           for (i in 1:N) { acc[i] = a * M[i, 1]; }
           target += sum(acc);
         }",
    );
    assert!(secs < 5.0, "{N} rows written in {secs:.1}s");
}
