# stan-runtime

Stan distributions, constraint transforms, and an AST evaluator.

Part of [stanwasm](https://github.com/habakan/stanwasm), which compiles and samples Stan models
entirely in the browser — no server, no cmdstan.

The largest crate in the workspace, and the one correctness rests on: density
and gradient implementations, constrained-to-unconstrained transforms with
their log-Jacobians, and a direct AST interpreter.

The interpreter is deliberately the slow path. It is the golden oracle the
compiled paths are tested against — an AOT-compiled model that disagrees with
it to more than 1e-12 is a bug in the compiler, not a tolerance to widen.

You almost certainly want [`stan-wasm-api`](https://crates.io/crates/stan-wasm-api), or the `stanwasm` npm
package, rather than this crate directly. It is published because they depend
on it.

Stan language coverage is a documented subset — see the workspace
[README](https://github.com/habakan/stanwasm#stan-language-coverage).

Licensed under Apache-2.0.
