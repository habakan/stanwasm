# stanwasm-codegen

Traces a Stan model onto an autodiff tape and hands the tape to the emitter.

Part of [stanwasm](https://github.com/habakan/stanwasm), which compiles and samples Stan models
entirely in the browser — no server, no cmdstan.

The emitter is [`tapewasm-codegen`](https://crates.io/crates/tapewasm-codegen),
which reads a tape and nothing else. What is here is the half that knows what a
model is: running the evaluator forward to record one, and reporting an
evaluation failure apart from a failure to emit. What the emitter exports is
re-exported unchanged, so a caller needs one dependency rather than two.

Emitted modules are plain wasm32 — linear memory and a manual heap, no wasm-gc.
A test in CI validates the output with `wasmparser` and the GC feature
explicitly disabled, so the target stays every browser rather than the newest
one.

You almost certainly want [`stanwasm`](https://crates.io/crates/stanwasm), or the `stanwasm` npm
package, rather than this crate directly. It is published because they depend
on it.

Stan language coverage is a documented subset — see the workspace
[README](https://github.com/habakan/stanwasm#stan-language-coverage).

Licensed under Apache-2.0.
