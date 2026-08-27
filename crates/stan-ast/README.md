# stan-ast

Shared AST types for the Stan probabilistic programming language.

Part of [stanwasm](https://github.com/habakan/stanwasm), which compiles and samples Stan models
entirely in the browser — no server, no cmdstan.

Data definitions only: blocks, declarations, expressions, distribution calls.
No parsing, no evaluation, no code generation. [`stan-parser`](https://crates.io/crates/stan-parser) builds these,
[`stan-runtime`](https://crates.io/crates/stan-runtime) evaluates them, and [`stan-codegen`](https://crates.io/crates/stan-codegen) compiles them —
keeping the definitions in a leaf crate is what lets those three agree on a
representation without depending on one another.

You almost certainly want [`stan-wasm-api`](https://crates.io/crates/stan-wasm-api), or the `stanwasm` npm
package, rather than this crate directly. It is published because they depend
on it.

Stan language coverage is a documented subset — see the workspace
[README](https://github.com/habakan/stanwasm#stan-language-coverage).

Licensed under Apache-2.0.
