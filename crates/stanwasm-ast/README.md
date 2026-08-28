# stanwasm-ast

Shared AST types for the Stan probabilistic programming language.

Part of [stanwasm](https://github.com/habakan/stanwasm), which compiles and samples Stan models
entirely in the browser — no server, no cmdstan.

Data definitions only: blocks, declarations, expressions, distribution calls.
No parsing, no evaluation, no code generation. [`stanwasm-parser`](https://crates.io/crates/stanwasm-parser) builds these,
[`stanwasm-runtime`](https://crates.io/crates/stanwasm-runtime) evaluates them, and [`stanwasm-codegen`](https://crates.io/crates/stanwasm-codegen) compiles them —
keeping the definitions in a leaf crate is what lets those three agree on a
representation without depending on one another.

You almost certainly want [`stanwasm`](https://crates.io/crates/stanwasm), or the `stanwasm` npm
package, rather than this crate directly. It is published because they depend
on it.

Stan language coverage is a documented subset — see the workspace
[README](https://github.com/habakan/stanwasm#stan-language-coverage).

Licensed under Apache-2.0.
