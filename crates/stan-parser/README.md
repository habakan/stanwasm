# stan-parser

Hand-written recursive-descent parser for a subset of the Stan language.

Part of [stanwasm](https://github.com/habakan/stanwasm), which compiles and samples Stan models
entirely in the browser — no server, no cmdstan.

Lexer, parser and error reporting, producing the types in [`stan-ast`](https://crates.io/crates/stan-ast).
Hand-written rather than generated, because the errors a modelling language
needs — pointing at the offending token, naming the block it appeared in — are
the part a grammar generator makes hardest.

The accepted subset is deliberate, not aspirational: constructs outside it are
rejected with a message saying so, never silently mis-parsed.

You almost certainly want [`stan-wasm-api`](https://crates.io/crates/stan-wasm-api), or the `stanwasm` npm
package, rather than this crate directly. It is published because they depend
on it.

Stan language coverage is a documented subset — see the workspace
[README](https://github.com/habakan/stanwasm#stan-language-coverage).

Licensed under Apache-2.0.
