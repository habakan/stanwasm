# stanwasm

[![crates.io](https://img.shields.io/crates/v/stanwasm?logo=rust&color=e43717)](https://crates.io/crates/stanwasm)
[![npm](https://img.shields.io/npm/v/stanwasm?logo=npm&color=cb3837)](https://www.npmjs.com/package/stanwasm)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/habakan/stanwasm/blob/main/LICENSE)

The wasm-bindgen API behind stanwasm: parse, compile and sample a Stan model
entirely inside the browser.

Part of [stanwasm](https://github.com/habakan/stanwasm), which compiles and samples Stan models
entirely in the browser — no server, no cmdstan.

This is the crate to look at first. It embeds
[nuts-rs](https://github.com/pymc-devs/nuts-rs) as the sampler and exposes
model compilation, `sample`, step-by-step sampling for live visualisation,
constrained-draw conversion, and `generated quantities`.

Built for `wasm32-unknown-unknown` and consumed from JavaScript as the
[`stanwasm`](https://www.npmjs.com/package/stanwasm) npm package. Most users
want that package rather than this crate — building it from source needs
`wasm-pack`, which the workspace `Makefile` wraps as `make wasm`.

Stan language coverage is a documented subset — see the workspace
[README](https://github.com/habakan/stanwasm#stan-language-coverage).

Licensed under Apache-2.0.
