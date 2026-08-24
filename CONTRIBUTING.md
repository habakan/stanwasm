# Contributing to stanwasm

Thanks for your interest. stanwasm is currently in **alpha**, maintained by one person on evenings and weekends. Contributions are welcome but please read this document first — it will save you and me time.

## Scope

stanwasm is **not** an attempt to reimplement the full Stan language. It is a deliberately scoped subset of Stan that runs end-to-end inside the browser, with an embedded NUTS sampler from [nuts-rs](https://github.com/pymc-devs/nuts-rs). The full Stan compiler / cmdstan / PyStan / RStan toolchain remains the canonical implementation; this project is complementary, not a replacement.

**In scope:**
- Stan grammar / parser improvements
- Additional distributions and constraint transforms (see `ROADMAP.md` for what is intentionally deferred)
- Performance work (AOT codegen, replay path, smaller wasm)
- Better diagnostics / error messages
- Browser and Node.js examples
- Documentation

**Out of scope (for now):**
- User-defined functions block (low priority)
- ADVI / fixed_param / pathfinder samplers (NUTS only by design)
- Anything requiring a backend server (the whole point is browser completion)
- Forking nuts-rs (we use it as-is via Cargo)

If you are not sure whether something is in scope, open a discussion or draft issue before writing code.

## Relationship to the Stan ecosystem

- **Stan official org**: [stan-dev](https://github.com/stan-dev) — canonical Stan, cmdstan, stanc3, Stan math, etc.
- **Stan Playground (Flatiron Institute)**: [flatironinstitute/stan-playground](https://github.com/flatironinstitute/stan-playground) — full-Stan, server-compile, browser-sample. The official browser playground.
- **stanwasm**: subset of Stan, browser-only, embedded sampler. **Complementary** to Stan Playground.

If you have a Stan problem that needs full language support, you almost certainly want Stan Playground or cmdstan, not this project.

## Development setup

```bash
git clone https://github.com/habakan/stanwasm
cd stanwasm

# Native build + tests (all crates)
cargo build --workspace --release
cargo test --workspace

# wasm32 build for the browser API
./scripts/build-wasm.sh  # requires wasm-pack
```

Required tools:
- Rust 1.88+ (the MSRV declared in the workspace `Cargo.toml`; `rust-toolchain.toml` pins the stable channel for development)
- `wasm-pack` (`cargo install wasm-pack`) for browser builds
- Node.js 22+ for Node integration tests (optional)

## Running tests

```bash
# Everything (native + integration)
cargo test --workspace

# Specific crate
cargo test -p stan-parser

# wasm-pack-built bundle in Node
./scripts/build-wasm.sh
cd ts && node --experimental-strip-types tests/smoke.ts
```

## Pull request guidelines

1. **One concern per PR.** Distribution additions, codegen optimizations, and doc tweaks all separate.
2. **Test coverage.** New distributions need at least:
   - one finite-difference gradient check in `crates/stan-runtime/tests/`
   - one oracle-vs-AOT comparison in `crates/stan-codegen/tests/`
3. **Document subset boundaries.** If a PR partially implements a Stan feature, the README should reflect what now works and what still doesn't.
4. **Don't break the no-wasm-gc invariant.** `crates/stan-codegen/tests/no_wasm_gc.rs` must keep passing — we ship plain wasm32 by design.
5. **CHANGELOG entry.** Add a bullet under `[Unreleased]` in `CHANGELOG.md`. Re-create that heading at the top of the file if the last release folded it into a version — see [`RELEASING.md`](RELEASING.md).
6. **Format and lint.** `cargo fmt --all` and `cargo clippy --all-targets`.

## Code style

- Rust 2021 edition, default `cargo fmt` style
- Prefer `forbid(unsafe_code)` per crate; only `stan-wasm-api` has FFI-flavored exceptions
- Comments explain *why*, not *what*
- No emoji in code or comments

## Issue reporting

When reporting a Stan model that fails to parse or evaluates incorrectly, please include:
- Minimal Stan code that reproduces the issue
- The data being fed (JSON inline is fine)
- Expected vs observed `log_prob_grad` output (use `Model::parse_and_load` and `Model::log_prob_grad`)
- Output of `cargo test --workspace` if relevant

## Maintainer responsiveness

This project is a side project. PR reviews and issue triage happen in batches, typically within 1-2 weeks. If something is urgent, say so in the issue and explain why.

For broader Stan ecosystem coordination, ping me on Stan Discourse or @habakan on GitHub.

## License

By contributing, you agree that your contributions will be licensed under the same Apache-2.0 license as the project.
