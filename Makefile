# Task runner for stanwasm.
#
# These targets are thin wrappers over cargo / wasm-pack / node. The point is
# one discoverable place for the commands that CONTRIBUTING.md, RELEASING.md
# and CI would otherwise each spell out in prose, so that the thing you run
# locally is literally the thing CI runs.
#
# Written for GNU Make 3.81 — the version macOS still ships — so no
# `.SHELLFLAGS`, and no other 3.82+ feature.

# `help` first, so a bare `make` lists targets rather than building anything.
.PHONY: help
help:
	@grep -hE '^[a-z][a-z-]*:.*##' $(MAKEFILE_LIST) \
	  | sed 's/:[^#]*##/|/' \
	  | awk -F'|' '{ printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2 }'

ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

# `node --experimental-strip-types` rather than a TypeScript build step: the
# tests are .ts, and this is what CI uses. Override for an older Node.
NODE ?= node --experimental-strip-types

# Passed through to `cargo test` — `make test TESTFLAGS=--release` is what
# RELEASING.md wants, without a second near-duplicate target.
TESTFLAGS ?=

.PHONY: fmt
fmt: ## Format all Rust code
	cargo fmt --all

.PHONY: fmt-check
fmt-check: ## Fail if anything is unformatted (what CI runs)
	cargo fmt --all -- --check

.PHONY: clippy
clippy: ## Lint the workspace with warnings denied
	cargo clippy --workspace --all-targets -- -D warnings

.PHONY: test
test: ## Native test suite, all crates
	cargo test --workspace $(TESTFLAGS)

.PHONY: check
check: fmt-check clippy test ## fmt-check + clippy + test

# The wasm bundle is a real file, not a phony target, so `make smoke` after an
# unrelated edit does not pay for a wasm-pack rebuild. Cargo does the
# fine-grained dependency tracking; this only has to answer "did any Rust
# source or manifest change since the bundle was written".
WASM_OUT := ts/pkg/stanwasm_bg.wasm
WASM_SRC := $(shell find crates -name '*.rs') Cargo.toml Cargo.lock

.PHONY: wasm
wasm: $(WASM_OUT) ## Build the wasm bundle into ts/pkg/

$(WASM_OUT): $(WASM_SRC)
	@command -v wasm-pack >/dev/null 2>&1 \
	  || { echo "error: wasm-pack not found. Install with: cargo install wasm-pack" >&2; exit 1; }
# wasm-pack writes into the out-dir without clearing it, so anything it no
# longer emits just stays. Renaming the crate once left the previous bundle's
# `stan_wasm_api*` files sitting beside the new ones, and `"files": ["pkg/"]`
# would have shipped both — 477 KB of dead wasm in the published package.
# Everything here is generated; `clean` removes the same directory.
	rm -rf $(ROOT)/ts/pkg
	wasm-pack build crates/stanwasm --target web --out-dir $(ROOT)/ts/pkg --release
# wasm-pack drops its own `.gitignore` (just `*`) into ts/pkg/. That's
# redundant here — the repo root .gitignore already excludes /ts/pkg — and
# actively harmful: npm's ignore-file resolution honors a nested .gitignore
# with no matching .npmignore, so `npm publish`/`npm pack` from ts/ would
# silently ship an empty pkg/ (no wasm, no glue JS) despite package.json's
# `"files": ["pkg/"]` saying to include it. Remove it so the package we would
# actually publish is the one we tested.
	@rm -f $(ROOT)/ts/pkg/.gitignore
	@ls -la $(ROOT)/ts/pkg/

.PHONY: smoke
smoke: wasm ## Node smoke test against the built bundle
	cd ts && $(NODE) tests/smoke.ts

.PHONY: bench
bench: wasm ## Node benchmark (replay vs AOT)
	cd ts && $(NODE) tests/bench.ts

.PHONY: bench-native
bench-native: ## Native Rust benchmark, no wasm involved
	cargo run --release -p stanwasm-cli -- bench all

# Data size for the vectorised models; the small ones ignore it.
N ?= 5000
BENCH_DIR := $(ROOT)/target/bench

.PHONY: bench-gradients
bench-gradients: wasm ## Per-gradient cost of both paths, over the model set
	cd ts && $(NODE) tests/bench_gradients.ts $(N) --emit $(BENCH_DIR)

# Needs a built CmdStan. Compiles each model there on first run, which takes a
# few seconds each and is then cached in target/bench.
CMDSTAN ?= ~/cmdstan

# posteriordb is stan-dev's collection of real posterior inference problems:
#   git clone --depth 1 https://github.com/stan-dev/posteriordb
PDB ?= ../posteriordb

.PHONY: posteriordb
posteriordb: wasm ## How much of posteriordb loads, and what stops the rest
	cd ts && $(NODE) tests/posteriordb_sweep.ts $(PDB)

.PHONY: compare-cmdstan
compare-cmdstan: bench-gradients ## Check log density and gradients against CmdStan, and time both
	cd ts && CMDSTAN=$(CMDSTAN) $(NODE) tests/compare_cmdstan.ts $(BENCH_DIR)

.PHONY: gallery
gallery: wasm ## Vite dev server for examples/gallery
	cd examples/gallery && npm install && npm run dev

# `npm ci` rather than `npm install`: this is the build GitHub Pages ships, so
# it resolves from the lockfile and nothing else.
.PHONY: gallery-build
gallery-build: wasm ## Production build of the gallery (what GitHub Pages ships)
	cd examples/gallery && npm ci && npm run build

# `--workspace` rather than a per-crate loop: `cargo package -p stanwasm-parser`
# on its own resolves `stanwasm-ast` from the crates.io index and fails until
# 0.1.0 is actually published there, whereas the workspace form resolves
# siblings locally. So this is runnable before the first release, which is
# exactly when a path dependency missing its `version` needs to surface — not
# halfway through publishing, when the crates already up cannot be taken back.
.PHONY: package
package: wasm ## Dry-run packaging every crate + the npm tarball
	cargo package --workspace --no-verify
# Apache-2.0 requires the licence text to travel with the artifact, and both
# `cargo package` and `npm pack` only collect files inside their own directory
# — the LICENSE at the repo root reaches no tarball on its own. The npm
# tarball has a second invisible failure: `wasm-pack` writes its own
# `.gitignore` (containing `*`) into `ts/pkg/`, which npm honours when no
# `.npmignore` sits beside it, and that once published a package carrying no
# wasm at all. A published version cannot be taken back, so both are asserted
# rather than assumed.
	@list=$$(mktemp); \
	for f in target/package/*.crate; do \
	  tar tzf "$$f" > "$$list"; \
	  grep -q '/LICENSE$$' "$$list" \
	    || { echo "error: $$f ships no LICENSE" >&2; rm -f "$$list"; exit 1; }; \
	done; \
	rm -f "$$list"
	@echo "LICENSE present in every .crate"
# The check is a script rather than a `node -e` here: make 4.3 and make 3.81
# disagree about a backslash-continued line inside a recipe, and the older one
# is what macOS ships, so the inline form passed locally and could never run
# on CI.
	@pack=$$(mktemp); \
	(cd ts && npm pack --dry-run --json) > "$$pack" \
	  && node ts/tests/check_pack.mjs "$$pack"; \
	status=$$?; rm -f "$$pack"; exit $$status

.PHONY: clean
clean: ## Remove cargo target/ and the generated ts/pkg/
	cargo clean
	rm -rf $(ROOT)/ts/pkg
