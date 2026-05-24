<!-- Thanks for the PR. Please fill in the sections below. -->

## What this changes

<!-- one or two sentences -->

## Why

<!-- linked issue, motivation, or use case -->

## Scope check

- [ ] In scope per `CONTRIBUTING.md` (Stan subset / nuts-rs / browser-only)
- [ ] If it's a Stan feature addition, documented in `README.md` under "Stan language coverage"
- [ ] Does not break the no-wasm-gc invariant (`cargo test -p stan-codegen --test no_wasm_gc`)

## Tests added

- [ ] Per-distribution finite-difference gradient check (if adding a distribution)
- [ ] AOT-vs-oracle agreement test (if changing codegen or constraints)
- [ ] Native + wasm32 builds both green

## CHANGELOG

- [ ] Entry added under `[Unreleased]` in `CHANGELOG.md`

## Notes for reviewer

<!-- anything worth flagging -->
