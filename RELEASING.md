# Releasing

Maintainer notes. Publishing is manual and local — no CI job holds a registry
credential. `.github/workflows/release.yml` runs on the tag and verifies; it
uploads nothing.

**A published version is permanent.** crates.io can yank and npm can
deprecate, but neither frees the version number or removes the code. Every
check below exists because something here is not reversible.

## 1. Set the version

It appears in four places, and nothing keeps them in sync automatically:

| File | Field |
|---|---|
| `Cargo.toml` | `workspace.package.version` |
| `Cargo.toml` | `version = "…"` on all six internal deps under `[workspace.dependencies]` |
| `ts/package.json` | `version` |
| `CITATION.cff` | `version`, `date-released` |

Cargo does not accept `version.workspace` inside `[workspace.dependencies]`,
which is why the six requirements are written out by hand.

Then move `CHANGELOG.md`'s `[Unreleased]` section under a `## [X.Y.Z] — DATE`
heading, and add the link definition at the bottom of the file. The date has
to be the day you actually tag: the GitHub release body is extracted from this
section by heading match, and the `guard` job fails the tag if no section
matches.

## 2. Check locally

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release

./scripts/build-wasm.sh
cd ts && node --experimental-strip-types tests/smoke.ts && cd ..

# Manifests are publishable (a path dependency with no version requirement
# fails here, not halfway through step 4).
for c in stan-ast stan-autodiff stan-parser stan-runtime stan-codegen stan-wasm-api; do
  cargo package --no-verify -p "$c"
done

# The npm tarball actually contains the wasm — see step 5.
cd ts && npm pack --dry-run && cd ..
```

Until the first crates.io publish, `cargo package` on everything except
`stan-ast` and `stan-autodiff` reports `no matching package named stan-ast
found`. That is expected: the packaged manifest resolves siblings from the
registry, and they are not there yet. Manifest errors look different — they
name the manifest and the field.

## 3. Tag, and let CI check the tree

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

Tag **before** publishing, not after. This is the last reversible moment: if
`guard` or `verify` fails, `git push --delete origin vX.Y.Z` and fix it. Once
step 4 runs, the version is spent.

`guard` compares the tag against every file in step 1. `verify` re-runs the
full gate against the exact tagged commit — `test.yml` only covers pushes to
`main` and pull requests, so a tag on a rebased or never-PR'd commit is
otherwise unverified. `github-release` then creates the GitHub release from
the CHANGELOG section.

## 4. Publish to crates.io

Strictly in this order. Each manifest resolves the ones before it from the
registry rather than from its path, so a crate cannot go up before its
dependencies:

```bash
cargo publish -p stan-ast
cargo publish -p stan-autodiff
cargo publish -p stan-parser
cargo publish -p stan-runtime
cargo publish -p stan-codegen
cargo publish -p stan-wasm-api
```

`cargo publish` blocks until the index carries what it just uploaded, so the
next command in the list can see it. If one fails partway through, the ones
already up stay up — fix forward with a patch version rather than trying to
re-publish the same number.

`stan-cli` is `publish = false`; it is a local development binary.

## 5. Publish to npm

```bash
./scripts/build-wasm.sh   # required — the wasm is not committed
cd ts
npm pack --dry-run        # confirm pkg/*.wasm is in the file list
npm publish --access public
```

The `npm pack --dry-run` line is not ceremony. `wasm-pack` writes its own
`.gitignore` (containing `*`) into `ts/pkg/`, and npm honors a nested
`.gitignore` when no `.npmignore` sits beside it — which once published a
package whose `pkg/` was empty, no wasm and no glue JS, despite
`package.json`'s `files` saying to include it. `scripts/build-wasm.sh` deletes
that file after every build; this check is what catches it coming back.

## 6. After

- Open `CHANGELOG.md` and start a fresh `## [Unreleased]` section.
- Check the GitHub release rendered the notes you expected.
- `npm view stanwasm` and `cargo search stan-wasm-api` to confirm what landed.
