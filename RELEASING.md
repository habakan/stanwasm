# Releasing

Maintainer notes. `.github/workflows/release.yml` runs on the tag: it verifies
the tagged tree, creates the GitHub release, and publishes the npm package.
crates.io is still published by hand from a local checkout. No CI job holds a
registry credential either way — the npm publish authenticates with the run's
own OIDC token.

**A published version is permanent.** crates.io can yank and npm can
deprecate, but neither frees the version number or removes the code. Every
check below exists because something here is not reversible.

Every published name is prefixed `stanwasm`, matching the npm package. A bare
`stan-parser` or `stan-runtime` on a flat registry namespace reads as a crate
belonging to Stan itself, and crates.io never frees a name once taken.

## 0. One-time: trust this workflow on npm

Only needed once per package, and again if `release.yml` is ever renamed.

On npmjs.com, under the `stanwasm` package's **Settings → Trusted publisher**,
add a GitHub Actions publisher:

| Field | Value |
|---|---|
| Organization or user | `habakan` |
| Repository | `stanwasm` |
| Workflow filename | `release.yml` |
| Environment | leave blank |

Leave any existing automation token in place until the first tag publishes
cleanly, then delete it. A stored token that nothing uses is the credential most
likely to leak, and trusted publishing exists to not have one.

Trusted publishing is also what produces the provenance attestation. A tarball
published from a laptop cannot carry one, which is why `SECURITY.md` used to say
the published bytes could not be traced back to a commit.

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
make check TESTFLAGS=--release   # fmt + clippy + the release test suite
make smoke                       # builds ts/pkg/, then exercises it in Node
make package                     # what both registries would actually receive
```

`make package` runs `cargo package --workspace`, not a per-crate loop. That
matters before the first release: `cargo package -p stanwasm-parser` on its
own resolves `stanwasm-ast` from the crates.io index and fails with `no
matching package named stanwasm-ast found` until 0.1.0 is really published
there, while the workspace form resolves siblings locally and can check all
six manifests today.

It then asserts two things that are invisible until someone installs the
result. That the Apache-2.0 licence text is inside every artifact — `cargo
package` and `npm pack` each collect only files under their own directory, so
the repo-root `LICENSE` reaches no tarball on its own. And that `pkg/` carries
the wasm: `wasm-pack` writes its own `.gitignore` (containing `*`) into
`ts/pkg/`, and npm honors a nested `.gitignore` when no `.npmignore` sits
beside it — which once published a package whose `pkg/` was empty, no wasm and
no glue JS, despite `package.json`'s `files` saying to include it. The `wasm`
target deletes that file after every build; this check is what catches it
coming back.

## 3. Tag, and let CI check the tree

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

**Pushing the tag spends the npm version.** `guard` and `verify` run first and
a failure in either stops `publish-npm` before it uploads, so a rejected tag can
still be deleted with `git push --delete origin vX.Y.Z`. But once both pass, npm
has the version and no one can take it back. The last fully reversible moment is
the local `git tag`, before the push.

`guard` compares the tag against every file in step 1. `verify` re-runs the
full gate against the exact tagged commit — `test.yml` only covers pushes to
`main` and pull requests, so a tag on a rebased or never-PR'd commit is
otherwise unverified. `github-release` then creates the GitHub release from the
CHANGELOG section, and `publish-npm` uploads the tarball.

Check that `release.yml` is actually enabled before tagging
(`gh workflow list --all`). A disabled workflow does not fail on its trigger —
the tag lands and nothing runs at all.

## 4. Publish to crates.io

Strictly in this order. Each manifest resolves the ones before it from the
registry rather than from its path, so a crate cannot go up before its
dependencies:

```bash
cargo publish -p stanwasm-ast
cargo publish -p stanwasm-autodiff
cargo publish -p stanwasm-parser
cargo publish -p stanwasm-runtime
cargo publish -p stanwasm-codegen
cargo publish -p stanwasm
```

`cargo publish` blocks until the index carries what it just uploaded, so the
next command in the list can see it. If one fails partway through, the ones
already up stay up — fix forward with a patch version rather than trying to
re-publish the same number.

`stanwasm-cli` is `publish = false`; it is a local development binary. The
five crates before `stanwasm` are published only because cargo requires a
dependency to be on the registry before its dependent can be — their
descriptions say so, and they carry no API stability guarantee.

## 5. Confirm the npm publish

Step 3 already did it. Check that it carried provenance:

```bash
npm view stanwasm@X.Y.Z dist.attestations
```

Nothing there means the publish fell back to a token, or the trusted publisher
in step 0 does not match this workflow. The version is still published and
usable; fix the trust before the next release rather than re-publishing.

If `publish-npm` failed outright, read the log before reaching for a local
`npm publish` — a tarball published from a laptop carries no provenance, and
`npm view stanwasm` will show the gap for that version forever.

## 6. After

- Open `CHANGELOG.md` and start a fresh `## [Unreleased]` section.
- Check the GitHub release rendered the notes you expected.
- `npm view stanwasm` and `cargo search stanwasm` to confirm what landed.
