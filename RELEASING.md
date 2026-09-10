# Releasing

Maintainer notes. `.github/workflows/release.yml` runs on the tag: it verifies
the tagged tree and creates the GitHub release. Both registries are published
by hand from a local checkout, and no CI job holds a credential for either.

npm can be published from the workflow instead, with a provenance attestation,
by configuring a trusted publisher on npmjs.com — see **Publishing from CI**
at the end. It is deliberately not set up: publishing by hand is what has
always happened here, and a job that fails on every tag because the trust was
never configured is worse than no job.

**A published version is permanent.** crates.io can yank and npm can
deprecate, but neither frees the version number or removes the code. Every
check below exists because something here is not reversible.

Every published name is prefixed `stanwasm`, matching the npm package. A bare
`stan-parser` or `stan-runtime` on a flat registry namespace reads as a crate
belonging to Stan itself, and crates.io never frees a name once taken.

## 1. Set the version

It appears in four places, and nothing keeps them in sync automatically:

| File | Field |
|---|---|
| `Cargo.toml` | `workspace.package.version` |
| `Cargo.toml` | `version = "…"` on all six internal deps under `[workspace.dependencies]` |
| `ts/package.json` | `version` |
| `CITATION.cff` | `version`, `date-released` |

`LICENSE-APACHE` and `LICENSE-MIT` both have to sit beside every crate
manifest and in `ts/`; `make package-npm` fails if either is missing.

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
make package-npm                 # what npm would actually receive
```

`make package` is the pair — `package-crates` and `package-npm` — but the
crates half cannot run while the engine is a git dependency: `cargo package`
rewrites a git spec to the version beside it, and tapewasm is not on crates.io
yet, so it fails resolving rather than checking anything. It waits on the same
nuts-rs release step 5 does. `package-npm` is the half that runs, and it is the
half these releases actually ship.

The two halves assert what is invisible until someone installs the result. That
both licence texts are inside every artifact — the offer is either one, so
shipping half of it is not the offer, and `cargo
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

Nothing is published by this. `guard` and `verify` check the tagged tree and
`github-release` creates the release; a tag that fails either can be deleted
with `git push --delete origin vX.Y.Z`. The registries are steps 4 and 5, and
those are what cannot be taken back.

`guard` compares the tag against every file in step 1. `verify` re-runs the
full gate against the exact tagged commit — `test.yml` only covers pushes to
`main` and pull requests, so a tag on a rebased or never-PR'd commit is
otherwise unverified. `github-release` then creates the GitHub release from the
CHANGELOG section.

Check that `release.yml` is actually enabled before tagging
(`gh workflow list --all`). A disabled workflow does not fail on its trigger —
the tag lands and nothing runs at all.

## 4. Publish to npm

From a checkout of the tag, with nothing uncommitted — `make wasm` bakes the
working tree into the bundle, so a stray edit ships as the release:

```bash
git status --short          # must be clean
git rev-parse HEAD          # must be the tagged commit
make wasm
cd ts && npm publish --access public
```

No `--provenance`: a tarball published from a laptop cannot carry an
attestation, and passing the flag fails rather than being ignored.

**This spends the version.** npm can deprecate but never frees a number.
Confirm with `npm view stanwasm version`.

## 5. Publish to crates.io

**Blocked as of 0.5.0.** The workspace `[patch.crates-io]` takes nuts-rs from
their main branch for a wasm fix that no published version carries, and a patch
does not travel into a published crate — so a crate published today builds, for
anyone who depends on it, a module WebKit refuses. `cargo publish` succeeds at
this, and a version cannot be taken back. Wait for a nuts-rs release, drop the
patch, and let `cargo test -p stanwasm-codegen --test no_wasm_gc` confirm it.
Skip this step until then and mark the CHANGELOG heading "(npm only)" with the
reason.

`tapewasm-autodiff`, `tapewasm-codegen` and `tapewasm` come from their own
repository and have to be up first — the manifests here name them by git rev
with a version beside it, and that version is what `cargo publish` writes. So
this step also means replacing the rev with the version, which is a commit of
its own, not something to do mid-publish.

Then strictly in this order. Each manifest resolves the ones before it from the
registry rather than from its path, so a crate cannot go up before its
dependencies:

```bash
cargo publish -p stanwasm-ast
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

## 6. After

- Open `CHANGELOG.md` and start a fresh `## [Unreleased]` section.
- Check the GitHub release rendered the notes you expected.
- `npm view stanwasm` and `cargo search stanwasm` to confirm what landed.

## Publishing from CI, if that is ever wanted

npm can publish from `release.yml` with a provenance attestation naming the
commit and the run that built the tarball. It needs two things, and the second
is what makes it a decision rather than a switch.

A `publish-npm` job in `release.yml` with `permissions: id-token: write`, ending
in `npm publish --provenance --access public`. `git log -S publish-npm --
.github/workflows/release.yml` has one that worked as far as npm's door.

And a trusted publisher on npmjs.com, under the `stanwasm` package's
**Settings → Trusted publisher**:

| Field | Value |
|---|---|
| Organization or user | `habakan` |
| Repository | `stanwasm` |
| Workflow filename | `release.yml` |
| Environment | leave blank |

Without it npm rejects the OIDC token, and the failure reads `E404 Not Found -
PUT https://registry.npmjs.org/stanwasm` — npm answers a rejected credential
with a 404 so that it does not confirm the package exists. That is what
happened at 0.5.0, tagged with the job present and the trust never configured:
the tag was spent, `github-release` succeeded, and the publish failed. Configure
the trust first, then add the job.
