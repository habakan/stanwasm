# Security Policy

## Reporting a vulnerability

Please report security issues privately through GitHub's
[private vulnerability reporting](https://github.com/habakan/stanwasm/security/advisories/new)
rather than by opening a public issue. If that form isn't available to you, open
a normal issue saying only that you have a security report and asking for a
private channel — no details in the public issue.

This project is maintained on evenings and weekends, so expect an initial
acknowledgement within about a week. If a report turns out to be valid, the fix
and the advisory go out together.

## Supported versions

Only the latest release is supported. The project is pre-1.0 and alpha; there
are no backported security fixes for earlier versions.

## What the threat model actually is

stanwasm is a client-side library. It has no server, no database, no
authentication, and makes no network requests of its own. That shapes what
counts as a vulnerability here.

**In scope**

- Memory-safety or sandbox-escape issues in the emitted AOT WebAssembly, or in
  the `stanwasm-codegen` path that emits it. Every crate is
  `#![forbid(unsafe_code)]` and the emitted module imports only `Math.*` and the
  host's linear memory, so this would be a real finding.
- A crafted Stan model or data JSON that makes the parser or evaluator do
  something other than fail cleanly — for example non-terminating evaluation a
  host can't interrupt.
- Supply-chain problems: a compromised dependency, or a published npm/crates.io
  artifact that doesn't match this source tree.
- Anything that causes the published package to exfiltrate data. The library
  reads only the model source and data you hand it.

**Out of scope**

- A malicious Stan model crashing its own page. Untrusted model source is an
  expected input — the examples gallery lets you type one — and the worst case
  is a wasm trap that kills the module instance, which the browser sandbox
  contains and which has nothing else in it to steal. Reports of "model X
  panics" are still welcome as ordinary bug reports.
- Wrong numerical results. Those are correctness bugs, not vulnerabilities;
  please open a normal issue. They are taken seriously — see `CHANGELOG.md`.
- Development-server advisories affecting `examples/gallery`. Those are
  `devDependencies` of an example app and never reach the published package or
  the built static site.

## Verifying what you install

`stanwasm` 0.1.0 is on [npm](https://www.npmjs.com/package/stanwasm) and
[crates.io](https://crates.io/crates/stanwasm). It was published by hand from a
local checkout of the `v0.1.0` tag, so it carries **no**
[npm provenance](https://docs.npmjs.com/generating-provenance-statements)
attestation — provenance requires publishing from a CI workflow, which this
project does not do yet. Until it does, a tarball cannot be cryptographically
traced back to the commit that built it.

Independently of that, `ts/pkg/` is generated entirely by
`make wasm` from this source tree, so you can rebuild and compare
rather than trusting the published bytes.

## Hardening already in place

- Every crate is `#![forbid(unsafe_code)]`; there is no `unsafe` in the tree.
- CI pins third-party actions by commit SHA, and every workflow declares
  `permissions: contents: read` at the top level. Exactly two jobs widen that,
  each in its own job block so nothing else in the run inherits it:
  `release.yml`'s `github-release` (`contents: write`, to create the release)
  and `pages.yml`'s `deploy` (`pages: write` + `id-token: write`, which is what
  `actions/deploy-pages` mints its OIDC token with). No workflow uses secrets
  or `pull_request_target`, and publishing to crates.io and npm is done by
  hand — CI holds no registry credentials.
- Dependabot watches Cargo, npm, and GitHub Actions — see
  [`.github/dependabot.yml`](.github/dependabot.yml).
