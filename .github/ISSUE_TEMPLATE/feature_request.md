---
name: Feature request
about: Propose an addition or change
title: 'feat: '
labels: enhancement
---

## What problem does this solve?

<!-- the underlying need, not the proposed solution -->

## Proposed change

<!-- API sketch, distribution to add, constraint transform, etc. -->

## Is this in scope?

See `CONTRIBUTING.md` for the project scope. In particular:
- We target a Stan subset, not full Stan compatibility
- The sampler is `nuts-rs` only — no PRs adding ADVI / pathfinder / fixed_param
- Browser-only by design — no server backend features

If you are not sure whether the proposal is in scope, that is OK — flag it explicitly and we'll discuss.

## Alternatives considered

<!-- workarounds, related tools (cmdstan, Stan Playground, PyMC, etc.) -->
