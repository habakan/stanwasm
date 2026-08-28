---
name: Bug report
about: Something broke that you expected to work
title: 'bug: '
labels: bug
---

## What happened

<!-- short description -->

## How to reproduce

<!-- minimal code or commands -->

```bash
cargo test -p stanwasm-runtime --test log_prob
```

or, for browser / Node:

```ts
import init, { StanModel } from "stanwasm";
await init();
// ...
```

## What you expected

<!-- briefly -->

## Environment

- stanwasm version / commit:
- Rust version (`rustc --version`):
- Browser / Node.js version (if applicable):
- OS:

## Additional context

<!-- stack traces, screenshots, logs -->
