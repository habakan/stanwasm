---
name: Stan model parsing or evaluation failure
about: A Stan model that should parse / evaluate but does not
title: 'parser: '
labels: parser, help-wanted
---

## Stan code (minimal repro)

```stan
data {
  // ...
}
parameters {
  // ...
}
model {
  // ...
}
```

## Data (JSON)

```json
{ "N": 10, "y": [...] }
```

## What you expected

- [ ] Should parse cleanly
- [ ] Parses but `log_prob_grad` returns wrong value
- [ ] Parses but `log_prob_grad` returns NaN / Inf
- [ ] Other:

## What stanwasm does

<!-- error message, stack trace, or numeric output -->

## What cmdstan / PyStan does (if you checked)

<!-- optional but very helpful for oracle comparison -->

## Coverage check

Is the model using a feature listed under "Not yet supported" in the README? If yes, please confirm — these are tracked separately.

## Environment

- stanwasm version / commit:
