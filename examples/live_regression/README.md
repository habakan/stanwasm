# live_regression — stan-wasm-rs demo

A single-purpose demo: drag a point on the scatter plot and watch two
Bayesian regressions over the same data refit **live**, every animation
frame, with no server in the loop.

Both fits share priors and data; they differ only in likelihood:

- **robust** — `y ~ student_t(4, alpha + beta * x, sigma)`. No conjugate
  posterior; this is a genuine case for MCMC.
- **normal** — `y ~ normal(alpha + beta * x, sigma)`. Conjugate under a
  normal prior — closed-form, no sampler needed.

Drag one point far from the rest: the normal fit gets pulled toward it,
the robust fit barely moves. That's the point of the demo — not just "it's
fast," but a model (Student-t likelihood) that has no closed form and
genuinely needs sampling, made tangible by watching it happen live. This is
the UX that only makes sense once sampling runs entirely in the browser — a
server-backed tool would need a debounced network round trip per drag frame
and would visibly lag.

## Run locally

```bash
# From the repo root: build the wasm bundle first.
./scripts/build-wasm.sh

# Then run the demo dev server.
cd examples/live_regression
npm install
npm run dev          # auto-copies the wasm into public/ before starting
```

Open `http://localhost:5173`.

## What it shows

- Dragging a point re-runs `new StanModel(...)` + `model.sample(...)` for
  *both* models on every animation frame (throttled via
  `requestAnimationFrame`), each warm-started from its own previous
  posterior mean (`sample()`'s `init` argument) so NUTS converges fast
  enough for a few hundred warmup/draws per frame to feel instant — both
  fits together typically resample in single-digit milliseconds for a
  dozen points.
- `model.constrainDraw(...)` on a subsample of the robust fit's draws, to
  plot both its posterior mean line and a faint "spaghetti" band of
  individual posterior draws — the uncertainty visibly widens/narrows as
  points move.
- Click empty space to add a point, double-click a point to remove it
  (down to a minimum of 3).

For a broader tour of the API (CSV upload, editable Stan source, multiple
model presets, posterior summary table), see
[`examples/get_started`](../get_started).
