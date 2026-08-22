# live_regression — stan-wasm-rs demo

A single-purpose demo: drag a point on the scatter plot and watch the
Bayesian linear regression refit **live**, every animation frame, with no
server in the loop. This is the UX that only makes sense once sampling runs
entirely in the browser — a server-backed tool would need a debounced
network round trip per drag frame and would visibly lag.

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

- Dragging a point re-runs `new StanModel(...)` + `model.sample(...)` on
  every animation frame (throttled via `requestAnimationFrame`), warm-started
  from the previous posterior mean (`sample()`'s `init` argument) so NUTS
  converges fast enough for a few hundred warmup/draws per frame to feel
  instant.
- `model.constrainDraw(...)` on a subsample of draws, to plot both the
  posterior mean fit line and a faint "spaghetti" band of individual
  posterior draws — the fit's uncertainty visibly widens/narrows as points
  move.
- Click empty space to add a point, double-click a point to remove it
  (down to a minimum of 3).

For a broader tour of the API (CSV upload, editable Stan source, multiple
model presets, posterior summary table), see
[`examples/get_started`](../get_started).

About 200 lines in `src/App.tsx`, no chart library.
