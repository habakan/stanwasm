# gallery — stan-wasm-rs examples

One Vite + React app, four tabs, each showing a different reason sampling
entirely in the browser is worth having.

## Run locally

```bash
# From the repo root: build the wasm bundle first.
./scripts/build-wasm.sh

# Then run the demo dev server.
cd examples/gallery
npm install
npm run dev          # auto-copies the wasm into public/ before starting
```

Open `http://localhost:5173`.

## Tabs

- **MCMC Visualizer** — NUTS and Random-Walk Metropolis step the same hard 2D
  posterior (Neal's funnel — wide at large `y`, vanishingly narrow at small
  `y`) side by side, one real draw per animation frame, adjustable chain
  count. This uses `StanModel`'s step-by-step sampling API
  (`startStepSampling`/`stepDraw`), not `sample()` — the sampler's state
  stays alive between calls so what's on screen is the actual computation
  happening now, not a replay of an already-finished chain. Each panel is
  also a live "fog of war": a gray haze over the true density clears wherever
  a draw actually lands, so which regions a method has (and hasn't) explored
  is visible at a glance. A third "Ground Truth" panel shows the same true
  density with no fog, as a fixed reference to compare both methods against.
  The NUTS panel also shows a live `step_size`/leapfrog-step readout, read
  straight off nuts-rs's own adaptation state each draw — real sampler
  internals, not a value the app computes, and a way to see the actual wasm
  computation at work since raw speed isn't a meaningful difference at this
  model's tiny size. 500 warmup + 500 sampling draws per chain.
  `src/tabs/McmcVisualizer.tsx`.
- **Live Regression** — drag a point; a robust (Student-t, no closed form)
  and a normal (conjugate, closed form) regression refit **live**, every
  animation frame, over the same data. Drag one point into outlier
  territory and watch them diverge — the normal fit gets dragged toward it,
  the robust fit barely moves. `src/tabs/LiveRegression.tsx`.
- **Hierarchical Shrinkage** — six marketing campaigns' observed A/B test
  CTR lift: three ran at full traffic (small standard error), three were
  small-sample pilots (large standard error), fit with a partial-pooling
  model (the classic "eight schools" structure). Drag a campaign's observed
  value and watch its posterior estimate resist by an amount that depends
  on that campaign's own sample size, not a hand-tuned rule — that's what
  falls out of the joint posterior, and it's exactly the real-world
  argument for not trusting a dramatic small-sample result at face value.
  Also shows the population distribution N(μ, τ) (band or density-curve
  toggle) and, per campaign, a KDE over its own real posterior draws.
  `src/tabs/HierarchicalShrinkage.tsx`.
- **Get Started** — a fuller API tour: CSV upload, editable Stan source,
  multiple model presets, posterior summary table with histograms.
  `src/tabs/GetStarted.tsx`. Sample CSVs for its presets live under
  `sample-csv/`.

Every drag frame reconstructs the relevant `StanModel`(s) and runs NUTS
(warm-started from the previous posterior mean via `sample()`'s `init`
argument) — single-digit milliseconds for the small models here. wasm is
loaded once in `src/App.tsx` and shared across tabs; only the active tab is
mounted, so switching tabs frees the inactive one's compiled model.

## Graphical models

Every "graphical model" diagram (the node-and-plate plot next to each Stan
code block, including a live one in Get Started that follows the editor) is
parsed straight out of the Stan source in `src/graphicalModel.tsx` — nodes
from `data`/`parameters`/`transformed parameters` declarations, edges and
distribution formulas from `model`-block sampling statements — rather than
hand-drawn per tab. Distribution formulas render via MathJax, served from a
locally-copied bundle (`copy-mathjax` in `package.json`, output to
`public/mathjax-tex-mml-chtml.js`) rather than a CDN, so the "runs fully
offline" story holds for this too.
