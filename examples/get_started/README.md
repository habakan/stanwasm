# get_started — stan-wasm-rs demo

Minimal Vite + React app showing how to embed `stan-wasm-rs` into a frontend.
Three preset Stan models (linear regression, Poisson regression, eight schools
non-centered) sample entirely in the browser.

## Run locally

```bash
# From the repo root: build the wasm bundle first.
./scripts/build-wasm.sh

# Then run the demo dev server.
cd examples/get_started
npm install
npm run dev          # auto-copies the wasm into public/ before starting
```

Open `http://localhost:5173`.

## Build for static hosting

```bash
npm run build
# dist/ is fully static — drop into GitHub Pages, Vercel, S3, etc.
```

## What it shows

- `init()` to load the wasm
- `new StanModel(stanCode, dataJson)` to compile a model
- `model.sample(init, nWarmup, nDraws, seed)` to run NUTS
- `model.n_params` and `model.paramNames()` for metadata
- Posterior summarisation in plain JS (mean / sd / 80% interval) plus an
  inline SVG histogram per parameter

About 220 lines across all `src/` files. Use it as a template for your own
integration.
