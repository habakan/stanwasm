import { useEffect, useRef, useState } from "react";
import init, { StanModel } from "stan-wasm-rs";

const STAN_CODE = `data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 10);
  beta  ~ normal(0, 10);
  sigma ~ exponential(1);
  y ~ normal(alpha + beta * x, sigma);
}`;

// Fixed data-space domain: dragging never rescales the plot, so the fit
// visibly moving is the only thing that changes frame to frame.
const X_DOMAIN: [number, number] = [-3, 3];
const Y_DOMAIN: [number, number] = [-8, 8];
const W = 640;
const H = 400;
const PAD = 32;
const MIN_POINTS = 3;
const N_WARMUP = 150;
const N_DRAWS = 150;
const N_SPAGHETTI = 25;
const SEED = 42n;

const xToPx = (x: number) => PAD + ((x - X_DOMAIN[0]) / (X_DOMAIN[1] - X_DOMAIN[0])) * (W - 2 * PAD);
const yToPx = (y: number) => H - PAD - ((y - Y_DOMAIN[0]) / (Y_DOMAIN[1] - Y_DOMAIN[0])) * (H - 2 * PAD);
const pxToX = (px: number) => X_DOMAIN[0] + ((px - PAD) / (W - 2 * PAD)) * (X_DOMAIN[1] - X_DOMAIN[0]);
const pxToY = (py: number) => Y_DOMAIN[0] + ((H - PAD - py) / (H - 2 * PAD)) * (Y_DOMAIN[1] - Y_DOMAIN[0]);
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

interface Point {
  x: number;
  y: number;
}

interface Fit {
  /** [alpha, beta] for a subsample of posterior draws, for the "spaghetti" band. */
  lines: [number, number][];
  meanAlpha: number;
  meanBeta: number;
  meanSigma: number;
  elapsedMs: number;
}

const INITIAL_POINTS: Point[] = [
  { x: -2.4, y: -4.6 }, { x: -2.0, y: -3.3 }, { x: -1.6, y: -3.0 },
  { x: -1.2, y: -1.4 }, { x: -0.8, y: -1.6 }, { x: -0.4, y: 0.2 },
  { x: 0.0, y: 0.6 }, { x: 0.4, y: 0.9 }, { x: 0.8, y: 2.1 },
  { x: 1.2, y: 1.9 }, { x: 1.6, y: 3.4 }, { x: 2.0, y: 3.6 },
  { x: 2.4, y: 4.8 },
];

export function App() {
  const [loaded, setLoaded] = useState(false);
  const [points, setPoints] = useState<Point[]>(INITIAL_POINTS);
  const [fit, setFit] = useState<Fit | null>(null);
  const [dragging, setDragging] = useState<number | null>(null);

  const modelRef = useRef<StanModel | null>(null);
  const warmStart = useRef<number[] | null>(null);
  const rafRef = useRef<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const wasmUrl = `${import.meta.env.BASE_URL}stan_wasm_api_bg.wasm`;
    init({ module_or_path: wasmUrl }).then(() => setLoaded(true));
    return () => {
      modelRef.current?.free();
    };
  }, []);

  // Resample on every point change, throttled to one run per animation
  // frame — during a drag this reads as continuous, not stepped.
  useEffect(() => {
    if (!loaded) return;
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      resample(points);
    });
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points, loaded]);

  function resample(pts: Point[]) {
    const t0 = performance.now();
    const data = { N: pts.length, x: pts.map((p) => p.x), y: pts.map((p) => p.y) };
    modelRef.current?.free();
    let model: StanModel;
    try {
      model = new StanModel(STAN_CODE, JSON.stringify(data));
    } catch {
      modelRef.current = null;
      return;
    }
    modelRef.current = model;
    const n = model.n_params; // alpha, beta, sigma — always 3 for this model
    const init = warmStart.current ?? [0, 1, 0];
    const draws = model.sample(new Float64Array(init), N_WARMUP, N_DRAWS, SEED);
    const post = draws.subarray(N_WARMUP * n);

    const meansUnconstrained = [0, 0, 0];
    for (let i = 0; i < N_DRAWS; i++) {
      for (let j = 0; j < n; j++) meansUnconstrained[j] += post[i * n + j];
    }
    for (let j = 0; j < n; j++) meansUnconstrained[j] /= N_DRAWS;
    warmStart.current = meansUnconstrained;

    const step = Math.max(1, Math.floor(N_DRAWS / N_SPAGHETTI));
    const lines: [number, number][] = [];
    let sumAlpha = 0, sumBeta = 0, sumSigma = 0, count = 0;
    for (let i = 0; i < N_DRAWS; i += step) {
      const c = model.constrainDraw(post.subarray(i * n, (i + 1) * n));
      lines.push([c[0], c[1]]);
      sumAlpha += c[0];
      sumBeta += c[1];
      sumSigma += c[2];
      count++;
    }
    setFit({
      lines,
      meanAlpha: sumAlpha / count,
      meanBeta: sumBeta / count,
      meanSigma: sumSigma / count,
      elapsedMs: performance.now() - t0,
    });
  }

  function svgPointFromEvent(e: { clientX: number; clientY: number }) {
    const rect = svgRef.current!.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const py = ((e.clientY - rect.top) / rect.height) * H;
    return {
      x: clamp(pxToX(px), X_DOMAIN[0], X_DOMAIN[1]),
      y: clamp(pxToY(py), Y_DOMAIN[0], Y_DOMAIN[1]),
    };
  }

  const onBackgroundPointerDown = (e: React.PointerEvent<SVGRectElement>) => {
    const p = svgPointFromEvent(e);
    setPoints((prev) => [...prev, p]);
  };

  const onPointPointerDown = (i: number) => (e: React.PointerEvent<SVGCircleElement>) => {
    e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    setDragging(i);
  };

  const onPointPointerMove = (i: number) => (e: React.PointerEvent<SVGCircleElement>) => {
    if (dragging !== i) return;
    e.stopPropagation();
    const p = svgPointFromEvent(e);
    setPoints((prev) => prev.map((pt, j) => (j === i ? p : pt)));
  };

  const onPointPointerUp = (e: React.PointerEvent<SVGCircleElement>) => {
    e.stopPropagation();
    setDragging(null);
  };

  const onPointDoubleClick = (i: number) => (e: React.MouseEvent) => {
    e.stopPropagation();
    if (points.length <= MIN_POINTS) return;
    setPoints((prev) => prev.filter((_, j) => j !== i));
  };

  const lineAt = (alpha: number, beta: number) => ({
    x1: xToPx(X_DOMAIN[0]),
    y1: yToPx(alpha + beta * X_DOMAIN[0]),
    x2: xToPx(X_DOMAIN[1]),
    y2: yToPx(alpha + beta * X_DOMAIN[1]),
  });

  return (
    <div className="app">
      <h1>stan-wasm-rs — live regression</h1>
      <p className="tagline">
        Drag a point. Every frame you see is a fresh NUTS run, entirely in this tab —{" "}
        no server, no network round trip.{" "}
        <a href="https://github.com/habakan/stan-wasm-rs" target="_blank" rel="noreferrer">
          GitHub
        </a>
        .
      </p>

      {!loaded && <p>Loading WebAssembly bundle…</p>}

      {loaded && (
        <>
          <p className="hint">
            Click empty space to add a point · drag a point to move it · double-click a point to remove it
            (min {MIN_POINTS}).
          </p>

          <svg ref={svgRef} className="plot-wrap" viewBox={`0 0 ${W} ${H}`} width={W} height={H}>
            <rect x={0} y={0} width={W} height={H} fill="white" onPointerDown={onBackgroundPointerDown} />
            {/* axes and fit lines are decorative only — pointerEvents="none" so
                they never steal a click from the background "add point" handler */}
            <line
              x1={xToPx(X_DOMAIN[0])}
              y1={yToPx(0)}
              x2={xToPx(X_DOMAIN[1])}
              y2={yToPx(0)}
              stroke="#eee"
              pointerEvents="none"
            />
            <line
              x1={xToPx(0)}
              y1={yToPx(Y_DOMAIN[0])}
              x2={xToPx(0)}
              y2={yToPx(Y_DOMAIN[1])}
              stroke="#eee"
              pointerEvents="none"
            />

            {/* posterior draws, faint — the "uncertainty band" */}
            {fit?.lines.map(([a, b], i) => {
              const l = lineAt(a, b);
              return <line key={i} {...l} stroke="#c2410c" strokeWidth={1} opacity={0.1} pointerEvents="none" />;
            })}

            {/* posterior mean fit */}
            {fit && (
              <line {...lineAt(fit.meanAlpha, fit.meanBeta)} stroke="#c2410c" strokeWidth={2.5} pointerEvents="none" />
            )}

            {points.map((p, i) => (
              <circle
                key={i}
                className="point"
                cx={xToPx(p.x)}
                cy={yToPx(p.y)}
                r={7}
                fill="white"
                stroke="#c2410c"
                strokeWidth={2}
                onPointerDown={onPointPointerDown(i)}
                onPointerMove={onPointPointerMove(i)}
                onPointerUp={onPointPointerUp}
                onDoubleClick={onPointDoubleClick(i)}
              />
            ))}
          </svg>

          <div className="readout">
            <span className="stat">
              α = <b>{fit ? fit.meanAlpha.toFixed(2) : "…"}</b>
            </span>
            <span className="stat">
              β = <b>{fit ? fit.meanBeta.toFixed(2) : "…"}</b>
            </span>
            <span className="stat">
              σ = <b>{fit ? fit.meanSigma.toFixed(2) : "…"}</b>
            </span>
            {fit && <span className="timing">resampled in {fit.elapsedMs.toFixed(1)}ms</span>}
          </div>

          <button onClick={() => setPoints(INITIAL_POINTS)}>Reset points</button>

          <div className="note">
            Each drag frame recompiles the model, runs {N_WARMUP} warmup + {N_DRAWS} NUTS draws, and
            re-derives the constrained posterior — all inside WebAssembly, in this tab. Nothing is sent
            anywhere. See <code>examples/get_started</code> for a fuller API tour (CSV upload, editable
            Stan source, multiple presets).
          </div>

          <footer>
            stan-wasm-rs · alpha · Apache-2.0 · embedded{" "}
            <a href="https://github.com/pymc-devs/nuts-rs" target="_blank" rel="noreferrer">
              nuts-rs
            </a>{" "}
            sampler
          </footer>
        </>
      )}
    </div>
  );
}
