import { useEffect, useRef, useState } from "react";
import init, { StanModel } from "stan-wasm-rs";

// Two likelihoods over the same data, fit side by side. `normal` has a
// closed-form (conjugate) posterior — no sampler needed. `student_t` with a
// small nu (heavy tails) does not: it's genuinely a case where you need
// NUTS. Dragging one point far from the rest makes the difference obvious —
// the normal fit gets dragged toward the outlier, the robust fit barely
// moves.
const STAN_ROBUST = `data {
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
  y ~ student_t(4, alpha + beta * x, sigma);
}`;

const STAN_NORMAL = `data {
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

// Fixed data-space domain: dragging never rescales the plot, so the fits
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
  meanAlpha: number;
  meanBeta: number;
  meanSigma: number;
  /** [alpha, beta] for a subsample of posterior draws — the "spaghetti" band. */
  lines: [number, number][];
}

const INITIAL_POINTS: Point[] = [
  { x: -2.4, y: -4.6 }, { x: -2.0, y: -3.3 }, { x: -1.6, y: -3.0 },
  { x: -1.2, y: -1.4 }, { x: -0.8, y: -1.6 }, { x: -0.4, y: 0.2 },
  { x: 0.0, y: 0.6 }, { x: 0.4, y: 0.9 }, { x: 0.8, y: 2.1 },
  { x: 1.2, y: 1.9 }, { x: 1.6, y: 3.4 }, { x: 2.0, y: 3.6 },
  { x: 2.4, y: 4.8 },
];

/** One model's worth of persistent wasm state: the compiled StanModel (freed
 *  and rebuilt every resample, since data changes) and the previous
 *  posterior mean, reused as the next `sample()` call's `init` so NUTS
 *  starts close to where it'll end up. */
function useModelSlot() {
  const modelRef = useRef<StanModel | null>(null);
  const warmStart = useRef<number[] | null>(null);
  return { modelRef, warmStart };
}

function fitModel(
  stanCode: string,
  pts: Point[],
  slot: ReturnType<typeof useModelSlot>,
  wantSpaghetti: boolean,
): Fit | null {
  const data = { N: pts.length, x: pts.map((p) => p.x), y: pts.map((p) => p.y) };
  slot.modelRef.current?.free();
  let model: StanModel;
  try {
    model = new StanModel(stanCode, JSON.stringify(data));
  } catch {
    slot.modelRef.current = null;
    return null;
  }
  slot.modelRef.current = model;
  const n = model.n_params; // alpha, beta, sigma — always 3 for both models here
  const init = slot.warmStart.current ?? [0, 1, 0];
  const draws = model.sample(new Float64Array(init), N_WARMUP, N_DRAWS, SEED);
  const post = draws.subarray(N_WARMUP * n);

  const meansUnconstrained = [0, 0, 0];
  for (let i = 0; i < N_DRAWS; i++) {
    for (let j = 0; j < n; j++) meansUnconstrained[j] += post[i * n + j];
  }
  for (let j = 0; j < n; j++) meansUnconstrained[j] /= N_DRAWS;
  slot.warmStart.current = meansUnconstrained;

  if (!wantSpaghetti) {
    // alpha/beta have no constraint transform, so their unconstrained mean
    // already is the constrained mean — no need to walk every draw.
    return {
      meanAlpha: meansUnconstrained[0],
      meanBeta: meansUnconstrained[1],
      meanSigma: Math.exp(meansUnconstrained[2]),
      lines: [],
    };
  }

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
  return { meanAlpha: sumAlpha / count, meanBeta: sumBeta / count, meanSigma: sumSigma / count, lines };
}

export function App() {
  const [loaded, setLoaded] = useState(false);
  const [points, setPoints] = useState<Point[]>(INITIAL_POINTS);
  const [robustFit, setRobustFit] = useState<Fit | null>(null);
  const [normalFit, setNormalFit] = useState<Fit | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [dragging, setDragging] = useState<number | null>(null);

  const robustSlot = useModelSlot();
  const normalSlot = useModelSlot();
  const rafRef = useRef<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const wasmUrl = `${import.meta.env.BASE_URL}stan_wasm_api_bg.wasm`;
    init({ module_or_path: wasmUrl }).then(() => setLoaded(true));
    return () => {
      robustSlot.modelRef.current?.free();
      normalSlot.modelRef.current?.free();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Resample on every point change, throttled to one run per animation
  // frame — during a drag this reads as continuous, not stepped.
  useEffect(() => {
    if (!loaded) return;
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      const t0 = performance.now();
      const robust = fitModel(STAN_ROBUST, points, robustSlot, true);
      const normal = fitModel(STAN_NORMAL, points, normalSlot, false);
      setElapsedMs(performance.now() - t0);
      if (robust) setRobustFit(robust);
      if (normal) setNormalFit(normal);
    });
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points, loaded]);

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
        Two Bayesian fits over the same points: <b>robust</b> (Student-t likelihood, no closed form) and{" "}
        <b>normal</b> (conjugate — a spreadsheet could do this one). Drag a point far from the rest and
        watch them diverge. Every frame is a fresh NUTS run, entirely in this tab — no server, no network
        round trip.{" "}
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
            (min {MIN_POINTS}). Try dragging one point far away — that's where the two fits split.
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

            {/* robust fit: posterior draws, faint — the "uncertainty band" */}
            {robustFit?.lines.map(([a, b], i) => {
              const l = lineAt(a, b);
              return <line key={i} {...l} stroke="#c2410c" strokeWidth={1} opacity={0.1} pointerEvents="none" />;
            })}

            {/* normal (OLS-like) fit: dashed, for contrast */}
            {normalFit && (
              <line
                {...lineAt(normalFit.meanAlpha, normalFit.meanBeta)}
                stroke="#64748b"
                strokeWidth={2}
                strokeDasharray="7 5"
                pointerEvents="none"
              />
            )}

            {/* robust fit: posterior mean */}
            {robustFit && (
              <line
                {...lineAt(robustFit.meanAlpha, robustFit.meanBeta)}
                stroke="#c2410c"
                strokeWidth={2.5}
                pointerEvents="none"
              />
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

          <div className="legend">
            <span><i className="swatch solid" /> robust (Student-t, ν=4)</span>
            <span><i className="swatch dashed" /> normal (conjugate)</span>
          </div>

          <div className="readout">
            <span className="stat">
              robust: α = <b>{robustFit ? robustFit.meanAlpha.toFixed(2) : "…"}</b> β ={" "}
              <b>{robustFit ? robustFit.meanBeta.toFixed(2) : "…"}</b> σ ={" "}
              <b>{robustFit ? robustFit.meanSigma.toFixed(2) : "…"}</b>
            </span>
            <span className="stat">
              normal: α = <b>{normalFit ? normalFit.meanAlpha.toFixed(2) : "…"}</b> β ={" "}
              <b>{normalFit ? normalFit.meanBeta.toFixed(2) : "…"}</b>
            </span>
            {elapsedMs !== null && <span className="timing">both resampled in {elapsedMs.toFixed(1)}ms</span>}
          </div>

          <button onClick={() => setPoints(INITIAL_POINTS)}>Reset points</button>

          <div className="note">
            Each drag frame recompiles both models, runs {N_WARMUP} warmup + {N_DRAWS} NUTS draws for each,
            and re-derives the constrained posteriors — all inside WebAssembly, in this tab. Nothing is
            sent anywhere. The robust fit's Student-t likelihood has no conjugate posterior — this is a
            case sampling is actually for. See <code>examples/get_started</code> for a fuller API tour
            (CSV upload, editable Stan source, multiple presets).
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
