import { useEffect, useRef, useState } from "react";
import { StanModel } from "stan-wasm-rs";
import { GraphicalModel } from "../graphicalModel";

// Classic partial-pooling model (the "eight schools" structure, non-centered
// parameterization). Each group has an observed value y_j with known noise
// sigma_j; theta_j is the partially-pooled estimate, shrunk toward the
// population mean mu by an amount that grows with sigma_j relative to tau
// (the between-group SD) — groups you're less sure about individually
// borrow more strength from the rest. No closed form; this is what NUTS is
// for.
const STAN_CODE = `data {
  int<lower=0> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}
parameters {
  real mu;
  real<lower=0> tau;
  vector[J] theta_tilde;
}
transformed parameters {
  vector[J] theta = mu + tau * theta_tilde;
}
model {
  mu ~ normal(0, 10);
  tau ~ half_normal(10);
  theta_tilde ~ normal(0, 1);
  y ~ normal(theta, sigma);
}`;

interface Group {
  label: string;
  y: number;
  /** Known observation noise — fixed per group, not draggable. Three groups
   *  are "confident" (small sigma), three are "noisy" (large sigma), so the
   *  contrast in how hard each shrinks is visible without any interaction. */
  sigma: number;
}

const INITIAL_GROUPS: Group[] = [
  { label: "A", y: 8, sigma: 3 },
  { label: "B", y: 5, sigma: 3 },
  { label: "C", y: -2, sigma: 3 },
  { label: "D", y: 6, sigma: 12 },
  { label: "E", y: 10, sigma: 12 },
  { label: "F", y: 2, sigma: 12 },
];

const Y_DOMAIN: [number, number] = [-30, 35];
const W = 640;
const H = 380;
const PAD = 40;
const N_WARMUP = 200;
const N_DRAWS = 200;
const SEED = 42n;
const MARK_OFFSET = 15;

const J = INITIAL_GROUPS.length;
const slotW = (W - 2 * PAD) / J;
const xForIndex = (i: number) => PAD + (i + 0.5) * slotW;
const yToPx = (y: number) => H - PAD - ((y - Y_DOMAIN[0]) / (Y_DOMAIN[1] - Y_DOMAIN[0])) * (H - 2 * PAD);
const pxToY = (py: number) => Y_DOMAIN[0] + ((H - PAD - py) / (H - 2 * PAD)) * (Y_DOMAIN[1] - Y_DOMAIN[0]);
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

interface Fit {
  mu: number;
  tau: number;
  thetaMean: number[];
  thetaSd: number[];
  elapsedMs: number;
}

export function HierarchicalShrinkage() {
  const [groups, setGroups] = useState<Group[]>(INITIAL_GROUPS);
  const [fit, setFit] = useState<Fit | null>(null);
  const [dragging, setDragging] = useState<number | null>(null);

  const modelRef = useRef<StanModel | null>(null);
  const warmStart = useRef<number[] | null>(null);
  const rafRef = useRef<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    return () => {
      modelRef.current?.free();
    };
  }, []);

  useEffect(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      resample(groups);
    });
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [groups]);

  function resample(gs: Group[]) {
    const t0 = performance.now();
    const data = { J: gs.length, y: gs.map((g) => g.y), sigma: gs.map((g) => g.sigma) };
    modelRef.current?.free();
    let model: StanModel;
    try {
      model = new StanModel(STAN_CODE, JSON.stringify(data));
    } catch {
      modelRef.current = null;
      return;
    }
    modelRef.current = model;
    const n = model.n_params; // mu, tau, theta_tilde[1..J] -> 2 + J
    const init = warmStart.current ?? [0, 1, ...new Array(gs.length).fill(0)];
    const draws = model.sample(new Float64Array(init), N_WARMUP, N_DRAWS, SEED);
    const post = draws.subarray(N_WARMUP * n);

    const means = new Array(n).fill(0);
    for (let i = 0; i < N_DRAWS; i++) {
      for (let j = 0; j < n; j++) means[j] += post[i * n + j];
    }
    for (let j = 0; j < n; j++) means[j] /= N_DRAWS;
    warmStart.current = means;

    // constrainDraw() output order: parameters (mu, tau, theta_tilde[1..J])
    // then transformed_params (theta[1..J]) — theta starts right after n.
    const thetaStart = n;
    const thetaSum = new Array(gs.length).fill(0);
    const thetaSqSum = new Array(gs.length).fill(0);
    let muSum = 0, tauSum = 0;
    for (let i = 0; i < N_DRAWS; i++) {
      const c = model.constrainDraw(post.subarray(i * n, (i + 1) * n));
      muSum += c[0];
      tauSum += c[1];
      for (let j = 0; j < gs.length; j++) {
        const th = c[thetaStart + j];
        thetaSum[j] += th;
        thetaSqSum[j] += th * th;
      }
    }
    const thetaMean = thetaSum.map((s) => s / N_DRAWS);
    const thetaSd = thetaMean.map((m, j) => Math.sqrt(Math.max(0, thetaSqSum[j] / N_DRAWS - m * m)));
    setFit({ mu: muSum / N_DRAWS, tau: tauSum / N_DRAWS, thetaMean, thetaSd, elapsedMs: performance.now() - t0 });
  }

  const onPointerDown = (i: number) => (e: React.PointerEvent<SVGCircleElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    setDragging(i);
  };

  const onPointerMove = (i: number) => (e: React.PointerEvent<SVGCircleElement>) => {
    if (dragging !== i) return;
    const rect = svgRef.current!.getBoundingClientRect();
    const py = ((e.clientY - rect.top) / rect.height) * H;
    const y = clamp(pxToY(py), Y_DOMAIN[0], Y_DOMAIN[1]);
    setGroups((prev) => prev.map((g, j) => (j === i ? { ...g, y } : g)));
  };

  const onPointerUp = () => setDragging(null);

  return (
    <div className="demo-layout">
      <div className="demo-model">
        <div className="model-diagram">
          <GraphicalModel stanCode={STAN_CODE} />
        </div>
        <div className="code-blocks">
          <div className="code-block">
            <h4>Stan model</h4>
            <pre>{STAN_CODE}</pre>
          </div>
        </div>
      </div>

      <div className="demo-interactive">
      <p className="hint">
        Drag a group's open circle (its observed value). Groups A–C are confident (σ=3); D–F are noisy
        (σ=12). Drag one far from the rest and compare how much its orange estimate follows versus how
        much it gets pulled back toward the dashed population mean.
      </p>

      <svg ref={svgRef} className="plot-wrap" viewBox={`0 0 ${W} ${H}`} width={W} height={H}>
        <rect x={0} y={0} width={W} height={H} fill="white" />
        <line x1={PAD} y1={yToPx(0)} x2={W - PAD} y2={yToPx(0)} stroke="#eee" pointerEvents="none" />
        {fit && (
          <line
            x1={PAD}
            y1={yToPx(fit.mu)}
            x2={W - PAD}
            y2={yToPx(fit.mu)}
            stroke="#c2410c"
            strokeWidth={1.5}
            strokeDasharray="6 4"
            pointerEvents="none"
          />
        )}

        {groups.map((g, i) => {
          const cx = xForIndex(i);
          const rawX = cx - MARK_OFFSET;
          const shrunkX = cx + MARK_OFFSET;
          const rawY = yToPx(g.y);
          const thetaMean = fit?.thetaMean[i];
          const thetaSd = fit?.thetaSd[i] ?? 0;
          const shrunkY = thetaMean !== undefined ? yToPx(thetaMean) : null;
          return (
            <g key={g.label}>
              {shrunkY !== null && (
                <line x1={rawX} y1={rawY} x2={shrunkX} y2={shrunkY} stroke="#ccc" strokeWidth={1} pointerEvents="none" />
              )}
              {/* observed value ± known sigma */}
              <line
                x1={rawX}
                y1={yToPx(g.y - g.sigma)}
                x2={rawX}
                y2={yToPx(g.y + g.sigma)}
                stroke="#64748b"
                strokeWidth={1.5}
                pointerEvents="none"
              />
              {/* partially-pooled estimate ± posterior sd */}
              {shrunkY !== null && thetaMean !== undefined && (
                <line
                  x1={shrunkX}
                  y1={yToPx(thetaMean - thetaSd)}
                  x2={shrunkX}
                  y2={yToPx(thetaMean + thetaSd)}
                  stroke="#c2410c"
                  strokeWidth={1.5}
                  opacity={0.5}
                  pointerEvents="none"
                />
              )}
              {shrunkY !== null && (
                <circle cx={shrunkX} cy={shrunkY} r={6} fill="#c2410c" stroke="#c2410c" pointerEvents="none" />
              )}
              <circle
                className="draggable"
                cx={rawX}
                cy={rawY}
                r={7}
                fill="white"
                stroke="#64748b"
                strokeWidth={2}
                onPointerDown={onPointerDown(i)}
                onPointerMove={onPointerMove(i)}
                onPointerUp={onPointerUp}
              />
              <text x={cx} y={H - PAD + 20} textAnchor="middle" fontSize={12} fill="#888">
                {g.label} (σ={g.sigma})
              </text>
            </g>
          );
        })}
      </svg>

      <div className="legend">
        <span><i className="dot raw" /> observed yⱼ ± σⱼ</span>
        <span><i className="dot shrunk" /> partially-pooled θⱼ ± posterior sd</span>
        <span><i className="swatch dashed" /> population mean μ</span>
      </div>

      <div className="readout">
        <span className="stat">
          μ = <b>{fit ? fit.mu.toFixed(2) : "…"}</b>
        </span>
        <span className="stat">
          τ = <b>{fit ? fit.tau.toFixed(2) : "…"}</b>
        </span>
        {fit && <span className="timing">resampled in {fit.elapsedMs.toFixed(1)}ms</span>}
      </div>

      <button onClick={() => setGroups(INITIAL_GROUPS)}>Reset groups</button>

      <div className="note">
        Every drag frame recompiles the model and runs {N_WARMUP} warmup + {N_DRAWS} NUTS draws — all
        inside WebAssembly, in this tab. The amount of shrinkage (how far the orange estimate sits from
        the gray observed value) isn't a fixed rule someone hand-tuned — it falls out of the posterior,
        different for every group, every frame.
      </div>
      </div>
    </div>
  );
}
