import { useEffect, useRef, useState } from "react";
import { StanModel } from "stanwasm";
import { GraphicalModel } from "../graphicalModel";

// Classic partial-pooling model (the "eight schools" structure, non-centered
// parameterization), framed here as six marketing campaigns' observed CTR
// lift (y_j) with known standard error (sigma_j, driven by each campaign's
// sample size — a handful of visitors gives a noisy estimate, tens of
// thousands gives a tight one). theta_j is the partially-pooled estimate,
// shrunk toward the population mean mu by an amount that grows with
// sigma_j relative to tau (the between-campaign SD) — this is the concrete
// business case for partial pooling: a flashy lift number from a
// small-sample pilot shouldn't be trusted at face value the way a
// well-powered campaign's number is. No closed form; this is what NUTS is
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
  /** Descriptive-only sample-size hint shown under the label — not fed to
   *  the model, `sigma` is; this just explains where that sigma comes from. */
  n: string;
  /** Observed CTR lift, percentage points. */
  y: number;
  /** Known standard error — fixed per group, not draggable. Three campaigns
   *  are well-powered (small sigma), three are small pilots (large sigma),
   *  so the contrast in how hard each shrinks is visible without any
   *  interaction. */
  sigma: number;
}

// Chart labels are kept short enough to fit their slot without overlapping
// neighbors; see the hint text below for each one's full campaign name.
const INITIAL_GROUPS: Group[] = [
  { label: "Homepage", n: "n ≈ 48,000", y: 7, sigma: 3 },
  { label: "Email Subj.", n: "n ≈ 61,000", y: 4, sigma: 3 },
  { label: "Checkout", n: "n ≈ 35,000", y: -3, sigma: 3 },
  { label: "Landing Pg", n: "n ≈ 300", y: 22, sigma: 12 },
  { label: "Influencer", n: "n ≈ 450", y: -8, sigma: 12 },
  { label: "Referral", n: "n ≈ 260", y: 14, sigma: 12 },
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
  /** Every theta_j draw this resample produced, kept per group so the UI can
   *  show each group's actual posterior shape (a KDE over these) rather than
   *  just its mean/sd. */
  thetaDraws: number[][];
  elapsedMs: number;
}

type PopView = "band" | "curve";

/** Normal density, used only to draw the population distribution N(mu, tau)
 *  that every group's theta is pooled toward — not part of the model or the
 *  sampler, purely a visualization of the posterior mu/tau this tab already
 *  computes. */
function normalPdf(y: number, mu: number, tau: number): number {
  if (tau <= 0) return 0;
  const z = (y - mu) / tau;
  return Math.exp(-0.5 * z * z) / (tau * Math.sqrt(2 * Math.PI));
}

const CURVE_SAMPLES = 48;
const CURVE_MAX_HALF_WIDTH = (W / 2 - PAD) * 0.9;

/** A violin-style outline: width at each y is the population density there,
 *  normalized so the peak (at y = mu) spans CURVE_MAX_HALF_WIDTH on each
 *  side — this is what makes "how much does tau narrow/widen this" visible
 *  at a glance, the same way a wider or narrower bell curve would. */
function populationCurvePath(mu: number, tau: number): string {
  const peak = normalPdf(mu, mu, tau);
  if (peak <= 0) return "";
  const cx = W / 2;
  const yLo = Math.max(Y_DOMAIN[0], mu - 4 * tau);
  const yHi = Math.min(Y_DOMAIN[1], mu + 4 * tau);
  const pts: [number, number][] = [];
  for (let i = 0; i <= CURVE_SAMPLES; i++) {
    const y = yLo + ((yHi - yLo) * i) / CURVE_SAMPLES;
    const halfW = (normalPdf(y, mu, tau) / peak) * CURVE_MAX_HALF_WIDTH;
    pts.push([cx + halfW, yToPx(y)]);
  }
  for (let i = CURVE_SAMPLES; i >= 0; i--) {
    const y = yLo + ((yHi - yLo) * i) / CURVE_SAMPLES;
    const halfW = (normalPdf(y, mu, tau) / peak) * CURVE_MAX_HALF_WIDTH;
    pts.push([cx - halfW, yToPx(y)]);
  }
  return `M${pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" L")} Z`;
}

/** Gaussian-kernel density estimate — used only for the per-group violin
 *  below, evaluated directly over that group's own posterior draws (not a
 *  normal-distribution approximation from mean/sd, the way the population
 *  curve above necessarily is — theta_j's actual posterior needn't be
 *  symmetric, and this shows whatever shape 200 real NUTS draws produced). */
function kde(draws: number[], y: number, bandwidth: number): number {
  let sum = 0;
  for (const d of draws) {
    const z = (y - d) / bandwidth;
    sum += Math.exp(-0.5 * z * z);
  }
  return sum / (draws.length * bandwidth * Math.sqrt(2 * Math.PI));
}

const GROUP_CURVE_SAMPLES = 32;
const GROUP_CURVE_MAX_HALF_WIDTH = 16;

function groupCurvePath(draws: number[], mean: number, sd: number, cx: number): string {
  if (draws.length === 0 || sd <= 0) return "";
  // Silverman's rule of thumb for kernel bandwidth.
  const bandwidth = Math.max(1e-6, 1.06 * sd * Math.pow(draws.length, -0.2));
  const yLo = mean - 3.5 * sd;
  const yHi = mean + 3.5 * sd;
  const density = Array.from({ length: GROUP_CURVE_SAMPLES + 1 }, (_, i) =>
    kde(draws, yLo + ((yHi - yLo) * i) / GROUP_CURVE_SAMPLES, bandwidth),
  );
  const peak = Math.max(...density);
  if (peak <= 0) return "";
  const pts: [number, number][] = [];
  for (let i = 0; i <= GROUP_CURVE_SAMPLES; i++) {
    const y = yLo + ((yHi - yLo) * i) / GROUP_CURVE_SAMPLES;
    pts.push([cx + (density[i] / peak) * GROUP_CURVE_MAX_HALF_WIDTH, yToPx(y)]);
  }
  for (let i = GROUP_CURVE_SAMPLES; i >= 0; i--) {
    const y = yLo + ((yHi - yLo) * i) / GROUP_CURVE_SAMPLES;
    pts.push([cx - (density[i] / peak) * GROUP_CURVE_MAX_HALF_WIDTH, yToPx(y)]);
  }
  return `M${pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" L")} Z`;
}

export function HierarchicalShrinkage() {
  const [groups, setGroups] = useState<Group[]>(INITIAL_GROUPS);
  const [fit, setFit] = useState<Fit | null>(null);
  const [dragging, setDragging] = useState<number | null>(null);
  const [popView, setPopView] = useState<PopView>("curve");

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
    const thetaDraws: number[][] = Array.from({ length: gs.length }, () => []);
    let muSum = 0, tauSum = 0;
    for (let i = 0; i < N_DRAWS; i++) {
      const c = model.constrainDraw(post.subarray(i * n, (i + 1) * n));
      muSum += c[0];
      tauSum += c[1];
      for (let j = 0; j < gs.length; j++) {
        const th = c[thetaStart + j];
        thetaSum[j] += th;
        thetaSqSum[j] += th * th;
        thetaDraws[j].push(th);
      }
    }
    const thetaMean = thetaSum.map((s) => s / N_DRAWS);
    const thetaSd = thetaMean.map((m, j) => Math.sqrt(Math.max(0, thetaSqSum[j] / N_DRAWS - m * m)));
    setFit({ mu: muSum / N_DRAWS, tau: tauSum / N_DRAWS, thetaMean, thetaSd, thetaDraws, elapsedMs: performance.now() - t0 });
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
        Six A/B tests' observed CTR lift. Homepage Banner, Email Subject Line, and Checkout Redesign ran
        at full traffic (tens of thousands of visitors each), so their numbers are trustworthy as-is. New
        Landing Page, Influencer Partnership, and Referral Program Pilot were small-sample pilots (a few
        hundred visitors) — drag one's open circle and watch how little its flashy-looking number actually
        moves the partially-pooled estimate next to it.
      </p>

      <div className="controls">
        <label>
          Population N(μ, τ):
          <select value={popView} onChange={(e) => setPopView(e.target.value as PopView)}>
            <option value="band">±τ / ±2τ band</option>
            <option value="curve">density curve</option>
          </select>
        </label>
      </div>

      <svg ref={svgRef} className="plot-wrap" viewBox={`0 0 ${W} ${H}`} width={W} height={H}>
        <rect x={0} y={0} width={W} height={H} fill="white" />
        <text x={PAD} y={PAD - 14} fontSize={11} fill="#888">CTR lift (%)</text>
        <line x1={PAD} y1={yToPx(0)} x2={W - PAD} y2={yToPx(0)} stroke="#eee" pointerEvents="none" />

        {fit && popView === "band" && (
          <>
            <rect
              x={PAD}
              y={Math.min(yToPx(fit.mu - 2 * fit.tau), yToPx(fit.mu + 2 * fit.tau))}
              width={W - 2 * PAD}
              height={Math.abs(yToPx(fit.mu - 2 * fit.tau) - yToPx(fit.mu + 2 * fit.tau))}
              fill="#c2410c"
              opacity={0.07}
              pointerEvents="none"
            />
            <rect
              x={PAD}
              y={Math.min(yToPx(fit.mu - fit.tau), yToPx(fit.mu + fit.tau))}
              width={W - 2 * PAD}
              height={Math.abs(yToPx(fit.mu - fit.tau) - yToPx(fit.mu + fit.tau))}
              fill="#c2410c"
              opacity={0.14}
              pointerEvents="none"
            />
          </>
        )}
        {fit && popView === "curve" && fit.tau > 0 && (
          <path
            d={populationCurvePath(fit.mu, fit.tau)}
            fill="#c2410c"
            opacity={0.16}
            stroke="#c2410c"
            strokeWidth={1}
            strokeOpacity={0.4}
            pointerEvents="none"
          />
        )}

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
              {/* theta_j's actual posterior shape — a KDE over its own real
                  draws from this resample, not a mean±sd approximation. */}
              {thetaMean !== undefined && fit && (
                <path
                  d={groupCurvePath(fit.thetaDraws[i], thetaMean, thetaSd, shrunkX)}
                  fill="#c2410c"
                  opacity={0.35}
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
              <text x={cx} y={H - PAD + 16} textAnchor="middle" fontSize={11} fill="#888">
                {g.label}
              </text>
              <text x={cx} y={H - PAD + 28} textAnchor="middle" fontSize={9} fill="#aaa">
                {g.n}
              </text>
            </g>
          );
        })}
      </svg>

      <div className="legend">
        <span><i className="dot raw" /> observed CTR lift ± standard error</span>
        <span><i className="dot shrunk" /> partially-pooled estimate (posterior mean)</span>
        <span><i className="dot raw" style={{ background: "#c2410c", opacity: 0.4, border: "none" }} /> that campaign's actual posterior (KDE over its own draws)</span>
        <span><i className="swatch dashed" /> population mean lift μ</span>
        <span><i className="dot raw" style={{ background: "#c2410c", opacity: 0.2, border: "none" }} /> population N(μ, τ) — what every campaign is pooled toward</span>
      </div>

      <div className="readout">
        <span className="stat">
          population mean lift μ = <b>{fit ? fit.mu.toFixed(2) : "…"}%</b>
        </span>
        <span className="stat">
          between-campaign SD τ = <b>{fit ? fit.tau.toFixed(2) : "…"}%</b>
        </span>
        {fit && <span className="timing">resampled in {fit.elapsedMs.toFixed(1)}ms</span>}
      </div>

      <button onClick={() => setGroups(INITIAL_GROUPS)}>Reset campaigns</button>

      <div className="note">
        Every drag frame recompiles the model and runs {N_WARMUP} warmup + {N_DRAWS} NUTS draws — all
        inside WebAssembly, in this tab. Notice that Landing Pg's, Influencer's, and Referral's flashy
        small-sample numbers (+22%, −8%, +14%) all sit much closer to each other's partially-pooled
        estimate than their raw observed values do — a hierarchical model automatically discounts a
        dramatic result that came from a few hundred visitors, without anyone hand-writing a "small
        sample size" rule. The wide shaded curve is the population distribution N(μ, τ), and its width
        (τ) is itself estimated from how much the six campaigns agree with each other: make them more
        consistent and watch it narrow; spread them out and watch it widen. The small violin at each
        campaign is that campaign's own posterior specifically — a kernel density estimate over its real
        200 NUTS draws from this resample, not a mean±sd stand-in — so you can compare an individual
        campaign's actual uncertainty against the population it's being shrunk toward.
      </div>
      </div>
    </div>
  );
}
