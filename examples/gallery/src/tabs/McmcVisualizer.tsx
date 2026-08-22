import { useEffect, useRef, useState } from "react";
import { StanModel } from "stan-wasm-rs";
import { GraphicalModel } from "../graphicalModel";

// Neal's funnel: a hard 2D posterior with no closed form and a notoriously
// difficult geometry (wide at large y, vanishingly narrow at small y) — the
// canonical illustration of why a naive random-walk proposal struggles and
// gradient-informed samplers like NUTS matter.
const STAN_CODE = `parameters {
  real y;
  real x;
}
model {
  y ~ normal(0, 3);
  x ~ normal(0, exp(y / 2));
}`;

// [0,0] has a zero gradient in x (the mean of x's conditional distribution)
// and nuts-rs rejects exactly-zero-gradient initial points — same reason
// other demos in this gallery seed away from zero.
const INIT: [number, number] = [0.3, 0.3];

const Y_DOMAIN: [number, number] = [-9, 9]; // vertical: top = wide mouth, bottom = narrow neck
const X_DOMAIN: [number, number] = [-20, 20];
const PANEL_W = 320;
const PANEL_H = 320;
const GRID_N = 90;
const N_WARMUP = 500;
const N_DRAWS = 500;
const TOTAL_ITERS = N_WARMUP + N_DRAWS;
const MIN_CHAINS = 1;
const MAX_CHAINS = 8;
const DEFAULT_CHAINS = 4;
const RWM_STEP = 1.0;
const TRAIL_LEN = 15;
const CHAIN_COLORS = [
  "#c2410c", "#0369a1", "#15803d", "#7c3aed",
  "#be185d", "#a16207", "#0e7490", "#4d7c0f",
];

const xToPx = (x: number) => ((x - X_DOMAIN[0]) / (X_DOMAIN[1] - X_DOMAIN[0])) * PANEL_W;
const yToPx = (y: number) => PANEL_H - ((y - Y_DOMAIN[0]) / (Y_DOMAIN[1] - Y_DOMAIN[0])) * PANEL_H;

// ---- seeded RNG (mulberry32) + Box-Muller gaussian, for the RWM baseline
// only — NUTS's randomness lives entirely on the Rust side. ----
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianFactory(rng: () => number) {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u: number, v: number, s: number;
    do {
      u = rng() * 2 - 1;
      v = rng() * 2 - 1;
      s = u * u + v * v;
    } while (s === 0 || s >= 1);
    const mul = Math.sqrt((-2 * Math.log(s)) / s);
    spare = v * mul;
    return u * mul;
  };
}

type Point = [number, number]; // [y, x]

interface RwmChainState {
  pos: Point;
  lp: number;
  rng: () => number;
  gauss: () => number;
}

interface ChainLogEntry {
  draw: number;
  tuning: boolean;
  diverging: boolean;
  stepSize: number;
  numSteps: number;
}

function computeHeatmap(model: StanModel): ImageData {
  const img = new ImageData(GRID_N, GRID_N);
  for (let j = 0; j < GRID_N; j++) {
    const yVal = Y_DOMAIN[1] - (j / (GRID_N - 1)) * (Y_DOMAIN[1] - Y_DOMAIN[0]);
    const rowLp: number[] = [];
    for (let i = 0; i < GRID_N; i++) {
      const xVal = X_DOMAIN[0] + (i / (GRID_N - 1)) * (X_DOMAIN[1] - X_DOMAIN[0]);
      rowLp.push(model.logProbGrad(new Float64Array([yVal, xVal]))[0]);
    }
    // Row-normalized (per y-slice), not globally: the funnel's true density
    // scale varies by orders of magnitude across rows (sigma_x = exp(y/2)),
    // which would make a global-max heatmap look blank except at one row.
    // This shows the funnel's *shape*, which is the pedagogical point.
    const rowMax = Math.max(...rowLp);
    for (let i = 0; i < GRID_N; i++) {
      const rel = Math.min(1, Math.max(0, (rowLp[i] - rowMax) / 8 + 1));
      const idx = (j * GRID_N + i) * 4;
      img.data[idx] = Math.round(250 + (194 - 250) * rel);
      img.data[idx + 1] = Math.round(250 + (65 - 250) * rel);
      img.data[idx + 2] = Math.round(250 + (12 - 250) * rel);
      img.data[idx + 3] = 255;
    }
  }
  return img;
}

/** "Fog of war" veil over the true-density panel: a light gray haze that
 *  starts opaque everywhere and clears wherever draws have actually landed,
 *  revealing the orange truth underneath. (An earlier version tinted
 *  covered cells blue instead — additively mixing a color on top of the
 *  funnel's already-saturated orange neck just produced murky purple there,
 *  unreadable exactly where coverage mattered most. A brightness veil never
 *  has that problem: it only ever *un-obscures*, never mixes hues.)
 *  Computed on the same grid as the true-density heatmap so the two line up
 *  pixel-for-pixel; recomputed from scratch every frame — cheap, at most a
 *  few hundred points per chain into a small grid. */
const VEIL_SPLAT = [
  [0, 0, 1], [-1, 0, 0.5], [1, 0, 0.5], [0, -1, 0.5], [0, 1, 0.5],
  [-1, -1, 0.25], [-1, 1, 0.25], [1, -1, 0.25], [1, 1, 0.25],
] as const;

function computeExplorationVeil(paths: Point[][]): ImageData {
  const counts = new Float64Array(GRID_N * GRID_N);
  let maxCount = 0;
  for (const path of paths) {
    for (const [yv, xv] of path) {
      if (yv < Y_DOMAIN[0] || yv > Y_DOMAIN[1] || xv < X_DOMAIN[0] || xv > X_DOMAIN[1]) continue;
      const j = Math.min(GRID_N - 1, Math.floor(((Y_DOMAIN[1] - yv) / (Y_DOMAIN[1] - Y_DOMAIN[0])) * GRID_N));
      const i = Math.min(GRID_N - 1, Math.floor(((xv - X_DOMAIN[0]) / (X_DOMAIN[1] - X_DOMAIN[0])) * GRID_N));
      // Splat each draw over its 3x3 neighborhood, not just its exact cell —
      // a lone visited cell reads as a nearly invisible fleck at this grid
      // resolution; a small blob is what actually looks "revealed".
      for (const [di, dj, w] of VEIL_SPLAT) {
        const ni = i + di;
        const nj = j + dj;
        if (ni < 0 || ni >= GRID_N || nj < 0 || nj >= GRID_N) continue;
        const idx = nj * GRID_N + ni;
        counts[idx] += w;
        if (counts[idx] > maxCount) maxCount = counts[idx];
      }
    }
  }
  const img = new ImageData(GRID_N, GRID_N);
  for (let k = 0; k < counts.length; k++) {
    const rel = maxCount > 0 ? counts[k] / maxCount : 0;
    // Any visited cell jumps to at least half-revealed immediately — sqrt
    // alone left a single early visit almost imperceptible against a
    // hundred-plus-cell hot spot, which read as "too faint to tell anything
    // happened here" rather than "lightly explored".
    const revealed = counts[k] > 0 ? Math.min(1, 0.5 + 0.5 * Math.sqrt(rel)) : 0;
    const idx = k * 4;
    img.data[idx] = 245;
    img.data[idx + 1] = 245;
    img.data[idx + 2] = 245;
    img.data[idx + 3] = Math.round((1 - revealed) * 235); // opaque fog when unexplored, clears toward 0
  }
  return img;
}

function drawImageToCanvas(canvas: HTMLCanvasElement | null, img: ImageData | null) {
  if (!canvas || !img) return;
  const off = document.createElement("canvas");
  off.width = img.width;
  off.height = img.height;
  off.getContext("2d")!.putImageData(img, 0, 0);
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, PANEL_W, PANEL_H);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, 0, 0, PANEL_W, PANEL_H);
}

function drawForeground(canvas: HTMLCanvasElement | null, paths: Point[][]) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, PANEL_W, PANEL_H);
  paths.forEach((path, c) => {
    if (path.length === 0) return;
    const color = CHAIN_COLORS[c % CHAIN_COLORS.length];

    // Where every draw so far landed is already shown by the fog-of-war
    // veil (see computeExplorationVeil) — this layer only needs the
    // recent trail, so it doesn't re-encode the same information twice.
    const start = Math.max(0, path.length - TRAIL_LEN);
    ctx.beginPath();
    for (let i = start; i < path.length; i++) {
      const [yv, xv] = path[i];
      const px = xToPx(xv);
      const py = yToPx(yv);
      if (i === start) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    const [yv, xv] = path[path.length - 1];
    ctx.beginPath();
    ctx.arc(xToPx(xv), yToPx(yv), 4, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = "white";
    ctx.lineWidth = 1;
    ctx.stroke();
  });
}

export function McmcVisualizer() {
  const [numChains, setNumChains] = useState(DEFAULT_CHAINS);
  const [restartKey, setRestartKey] = useState(0);
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [frame, setFrame] = useState(0);
  const [rwmAccepts, setRwmAccepts] = useState<number[]>([]);
  const [nutsDivergences, setNutsDivergences] = useState<number[]>([]);
  // One line of live status per NUTS chain, read straight off each chain's
  // own stepDraw() return — step_size/num_steps are nuts-rs's own
  // dual-averaging adaptation and trajectory-length search, not a value
  // this app computes. A JS reimplementation of "a NUTS demo" would have to
  // fake these; the real sampler produces them, independently per chain, as
  // a side effect of actually running.
  const [nutsChainLog, setNutsChainLog] = useState<Array<ChainLogEntry | null>>([]);
  const nutsChainLogRef = useRef<Array<ChainLogEntry | null>>([]);

  const sharedModelRef = useRef<StanModel | null>(null); // heatmap + all RWM logProbGrad calls
  const heatmapRef = useRef<ImageData | null>(null);
  const nutsModelsRef = useRef<StanModel[]>([]);
  const nutsPathsRef = useRef<Point[][]>([]);
  const nutsDoneRef = useRef<boolean[]>([]);
  const nutsDivergeCountRef = useRef<number[]>([]);
  const rwmChainsRef = useRef<RwmChainState[]>([]);
  const rwmPathsRef = useRef<Point[][]>([]);
  const rwmAcceptCountRef = useRef<number[]>([]);
  // Play/Pause toggles this, not a dependency the animation effect restarts
  // on — see the comment on that effect for why.
  const playingRef = useRef(false);

  const nutsBgRef = useRef<HTMLCanvasElement>(null);
  const nutsFgRef = useRef<HTMLCanvasElement>(null);
  const rwmBgRef = useRef<HTMLCanvasElement>(null);
  const rwmFgRef = useRef<HTMLCanvasElement>(null);
  const nutsHistRef = useRef<HTMLCanvasElement>(null);
  const rwmHistRef = useRef<HTMLCanvasElement>(null);
  const truthRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    playingRef.current = playing;
  }, [playing]);

  function tick() {
    let anyActive = false;

    nutsModelsRef.current.forEach((m, c) => {
      if (nutsDoneRef.current[c]) return;
      try {
        const out = m.stepDraw();
        nutsPathsRef.current[c].push([out[0], out[1]]);
        if (out[3] === 1.0) nutsDivergeCountRef.current[c] += 1;
        nutsChainLogRef.current[c] = {
          draw: nutsPathsRef.current[c].length,
          tuning: out[2] === 1.0,
          diverging: out[3] === 1.0,
          stepSize: out[4],
          numSteps: out[5],
        };
      } catch {
        nutsDoneRef.current[c] = true;
        return;
      }
      if (nutsPathsRef.current[c].length >= TOTAL_ITERS) nutsDoneRef.current[c] = true;
      else anyActive = true;
    });

    const shared = sharedModelRef.current;
    if (shared) {
      rwmChainsRef.current.forEach((st, c) => {
        if (rwmPathsRef.current[c].length >= TOTAL_ITERS) return;
        try {
          const prop: Point = [st.pos[0] + st.gauss() * RWM_STEP, st.pos[1] + st.gauss() * RWM_STEP];
          const lpProp = shared.logProbGrad(new Float64Array(prop))[0];
          if (Math.log(st.rng()) < lpProp - st.lp) {
            st.pos = prop;
            st.lp = lpProp;
            rwmAcceptCountRef.current[c] += 1;
          }
          rwmPathsRef.current[c].push([...st.pos]);
          anyActive = true;
        } catch {
          rwmPathsRef.current[c].push([...st.pos]); // stop advancing this chain, but keep its length in sync
        }
      });
    }

    setFrame((f) => f + 1);
    setRwmAccepts([...rwmAcceptCountRef.current]);
    setNutsDivergences([...nutsDivergeCountRef.current]);
    setNutsChainLog([...nutsChainLogRef.current]);
    drawForeground(nutsFgRef.current, nutsPathsRef.current);
    drawForeground(rwmFgRef.current, rwmPathsRef.current);
    drawImageToCanvas(nutsHistRef.current, computeExplorationVeil(nutsPathsRef.current));
    drawImageToCanvas(rwmHistRef.current, computeExplorationVeil(rwmPathsRef.current));

    if (!anyActive) setPlaying(false);
  }

  // One effect builds the chains AND drives the whole animation loop, rather
  // than a setup effect plus a separate ready/playing-driven "start the raf
  // loop" effect — simpler to reason about, and scheduling/cancelling the
  // animation frame happen inside one synchronous setup/cleanup pair.
  // Play/Pause reads `playingRef` (not a dependency), so toggling it never
  // tears down and rebuilds this effect — it only skips calling `tick()`
  // while paused.
  //
  // Freeing a `StanModel` while its step-sampling chain is mid-flight can
  // occasionally throw a wasm-level trap (a real fault, not just a "null
  // pointer" JS error) — reproducible by unmounting this component while it
  // is actively animating. `tick()` already catches per-chain so one bad
  // draw doesn't take down the others; the cleanup's `free()` calls must be
  // caught the same way, since an uncaught exception in a React effect
  // cleanup is treated as fatal and unmounts the whole tree. Note: React
  // StrictMode's dev-only mount→cleanup→remount simulation creates and
  // immediately discards one throwaway generation of these models on every
  // mount, which visibly triggers this far more often in `npm run dev` than
  // `npm run build` ever would (verified clean in a production build across
  // repeated full runs and repeated tab-switches mid-animation) — the catch
  // here is what keeps a dev-mode trap from being visibly disruptive either
  // way.
  useEffect(() => {
    let cancelled = false;

    const shared = new StanModel(STAN_CODE, "{}");
    sharedModelRef.current = shared;
    if (!heatmapRef.current) heatmapRef.current = computeHeatmap(shared);
    drawImageToCanvas(nutsBgRef.current, heatmapRef.current);
    drawImageToCanvas(rwmBgRef.current, heatmapRef.current);
    drawImageToCanvas(truthRef.current, heatmapRef.current);

    const nutsModels: StanModel[] = [];
    for (let c = 0; c < numChains; c++) {
      const m = new StanModel(STAN_CODE, "{}");
      m.startStepSampling(new Float64Array(INIT), N_WARMUP, N_DRAWS, BigInt(1000 + c));
      nutsModels.push(m);
    }
    nutsModelsRef.current = nutsModels;
    nutsPathsRef.current = Array.from({ length: numChains }, () => []);
    nutsDoneRef.current = new Array(numChains).fill(false);
    nutsDivergeCountRef.current = new Array(numChains).fill(0);

    const lp0 = shared.logProbGrad(new Float64Array(INIT))[0];
    const rwmChains: RwmChainState[] = [];
    for (let c = 0; c < numChains; c++) {
      const rng = mulberry32(2000 + c);
      rwmChains.push({ pos: [...INIT], lp: lp0, rng, gauss: gaussianFactory(rng) });
    }
    rwmChainsRef.current = rwmChains;
    rwmPathsRef.current = Array.from({ length: numChains }, () => []);
    rwmAcceptCountRef.current = new Array(numChains).fill(0);

    setFrame(0);
    setRwmAccepts(new Array(numChains).fill(0));
    setNutsDivergences(new Array(numChains).fill(0));
    nutsChainLogRef.current = new Array(numChains).fill(null);
    setNutsChainLog(nutsChainLogRef.current);
    drawForeground(nutsFgRef.current, nutsPathsRef.current);
    drawForeground(rwmFgRef.current, rwmPathsRef.current);
    drawImageToCanvas(nutsHistRef.current, computeExplorationVeil(nutsPathsRef.current));
    drawImageToCanvas(rwmHistRef.current, computeExplorationVeil(rwmPathsRef.current));
    setReady(true);
    setPlaying(true);

    let raf = requestAnimationFrame(function loop() {
      if (cancelled) return;
      if (playingRef.current) tick();
      raf = requestAnimationFrame(loop);
    });

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      nutsModels.forEach((m) => {
        try {
          m.free();
        } catch {
          // See the comment above this effect — freeing a mid-flight chain
          // can trap; already-caught inside tick() calls, must also be
          // caught here so an uncaught cleanup exception doesn't take down
          // the whole React tree.
        }
      });
      try {
        shared.free();
      } catch {
        // ditto
      }
      if (nutsModelsRef.current === nutsModels) nutsModelsRef.current = [];
      if (sharedModelRef.current === shared) sharedModelRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [numChains, restartKey]);

  const phase = frame < N_WARMUP ? "warmup" : "sampling";
  const totalAccept = rwmAccepts.reduce((a, b) => a + b, 0);
  const totalDiverge = nutsDivergences.reduce((a, b) => a + b, 0);
  const stepsShown = Math.min(frame, TOTAL_ITERS);

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
          Both panels step the same funnel posterior, one real draw at a time, live — this is not a
          precomputed replay. NUTS uses the gradient to take large, well-aimed steps; Random-Walk
          Metropolis only sees the log-density at each proposal.
        </p>

        <div className="controls">
          <label>
            Chains:
            <button
              className="secondary"
              onClick={() => setNumChains((c) => Math.max(MIN_CHAINS, c - 1))}
              disabled={numChains <= MIN_CHAINS}
            >
              −
            </button>
            <span className="chain-count">{numChains}</span>
            <button
              className="secondary"
              onClick={() => setNumChains((c) => Math.min(MAX_CHAINS, c + 1))}
              disabled={numChains >= MAX_CHAINS}
            >
              +
            </button>
          </label>
          <button onClick={() => setPlaying((p) => !p)} disabled={!ready}>
            {playing ? "Pause" : stepsShown >= TOTAL_ITERS ? "Done" : "Resume"}
          </button>
          <button className="secondary" onClick={() => setRestartKey((k) => k + 1)}>
            Restart
          </button>
          <span className="timing">
            step {stepsShown}/{TOTAL_ITERS} — {phase}
          </span>
        </div>

        <div className="mcmc-panels">
          <div className="mcmc-panel">
            <h4>NUTS</h4>
            <div className="canvas-stack">
              <canvas ref={nutsBgRef} width={PANEL_W} height={PANEL_H} className="plot-wrap" />
              <canvas ref={nutsHistRef} width={PANEL_W} height={PANEL_H} className="plot-wrap overlay" />
              <canvas ref={nutsFgRef} width={PANEL_W} height={PANEL_H} className="plot-wrap overlay" />
            </div>
            <p className="panel-stat">{totalDiverge} divergence{totalDiverge === 1 ? "" : "s"} across all chains</p>
          </div>
          <div className="mcmc-panel">
            <h4>Random-Walk Metropolis</h4>
            <div className="canvas-stack">
              <canvas ref={rwmBgRef} width={PANEL_W} height={PANEL_H} className="plot-wrap" />
              <canvas ref={rwmHistRef} width={PANEL_W} height={PANEL_H} className="plot-wrap overlay" />
              <canvas ref={rwmFgRef} width={PANEL_W} height={PANEL_H} className="plot-wrap overlay" />
            </div>
            <p className="panel-stat">
              acceptance rate: {stepsShown > 0 ? ((totalAccept / (stepsShown * numChains)) * 100).toFixed(0) : "…"}%
            </p>
          </div>
          <div className="mcmc-panel">
            <h4>Ground Truth</h4>
            <div className="canvas-stack">
              <canvas ref={truthRef} width={PANEL_W} height={PANEL_H} className="plot-wrap" />
            </div>
            <p className="panel-stat">the funnel's actual density — no fog, nothing sampled</p>
          </div>
        </div>

        <pre className="wasm-log">
          {nutsChainLog.map((entry, c) => {
            const color = CHAIN_COLORS[c % CHAIN_COLORS.length];
            const label = `chain ${c + 1}`.padEnd(8);
            if (!entry) return `${label} …\n`;
            const phaseTag = entry.tuning ? "warmup  " : "sampling";
            const divergedTag = entry.diverging ? "  DIVERGED" : "";
            return (
              <span key={c} style={{ color }}>
                {label}
                draw {String(entry.draw).padStart(4)}/{TOTAL_ITERS}  {phaseTag}  step_size={entry.stepSize.toFixed(4)}  leapfrog_steps={String(entry.numSteps).padStart(2)}
                {divergedTag}
                {"\n"}
              </span>
            );
          })}
        </pre>

        <div className="legend">
          <span><i className="dot raw" style={{ background: "#c2410c", borderColor: "#c2410c" }} /> revealed (sampled)</span>
          <span><i className="dot raw" style={{ background: "#d4d4d4", borderColor: "#a3a3a3" }} /> fogged (not yet sampled)</span>
        </div>
        <div className="legend">
          {Array.from({ length: numChains }, (_, c) => (
            <span key={c}>
              <i className="swatch solid" style={{ borderTopColor: CHAIN_COLORS[c % CHAIN_COLORS.length] }} />
              chain {c + 1}
            </span>
          ))}
        </div>

        <div className="note">
          Both panels call the same underlying wasm model, one draw per animation frame, for every chain
          — {N_WARMUP} warmup + {N_DRAWS} sampling draws each. NUTS's step-by-step API
          (<code>startStepSampling</code>/<code>stepDraw</code>) keeps the sampler's state alive between
          calls instead of running the whole chain in one shot, so what you're watching is the actual
          computation happening now, not a replay of a finished run. The gray haze is fog of war: it
          starts opaque everywhere and clears — revealing the true orange density underneath — wherever
          a draw actually lands, recomputed from a live 2D histogram of every draw so far each frame.
          The log below is one line per NUTS chain, straight from that chain's own <code>stepDraw()</code>
          call each frame — <code>step_size</code> and <code>leapfrog_steps</code> are nuts-rs's own live
          dual-averaging adaptation and trajectory-length search, not values this app computes, and they
          vary chain to chain because each chain really is its own independent sampler instance running
          in wasm right now, not a shared script driving identical-looking lines.
        </div>
      </div>
    </div>
  );
}
