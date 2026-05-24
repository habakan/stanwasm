import { useEffect, useState } from "react";
import init, { StanModel } from "stan-wasm-rs";
import { PRESETS, type Preset } from "./models";
import { Histogram } from "./Histogram";

interface ParamSummary {
  name: string;
  mean: number;
  sd: number;
  q10: number;
  q90: number;
  samples: number[];
}

function quantile(sorted: number[], p: number): number {
  const idx = (sorted.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

function summarize(name: string, samples: number[]): ParamSummary {
  const sorted = [...samples].sort((a, b) => a - b);
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  const variance =
    samples.reduce((acc, v) => acc + (v - mean) ** 2, 0) / samples.length;
  return {
    name,
    mean,
    sd: Math.sqrt(variance),
    q10: quantile(sorted, 0.1),
    q90: quantile(sorted, 0.9),
    samples,
  };
}

export function App() {
  const [loaded, setLoaded] = useState(false);
  const [presetKey, setPresetKey] = useState<keyof typeof PRESETS>("linear_regression");
  const [nWarmup, setNWarmup] = useState(500);
  const [nDraws, setNDraws] = useState(1000);
  const [seed, setSeed] = useState(42);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [summary, setSummary] = useState<ParamSummary[] | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);

  useEffect(() => {
    init().then(() => setLoaded(true));
  }, []);

  const preset: Preset = PRESETS[presetKey];

  const onRun = async () => {
    setRunning(true);
    setError(null);
    setSummary(null);
    setElapsedMs(null);
    try {
      const model = new StanModel(preset.stanCode, JSON.stringify(preset.data));
      const init = new Float64Array(preset.init);
      const t0 = performance.now();
      const samples = model.sample(init, nWarmup, nDraws, BigInt(seed));
      const elapsed = performance.now() - t0;
      const n = model.n_params;
      const names = model.paramNames();
      const post = samples.subarray(nWarmup * n);
      const draws: number[][] = Array.from({ length: n }, () => []);
      for (let i = 0; i < nDraws; i++) {
        for (let j = 0; j < n; j++) draws[j].push(post[i * n + j]);
      }
      setSummary(draws.map((vals, i) => summarize(names[i], vals)));
      setElapsedMs(elapsed);
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="app">
      <h1>stan-wasm-rs — get started</h1>
      <p className="tagline">
        Stan probabilistic models sampling entirely in your browser.{" "}
        <a href="https://github.com/habakan/stan-wasm-rs" target="_blank" rel="noreferrer">
          GitHub
        </a>
        .
      </p>

      {!loaded && <p>Loading WebAssembly bundle…</p>}

      {loaded && (
        <>
          <div className="row">
            <label>Model:</label>
            <select
              value={presetKey}
              onChange={(e) => setPresetKey(e.target.value as keyof typeof PRESETS)}
            >
              {Object.entries(PRESETS).map(([k, p]) => (
                <option key={k} value={k}>
                  {p.name}
                </option>
              ))}
            </select>
            <span style={{ color: "#666", fontSize: 13 }}>{preset.description}</span>
          </div>

          <div className="row">
            <label>Warmup:</label>
            <input
              type="number"
              value={nWarmup}
              min={0}
              step={100}
              onChange={(e) => setNWarmup(Number(e.target.value))}
            />
            <label>Draws:</label>
            <input
              type="number"
              value={nDraws}
              min={1}
              step={100}
              onChange={(e) => setNDraws(Number(e.target.value))}
            />
            <label>Seed:</label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
            />
            <button onClick={onRun} disabled={running}>
              {running ? "Sampling…" : "Run NUTS"}
            </button>
          </div>

          <div className="code-section">
            <h3>Stan program</h3>
            <pre>{preset.stanCode}</pre>
          </div>

          <div className="code-section">
            <h3>Data (JSON)</h3>
            <pre>{JSON.stringify(preset.data, null, 2)}</pre>
          </div>

          {error && (
            <div className="note" style={{ borderLeftColor: "#b91c1c", background: "#fee2e2" }}>
              <strong>Error:</strong> {error}
            </div>
          )}

          {summary && (
            <div className="results">
              <h2>
                Posterior summary{" "}
                {elapsedMs && (
                  <span style={{ fontSize: 13, fontWeight: 400, color: "#666" }}>
                    — sampled in {elapsedMs.toFixed(0)}ms
                  </span>
                )}
              </h2>

              <div className="param-row header">
                <span>parameter</span>
                <span className="num">mean</span>
                <span className="num">sd</span>
                <span className="num">q10</span>
                <span className="num">q90</span>
                <span>posterior</span>
              </div>
              {summary.map((p) => (
                <div className="param-row" key={p.name}>
                  <span className="param-name">{p.name}</span>
                  <span className="num">{p.mean.toFixed(3)}</span>
                  <span className="num">{p.sd.toFixed(3)}</span>
                  <span className="num">{p.q10.toFixed(3)}</span>
                  <span className="num">{p.q90.toFixed(3)}</span>
                  <Histogram values={p.samples} />
                </div>
              ))}
            </div>
          )}

          <div className="note">
            All sampling runs in WebAssembly inside your browser. No data leaves your device.
            For full Stan language coverage and a polished UI, see{" "}
            <a href="https://stan-playground.flatironinstitute.org" target="_blank" rel="noreferrer">
              Stan Playground
            </a>
            .
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
