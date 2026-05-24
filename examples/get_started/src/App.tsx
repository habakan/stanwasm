import { useEffect, useRef, useState } from "react";
import init, { StanModel } from "stan-wasm-rs";
import { PRESETS, type Preset } from "./models";
import { Histogram } from "./Histogram";
import { DataTable } from "./DataTable";
import { csvToData } from "./csv";

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
  /** When set, overrides the preset's bundled data. */
  const [customData, setCustomData] = useState<Record<string, number | number[]> | null>(null);
  const [csvError, setCsvError] = useState<string | null>(null);
  const [csvFilename, setCsvFilename] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  /** When set, overrides the preset's bundled Stan code. */
  const [customStan, setCustomStan] = useState<string | null>(null);

  useEffect(() => {
    // wasm is copied into public/ by the `copy-wasm` npm script and served
    // at BASE_URL + filename. We pass it explicitly because the default
    // wasm-bindgen resolution uses the JS file's location, which lives in
    // ../../ts/pkg/ and is outside Vite's served scope by default.
    const wasmUrl = `${import.meta.env.BASE_URL}stan_wasm_api_bg.wasm`;
    init({ module_or_path: wasmUrl }).then(() => setLoaded(true));
  }, []);

  const preset: Preset = PRESETS[presetKey];
  const effectiveData = customData ?? preset.data;
  const effectiveStan = customStan ?? preset.stanCode;

  const onPresetChange = (key: keyof typeof PRESETS) => {
    setPresetKey(key);
    setCustomData(null);
    setCsvError(null);
    setCsvFilename(null);
    setCustomStan(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const resetStan = () => setCustomStan(null);

  const onCsvFile = async (file: File) => {
    setCsvError(null);
    const text = await file.text();
    const result = csvToData(text);
    if ("message" in result) {
      setCsvError(result.message);
      setCustomData(null);
      setCsvFilename(null);
      return;
    }
    setCustomData(result.data);
    setCsvFilename(file.name);
  };

  const resetCsv = () => {
    setCustomData(null);
    setCsvError(null);
    setCsvFilename(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const onRun = async () => {
    setRunning(true);
    setError(null);
    setSummary(null);
    setElapsedMs(null);
    try {
      const model = new StanModel(effectiveStan, JSON.stringify(effectiveData));
      // Use the preset's hand-tuned init when sizes match, else default to
      // a small non-zero vector (avoids degenerate gradients at exactly 0).
      const init =
        preset.init.length === model.n_params
          ? new Float64Array(preset.init)
          : new Float64Array(model.n_params).fill(0.1);
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
              onChange={(e) => onPresetChange(e.target.value as keyof typeof PRESETS)}
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
            <label>Data:</label>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,text/csv"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onCsvFile(f);
              }}
            />
            {customData ? (
              <>
                <span style={{ fontSize: 13, color: "#047857" }}>
                  ✓ {csvFilename}
                </span>
                <button className="secondary" onClick={resetCsv}>
                  Reset to preset
                </button>
              </>
            ) : (
              <span style={{ fontSize: 13, color: "#666" }}>
                any CSV — columns become Stan vectors; row count exposed as{" "}
                <code>N</code> / <code>J</code> / <code>K</code>. Edit the Stan
                program if your columns differ from the preset.
              </span>
            )}
          </div>

          {csvError && (
            <div className="note" style={{ borderLeftColor: "#b91c1c", background: "#fee2e2" }}>
              <strong>CSV error:</strong> {csvError}
            </div>
          )}

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
            <h3>
              Stan program
              {customStan !== null && (
                <>
                  {" "}
                  <span style={{ fontSize: 12, color: "#047857", fontWeight: 400 }}>
                    (edited)
                  </span>
                  {" "}
                  <button
                    className="secondary"
                    style={{ padding: "1px 8px", fontSize: 12 }}
                    onClick={resetStan}
                  >
                    Reset to preset
                  </button>
                </>
              )}
            </h3>
            <textarea
              className="stan-editor"
              value={effectiveStan}
              onChange={(e) => setCustomStan(e.target.value)}
              spellCheck={false}
            />
          </div>

          <div className="code-section">
            <h3>Data {customData && <span style={{ fontSize: 12, color: "#047857", fontWeight: 400 }}>(custom CSV)</span>}</h3>
            <DataTable data={effectiveData} />
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
