import { useEffect, useMemo, useRef, useState } from "react";
import { StanModel } from "stan-wasm-rs";
import { PRESETS, type Preset } from "../models";
import { Histogram } from "../Histogram";
import { DataTable } from "../DataTable";
import { csvToData } from "../csv";
import { GraphicalModel } from "../graphicalModel";

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

export function WasmSandbox() {
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
  const [csvSkipped, setCsvSkipped] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  /** When set, overrides the preset's bundled Stan code. */
  const [customStan, setCustomStan] = useState<string | null>(null);
  /** Cached compiled model. null when never compiled or after reset. */
  const [compiledModel, setCompiledModel] = useState<StanModel | null>(null);
  const [compileError, setCompileError] = useState<string | null>(null);
  const [lastCompiledKey, setLastCompiledKey] = useState<string | null>(null);
  const [compiling, setCompiling] = useState(false);
  /** Bottom panel (Data / Posterior summary) — collapsible so the whole tab
   *  fits one viewport without ever needing to scroll the page itself; only
   *  this panel's own body scrolls, and only when its content outgrows it. */
  const [panelOpen, setPanelOpen] = useState(true);
  const [panelTab, setPanelTab] = useState<"data" | "results">("data");

  const preset: Preset = PRESETS[presetKey];
  const effectiveData = customData ?? preset.data;
  const effectiveStan = customStan ?? preset.stanCode;

  // Debounced so the graphical-model diagram (which re-parses the Stan
  // source and re-typesets MathJax) doesn't redo that work on every
  // keystroke while editing.
  const [debouncedStan, setDebouncedStan] = useState(effectiveStan);
  useEffect(() => {
    const t = setTimeout(() => setDebouncedStan(effectiveStan), 250);
    return () => clearTimeout(t);
  }, [effectiveStan]);

  /** Identifies the (stan, data) pair currently in the editor. When the
   *  cached compile was for a different key, the model is "stale". */
  const compileKey = useMemo(
    () => JSON.stringify({ s: effectiveStan, d: effectiveData }),
    [effectiveStan, effectiveData],
  );
  const stale = compiledModel !== null && lastCompiledKey !== compileKey;

  const compile = () => {
    setCompiling(true);
    setCompileError(null);
    // Tiny defer so the spinner state lands before the synchronous parse
    // and trace kicks the main thread.
    setTimeout(() => {
      try {
        if (compiledModel) compiledModel.free();
        const m = new StanModel(effectiveStan, JSON.stringify(effectiveData));
        setCompiledModel(m);
        setLastCompiledKey(compileKey);
      } catch (e) {
        setCompileError(String(e));
        setCompiledModel(null);
      } finally {
        setCompiling(false);
      }
    }, 0);
  };

  const onPresetChange = (key: keyof typeof PRESETS) => {
    setPresetKey(key);
    setCustomData(null);
    setCsvError(null);
    setCsvFilename(null);
    setCustomStan(null);
    if (compiledModel) compiledModel.free();
    setCompiledModel(null);
    setCompileError(null);
    setLastCompiledKey(null);
    setSummary(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const resetStan = () => setCustomStan(null);

  const onCsvFile = async (file: File) => {
    setCsvError(null);
    setCsvSkipped([]);
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
    setCsvSkipped(result.skippedColumns);
  };

  const resetCsv = () => {
    setCustomData(null);
    setCsvError(null);
    setCsvFilename(null);
    setCsvSkipped([]);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const onRun = async () => {
    if (!compiledModel || stale) return;
    setRunning(true);
    setError(null);
    setSummary(null);
    setElapsedMs(null);
    try {
      const initVec =
        preset.init.length === compiledModel.n_params
          ? new Float64Array(preset.init)
          : new Float64Array(compiledModel.n_params).fill(0.1);
      const t0 = performance.now();
      const samples = compiledModel.sample(initVec, nWarmup, nDraws, BigInt(seed));
      const elapsed = performance.now() - t0;
      const n = compiledModel.n_params;
      const names = compiledModel.paramNames();
      const post = samples.subarray(nWarmup * n);
      // `sample()` returns unconstrained draws (e.g. sigma on the log scale);
      // constrainDraw() maps each draw back to the natural scale and also
      // fills in transformed parameters, which paramNames() includes but the
      // raw draw doesn't carry.
      const draws: number[][] = Array.from({ length: names.length }, () => []);
      for (let i = 0; i < nDraws; i++) {
        const row = post.subarray(i * n, (i + 1) * n);
        const constrained = compiledModel.constrainDraw(row);
        for (let j = 0; j < names.length; j++) draws[j].push(constrained[j]);
      }
      setSummary(draws.map((vals, i) => summarize(names[i], vals)));
      setElapsedMs(elapsed);
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
      setPanelTab("results");
      setPanelOpen(true);
    }
  };

  return (
    <div className="sandbox">
      <div className="sandbox-toolbar">
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
          <label style={{ display: "inline-flex", alignItems: "center", gap: 4, fontWeight: 400 }}>
            <input
              type="radio"
              name="data-source"
              checked={customData === null}
              onChange={resetCsv}
            />
            Use preset
          </label>
          <label style={{ display: "inline-flex", alignItems: "center", gap: 4, fontWeight: 400 }}>
            <input
              type="radio"
              name="data-source"
              checked={customData !== null}
              onChange={() => fileInputRef.current?.click()}
            />
            Upload CSV
          </label>
          <button className="secondary" onClick={() => fileInputRef.current?.click()}>
            Choose File
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,text/csv"
            style={{ display: "none" }}
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
              {csvSkipped.length > 0 && (
                <span style={{ fontSize: 12, color: "#b45309" }}>
                  skipped non-numeric column(s): {csvSkipped.join(", ")}
                </span>
              )}
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
          <button className="secondary" onClick={compile} disabled={compiling}>
            {compiling ? "Compiling…" : compiledModel && !stale ? "Recompile" : "Compile"}
          </button>
          <button onClick={onRun} disabled={running || !compiledModel || stale}>
            {running ? "Sampling…" : "Run NUTS"}
          </button>
          <span className="compile-status">
            {compileError ? (
              <span style={{ color: "#b91c1c" }}>✗ compile error</span>
            ) : compiledModel && !stale ? (
              <span style={{ color: "#047857" }}>
                ✓ compiled ({compiledModel.n_params} params)
              </span>
            ) : compiledModel && stale ? (
              <span style={{ color: "#b45309" }}>
                ⚠ stale — recompile
              </span>
            ) : (
              <span style={{ color: "#888" }}>not compiled</span>
            )}
          </span>
        </div>

        {compileError && (
          <div className="note" style={{ borderLeftColor: "#b91c1c", background: "#fee2e2" }}>
            <strong>Compile error:</strong>
            <pre style={{ background: "transparent", border: "none", padding: "4px 0", margin: 0 }}>
              {compileError}
            </pre>
          </div>
        )}
      </div>

      <div className="stan-editor-row">
        <div className="code-section stan-editor-col">
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

        <div className="code-section graph-col">
          <h3>Graphical model</h3>
          <div className="model-diagram-card">
            <GraphicalModel stanCode={debouncedStan} />
          </div>
        </div>
      </div>

      <div className={`sandbox-panel${panelOpen ? "" : " collapsed"}`}>
        <div className="sandbox-panel-tabs">
          <button
            className={`sandbox-panel-tab${panelTab === "data" ? " active" : ""}`}
            onClick={() => {
              setPanelTab("data");
              setPanelOpen(true);
            }}
          >
            Data{customData && " (custom CSV)"}
          </button>
          <button
            className={`sandbox-panel-tab${panelTab === "results" ? " active" : ""}`}
            onClick={() => {
              setPanelTab("results");
              setPanelOpen(true);
            }}
          >
            Posterior summary{summary && ` (${summary.length})`}
          </button>
          <button
            className="secondary sandbox-panel-toggle"
            onClick={() => setPanelOpen((o) => !o)}
          >
            {panelOpen ? "▾ Hide" : "▴ Show"}
          </button>
        </div>

        {panelOpen && (
          <div className="sandbox-panel-body">
            {panelTab === "data" && <DataTable data={effectiveData} />}
            {panelTab === "results" &&
              (error ? (
                <div className="note" style={{ borderLeftColor: "#b91c1c", background: "#fee2e2" }}>
                  <strong>Error:</strong> {error}
                </div>
              ) : summary ? (
                <div className="results">
                  {elapsedMs && (
                    <p style={{ fontSize: 13, color: "#666", margin: "0 0 8px" }}>
                      sampled in {elapsedMs.toFixed(0)}ms
                    </p>
                  )}
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
              ) : (
                <p className="hint">Compile and Run NUTS to see posterior summaries here.</p>
              ))}
          </div>
        )}
      </div>

      <div className="note sandbox-footnote">
        All sampling runs in WebAssembly inside your browser. No data leaves your device.
        For full Stan language coverage and a polished UI, see{" "}
        <a href="https://stan-playground.flatironinstitute.org" target="_blank" rel="noreferrer">
          Stan Playground
        </a>
        .
      </div>
    </div>
  );
}
