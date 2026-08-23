import { useEffect, useState } from "react";
import init from "stan-wasm-rs";
import { WasmSandbox } from "./tabs/WasmSandbox";
import { LiveRegression } from "./tabs/LiveRegression";
import { HierarchicalShrinkage } from "./tabs/HierarchicalShrinkage";
import { McmcVisualizer } from "./tabs/McmcVisualizer";
import { MathJaxProvider } from "./graphicalModel";

const TABS = [
  {
    key: "mcmc-visualizer",
    label: "MCMC Visualizer",
    description:
      "Watch NUTS and Random-Walk Metropolis sample the same hard posterior (Neal's funnel), step by step, live — not a replay.",
    Component: McmcVisualizer,
  },
  {
    key: "live-regression",
    label: "Live Regression",
    description:
      "Drag a point; a robust (Student-t) and a normal (conjugate) fit refit live, every frame, and diverge on the outlier.",
    Component: LiveRegression,
  },
  {
    key: "hierarchical-shrinkage",
    label: "Hierarchical Shrinkage",
    description:
      "Six A/B tests' CTR lift — drag a small-sample pilot's observed value and watch how little it moves the partially-pooled estimate next to it.",
    Component: HierarchicalShrinkage,
  },
  {
    key: "wasm-sandbox",
    label: "Wasm Sandbox",
    description: "A fuller API tour: CSV upload, editable Stan source, multiple presets, posterior summary table.",
    Component: WasmSandbox,
  },
] as const;

type TabKey = (typeof TABS)[number]["key"];

export function App() {
  const [loaded, setLoaded] = useState(false);
  const [tab, setTab] = useState<TabKey>(TABS[0].key);

  useEffect(() => {
    // wasm is copied into public/ by the `copy-wasm` npm script and served
    // at BASE_URL + filename. We pass it explicitly because the default
    // wasm-bindgen resolution uses the JS file's location, which lives in
    // ../../ts/pkg/ and is outside Vite's served scope by default.
    // Loaded once here, shared by every tab, so switching tabs never
    // re-instantiates the wasm module.
    const wasmUrl = `${import.meta.env.BASE_URL}stan_wasm_api_bg.wasm`;
    init({ module_or_path: wasmUrl }).then(() => setLoaded(true));
  }, []);

  const active = TABS.find((t) => t.key === tab)!;

  return (
    <MathJaxProvider>
    <div className="app">
      <h1>stan-wasm-rs — examples</h1>
      <p className="tagline">
        Stan probabilistic models sampling entirely in your browser. No server, no network round trip.{" "}
        <a href="https://github.com/habakan/stan-wasm-rs" target="_blank" rel="noreferrer">
          GitHub
        </a>
        .
      </p>

      <div className="tabs">
        {TABS.map((t) => (
          <button
            key={t.key}
            className={`tab-btn${t.key === tab ? " active" : ""}`}
            onClick={() => setTab(t.key)}
          >
            {t.label}
          </button>
        ))}
      </div>
      <p className="tab-desc">{active.description}</p>

      {!loaded && <p>Loading WebAssembly bundle…</p>}
      {loaded && <active.Component />}

      <footer>
        stan-wasm-rs · alpha · Apache-2.0 · embedded{" "}
        <a href="https://github.com/pymc-devs/nuts-rs" target="_blank" rel="noreferrer">
          nuts-rs
        </a>{" "}
        sampler
      </footer>
    </div>
    </MathJaxProvider>
  );
}
