import { useEffect, useState } from "react";
import init from "stanwasm";
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
  const [initError, setInitError] = useState<string | null>(null);
  const [tab, setTab] = useState<TabKey>(TABS[0].key);

  useEffect(() => {
    // No explicit URL: wasm-bindgen resolves the wasm relative to the glue, which
    // Vite content-hashes. `.catch` is required or a failed init sits on "Loading…".
    init()
      .then(() => setLoaded(true))
      .catch((e: unknown) => setInitError(String((e as Error)?.message ?? e)));
  }, []);

  const active = TABS.find((t) => t.key === tab)!;
  // Wasm Sandbox is a full IDE in its own right; the gallery chrome above it just
  // eats into its one-viewport layout. Collapse to a single "back" affordance.
  const isSandbox = tab === "wasm-sandbox";

  return (
    <MathJaxProvider>
    <div className="app">
      {isSandbox ? (
        <div className="app-header app-header-minimal">
          <button className="secondary" onClick={() => setTab(TABS[0].key)}>
            ← Back to gallery
          </button>
        </div>
      ) : (
        <div className="app-header">
          <h1>stanwasm — examples</h1>
          <p className="tagline">
            Stan probabilistic models sampling entirely in your browser. No server, no network round trip.{" "}
            <a href="https://github.com/habakan/stanwasm" target="_blank" rel="noreferrer">
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
        </div>
      )}

      <div className="tab-body">
        {initError ? (
          <div className="init-error">
            <p>
              <strong>This browser could not load the WebAssembly bundle.</strong>
            </p>
            {/^.*relaxed simd.*$/i.test(initError) ? (
              <p>
                The bundle contains <code>relaxed SIMD</code> instructions, which Safari
                and every browser on iOS still reject at compile time. Chrome, Edge and
                Firefox on the desktop run it. This is a packaging bug on our side, not
                something you can work around —{" "}
                <a
                  href="https://github.com/habakan/stanwasm/issues"
                  target="_blank"
                  rel="noreferrer"
                >
                  tracked here
                </a>
                .
              </p>
            ) : null}
            <pre>{initError}</pre>
          </div>
        ) : !loaded ? (
          <p>Loading WebAssembly bundle…</p>
        ) : (
          <active.Component />
        )}
      </div>

      {!isSandbox && (
        <footer>stanwasm · alpha · Apache-2.0</footer>
      )}
    </div>
    </MathJaxProvider>
  );
}
