import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",
  plugins: [react()],
  server: {
    fs: {
      // `stanwasm` is a file: dep at ../../ts, outside this Vite root, and Vite
      // refuses to serve outside the root unless told to.
      allow: ["..", "../.."],
    },
  },
  optimizeDeps: {
    // Don't pre-bundle the wasm-pack output; let Vite serve the .wasm verbatim.
    // posteriorwasm starts its worker from `new URL(..., import.meta.url)`, which pre-bundling breaks.
    exclude: ["stanwasm", "posteriorwasm"],
  },
  // posteriorwasm's worker imports Pyodide at run time, which a classic (iife) worker cannot.
  worker: {
    format: "es",
  },
  build: {
    target: "esnext",
  },
});
