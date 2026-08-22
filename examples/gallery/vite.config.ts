import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",
  plugins: [react()],
  server: {
    fs: {
      // The `stan-wasm-rs` package lives at ../../ts via a file: dep, and the
      // wasm-pack output (ts/pkg/) sits outside this Vite project root.
      // Vite blocks serving files outside the root by default — allow the
      // repo root explicitly so wasm fetch works during `npm run dev`.
      allow: ["..", "../.."],
    },
  },
  optimizeDeps: {
    // Don't pre-bundle the wasm-pack output; let Vite serve the .wasm verbatim.
    exclude: ["stan-wasm-rs"],
  },
  build: {
    target: "esnext",
  },
});
