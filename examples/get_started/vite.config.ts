import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",
  plugins: [react()],
  optimizeDeps: {
    // Don't pre-bundle the wasm-pack output; let Vite serve the .wasm verbatim.
    exclude: ["stan-wasm-rs"],
  },
  build: {
    target: "esnext",
  },
  // The .wasm file shipped from the stan-wasm-rs package needs the right MIME
  // type. Vite handles `.wasm` by default; nothing extra needed.
});
