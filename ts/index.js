// Public TS facade for stanwasm.
//
// Re-exports the wasm-bindgen-generated bindings under a stable name so
// downstream callers can `import { StanModel } from "stanwasm"` without
// poking into `pkg/`.
//
// Plain `.js` (with a hand-written `index.d.ts` alongside) rather than `.ts`:
// this is the package entry point, so it has to load in a plain-JS project, in
// a bundler, and in Node without `--experimental-strip-types`. There is no
// TypeScript syntax here to lose.

import init from "./pkg/stan_wasm_api.js";
export {
  StanModel,
  version,
  setAotExports,
  clearAotExports,
  sharedMemory,
} from "./pkg/stan_wasm_api.js";
export default init;
