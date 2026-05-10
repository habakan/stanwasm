// Public TS facade for stan-wasm-rs.
//
// Re-exports the wasm-bindgen-generated bindings under a stable name so
// downstream callers can `import { StanModel } from "stan-wasm-rs"` without
// poking into `pkg/`.

import init from "./pkg/stan_wasm_api.js";
export {
  StanModel,
  version,
  setAotExports,
  clearAotExports,
  sharedMemory,
} from "./pkg/stan_wasm_api.js";
export default init;

