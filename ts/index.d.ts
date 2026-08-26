// Types for `index.js`. The shapes themselves come from the wasm-bindgen
// output in `pkg/`, which `make wasm` regenerates.
export {
  StanModel,
  version,
  setAotExports,
  clearAotExports,
  sharedMemory,
} from "./pkg/stan_wasm_api.js";
export { default } from "./pkg/stan_wasm_api.js";
