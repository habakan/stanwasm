// Types for `index.js`; the shapes come from `pkg/`, which `make wasm` writes.
export {
  AotSampler,
  StanModel,
  version,
  setAotExports,
  clearAotExports,
  sharedMemory,
} from "./pkg/stanwasm.js";
export { default } from "./pkg/stanwasm.js";
