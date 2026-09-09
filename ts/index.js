// Public facade: re-exports the wasm-bindgen bindings so callers never reach
// into `pkg/`. Plain `.js` because a package entry point has to load without
// `--experimental-strip-types`, and there is no TypeScript syntax here to lose.

import init from "./pkg/stanwasm.js";
export {
  AotSampler,
  StanModel,
  compileTape,
  version,
  setAotExports,
  clearAotExports,
  sharedMemory,
} from "./pkg/stanwasm.js";
export default init;
