// Public facade: re-exports the wasm-bindgen bindings so callers never reach
// into `pkg/`. Plain `.js` because a package entry point has to load without
// `--experimental-strip-types`, and there is no TypeScript syntax here to lose.
//
// The list is written out rather than a `*` re-export, so it has to be kept in
// step with what the bundle exports — `tests/facade_exports.mjs` is what says
// when it is not.

import init from "./pkg/stanwasm.js";
export {
  AotSampler,
  CompiledTape,
  StanModel,
  compileTape,
  version,
  tapewasmVersion,
  setAotExports,
  clearAotExports,
  sharedMemory,
} from "./pkg/stanwasm.js";
export default init;
