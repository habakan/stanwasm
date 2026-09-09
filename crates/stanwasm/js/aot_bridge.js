// Bridge between stanwasm.wasm and a per-model AOT-compiled wasm, which imports
// stanwasm's memory and exports `log_prob_grad`, `tapewasm_layout_id` and
// `tapewasm_abi_version`.
//
// The binding is per page while the scratch buffer belongs to one StanModel, so
// `sampleViaAot` reads the id back to refuse a mismatched pair. See
// `ts/tests/aot_smoke.ts` for how a host binds one.

let aotLogProbGrad = null;
// NaN means nothing is bound, or no id is exported; no u32 id collides with it.
let aotLayoutId = NaN;
// Likewise: a module from before the global existed reads as unknown, not as 0.
let aotAbiVersion = NaN;

export function set_aot_exports(exports) {
  aotLogProbGrad = exports.log_prob_grad;
  const g = exports.tapewasm_layout_id;
  aotLayoutId = g ? g.value >>> 0 : NaN;
  const v = exports.tapewasm_abi_version;
  aotAbiVersion = v ? v.value >>> 0 : NaN;
}

export function clear_aot_exports() {
  aotLogProbGrad = null;
  aotLayoutId = NaN;
  aotAbiVersion = NaN;
}

export function aot_layout_id() {
  return aotLayoutId;
}

export function aot_abi_version() {
  return aotAbiVersion;
}

export function aot_logp(paramsPtr, gradsPtr, nParams, scratchPtr) {
  if (!aotLogProbGrad) {
    throw new Error("AOT not bound — call setAotExports() before sampleViaAot()");
  }
  return aotLogProbGrad(paramsPtr, gradsPtr, nParams, scratchPtr);
}
