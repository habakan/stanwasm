// Bridge between stanwasm.wasm and a per-model AOT-compiled wasm, which imports
// stanwasm's memory and exports `log_prob_grad` plus `stanwasm_layout_id`.
//
// The binding is per page while the scratch buffer belongs to one StanModel, so
// `sampleViaAot` reads the id back to refuse a mismatched pair. See
// `ts/tests/aot_smoke.ts` for how a host binds one.

let aotLogProbGrad = null;
// NaN means nothing is bound, or no id is exported; no u32 id collides with it.
let aotLayoutId = NaN;

export function set_aot_exports(exports) {
  aotLogProbGrad = exports.log_prob_grad;
  const g = exports.stanwasm_layout_id;
  aotLayoutId = g ? g.value >>> 0 : NaN;
}

export function clear_aot_exports() {
  aotLogProbGrad = null;
  aotLayoutId = NaN;
}

export function aot_layout_id() {
  return aotLayoutId;
}

export function aot_logp(paramsPtr, gradsPtr, nParams, scratchPtr) {
  if (!aotLogProbGrad) {
    throw new Error("AOT not bound — call setAotExports() before sampleViaAot()");
  }
  return aotLogProbGrad(paramsPtr, gradsPtr, nParams, scratchPtr);
}
