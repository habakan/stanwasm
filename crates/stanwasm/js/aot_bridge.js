// Bridge between stanwasm.wasm and a per-model AOT-compiled wasm.
//
// The AOT module imports memory from stanwasm (zero-copy) and exports
// `log_prob_grad(params_ptr, grads_ptr, n_params, scratch_ptr)` plus the global
// `stanwasm_layout_id`. This snippet stores the active AOT exports in a
// module-local variable and forwards calls.
//
// The binding is per page, not per model, while the scratch buffer it works in
// belongs to one StanModel. `sampleViaAot` reads the id back to refuse a pair
// that does not match; without it, a larger model's module would write past a
// smaller model's scratch.
//
// Usage from app code:
//   import init, { StanModel, setAotExports } from "stanwasm";
//   await init();
//   const model = new StanModel(src, data);
//   const wasmBytes = model.compileToWasm();
//   const stanMemory = /* obtain from init output */;
//   const aot = await WebAssembly.instantiate(wasmBytes, {
//     stan: { memory: stanMemory },
//     Math: { exp: Math.exp, log: Math.log, pow: Math.pow, sin, cos, /* + a phi shim */ },
//   });
//   setAotExports(aot.instance.exports);
//   const samples = model.sampleViaAot(init, warmup, draws, seed);

let aotLogProbGrad = null;
// NaN means nothing is bound, or what is bound exports no id. u32 ids are exact
// as doubles, so no real id collides with it.
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
