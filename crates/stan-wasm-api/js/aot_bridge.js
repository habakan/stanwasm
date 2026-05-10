// Bridge between stan-wasm-rs.wasm and a per-model AOT-compiled wasm.
//
// The AOT module imports memory from stan-wasm-rs (zero-copy) and exports
// `log_prob_grad(params_ptr, grads_ptr, n_params)`. This snippet stores the
// active AOT exports in a module-local variable and forwards calls.
//
// Usage from app code:
//   import init, { StanModel, setAotExports } from "stan-wasm-rs";
//   await init();
//   const model = new StanModel(src, data);
//   const wasmBytes = model.compileToWasm();
//   const stanMemory = /* obtain from init output */;
//   const aot = await WebAssembly.instantiate(wasmBytes, {
//     stan: { memory: stanMemory },
//     Math: { exp: Math.exp, log: Math.log, /* ... + lgamma/digamma/phi shims */ },
//   });
//   setAotExports(aot.instance.exports);
//   const samples = model.sampleViaAot(init, warmup, draws, seed);

let aotLogProbGrad = null;

export function set_aot_exports(exports) {
  aotLogProbGrad = exports.log_prob_grad;
}

export function clear_aot_exports() {
  aotLogProbGrad = null;
}

export function aot_logp(paramsPtr, gradsPtr, nParams) {
  if (!aotLogProbGrad) {
    throw new Error("AOT not bound — call setAotExports() before sampleViaAot()");
  }
  return aotLogProbGrad(paramsPtr, gradsPtr, nParams);
}
