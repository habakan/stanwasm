# Architecture

A contributor-facing tour of `stanwasm` internals: workspace layout, data flow, key design choices, and the boundary between native and wasm builds. Pairs with [`README.md`](README.md) (user-facing).

## Goals

- Parse and sample a useful subset of Stan models **entirely inside the browser**, with no compile server, no JS NUTS, no separate sampler binary.
- Keep total wasm payload small after `wasm-opt -Oz` (currently ~482 KB, ~180 KB gzipped, including `console_error_panic_hook` for browser diagnostics).
- Allow the same Rust code to be exercised natively (`cargo test`) and from the browser (`wasm-pack build`) with a single source tree.
- Sample with `nuts-rs` (the same NUTS implementation PyMC uses) so the sampler quality matches established Bayesian tooling.

## Component diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  Application code (browser / Node.js)                            │
│    import init, { StanModel } from "stanwasm"                    │
└──────────┬──────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  stanwasm.wasm  (single bundle, ~482 KB after wasm-opt)          │
│                                                                  │
│   parser ──► AST ──► trace forward pass ──► autodiff tape        │
│                              │                                   │
│                              ├──► tape replay (default sample)   │
│                              │      └─► nuts-rs (embedded crate) │
│                              │                                   │
│                              └──► wasm-encoder ──► AOT model     │
│                                                       wasm bytes │
└────────────────────────────────────────────────────────┬─────────┘
                                                         │
                                                         ▼
                                   ┌──────────────────────────────┐
                                   │  Per-model AOT wasm           │
                                   │  imports ("stan" "memory")    │
                                   │  exports log_prob_grad(p,g,n) │
                                   │                               │
                                   │  V8 JITs the unrolled         │
                                   │  forward + backward pass      │
                                   └──────────────────────────────┘
                                              ▲
                                              │  shared linear memory
                                              │  (zero-copy bridge)
                                              ▼
                                   ┌──────────────────────────────┐
                                   │  JS adapter: 5 lines          │
                                   │  setAotExports() binds the    │
                                   │  AOT wasm's log_prob_grad to  │
                                   │  the host's aot_logp import   │
                                   └──────────────────────────────┘
```

The "tape replay" and "AOT" paths are both available via `StanModel::sample` and `StanModel::sampleViaAot`. Replay is self-contained inside the single wasm; AOT delivers V8-JIT'd per-model code via the shared-memory bridge.

## Workspace layout

Seven crates, one TS facade.

| Crate | Target | Role |
|---|---|---|
| `stanwasm-ast` | lib (native + wasm) | AST type definitions shared by parser, runtime, codegen. Optional `serde` for golden-value tests. |
| `stanwasm-parser` | lib | Hand-written recursive-descent parser. Token enum + lexer + Pratt precedence climbing for expressions. |
| `stanwasm-autodiff` | lib | Reverse-mode autodiff tape (SoA `Vec<f64>` / `Vec<u32>`). Per-op enum, 28 supported ops, CSE caches for log/exp, O(1) reset via generation counter, and `forward_replay` for the sampling hot loop. |
| `stanwasm-runtime` | lib (native by default) | Distributions, constraint transforms, Stan-program evaluator. `Compiled` struct wraps a frozen tape + root index for the replay path. The native-only AST evaluator is the **golden oracle** used in tests; production wasm does not reach it. |
| `stanwasm-codegen` | lib | Emits per-model wasm via `wasm-encoder`. ABI imports memory from the host; exports `log_prob_grad(params_ptr, grads_ptr, n_params)`. No WAT, no `wabt`. |
| `stanwasm` | cdylib (wasm32) | wasm-bindgen public API. Embeds `nuts-rs` (Rust crate) for sampling. Exposes `StanModel` class, `setAotExports` bridge, and the `aot_logp` JS shim. |
| `stanwasm-cli` | bin (native) | Development CLI. `bench all` times AST eval / replay / AOT (via `wasmi`) / end-to-end sampling. |

The `ts/` directory holds the wasm-pack output, hand-written facade, and Node.js integration tests. `examples/gallery/` is a Vite + React demo (tabbed: live regression, hierarchical shrinkage, a fuller API tour) that consumes the local `ts/` package as a `file:` dep.

## Critical paths

### Cold path: building a `StanModel`

```
new StanModel(stanCode, dataJson)
       │
       ▼ stanwasm-parser
   tokenize → AST
       │
       ▼ stanwasm-runtime::data_from_json
   serde_json → Env { name -> Val }
       │
       ▼ stanwasm-runtime::Model::parse_and_load
       │ - resolves data sizes (N, K, …) via eval_plain
       │ - counts unconstrained parameter dimensions
       ▼
   Model { prog, data_env, n_params }
       │
       ▼ stanwasm-runtime::Compiled::from
   tape.new_var × n_params  → leaf tape nodes
   apply constraints + Jacobians, evaluate model block
   forward pass on tape, returns root tape index
       │
       ▼
   Compiled { tape, root, n_params, param_names }
```

The `Compiled` struct is frozen after construction: the tape's op array, arg arrays, and primal/grad arrays no longer grow. Sampling reuses this graph.

### Hot path: `sample` (tape replay)

```
for each leapfrog step in nuts-rs:
    tape.forward_replay(position[..])     // overwrite leaf vals, recompute primals
    tape.reset_grads()
    tape.backward(root)                   // reverse pass over the recorded op array
    copy tape.grad[..n_params] into gradient
    return tape.val[root] as log_prob
```

`forward_replay` walks the op array once with no allocation, no dispatch on `Val`, no AST traversal. The backward pass walks the same array in reverse, accumulating into the persistent `grad` array.

### Hot path: `sampleViaAot` (V8-JIT'd AOT)

The recorded tape is one-shot rewritten to wasm32 by `stanwasm-codegen`. Each tape node becomes a sequence of wasm instructions writing to a function-local `f64`. There are no loops, no conditionals, and no memory access in the body except for parameter loads and gradient stores. V8 JITs this aggressively.

Per-leapfrog step:
- `nuts-rs` calls `aot_logp(params_ptr, grads_ptr, n)` (a wasm-bindgen import)
- The JS shim forwards directly to `aotInstance.exports.log_prob_grad(p, g, n)`
- Because both wasm instances share the same `WebAssembly.Memory`, the pointers are linear-memory offsets that both modules read/write in place — no `memcpy`

## Memory model

Default: each wasm-pack-built bundle has one linear memory exported as `memory`. `stanwasm` exports this via `sharedMemory()` so the host JS can pass it as an import to the AOT model.

```js
const stanMemory = sharedMemory();             // WebAssembly.Memory of stanwasm
const aot = await WebAssembly.instantiate(model.compileToWasm(), {
  stan: { memory: stanMemory },                 // AOT imports stan's memory
  Math: { exp, log, sin, cos, pow, lgamma, digamma, phi },
});
setAotExports(aot.instance.exports);            // bind aot_logp -> aot.log_prob_grad
const samples = model.sampleViaAot(init, 1000, 1000, 42n);
```

Inside the wasm:
- `nuts-rs` allocates its `position` and `gradient` buffers in `stanwasm`'s linear memory (Rust `Vec<f64>`)
- It calls `aot_logp` with the raw pointers
- The AOT wasm reads from and writes to those same byte offsets
- nuts-rs reads the gradients back as if it owned them, because it does

This shared-memory design avoids any per-call JS-side `memcpy` between the two wasm modules.

## Autodiff tape design

`stanwasm-autodiff::Tape` is a struct of arrays:

```rust
val:   Vec<f64>     // primal value at each node
grad:  Vec<f64>     // gradient accumulator
op:    Vec<Op>      // 28-variant enum, repr(i32)
arg1:  Vec<u32>     // first operand: tape index or unused
arg2i: Vec<u32>     // second operand if integer-indexed (binary op partner)
arg2f: Vec<f64>     // second operand if scalar constant (e.g. pow exponent)
```

Each `tape.add(a, b)` etc. pushes one element to each array. The SoA layout means the backward pass reads `op[i]`, `arg1[i]`, `arg2i[i]`, `grad[i]`, `val[a1]` in roughly sequential order — friendly to the cache.

Direct-mapped CSE caches dedupe `log(x)` and `exp(x)` on the same tape index (very common in distribution lpdf computations). `reset_grads` zeroes the `grad` array. `forward_replay` overwrites leaf vals from a parameter slice and recomputes all non-leaf vals.

Initial capacity: 65 536 nodes. Vec grows automatically; we do not pre-reserve from JS hints.

## AOT codegen design (`stanwasm-codegen`)

After `Compiled::from` runs the trace on a `Tape`, `stanwasm-codegen::compile` walks `tape.op_at(k)` once and emits:

```
fn log_prob_grad(params_ptr: i32, grads_ptr: i32, n_params: i32) -> f64
    locals: 2*n f64    (n primal slots, n adjoint slots)

    // Forward pass — for each tape node k in order:
    if Op::Leaf:
        f64.load (params_ptr + k*8)        if k < n_params
        f64.const tape.value(k)            otherwise (constants captured in trace)
    elif Op::Add:
        local.get primal[arg1[k]]
        local.get primal[arg2i[k]]
        f64.add
    elif Op::Exp:
        local.get primal[arg1[k]]
        call $exp                          (imported "Math" "exp")
    …
    local.set primal[k]

    // Initialize root adjoint = 1.0
    f64.const 1.0
    local.set adjoint[root]

    // Backward pass — for each tape node k in reverse order:
    //   each op contributes a fixed wasm instruction sequence that updates
    //   adjoint[arg1[k]] (and arg2i[k] for binary ops) from adjoint[k]
    //   and the recorded primals
    …

    // Store gradients to memory
    for pi in 0..n_params:
        f64.store (grads_ptr + pi*8) <- adjoint[pi]

    // Return root primal
    local.get primal[root]
```

Imports are emitted only for math functions the recorded tape actually used (`scan_imports` walks the tape ahead of section emission). Memory is imported, never defined, so the AOT module can share linear memory with the host.

### Size limit on the AOT path

The emitted function declares one primal and one adjoint local per tape node, so a trace of *n* nodes needs *2n* wasm locals. V8 accepts at most **50,000 locals per function**, which caps the AOT path at a tape of ~25,000 nodes. Because the trace is fully unrolled, tape length grows with the data: a vectorized `y ~ normal(alpha + beta * x, sigma)` uses roughly a dozen nodes per observation, so the ceiling lands somewhere around `N ≈ 2,000` for that model and lower for models that do more work per observation.

`stanwasm-codegen::compile` checks this before emitting and returns `CodegenError::TooManyLocals` rather than producing a module that fails to instantiate in the browser with an opaque `CompileError: local count too large`. The tape-replay path (`StanModel::sample`) has no such limit — it interprets the same tape and is the fallback for large models.

The output validates without the `GC` feature in `wasmparser` — see `crates/stanwasm-codegen/tests/no_wasm_gc.rs`.

## Native vs wasm builds

Same source tree, different feature surfaces:

| Concern | Native (`cargo test`) | wasm32 (`wasm-pack build`) |
|---|---|---|
| AST evaluator (`stanwasm-runtime::eval`) | Compiled in, used as golden oracle | Compiled in and used at runtime: it produces the trace `sample` replays, and evaluates `generated quantities` natively per draw |
| Parser + tape + codegen | Compiled in | Compiled in |
| `nuts-rs` | Used via `Compiled` adapter | Same, via `wasm-bindgen` types |
| AOT execution | Verified by spawning `wasmi` in tests | Done by V8 in the browser |
| `wasmi` / `wasmparser` | Dev-deps for verification | Excluded |
| `wasm-bindgen` glue | Compiled but not active | Drives the public API |

This lets us write tests like "the AOT-emitted wasm produces the same `(logp, grad)` as the AST oracle to 1e-12" without any browser involvement.

## Build pipeline

```bash
# 1. Native: parser + autodiff + runtime + codegen tests, plus the CLI bench
cargo test --workspace
cargo run --release -p stanwasm-cli -- bench all

# 2. wasm32: produce the wasm-pack output consumed by ts/ and examples/
make wasm
#   ├─ cargo build --release --target wasm32-unknown-unknown -p stanwasm
#   ├─ wasm-bindgen processes the cdylib, generates JS glue
#   └─ wasm-opt (wasm-pack's release default) shrinks the bundle to ~482 KB

# 3. TS facade smoke + Node bench in V8 (both rebuild the wasm if it is stale)
make smoke
make bench

# 4. Demo site (Vite + React)
make gallery
```

The CI workflow (`.github/workflows/test.yml`) reproduces step 1 + 2 on Linux and macOS, plus a Node smoke step.

## Validation strategy

Three layers of confidence:

1. **Per-operation unit tests** in `stanwasm-autodiff/tests/gradients.rs`. Each derivative is checked analytically against hand-computed values.

2. **Whole-model finite-difference tests** in `stanwasm-runtime/tests/log_prob.rs`. For each model that exercises a distribution / constraint, the autodiff-produced gradient is compared to a central-difference numerical gradient and required to agree to ~1e-4.

3. **AOT-vs-oracle equivalence** in `stanwasm-codegen/tests/aot_vs_oracle.rs`. The codegen-emitted wasm is instantiated under `wasmi`, fed the same parameters as the AST evaluator, and the resulting `(log_prob, gradient)` pair is required to agree to 1e-12. The `no_wasm_gc.rs` companion pins that the output does not use `WasmFeatures::GC`.

End-to-end sampling is exercised by `stanwasm/tests/sampling.rs`: it runs the full nuts-rs loop on `linear_regression` and asserts the posterior mean of β recovers the true slope to within 0.3 over 400 post-warmup draws.

## Why wasm32, not wasm-gc

- Stable Rust does not target wasm-gc. The wasm-gc Rust backend is experimental and currently produces larger / slower binaries than `wasm32-unknown-unknown` + `dlmalloc`.
- The bottleneck for sampling is the per-leapfrog `log_prob_grad` call, and the AOT model wasm is emitted as plain wasm32 with no `struct.new` / `array.new` — there is nothing to GC.
- Skipping wasm-gc keeps the runtime predictable across all current browser versions and avoids the `wasm-gc` feature negotiation that some embedders still gate.

The `crates/stanwasm-codegen/tests/no_wasm_gc.rs` test enforces this invariant: the validator is run with `WasmFeatures::default() - WasmFeatures::GC` and must accept the artifact.

## References

- [`README.md`](README.md) — user-facing intro and quick start
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — how to propose changes
- [`docs/en/BENCHMARKS.md`](docs/en/BENCHMARKS.md) — performance methodology and current numbers
- [`crates/stanwasm-codegen/tests/no_wasm_gc.rs`](crates/stanwasm-codegen/tests/no_wasm_gc.rs) — wasm-gc absence invariant
