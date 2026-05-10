//! AOT compilation: trace a Stan model on the autodiff tape, then emit a
//! self-contained wasm module that computes log_prob and gradients in one call.
//!
//! Replaces `compiler/stan/codegen.mbt` (which emitted WAT text). This crate
//! emits wasm binary directly via `wasm-encoder`, removing the browser-side
//! `wabt` dependency.
//!
//! Generated module ABI (matches MoonBit):
//!   memory          (export)             — linear memory for params + grads
//!   params_ptr()    -> i32   = 0         — base address of param input
//!   grad_ptr()      -> i32   = n_params*8 — base address of gradient output
//!   log_prob_grad() -> f64               — runs forward + backward, returns lp
//!
//! Required imports (host-provided):
//!   ("Math","exp"|"log"|"sin"|"cos"|"pow"|"lgamma"|"digamma"|"phi")
//!   Only the imports actually needed by the recorded tape are emitted.

#![forbid(unsafe_code)]

use stan_autodiff::{Op, Tape};
use stan_runtime::Model;
use thiserror::Error;
use wasm_encoder::{
    CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection,
    ImportSection, Instruction, MemorySection, MemoryType, Module, TypeSection, ValType,
};

#[derive(Debug, Error)]
pub enum CodegenError {
    #[error("trace produced empty tape — no log_prob computation recorded")]
    EmptyTape,
    #[error("internal: {0}")]
    Internal(String),
}

#[derive(Debug, Clone)]
pub struct Compiled {
    pub wasm: Vec<u8>,
    pub n_params: usize,
}

/// Trace `model` on a fresh tape using `dummy_params` (all 0.1 is a reasonable
/// default — eight_schools needs non-zero seeds), then emit a model-specific
/// wasm module.
pub fn compile(model: &Model, dummy_params: &[f64]) -> Result<Compiled, CodegenError> {
    if dummy_params.len() != model.n_params() {
        return Err(CodegenError::Internal(format!(
            "dummy_params len {} != model n_params {}",
            dummy_params.len(),
            model.n_params()
        )));
    }
    let mut tape = Tape::new();
    let leaves: Vec<u32> = dummy_params.iter().map(|p| tape.new_var(*p)).collect();
    let root = model.trace_forward(&mut tape, &leaves);
    if tape.len() == 0 {
        return Err(CodegenError::EmptyTape);
    }
    let wasm = emit(&tape, dummy_params.len(), root);
    Ok(Compiled { wasm, n_params: dummy_params.len() })
}

/// Lower the recorded tape to a wasm module.
fn emit(tape: &Tape, n_params: usize, root: u32) -> Vec<u8> {
    let n = tape.len() as u32;
    let needs = scan_imports(tape);

    // ---- type section ------------------------------------------------------
    // type 0: () -> i32           (params_ptr, grad_ptr)
    // type 1: () -> f64           (log_prob_grad)
    // type 2: (f64) -> f64        (unary math: exp/log/sin/cos/lgamma/digamma/phi)
    // type 3: (f64,f64) -> f64    (pow)
    let mut types = TypeSection::new();
    types.ty().function([], [ValType::I32]);
    types.ty().function([], [ValType::F64]);
    types.ty().function([ValType::F64], [ValType::F64]);
    types.ty().function([ValType::F64, ValType::F64], [ValType::F64]);

    // ---- import section ----------------------------------------------------
    let mut imports = ImportSection::new();
    let mut math_idx = MathImportIndex::default();

    if needs.exp {
        math_idx.exp = Some(import_unary(&mut imports, "exp"));
    }
    if needs.log {
        math_idx.log = Some(import_unary(&mut imports, "log"));
    }
    if needs.sin {
        math_idx.sin = Some(import_unary(&mut imports, "sin"));
    }
    if needs.cos {
        math_idx.cos = Some(import_unary(&mut imports, "cos"));
    }
    if needs.lgamma {
        math_idx.lgamma = Some(import_unary(&mut imports, "lgamma"));
    }
    if needs.digamma {
        math_idx.digamma = Some(import_unary(&mut imports, "digamma"));
    }
    if needs.phi {
        math_idx.phi = Some(import_unary(&mut imports, "phi"));
    }
    if needs.pow {
        math_idx.pow = Some(import_binary(&mut imports));
    }

    let n_imports = imports.len();

    // ---- function section --------------------------------------------------
    let mut functions = FunctionSection::new();
    functions.function(0); // params_ptr
    functions.function(0); // grad_ptr
    functions.function(1); // log_prob_grad

    let params_ptr_idx = n_imports;
    let grad_ptr_idx = n_imports + 1;
    let log_prob_grad_idx = n_imports + 2;

    // ---- memory section ----------------------------------------------------
    let mut memories = MemorySection::new();
    memories.memory(MemoryType {
        minimum: 1,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });

    // ---- export section ----------------------------------------------------
    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, 0);
    exports.export("params_ptr", ExportKind::Func, params_ptr_idx);
    exports.export("grad_ptr", ExportKind::Func, grad_ptr_idx);
    exports.export("log_prob_grad", ExportKind::Func, log_prob_grad_idx);

    // ---- code section ------------------------------------------------------
    let mut codes = CodeSection::new();

    // params_ptr() = 0
    let mut params_ptr_fn = Function::new([]);
    params_ptr_fn.instruction(&Instruction::I32Const(0));
    params_ptr_fn.instruction(&Instruction::End);
    codes.function(&params_ptr_fn);

    // grad_ptr() = n_params * 8
    let grad_base = (n_params as i32) * 8;
    let mut grad_ptr_fn = Function::new([]);
    grad_ptr_fn.instruction(&Instruction::I32Const(grad_base));
    grad_ptr_fn.instruction(&Instruction::End);
    codes.function(&grad_ptr_fn);

    // log_prob_grad(): the main pass
    let lpg = build_log_prob_grad(tape, n_params, root, n, &math_idx, grad_base);
    codes.function(&lpg);

    // ---- assemble ----------------------------------------------------------
    let mut module = Module::new();
    module.section(&types);
    module.section(&imports);
    module.section(&functions);
    module.section(&memories);
    module.section(&exports);
    module.section(&codes);
    module.finish()
}

#[derive(Default, Debug)]
struct ImportNeeds {
    exp: bool,
    log: bool,
    sin: bool,
    cos: bool,
    pow: bool,
    lgamma: bool,
    digamma: bool,
    phi: bool,
}

#[derive(Default)]
struct MathImportIndex {
    exp: Option<u32>,
    log: Option<u32>,
    sin: Option<u32>,
    cos: Option<u32>,
    pow: Option<u32>,
    lgamma: Option<u32>,
    digamma: Option<u32>,
    phi: Option<u32>,
}

fn scan_imports(tape: &Tape) -> ImportNeeds {
    let mut needs = ImportNeeds::default();
    for k in 0..tape.len() {
        match tape.op_at(k as u32) {
            Op::Exp => needs.exp = true,
            Op::Log => needs.log = true,
            Op::Sin => {
                needs.sin = true;
                needs.cos = true; // backward of sin uses cos
            }
            Op::Cos => {
                needs.cos = true;
                needs.sin = true; // backward of cos uses sin
            }
            Op::Pow => needs.pow = true,
            Op::Lgamma => {
                needs.lgamma = true;
                needs.digamma = true; // backward
            }
            Op::Phi => {
                needs.phi = true;
                needs.exp = true; // backward uses exp
            }
            // Tan/Asin/Acos/Atan/Erf/Erfc/Digamma/Sqrt/Abs and arithmetic ops
            // are handled inline (sqrt/abs as f64 instructions; others not yet
            // emitted because the runtime currently does not produce them).
            _ => {}
        }
    }
    needs
}

fn import_unary(imports: &mut ImportSection, name: &str) -> u32 {
    let idx = imports.len();
    imports.import("Math", name, EntityType::Function(2));
    idx
}

fn import_binary(imports: &mut ImportSection) -> u32 {
    let idx = imports.len();
    imports.import("Math", "pow", EntityType::Function(3));
    idx
}

fn build_log_prob_grad(
    tape: &Tape,
    n_params: usize,
    root: u32,
    n: u32,
    m: &MathImportIndex,
    grad_base: i32,
) -> Function {
    // Locals: 2*n f64 (primals 0..n-1, adjoints n..2n-1)
    let total_locals = (2 * n) as u32;
    let mut f = Function::new([(total_locals, ValType::F64)]);

    // ---- forward pass ------------------------------------------------------
    for k in 0..n {
        emit_forward(&mut f, tape, k, n_params as u32, m);
        f.instruction(&Instruction::LocalSet(k));
    }

    // ---- initialize root adjoint = 1.0 ------------------------------------
    f.instruction(&Instruction::F64Const(1.0));
    f.instruction(&Instruction::LocalSet(root + n));

    // ---- backward pass (reverse order) ------------------------------------
    for k_rev in (0..n).rev() {
        emit_backward(&mut f, tape, k_rev, n, m);
    }

    // ---- store gradients to memory ----------------------------------------
    for pi in 0..(n_params as u32) {
        let addr = grad_base + (pi as i32) * 8;
        f.instruction(&Instruction::I32Const(addr));
        f.instruction(&Instruction::LocalGet(pi + n));
        f.instruction(&Instruction::F64Store(wasm_encoder::MemArg {
            offset: 0,
            align: 3,
            memory_index: 0,
        }));
    }

    // ---- return log_prob ---------------------------------------------------
    f.instruction(&Instruction::LocalGet(root));
    f.instruction(&Instruction::End);
    f
}

fn emit_forward(
    f: &mut Function,
    tape: &Tape,
    k: u32,
    n_params: u32,
    m: &MathImportIndex,
) {
    let op = tape.op_at(k);
    let a1 = tape.arg1_at(k);
    let a2i = tape.arg2i_at(k);
    let a2f = tape.arg2f_at(k);
    match op {
        Op::Leaf => {
            if k < n_params {
                // memory[k*8] (i.e. params_ptr + k*8)
                f.instruction(&Instruction::I32Const((k * 8) as i32));
                f.instruction(&Instruction::F64Load(wasm_encoder::MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: 0,
                }));
            } else {
                f.instruction(&Instruction::F64Const(tape.value(k)));
            }
        }
        Op::Add => {
            f.instruction(&Instruction::LocalGet(a1));
            f.instruction(&Instruction::LocalGet(a2i));
            f.instruction(&Instruction::F64Add);
        }
        Op::Sub => {
            f.instruction(&Instruction::LocalGet(a1));
            f.instruction(&Instruction::LocalGet(a2i));
            f.instruction(&Instruction::F64Sub);
        }
        Op::Mul => {
            f.instruction(&Instruction::LocalGet(a1));
            f.instruction(&Instruction::LocalGet(a2i));
            f.instruction(&Instruction::F64Mul);
        }
        Op::Div => {
            f.instruction(&Instruction::LocalGet(a1));
            f.instruction(&Instruction::LocalGet(a2i));
            f.instruction(&Instruction::F64Div);
        }
        Op::Neg => {
            f.instruction(&Instruction::LocalGet(a1));
            f.instruction(&Instruction::F64Neg);
        }
        Op::Exp => {
            f.instruction(&Instruction::LocalGet(a1));
            f.instruction(&Instruction::Call(m.exp.expect("exp import missing")));
        }
        Op::Log => {
            f.instruction(&Instruction::LocalGet(a1));
            f.instruction(&Instruction::Call(m.log.expect("log import missing")));
        }
        Op::Sin => {
            f.instruction(&Instruction::LocalGet(a1));
            f.instruction(&Instruction::Call(m.sin.expect("sin import missing")));
        }
        Op::Cos => {
            f.instruction(&Instruction::LocalGet(a1));
            f.instruction(&Instruction::Call(m.cos.expect("cos import missing")));
        }
        Op::Sqrt => {
            f.instruction(&Instruction::LocalGet(a1));
            f.instruction(&Instruction::F64Sqrt);
        }
        Op::Pow => {
            f.instruction(&Instruction::LocalGet(a1));
            f.instruction(&Instruction::F64Const(a2f));
            f.instruction(&Instruction::Call(m.pow.expect("pow import missing")));
        }
        Op::Abs => {
            f.instruction(&Instruction::LocalGet(a1));
            f.instruction(&Instruction::F64Abs);
        }
        Op::Lgamma => {
            f.instruction(&Instruction::LocalGet(a1));
            f.instruction(&Instruction::Call(m.lgamma.expect("lgamma import missing")));
        }
        Op::AddC => {
            f.instruction(&Instruction::LocalGet(a1));
            f.instruction(&Instruction::F64Const(a2f));
            f.instruction(&Instruction::F64Add);
        }
        Op::SubC => {
            f.instruction(&Instruction::LocalGet(a1));
            f.instruction(&Instruction::F64Const(a2f));
            f.instruction(&Instruction::F64Sub);
        }
        Op::RsubC => {
            f.instruction(&Instruction::F64Const(a2f));
            f.instruction(&Instruction::LocalGet(a1));
            f.instruction(&Instruction::F64Sub);
        }
        Op::MulC => {
            f.instruction(&Instruction::LocalGet(a1));
            f.instruction(&Instruction::F64Const(a2f));
            f.instruction(&Instruction::F64Mul);
        }
        Op::DivC => {
            f.instruction(&Instruction::LocalGet(a1));
            f.instruction(&Instruction::F64Const(a2f));
            f.instruction(&Instruction::F64Div);
        }
        Op::RdivC => {
            f.instruction(&Instruction::F64Const(a2f));
            f.instruction(&Instruction::LocalGet(a1));
            f.instruction(&Instruction::F64Div);
        }
        Op::Phi => {
            f.instruction(&Instruction::LocalGet(a1));
            f.instruction(&Instruction::Call(m.phi.expect("phi import missing")));
        }
        // Erf/Erfc/Tan/Asin/Acos/Atan/Digamma not currently emitted by the
        // runtime, so skip for now. Encountering one would push nothing onto
        // the stack and produce malformed wasm, so guard with unimplemented.
        Op::Erf | Op::Erfc | Op::Tan | Op::Asin | Op::Acos | Op::Atan | Op::Digamma => {
            unimplemented!("codegen for op {op:?}");
        }
    }
}

fn emit_backward(f: &mut Function, tape: &Tape, k: u32, n: u32, m: &MathImportIndex) {
    let op = tape.op_at(k);
    let a1 = tape.arg1_at(k);
    let a2i = tape.arg2i_at(k);
    let a2f = tape.arg2f_at(k);
    let dk = k + n; // adjoint local index for node k

    match op {
        Op::Leaf => {} // adjoint already accumulated by callers
        Op::Add => {
            adj_incr(f, a1 + n, dk);
            adj_incr(f, a2i + n, dk);
        }
        Op::Sub => {
            adj_incr(f, a1 + n, dk);
            adj_decr(f, a2i + n, dk);
        }
        Op::Mul => {
            adj_incr_mul(f, a1 + n, dk, a2i);
            adj_incr_mul(f, a2i + n, dk, a1);
        }
        Op::Div => {
            adj_incr_div(f, a1 + n, dk, a2i);
            adj_decr_mul_div2(f, a2i + n, dk, a1, a2i);
        }
        Op::Neg => {
            adj_decr(f, a1 + n, dk);
        }
        Op::Exp => {
            // d/dx exp(x) = exp(x) = primal[k]
            adj_incr_mul(f, a1 + n, dk, k);
        }
        Op::Log => {
            adj_incr_div(f, a1 + n, dk, a1);
        }
        Op::Sin => {
            adj_incr_fn1(f, a1 + n, dk, a1, m.cos.unwrap());
        }
        Op::Cos => {
            adj_decr_fn1(f, a1 + n, dk, a1, m.sin.unwrap());
        }
        Op::Sqrt => {
            // d/dx sqrt(x) = 1 / (2 * sqrt(x)) = 1 / (2 * primal[k])
            adj_incr_div2(f, a1 + n, dk, k);
        }
        Op::Pow => {
            adj_incr_pow(f, a1 + n, dk, a1, a2f, m.pow.unwrap());
        }
        Op::Abs => {
            adj_incr_sign(f, a1 + n, dk, a1);
        }
        Op::Lgamma => {
            adj_incr_fn1(f, a1 + n, dk, a1, m.digamma.unwrap());
        }
        Op::AddC | Op::SubC => {
            adj_incr(f, a1 + n, dk);
        }
        Op::RsubC => {
            adj_decr(f, a1 + n, dk);
        }
        Op::MulC => {
            adj_incr_mulc(f, a1 + n, dk, a2f);
        }
        Op::DivC => {
            adj_incr_divc(f, a1 + n, dk, a2f);
        }
        Op::RdivC => {
            adj_decr_rdivc(f, a1 + n, dk, a2f, a1);
        }
        Op::Phi => {
            adj_incr_phi(f, a1 + n, dk, a1, m.exp.unwrap());
        }
        Op::Erf | Op::Erfc | Op::Tan | Op::Asin | Op::Acos | Op::Atan | Op::Digamma => {
            unimplemented!("backward for op {op:?}");
        }
    }
}

// ---- adjoint update emitters ----
// Each takes the local indices for adjoint `da` and source adjoint `dk`, plus
// (optionally) primal locals `tv`, `ta`, `tb`, and writes the wasm sequence
// that performs `da += <expression involving dk and primals>`.

// d[da] += d[dk]
fn adj_incr(f: &mut Function, da: u32, dk: u32) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] -= d[dk]
fn adj_decr(f: &mut Function, da: u32, dk: u32) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] += d[dk] * t[tv]
fn adj_incr_mul(f: &mut Function, da: u32, dk: u32, tv: u32) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::LocalGet(tv));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] += d[dk] / t[tv]
fn adj_incr_div(f: &mut Function, da: u32, dk: u32, tv: u32) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::LocalGet(tv));
    f.instruction(&Instruction::F64Div);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] -= d[dk] * t[ta] / (t[tb] * t[tb])
fn adj_decr_mul_div2(f: &mut Function, da: u32, dk: u32, ta: u32, tb: u32) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::LocalGet(ta));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::LocalGet(tb));
    f.instruction(&Instruction::LocalGet(tb));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Div);
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] += d[dk] / (2 * t[tk])
fn adj_incr_div2(f: &mut Function, da: u32, dk: u32, tk: u32) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::F64Const(2.0));
    f.instruction(&Instruction::LocalGet(tk));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Div);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] += d[dk] * fn(t[tv])
fn adj_incr_fn1(f: &mut Function, da: u32, dk: u32, tv: u32, fn_idx: u32) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::LocalGet(tv));
    f.instruction(&Instruction::Call(fn_idx));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] -= d[dk] * fn(t[tv])
fn adj_decr_fn1(f: &mut Function, da: u32, dk: u32, tv: u32, fn_idx: u32) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::LocalGet(tv));
    f.instruction(&Instruction::Call(fn_idx));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] += d[dk] * c
fn adj_incr_mulc(f: &mut Function, da: u32, dk: u32, c: f64) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::F64Const(c));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] += d[dk] / c
fn adj_incr_divc(f: &mut Function, da: u32, dk: u32, c: f64) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::F64Const(c));
    f.instruction(&Instruction::F64Div);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] -= d[dk] * c / (t[ta] * t[ta])
fn adj_decr_rdivc(f: &mut Function, da: u32, dk: u32, c: f64, ta: u32) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::F64Const(c));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::LocalGet(ta));
    f.instruction(&Instruction::LocalGet(ta));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Div);
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] += d[dk] * exp * pow(t[tv], exp - 1)
fn adj_incr_pow(f: &mut Function, da: u32, dk: u32, tv: u32, exponent: f64, pow_idx: u32) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::F64Const(exponent));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::LocalGet(tv));
    f.instruction(&Instruction::F64Const(exponent - 1.0));
    f.instruction(&Instruction::Call(pow_idx));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] += d[dk] * copysign(1.0, t[ta])  (ABS backward)
fn adj_incr_sign(f: &mut Function, da: u32, dk: u32, ta: u32) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::F64Const(1.0));
    f.instruction(&Instruction::LocalGet(ta));
    f.instruction(&Instruction::F64Copysign);
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] += d[dk] * (1/sqrt(2π)) * exp(-0.5 * t[ta]²)  (Phi backward)
fn adj_incr_phi(f: &mut Function, da: u32, dk: u32, ta: u32, exp_idx: u32) {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::F64Const(-0.5));
    f.instruction(&Instruction::LocalGet(ta));
    f.instruction(&Instruction::LocalGet(ta));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::Call(exp_idx));
    f.instruction(&Instruction::F64Const(INV_SQRT_2PI));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalSet(da));
}

