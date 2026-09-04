//! Variable environment. Most-recent binding wins (for nested scopes).

use std::cell::RefCell;
use std::rc::Rc;

use crate::value::Val;
use rand::rngs::ChaCha8Rng;
use stanwasm_ast::FuncDef;
use stanwasm_autodiff::Tape;

/// One binding, carrying whether the *declared* Stan type is integral. It lives
/// here and not in `Val` because it is static, and only `/` cares.
#[derive(Debug, Clone)]
struct Binding {
    name: String,
    val: Val,
    is_int: bool,
}

#[derive(Debug, Default, Clone)]
pub struct Env {
    vars: Vec<Binding>,
    /// Bindings read through rather than copied. The data block is the largest
    /// value here, and a scope per call or per trace duplicated all of it.
    base: Option<Rc<Env>>,
    /// Shared RNG for `_rng` calls, set only while evaluating `generated
    /// quantities`. `Rc` means a cloned `Env` advances the same stream.
    rng: Option<Rc<RefCell<ChaCha8Rng>>>,
    /// Set while tracing for the one-shot `Compiled` tape, which is replayed for
    /// every draw — a parameter-dependent branch would freeze at its trace-time value.
    strict_no_param_branch: bool,
    /// User-defined functions, shared by `Rc` because `Env` is cloned per scope.
    funcs: Option<Rc<Vec<(String, FuncDef)>>>,
    /// Names currently being inlined. Calls are unrolled into the tape, so a
    /// recursive one would expand forever rather than loop.
    call_stack: Vec<String>,
}

impl Env {
    pub fn new() -> Self {
        Self::default()
    }

    /// A scope that reads through to `base` and writes into its own bindings.
    pub fn nested(base: Rc<Env>) -> Self {
        Self {
            vars: Vec::new(),
            rng: base.rng.clone(),
            strict_no_param_branch: base.strict_no_param_branch,
            funcs: base.funcs.clone(),
            call_stack: base.call_stack.clone(),
            base: Some(base),
        }
    }

    fn binding(&self, name: &str) -> Option<&Binding> {
        self.vars
            .iter()
            .rev()
            .find(|b| b.name == name)
            .or_else(|| self.base.as_ref()?.binding(name))
    }

    /// Rebind `name`, keeping whatever int-ness the existing binding declared
    /// (`int k; k = k + 1;` stays an int). New names default to real.
    pub fn set(&mut self, name: &str, val: Val) {
        for slot in self.vars.iter_mut() {
            if slot.name == name {
                slot.val = val;
                return;
            }
        }
        let is_int = self.is_int(name);
        self.vars.push(Binding {
            name: name.to_string(),
            val,
            is_int,
        });
    }

    /// Like `set`, but declares the binding int-typed (see `Binding::is_int`).
    pub fn set_int_typed(&mut self, name: &str, val: Val) {
        for slot in self.vars.iter_mut() {
            if slot.name == name {
                slot.val = val;
                slot.is_int = true;
                return;
            }
        }
        self.vars.push(Binding {
            name: name.to_string(),
            val,
            is_int: true,
        });
    }

    /// Push a new binding without overwriting earlier ones — caller can
    /// later `pop_to(saved_len)` to restore the scope.
    pub fn push(&mut self, name: &str, val: Val) {
        self.vars.push(Binding {
            name: name.to_string(),
            val,
            is_int: false,
        });
    }

    pub fn get(&self, name: &str) -> Option<&Val> {
        self.binding(name).map(|b| &b.val)
    }

    /// Mutable access to a binding, so an indexed write can edit in place
    /// rather than rebuild the container it lives in.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Val> {
        if !self.vars.iter().any(|b| b.name == name) {
            // Writing through to a shared binding would edit every scope holding
            // it, so the write gets its own copy. Stan data is never assigned.
            let b = self.base.as_ref()?.binding(name)?.clone();
            self.vars.push(b);
        }
        self.vars
            .iter_mut()
            .rev()
            .find(|b| b.name == name)
            .map(|b| &mut b.val)
    }

    /// Mark an existing binding int-typed, leaving its value where it is.
    pub fn mark_int(&mut self, name: &str) {
        if let Some(b) = self.vars.iter_mut().rev().find(|b| b.name == name) {
            b.is_int = true;
        }
    }

    /// Whether `name` is bound in this scope or any it reads through.
    pub fn has(&self, name: &str) -> bool {
        self.binding(name).is_some()
    }

    /// Whether `name` was declared with an integral Stan type.
    pub fn is_int(&self, name: &str) -> bool {
        self.binding(name).is_some_and(|b| b.is_int)
    }

    pub fn len(&self) -> usize {
        self.vars.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vars.is_empty()
    }

    pub fn truncate(&mut self, len: usize) {
        self.vars.truncate(len);
    }

    /// Replace every tape-backed binding with its constant value. `transformed
    /// data` runs on a scratch tape whose indices must not outlive it.
    pub fn freeze(&mut self, tape: &Tape) {
        fn go(tape: &Tape, v: &mut Val) {
            match v {
                Val::Tape(i) => *v = Val::Num(tape.value(*i)),
                Val::Vec(xs) | Val::Row(xs) => xs.iter_mut().for_each(|x| go(tape, x)),
                Val::Num(_) => {}
            }
        }
        for b in self.vars.iter_mut() {
            go(tape, &mut b.val);
        }
    }

    pub fn set_scalar(&mut self, name: &str, v: f64) {
        self.set(name, Val::Num(v));
    }

    pub fn set_vector(&mut self, name: &str, xs: &[f64]) {
        self.set(name, Val::Vec(xs.iter().map(|x| Val::Num(*x)).collect()));
    }

    pub fn set_rng(&mut self, rng: Rc<RefCell<ChaCha8Rng>>) {
        self.rng = Some(rng);
    }

    pub fn rng(&self) -> Option<Rc<RefCell<ChaCha8Rng>>> {
        self.rng.clone()
    }

    pub fn set_strict_no_param_branch(&mut self, v: bool) {
        self.strict_no_param_branch = v;
    }

    pub fn set_funcs(&mut self, funcs: Rc<Vec<(String, FuncDef)>>) {
        self.funcs = Some(funcs);
    }

    pub fn func(&self, name: &str) -> Option<FuncDef> {
        let fs = self.funcs.as_ref()?;
        fs.iter().find(|(n, _)| n == name).map(|(_, f)| f.clone())
    }

    pub fn in_call(&self, name: &str) -> bool {
        self.call_stack.iter().any(|n| n == name)
    }

    pub fn enter_call(&mut self, name: &str) {
        self.call_stack.push(name.to_string());
    }

    pub fn strict_no_param_branch(&self) -> bool {
        self.strict_no_param_branch
    }
}
