//! Errors surfaced while evaluating a Stan model: user-reachable mistakes
//! (typos, wrong arity, an RNG call outside `generated quantities`, an
//! assignment form we don't support) that must be reported cleanly instead
//! of silently contributing zero to the log density or panicking.

use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum EvalError {
    #[error("undefined variable: {0}")]
    UndefinedVariable(String),
    #[error("unknown function: {0}")]
    UnknownFunction(String),
    #[error("unknown distribution: {0}")]
    UnknownDistribution(String),
    #[error("{0}_rng called outside generated quantities")]
    RngOutsideGeneratedQuantities(String),
    #[error("unsupported or wrong-arity call: {0}_rng")]
    UnknownRng(String),
    #[error(
        "assignment to indexed/compound targets (e.g. `arr[i] = ...`) is not \
         yet supported — only `name = expr` works"
    )]
    UnsupportedAssignmentTarget,
    #[error("while loop exceeded {0} iterations — possible infinite loop")]
    WhileLoopOverflow(u64),
    #[error("invalid parameters: {0}")]
    InvalidRngParams(String),
    #[error(
        "if/while condition in `model`/`transformed parameters` depends on a \
         sampled parameter — not supported. NUTS traces this block once and \
         replays the same computation graph for every draw, so which branch \
         is taken can't change per-draw once traced. Restructure the model to \
         avoid parameter-dependent control flow here (`generated quantities` \
         doesn't have this limitation — it re-evaluates natively per draw)."
    )]
    ParamDependentBranch,
}
