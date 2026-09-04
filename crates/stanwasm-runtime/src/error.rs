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
    #[error(
        "`{0}` is not implemented. An adaptive solver takes a number of steps \
         that depends on the parameters, and this runtime records the log \
         density once and replays the same computation graph for every draw, \
         so the step sequence would freeze at whatever the first evaluation \
         chose. Solving the system on a fixed grid outside the model, and \
         passing the result in as data, is the form that works here"
    )]
    UnsupportedOdeIntegrator(String),
    #[error("unknown distribution: {0}")]
    UnknownDistribution(String),
    #[error("{0}_rng called outside generated quantities")]
    RngOutsideGeneratedQuantities(String),
    #[error("unsupported or wrong-arity call: {0}_rng")]
    UnknownRng(String),
    #[error(
        "assignment target must be a name, optionally indexed or sliced \
         (`x`, `x[i]`, `M[1:n, k]`) — this one is neither"
    )]
    UnsupportedAssignmentTarget,
    #[error("while loop exceeded {0} iterations — possible infinite loop")]
    WhileLoopOverflow(u64),
    #[error(
        "the recorded computation graph passed {0} nodes, which is more than a \
         32-bit address space holds. Every scalar operation the model performs \
         is one node, so observations times parameters is the count that grows: \
         fit a subset of the data, or narrow the model"
    )]
    TapeTooLarge(usize),
    #[error("invalid parameters: {0}")]
    InvalidRngParams(String),
    #[error("index {index} out of bounds for a length-{len} array/vector")]
    IndexOutOfBounds { index: i32, len: usize },
    #[error(
        "expected a scalar but got a vector/matrix — this operation is not \
         vectorized. Common causes: comparing containers with `==`, or a \
         function given a container where Stan defines it only on reals; \
         write the loop form instead"
    )]
    NotAScalar,
    #[error(
        "`{func}`'s {arg} depends on a parameter, and its derivative there has \
         no closed form in this runtime. Move it to `data`/`transformed data`"
    )]
    NonDifferentiableArgument { func: String, arg: String },
    #[error("shape mismatch: cannot apply `{op}` to {lhs} and {rhs}")]
    ShapeMismatch {
        op: String,
        lhs: String,
        rhs: String,
    },
    #[error("{name} expects {expected} argument(s) after the variate, got {got}")]
    DistributionArity {
        name: String,
        expected: usize,
        got: usize,
    },
    #[error(
        "{name}: distribution argument has length {arg_len} but the variate \
         has length {var_len} — vectorized arguments must match element-wise"
    )]
    DistributionArgLength {
        name: String,
        arg_len: usize,
        var_len: usize,
    },
    #[error("{name} expects {expected} — got {got}")]
    DistributionArgType {
        name: String,
        expected: String,
        got: String,
    },
    #[error(
        "{name} takes a single vector as its variate, but got {got}. An array \
         of vectors (`array[N] vector[K] y; y ~ {name}(mu, L);`) is not \
         vectorized here — write the loop form: \
         `for (n in 1:N) y[n] ~ {name}(mu, L);`"
    )]
    MultivariateNotVectorized { name: String, got: String },
    #[error("integer division by zero")]
    IntDivisionByZero,
    #[error(
        "parameter `{name}` is declared `{typ}`, which has no constraint \
         transform in this runtime yet — it would otherwise be sampled \
         unconstrained, giving a silently wrong posterior. See the \"Not yet \
         supported\" list in the README"
    )]
    UnsupportedConstraint { name: String, typ: String },
    #[error(
        "parameter `{0}` is declared `int`. Stan parameters must be \
         continuous — NUTS differentiates the log density with respect to \
         them. Move it to `data`/`transformed data`, or marginalize the \
         discrete variable out of the model"
    )]
    IntParameter(String),
    #[error("parameter `{name}`: {detail}")]
    BadParameterDeclaration { name: String, detail: String },
    #[error(
        "if/while condition in `model`/`transformed parameters` depends on a \
         sampled parameter — not supported. NUTS traces this block once and \
         replays the same computation graph for every draw, so which branch \
         is taken can't change per-draw once traced. Restructure the model to \
         avoid parameter-dependent control flow here (`generated quantities` \
         doesn't have this limitation — it re-evaluates natively per draw)."
    )]
    ParamDependentBranch,
    #[error(
        "generated quantity `{name}` is declared to hold {expected} value(s) but its \
         expression produced {got}. A scalar `_rng` given only scalars returns a \
         scalar — pass it a container (`normal_rng(rep_vector(mu, N), sigma)`) or \
         fill the declaration element by element."
    )]
    GenQuantityShape {
        name: String,
        expected: usize,
        got: usize,
    },
    #[error(
        "user-defined function `{0}` calls itself. Calls are inlined into one recorded \
         computation graph, which a recursive one would expand forever — rewrite it as \
         a loop."
    )]
    RecursiveCall(String),
    #[error("user-defined function `{name}` takes {expected} argument(s), got {got}")]
    WrongArity {
        name: String,
        expected: usize,
        got: usize,
    },
}
