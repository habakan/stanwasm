//! Targeted unit-level tests pinning down specific parsing behavior that the
//! whole-model integration test would not detect at fine granularity.

use stanwasm_ast::{Constraint, Expr, SliceIdx, StanType, Stmt};
use stanwasm_parser::parse;

#[test]
fn lower_constraint_on_real() {
    let src = r#"
parameters {
  real<lower=0> sigma;
}
"#;
    let prog = parse(src).unwrap();
    let p = &prog.parameters[0];
    assert_eq!(p.name, "sigma");
    match &p.typ {
        StanType::Real(Constraint::Lower(Expr::IntNum(v))) => assert_eq!(*v, 0),
        other => panic!("unexpected type: {other:?}"),
    }
}

#[test]
fn lower_upper_constraint() {
    let src = r#"
parameters {
  real<lower=0, upper=1> p;
}
"#;
    let prog = parse(src).unwrap();
    match &prog.parameters[0].typ {
        StanType::Real(Constraint::LowerUpper(lo, hi)) => {
            assert_eq!(lo, &Expr::IntNum(0));
            assert_eq!(hi, &Expr::IntNum(1));
        }
        other => panic!("unexpected type: {other:?}"),
    }
}

#[test]
fn sample_statement_decomposes_distribution() {
    let src = r#"
data { vector[3] y; }
parameters { real mu; }
model {
  y ~ normal(mu, 1);
}
"#;
    let prog = parse(src).unwrap();
    match &prog.model[0] {
        Stmt::Sample(_lhs, dist, args) => {
            assert_eq!(dist, "normal");
            assert_eq!(args.len(), 2);
        }
        other => panic!("expected Sample, got {other:?}"),
    }
}

#[test]
fn a_range_index_becomes_a_slice() {
    let src = r#"
data { vector[10] y; }
parameters { real mu; }
model {
  target += sum(y[2:5]);
}
"#;
    let prog = parse(src).unwrap();
    let Stmt::TargetIncr(Expr::Call(name, args)) = &prog.model[0] else {
        panic!("expected target += sum(...), got {:?}", prog.model[0]);
    };
    assert_eq!(name, "sum");
    let Expr::Slice(base, idxs) = &args[0] else {
        panic!("expected a slice, got {:?}", args[0]);
    };
    assert_eq!(**base, Expr::Var("y".into()));
    assert_eq!(
        idxs.as_slice(),
        [SliceIdx::Range(
            Some(Expr::IntNum(2)),
            Some(Expr::IntNum(5))
        )]
    );
}

/// A range on one dimension and an index on another — the form that a plain
/// `segment` lowering could not express.
#[test]
fn a_range_may_be_mixed_with_an_index() {
    let prog = parse("model { target += sum(W[1:rows(W), k]); }").unwrap();
    let Stmt::TargetIncr(Expr::Call(_, args)) = &prog.model[0] else {
        panic!("expected a call, got {:?}", prog.model[0]);
    };
    let Expr::Slice(_, idxs) = &args[0] else {
        panic!("expected a slice, got {:?}", args[0]);
    };
    assert!(matches!(idxs[0], SliceIdx::Range(Some(_), Some(_))));
    assert_eq!(idxs[1], SliceIdx::At(Expr::Var("k".into())));
}

/// `x[ : ]` and `x[2 : ]` leave a bound to the container's own end.
#[test]
fn a_range_bound_may_be_omitted() {
    let prog = parse("model { target += sum(M[ : , 2]) + sum(v[3 : ]); }").unwrap();
    let Stmt::TargetIncr(Expr::BinOp(_, lhs, rhs)) = &prog.model[0] else {
        panic!("expected a sum of two, got {:?}", prog.model[0]);
    };
    let (Expr::Call(_, la), Expr::Call(_, ra)) = (lhs.as_ref(), rhs.as_ref()) else {
        panic!("expected two calls");
    };
    let (Expr::Slice(_, li), Expr::Slice(_, ri)) = (&la[0], &ra[0]) else {
        panic!("expected two slices");
    };
    assert_eq!(li[0], SliceIdx::Range(None, None));
    assert_eq!(ri[0], SliceIdx::Range(Some(Expr::IntNum(3)), None));
}

#[test]
fn array_type() {
    let src = r#"
data {
  array[5] int counts;
}
parameters { real mu; }
"#;
    let prog = parse(src).unwrap();
    match &prog.data[0].typ {
        StanType::Array(size, elem) => {
            assert_eq!(size, &Expr::IntNum(5));
            assert!(matches!(elem.as_ref(), StanType::Int(_)));
        }
        other => panic!("unexpected type: {other:?}"),
    }
}

#[test]
fn pratt_precedence_arith() {
    // a + b * c → BinOp("+", a, BinOp("*", b, c))
    let src = r#"
data { real a; real b; real c; }
parameters { real mu; }
model {
  target += a + b * c;
}
"#;
    let prog = parse(src).unwrap();
    match &prog.model[0] {
        Stmt::TargetIncr(Expr::BinOp(op, lhs, rhs)) => {
            assert_eq!(op, "+");
            assert_eq!(lhs.as_ref(), &Expr::Var("a".into()));
            match rhs.as_ref() {
                Expr::BinOp(op2, _, _) => assert_eq!(op2, "*"),
                other => panic!("expected b*c, got {other:?}"),
            }
        }
        other => panic!("unexpected stmt: {other:?}"),
    }
}

#[test]
fn for_loop() {
    let src = r#"
data { int N; vector[10] y; }
parameters { real mu; real<lower=0> sigma; }
model {
  for (i in 1:N) y[i] ~ normal(mu, sigma);
}
"#;
    let prog = parse(src).unwrap();
    match &prog.model[0] {
        Stmt::For(var, _lo, _hi, body) => {
            assert_eq!(var, "i");
            assert_eq!(body.len(), 1);
            assert!(matches!(body[0], Stmt::Sample(..)));
        }
        other => panic!("expected For, got {other:?}"),
    }
}

#[test]
fn unknown_non_ascii_character_is_reported_verbatim() {
    // The old byte-to-`char` cast rendered a UTF-8 continuation byte as Latin-1,
    // so the message named a character that isn't in the source.
    let err = parse("parameters { real α; }").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains('α'), "{msg}");
}

/// `array[a, b] T` is `array[a] array[b] T`, and the rightmost size is the
/// innermost. It is how capture-history and panel data get declared, which is
/// most of what a real model corpus wanted that this could not read.
#[test]
fn an_array_can_be_declared_with_more_than_one_dimension() {
    let prog = parse("data { int M; int T; array[M, T] int<lower=0, upper=1> y; }").unwrap();
    let y = prog
        .data
        .iter()
        .find(|d| d.name == "y")
        .expect("y is declared");
    let StanType::Array(outer, inner) = &y.typ else {
        panic!("expected an array, got {:?}", y.typ);
    };
    assert_eq!(*outer, Expr::Var("M".into()));
    let StanType::Array(mid, elem) = inner.as_ref() else {
        panic!("expected a nested array, got {inner:?}");
    };
    assert_eq!(*mid, Expr::Var("T".into()));
    assert!(matches!(elem.as_ref(), StanType::Int(_)));
}

/// `'` is postfix and binds tighter than `^`, so `x'` is the operand of `*`
/// rather than the whole product being transposed.
#[test]
fn transpose_is_postfix_and_binds_tightest() {
    let prog = parse("model { target += x' * y; }").unwrap();
    let Stmt::TargetIncr(Expr::BinOp(op, lhs, rhs)) = &prog.model[0] else {
        panic!("expected a product, got {:?}", prog.model[0]);
    };
    assert_eq!(op, "*");
    assert_eq!(
        **lhs,
        Expr::UnOp("'".into(), Box::new(Expr::Var("x".into())))
    );
    assert_eq!(**rhs, Expr::Var("y".into()));
}

/// `M[i]'` transposes the indexed element, not `M`.
#[test]
fn transpose_applies_after_indexing() {
    let prog = parse("model { target += sum(M[i]'); }").unwrap();
    let Stmt::TargetIncr(Expr::Call(_, args)) = &prog.model[0] else {
        panic!("expected a call, got {:?}", prog.model[0]);
    };
    let Expr::UnOp(op, inner) = &args[0] else {
        panic!("expected a transpose, got {:?}", args[0]);
    };
    assert_eq!(op, "'");
    assert!(matches!(**inner, Expr::Index(..)));
}

/// `data` on a function argument promises it is not a parameter, which changes
/// nothing about the shape the argument carries.
#[test]
fn a_data_qualified_argument_parses() {
    let src =
        "functions { real f(real a, data array[] real x_r, data array[] int x_i) { return a; } }
               model { target += f(1.0, x_r, x_i); }";
    let prog = parse(src).unwrap();
    let (name, def) = &prog.functions[0];
    assert_eq!(name, "f");
    assert_eq!(def.params.len(), 3);
    assert_eq!(def.params[1].1, "x_r");
    assert_eq!(def.params[2].1, "x_i");
}
