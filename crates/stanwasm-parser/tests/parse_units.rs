//! Targeted unit-level tests pinning down specific parsing behavior that the
//! whole-model integration test would not detect at fine granularity.

use stanwasm_ast::{Constraint, Expr, StanType, Stmt};
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
fn range_index_lowers_to_segment_call() {
    let src = r#"
data { vector[10] y; }
parameters { real mu; }
model {
  target += sum(y[2:5]);
}
"#;
    let prog = parse(src).unwrap();
    // model[0] is target += sum(y[2:5]); the y[2:5] should become segment(y, 2, 5-2+1)
    let s = &prog.model[0];
    match s {
        Stmt::TargetIncr(Expr::Call(name, args)) if name == "sum" => match &args[0] {
            Expr::Call(seg, seg_args) if seg == "segment" => {
                assert_eq!(seg_args.len(), 3);
                assert_eq!(&seg_args[1], &Expr::IntNum(2));
            }
            other => panic!("expected segment call, got {other:?}"),
        },
        other => panic!("expected target += sum(...), got {other:?}"),
    }
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
    // The byte-to-`char` cast this used to do rendered a UTF-8 continuation
    // byte as a Latin-1 glyph, so the message named a character that isn't in
    // the source.
    let err = parse("parameters { real α; }").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains('α'), "{msg}");
}
