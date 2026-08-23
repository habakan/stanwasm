//! Stan recursive-descent parser.

use crate::lexer::tokenize;
use crate::token::Token;
use stan_ast::{Constraint, Expr, FuncDef, StanProgram, StanType, Stmt, VarDecl};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("expected {expected}, got {got:?}")]
    Expected { expected: String, got: Token },
    #[error("expected identifier, got {got:?}")]
    ExpectedIdent { got: Token },
    #[error("expected type keyword, got {got:?}")]
    ExpectedType { got: Token },
    #[error("expected lower/upper in constraint, got {got:?}")]
    BadConstraint { got: Token },
    #[error("unexpected token in expression: {got:?}")]
    UnexpectedInExpr { got: Token },
    #[error("expected distribution call after ~")]
    BadSample,
    #[error("expected block name, got {got:?}")]
    BadBlockName { got: Token },
    #[error(
        "unknown top-level block `{name}` — expected one of: functions, data, \
         parameters, transformed data, transformed parameters, model, \
         generated quantities"
    )]
    UnknownBlock { name: String },
    #[error(
        "unsupported statement: a bare expression by itself isn't a valid \
         Stan statement here (e.g. `print(...)`/`reject(...)`/void function \
         calls aren't supported yet) — did you mean `~`, `=`, or `+=`?"
    )]
    UnsupportedStatement,
    #[error(
        "unrecognized character `{0}` — note: elementwise operators (`.*`, \
         `./`, `.^`) aren't supported yet"
    )]
    UnknownChar(#[from] crate::lexer::UnknownChar),
}

type Result<T> = std::result::Result<T, ParseError>;

pub struct Parser {
    toks: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(src: &str) -> Result<Self> {
        Ok(Self {
            toks: tokenize(src)?,
            pos: 0,
        })
    }

    fn peek(&self) -> &Token {
        &self.toks[self.pos]
    }

    fn consume(&mut self) -> Token {
        let t = self.toks[self.pos].clone();
        self.pos += 1;
        t
    }

    fn check_tok(&self, t: &Token) -> bool {
        self.peek() == t
    }

    fn check_kw(&self, kw: &str) -> bool {
        matches!(self.peek(), Token::Kw(s) if s == kw)
    }

    fn check_id(&self, id: &str) -> bool {
        matches!(self.peek(), Token::Id(s) if s == id)
    }

    fn try_tok(&mut self, t: &Token) -> bool {
        if self.peek() == t {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn try_kw(&mut self, kw: &str) -> bool {
        if self.check_kw(kw) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect_tok(&mut self, t: &Token) -> Result<()> {
        let got = self.consume();
        if &got == t {
            Ok(())
        } else {
            Err(ParseError::Expected {
                expected: format!("{t:?}"),
                got,
            })
        }
    }

    fn expect_kw(&mut self, kw: &str) -> Result<()> {
        let got = self.consume();
        match &got {
            Token::Kw(s) if s == kw => Ok(()),
            _ => Err(ParseError::Expected {
                expected: format!("Kw({kw})"),
                got,
            }),
        }
    }

    fn expect_id(&mut self) -> Result<String> {
        match self.consume() {
            Token::Id(s) | Token::Kw(s) => Ok(s),
            got => Err(ParseError::ExpectedIdent { got }),
        }
    }

    // ---- Constraints: <lower=0>, <lower=0,upper=1> -----------------------

    fn parse_constraints(&mut self) -> Result<Constraint> {
        if !self.check_tok(&Token::Lt) {
            return Ok(Constraint::None);
        }
        let saved = self.pos;
        self.pos += 1;
        let is_constraint = matches!(
            self.peek(),
            Token::Kw(s) if s == "lower" || s == "upper"
        );
        if !is_constraint {
            self.pos = saved;
            return Ok(Constraint::None);
        }

        let mut lower: Option<Expr> = None;
        let mut upper: Option<Expr> = None;
        loop {
            match self.consume() {
                Token::Kw(ref s) if s == "lower" => {
                    self.expect_tok(&Token::Equals)?;
                    lower = Some(self.parse_expr(5)?);
                }
                Token::Kw(ref s) if s == "upper" => {
                    self.expect_tok(&Token::Equals)?;
                    upper = Some(self.parse_expr(5)?);
                }
                got => return Err(ParseError::BadConstraint { got }),
            }
            if !self.try_tok(&Token::Comma) {
                break;
            }
        }
        self.expect_tok(&Token::Gt)?;

        Ok(match (lower, upper) {
            (Some(lo), None) => Constraint::Lower(lo),
            (None, Some(hi)) => Constraint::Upper(hi),
            (Some(lo), Some(hi)) => Constraint::LowerUpper(lo, hi),
            (None, None) => Constraint::None,
        })
    }

    // ---- Type declaration ------------------------------------------------

    fn parse_type(&mut self) -> Result<StanType> {
        if self.check_kw("array") {
            self.consume();
            self.expect_tok(&Token::LBrack)?;
            let size = self.parse_expr(0)?;
            self.expect_tok(&Token::RBrack)?;
            let elem = self.parse_base_type()?;
            return Ok(StanType::Array(size, Box::new(elem)));
        }
        self.parse_base_type()
    }

    fn parse_base_type(&mut self) -> Result<StanType> {
        let tok = self.consume();
        match &tok {
            Token::Kw(s) if s == "real" => {
                let c = self.parse_constraints()?;
                Ok(StanType::Real(c))
            }
            Token::Kw(s) if s == "int" => {
                // e.g. `int<lower=0> N;` — the bound is checked against the
                // supplied data (see `Model::parse_and_load`).
                Ok(StanType::Int(self.parse_constraints()?))
            }
            Token::Kw(s) if s == "vector" => {
                let c = self.parse_constraints()?;
                self.expect_tok(&Token::LBrack)?;
                let size = self.parse_expr(0)?;
                self.expect_tok(&Token::RBrack)?;
                Ok(StanType::Vector(size, c))
            }
            Token::Kw(s) if s == "matrix" => {
                self.expect_tok(&Token::LBrack)?;
                let rows = self.parse_expr(0)?;
                self.expect_tok(&Token::Comma)?;
                let cols = self.parse_expr(0)?;
                self.expect_tok(&Token::RBrack)?;
                Ok(StanType::Matrix(rows, cols))
            }
            Token::Kw(s) if s == "simplex" => {
                self.expect_tok(&Token::LBrack)?;
                let k = self.parse_expr(0)?;
                self.expect_tok(&Token::RBrack)?;
                Ok(StanType::Simplex(k))
            }
            Token::Kw(s) if s == "ordered" => {
                self.expect_tok(&Token::LBrack)?;
                let k = self.parse_expr(0)?;
                self.expect_tok(&Token::RBrack)?;
                Ok(StanType::Ordered(k))
            }
            Token::Id(s) if s == "cholesky_factor_corr" => {
                self.expect_tok(&Token::LBrack)?;
                let k = self.parse_expr(0)?;
                self.expect_tok(&Token::RBrack)?;
                Ok(StanType::CholeskyFactorCorr(k))
            }
            Token::Id(s) if s == "cholesky_factor_cov" => {
                self.expect_tok(&Token::LBrack)?;
                let k = self.parse_expr(0)?;
                self.expect_tok(&Token::RBrack)?;
                Ok(StanType::CholeskyFactorCov(k))
            }
            Token::Id(s) if s == "cov_matrix" => {
                self.expect_tok(&Token::LBrack)?;
                let k = self.parse_expr(0)?;
                self.expect_tok(&Token::RBrack)?;
                Ok(StanType::CovMatrix(k))
            }
            Token::Id(s) if s == "corr_matrix" => {
                self.expect_tok(&Token::LBrack)?;
                let k = self.parse_expr(0)?;
                self.expect_tok(&Token::RBrack)?;
                Ok(StanType::CorrMatrix(k))
            }
            Token::Kw(s) if s == "positive_ordered" => {
                self.expect_tok(&Token::LBrack)?;
                let k = self.parse_expr(0)?;
                self.expect_tok(&Token::RBrack)?;
                Ok(StanType::PositiveOrdered(k))
            }
            Token::Id(s) if s == "unit_vector" => {
                self.expect_tok(&Token::LBrack)?;
                let k = self.parse_expr(0)?;
                self.expect_tok(&Token::RBrack)?;
                Ok(StanType::UnitVector(k))
            }
            _ => Err(ParseError::ExpectedType { got: tok }),
        }
    }

    // ---- Expressions (Pratt precedence climbing) ------------------------

    fn parse_expr(&mut self, min_prec: i32) -> Result<Expr> {
        let mut left = self.parse_unary()?;
        loop {
            let p = prec(self.peek());
            if p <= min_prec {
                break;
            }
            let op = self.consume();
            let op_str = tok_op_str(&op);
            let right = self.parse_expr(p)?;
            left = Expr::BinOp(op_str.to_string(), Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr> {
        match self.peek() {
            Token::Minus => {
                self.consume();
                Ok(Expr::UnOp("-".into(), Box::new(self.parse_unary()?)))
            }
            Token::Plus => {
                self.consume();
                self.parse_unary()
            }
            Token::Bang => {
                self.consume();
                Ok(Expr::UnOp("!".into(), Box::new(self.parse_unary()?)))
            }
            _ => self.parse_power(),
        }
    }

    /// `^` is handled here rather than in the `prec` table because Stan gives
    /// it two properties the precedence-climbing loop can't express:
    /// it binds *tighter than unary minus* (`-a^2` is `-(a^2)`, not `(-a)^2`)
    /// and it is *right-associative* (`2^3^2` is `2^(3^2)` = 512, not 64).
    /// Recursing into `parse_unary` for the exponent gives both: the base
    /// comes from `parse_postfix` (so a leading `-` never gets swallowed into
    /// it), and the exponent may itself be a signed power (`a^-b`, `2^3^2`).
    fn parse_power(&mut self) -> Result<Expr> {
        let base = self.parse_postfix()?;
        if matches!(self.peek(), Token::Caret) {
            self.consume();
            let exp = self.parse_unary()?;
            return Ok(Expr::BinOp("^".into(), Box::new(base), Box::new(exp)));
        }
        Ok(base)
    }

    fn parse_postfix(&mut self) -> Result<Expr> {
        let mut e = self.parse_primary()?;
        loop {
            if !matches!(self.peek(), Token::LBrack) {
                break;
            }
            self.consume(); // [
            let idx = self.parse_expr(0)?;
            // Range index v[a:b] → segment(v, a, b-a+1)
            if self.try_tok(&Token::Colon) {
                let hi = self.parse_expr(0)?;
                self.expect_tok(&Token::RBrack)?;
                // `IntNum(1)`, not `Num(1.0)`: the length has to stay
                // int-typed so `/` inside a slice bound keeps Stan's integer
                // division semantics.
                let len = Expr::BinOp(
                    "+".into(),
                    Box::new(Expr::BinOp("-".into(), Box::new(hi), Box::new(idx.clone()))),
                    Box::new(Expr::IntNum(1)),
                );
                e = Expr::Call("segment".into(), vec![e, idx, len]);
            } else {
                e = Expr::Index(Box::new(e), Box::new(idx));
                // A[i,j] → Index(Index(A, i), j)
                while self.try_tok(&Token::Comma) {
                    let idx2 = self.parse_expr(0)?;
                    e = Expr::Index(Box::new(e), Box::new(idx2));
                }
                self.expect_tok(&Token::RBrack)?;
            }
        }
        Ok(e)
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        match self.peek().clone() {
            Token::Num(v) => {
                self.consume();
                Ok(Expr::Num(v))
            }
            Token::IntNum(v) => {
                self.consume();
                Ok(Expr::IntNum(v))
            }
            Token::Id(name) | Token::Kw(name) => {
                self.consume();
                if matches!(self.peek(), Token::LParen) {
                    self.consume();
                    let mut args: Vec<Expr> = Vec::new();
                    if !matches!(self.peek(), Token::RParen) {
                        args.push(self.parse_expr(0)?);
                        if self.try_tok(&Token::Pipe) {
                            args.push(self.parse_expr(0)?);
                        }
                        while self.try_tok(&Token::Comma) {
                            args.push(self.parse_expr(0)?);
                        }
                    }
                    self.expect_tok(&Token::RParen)?;
                    Ok(Expr::Call(name, args))
                } else {
                    Ok(Expr::Var(name))
                }
            }
            Token::LParen => {
                self.consume();
                let e = self.parse_expr(0)?;
                self.expect_tok(&Token::RParen)?;
                Ok(e)
            }
            got => Err(ParseError::UnexpectedInExpr { got }),
        }
    }

    // ---- Statements ------------------------------------------------------

    fn parse_block(&mut self) -> Result<Vec<Stmt>> {
        self.expect_tok(&Token::LBrace)?;
        let mut stmts: Vec<Stmt> = Vec::new();
        while !self.check_tok(&Token::RBrace) && !self.check_tok(&Token::Eof) {
            stmts.push(self.parse_stmt()?);
        }
        self.expect_tok(&Token::RBrace)?;
        Ok(stmts)
    }

    fn parse_stmt_block_or_single(&mut self) -> Result<Vec<Stmt>> {
        if self.check_tok(&Token::LBrace) {
            self.parse_block()
        } else {
            Ok(vec![self.parse_stmt()?])
        }
    }

    fn parse_stmt(&mut self) -> Result<Stmt> {
        match self.peek().clone() {
            Token::Kw(s) if s == "for" => {
                self.consume();
                self.expect_tok(&Token::LParen)?;
                let loop_var = self.expect_id()?;
                self.expect_kw("in")?;
                let lo = self.parse_expr(0)?;
                self.expect_tok(&Token::Colon)?;
                let hi = self.parse_expr(0)?;
                self.expect_tok(&Token::RParen)?;
                let body = self.parse_stmt_block_or_single()?;
                Ok(Stmt::For(loop_var, lo, hi, body))
            }
            Token::Kw(s) if s == "while" => {
                self.consume();
                self.expect_tok(&Token::LParen)?;
                let cond = self.parse_expr(0)?;
                self.expect_tok(&Token::RParen)?;
                let body = self.parse_stmt_block_or_single()?;
                Ok(Stmt::While(cond, body))
            }
            Token::Kw(s) if s == "if" => {
                self.consume();
                self.expect_tok(&Token::LParen)?;
                let cond = self.parse_expr(0)?;
                self.expect_tok(&Token::RParen)?;
                let then_body = self.parse_stmt_block_or_single()?;
                let else_body = if self.try_kw("else") {
                    self.parse_stmt_block_or_single()?
                } else {
                    Vec::new()
                };
                Ok(Stmt::If(cond, then_body, else_body))
            }
            Token::Kw(s) if s == "target" => {
                self.consume();
                self.expect_tok(&Token::AddEq)?;
                let e = self.parse_expr(0)?;
                self.expect_tok(&Token::Semi)?;
                Ok(Stmt::TargetIncr(e))
            }
            Token::Kw(s) if s == "return" => {
                self.consume();
                let e = self.parse_expr(0)?;
                self.expect_tok(&Token::Semi)?;
                Ok(Stmt::Return(e))
            }
            Token::Kw(s) if s == "break" => {
                self.consume();
                self.expect_tok(&Token::Semi)?;
                Ok(Stmt::Break)
            }
            Token::Kw(s) if s == "continue" => {
                self.consume();
                self.expect_tok(&Token::Semi)?;
                Ok(Stmt::Continue)
            }
            tok => {
                if is_type_kw(&tok) {
                    let typ = self.parse_type()?;
                    let name = self.expect_id()?;
                    let init = if self.try_tok(&Token::Equals) {
                        Some(self.parse_expr(0)?)
                    } else {
                        None
                    };
                    self.expect_tok(&Token::Semi)?;
                    return Ok(Stmt::LocalDecl(typ, name, init));
                }
                let lhs = self.parse_expr(0)?;
                match self.peek() {
                    Token::Tilde => {
                        self.consume();
                        let dist_expr = self.parse_expr(0)?;
                        self.expect_tok(&Token::Semi)?;
                        match dist_expr {
                            Expr::Call(name, args) => Ok(Stmt::Sample(lhs, name, args)),
                            _ => Err(ParseError::BadSample),
                        }
                    }
                    Token::Equals => {
                        self.consume();
                        let rhs = self.parse_expr(0)?;
                        self.expect_tok(&Token::Semi)?;
                        Ok(Stmt::Assign(lhs, rhs))
                    }
                    Token::AddEq => {
                        self.consume();
                        let rhs = self.parse_expr(0)?;
                        self.expect_tok(&Token::Semi)?;
                        Ok(Stmt::IncrAssign(lhs, rhs))
                    }
                    _ => Err(ParseError::UnsupportedStatement),
                }
            }
        }
    }

    // ---- Variable declarations (data / parameters) ----------------------

    fn parse_var_decls(&mut self) -> Result<Vec<VarDecl>> {
        let mut decls: Vec<VarDecl> = Vec::new();
        while !self.check_tok(&Token::RBrace) && !self.check_tok(&Token::Eof) {
            let typ = self.parse_type()?;
            let name = self.expect_id()?;
            let init = if self.try_tok(&Token::Equals) {
                Some(self.parse_expr(0)?)
            } else {
                None
            };
            self.expect_tok(&Token::Semi)?;
            decls.push(VarDecl { typ, name, init });
        }
        Ok(decls)
    }

    // ---- Top-level ------------------------------------------------------

    pub fn parse(&mut self) -> Result<StanProgram> {
        let mut prog = StanProgram::default();

        while !self.check_tok(&Token::Eof) {
            let block_name = match self.consume() {
                Token::Kw(s) | Token::Id(s) => s,
                got => return Err(ParseError::BadBlockName { got }),
            };

            let full_name: String = match block_name.as_str() {
                "generated" => {
                    if self.check_kw("quantities") || self.check_id("quantities") {
                        self.consume();
                        "generated_quantities".into()
                    } else {
                        "generated".into()
                    }
                }
                "transformed" => {
                    if self.check_kw("parameters") {
                        self.consume();
                        "transformed_parameters".into()
                    } else if self.check_kw("data") || self.check_id("data") {
                        self.consume();
                        "transformed_data".into()
                    } else {
                        "transformed".into()
                    }
                }
                _ => block_name,
            };

            self.expect_tok(&Token::LBrace)?;
            match full_name.as_str() {
                "functions" => {
                    prog.functions = self.parse_functions_block()?;
                }
                "data" => {
                    prog.data = self.parse_var_decls()?;
                }
                "parameters" => {
                    prog.parameters = self.parse_var_decls()?;
                }
                "transformed_parameters" => {
                    let (decls, stmts) = self.parse_mixed_block()?;
                    prog.transformed_params = decls;
                    prog.transformed_stmts = stmts;
                }
                "model" => {
                    for s in self.parse_block_body()? {
                        prog.model.push(s);
                    }
                }
                "transformed_data" => {
                    // transformed_data is pre-processed before parameters; treat as model preamble
                    for s in self.parse_block_body()? {
                        prog.model.push(s);
                    }
                }
                "generated_quantities" => {
                    for s in self.parse_block_body()? {
                        prog.gen_quantities.push(s);
                    }
                }
                _ => {
                    return Err(ParseError::UnknownBlock { name: full_name });
                }
            }
            self.expect_tok(&Token::RBrace)?;
        }

        Ok(prog)
    }

    fn parse_functions_block(&mut self) -> Result<Vec<(String, FuncDef)>> {
        let mut funcs: Vec<(String, FuncDef)> = Vec::new();
        while !self.check_tok(&Token::RBrace) && !self.check_tok(&Token::Eof) {
            let _ret = self.parse_type()?; // ignored: scalar
            let fname = self.expect_id()?;
            self.expect_tok(&Token::LParen)?;
            let mut params: Vec<(StanType, String)> = Vec::new();
            while !self.check_tok(&Token::RParen) && !self.check_tok(&Token::Eof) {
                let ptype = self.parse_type()?;
                let pname = self.expect_id()?;
                params.push((ptype, pname));
                let _ = self.try_tok(&Token::Comma);
            }
            self.expect_tok(&Token::RParen)?;
            self.expect_tok(&Token::LBrace)?;
            let mut body: Vec<Stmt> = Vec::new();
            let mut ret_expr = Expr::Num(0.0);
            while !self.check_tok(&Token::RBrace) && !self.check_tok(&Token::Eof) {
                let stmt = self.parse_stmt()?;
                if let Stmt::Return(e) = stmt {
                    ret_expr = e;
                    break;
                } else {
                    body.push(stmt);
                }
            }
            while !self.check_tok(&Token::RBrace) && !self.check_tok(&Token::Eof) {
                self.consume();
            }
            self.expect_tok(&Token::RBrace)?;
            funcs.push((
                fname,
                FuncDef {
                    params,
                    body,
                    ret_expr,
                },
            ));
        }
        Ok(funcs)
    }

    fn parse_block_body(&mut self) -> Result<Vec<Stmt>> {
        let mut stmts: Vec<Stmt> = Vec::new();
        while !self.check_tok(&Token::RBrace) && !self.check_tok(&Token::Eof) {
            stmts.push(self.parse_stmt()?);
        }
        Ok(stmts)
    }

    fn parse_mixed_block(&mut self) -> Result<(Vec<VarDecl>, Vec<Stmt>)> {
        let mut decls: Vec<VarDecl> = Vec::new();
        let mut stmts: Vec<Stmt> = Vec::new();
        while !self.check_tok(&Token::RBrace) && !self.check_tok(&Token::Eof) {
            if is_type_kw(self.peek()) {
                let saved = self.pos;
                let _ = self.parse_type()?;
                let is_decl = matches!(self.peek(), Token::Id(_) | Token::Kw(_));
                self.pos = saved;
                if is_decl {
                    let typ = self.parse_type()?;
                    let name = self.expect_id()?;
                    let init = if self.try_tok(&Token::Equals) {
                        Some(self.parse_expr(0)?)
                    } else {
                        None
                    };
                    self.expect_tok(&Token::Semi)?;
                    decls.push(VarDecl { typ, name, init });
                } else {
                    stmts.push(self.parse_stmt()?);
                }
            } else {
                stmts.push(self.parse_stmt()?);
            }
        }
        Ok((decls, stmts))
    }
}

fn prec(tok: &Token) -> i32 {
    match tok {
        Token::OrOr => 1,
        Token::AndAnd => 2,
        Token::EqEq | Token::Ne => 3,
        Token::Lt | Token::Gt | Token::Le | Token::Ge => 4,
        Token::Plus | Token::Minus => 6,
        Token::Star | Token::Slash => 7,
        // `Token::Caret` is deliberately absent: see `parse_power`.
        _ => -1,
    }
}

fn tok_op_str(tok: &Token) -> &'static str {
    match tok {
        Token::OrOr => "||",
        Token::AndAnd => "&&",
        Token::EqEq => "==",
        Token::Ne => "!=",
        Token::Lt => "<",
        Token::Gt => ">",
        Token::Le => "<=",
        Token::Ge => ">=",
        Token::Plus => "+",
        Token::Minus => "-",
        Token::Star => "*",
        Token::Slash => "/",
        Token::Caret => "^",
        _ => "?",
    }
}

fn is_type_kw(tok: &Token) -> bool {
    match tok {
        Token::Kw(s) => matches!(
            s.as_str(),
            "real"
                | "int"
                | "vector"
                | "matrix"
                | "array"
                | "simplex"
                | "ordered"
                | "positive_ordered"
        ),
        Token::Id(s) => matches!(
            s.as_str(),
            "cholesky_factor_corr"
                | "cholesky_factor_cov"
                | "cov_matrix"
                | "corr_matrix"
                | "unit_vector"
        ),
        _ => false,
    }
}

/// Convenience: tokenize, parse, return AST.
pub fn parse(src: &str) -> Result<StanProgram> {
    Parser::new(src)?.parse()
}
