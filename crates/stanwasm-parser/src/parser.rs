//! Stan recursive-descent parser.

use crate::lexer::tokenize;
use crate::token::Token;
use stanwasm_ast::{Constraint, Expr, FuncDef, SliceIdx, StanProgram, StanType, Stmt, VarDecl};
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
    #[error("{0}")]
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
            let mut sizes = vec![self.parse_expr(0)?];
            while self.try_tok(&Token::Comma) {
                sizes.push(self.parse_expr(0)?);
            }
            self.expect_tok(&Token::RBrack)?;
            let elem = self.parse_base_type()?;
            // `array[a, b] T` is `array[a] array[b] T`, so the rightmost size is
            // the innermost array.
            return Ok(sizes
                .into_iter()
                .rev()
                .fold(elem, |inner, size| StanType::Array(size, Box::new(inner))));
        }
        self.parse_base_type()
    }

    /// Types as they appear in a function signature, where Stan omits the sizes
    /// (`vector v`, not `vector[N] v`) because the argument carries its own length.
    fn parse_param_type(&mut self) -> Result<StanType> {
        if self.check_kw("array") {
            self.consume();
            self.expect_tok(&Token::LBrack)?;
            // `array[] real x` and `array[,] real x` — the dimensions are unsized.
            while !self.check_tok(&Token::RBrack) && !self.check_tok(&Token::Eof) {
                self.consume();
            }
            self.expect_tok(&Token::RBrack)?;
            let elem = self.parse_param_type()?;
            return Ok(StanType::Array(Expr::Num(0.0), Box::new(elem)));
        }
        for (kw, build) in [
            ("vector", 1usize),
            ("row_vector", 1),
            ("matrix", 2),
            ("simplex", 1),
            ("ordered", 1),
        ] {
            if self.check_kw(kw) || self.check_id(kw) {
                self.consume();
                let c = self.parse_constraints()?;
                if self.check_tok(&Token::LBrack) {
                    // A sized signature is not Stan, but accepting it costs nothing
                    // and keeps the error away from a merely unusual spelling.
                    while !self.check_tok(&Token::RBrack) && !self.check_tok(&Token::Eof) {
                        self.consume();
                    }
                    self.expect_tok(&Token::RBrack)?;
                }
                let zero = Expr::Num(0.0);
                return Ok(match (kw, build) {
                    ("matrix", _) => StanType::Matrix(zero.clone(), zero, c),
                    ("simplex", _) => StanType::Simplex(zero),
                    ("ordered", _) => StanType::Ordered(zero),
                    ("row_vector", _) => StanType::RowVector(zero, c),
                    _ => StanType::Vector(zero, c),
                });
            }
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
                let c = self.parse_constraints()?;
                self.expect_tok(&Token::LBrack)?;
                let rows = self.parse_expr(0)?;
                self.expect_tok(&Token::Comma)?;
                let cols = self.parse_expr(0)?;
                self.expect_tok(&Token::RBrack)?;
                Ok(StanType::Matrix(rows, cols, c))
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
            Token::Id(s) if s == "row_vector" => {
                let c = self.parse_constraints()?;
                self.expect_tok(&Token::LBrack)?;
                let size = self.parse_expr(0)?;
                self.expect_tok(&Token::RBrack)?;
                Ok(StanType::RowVector(size, c))
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
        // Lowest precedence and right-associative, so it binds after every binary
        // operator and `a ? b : c ? d : e` groups to the right the way Stan does.
        if min_prec <= 0 && self.try_tok(&Token::Question) {
            let then_e = self.parse_expr(0)?;
            self.expect_tok(&Token::Colon)?;
            let else_e = self.parse_expr(0)?;
            return Ok(Expr::Ternary(
                Box::new(left),
                Box::new(then_e),
                Box::new(else_e),
            ));
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

    /// `^` is handled here rather than in `prec`: it binds tighter than unary minus
    /// (`-a^2` is `-(a^2)`) and is right-associative (`2^3^2` is 512).
    fn parse_power(&mut self) -> Result<Expr> {
        let base = self.parse_postfix()?;
        // `.^` shares `^`'s shape: right-associative and tighter than unary minus.
        if matches!(self.peek(), Token::Caret | Token::DotCaret) {
            let op = if matches!(self.peek(), Token::DotCaret) {
                ".^"
            } else {
                "^"
            };
            self.consume();
            let exp = self.parse_unary()?;
            return Ok(Expr::BinOp(op.into(), Box::new(base), Box::new(exp)));
        }
        Ok(base)
    }

    fn parse_postfix(&mut self) -> Result<Expr> {
        let mut e = self.parse_primary()?;
        loop {
            // `'` binds tighter than `^`, so it lives here with indexing.
            if self.try_tok(&Token::Quote) {
                e = Expr::UnOp("'".into(), Box::new(e));
                continue;
            }
            if !matches!(self.peek(), Token::LBrack) {
                break;
            }
            self.consume(); // [
            let idxs = self.parse_index_list()?;
            // Every dimension a plain index is `A[i, j]` → `Index(Index(A, i), j)`,
            // which is the shape the evaluator and `is_int_expr` already walk.
            let plain: Option<Vec<Expr>> = idxs
                .iter()
                .map(|i| match i {
                    SliceIdx::At(x) => Some(x.clone()),
                    SliceIdx::Range(..) => None,
                })
                .collect();
            e = match plain {
                Some(xs) => xs
                    .into_iter()
                    .fold(e, |acc, x| Expr::Index(Box::new(acc), Box::new(x))),
                None => Expr::Slice(Box::new(e), idxs),
            };
        }
        Ok(e)
    }

    /// The comma-separated indices inside one `[...]`, up to and including the
    /// closing bracket.
    fn parse_index_list(&mut self) -> Result<Vec<SliceIdx>> {
        let mut out = Vec::new();
        loop {
            let lo = if self.check_tok(&Token::Colon) {
                None
            } else {
                Some(self.parse_expr(0)?)
            };
            if self.try_tok(&Token::Colon) {
                let hi = if self.check_tok(&Token::RBrack) || self.check_tok(&Token::Comma) {
                    None
                } else {
                    Some(self.parse_expr(0)?)
                };
                out.push(SliceIdx::Range(lo, hi));
            } else {
                match lo {
                    Some(e) => out.push(SliceIdx::At(e)),
                    None => {
                        return Err(ParseError::UnexpectedInExpr {
                            got: self.consume(),
                        })
                    }
                }
            }
            if !self.try_tok(&Token::Comma) {
                break;
            }
        }
        self.expect_tok(&Token::RBrack)?;
        Ok(out)
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
            // `[a, b, c]` — a row vector of scalars, or a matrix of its rows.
            // Spelled as a call because it needs no evaluation rule of its own,
            // and `[]` is not something a Stan program can name.
            Token::LBrack => {
                self.consume();
                let mut args = vec![self.parse_expr(0)?];
                while self.try_tok(&Token::Comma) {
                    args.push(self.parse_expr(0)?);
                }
                self.expect_tok(&Token::RBrack)?;
                Ok(Expr::Call("[]".into(), args))
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
            // A block on its own, which Stan uses to scope a local declaration
            // to the middle of another block.
            Token::LBrace => Ok(Stmt::Block(self.parse_block()?)),
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
                    for s in self.parse_block_body()? {
                        prog.transformed_data.push(s);
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
            // The return type is unsized too (`matrix f(...)`), and nothing downstream
            // needs it — the returned value carries its own shape.
            let _ret = self.parse_param_type()?;
            let fname = self.expect_id()?;
            self.expect_tok(&Token::LParen)?;
            let mut params: Vec<(StanType, String)> = Vec::new();
            while !self.check_tok(&Token::RParen) && !self.check_tok(&Token::Eof) {
                let ptype = self.parse_param_type()?;
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
        Token::Star | Token::Slash | Token::DotStar | Token::DotSlash => 7,
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
        Token::DotStar => ".*",
        Token::DotSlash => "./",
        Token::DotCaret => ".^",
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
            "row_vector"
                | "cholesky_factor_corr"
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
