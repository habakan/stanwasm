//! Stan tokens.

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Num(f64),
    Id(String),
    Kw(String),
    // Two-char operators
    Le,     // <=
    Ge,     // >=
    EqEq,   // ==
    Ne,     // !=
    AddEq,  // +=
    AndAnd, // &&
    OrOr,   // ||
    Arrow,  // ->
    // Single-char tokens
    LBrace, // {
    RBrace, // }
    LParen, // (
    RParen, // )
    LBrack, // [
    RBrack, // ]
    Semi,   // ;
    Comma,  // ,
    Colon,  // :
    Lt,     // <
    Gt,     // >
    Plus,   // +
    Minus,  // -
    Star,   // *
    Slash,  // /
    Caret,  // ^
    Tilde,  // ~
    Equals, // =
    Pipe,   // |
    Bang,   // !
    Eof,
}

pub const KEYWORDS: &[&str] = &[
    "data",
    "parameters",
    "transformed",
    "model",
    "generated",
    "quantities",
    "functions",
    "real",
    "int",
    "vector",
    "matrix",
    "array",
    "simplex",
    "ordered",
    "positive_ordered",
    "for",
    "in",
    "if",
    "else",
    "while",
    "return",
    "break",
    "continue",
    "target",
    "lower",
    "upper",
    "print",
    "reject",
];

pub fn is_keyword(s: &str) -> bool {
    KEYWORDS.contains(&s)
}
