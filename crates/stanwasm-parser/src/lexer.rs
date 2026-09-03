//! Stan tokenizer.

use crate::token::{is_keyword, Token};

/// A byte that starts no token, comment or whitespace. Previously dropped, which
/// made `.` in `.*`/`./`/`.^` vanish and leave a silently wrong matrix operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnknownChar(pub char);

impl std::fmt::Display for UnknownChar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unrecognized character `{}`", self.0)?;
        // The reason belongs to the runtime rather than the lexer, but this is
        // where the character is, and a bare "unrecognized character `'`" sends
        // the reader looking for a typo.
        if self.0 == '\'' {
            write!(
                f,
                " — transpose isn't supported: without a row vector `x'` cannot \
                 be told apart from `x`, so `x' * y` would quietly be an \
                 element-wise product where Stan means a dot product. Write \
                 `dot_product(x, y)`"
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for UnknownChar {}

pub fn tokenize(src: &str) -> Result<Vec<Token>, UnknownChar> {
    let bytes = src.as_bytes();
    let n = bytes.len();
    let mut tokens = Vec::new();
    let mut i = 0;

    while i < n {
        let c = bytes[i];

        // whitespace
        if matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
            i += 1;
            continue;
        }

        // line comment: //
        if c == b'/' && i + 1 < n && bytes[i + 1] == b'/' {
            while i < n && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        // block comment: /* ... */
        if c == b'/' && i + 1 < n && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < n && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i += 2;
            continue;
        }

        // two-char operators
        if i + 1 < n {
            let c2 = bytes[i + 1];
            let matched: Option<Token> = match (c, c2) {
                (b'<', b'=') => Some(Token::Le),
                (b'>', b'=') => Some(Token::Ge),
                (b'=', b'=') => Some(Token::EqEq),
                (b'!', b'=') => Some(Token::Ne),
                (b'+', b'=') => Some(Token::AddEq),
                (b'&', b'&') => Some(Token::AndAnd),
                (b'|', b'|') => Some(Token::OrOr),
                (b'-', b'>') => Some(Token::Arrow),
                _ => None,
            };
            if let Some(t) = matched {
                tokens.push(t);
                i += 2;
                continue;
            }
        }

        // number: digit-led, or '.' followed by digit
        if c.is_ascii_digit() || (c == b'.' && i + 1 < n && bytes[i + 1].is_ascii_digit()) {
            let start = i;
            while i < n && bytes[i].is_ascii_digit() {
                i += 1;
            }
            if i < n && bytes[i] == b'.' {
                i += 1;
                while i < n && bytes[i].is_ascii_digit() {
                    i += 1;
                }
            }
            if i < n && (bytes[i] == b'e' || bytes[i] == b'E') {
                i += 1;
                if i < n && (bytes[i] == b'+' || bytes[i] == b'-') {
                    i += 1;
                }
                while i < n && bytes[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let s = std::str::from_utf8(&bytes[start..i]).expect("ascii numeric");
            // No `.` and no exponent means an *int* in Stan, which changes what `/`
            // does. Anything past i64 falls back to the real path rather than wrapping.
            let is_int = !s.contains(['.', 'e', 'E']);
            match (is_int, s.parse::<i64>()) {
                (true, Ok(iv)) => tokens.push(Token::IntNum(iv)),
                _ => tokens.push(Token::Num(s.parse().unwrap_or(0.0))),
            }
            continue;
        }

        // identifier / keyword
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            while i < n && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let s = std::str::from_utf8(&bytes[start..i]).expect("ascii ident");
            if is_keyword(s) {
                tokens.push(Token::Kw(s.to_string()));
            } else {
                tokens.push(Token::Id(s.to_string()));
            }
            continue;
        }

        // single-char tokens
        let tok: Option<Token> = match c {
            b'{' => Some(Token::LBrace),
            b'}' => Some(Token::RBrace),
            b'(' => Some(Token::LParen),
            b')' => Some(Token::RParen),
            b'[' => Some(Token::LBrack),
            b']' => Some(Token::RBrack),
            b';' => Some(Token::Semi),
            b',' => Some(Token::Comma),
            b':' => Some(Token::Colon),
            // `.5` was already taken as a number above, so a `.` here starts an
            // elementwise operator.
            b'.' if i + 1 < n => match bytes[i + 1] {
                b'*' => {
                    i += 1;
                    Some(Token::DotStar)
                }
                b'/' => {
                    i += 1;
                    Some(Token::DotSlash)
                }
                b'^' => {
                    i += 1;
                    Some(Token::DotCaret)
                }
                _ => None,
            },
            b'?' => Some(Token::Question),
            b'<' => Some(Token::Lt),
            b'>' => Some(Token::Gt),
            b'+' => Some(Token::Plus),
            b'-' => Some(Token::Minus),
            b'*' => Some(Token::Star),
            b'/' => Some(Token::Slash),
            b'^' => Some(Token::Caret),
            b'~' => Some(Token::Tilde),
            b'=' => Some(Token::Equals),
            b'|' => Some(Token::Pipe),
            b'!' => Some(Token::Bang),
            _ => None,
        };
        match tok {
            Some(t) => tokens.push(t),
            // `c as char` renders a UTF-8 continuation byte as a Latin-1 glyph, so
            // decode the whole character from the source.
            None => return Err(UnknownChar(src[i..].chars().next().unwrap_or('\u{fffd}'))),
        }
        i += 1;
    }

    tokens.push(Token::Eof);
    Ok(tokens)
}
