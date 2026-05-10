//! Stan tokenizer. Mirrors `compiler/stan/lexer.mbt`.

use crate::token::{is_keyword, Token};

pub fn tokenize(src: &str) -> Vec<Token> {
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
            let v: f64 = s.parse().unwrap_or(0.0);
            tokens.push(Token::Num(v));
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
        if let Some(t) = tok {
            tokens.push(t);
        }
        i += 1;
    }

    tokens.push(Token::Eof);
    tokens
}
