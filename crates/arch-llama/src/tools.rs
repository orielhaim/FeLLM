//! Assistant output parsing for Llama-family tool calling.
//!
//! Chat *prompt* formatting comes from the GGUF Jinja template (generic
//! tokenizer engine). This module only interprets the model's *response*
//! formats that are specific to Llama 3.1 / 3.2 Instruct:
//! - JSON: `{"name": "...", "parameters": {...}}`
//! - Pythonic: `[fn(a=1, b='x'), ...]`
//! - Optional `<|python_tag|>` prefix

use fellm_core::error::{FellmError, Result};
use fellm_tokenizer::{AssistantOutput, ToolCall};

/// Parse an assistant completion into text or tool call(s).
#[must_use]
pub fn parse_assistant_output(text: &str) -> AssistantOutput {
    let t = text.trim();
    let t = t.strip_prefix("<|python_tag|>").map(str::trim).unwrap_or(t);

    if let Some(calls) = try_parse_json_tool_call(t) {
        return AssistantOutput::ToolCalls(calls);
    }
    if let Some(calls) = try_parse_pythonic_tool_calls(t) {
        return AssistantOutput::ToolCalls(calls);
    }
    AssistantOutput::Text(text.to_string())
}

fn map_get<'a>(map: &'a [(String, JsonValue)], key: &str) -> Option<&'a JsonValue> {
    map.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn try_parse_json_tool_call(t: &str) -> Option<Vec<ToolCall>> {
    let t = t.trim();
    if !t.starts_with('{') {
        return None;
    }
    let v = parse_json_value(t).ok()?;
    let JsonValue::Object(map) = v else {
        return None;
    };
    let name = match map_get(&map, "name")? {
        JsonValue::String(s) => s.clone(),
        _ => return None,
    };
    let params = map_get(&map, "parameters")
        .or_else(|| map_get(&map, "arguments"))
        .cloned()
        .unwrap_or(JsonValue::Object(Vec::new()));
    let arguments = json_value_to_string(&params);
    Some(vec![ToolCall {
        id: String::new(),
        name,
        arguments,
    }])
}

fn try_parse_pythonic_tool_calls(t: &str) -> Option<Vec<ToolCall>> {
    let t = t.trim();
    if !t.starts_with('[') || !t.ends_with(']') {
        return None;
    }
    if t.contains("\"name\"") {
        return None;
    }
    let inner = t[1..t.len() - 1].trim();
    if inner.is_empty() {
        return None;
    }
    let mut calls = Vec::new();
    for piece in split_top_level_calls(inner) {
        let piece = piece.trim();
        let paren = piece.find('(')?;
        if !piece.ends_with(')') {
            return None;
        }
        let name = piece[..paren].trim().to_string();
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return None;
        }
        let args_src = &piece[paren + 1..piece.len() - 1];
        let arguments = py_kwargs_to_json(args_src).ok()?;
        calls.push(ToolCall {
            id: String::new(),
            name,
            arguments,
        });
    }
    if calls.is_empty() {
        None
    } else {
        Some(calls)
    }
}

fn split_top_level_calls(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut in_s = false;
    let mut in_d = false;
    let mut start = 0usize;
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == '\'' && !in_d {
            in_s = !in_s;
        } else if c == '"' && !in_s {
            in_d = !in_d;
        } else if !in_s && !in_d {
            match c {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                ',' if depth == 0 => {
                    out.push(s[start..i].to_string());
                    start = i + 1;
                }
                _ => {}
            }
        }
        i += 1;
    }
    if start < s.len() {
        out.push(s[start..].to_string());
    }
    out
}

fn py_kwargs_to_json(src: &str) -> Result<String> {
    let src = src.trim();
    if src.is_empty() {
        return Ok("{}".into());
    }
    let mut parts = Vec::new();
    for piece in split_top_level_calls(src) {
        let piece = piece.trim();
        let eq = piece
            .find('=')
            .ok_or_else(|| FellmError::Tokenization(format!("bad kwarg: {piece}")))?;
        let key = piece[..eq].trim();
        let val = piece[eq + 1..].trim();
        parts.push(format!(
            "\"{}\":{}",
            escape_json_string(key),
            py_literal_to_json(val)?
        ));
    }
    Ok(format!("{{{}}}", parts.join(",")))
}

fn py_literal_to_json(src: &str) -> Result<String> {
    let s = src.trim();
    if s == "None" {
        return Ok("null".into());
    }
    if s == "True" {
        return Ok("true".into());
    }
    if s == "False" {
        return Ok("false".into());
    }
    if (s.starts_with('\'') && s.ends_with('\'')) || (s.starts_with('"') && s.ends_with('"')) {
        let inner = &s[1..s.len() - 1];
        return Ok(format!("\"{}\"", escape_json_string(inner)));
    }
    if s.parse::<f64>().is_ok() {
        return Ok(s.to_string());
    }
    if (s.starts_with('[') || s.starts_with('{')) && parse_json_value(s).is_ok() {
        return Ok(s.to_string());
    }
    Ok(format!("\"{}\"", escape_json_string(s)))
}

fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                use std::fmt::Write as _;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

#[derive(Debug, Clone)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

fn json_value_to_string(v: &JsonValue) -> String {
    match v {
        JsonValue::Null => "null".into(),
        JsonValue::Bool(b) => b.to_string(),
        JsonValue::Number(n) => n.clone(),
        JsonValue::String(s) => format!("\"{}\"", escape_json_string(s)),
        JsonValue::Array(a) => {
            let inner: Vec<_> = a.iter().map(json_value_to_string).collect();
            format!("[{}]", inner.join(","))
        }
        JsonValue::Object(o) => {
            let inner: Vec<_> = o
                .iter()
                .map(|(k, v)| format!("\"{}\":{}", escape_json_string(k), json_value_to_string(v)))
                .collect();
            format!("{{{}}}", inner.join(","))
        }
    }
}

fn parse_json_value(s: &str) -> Result<JsonValue> {
    let mut p = JsonParser {
        src: s.as_bytes(),
        i: 0,
    };
    let v = p.parse_value()?;
    p.skip_ws();
    if p.i != p.src.len() {
        return Err(FellmError::Tokenization("trailing junk in JSON".into()));
    }
    Ok(v)
}

struct JsonParser<'a> {
    src: &'a [u8],
    i: usize,
}

impl<'a> JsonParser<'a> {
    fn skip_ws(&mut self) {
        while self.i < self.src.len() && self.src[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.i).copied()
    }

    fn bump(&mut self) -> Result<u8> {
        let c = self
            .peek()
            .ok_or_else(|| FellmError::Tokenization("unexpected end of JSON".into()))?;
        self.i += 1;
        Ok(c)
    }

    fn parse_value(&mut self) -> Result<JsonValue> {
        self.skip_ws();
        match self.peek() {
            Some(b'n') => self.parse_literal(b"null", JsonValue::Null),
            Some(b't') => self.parse_literal(b"true", JsonValue::Bool(true)),
            Some(b'f') => self.parse_literal(b"false", JsonValue::Bool(false)),
            Some(b'"') => Ok(JsonValue::String(self.parse_string()?)),
            Some(b'[') => self.parse_array(),
            Some(b'{') => self.parse_object(),
            Some(b'-') | Some(b'0'..=b'9') => self.parse_number(),
            other => Err(FellmError::Tokenization(format!(
                "unexpected JSON byte {other:?}"
            ))),
        }
    }

    fn parse_literal(&mut self, lit: &[u8], v: JsonValue) -> Result<JsonValue> {
        for &b in lit {
            if self.bump()? != b {
                return Err(FellmError::Tokenization("bad JSON literal".into()));
            }
        }
        Ok(v)
    }

    fn parse_string(&mut self) -> Result<String> {
        if self.bump()? != b'"' {
            return Err(FellmError::Tokenization("expected string".into()));
        }
        let mut out = String::new();
        loop {
            match self.bump()? {
                b'"' => return Ok(out),
                b'\\' => match self.bump()? {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let mut hex = 0u32;
                        for _ in 0..4 {
                            let c = self.bump()? as char;
                            hex = hex * 16
                                + c.to_digit(16).ok_or_else(|| {
                                    FellmError::Tokenization("bad unicode escape".into())
                                })?;
                        }
                        out.push(char::from_u32(hex).unwrap_or('\u{FFFD}'));
                    }
                    c => out.push(c as char),
                },
                c => out.push(c as char),
            }
        }
    }

    fn parse_number(&mut self) -> Result<JsonValue> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.i += 1;
        }
        if self.peek() == Some(b'.') {
            self.i += 1;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.i += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.i += 1;
            }
        }
        let s = std::str::from_utf8(&self.src[start..self.i])
            .map_err(|_| FellmError::Tokenization("bad number utf8".into()))?;
        Ok(JsonValue::Number(s.to_string()))
    }

    fn parse_array(&mut self) -> Result<JsonValue> {
        self.bump()?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.bump()? {
                b']' => return Ok(JsonValue::Array(items)),
                b',' => continue,
                _ => return Err(FellmError::Tokenization("bad JSON array".into())),
            }
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue> {
        self.bump()?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(JsonValue::Object(items));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            if self.bump()? != b':' {
                return Err(FellmError::Tokenization("expected ':'".into()));
            }
            let val = self.parse_value()?;
            items.push((key, val));
            self.skip_ws();
            match self.bump()? {
                b'}' => return Ok(JsonValue::Object(items)),
                b',' => continue,
                _ => return Err(FellmError::Tokenization("bad JSON object".into())),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_json_tool_call() {
        let out = parse_assistant_output(
            r#"{"name": "get_weather", "parameters": {"city": "SF"}}"#,
        );
        match out {
            AssistantOutput::ToolCalls(c) => {
                assert_eq!(c[0].name, "get_weather");
                assert!(c[0].arguments.contains("SF"));
            }
            AssistantOutput::Text(t) => panic!("expected tool call, got {t}"),
        }
    }

    #[test]
    fn parses_pythonic_tool_calls() {
        let out =
            parse_assistant_output("[get_weather(city='San Francisco', metric='celsius')]");
        match out {
            AssistantOutput::ToolCalls(c) => {
                assert_eq!(c[0].name, "get_weather");
                assert!(c[0].arguments.contains("San Francisco"));
            }
            AssistantOutput::Text(t) => panic!("expected tool call, got {t}"),
        }
    }
}
