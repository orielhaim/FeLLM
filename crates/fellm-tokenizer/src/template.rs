//! Extremely minimal Jinja2-subset chat template engine.
//!
//! Supports:
//!   - `{{ variable }}` (with optional `| trim` filter)
//!   - `{% for m in messages %} ... {% endfor %}`
//!   - `{% if cond %} ... {% elif cond %} ... {% else %} ... {% endif %}`
//!   - `{% set name = expr %}`
//!   - Simple boolean expressions: `a == "x"`, `a != "x"`, `not a`, `a and b`, `a or b`
//!   - Member access: `m.role`, `m.content`
//!
//! This is deliberately small — enough for Llama/Mistral/Qwen chat templates.

use fellm_core::error::{FellmError, Result};
use std::collections::BTreeMap;

/// A message in a chat conversation.
#[derive(Debug, Clone)]
pub struct Message {
    /// Role: "system", "user", "assistant", "tool".
    pub role: String,
    /// Content text.
    pub content: String,
}

/// A template context.
#[derive(Debug, Default)]
pub struct TemplateContext {
    /// Messages.
    pub messages: Vec<Message>,
    /// Whether to add the generation prompt (assistant header).
    pub add_generation_prompt: bool,
    /// Extra scalar variables.
    pub vars: BTreeMap<String, Value>,
}

/// A value in the template engine.
#[derive(Debug, Clone)]
pub enum Value {
    /// A string.
    String(String),
    /// A bool.
    Bool(bool),
    /// A list of messages (only `messages` is expected).
    Messages(Vec<Message>),
    /// A message (loop variable inside a for).
    Message(Message),
    /// Nothing.
    None,
}

impl Value {
    fn is_truthy(&self) -> bool {
        match self {
            Self::String(s) => !s.is_empty(),
            Self::Bool(b) => *b,
            Self::Messages(v) => !v.is_empty(),
            Self::Message(_) => true,
            Self::None => false,
        }
    }

    fn as_string(&self) -> String {
        match self {
            Self::String(s) => s.clone(),
            Self::Bool(b) => b.to_string(),
            _ => String::new(),
        }
    }

    fn member(&self, name: &str) -> Value {
        match self {
            Self::Message(m) => match name {
                "role" => Value::String(m.role.clone()),
                "content" => Value::String(m.content.clone()),
                _ => Value::None,
            },
            _ => Value::None,
        }
    }
}

/// Render a chat template.
pub fn render(template: &str, ctx: &TemplateContext) -> Result<String> {
    let tokens = tokenize(template)?;
    let mut env = Env::new(ctx);
    let mut out = String::new();
    render_tokens(&tokens, &mut env, &mut out, 0)?;
    Ok(out)
}

// ---------- Lexer ----------

#[derive(Debug, Clone)]
enum Tok {
    Text(String),
    Expr(String), // {{ ... }}
    Stmt(String), // {% ... %}
}

fn tokenize(src: &str) -> Result<Vec<Tok>> {
    let mut out = Vec::new();
    let bytes = src.as_bytes();
    let mut i = 0;
    let mut cur_text = String::new();
    while i < bytes.len() {
        if i + 1 < bytes.len() && bytes[i] == b'{' && (bytes[i + 1] == b'{' || bytes[i + 1] == b'%')
        {
            if !cur_text.is_empty() {
                out.push(Tok::Text(std::mem::take(&mut cur_text)));
            }
            let is_expr = bytes[i + 1] == b'{';
            let close: &[u8] = if is_expr { b"}}" } else { b"%}" };
            i += 2;
            // Skip optional whitespace-trim '-' after the opening.
            let mut lstrip = false;
            if i < bytes.len() && bytes[i] == b'-' {
                lstrip = true;
                i += 1;
            }
            let start = i;
            let mut end = start;
            while end + 1 < bytes.len() {
                // Allow rstrip '-' before closing.
                if bytes[end] == b'-'
                    && bytes[end + 1] == close[0]
                    && end + 2 < bytes.len()
                    && bytes[end + 2] == close[1]
                {
                    let body = &src[start..end];
                    i = end + 3;
                    if lstrip {
                        // Trim trailing whitespace of previous text.
                        if let Some(Tok::Text(t)) = out.last_mut() {
                            *t = t.trim_end().to_string();
                        }
                    }
                    // Trim leading whitespace of next text.
                    let body_trim = body.trim().to_string();
                    push_stmt_or_expr(&mut out, is_expr, body_trim);
                    // Consume leading whitespace after tag.
                    while i < bytes.len() && matches!(bytes[i], b' ' | b'\t' | b'\n' | b'\r') {
                        i += 1;
                    }
                    break;
                }
                if bytes[end] == close[0] && bytes[end + 1] == close[1] {
                    let body = &src[start..end];
                    i = end + 2;
                    if lstrip {
                        if let Some(Tok::Text(t)) = out.last_mut() {
                            *t = t.trim_end().to_string();
                        }
                    }
                    let body_trim = body.trim().to_string();
                    push_stmt_or_expr(&mut out, is_expr, body_trim);
                    break;
                }
                end += 1;
            }
            if end + 1 >= bytes.len() {
                return Err(FellmError::Tokenization("unterminated template tag".into()));
            }
        } else {
            cur_text.push(bytes[i] as char);
            i += 1;
        }
    }
    if !cur_text.is_empty() {
        out.push(Tok::Text(cur_text));
    }
    Ok(out)
}

fn push_stmt_or_expr(out: &mut Vec<Tok>, is_expr: bool, body: String) {
    if is_expr {
        out.push(Tok::Expr(body));
    } else {
        out.push(Tok::Stmt(body));
    }
}

// ---------- Renderer ----------

struct Env<'a> {
    ctx: &'a TemplateContext,
    scopes: Vec<BTreeMap<String, Value>>,
}

impl<'a> Env<'a> {
    fn new(ctx: &'a TemplateContext) -> Self {
        let mut scopes = vec![BTreeMap::new()];
        scopes[0].insert("messages".into(), Value::Messages(ctx.messages.clone()));
        scopes[0].insert(
            "add_generation_prompt".into(),
            Value::Bool(ctx.add_generation_prompt),
        );
        for (k, v) in &ctx.vars {
            scopes[0].insert(k.clone(), v.clone());
        }
        Self { ctx, scopes }
    }

    fn get(&self, name: &str) -> Value {
        for s in self.scopes.iter().rev() {
            if let Some(v) = s.get(name) {
                return v.clone();
            }
        }
        // Also check ctx.vars.
        if let Some(v) = self.ctx.vars.get(name) {
            return v.clone();
        }
        Value::None
    }

    fn set(&mut self, name: &str, v: Value) {
        self.scopes.last_mut().unwrap().insert(name.into(), v);
    }

    fn push(&mut self) {
        self.scopes.push(BTreeMap::new());
    }

    fn pop(&mut self) {
        self.scopes.pop();
    }
}

fn render_tokens(
    tokens: &[Tok],
    env: &mut Env<'_>,
    out: &mut String,
    start: usize,
) -> Result<usize> {
    let mut i = start;
    while i < tokens.len() {
        match &tokens[i] {
            Tok::Text(t) => {
                out.push_str(t);
                i += 1;
            }
            Tok::Expr(body) => {
                let v = eval_expr(body, env)?;
                out.push_str(&v.as_string());
                i += 1;
            }
            Tok::Stmt(body) => {
                let bt = body.trim();
                if let Some(_rest) = bt.strip_prefix("if ") {
                    let end = find_end(tokens, i, &["endif"])?;
                    // Find else/elif branches.
                    let branches = split_if(tokens, i, end);
                    let mut chosen: Option<(usize, usize)> = None;
                    for (cond, from, to) in branches {
                        let cond_val = if cond == "__else__" {
                            true
                        } else {
                            // `cond` is like "if <expr>" or "elif <expr>" —
                            // strip either prefix and evaluate the rest.
                            let expr = cond
                                .strip_prefix("if ")
                                .or_else(|| cond.strip_prefix("elif "))
                                .unwrap_or(cond.as_str())
                                .trim();
                            eval_bool(expr, env)?
                        };
                        if cond_val {
                            chosen = Some((from, to));
                            break;
                        }
                    }
                    if let Some((from, to)) = chosen {
                        render_tokens(&tokens[..to], env, out, from)?;
                    }
                    i = end + 1;
                } else if let Some(rest) = bt.strip_prefix("for ") {
                    // for X in Y
                    let (var, list_expr) = parse_for(rest)?;
                    let end = find_end(tokens, i, &["endfor"])?;
                    let val = eval_expr(&list_expr, env)?;
                    let items: Vec<Value> = match val {
                        Value::Messages(v) => v.into_iter().map(Value::Message).collect(),
                        _ => vec![],
                    };
                    for item in items {
                        env.push();
                        env.set(&var, item);
                        render_tokens(&tokens[..end], env, out, i + 1)?;
                        env.pop();
                    }
                    i = end + 1;
                } else if let Some(rest) = bt.strip_prefix("set ") {
                    let eq = rest
                        .find('=')
                        .ok_or_else(|| FellmError::Tokenization("bad set".into()))?;
                    let name = rest[..eq].trim().to_string();
                    let expr = rest[eq + 1..].trim();
                    let v = eval_expr(expr, env)?;
                    env.set(&name, v);
                    i += 1;
                } else if bt == "endif"
                    || bt == "endfor"
                    || bt.starts_with("else")
                    || bt.starts_with("elif ")
                {
                    // Handled by parent — return to allow caller to see the closer.
                    return Ok(i);
                } else {
                    return Err(FellmError::Tokenization(format!("unknown stmt: {bt}")));
                }
            }
        }
    }
    Ok(i)
}

fn find_end(tokens: &[Tok], from: usize, closers: &[&str]) -> Result<usize> {
    let mut depth = 1i32;
    for (j, t) in tokens.iter().enumerate().skip(from + 1) {
        if let Tok::Stmt(s) = t {
            let s = s.trim();
            if s.starts_with("if ") || s.starts_with("for ") {
                depth += 1;
            } else if closers.iter().any(|c| s == *c) {
                depth -= 1;
                if depth == 0 {
                    return Ok(j);
                }
            }
        }
    }
    Err(FellmError::Tokenization("unterminated block".into()))
}

fn split_if(tokens: &[Tok], start: usize, end: usize) -> Vec<(String, usize, usize)> {
    // Returns list of (cond_string, body_start, body_end_exclusive)
    let mut branches = Vec::new();
    let mut current_cond = match &tokens[start] {
        Tok::Stmt(s) => s.trim().to_string(),
        _ => return branches,
    };
    let mut body_start = start + 1;
    let mut depth = 0i32;
    for j in body_start..end {
        if let Tok::Stmt(s) = &tokens[j] {
            let s = s.trim();
            if s.starts_with("if ") || s.starts_with("for ") {
                depth += 1;
            } else if s == "endif" || s == "endfor" {
                depth -= 1;
            } else if depth == 0 && (s.starts_with("elif ") || s == "else") {
                branches.push((current_cond.clone(), body_start, j));
                current_cond = if s == "else" {
                    "__else__".to_string()
                } else {
                    s.to_string()
                };
                body_start = j + 1;
            }
        }
    }
    branches.push((current_cond, body_start, end));
    branches
}

fn parse_for(rest: &str) -> Result<(String, String)> {
    // "X in Y"
    let idx = rest
        .find(" in ")
        .ok_or_else(|| FellmError::Tokenization("bad for".into()))?;
    let var = rest[..idx].trim().to_string();
    let expr = rest[idx + 4..].trim().to_string();
    Ok((var, expr))
}

fn eval_expr(src: &str, env: &Env<'_>) -> Result<Value> {
    let s = src.trim();
    // Filter: X | trim
    if let Some((base, filter)) = s.rsplit_once('|') {
        let base_v = eval_expr(base.trim(), env)?;
        let filter = filter.trim();
        return Ok(match filter {
            "trim" => Value::String(base_v.as_string().trim().to_string()),
            "upper" => Value::String(base_v.as_string().to_uppercase()),
            "lower" => Value::String(base_v.as_string().to_lowercase()),
            _ => base_v,
        });
    }
    // String literal
    if (s.starts_with('"') && s.ends_with('"') && s.len() >= 2)
        || (s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2)
    {
        return Ok(Value::String(s[1..s.len() - 1].to_string()));
    }
    // Member access
    if let Some((head, tail)) = s.split_once('.') {
        let hv = env.get(head.trim());
        return Ok(hv.member(tail.trim()));
    }
    // Plain identifier
    Ok(env.get(s))
}

fn eval_bool(src: &str, env: &Env<'_>) -> Result<bool> {
    let s = src.trim();
    if let Some(rest) = s.strip_prefix("not ") {
        return Ok(!eval_bool(rest, env)?);
    }
    if let Some((l, r)) = split_top(s, " or ") {
        return Ok(eval_bool(&l, env)? || eval_bool(&r, env)?);
    }
    if let Some((l, r)) = split_top(s, " and ") {
        return Ok(eval_bool(&l, env)? && eval_bool(&r, env)?);
    }
    if let Some((l, r)) = split_top(s, "==") {
        let lv = eval_expr(&l, env)?;
        let rv = eval_expr(&r, env)?;
        return Ok(lv.as_string() == rv.as_string());
    }
    if let Some((l, r)) = split_top(s, "!=") {
        let lv = eval_expr(&l, env)?;
        let rv = eval_expr(&r, env)?;
        return Ok(lv.as_string() != rv.as_string());
    }
    Ok(eval_expr(s, env)?.is_truthy())
}

fn split_top(s: &str, sep: &str) -> Option<(String, String)> {
    // Split at the top-level (ignore inside quotes).
    let bytes = s.as_bytes();
    let sb = sep.as_bytes();
    let mut i = 0;
    let mut in_s = false;
    let mut in_d = false;
    while i + sb.len() <= bytes.len() {
        let c = bytes[i];
        if c == b'\'' && !in_d {
            in_s = !in_s;
        } else if c == b'"' && !in_s {
            in_d = !in_d;
        } else if !in_s && !in_d && &bytes[i..i + sb.len()] == sb {
            return Some((s[..i].to_string(), s[i + sb.len()..].to_string()));
        }
        i += 1;
    }
    None
}
