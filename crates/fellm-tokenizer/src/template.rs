//! General Jinja2-subset renderer for GGUF `tokenizer.chat_template` strings.
//!
//! This intentionally implements only the subset commonly used by chat
//! templates: variables, conditionals, loops, assignments, comments,
//! whitespace control, filters, tests, indexing, slicing, simple function
//! calls, and JSON conversion. It contains no model-specific formatting logic.

use fellm_core::error::{FellmError, Result};
use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// A message in a chat conversation.
#[derive(Debug, Clone)]
pub struct Message {
    /// Role: "system", "user", "assistant", "tool" / "ipython".
    pub role: String,
    /// Content text (tool results go here for role=tool/ipython).
    pub content: String,
    /// Assistant tool calls (when the model requested tools).
    pub tool_calls: Vec<ToolCall>,
}

impl Message {
    /// User / system / assistant text message.
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            tool_calls: Vec::new(),
        }
    }

    /// Tool / ipython result message.
    pub fn tool_result(content: impl Into<String>) -> Self {
        Self {
            role: "ipython".into(),
            content: content.into(),
            tool_calls: Vec::new(),
        }
    }

    /// Assistant turn that invoked tools.
    pub fn assistant_tools(calls: Vec<ToolCall>) -> Self {
        Self {
            role: "assistant".into(),
            content: String::new(),
            tool_calls: calls,
        }
    }
}

/// A tool the model may call (OpenAI-style function schema, flattened).
#[derive(Debug, Clone)]
pub struct ToolDef {
    /// Function name.
    pub name: String,
    /// Human description.
    pub description: String,
    /// JSON Schema object for parameters (as a JSON string).
    pub parameters_json: String,
}

/// One tool invocation requested by the assistant.
#[derive(Debug, Clone)]
pub struct ToolCall {
    /// Optional call id (OpenAI-style); may be empty for native formats.
    pub id: String,
    /// Function name.
    pub name: String,
    /// Arguments as a JSON object string.
    pub arguments: String,
}

/// Result of parsing an assistant turn (text vs tool calls).
///
/// Architecture plugins decide *how* to parse model output into this shape;
/// the type itself is format-agnostic.
#[derive(Debug, Clone)]
pub enum AssistantOutput {
    /// Normal text reply.
    Text(String),
    /// One or more tool invocations.
    ToolCalls(Vec<ToolCall>),
}

/// OpenAI-style function wrapper used when ingesting tool lists.
#[derive(Debug, Clone)]
pub struct ToolFunctionCall {
    /// Function name.
    pub name: String,
    /// Optional description.
    pub description: Option<String>,
    /// Optional JSON-schema parameters object (as string).
    pub parameters_json: Option<String>,
}

/// A template context.
#[derive(Debug, Default)]
pub struct TemplateContext {
    /// Messages.
    pub messages: Vec<Message>,
    /// Whether to add the generation prompt (assistant header).
    pub add_generation_prompt: bool,
    /// Extra template variables, for example `bos_token` and `eos_token`.
    pub vars: BTreeMap<String, Value>,
    /// Optional tools for templates that understand them.
    pub tools: Vec<ToolDef>,
}

/// A value in the template engine.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// A string.
    String(String),
    /// A boolean.
    Bool(bool),
    /// An integer.
    Int(i64),
    /// A list.
    List(Vec<Value>),
    /// A string-keyed map.
    Map(BTreeMap<String, Value>),
    /// Nothing / null.
    None,
}

impl Value {
    fn is_truthy(&self) -> bool {
        match self {
            Self::String(s) => !s.is_empty(),
            Self::Bool(b) => *b,
            Self::Int(i) => *i != 0,
            Self::List(v) => !v.is_empty(),
            Self::Map(m) => !m.is_empty(),
            Self::None => false,
        }
    }

    fn as_output(&self) -> String {
        match self {
            Self::String(s) => s.clone(),
            Self::Bool(true) => "true".to_string(),
            Self::Bool(false) => "false".to_string(),
            Self::Int(i) => i.to_string(),
            Self::List(_) | Self::Map(_) => to_json(self, None),
            Self::None => String::new(),
        }
    }

    fn get_member(&self, name: &str) -> Option<Value> {
        match self {
            Self::Map(m) => m.get(name).cloned(),
            _ => None,
        }
    }

    fn get_index(&self, idx: i64) -> Option<Value> {
        match self {
            Self::List(v) => {
                let len = i64::try_from(v.len()).ok()?;
                let idx = if idx < 0 { len + idx } else { idx };
                usize::try_from(idx).ok().and_then(|i| v.get(i).cloned())
            }
            Self::String(s) => {
                let chars: Vec<char> = s.chars().collect();
                let len = i64::try_from(chars.len()).ok()?;
                let idx = if idx < 0 { len + idx } else { idx };
                usize::try_from(idx)
                    .ok()
                    .and_then(|i| chars.get(i).copied())
                    .map(|c| Value::String(c.to_string()))
            }
            _ => None,
        }
    }

    fn slice(&self, start: Option<i64>, end: Option<i64>) -> Value {
        match self {
            Self::List(v) => {
                let len = i64::try_from(v.len()).unwrap_or(0);
                let start = normalize_slice_bound(start.unwrap_or(0), len);
                let end = normalize_slice_bound(end.unwrap_or(len), len);
                if start >= end {
                    return Value::List(Vec::new());
                }
                Value::List(v[start as usize..end as usize].to_vec())
            }
            Self::String(s) => {
                let chars: Vec<char> = s.chars().collect();
                let len = i64::try_from(chars.len()).unwrap_or(0);
                let start = normalize_slice_bound(start.unwrap_or(0), len);
                let end = normalize_slice_bound(end.unwrap_or(len), len);
                if start >= end {
                    return Value::String(String::new());
                }
                Value::String(chars[start as usize..end as usize].iter().collect())
            }
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

// ---------- Environment ----------

struct Env {
    scopes: Vec<BTreeMap<String, Value>>,
}

impl Env {
    fn new(ctx: &TemplateContext) -> Self {
        let mut root = ctx.vars.clone();
        root.insert("messages".to_string(), messages_to_value(&ctx.messages));
        root.insert(
            "add_generation_prompt".to_string(),
            Value::Bool(ctx.add_generation_prompt),
        );
        root.insert("tools".to_string(), tools_to_value(&ctx.tools));
        Self { scopes: vec![root] }
    }

    fn get(&self, name: &str) -> Value {
        self.lookup(name).unwrap_or(Value::None)
    }

    fn lookup(&self, name: &str) -> Option<Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.get(name) {
                return Some(value.clone());
            }
        }
        None
    }

    fn is_defined_name(&self, name: &str) -> bool {
        name == "strftime_now" || self.lookup(name).is_some()
    }

    fn set(&mut self, name: &str, value: Value) {
        if self.scopes.len() == 1 {
            self.scopes[0].insert(name.to_string(), value);
            return;
        }

        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), value);
                return;
            }
        }
        self.scopes
            .last_mut()
            .expect("env has at least one scope")
            .insert(name.to_string(), value);
    }

    fn set_local(&mut self, name: &str, value: Value) {
        self.scopes
            .last_mut()
            .expect("env has at least one scope")
            .insert(name.to_string(), value);
    }

    fn push(&mut self) {
        self.scopes.push(BTreeMap::new());
    }

    fn pop(&mut self) {
        self.scopes.pop();
    }

    fn path_defined(&self, src: &str) -> bool {
        let mut parser = PathParser::new(src);
        let Some(first) = parser.parse_identifier() else {
            return false;
        };
        if !self.is_defined_name(&first) {
            return false;
        }
        if first == "strftime_now" {
            return parser.remaining_is_blank();
        }

        let mut cur = self.get(&first);
        while let Some(op) = parser.parse_path_op() {
            match op {
                PathOp::Member(name) => {
                    let Some(next) = cur.get_member(&name) else {
                        return false;
                    };
                    cur = next;
                }
                PathOp::Index(index) => {
                    let Some(next) = cur.get_index(index) else {
                        return false;
                    };
                    cur = next;
                }
                PathOp::Key(key) => {
                    let Some(next) = cur.get_member(&key) else {
                        return false;
                    };
                    cur = next;
                }
                PathOp::Slice { .. } => {
                    cur = cur.slice(None, None);
                }
            }
        }
        parser.remaining_is_blank()
    }
}

fn messages_to_value(messages: &[Message]) -> Value {
    Value::List(
        messages
            .iter()
            .map(|message| {
                let mut map = BTreeMap::new();
                map.insert("role".to_string(), Value::String(message.role.clone()));
                map.insert("content".to_string(), Value::String(message.content.clone()));
                if !message.tool_calls.is_empty() {
                    map.insert(
                        "tool_calls".to_string(),
                        Value::List(
                            message
                                .tool_calls
                                .iter()
                                .map(tool_call_to_value)
                                .collect(),
                        ),
                    );
                }
                Value::Map(map)
            })
            .collect(),
    )
}

fn tool_call_to_value(call: &ToolCall) -> Value {
    let mut function = BTreeMap::new();
    function.insert("name".to_string(), Value::String(call.name.clone()));
    function.insert(
        "arguments".to_string(),
        parse_json_value(&call.arguments).unwrap_or_else(|| Value::String(call.arguments.clone())),
    );

    let mut map = BTreeMap::new();
    if !call.id.is_empty() {
        map.insert("id".to_string(), Value::String(call.id.clone()));
    }
    map.insert("function".to_string(), Value::Map(function));
    Value::Map(map)
}

fn tools_to_value(tools: &[ToolDef]) -> Value {
    if tools.is_empty() {
        return Value::None;
    }

    Value::List(
        tools
            .iter()
            .map(|tool| {
                let mut map = BTreeMap::new();
                map.insert("name".to_string(), Value::String(tool.name.clone()));
                map.insert(
                    "description".to_string(),
                    Value::String(tool.description.clone()),
                );
                map.insert(
                    "parameters".to_string(),
                    parse_json_value(&tool.parameters_json)
                        .unwrap_or_else(|| Value::Map(BTreeMap::new())),
                );
                Value::Map(map)
            })
            .collect(),
    )
}

// ---------- Template lexer ----------

#[derive(Debug, Clone)]
enum Tok {
    Text(String),
    Expr(String),
    Stmt(String),
}

fn tokenize(src: &str) -> Result<Vec<Tok>> {
    let bytes = src.as_bytes();
    let mut tokens = Vec::new();
    let mut text_start = 0usize;
    let mut i = 0usize;

    while i + 1 < bytes.len() {
        if bytes[i] != b'{' || !matches!(bytes[i + 1], b'{' | b'%' | b'#') {
            i += 1;
            continue;
        }

        let kind = bytes[i + 1];
        let open_trim = i + 2 < bytes.len() && bytes[i + 2] == b'-';
        push_text(&mut tokens, &src[text_start..i], open_trim);

        let body_start = i + if open_trim { 3 } else { 2 };
        let close = match kind {
            b'{' => b"}}",
            b'%' => b"%}",
            b'#' => b"#}",
            _ => unreachable!(),
        };
        let (body_end, close_end, close_trim) = find_tag_close(src, body_start, close)?;
        let body = src[body_start..body_end].trim().to_string();

        match kind {
            b'{' => tokens.push(Tok::Expr(body)),
            b'%' => tokens.push(Tok::Stmt(body)),
            b'#' => {}
            _ => unreachable!(),
        }

        i = close_end;
        if close_trim {
            while i < bytes.len() && matches!(bytes[i], b' ' | b'\t' | b'\n' | b'\r') {
                i += 1;
            }
        }
        text_start = i;
    }

    push_text(&mut tokens, &src[text_start..], false);
    Ok(tokens)
}

fn push_text(tokens: &mut Vec<Tok>, text: &str, trim_end: bool) {
    let text = if trim_end { text.trim_end() } else { text };
    if text.is_empty() {
        return;
    }
    if let Some(Tok::Text(prev)) = tokens.last_mut() {
        prev.push_str(text);
    } else {
        tokens.push(Tok::Text(text.to_string()));
    }
}

fn find_tag_close(src: &str, from: usize, close: &[u8; 2]) -> Result<(usize, usize, bool)> {
    let bytes = src.as_bytes();
    let mut i = from;
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;

    while i + 1 < bytes.len() {
        let b = bytes[i];
        if escaped {
            escaped = false;
            i += 1;
            continue;
        }
        if b == b'\\' && (in_single || in_double) {
            escaped = true;
            i += 1;
            continue;
        }
        if b == b'\'' && !in_double {
            in_single = !in_single;
            i += 1;
            continue;
        }
        if b == b'"' && !in_single {
            in_double = !in_double;
            i += 1;
            continue;
        }
        if !in_single && !in_double {
            if b == b'-' && i + 2 < bytes.len() && bytes[i + 1] == close[0] && bytes[i + 2] == close[1] {
                return Ok((i, i + 3, true));
            }
            if b == close[0] && bytes[i + 1] == close[1] {
                return Ok((i, i + 2, false));
            }
        }
        i += 1;
    }

    Err(FellmError::Tokenization("unterminated template tag".to_string()))
}

// ---------- Renderer ----------

fn render_tokens(tokens: &[Tok], env: &mut Env, out: &mut String, start: usize) -> Result<usize> {
    let mut i = start;
    while i < tokens.len() {
        match &tokens[i] {
            Tok::Text(text) => {
                out.push_str(text);
                i += 1;
            }
            Tok::Expr(body) => {
                let value = eval_expr(body, env)?;
                out.push_str(&value.as_output());
                i += 1;
            }
            Tok::Stmt(body) => {
                let stmt = body.trim();
                if stmt.starts_with("if ") {
                    let end = find_end(tokens, i, "if")?;
                    let branches = split_if(tokens, i, end);
                    for (condition, body_start, body_end) in branches {
                        let selected = if condition == "else" {
                            true
                        } else {
                            eval_bool(condition, env)?
                        };
                        if selected {
                            render_tokens(&tokens[..body_end], env, out, body_start)?;
                            break;
                        }
                    }
                    i = end + 1;
                } else if let Some(rest) = stmt.strip_prefix("for ") {
                    let (var, expr) = parse_for(rest)?;
                    let end = find_end(tokens, i, "for")?;
                    let items = match eval_expr(&expr, env)? {
                        Value::List(items) => items,
                        Value::Map(map) => map.into_values().collect(),
                        Value::String(s) => s.chars().map(|c| Value::String(c.to_string())).collect(),
                        Value::Bool(_) | Value::Int(_) | Value::None => Vec::new(),
                    };
                    render_for_loop(tokens, env, out, i + 1, end, &var, items)?;
                    i = end + 1;
                } else if let Some(rest) = stmt.strip_prefix("set ") {
                    let (name, expr) = parse_set(rest)?;
                    let value = eval_expr(expr, env)?;
                    env.set(name, value);
                    i += 1;
                } else if matches!(stmt, "endif" | "endfor" | "else") || stmt.starts_with("elif ") {
                    return Ok(i);
                } else if stmt.is_empty() {
                    i += 1;
                } else {
                    eval_expr(stmt, env)?;
                    i += 1;
                }
            }
        }
    }
    Ok(i)
}

fn find_end(tokens: &[Tok], from: usize, block: &str) -> Result<usize> {
    let mut depth = 1i32;
    for (idx, token) in tokens.iter().enumerate().skip(from + 1) {
        let Tok::Stmt(stmt) = token else {
            continue;
        };
        let stmt = stmt.trim();
        if stmt.starts_with("if ") || stmt.starts_with("for ") {
            depth += 1;
        } else if stmt == "endif" || stmt == "endfor" {
            depth -= 1;
            if depth == 0 {
                let expected = if block == "if" { "endif" } else { "endfor" };
                if stmt == expected {
                    return Ok(idx);
                }
                return Err(FellmError::Tokenization(format!(
                    "unexpected {stmt}, expected {expected}"
                )));
            }
        }
    }

    Err(FellmError::Tokenization(format!("unterminated {block} block")))
}

fn split_if(tokens: &[Tok], start: usize, end: usize) -> Vec<(&str, usize, usize)> {
    let mut branches = Vec::new();
    let mut condition = match &tokens[start] {
        Tok::Stmt(stmt) => stmt.trim().strip_prefix("if ").unwrap_or("").trim(),
        _ => "",
    };
    let mut body_start = start + 1;
    let mut depth = 0i32;

    for idx in body_start..end {
        let Tok::Stmt(stmt) = &tokens[idx] else {
            continue;
        };
        let stmt = stmt.trim();
        if stmt.starts_with("if ") || stmt.starts_with("for ") {
            depth += 1;
        } else if stmt == "endif" || stmt == "endfor" {
            depth -= 1;
        } else if depth == 0 && (stmt == "else" || stmt.starts_with("elif ")) {
            branches.push((condition, body_start, idx));
            condition = if stmt == "else" {
                "else"
            } else {
                stmt.strip_prefix("elif ").unwrap_or("").trim()
            };
            body_start = idx + 1;
        }
    }

    branches.push((condition, body_start, end));
    branches
}

fn parse_for(rest: &str) -> Result<(String, String)> {
    let Some((var, expr)) = split_top_word(rest, "in") else {
        return Err(FellmError::Tokenization("bad for statement".to_string()));
    };
    let var = var.trim();
    if var.is_empty() {
        return Err(FellmError::Tokenization("missing for-loop variable".to_string()));
    }
    Ok((var.to_string(), expr.trim().to_string()))
}

fn parse_set(rest: &str) -> Result<(&str, &str)> {
    let Some(idx) = find_top_char(rest, '=') else {
        return Err(FellmError::Tokenization("bad set statement".to_string()));
    };
    let name = rest[..idx].trim();
    let expr = rest[idx + 1..].trim();
    if name.is_empty() || expr.is_empty() {
        return Err(FellmError::Tokenization("bad set statement".to_string()));
    }
    Ok((name, expr))
}

fn render_for_loop(
    tokens: &[Tok],
    env: &mut Env,
    out: &mut String,
    body_start: usize,
    body_end: usize,
    var: &str,
    items: Vec<Value>,
) -> Result<()> {
    let len = items.len();
    for (idx, item) in items.into_iter().enumerate() {
        env.push();
        env.set_local(var, item);

        let mut loop_map = BTreeMap::new();
        loop_map.insert("index0".to_string(), Value::Int(idx as i64));
        loop_map.insert("index".to_string(), Value::Int((idx + 1) as i64));
        loop_map.insert("first".to_string(), Value::Bool(idx == 0));
        loop_map.insert("last".to_string(), Value::Bool(idx + 1 == len));
        env.set_local("loop", Value::Map(loop_map));

        render_tokens(&tokens[..body_end], env, out, body_start)?;
        env.pop();
    }
    Ok(())
}

// ---------- Expression evaluator ----------

fn eval_expr(src: &str, env: &Env) -> Result<Value> {
    eval_or(src.trim(), env)
}

fn eval_bool(src: &str, env: &Env) -> Result<bool> {
    Ok(eval_expr(src, env)?.is_truthy())
}

fn eval_or(src: &str, env: &Env) -> Result<Value> {
    if let Some((left, right)) = split_top_word(src, "or") {
        let left_truthy = eval_and(left.trim(), env)?.is_truthy();
        if left_truthy {
            return Ok(Value::Bool(true));
        }
        return Ok(Value::Bool(eval_or(right.trim(), env)?.is_truthy()));
    }
    eval_and(src, env)
}

fn eval_and(src: &str, env: &Env) -> Result<Value> {
    if let Some((left, right)) = split_top_word(src, "and") {
        let left_truthy = eval_not(left.trim(), env)?.is_truthy();
        if !left_truthy {
            return Ok(Value::Bool(false));
        }
        return Ok(Value::Bool(eval_and(right.trim(), env)?.is_truthy()));
    }
    eval_not(src, env)
}

fn eval_not(src: &str, env: &Env) -> Result<Value> {
    let src = src.trim();
    if let Some(rest) = strip_word_prefix(src, "not") {
        return Ok(Value::Bool(!eval_not(rest.trim(), env)?.is_truthy()));
    }
    eval_compare(src, env)
}

fn eval_compare(src: &str, env: &Env) -> Result<Value> {
    if let Some((expr, test)) = split_top_words(src, &["is", "not", "defined"]) {
        let _ = expr;
        return Ok(Value::Bool(!env.path_defined(src[..src.len() - test.len()].trim())));
    }
    if let Some((expr, _test)) = split_top_words(src, &["is", "defined"]) {
        return Ok(Value::Bool(env.path_defined(expr.trim())));
    }
    if let Some((expr, test)) = split_top_words(src, &["is", "not", "none"]) {
        let _ = test;
        return Ok(Value::Bool(!matches!(eval_concat(expr.trim(), env)?, Value::None)));
    }
    if let Some((expr, _test)) = split_top_words(src, &["is", "none"]) {
        return Ok(Value::Bool(matches!(eval_concat(expr.trim(), env)?, Value::None)));
    }
    if let Some((expr, _test)) = split_top_words(src, &["is", "mapping"]) {
        return Ok(Value::Bool(matches!(eval_concat(expr.trim(), env)?, Value::Map(_))));
    }
    if let Some((expr, _test)) = split_top_words(src, &["is", "iterable"]) {
        return Ok(Value::Bool(matches!(
            eval_concat(expr.trim(), env)?,
            Value::String(_) | Value::List(_) | Value::Map(_)
        )));
    }
    if let Some((left, right)) = split_top_word(src, "in") {
        let needle = eval_concat(left.trim(), env)?;
        let haystack = eval_concat(right.trim(), env)?;
        return Ok(Value::Bool(value_contains(&haystack, &needle)));
    }
    if let Some((left, right)) = split_top_op(src, "==") {
        return Ok(Value::Bool(
            eval_concat(left.trim(), env)? == eval_concat(right.trim(), env)?,
        ));
    }
    if let Some((left, right)) = split_top_op(src, "!=") {
        return Ok(Value::Bool(
            eval_concat(left.trim(), env)? != eval_concat(right.trim(), env)?,
        ));
    }
    eval_concat(src, env)
}

fn eval_concat(src: &str, env: &Env) -> Result<Value> {
    let parts = split_all_top_op(src, "+");
    if parts.len() == 1 {
        return eval_filter(parts[0].trim(), env);
    }

    let mut out = String::new();
    for part in parts {
        out.push_str(&eval_filter(part.trim(), env)?.as_output());
    }
    Ok(Value::String(out))
}

fn eval_filter(src: &str, env: &Env) -> Result<Value> {
    let parts = split_all_top_op(src, "|");
    let mut iter = parts.into_iter();
    let Some(base) = iter.next() else {
        return Ok(Value::None);
    };
    let mut value = eval_atom(base.trim(), env)?;

    for filter in iter {
        let filter = filter.trim();
        let (name, args) = parse_filter_call(filter);
        value = match name {
            "trim" => Value::String(value.as_output().trim().to_string()),
            "length" => Value::Int(value_length(&value)),
            "tojson" => {
                let indent = parse_indent_arg(args, env)?;
                Value::String(to_json(&value, indent))
            }
            _ => value,
        };
    }

    Ok(value)
}

fn eval_atom(src: &str, env: &Env) -> Result<Value> {
    let src = src.trim();
    if src.starts_with('(') && src.ends_with(')') && matching_outer_parens(src) {
        return eval_expr(&src[1..src.len() - 1], env);
    }
    let src = strip_balanced_parens(src);
    if src.is_empty() {
        return Ok(Value::None);
    }

    if let Some(s) = parse_string_literal(src) {
        return Ok(Value::String(s));
    }
    if src == "true" {
        return Ok(Value::Bool(true));
    }
    if src == "false" {
        return Ok(Value::Bool(false));
    }
    if src == "none" || src == "None" || src == "null" {
        return Ok(Value::None);
    }
    if let Ok(value) = src.parse::<i64>() {
        return Ok(Value::Int(value));
    }
    if let Some((name, args)) = parse_function_call(src) {
        return eval_function_call(name, args, env);
    }

    Ok(eval_path(src, env))
}

fn eval_function_call(name: &str, args: &str, env: &Env) -> Result<Value> {
    match name {
        "raise_exception" => {
            let message = first_arg(args)
                .map(|arg| eval_expr(arg.trim(), env))
                .transpose()?
                .unwrap_or_else(|| Value::String("template raised exception".to_string()))
                .as_output();
            Err(FellmError::Tokenization(message))
        }
        "strftime_now" => {
            let fmt = first_arg(args)
                .map(|arg| eval_expr(arg.trim(), env))
                .transpose()?
                .unwrap_or_else(|| Value::String("%d %b %Y".to_string()))
                .as_output();
            Ok(Value::String(strftime_now_utc(&fmt)))
        }
        _ => Ok(Value::None),
    }
}

fn eval_path(src: &str, env: &Env) -> Value {
    let mut parser = PathParser::new(src);
    let Some(first) = parser.parse_identifier() else {
        return Value::None;
    };
    let mut cur = env.get(&first);

    while let Some(op) = parser.parse_path_op() {
        cur = match op {
            PathOp::Member(name) | PathOp::Key(name) => cur.get_member(&name).unwrap_or(Value::None),
            PathOp::Index(index) => cur.get_index(index).unwrap_or(Value::None),
            PathOp::Slice { start, end } => cur.slice(start, end),
        };
    }

    if parser.remaining_is_blank() {
        cur
    } else {
        Value::None
    }
}

fn value_contains(haystack: &Value, needle: &Value) -> bool {
    match (haystack, needle) {
        (Value::Map(map), Value::String(key)) => map.contains_key(key),
        (Value::List(items), _) => items.iter().any(|item| item == needle),
        (Value::String(haystack), Value::String(needle)) => haystack.contains(needle),
        _ => false,
    }
}

fn value_length(value: &Value) -> i64 {
    match value {
        Value::String(s) => i64::try_from(s.chars().count()).unwrap_or(0),
        Value::List(v) => i64::try_from(v.len()).unwrap_or(0),
        Value::Map(m) => i64::try_from(m.len()).unwrap_or(0),
        Value::Bool(_) | Value::Int(_) | Value::None => 0,
    }
}

// ---------- Path parsing ----------

#[derive(Debug)]
enum PathOp {
    Member(String),
    Key(String),
    Index(i64),
    Slice { start: Option<i64>, end: Option<i64> },
}

struct PathParser<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> PathParser<'a> {
    fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    fn parse_identifier(&mut self) -> Option<String> {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.src.len() {
            let b = self.src.as_bytes()[self.pos];
            if b.is_ascii_alphanumeric() || b == b'_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        (self.pos > start).then(|| self.src[start..self.pos].to_string())
    }

    fn parse_path_op(&mut self) -> Option<PathOp> {
        self.skip_ws();
        let b = *self.src.as_bytes().get(self.pos)?;
        match b {
            b'.' => {
                self.pos += 1;
                self.parse_identifier().map(PathOp::Member)
            }
            b'[' => self.parse_bracket(),
            _ => None,
        }
    }

    fn parse_bracket(&mut self) -> Option<PathOp> {
        self.pos += 1;
        let start = self.pos;
        let mut in_single = false;
        let mut in_double = false;
        let mut escaped = false;
        while self.pos < self.src.len() {
            let b = self.src.as_bytes()[self.pos];
            if escaped {
                escaped = false;
                self.pos += 1;
                continue;
            }
            if b == b'\\' && (in_single || in_double) {
                escaped = true;
                self.pos += 1;
                continue;
            }
            if b == b'\'' && !in_double {
                in_single = !in_single;
            } else if b == b'"' && !in_single {
                in_double = !in_double;
            } else if b == b']' && !in_single && !in_double {
                let inner = self.src[start..self.pos].trim();
                self.pos += 1;
                return parse_bracket_inner(inner);
            }
            self.pos += 1;
        }
        None
    }

    fn remaining_is_blank(&mut self) -> bool {
        self.skip_ws();
        self.pos == self.src.len()
    }

    fn skip_ws(&mut self) {
        while self
            .src
            .as_bytes()
            .get(self.pos)
            .is_some_and(u8::is_ascii_whitespace)
        {
            self.pos += 1;
        }
    }
}

fn parse_bracket_inner(inner: &str) -> Option<PathOp> {
    if let Some(key) = parse_string_literal(inner) {
        return Some(PathOp::Key(key));
    }
    if let Some((start, end)) = inner.split_once(':') {
        let start = parse_optional_i64(start.trim())?;
        let end = parse_optional_i64(end.trim())?;
        return Some(PathOp::Slice { start, end });
    }
    inner.parse::<i64>().ok().map(PathOp::Index)
}

fn parse_optional_i64(src: &str) -> Option<Option<i64>> {
    if src.is_empty() {
        Some(None)
    } else {
        src.parse::<i64>().ok().map(Some)
    }
}

fn normalize_slice_bound(bound: i64, len: i64) -> i64 {
    let bound = if bound < 0 { len + bound } else { bound };
    bound.clamp(0, len)
}

// ---------- Expression string helpers ----------

fn split_top_words<'a>(src: &'a str, words: &[&str]) -> Option<(&'a str, &'a str)> {
    let mut spans = top_level_word_spans(src);
    let needed = words.len();
    if spans.len() < needed {
        return None;
    }
    spans.reverse();
    for window in spans.windows(needed) {
        let mut matched = true;
        for (idx, expected) in words.iter().rev().enumerate() {
            if &src[window[idx].0..window[idx].1] != *expected {
                matched = false;
                break;
            }
        }
        if matched {
            let first = window[needed - 1].0;
            return Some((&src[..first], &src[first..]));
        }
    }
    None
}

fn split_top_word<'a>(src: &'a str, word: &str) -> Option<(&'a str, &'a str)> {
    for (start, end) in top_level_word_spans(src) {
        if &src[start..end] == word {
            return Some((&src[..start], &src[end..]));
        }
    }
    None
}

fn top_level_word_spans(src: &str) -> Vec<(usize, usize)> {
    let bytes = src.as_bytes();
    let mut spans = Vec::new();
    let mut i = 0usize;
    let mut depth = 0i32;
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;

    while i < bytes.len() {
        let b = bytes[i];
        if escaped {
            escaped = false;
            i += 1;
            continue;
        }
        if b == b'\\' && (in_single || in_double) {
            escaped = true;
            i += 1;
            continue;
        }
        if b == b'\'' && !in_double {
            in_single = !in_single;
            i += 1;
            continue;
        }
        if b == b'"' && !in_single {
            in_double = !in_double;
            i += 1;
            continue;
        }
        if !in_single && !in_double {
            match b {
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => depth -= 1,
                _ if depth == 0 && (b.is_ascii_alphabetic() || b == b'_') => {
                    let start = i;
                    i += 1;
                    while i < bytes.len() {
                        let c = bytes[i];
                        if c.is_ascii_alphanumeric() || c == b'_' {
                            i += 1;
                        } else {
                            break;
                        }
                    }
                    spans.push((start, i));
                    continue;
                }
                _ => {}
            }
        }
        i += 1;
    }

    spans
}

fn split_top_op<'a>(src: &'a str, op: &str) -> Option<(&'a str, &'a str)> {
    let bytes = src.as_bytes();
    let op_bytes = op.as_bytes();
    let mut i = 0usize;
    let mut depth = 0i32;
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;

    while i + op_bytes.len() <= bytes.len() {
        let b = bytes[i];
        if escaped {
            escaped = false;
            i += 1;
            continue;
        }
        if b == b'\\' && (in_single || in_double) {
            escaped = true;
            i += 1;
            continue;
        }
        if b == b'\'' && !in_double {
            in_single = !in_single;
            i += 1;
            continue;
        }
        if b == b'"' && !in_single {
            in_double = !in_double;
            i += 1;
            continue;
        }
        if !in_single && !in_double {
            match b {
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => depth -= 1,
                _ => {
                    if depth == 0 && &bytes[i..i + op_bytes.len()] == op_bytes {
                        return Some((&src[..i], &src[i + op_bytes.len()..]));
                    }
                }
            }
        }
        i += 1;
    }

    None
}

fn split_all_top_op<'a>(src: &'a str, op: &str) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut rest = src;
    while let Some((left, right)) = split_top_op(rest, op) {
        out.push(left);
        rest = right;
    }
    out.push(rest);
    out
}

fn find_top_char(src: &str, needle: char) -> Option<usize> {
    let needle = needle as u8;
    let bytes = src.as_bytes();
    let mut depth = 0i32;
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;

    for (idx, b) in bytes.iter().copied().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        if b == b'\\' && (in_single || in_double) {
            escaped = true;
            continue;
        }
        if b == b'\'' && !in_double {
            in_single = !in_single;
            continue;
        }
        if b == b'"' && !in_single {
            in_double = !in_double;
            continue;
        }
        if in_single || in_double {
            continue;
        }
        if depth == 0 && b == needle {
            return Some(idx);
        }
        match b {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            _ => {}
        }
    }

    None
}

fn strip_word_prefix<'a>(src: &'a str, word: &str) -> Option<&'a str> {
    let rest = src.strip_prefix(word)?;
    if rest
        .as_bytes()
        .first()
        .is_some_and(|b| b.is_ascii_whitespace())
    {
        Some(rest)
    } else {
        None
    }
}

fn strip_balanced_parens(mut src: &str) -> &str {
    loop {
        let s = src.trim();
        if !(s.starts_with('(') && s.ends_with(')')) {
            return s;
        }
        if matching_outer_parens(s) {
            src = &s[1..s.len() - 1];
        } else {
            return s;
        }
    }
}

fn matching_outer_parens(src: &str) -> bool {
    let bytes = src.as_bytes();
    let mut depth = 0i32;
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;

    for (idx, b) in bytes.iter().copied().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        if b == b'\\' && (in_single || in_double) {
            escaped = true;
            continue;
        }
        if b == b'\'' && !in_double {
            in_single = !in_single;
            continue;
        }
        if b == b'"' && !in_single {
            in_double = !in_double;
            continue;
        }
        if in_single || in_double {
            continue;
        }
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 && idx != bytes.len() - 1 {
                    return false;
                }
            }
            _ => {}
        }
    }
    depth == 0
}

fn parse_string_literal(src: &str) -> Option<String> {
    let bytes = src.as_bytes();
    if bytes.len() < 2 {
        return None;
    }
    let quote = bytes[0];
    if !matches!(quote, b'\'' | b'"') || bytes[bytes.len() - 1] != quote {
        return None;
    }
    unescape_string(&src[1..src.len() - 1]).ok()
}

fn parse_function_call(src: &str) -> Option<(&str, &str)> {
    let open = find_top_char(src, '(')?;
    if !src.ends_with(')') || !matching_outer_call(src, open) {
        return None;
    }
    let name = src[..open].trim();
    if name.is_empty()
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        return None;
    }
    Some((name, &src[open + 1..src.len() - 1]))
}

fn matching_outer_call(src: &str, open: usize) -> bool {
    let call = &src[open..];
    matching_outer_parens(call)
}

fn first_arg(args: &str) -> Option<&str> {
    split_args(args).into_iter().next()
}

fn split_args(args: &str) -> Vec<&str> {
    split_all_top_op(args, ",")
        .into_iter()
        .map(str::trim)
        .filter(|arg| !arg.is_empty())
        .collect()
}

fn parse_filter_call(filter: &str) -> (&str, Option<&str>) {
    if let Some((name, args)) = parse_function_call(filter) {
        (name.trim(), Some(args))
    } else {
        (filter.split_whitespace().next().unwrap_or(filter), None)
    }
}

fn parse_indent_arg(args: Option<&str>, env: &Env) -> Result<Option<usize>> {
    let Some(args) = args else {
        return Ok(None);
    };
    for arg in split_args(args) {
        let expr = arg
            .trim()
            .strip_prefix("indent")
            .and_then(|rest| rest.trim().strip_prefix('='))
            .map(str::trim)
            .unwrap_or(arg.trim());
        if expr.is_empty() {
            continue;
        }
        if let Value::Int(indent) = eval_expr(expr, env)? {
            return Ok(usize::try_from(indent).ok());
        }
    }
    Ok(None)
}

fn unescape_string(src: &str) -> Result<String> {
    let mut out = String::new();
    let mut chars = src.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        let Some(escaped) = chars.next() else {
            out.push('\\');
            break;
        };
        match escaped {
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            't' => out.push('\t'),
            '\\' => out.push('\\'),
            '\'' => out.push('\''),
            '"' => out.push('"'),
            'u' => {
                let mut hex = String::new();
                for _ in 0..4 {
                    let Some(h) = chars.next() else {
                        return Err(FellmError::Tokenization("bad unicode escape".to_string()));
                    };
                    hex.push(h);
                }
                let code = u32::from_str_radix(&hex, 16)
                    .map_err(|_| FellmError::Tokenization("bad unicode escape".to_string()))?;
                let ch = char::from_u32(code)
                    .ok_or_else(|| FellmError::Tokenization("bad unicode escape".to_string()))?;
                out.push(ch);
            }
            other => out.push(other),
        }
    }
    Ok(out)
}

// ---------- Date helper ----------

fn strftime_now_utc(fmt: &str) -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs());
    let days = i64::try_from(seconds / 86_400).unwrap_or(0);
    let (year, month, day) = civil_from_days(days);
    let month_name = [
        "", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ][month as usize];

    let mut out = String::new();
    let mut chars = fmt.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('d') => out.push_str(&day.to_string()),
            Some('m') => out.push_str(&format!("{month:02}")),
            Some('b') => out.push_str(month_name),
            Some('Y') => out.push_str(&year.to_string()),
            Some('%') => out.push('%'),
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
}

fn civil_from_days(days_since_epoch: i64) -> (i32, u32, u32) {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    let year = y + if m <= 2 { 1 } else { 0 };
    (year as i32, m as u32, d as u32)
}

// ---------- JSON helpers ----------

fn parse_json_value(src: &str) -> Option<Value> {
    let mut parser = JsonParser::new(src);
    let value = parser.parse_value().ok()?;
    parser.skip_ws();
    (parser.pos == parser.src.len()).then_some(value)
}

struct JsonParser<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    fn parse_value(&mut self) -> Result<Value> {
        self.skip_ws();
        match self.peek() {
            Some(b'"') => self.parse_string().map(Value::String),
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b't') => {
                self.expect_literal("true")?;
                Ok(Value::Bool(true))
            }
            Some(b'f') => {
                self.expect_literal("false")?;
                Ok(Value::Bool(false))
            }
            Some(b'n') => {
                self.expect_literal("null")?;
                Ok(Value::None)
            }
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            _ => Err(FellmError::Tokenization("bad JSON value".to_string())),
        }
    }

    fn parse_object(&mut self) -> Result<Value> {
        self.expect(b'{')?;
        let mut map = BTreeMap::new();
        loop {
            self.skip_ws();
            if self.consume(b'}') {
                break;
            }
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect(b':')?;
            let value = self.parse_value()?;
            map.insert(key, value);
            self.skip_ws();
            if self.consume(b'}') {
                break;
            }
            self.expect(b',')?;
        }
        Ok(Value::Map(map))
    }

    fn parse_array(&mut self) -> Result<Value> {
        self.expect(b'[')?;
        let mut list = Vec::new();
        loop {
            self.skip_ws();
            if self.consume(b']') {
                break;
            }
            list.push(self.parse_value()?);
            self.skip_ws();
            if self.consume(b']') {
                break;
            }
            self.expect(b',')?;
        }
        Ok(Value::List(list))
    }

    fn parse_string(&mut self) -> Result<String> {
        self.expect(b'"')?;
        let start = self.pos;
        let mut escaped = false;
        while self.pos < self.src.len() {
            let b = self.src.as_bytes()[self.pos];
            if escaped {
                escaped = false;
                self.pos += 1;
                continue;
            }
            if b == b'\\' {
                escaped = true;
                self.pos += 1;
                continue;
            }
            if b == b'"' {
                let raw = &self.src[start..self.pos];
                self.pos += 1;
                return unescape_string(raw);
            }
            self.pos += 1;
        }
        Err(FellmError::Tokenization("unterminated JSON string".to_string()))
    }

    fn parse_number(&mut self) -> Result<Value> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while self.peek().is_some_and(|b| b.is_ascii_digit()) {
            self.pos += 1;
        }
        if self.peek() == Some(b'.') {
            while self
                .peek()
                .is_some_and(|b| b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E' | b'+' | b'-'))
            {
                self.pos += 1;
            }
            return Ok(Value::String(self.src[start..self.pos].to_string()));
        }
        self.src[start..self.pos]
            .parse::<i64>()
            .map(Value::Int)
            .map_err(|_| FellmError::Tokenization("bad JSON number".to_string()))
    }

    fn expect_literal(&mut self, literal: &str) -> Result<()> {
        if self.src[self.pos..].starts_with(literal) {
            self.pos += literal.len();
            Ok(())
        } else {
            Err(FellmError::Tokenization("bad JSON literal".to_string()))
        }
    }

    fn expect(&mut self, byte: u8) -> Result<()> {
        if self.consume(byte) {
            Ok(())
        } else {
            Err(FellmError::Tokenization("bad JSON syntax".to_string()))
        }
    }

    fn consume(&mut self, byte: u8) -> bool {
        if self.peek() == Some(byte) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.as_bytes().get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while self.peek().is_some_and(|b| b.is_ascii_whitespace()) {
            self.pos += 1;
        }
    }
}

fn to_json(value: &Value, indent: Option<usize>) -> String {
    let mut out = String::new();
    write_json(value, &mut out, indent, 0);
    out
}

fn write_json(value: &Value, out: &mut String, indent: Option<usize>, depth: usize) {
    match value {
        Value::String(s) => write_json_string(s, out),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Int(i) => out.push_str(&i.to_string()),
        Value::None => out.push_str("null"),
        Value::List(list) => write_json_list(list, out, indent, depth),
        Value::Map(map) => write_json_map(map, out, indent, depth),
    }
}

fn write_json_list(list: &[Value], out: &mut String, indent: Option<usize>, depth: usize) {
    out.push('[');
    for (idx, value) in list.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        if let Some(spaces) = indent {
            out.push('\n');
            out.push_str(&" ".repeat((depth + 1) * spaces));
        }
        write_json(value, out, indent, depth + 1);
    }
    if let Some(spaces) = indent {
        if !list.is_empty() {
            out.push('\n');
            out.push_str(&" ".repeat(depth * spaces));
        }
    }
    out.push(']');
}

fn write_json_map(
    map: &BTreeMap<String, Value>,
    out: &mut String,
    indent: Option<usize>,
    depth: usize,
) {
    out.push('{');
    for (idx, (key, value)) in map.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        if let Some(spaces) = indent {
            out.push('\n');
            out.push_str(&" ".repeat((depth + 1) * spaces));
        }
        write_json_string(key, out);
        out.push(':');
        if indent.is_some() {
            out.push(' ');
        }
        write_json(value, out, indent, depth + 1);
    }
    if let Some(spaces) = indent {
        if !map.is_empty() {
            out.push('\n');
            out.push_str(&" ".repeat(depth * spaces));
        }
    }
    out.push('}');
}

fn write_json_string(src: &str, out: &mut String) {
    out.push('"');
    for c in src.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    const LLAMA_3_2_TOOL_TEMPLATE: &str = r#"{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- if strftime_now is defined %}
        {%- set date_string = strftime_now("%d %b %Y") %}
    {%- else %}
        {%- set date_string = "26 Jul 2024" %}
    {%- endif %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
        {{- '{"name": "' + tool_call.name + '", ' }}
        {{- '"parameters": ' }}
        {{- tool_call.arguments | tojson }}
        {{- "}" }}
        {{- "<|eot_id|>" }}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"#;

    fn vars() -> BTreeMap<String, Value> {
        let mut vars = BTreeMap::new();
        vars.insert(
            "bos_token".to_string(),
            Value::String("<|begin_of_text|>".to_string()),
        );
        vars
    }

    #[test]
    fn renders_exact_llama_3_2_template_without_tools() {
        let ctx = TemplateContext {
            messages: vec![Message::text("user", "who are you")],
            add_generation_prompt: true,
            vars: vars(),
            tools: Vec::new(),
        };

        let out = render(LLAMA_3_2_TOOL_TEMPLATE, &ctx).unwrap();
        assert!(!out.contains("{#"));
        assert!(out.contains("who are you"));
        assert!(out.contains("<|start_header_id|>assistant<|end_header_id|>"));
        assert!(!out.contains("Given the following functions"));
    }

    #[test]
    fn renders_exact_llama_3_2_template_with_tools() {
        let ctx = TemplateContext {
            messages: vec![Message::text("user", "What is the weather in SF?")],
            add_generation_prompt: true,
            vars: vars(),
            tools: vec![ToolDef {
                name: "get_weather".to_string(),
                description: "Get the weather for a city".to_string(),
                parameters_json:
                    r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#
                        .to_string(),
            }],
        };

        let out = render(LLAMA_3_2_TOOL_TEMPLATE, &ctx).unwrap();
        assert!(out.contains("get_weather"));
        assert!(out.contains("Environment: ipython"));
        assert!(out.contains("Given the following functions"));
        assert!(out.contains("What is the weather in SF?"));
    }

    #[test]
    fn renders_concat_and_filters() {
        let ctx = TemplateContext {
            messages: vec![Message::text("user", "  hi  ")],
            add_generation_prompt: false,
            vars: BTreeMap::new(),
            tools: Vec::new(),
        };

        let out = render("{{ 'A' + messages[0].content | trim + 'B' }} {{ messages|length }}", &ctx)
            .unwrap();
        assert_eq!(out, "AhiB 1");
    }
}
