//! Chat messages and HuggingFace-style Jinja chat-template rendering via MiniJinja.

use chrono::Local;
use fellm_core::error::{FellmError, Result};
use minijinja::{Environment, ErrorKind, context};
use minijinja_contrib::pycompat;
use regex::regex;
use serde::Serialize;
use serde_json::Value as JsonValue;

/// A message in a chat conversation.
#[derive(Debug, Clone, Serialize)]
pub struct Message {
    /// Role: "system", "user", "assistant", "tool" / "ipython".
    pub role: String,
    /// Content text (tool results go here for role=tool/ipython).
    pub content: String,
    /// Assistant tool calls (when the model requested tools).
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
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

impl Serialize for ToolCall {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;
        let args: JsonValue = serde_json::from_str(&self.arguments)
            .unwrap_or_else(|_| JsonValue::Object(serde_json::Map::new()));
        let mut map = serializer.serialize_map(Some(2))?;
        if !self.id.is_empty() {
            map.serialize_entry("id", &self.id)?;
        }
        map.serialize_entry(
            "function",
            &serde_json::json!({
                "name": self.name,
                "arguments": args,
            }),
        )?;
        map.end()
    }
}

/// Result of parsing an assistant turn (text vs tool calls).
#[derive(Debug, Clone)]
pub enum AssistantOutput {
    /// Normal text reply.
    Text(String),
    /// One or more tool invocations.
    ToolCalls(Vec<ToolCall>),
}

#[derive(Serialize)]
struct ToolForTemplate {
    name: String,
    description: String,
    parameters: JsonValue,
}

fn raise_exception(err_text: String) -> std::result::Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::SyntaxError, err_text))
}

fn strftime_now(format_str: String) -> std::result::Result<String, minijinja::Error> {
    Ok(Local::now().format(&format_str).to_string())
}

/// Normalize HF chat templates for MiniJinja (same approach as TGI).
fn prepare_template(source: &str) -> String {
    let mut s = source.to_string();
    // Python reverse slice → MiniJinja filter.
    s = s.replace("[::-1]", "|reverse");
    // `{% generation %}` is training-only assistant masking; strip for inference.
    s = regex!(r"\{%-?\s*generation\s*-?%\}")
        .replace_all(&s, "")
        .into_owned();
    s = regex!(r"\{%-?\s*endgeneration\s*-?%\}")
        .replace_all(&s, "")
        .into_owned();
    s
}

fn make_env() -> Environment<'static> {
    let mut env = Environment::new();
    env.set_unknown_method_callback(pycompat::unknown_method_callback);
    env.add_function("raise_exception", raise_exception);
    env.add_function("strftime_now", strftime_now);
    env
}

fn messages_for_template(messages: &[Message]) -> Result<Vec<JsonValue>> {
    messages
        .iter()
        .map(|m| {
            let mut map = serde_json::Map::new();
            map.insert("role".into(), JsonValue::String(m.role.clone()));
            map.insert("content".into(), JsonValue::String(m.content.clone()));
            if !m.tool_calls.is_empty() {
                map.insert(
                    "tool_calls".into(),
                    serde_json::to_value(&m.tool_calls).map_err(|e| {
                        FellmError::Tokenization(format!("tool_calls serialize: {e}"))
                    })?,
                );
            }
            Ok(JsonValue::Object(map))
        })
        .collect()
}

fn tools_for_template(tools: &[ToolDef]) -> Result<Vec<ToolForTemplate>> {
    tools
        .iter()
        .map(|t| {
            let parameters: JsonValue = if t.parameters_json.trim().is_empty() {
                JsonValue::Object(serde_json::Map::new())
            } else {
                serde_json::from_str(&t.parameters_json)
                    .map_err(|e| FellmError::Tokenization(format!("tool parameters JSON: {e}")))?
            };
            Ok(ToolForTemplate {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters,
            })
        })
        .collect()
}

/// Render a GGUF / HuggingFace chat template with MiniJinja.
pub fn render_chat_template(
    template_src: &str,
    messages: &[Message],
    tools: &[ToolDef],
    add_generation_prompt: bool,
    bos_token: Option<&str>,
    eos_token: Option<&str>,
) -> Result<String> {
    let prepared = prepare_template(template_src);
    let env = make_env();
    let tmpl = env
        .template_from_str(&prepared)
        .map_err(|e| FellmError::Tokenization(format!("chat template parse: {e}")))?;

    let msgs = messages_for_template(messages)?;
    let tool_vals = tools_for_template(tools)?;
    let tools_opt = if tool_vals.is_empty() {
        None
    } else {
        Some(tool_vals)
    };

    let rendered = tmpl
        .render(context! {
            messages => msgs,
            add_generation_prompt => add_generation_prompt,
            bos_token => bos_token.unwrap_or(""),
            eos_token => eos_token.unwrap_or(""),
            tools => tools_opt,
        })
        .map_err(|e| FellmError::Tokenization(format!("chat template render: {e}")))?;
    Ok(rendered)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_simple_template() {
        let tmpl = "{% for message in messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.first %}{{ bos_token + content }}{% else %}{{ content }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}";
        let out = render_chat_template(
            tmpl,
            &[Message::text("user", "who are you")],
            &[],
            true,
            Some("<|begin_of_text|>"),
            Some("<|eot_id|>"),
        )
        .unwrap();
        assert!(out.contains("<|begin_of_text|>"));
        assert!(out.contains("who are you"));
        assert!(out.contains("<|start_header_id|>assistant<|end_header_id|>"));
    }

    #[test]
    fn strips_generation_and_supports_namespace() {
        let tmpl = r#"{%- set ns = namespace(x="hi") -%}{%- generation -%}{{ ns.x }}{%- endgeneration -%}"#;
        let out = render_chat_template(tmpl, &[], &[], false, None, None).unwrap();
        assert_eq!(out, "hi");
    }
}
