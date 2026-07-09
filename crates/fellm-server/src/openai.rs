//! OpenAI request/response mapping helpers.

use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestAssistantMessageContentPart,
    ChatCompletionRequestDeveloperMessageContent, ChatCompletionRequestDeveloperMessageContentPart,
    ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestSystemMessageContentPart,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestToolMessageContentPart,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    ChatCompletionTool, ChatCompletionTools, CreateChatCompletionRequest, FunctionObject,
};
use fellm_runtime::{GenParams, GenStats, Message, ToolCall, ToolDef};
use serde::Serialize;
use serde_json::{Value, json};
use std::time::{SystemTime, UNIX_EPOCH};

/// Map an OpenAI chat completion request into engine inputs.
pub fn map_request(
    req: &CreateChatCompletionRequest,
    defaults: GenParams,
) -> Result<(Vec<Message>, Vec<ToolDef>, GenParams, bool), String> {
    let messages = map_messages(&req.messages)?;
    let tools = map_tools(req.tools.as_deref().unwrap_or(&[]))?;
    let params = map_params(req, defaults);
    let stream = req.stream.unwrap_or(false);
    Ok((messages, tools, params, stream))
}

fn map_params(req: &CreateChatCompletionRequest, mut defaults: GenParams) -> GenParams {
    #[allow(deprecated)]
    if let Some(n) = req.max_completion_tokens.or(req.max_tokens) {
        defaults.max_tokens = n;
    }
    if let Some(t) = req.temperature {
        defaults.temperature = t;
    }
    if let Some(p) = req.top_p {
        defaults.top_p = p;
    }
    #[allow(deprecated)]
    if let Some(seed) = req.seed {
        defaults.seed = u64::try_from(seed).unwrap_or(0);
    }
    defaults
}

fn map_tools(tools: &[ChatCompletionTools]) -> Result<Vec<ToolDef>, String> {
    let mut out = Vec::with_capacity(tools.len());
    for tool in tools {
        match tool {
            ChatCompletionTools::Function(ChatCompletionTool { function }) => {
                out.push(function_to_tool_def(function)?);
            }
            ChatCompletionTools::Custom(_) => {
                return Err("custom tools are not supported".into());
            }
        }
    }
    Ok(out)
}

fn function_to_tool_def(f: &FunctionObject) -> Result<ToolDef, String> {
    let parameters_json = match &f.parameters {
        Some(v) => serde_json::to_string(v).map_err(|e| e.to_string())?,
        None => "{}".to_string(),
    };
    Ok(ToolDef {
        name: f.name.clone(),
        description: f.description.clone().unwrap_or_default(),
        parameters_json,
    })
}

fn map_messages(messages: &[ChatCompletionRequestMessage]) -> Result<Vec<Message>, String> {
    messages.iter().map(map_message).collect()
}

fn map_message(msg: &ChatCompletionRequestMessage) -> Result<Message, String> {
    match msg {
        ChatCompletionRequestMessage::System(m) => {
            Ok(Message::text("system", system_content_text(&m.content)?))
        }
        ChatCompletionRequestMessage::Developer(m) => {
            Ok(Message::text("system", developer_content_text(&m.content)?))
        }
        ChatCompletionRequestMessage::User(m) => {
            Ok(Message::text("user", user_content_text(&m.content)?))
        }
        ChatCompletionRequestMessage::Assistant(m) => {
            let tool_calls = m
                .tool_calls
                .as_ref()
                .map(|calls| {
                    calls
                        .iter()
                        .map(map_openai_tool_call)
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?
                .unwrap_or_default();

            if tool_calls.is_empty() {
                let content = m
                    .content
                    .as_ref()
                    .map(assistant_content_text)
                    .transpose()?
                    .unwrap_or_default();
                Ok(Message::text("assistant", content))
            } else {
                Ok(Message::assistant_tools(tool_calls))
            }
        }
        ChatCompletionRequestMessage::Tool(m) => {
            Ok(Message::tool_result(tool_content_text(&m.content)?))
        }
        ChatCompletionRequestMessage::Function(m) => {
            Ok(Message::tool_result(m.content.clone().unwrap_or_default()))
        }
    }
}

fn map_openai_tool_call(call: &ChatCompletionMessageToolCalls) -> Result<ToolCall, String> {
    match call {
        ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
            id,
            function,
        }) => Ok(ToolCall {
            id: id.clone(),
            name: function.name.clone(),
            arguments: function.arguments.clone(),
        }),
        ChatCompletionMessageToolCalls::Custom(_) => {
            Err("custom tool calls are not supported".into())
        }
    }
}

fn text_parts(parts: &[ChatCompletionRequestMessageContentPartText]) -> String {
    parts.iter().map(|p| p.text.as_str()).collect()
}

fn system_content_text(
    content: &ChatCompletionRequestSystemMessageContent,
) -> Result<String, String> {
    match content {
        ChatCompletionRequestSystemMessageContent::Text(s) => Ok(s.clone()),
        ChatCompletionRequestSystemMessageContent::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                match part {
                    ChatCompletionRequestSystemMessageContentPart::Text(t) => {
                        out.push_str(&t.text);
                    }
                }
            }
            Ok(out)
        }
    }
}

fn developer_content_text(
    content: &ChatCompletionRequestDeveloperMessageContent,
) -> Result<String, String> {
    match content {
        ChatCompletionRequestDeveloperMessageContent::Text(s) => Ok(s.clone()),
        ChatCompletionRequestDeveloperMessageContent::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                match part {
                    ChatCompletionRequestDeveloperMessageContentPart::Text(t) => {
                        out.push_str(&t.text);
                    }
                }
            }
            Ok(out)
        }
    }
}

fn user_content_text(content: &ChatCompletionRequestUserMessageContent) -> Result<String, String> {
    match content {
        ChatCompletionRequestUserMessageContent::Text(s) => Ok(s.clone()),
        ChatCompletionRequestUserMessageContent::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                match part {
                    ChatCompletionRequestUserMessageContentPart::Text(t) => {
                        out.push_str(&t.text);
                    }
                    _ => return Err("only text user content is supported".into()),
                }
            }
            Ok(out)
        }
    }
}

fn assistant_content_text(
    content: &ChatCompletionRequestAssistantMessageContent,
) -> Result<String, String> {
    match content {
        ChatCompletionRequestAssistantMessageContent::Text(s) => Ok(s.clone()),
        ChatCompletionRequestAssistantMessageContent::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                match part {
                    ChatCompletionRequestAssistantMessageContentPart::Text(t) => {
                        out.push_str(&t.text);
                    }
                    ChatCompletionRequestAssistantMessageContentPart::Refusal(r) => {
                        out.push_str(&r.refusal);
                    }
                }
            }
            Ok(out)
        }
    }
}

fn tool_content_text(content: &ChatCompletionRequestToolMessageContent) -> Result<String, String> {
    match content {
        ChatCompletionRequestToolMessageContent::Text(s) => Ok(s.clone()),
        ChatCompletionRequestToolMessageContent::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                match part {
                    ChatCompletionRequestToolMessageContentPart::Text(t) => {
                        out.push_str(&t.text);
                    }
                }
            }
            Ok(out)
        }
    }
}

/// Build a non-streaming chat completion response JSON value.
pub fn build_completion_response(
    model: &str,
    finish_reason: &str,
    full_text: &str,
    tool_calls: Option<&[ToolCall]>,
    usage: &GenStats,
) -> Value {
    let id = completion_id();
    let created = unix_now();

    let message = if let Some(calls) = tool_calls {
        json!({
            "role": "assistant",
            "content": null,
            "tool_calls": openai_tool_calls(calls),
        })
    } else {
        json!({
            "role": "assistant",
            "content": full_text,
        })
    };

    json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
            "logprobs": null,
        }],
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.predicted_tokens,
            "total_tokens": usage.prompt_tokens.saturating_add(usage.predicted_tokens),
        },
    })
}

/// Streaming chunk with content delta.
pub fn stream_content_chunk(id: &str, created: u64, model: &str, content: &str) -> String {
    let v = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": { "content": content },
            "finish_reason": null,
        }],
    });
    format!("data: {v}\n\n")
}

/// First streaming chunk with role.
pub fn stream_role_chunk(id: &str, created: u64, model: &str) -> String {
    let v = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": { "role": "assistant", "content": "" },
            "finish_reason": null,
        }],
    });
    format!("data: {v}\n\n")
}

/// Final streaming chunk with finish_reason and optional tool_calls.
pub fn stream_final_chunk(
    id: &str,
    created: u64,
    model: &str,
    finish_reason: &str,
    tool_calls: Option<&[ToolCall]>,
) -> String {
    let delta = if let Some(calls) = tool_calls {
        json!({ "tool_calls": openai_tool_calls_delta(calls) })
    } else {
        json!({})
    };
    let v = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    });
    format!("data: {v}\n\n")
}

fn openai_tool_calls(calls: &[ToolCall]) -> Vec<Value> {
    calls
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let id = if c.id.is_empty() {
                format!("call_{i}")
            } else {
                c.id.clone()
            };
            json!({
                "id": id,
                "type": "function",
                "function": {
                    "name": c.name,
                    "arguments": c.arguments,
                }
            })
        })
        .collect()
}

fn openai_tool_calls_delta(calls: &[ToolCall]) -> Vec<Value> {
    calls
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let id = if c.id.is_empty() {
                format!("call_{i}")
            } else {
                c.id.clone()
            };
            json!({
                "index": i,
                "id": id,
                "type": "function",
                "function": {
                    "name": c.name,
                    "arguments": c.arguments,
                }
            })
        })
        .collect()
}

/// Models list response.
#[derive(Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelCard>,
}

/// Single model card.
#[derive(Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

pub fn models_response(model_id: &str) -> ModelsResponse {
    ModelsResponse {
        object: "list",
        data: vec![ModelCard {
            id: model_id.to_string(),
            object: "model",
            created: unix_now(),
            owned_by: "fellm",
        }],
    }
}

pub fn completion_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("chatcmpl-{nanos}")
}

pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[allow(dead_code)]
fn _text_parts_anchor(parts: &[ChatCompletionRequestMessageContentPartText]) -> String {
    text_parts(parts)
}
