use cwc_core::traits::TokenCounter;

use crate::error::{Result, SessionError};
use crate::message::{MessageFlags, SessionMessage, SessionRole, ToolCall, ToolResult};

/// Convert OpenAI-style messages to SessionMessages.
///
/// Handles:
/// - { role: "system"|"user"|"assistant"|"tool", content: "..." }
/// - assistant messages with tool_calls array
/// - tool messages with tool_call_id
pub fn from_openai(
    messages: &[serde_json::Value],
    tokenizer: &dyn TokenCounter,
) -> Result<Vec<SessionMessage>> {
    let mut result = Vec::with_capacity(messages.len());

    for (i, msg) in messages.iter().enumerate() {
        let role_str = msg["role"]
            .as_str()
            .ok_or_else(|| SessionError::MissingField(format!("messages[{i}].role")))?;
        let role = match role_str {
            "system" => SessionRole::System,
            "user" => SessionRole::User,
            "assistant" => SessionRole::Assistant,
            "tool" => SessionRole::Tool,
            other => {
                return Err(SessionError::InvalidFormat(format!(
                    "unknown role: {other}"
                )))
            }
        };
        let content = msg["content"].as_str().unwrap_or("").to_string();

        let mut sm = SessionMessage::text(role, &content);

        if role == SessionRole::System {
            sm.flags.insert(MessageFlags::PRESERVE);
        }

        // Parse tool_calls if present
        if let Some(calls) = msg.get("tool_calls").and_then(|v| v.as_array()) {
            for call in calls {
                let call_id = call["id"].as_str().unwrap_or("").to_string();
                let tool_name = call["function"]["name"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                let arguments = call["function"]["arguments"].clone();
                let arguments = if let Some(s) = arguments.as_str() {
                    serde_json::from_str(s).unwrap_or(serde_json::Value::Null)
                } else {
                    arguments
                };
                sm.tool_calls.push(ToolCall {
                    call_id,
                    tool_name,
                    arguments,
                });
            }
        }

        // Parse tool result
        if role == SessionRole::Tool {
            let call_id = msg["tool_call_id"].as_str().unwrap_or("").to_string();
            let tool_name = msg["name"].as_str().unwrap_or("").to_string();
            sm.tool_result = Some(ToolResult {
                call_id,
                tool_name,
                output: content.clone(),
                is_error: false,
            });
        }

        // Set token count
        sm.token_count = tokenizer.count_tokens(&sm.content) + 4;
        for tc in &sm.tool_calls {
            sm.token_count += tokenizer.count_tokens(&tc.tool_name);
            sm.token_count += tokenizer.count_tokens(&tc.arguments.to_string());
        }

        result.push(sm);
    }

    Ok(result)
}

/// Convert SessionMessages back to OpenAI format.
pub fn to_openai(messages: &[SessionMessage]) -> Vec<serde_json::Value> {
    messages
        .iter()
        .map(|m| {
            let mut obj = serde_json::json!({
                "role": m.role.to_string(),
                "content": m.content,
            });
            if !m.tool_calls.is_empty() {
                let calls: Vec<serde_json::Value> = m
                    .tool_calls
                    .iter()
                    .map(|tc| {
                        serde_json::json!({
                            "id": tc.call_id,
                            "type": "function",
                            "function": {
                                "name": tc.tool_name,
                                "arguments": tc.arguments.to_string(),
                            }
                        })
                    })
                    .collect();
                obj["tool_calls"] = serde_json::Value::Array(calls);
            }
            if let Some(tr) = &m.tool_result {
                obj["tool_call_id"] = serde_json::Value::String(tr.call_id.clone());
                obj["name"] = serde_json::Value::String(tr.tool_name.clone());
            }
            obj
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct WordCounter;
    impl TokenCounter for WordCounter {
        fn count_tokens(&self, text: &str) -> u32 {
            text.split_whitespace().count() as u32
        }
        fn truncate_to_tokens(&self, text: &str, max: u32) -> String {
            text.split_whitespace()
                .take(max as usize)
                .collect::<Vec<_>>()
                .join(" ")
        }
    }
    fn tc() -> &'static dyn TokenCounter {
        &WordCounter
    }

    #[test]
    fn test_openai_system_user_assistant_roundtrip() {
        let input = vec![
            serde_json::json!({"role": "system", "content": "You are helpful"}),
            serde_json::json!({"role": "user", "content": "hello"}),
            serde_json::json!({"role": "assistant", "content": "hi there"}),
        ];
        let parsed = from_openai(&input, tc()).unwrap();
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0].role, SessionRole::System);
        assert!(parsed[0].flags.contains(MessageFlags::PRESERVE));
        assert_eq!(parsed[1].role, SessionRole::User);
        assert_eq!(parsed[2].role, SessionRole::Assistant);
        assert_eq!(parsed[2].content, "hi there");

        // Roundtrip
        let output = to_openai(&parsed);
        assert_eq!(output.len(), 3);
        assert_eq!(output[0]["role"], "system");
        assert_eq!(output[1]["content"], "hello");
    }

    #[test]
    fn test_openai_tool_calls_parsed() {
        let input = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "find bugs"}),
            serde_json::json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "file.grep",
                        "arguments": "{\"pattern\": \"bug\"}"
                    }
                }]
            }),
            serde_json::json!({
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "file.grep",
                "content": "src/main.rs:10: let bug = true;"
            }),
        ];
        let parsed = from_openai(&input, tc()).unwrap();
        assert_eq!(parsed.len(), 4);

        // Assistant tool call
        assert_eq!(parsed[2].tool_calls.len(), 1);
        assert_eq!(parsed[2].tool_calls[0].call_id, "call_1");
        assert_eq!(parsed[2].tool_calls[0].tool_name, "file.grep");
        assert_eq!(parsed[2].tool_calls[0].arguments["pattern"], "bug");

        // Tool result
        assert_eq!(parsed[3].role, SessionRole::Tool);
        let tr = parsed[3].tool_result.as_ref().unwrap();
        assert_eq!(tr.call_id, "call_1");
        assert_eq!(tr.tool_name, "file.grep");
    }

    #[test]
    fn test_openai_tool_call_id_linkage_preserved() {
        let input = vec![
            serde_json::json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "build", "arguments": "{}"}
                }]
            }),
            serde_json::json!({
                "role": "tool",
                "tool_call_id": "call_abc",
                "name": "build",
                "content": "success"
            }),
        ];
        let parsed = from_openai(&input, tc()).unwrap();
        let output = to_openai(&parsed);

        assert_eq!(output[0]["tool_calls"][0]["id"], "call_abc");
        assert_eq!(output[1]["tool_call_id"], "call_abc");
        assert_eq!(output[1]["name"], "build");
    }

    #[test]
    fn test_to_openai_produces_valid_format() {
        let messages = vec![
            SessionMessage::system("You are helpful"),
            SessionMessage::text(SessionRole::User, "hello"),
            SessionMessage::assistant_tool_calls(
                "",
                vec![ToolCall {
                    call_id: "c1".into(),
                    tool_name: "test".into(),
                    arguments: serde_json::json!({"x": 1}),
                }],
            ),
        ];
        let output = to_openai(&messages);
        assert_eq!(output[0]["role"], "system");
        assert_eq!(output[2]["tool_calls"][0]["function"]["name"], "test");
        assert_eq!(output[2]["tool_calls"][0]["id"], "c1");
    }
}
