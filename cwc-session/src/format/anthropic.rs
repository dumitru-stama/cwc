use cwc_core::traits::TokenCounter;

use crate::error::{Result, SessionError};
use crate::message::{SessionMessage, SessionRole, ToolCall, ToolResult};

/// Convert Anthropic-style messages to SessionMessages.
///
/// Handles:
/// - { role: "user"|"assistant", content: [...] } with content blocks
/// - tool_use content blocks (type: "tool_use", id, name, input)
/// - tool_result content blocks (type: "tool_result", tool_use_id, content)
/// - System prompt as separate top-level field
pub fn from_anthropic(
    system: Option<&str>,
    messages: &[serde_json::Value],
    tokenizer: &dyn TokenCounter,
) -> Result<Vec<SessionMessage>> {
    let mut result = Vec::with_capacity(messages.len() + 1);

    // System prompt is a separate field in Anthropic format
    if let Some(sys_text) = system {
        let mut sys = SessionMessage::system(sys_text);
        sys.token_count = tokenizer.count_tokens(sys_text) + 4;
        result.push(sys);
    }

    for (i, msg) in messages.iter().enumerate() {
        let role_str = msg["role"]
            .as_str()
            .ok_or_else(|| SessionError::MissingField(format!("messages[{i}].role")))?;

        let content = &msg["content"];

        // Anthropic uses content block arrays
        if let Some(blocks) = content.as_array() {
            parse_content_blocks(role_str, blocks, &mut result, tokenizer)?;
        } else {
            // Simple string content (some Anthropic messages may use this)
            let text = content.as_str().unwrap_or("");
            let role = match role_str {
                "user" => SessionRole::User,
                "assistant" => SessionRole::Assistant,
                other => {
                    return Err(SessionError::InvalidFormat(format!(
                        "unexpected Anthropic role: {other}"
                    )))
                }
            };
            let mut sm = SessionMessage::text(role, text);
            sm.token_count = tokenizer.count_tokens(text) + 4;
            result.push(sm);
        }
    }

    Ok(result)
}

fn parse_content_blocks(
    role_str: &str,
    blocks: &[serde_json::Value],
    result: &mut Vec<SessionMessage>,
    tokenizer: &dyn TokenCounter,
) -> Result<()> {
    // Collect text blocks into one message, tool_use into tool calls,
    // tool_result into Tool messages
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut tool_results: Vec<(String, String, String)> = Vec::new(); // (call_id, tool_name_hint, content)

    for block in blocks {
        let block_type = block["type"].as_str().unwrap_or("text");
        match block_type {
            "text" => {
                if let Some(text) = block["text"].as_str() {
                    text_parts.push(text.to_string());
                }
            }
            "tool_use" => {
                let id = block["id"].as_str().unwrap_or("").to_string();
                let name = block["name"].as_str().unwrap_or("").to_string();
                let input = block["input"].clone();
                tool_calls.push(ToolCall {
                    call_id: id,
                    tool_name: name,
                    arguments: input,
                });
            }
            "tool_result" => {
                let tool_use_id = block["tool_use_id"].as_str().unwrap_or("").to_string();
                let content_text = if let Some(arr) = block["content"].as_array() {
                    arr.iter()
                        .filter_map(|b| b["text"].as_str())
                        .collect::<Vec<_>>()
                        .join("\n")
                } else {
                    block["content"].as_str().unwrap_or("").to_string()
                };
                tool_results.push((tool_use_id, String::new(), content_text));
            }
            _ => {
                // Unknown block type — include as text
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    text_parts.push(text.to_string());
                }
            }
        }
    }

    let role = match role_str {
        "user" => SessionRole::User,
        "assistant" => SessionRole::Assistant,
        other => {
            return Err(SessionError::InvalidFormat(format!(
                "unexpected Anthropic role: {other}"
            )))
        }
    };

    // Emit assistant message with text + tool_calls
    if role == SessionRole::Assistant {
        let combined_text = text_parts.join("\n");
        let mut sm = if tool_calls.is_empty() {
            SessionMessage::text(role, &combined_text)
        } else {
            SessionMessage::assistant_tool_calls(&combined_text, tool_calls)
        };
        sm.token_count = tokenizer.count_tokens(&sm.content) + 4;
        for tc in &sm.tool_calls {
            sm.token_count += tokenizer.count_tokens(&tc.tool_name);
            sm.token_count += tokenizer.count_tokens(&tc.arguments.to_string());
        }
        result.push(sm);
    }

    // Emit user message (possibly with tool results mixed in)
    if role == SessionRole::User {
        // First emit tool results as Tool-role messages
        for (call_id, _tool_name, content) in &tool_results {
            let mut sm = SessionMessage::tool_result(ToolResult {
                call_id: call_id.clone(),
                tool_name: String::new(), // Anthropic doesn't include tool name in results
                output: content.clone(),
                is_error: false,
            });
            sm.token_count = tokenizer.count_tokens(content) + 4;
            result.push(sm);
        }
        // Then emit text as User message
        if !text_parts.is_empty() {
            let combined = text_parts.join("\n");
            let mut sm = SessionMessage::text(SessionRole::User, &combined);
            sm.token_count = tokenizer.count_tokens(&combined) + 4;
            result.push(sm);
        }
    }

    Ok(())
}

/// Convert SessionMessages back to Anthropic format.
///
/// Returns (system_prompt, messages) where system is extracted from messages[0]
/// if it has System role.
pub fn to_anthropic(messages: &[SessionMessage]) -> (Option<String>, Vec<serde_json::Value>) {
    let mut system = None;
    let mut output = Vec::new();
    let mut i = 0;

    // Extract system prompt
    if !messages.is_empty() && messages[0].role == SessionRole::System {
        system = Some(messages[0].content.clone());
        i = 1;
    }

    while i < messages.len() {
        let msg = &messages[i];
        match msg.role {
            SessionRole::System => {
                // Additional system messages → convert to user messages
                output.push(serde_json::json!({
                    "role": "user",
                    "content": [{"type": "text", "text": msg.content}]
                }));
            }
            SessionRole::User => {
                let mut blocks = Vec::new();
                if !msg.content.is_empty() {
                    blocks.push(serde_json::json!({"type": "text", "text": msg.content}));
                }
                output.push(serde_json::json!({
                    "role": "user",
                    "content": blocks,
                }));
            }
            SessionRole::Assistant => {
                let mut blocks = Vec::new();
                if !msg.content.is_empty() {
                    blocks.push(serde_json::json!({"type": "text", "text": msg.content}));
                }
                for tc in &msg.tool_calls {
                    blocks.push(serde_json::json!({
                        "type": "tool_use",
                        "id": tc.call_id,
                        "name": tc.tool_name,
                        "input": tc.arguments,
                    }));
                }
                output.push(serde_json::json!({
                    "role": "assistant",
                    "content": blocks,
                }));
            }
            SessionRole::Tool => {
                // Tool results go into a user message with tool_result blocks
                let tr = msg.tool_result.as_ref();
                let call_id = tr.map(|t| t.call_id.as_str()).unwrap_or("");
                let content_text = &msg.content;
                // Collect consecutive tool results into one user message
                let mut tool_result_blocks = vec![serde_json::json!({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": [{"type": "text", "text": content_text}],
                })];
                // Look ahead for more tool results
                while i + 1 < messages.len() && messages[i + 1].role == SessionRole::Tool {
                    i += 1;
                    let next = &messages[i];
                    let next_tr = next.tool_result.as_ref();
                    let next_id = next_tr.map(|t| t.call_id.as_str()).unwrap_or("");
                    tool_result_blocks.push(serde_json::json!({
                        "type": "tool_result",
                        "tool_use_id": next_id,
                        "content": [{"type": "text", "text": next.content}],
                    }));
                }
                output.push(serde_json::json!({
                    "role": "user",
                    "content": tool_result_blocks,
                }));
            }
        }
        i += 1;
    }

    (system, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::MessageFlags;

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
    fn test_anthropic_content_blocks_parsed() {
        let messages = vec![
            serde_json::json!({
                "role": "user",
                "content": [{"type": "text", "text": "find bugs"}]
            }),
            serde_json::json!({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll search"},
                    {"type": "tool_use", "id": "tu_1", "name": "file.grep", "input": {"pattern": "bug"}}
                ]
            }),
            serde_json::json!({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1", "content": "found 3 bugs"}
                ]
            }),
        ];
        let parsed = from_anthropic(Some("system prompt"), &messages, tc()).unwrap();

        // system + user + assistant(with tool_call) + tool_result
        assert_eq!(parsed[0].role, SessionRole::System);
        assert_eq!(parsed[1].role, SessionRole::User);
        assert_eq!(parsed[1].content, "find bugs");
        assert_eq!(parsed[2].role, SessionRole::Assistant);
        assert_eq!(parsed[2].tool_calls.len(), 1);
        assert_eq!(parsed[2].tool_calls[0].call_id, "tu_1");
        assert_eq!(parsed[3].role, SessionRole::Tool);
        assert_eq!(parsed[3].tool_result.as_ref().unwrap().call_id, "tu_1");
    }

    #[test]
    fn test_anthropic_system_prompt_separate() {
        let messages = vec![serde_json::json!({
            "role": "user",
            "content": [{"type": "text", "text": "hello"}]
        })];
        let parsed = from_anthropic(Some("You are a helpful assistant"), &messages, tc()).unwrap();
        assert_eq!(parsed[0].role, SessionRole::System);
        assert!(parsed[0].flags.contains(MessageFlags::PRESERVE));
        assert_eq!(parsed[0].content, "You are a helpful assistant");
        assert_eq!(parsed[1].role, SessionRole::User);
    }

    #[test]
    fn test_anthropic_tool_use_id_linkage() {
        let messages = vec![
            serde_json::json!({
                "role": "assistant",
                "content": [{
                    "type": "tool_use", "id": "tu_abc", "name": "build",
                    "input": {"target": "release"}
                }]
            }),
            serde_json::json!({
                "role": "user",
                "content": [{
                    "type": "tool_result", "tool_use_id": "tu_abc",
                    "content": "Build succeeded"
                }]
            }),
        ];
        let parsed = from_anthropic(None, &messages, tc()).unwrap();
        assert_eq!(parsed[0].tool_calls[0].call_id, "tu_abc");
        assert_eq!(parsed[1].tool_result.as_ref().unwrap().call_id, "tu_abc");
    }

    #[test]
    fn test_anthropic_roundtrip_preserves_semantics() {
        let original = vec![
            SessionMessage::system("You are helpful"),
            SessionMessage::text(SessionRole::User, "hello"),
            SessionMessage::text(SessionRole::Assistant, "hi"),
        ];
        let (system, anthropic_msgs) = to_anthropic(&original);
        assert_eq!(system.as_deref(), Some("You are helpful"));
        assert_eq!(anthropic_msgs.len(), 2); // user + assistant (system extracted)

        // Parse back
        let parsed = from_anthropic(system.as_deref(), &anthropic_msgs, tc()).unwrap();
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0].role, SessionRole::System);
        assert_eq!(parsed[1].role, SessionRole::User);
        assert_eq!(parsed[1].content, "hello");
        assert_eq!(parsed[2].role, SessionRole::Assistant);
        assert_eq!(parsed[2].content, "hi");
    }
}
