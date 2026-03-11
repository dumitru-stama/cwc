pub mod anthropic;
pub mod openai;
pub mod raw;

use crate::error::{Result, SessionError};
use crate::message::SessionMessage;
use cwc_core::traits::TokenCounter;

/// Auto-detect format and parse messages.
///
/// Detection heuristic:
/// - If any message has "content" as an array → Anthropic format
/// - If messages have a "role" key with string values → OpenAI format
/// - Otherwise → try raw JSON
pub fn auto_detect_and_parse(
    messages: &[serde_json::Value],
    system: Option<&str>,
    tokenizer: &dyn TokenCounter,
) -> Result<Vec<SessionMessage>> {
    if messages.is_empty() {
        return Ok(Vec::new());
    }

    // Check first non-empty message for format clues
    for msg in messages {
        // Anthropic: content is an array of content blocks
        if msg.get("content").and_then(|c| c.as_array()).is_some() {
            return anthropic::from_anthropic(system, messages, tokenizer);
        }
        // OpenAI: role is a string, content is a string (or null)
        if msg.get("role").and_then(|r| r.as_str()).is_some() {
            return openai::from_openai(messages, tokenizer);
        }
    }

    Err(SessionError::InvalidFormat(
        "could not auto-detect message format".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::SessionRole;

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
    fn test_auto_detect_openai_format() {
        let messages = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "hello"}),
        ];
        let parsed = auto_detect_and_parse(&messages, None, tc()).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].role, SessionRole::System);
        assert_eq!(parsed[1].role, SessionRole::User);
    }

    #[test]
    fn test_auto_detect_anthropic_format() {
        let messages = vec![
            serde_json::json!({
                "role": "user",
                "content": [{"type": "text", "text": "hello"}]
            }),
        ];
        let parsed = auto_detect_and_parse(&messages, Some("system prompt"), tc()).unwrap();
        // Should have system + user
        assert_eq!(parsed[0].role, SessionRole::System);
        assert_eq!(parsed[1].role, SessionRole::User);
    }
}
