use cwc_core::traits::TokenCounter;

use crate::error::{Result, SessionError};
use crate::message::{SessionMessage, SessionRole};

/// Convert a plain text conversation (alternating user/assistant lines) to SessionMessages.
///
/// Format: lines starting with "User:" or "Assistant:" or "System:".
/// Lines without a role prefix continue the previous message.
pub fn from_plain_text(text: &str, tokenizer: &dyn TokenCounter) -> Result<Vec<SessionMessage>> {
    let mut messages = Vec::new();
    let mut current_role: Option<SessionRole> = None;
    let mut current_content = String::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if current_role.is_some() {
                current_content.push('\n');
            }
            continue;
        }

        let (new_role, content) = if let Some(rest) = trimmed.strip_prefix("System:") {
            (Some(SessionRole::System), rest.trim())
        } else if let Some(rest) = trimmed.strip_prefix("User:") {
            (Some(SessionRole::User), rest.trim())
        } else if let Some(rest) = trimmed.strip_prefix("Assistant:") {
            (Some(SessionRole::Assistant), rest.trim())
        } else {
            (None, trimmed)
        };

        if let Some(role) = new_role {
            // Flush previous message
            if let Some(prev_role) = current_role.take() {
                let trimmed_content = current_content.trim().to_string();
                if !trimmed_content.is_empty() {
                    let mut msg = if prev_role == SessionRole::System {
                        SessionMessage::system(&trimmed_content)
                    } else {
                        SessionMessage::text(prev_role, &trimmed_content)
                    };
                    msg.token_count = tokenizer.count_tokens(&msg.content) + 4;
                    messages.push(msg);
                }
            }
            current_role = Some(role);
            current_content = content.to_string();
        } else {
            // Continuation line
            if current_role.is_some() {
                current_content.push('\n');
                current_content.push_str(content);
            }
        }
    }

    // Flush last message
    if let Some(role) = current_role {
        let trimmed_content = current_content.trim().to_string();
        if !trimmed_content.is_empty() {
            let mut msg = if role == SessionRole::System {
                SessionMessage::system(&trimmed_content)
            } else {
                SessionMessage::text(role, &trimmed_content)
            };
            msg.token_count = tokenizer.count_tokens(&msg.content) + 4;
            messages.push(msg);
        }
    }

    Ok(messages)
}

/// Convert SessionMessages to a JSON string (our native format).
pub fn to_json(messages: &[SessionMessage]) -> Result<String> {
    serde_json::to_string_pretty(messages).map_err(|e| {
        SessionError::InvalidFormat(format!("failed to serialize messages: {e}"))
    })
}

/// Parse SessionMessages from a JSON string (our native format).
pub fn from_json(json: &str, tokenizer: &dyn TokenCounter) -> Result<Vec<SessionMessage>> {
    let mut messages: Vec<SessionMessage> = serde_json::from_str(json)?;
    // Ensure token counts are set
    for msg in &mut messages {
        if msg.token_count == 0 {
            msg.token_count = tokenizer.count_tokens(&msg.content) + 4;
            for tc in &msg.tool_calls {
                msg.token_count += tokenizer.count_tokens(&tc.tool_name);
                msg.token_count += tokenizer.count_tokens(&tc.arguments.to_string());
            }
        }
    }
    Ok(messages)
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
    fn test_plain_text_roundtrip() {
        let text = "\
System: You are helpful
User: hello world
Assistant: hi there
User: how are you
Assistant: I'm fine";
        let messages = from_plain_text(text, tc()).unwrap();
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[0].role, SessionRole::System);
        assert!(messages[0].flags.contains(MessageFlags::PRESERVE));
        assert_eq!(messages[1].role, SessionRole::User);
        assert_eq!(messages[1].content, "hello world");
        assert_eq!(messages[2].role, SessionRole::Assistant);
        assert_eq!(messages[3].role, SessionRole::User);
        assert_eq!(messages[4].role, SessionRole::Assistant);
    }

    #[test]
    fn test_json_serialization_roundtrip() {
        let messages = vec![
            SessionMessage::system("system prompt"),
            SessionMessage::text(SessionRole::User, "hello"),
            SessionMessage::text(SessionRole::Assistant, "hi"),
        ];
        let json = to_json(&messages).unwrap();
        let back = from_json(&json, tc()).unwrap();
        assert_eq!(back.len(), 3);
        assert_eq!(back[0].role, SessionRole::System);
        assert_eq!(back[1].content, "hello");
        assert_eq!(back[2].content, "hi");
    }
}
