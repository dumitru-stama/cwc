use cwc_core::traits::TokenCounter;

use crate::message::{MessageFlags, SessionMessage, SessionRole};

use super::store::SessionMemoryStore;

/// Extract the goal text from the first real user message in the conversation.
///
/// "Real" = SessionRole::User and not flagged IS_NUDGE or IS_MEMORY.
/// Truncates to `max_chars` if needed.
pub fn extract_goal_text(messages: &[SessionMessage], max_chars: usize) -> Option<String> {
    for msg in messages {
        if msg.role == SessionRole::User
            && !msg.flags.contains(MessageFlags::IS_NUDGE)
            && !msg.flags.contains(MessageFlags::IS_MEMORY)
        {
            let text = msg.content.trim();
            if text.is_empty() {
                continue;
            }
            if text.chars().count() <= max_chars {
                return Some(text.to_string());
            }
            return Some(text.chars().take(max_chars).collect());
        }
    }
    None
}

/// Format a goal anchoring nudge to append after tool results.
pub fn format_goal_nudge(goal: &str) -> String {
    format!("Continue with your analysis.\nYour goal: {goal}")
}

/// Create a memory injection message for insertion at position 1 (after system prompt).
///
/// The message has role=User, flag=IS_MEMORY, and content starting with [WORKING_MEMORY].
pub fn create_memory_message(
    memory_store: &SessionMemoryStore,
    max_tokens: u32,
    tokenizer: &dyn TokenCounter,
) -> SessionMessage {
    let content = memory_store.render(max_tokens, tokenizer);
    let mut msg = SessionMessage::text(SessionRole::User, content);
    msg.flags.insert(MessageFlags::IS_MEMORY);
    msg
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
    fn test_extract_goal_text_basic() {
        let messages = vec![
            SessionMessage::system("sys"),
            SessionMessage::text(SessionRole::User, "Find all memory leaks in parser.rs"),
        ];
        let goal = extract_goal_text(&messages, 256).unwrap();
        assert_eq!(goal, "Find all memory leaks in parser.rs");
    }

    #[test]
    fn test_extract_goal_text_skips_nudge() {
        let messages = vec![
            SessionMessage::system("sys"),
            SessionMessage::nudge("keep going"),
            SessionMessage::text(SessionRole::User, "the real request"),
        ];
        let goal = extract_goal_text(&messages, 256).unwrap();
        assert_eq!(goal, "the real request");
    }

    #[test]
    fn test_extract_goal_text_skips_memory() {
        let mut mem_msg = SessionMessage::text(SessionRole::User, "[WORKING_MEMORY]\nGoal: old");
        mem_msg.flags.insert(MessageFlags::IS_MEMORY);
        let messages = vec![
            SessionMessage::system("sys"),
            mem_msg,
            SessionMessage::text(SessionRole::User, "actual goal"),
        ];
        let goal = extract_goal_text(&messages, 256).unwrap();
        assert_eq!(goal, "actual goal");
    }

    #[test]
    fn test_extract_goal_text_truncates() {
        let long = "x".repeat(300);
        let messages = vec![
            SessionMessage::text(SessionRole::User, &long),
        ];
        let goal = extract_goal_text(&messages, 100).unwrap();
        assert_eq!(goal.len(), 100);
    }

    #[test]
    fn test_extract_goal_text_empty() {
        let messages: Vec<SessionMessage> = vec![];
        assert!(extract_goal_text(&messages, 256).is_none());
    }

    #[test]
    fn test_extract_goal_text_only_system() {
        let messages = vec![SessionMessage::system("sys")];
        assert!(extract_goal_text(&messages, 256).is_none());
    }

    #[test]
    fn test_extract_goal_text_skips_empty_user() {
        let messages = vec![
            SessionMessage::text(SessionRole::User, "   "),
            SessionMessage::text(SessionRole::User, "real goal"),
        ];
        let goal = extract_goal_text(&messages, 256).unwrap();
        assert_eq!(goal, "real goal");
    }

    #[test]
    fn test_format_goal_nudge() {
        let nudge = format_goal_nudge("find all bugs");
        assert!(nudge.contains("Continue with your analysis"));
        assert!(nudge.contains("Your goal: find all bugs"));
    }

    #[test]
    fn test_create_memory_message_basic() {
        use super::super::types::{FactPriority, MemoryFact, MemorySource};

        let mut store = SessionMemoryStore::new(2048, 30);
        store.upsert(MemoryFact {
            key: "goal".into(),
            value: "find bugs".into(),
            source: MemorySource::UserMessage { message_index: 1 },
            created_at: 100,
            priority: FactPriority::Goal,
        });

        let msg = create_memory_message(&store, 1000, tc());
        assert_eq!(msg.role, SessionRole::User);
        assert!(msg.flags.contains(MessageFlags::IS_MEMORY));
        assert!(msg.content.starts_with("[WORKING_MEMORY]"));
        assert!(msg.content.contains("Goal: find bugs"));
    }

    #[test]
    fn test_create_memory_message_empty_store() {
        let store = SessionMemoryStore::new(2048, 30);
        let msg = create_memory_message(&store, 1000, tc());
        assert_eq!(msg.role, SessionRole::User);
        assert!(msg.flags.contains(MessageFlags::IS_MEMORY));
        assert!(msg.content.is_empty());
    }
}
