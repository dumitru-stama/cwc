use cwc_core::traits::TokenCounter;

use crate::memory::{create_memory_message, SessionMemoryStore};
use crate::message::{MessageFlags, SessionMessage};

/// Default token budget for memory rendering in rebuilt conversations.
pub(crate) const MEMORY_RENDER_TOKENS: u32 = 512;

/// Rebuild a conversation from components. Used after sliding window and hard reset.
///
/// Result order:
/// 1. System prompt (messages[0], PRESERVE)
/// 2. Memory message (IS_MEMORY flag) — only if store is non-empty
/// 3. Tail messages (from trim/reset)
///
/// Ensures:
/// - No duplicate memory messages
/// - Token counts set on new messages
/// - Flags set correctly
pub fn rebuild_conversation(
    system_prompt: &SessionMessage,
    memory_store: &SessionMemoryStore,
    tail_messages: Vec<SessionMessage>,
    tokenizer: &dyn TokenCounter,
) -> Vec<SessionMessage> {
    let mut result = Vec::with_capacity(2 + tail_messages.len());

    // 1. System prompt with PRESERVE flag
    let mut sys = system_prompt.clone();
    sys.flags.insert(MessageFlags::PRESERVE);
    if sys.token_count == 0 {
        sys.token_count = tokenizer.count_tokens(&sys.content) + 4;
    }
    result.push(sys);

    // 2. Memory message (if store has renderable content)
    if !memory_store.is_empty() {
        let mut mem_msg = create_memory_message(memory_store, MEMORY_RENDER_TOKENS, tokenizer);
        if !mem_msg.content.is_empty() {
            mem_msg.token_count = tokenizer.count_tokens(&mem_msg.content) + 4;
            result.push(mem_msg);
        }
    }

    // 3. Tail messages — filter out old memory messages to prevent duplicates
    for msg in tail_messages {
        if !msg.flags.contains(MessageFlags::IS_MEMORY) {
            result.push(msg);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::{FactPriority, MemoryFact, MemorySource};
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

    fn make_fact(key: &str, value: &str, priority: FactPriority) -> MemoryFact {
        MemoryFact {
            key: key.to_string(),
            value: value.to_string(),
            source: MemorySource::UserMessage { message_index: 0 },
            created_at: 100,
            priority,
        }
    }

    fn msg(role: SessionRole, content: &str, tokens: u32) -> SessionMessage {
        let mut m = SessionMessage::text(role, content);
        m.token_count = tokens;
        m
    }

    #[test]
    fn test_rebuild_system_memory_tail_order() {
        let sys = SessionMessage::system("system prompt");
        let mut store = SessionMemoryStore::new(2048, 30);
        store.upsert(make_fact("goal", "find bugs", FactPriority::Goal));
        let tail = vec![
            msg(SessionRole::User, "hello", 10),
            msg(SessionRole::Assistant, "hi", 8),
        ];

        let result = rebuild_conversation(&sys, &store, tail, tc());
        assert_eq!(result.len(), 4); // system + memory + 2 tail
        assert_eq!(result[0].role, SessionRole::System);
        assert!(result[0].flags.contains(MessageFlags::PRESERVE));
        assert_eq!(result[1].role, SessionRole::User);
        assert!(result[1].flags.contains(MessageFlags::IS_MEMORY));
        assert_eq!(result[2].role, SessionRole::User);
        assert_eq!(result[2].content, "hello");
        assert_eq!(result[3].role, SessionRole::Assistant);
    }

    #[test]
    fn test_rebuild_no_duplicate_memory() {
        let sys = SessionMessage::system("system prompt");
        let mut store = SessionMemoryStore::new(2048, 30);
        store.upsert(make_fact("goal", "find bugs", FactPriority::Goal));

        // Tail contains an old memory message that should be filtered out
        let mut old_mem = SessionMessage::text(SessionRole::User, "[WORKING_MEMORY] old");
        old_mem.flags.insert(MessageFlags::IS_MEMORY);
        old_mem.token_count = 10;
        let tail = vec![old_mem, msg(SessionRole::User, "hello", 10)];

        let result = rebuild_conversation(&sys, &store, tail, tc());
        let memory_count = result
            .iter()
            .filter(|m| m.flags.contains(MessageFlags::IS_MEMORY))
            .count();
        assert_eq!(memory_count, 1);
        assert!(result[1].flags.contains(MessageFlags::IS_MEMORY));
        assert!(result[1].content.contains("find bugs"));
    }

    #[test]
    fn test_rebuild_token_counts_set() {
        let sys = SessionMessage::system("system prompt");
        let mut store = SessionMemoryStore::new(2048, 30);
        store.upsert(make_fact("goal", "find bugs", FactPriority::Goal));

        let tail = vec![msg(SessionRole::User, "hello", 10)];
        let result = rebuild_conversation(&sys, &store, tail, tc());

        // Memory message should have non-zero token count
        let mem = &result[1];
        assert!(mem.flags.contains(MessageFlags::IS_MEMORY));
        assert!(mem.token_count > 0);
    }

    #[test]
    fn test_rebuild_empty_memory_no_memory_message() {
        let sys = SessionMessage::system("system prompt");
        let store = SessionMemoryStore::new(2048, 30);
        let tail = vec![
            msg(SessionRole::User, "hello", 10),
            msg(SessionRole::Assistant, "hi", 8),
        ];

        let result = rebuild_conversation(&sys, &store, tail, tc());
        assert_eq!(result.len(), 3); // system + 2 tail, no memory
        assert!(!result[1].flags.contains(MessageFlags::IS_MEMORY));
    }

    #[test]
    fn test_rebuild_memory_message_is_memory_flag() {
        let sys = SessionMessage::system("system prompt");
        let mut store = SessionMemoryStore::new(2048, 30);
        store.upsert(make_fact("build_result", "success", FactPriority::Status));

        let tail = vec![msg(SessionRole::User, "hello", 10)];
        let result = rebuild_conversation(&sys, &store, tail, tc());

        let mem = &result[1];
        assert!(mem.flags.contains(MessageFlags::IS_MEMORY));
        assert_eq!(mem.role, SessionRole::User);
    }
}
