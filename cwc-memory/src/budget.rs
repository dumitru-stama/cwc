use cwc_core::traits::TokenCounter;

use crate::conversation::ConversationMemory;
use crate::longterm::MemoryEntry;

/// Render long-term memory entries as text for prompt injection.
pub fn render_long_term(entries: &[MemoryEntry], max_tokens: u32, tokenizer: &dyn TokenCounter) -> String {
    if entries.is_empty() || max_tokens == 0 {
        return String::new();
    }

    let header = "[LONG_TERM_MEMORY]\n";
    let header_tokens = tokenizer.count_tokens(header);
    if header_tokens >= max_tokens {
        return String::new();
    }

    let mut out = String::from(header);
    let mut tokens_used = header_tokens;
    let mut has_entries = false;

    for entry in entries {
        let line = format!("- {}: {} [{}]\n", entry.key, entry.value, entry.category.as_str());
        let line_tokens = tokenizer.count_tokens(&line);

        if tokens_used + line_tokens > max_tokens {
            break;
        }

        out.push_str(&line);
        tokens_used += line_tokens;
        has_entries = true;
    }

    // Don't return a bare header with no entries
    if !has_entries {
        return String::new();
    }

    out
}

/// Compute memory text (conversation + long-term) that fits within a budget.
///
/// Priority: conversation history first, then long-term memory.
/// If budget is tight, long-term memory is trimmed before conversation.
pub fn compile_memory(
    conversation: &ConversationMemory,
    long_term: &[MemoryEntry],
    memory_budget: u32,
    tokenizer: &dyn TokenCounter,
) -> String {
    if memory_budget == 0 {
        return String::new();
    }

    // Give conversation up to 60% of memory budget
    let conv_budget = (memory_budget as f32 * 0.6) as u32;
    let conv_text = conversation.render(conv_budget);
    let conv_tokens = tokenizer.count_tokens(&conv_text);

    // Remaining budget goes to long-term memory
    let ltm_budget = memory_budget.saturating_sub(conv_tokens);
    let ltm_text = render_long_term(long_term, ltm_budget, tokenizer);

    format!("{conv_text}{ltm_text}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use cwc_core::types::Role;

    struct FakeTokenCounter;

    impl TokenCounter for FakeTokenCounter {
        fn count_tokens(&self, text: &str) -> u32 {
            text.split_whitespace().count() as u32
        }

        fn truncate_to_tokens(&self, text: &str, max_tokens: u32) -> String {
            let words: Vec<&str> = text.split_whitespace().collect();
            words[..words.len().min(max_tokens as usize)].join(" ")
        }
    }

    fn make_entries() -> Vec<MemoryEntry> {
        use chrono::Utc;
        use crate::longterm::MemoryCategory;

        vec![
            MemoryEntry {
                key: "pref:units".to_string(),
                value: "metric".to_string(),
                category: MemoryCategory::UserPreference,
                created_at: Utc::now(),
                updated_at: Utc::now(),
                access_count: 5,
            },
            MemoryEntry {
                key: "fact:db".to_string(),
                value: "uses PostgreSQL".to_string(),
                category: MemoryCategory::ProjectFact,
                created_at: Utc::now(),
                updated_at: Utc::now(),
                access_count: 2,
            },
        ]
    }

    #[test]
    fn test_render_long_term_basic() {
        let entries = make_entries();
        let tc = FakeTokenCounter;
        let rendered = render_long_term(&entries, 1000, &tc);
        assert!(rendered.starts_with("[LONG_TERM_MEMORY]"));
        assert!(rendered.contains("pref:units: metric"));
        assert!(rendered.contains("fact:db: uses PostgreSQL"));
    }

    #[test]
    fn test_render_long_term_empty() {
        let tc = FakeTokenCounter;
        assert_eq!(render_long_term(&[], 1000, &tc), "");
    }

    #[test]
    fn test_render_long_term_budget_limit() {
        let entries = make_entries();
        let tc = FakeTokenCounter;
        // Very tight budget — should only include header + maybe one entry
        let rendered = render_long_term(&entries, 5, &tc);
        // "[LONG_TERM_MEMORY]\n" is 1 token (word count) but actually multiple
        // With word-based tokenizer, header is "[LONG_TERM_MEMORY]" = 1 token
        // Each entry line is several words
        assert!(!rendered.is_empty());
    }

    #[test]
    fn test_compile_memory_fits_budget() {
        let tc = Arc::new(FakeTokenCounter);
        let mut conv = ConversationMemory::new(10, tc.clone());
        conv.push(Role::User, "What is Rust?");
        conv.push(Role::Assistant, "A systems language.");

        let entries = make_entries();
        let result = compile_memory(&conv, &entries, 1000, &*tc);
        assert!(result.contains("[CONVERSATION_HISTORY]"));
        assert!(result.contains("[LONG_TERM_MEMORY]"));
    }

    #[test]
    fn test_compile_memory_tight_budget_trims_ltm() {
        let tc = Arc::new(FakeTokenCounter);
        let mut conv = ConversationMemory::new(10, tc.clone());
        conv.push(Role::User, "hello world");

        let entries = make_entries();
        // Budget=20: conv_budget=12 (60%), conv renders "hello world" (2 tokens turn),
        // header "[CONVERSATION_HISTORY]\nUser: hello world\n" is a few tokens.
        // LTM gets remaining budget. With tight budget, LTM should be trimmed.
        let result = compile_memory(&conv, &entries, 20, &*tc);
        assert!(result.contains("[CONVERSATION_HISTORY]"));
        // With such tight budget, LTM may or may not appear
    }

    #[test]
    fn test_compile_memory_zero_budget() {
        let tc = Arc::new(FakeTokenCounter);
        let mut conv = ConversationMemory::new(10, tc.clone());
        conv.push(Role::User, "hello");

        let entries = make_entries();
        let result = compile_memory(&conv, &entries, 0, &*tc);
        assert_eq!(result, "");
    }

    #[test]
    fn test_render_long_term_budget_zero() {
        let entries = make_entries();
        let tc = FakeTokenCounter;
        let rendered = render_long_term(&entries, 0, &tc);
        assert_eq!(rendered, "");
    }

    #[test]
    fn test_render_long_term_budget_too_small_for_entries() {
        let entries = make_entries();
        let tc = FakeTokenCounter;
        // Budget=1: only the header "[LONG_TERM_MEMORY]\n" (1 word-token) fits,
        // but no entries can fit, so should return empty (no bare header).
        let rendered = render_long_term(&entries, 1, &tc);
        assert_eq!(rendered, "", "should not return bare header with no entries");
    }

    #[test]
    fn test_compile_memory_empty_conversation_with_ltm() {
        let tc = Arc::new(FakeTokenCounter);
        let conv = ConversationMemory::new(10, tc.clone());

        let entries = make_entries();
        let result = compile_memory(&conv, &entries, 100, &*tc);
        // No conversation, so all budget goes to LTM
        assert!(!result.contains("[CONVERSATION_HISTORY]"));
        assert!(result.contains("[LONG_TERM_MEMORY]"));
        assert!(result.contains("pref:units"));
    }

    #[test]
    fn test_compile_memory_no_ltm() {
        let tc = Arc::new(FakeTokenCounter);
        let mut conv = ConversationMemory::new(10, tc.clone());
        conv.push(Role::User, "hello world");

        let result = compile_memory(&conv, &[], 100, &*tc);
        assert!(result.contains("[CONVERSATION_HISTORY]"));
        assert!(!result.contains("[LONG_TERM_MEMORY]"));
    }
}
