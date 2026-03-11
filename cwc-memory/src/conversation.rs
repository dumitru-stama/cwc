use std::collections::VecDeque;
use std::sync::Arc;
use std::time::SystemTime;

use cwc_core::traits::TokenCounter;
use cwc_core::types::Role;

/// A single turn in the conversation.
#[derive(Debug, Clone)]
pub struct Turn {
    pub role: Role,
    pub content: String,
    pub token_count: u32,
    pub timestamp: SystemTime,
}

/// Short-term sliding window conversation memory.
pub struct ConversationMemory {
    turns: VecDeque<Turn>,
    max_turns: usize,
    tokenizer: Arc<dyn TokenCounter>,
}

impl ConversationMemory {
    pub fn new(max_turns: usize, tokenizer: Arc<dyn TokenCounter>) -> Self {
        Self {
            turns: VecDeque::new(),
            max_turns,
            tokenizer,
        }
    }

    /// Add a turn to the conversation. Drops oldest if at capacity.
    pub fn push(&mut self, role: Role, content: &str) {
        let token_count = self.tokenizer.count_tokens(content);
        self.turns.push_back(Turn {
            role,
            content: content.to_string(),
            token_count,
            timestamp: SystemTime::now(),
        });
        while self.turns.len() > self.max_turns {
            self.turns.pop_front();
        }
    }

    /// Get recent turns within a token budget.
    /// Returns most recent turns that fit, dropping oldest first.
    pub fn get_within_budget(&self, max_tokens: u32) -> Vec<&Turn> {
        let mut result = Vec::new();
        let mut budget = max_tokens;

        // Walk backwards (most recent first)
        for turn in self.turns.iter().rev() {
            if turn.token_count > budget {
                break;
            }
            result.push(turn);
            budget -= turn.token_count;
        }

        // Reverse to chronological order
        result.reverse();
        result
    }

    /// Render conversation history as text for prompt injection.
    /// Accounts for formatting overhead (header + role prefixes) in the budget.
    pub fn render(&self, max_tokens: u32) -> String {
        if self.turns.is_empty() || max_tokens == 0 {
            return String::new();
        }

        let header = "[CONVERSATION_HISTORY]\n";
        let header_tokens = self.tokenizer.count_tokens(header);
        if header_tokens >= max_tokens {
            return String::new();
        }

        let mut tokens_used = header_tokens;
        let mut lines = Vec::new();

        // Walk backwards (most recent first), accounting for formatted line cost
        for turn in self.turns.iter().rev() {
            let prefix = match turn.role {
                Role::User => "User",
                Role::Assistant => "Assistant",
                Role::System => "System",
            };
            let line = format!("{prefix}: {}\n", turn.content);
            let line_tokens = self.tokenizer.count_tokens(&line);

            if tokens_used + line_tokens > max_tokens {
                break;
            }

            lines.push(line);
            tokens_used += line_tokens;
        }

        if lines.is_empty() {
            return String::new();
        }

        // Reverse to chronological order
        lines.reverse();
        let mut out = String::from(header);
        for line in lines {
            out.push_str(&line);
        }
        out
    }

    /// Clear all history.
    pub fn clear(&mut self) {
        self.turns.clear();
    }

    /// Total token count of all stored turns.
    pub fn total_tokens(&self) -> u32 {
        self.turns.iter().map(|t| t.token_count).sum()
    }

    /// Number of stored turns.
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    /// Whether the conversation is empty.
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn make_mem(max_turns: usize) -> ConversationMemory {
        ConversationMemory::new(max_turns, Arc::new(FakeTokenCounter))
    }

    #[test]
    fn test_conversation_push_and_get_all() {
        let mut mem = make_mem(10);
        mem.push(Role::User, "What is Rust?");
        mem.push(Role::Assistant, "Rust is a systems language.");
        mem.push(Role::User, "Tell me more.");
        mem.push(Role::Assistant, "It has ownership and borrowing.");
        mem.push(Role::User, "Thanks!");

        let turns = mem.get_within_budget(1000);
        assert_eq!(turns.len(), 5);
        assert_eq!(turns[0].role, Role::User);
        assert!(turns[0].content.contains("What is Rust"));
    }

    #[test]
    fn test_conversation_sliding_window() {
        let mut mem = make_mem(10);
        for i in 0..20 {
            mem.push(Role::User, &format!("message {i}"));
        }
        assert_eq!(mem.len(), 10);
        // Should have messages 10-19
        let turns = mem.get_within_budget(10000);
        assert_eq!(turns.len(), 10);
        assert!(turns[0].content.contains("message 10"));
        assert!(turns[9].content.contains("message 19"));
    }

    #[test]
    fn test_conversation_budget_constraint() {
        let mut mem = make_mem(100);
        // Each message is 3 words = 3 tokens
        mem.push(Role::User, "one two three");
        mem.push(Role::Assistant, "four five six");
        mem.push(Role::User, "seven eight nine");

        // Budget of 5 tokens → only the last 1 turn fits (3 tokens)
        // (walking from most recent: "seven eight nine" = 3, next "four five six" = 3, total = 6 > 5)
        let turns = mem.get_within_budget(5);
        assert_eq!(turns.len(), 1);
        assert!(turns[0].content.contains("seven eight nine"));
    }

    #[test]
    fn test_conversation_render_format() {
        let mut mem = make_mem(10);
        mem.push(Role::User, "What is Rust?");
        mem.push(Role::Assistant, "A systems language.");
        mem.push(Role::User, "Tell me more.");

        let rendered = mem.render(1000);
        assert!(rendered.starts_with("[CONVERSATION_HISTORY]"));
        assert!(rendered.contains("User: What is Rust?"));
        assert!(rendered.contains("Assistant: A systems language."));
        assert!(rendered.contains("User: Tell me more."));
    }

    #[test]
    fn test_conversation_render_empty() {
        let mem = make_mem(10);
        assert_eq!(mem.render(1000), "");
    }

    #[test]
    fn test_conversation_total_tokens() {
        let mut mem = make_mem(10);
        mem.push(Role::User, "one two three"); // 3
        mem.push(Role::Assistant, "four five");  // 2
        assert_eq!(mem.total_tokens(), 5);
    }

    #[test]
    fn test_conversation_clear() {
        let mut mem = make_mem(10);
        mem.push(Role::User, "hello");
        mem.push(Role::Assistant, "world");
        assert_eq!(mem.len(), 2);
        mem.clear();
        assert!(mem.is_empty());
        assert_eq!(mem.total_tokens(), 0);
    }

    #[test]
    fn test_conversation_budget_exactly_fits() {
        let mut mem = make_mem(100);
        mem.push(Role::User, "one two");     // 2
        mem.push(Role::Assistant, "three");   // 1
        mem.push(Role::User, "four five");    // 2
        // Total raw = 5; budget = 5 → all fit (get_within_budget uses raw content tokens)
        let turns = mem.get_within_budget(5);
        assert_eq!(turns.len(), 3);
    }

    #[test]
    fn test_conversation_render_accounts_for_overhead() {
        let mut mem = make_mem(100);
        mem.push(Role::User, "one two");     // raw: 2
        mem.push(Role::Assistant, "three");   // raw: 1
        mem.push(Role::User, "four five");    // raw: 2
        // render with budget=5: header "[CONVERSATION_HISTORY]\n" = 1 word-token
        // "User: four five\n" = 3, total = 4. "Assistant: three\n" = 2, total = 6 > 5.
        // So only the most recent turn fits in render.
        let rendered = mem.render(5);
        assert!(rendered.contains("four five"));
        assert!(!rendered.contains("three"));
        assert!(!rendered.contains("one two"));
    }

    #[test]
    fn test_conversation_render_generous_budget() {
        let mut mem = make_mem(100);
        mem.push(Role::User, "one two");
        mem.push(Role::Assistant, "three");
        mem.push(Role::User, "four five");
        // With large budget, all turns should render
        let rendered = mem.render(1000);
        assert!(rendered.contains("one two"));
        assert!(rendered.contains("three"));
        assert!(rendered.contains("four five"));
    }

    #[test]
    fn test_conversation_budget_zero() {
        let mut mem = make_mem(10);
        mem.push(Role::User, "hello");
        let turns = mem.get_within_budget(0);
        assert!(turns.is_empty());
    }

    #[test]
    fn test_conversation_render_budget_zero() {
        let mut mem = make_mem(10);
        mem.push(Role::User, "hello");
        assert_eq!(mem.render(0), "");
    }

    #[test]
    fn test_conversation_max_turns_one() {
        let mut mem = make_mem(1);
        mem.push(Role::User, "first");
        mem.push(Role::User, "second");
        mem.push(Role::User, "third");
        assert_eq!(mem.len(), 1);
        let turns = mem.get_within_budget(1000);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].content, "third");
    }

    #[test]
    fn test_conversation_render_system_role() {
        let mut mem = make_mem(10);
        mem.push(Role::System, "You are a helpful assistant.");
        mem.push(Role::User, "Hello");
        let rendered = mem.render(1000);
        assert!(rendered.contains("System: You are a helpful assistant."));
        assert!(rendered.contains("User: Hello"));
    }
}
