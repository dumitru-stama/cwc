use cwc_core::traits::TokenCounter;

use crate::budget::SessionBudget;
use crate::error::Result;
use crate::memory::{extract_from_turns, extract_goal, MemoryFact, SessionMemoryStore};
use crate::message::SessionMessage;
use crate::rebuild::rebuild_conversation;
use crate::turn::parse_turns;

/// Result of a hard context reset.
#[derive(Debug, Clone)]
pub struct ResetResult {
    /// The rebuilt conversation (system + memory + tail).
    pub messages: Vec<SessionMessage>,
    /// Facts extracted from all dropped messages.
    pub extracted_facts: Vec<MemoryFact>,
    /// Tokens before reset.
    pub tokens_before: u32,
    /// Tokens after reset.
    pub tokens_after: u32,
}

/// Hard context reset: rebuild conversation from scratch.
///
/// Algorithm:
/// 1. Extract memory from ALL turns except system prompt
/// 2. Keep only:
///    - messages[0]: system prompt (PRESERVE flag)
///    - Fresh memory message (rendered from updated store)
///    - Last N atomic turns (default from budget.min_tail_turns)
/// 3. Replace conversation with this minimal context
///
/// This is instant and free — no LLM call needed.
pub fn hard_reset(
    messages: &[SessionMessage],
    memory_store: &mut SessionMemoryStore,
    budget: &SessionBudget,
    tokenizer: &dyn TokenCounter,
) -> Result<ResetResult> {
    if messages.is_empty() {
        return Ok(ResetResult {
            messages: Vec::new(),
            extracted_facts: Vec::new(),
            tokens_before: 0,
            tokens_after: 0,
        });
    }

    let tokens_before: u32 = messages.iter().map(|m| m.token_count).sum();
    let turns = parse_turns(messages);

    // Extract memory from ALL non-system turns
    let non_system_turns: Vec<_> = if turns.len() > 1 {
        turns[1..].to_vec()
    } else {
        Vec::new()
    };
    let mut extracted_facts = extract_from_turns(messages, &non_system_turns);

    // Extract goal from full conversation
    if let Some(goal) = extract_goal(messages) {
        if memory_store.get("goal").is_none() {
            extracted_facts.push(goal.clone());
        }
        memory_store.upsert(goal);
    }

    // Upsert all extracted facts
    memory_store.upsert_batch(extracted_facts.clone());
    memory_store.enforce_limits();

    // Keep last N turns
    let keep_count = budget.min_tail_turns.min(non_system_turns.len());
    let kept_turns = if keep_count > 0 {
        &non_system_turns[non_system_turns.len() - keep_count..]
    } else {
        &[]
    };

    let tail_start = kept_turns
        .first()
        .map(|t| t.start)
        .unwrap_or(messages.len());
    let tail_messages: Vec<SessionMessage> = messages[tail_start..].to_vec();

    // Rebuild
    let system_prompt = &messages[0];
    let rebuilt = rebuild_conversation(system_prompt, memory_store, tail_messages, tokenizer);
    let tokens_after: u32 = rebuilt.iter().map(|m| m.token_count).sum();

    Ok(ResetResult {
        messages: rebuilt,
        extracted_facts,
        tokens_before,
        tokens_after,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::FactPriority;
    use crate::message::{MessageFlags, SessionRole, ToolCall, ToolResult};

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

    fn msg(role: SessionRole, content: &str, tokens: u32) -> SessionMessage {
        let mut m = SessionMessage::text(role, content);
        m.token_count = tokens;
        m
    }

    fn sys_msg(tokens: u32) -> SessionMessage {
        let mut m = SessionMessage::system("system prompt");
        m.token_count = tokens;
        m
    }

    fn tool_call_msg(name: &str, tokens: u32) -> SessionMessage {
        let mut m = SessionMessage::assistant_tool_calls(
            "",
            vec![ToolCall {
                call_id: "c1".into(),
                tool_name: name.into(),
                arguments: serde_json::json!({"pattern": "test"}),
            }],
        );
        m.token_count = tokens;
        m
    }

    fn tool_result_msg(name: &str, output: &str, tokens: u32) -> SessionMessage {
        let mut m = SessionMessage::tool_result(ToolResult {
            call_id: "c1".into(),
            tool_name: name.into(),
            output: output.into(),
            is_error: false,
        });
        m.token_count = tokens;
        m
    }

    fn test_budget(min_turns: usize) -> SessionBudget {
        SessionBudget {
            context_window: 32768,
            max_output: 4096,
            usable_budget: 10000,
            sliding_window_trigger: 5000,
            hard_reset_trigger: 6000,
            compaction_trigger: 8500,
            tail_tokens: 2000,
            min_tail_turns: min_turns,
        }
    }

    #[test]
    fn test_hard_reset_extracts_from_all_turns() {
        let messages = vec![
            sys_msg(50),
            msg(SessionRole::User, "find malloc bugs", 100),
            tool_call_msg("file.grep", 20),
            tool_result_msg("file.grep", "src/main.rs:10: malloc(256)", 200),
            msg(SessionRole::Assistant, "Found malloc", 50),
            msg(SessionRole::User, "now fix them", 100),
            msg(SessionRole::Assistant, "fixing...", 100),
        ];
        let mut store = SessionMemoryStore::default();

        let result = hard_reset(&messages, &mut store, &test_budget(2), tc()).unwrap();
        assert!(!result.extracted_facts.is_empty(), "should extract facts");
        assert!(store.get("goal").is_some(), "should extract goal");
    }

    #[test]
    fn test_hard_reset_system_memory_last_2_turns() {
        let messages = vec![
            sys_msg(50),
            msg(SessionRole::User, "q1", 100),
            msg(SessionRole::Assistant, "a1", 100),
            msg(SessionRole::User, "q2", 100),
            msg(SessionRole::Assistant, "a2", 100),
            msg(SessionRole::User, "q3", 100),
            msg(SessionRole::Assistant, "a3", 100),
        ];
        let mut store = SessionMemoryStore::default();

        let result = hard_reset(&messages, &mut store, &test_budget(2), tc()).unwrap();
        // Should have: system + memory + last 2 turns (q2+a2, q3+a3)
        assert_eq!(result.messages[0].role, SessionRole::System);
        assert!(result.messages[0].flags.contains(MessageFlags::PRESERVE));
        // Memory at position 1
        assert!(result.messages[1].flags.contains(MessageFlags::IS_MEMORY));
        // Last 2 turns: 4 messages (q2,a2,q3,a3)
        // Total: sys + memory + 4 = 6
        assert_eq!(result.messages.len(), 6);
    }

    #[test]
    fn test_hard_reset_tokens_much_less() {
        let mut messages = vec![sys_msg(50)];
        for i in 0..10 {
            messages.push(msg(SessionRole::User, &format!("question {i} with many words"), 100));
            messages.push(msg(SessionRole::Assistant, &format!("answer {i} with detail"), 100));
        }
        let mut store = SessionMemoryStore::default();

        let result = hard_reset(&messages, &mut store, &test_budget(2), tc()).unwrap();
        assert!(
            result.tokens_after < result.tokens_before / 2,
            "tokens_after {} should be much less than tokens_before {}",
            result.tokens_after,
            result.tokens_before
        );
    }

    #[test]
    fn test_hard_reset_goal_preserved_in_store() {
        let messages = vec![
            sys_msg(50),
            msg(SessionRole::User, "Find all security vulnerabilities", 100),
            msg(SessionRole::Assistant, "I will search", 50),
        ];
        let mut store = SessionMemoryStore::default();

        hard_reset(&messages, &mut store, &test_budget(2), tc()).unwrap();
        let goal = store.get("goal").expect("goal should be in store");
        assert!(goal.value.contains("security vulnerabilities"));
        assert_eq!(goal.priority, FactPriority::Goal);
    }

    #[test]
    fn test_hard_reset_subsequent_call_updates_not_duplicates() {
        let messages1 = vec![
            sys_msg(50),
            msg(SessionRole::User, "find bugs", 100),
            msg(SessionRole::Assistant, "working on it", 100),
        ];
        let mut store = SessionMemoryStore::default();
        hard_reset(&messages1, &mut store, &test_budget(1), tc()).unwrap();
        let count_after_first = store.len();

        // Second reset with more turns
        let messages2 = vec![
            sys_msg(50),
            msg(SessionRole::User, "find bugs", 100),
            msg(SessionRole::Assistant, "found 3 issues", 100),
            msg(SessionRole::User, "fix them", 100),
            msg(SessionRole::Assistant, "fixing", 100),
        ];
        hard_reset(&messages2, &mut store, &test_budget(1), tc()).unwrap();

        // Goal should not be duplicated (same key "goal")
        let goal_count = store.all().iter().filter(|f| f.key == "goal").count();
        assert_eq!(goal_count, 1, "goal should not be duplicated");
        // Store may have new facts but goal count stays 1
        assert!(store.len() >= count_after_first);
    }

    #[test]
    fn test_hard_reset_single_turn() {
        let messages = vec![
            sys_msg(50),
            msg(SessionRole::User, "hello", 10),
            msg(SessionRole::Assistant, "hi", 8),
        ];
        let mut store = SessionMemoryStore::default();

        let result = hard_reset(&messages, &mut store, &test_budget(2), tc()).unwrap();
        // Should keep: system + memory + the single turn
        assert_eq!(result.messages[0].role, SessionRole::System);
        // The turn's messages should be present
        let has_user = result
            .messages
            .iter()
            .any(|m| m.role == SessionRole::User && !m.flags.contains(MessageFlags::IS_MEMORY));
        assert!(has_user, "should keep the user message");
    }

    // Integration test: hard reset + verify memory < 30% of budget
    #[test]
    fn test_hard_reset_full_cycle() {
        let mut messages = vec![sys_msg(50)];
        for i in 0..20 {
            messages.push(msg(SessionRole::User, &format!("task {i}"), 50));
            messages.push(tool_call_msg("build", 20));
            messages.push(tool_result_msg("build", &format!("result {i}"), 200));
            messages.push(msg(SessionRole::Assistant, &format!("done {i}"), 30));
        }
        let budget = test_budget(2);
        let mut store = SessionMemoryStore::default();

        let result = hard_reset(&messages, &mut store, &budget, tc()).unwrap();
        // Tokens after should be well under 30% of usable budget
        assert!(
            result.tokens_after < budget.usable_budget * 30 / 100,
            "tokens_after {} should be < 30% of {} = {}",
            result.tokens_after,
            budget.usable_budget,
            budget.usable_budget * 30 / 100
        );
        assert!(!store.is_empty(), "memory should contain extracted facts");
    }
}
