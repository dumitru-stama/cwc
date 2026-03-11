use cwc_core::traits::TokenCounter;

use crate::budget::SessionBudget;
use crate::error::Result;
use crate::memory::{extract_from_turns, extract_goal, MemoryFact, SessionMemoryStore};
use crate::message::{MessageFlags, SessionMessage};
use crate::rebuild::rebuild_conversation;
use crate::turn::{parse_turns, Turn};

/// Result of a sliding window trim operation.
#[derive(Debug, Clone)]
pub struct TrimResult {
    /// Messages to keep (new conversation state).
    pub kept_messages: Vec<SessionMessage>,
    /// Facts extracted from dropped messages (to inject into memory).
    pub extracted_facts: Vec<MemoryFact>,
    /// Number of turns dropped.
    pub turns_dropped: usize,
    /// Tokens before trim.
    pub tokens_before: u32,
    /// Tokens after trim.
    pub tokens_after: u32,
}

/// Token-budget sliding window trimmer.
///
/// Algorithm:
/// 1. Parse messages into atomic turns
/// 2. Walk turns back-to-front, accumulating token count
/// 3. Stop when ALL of: accumulated >= tail_tokens, min_tail_turns met, 1 user request
/// 4. Everything before boundary = "to drop"
/// 5. Extract memory from dropped turns
/// 6. Rebuild: system prompt + memory message + kept turns
pub fn sliding_window_trim(
    messages: &[SessionMessage],
    budget: &SessionBudget,
    memory_store: &mut SessionMemoryStore,
    tokenizer: &dyn TokenCounter,
) -> Result<TrimResult> {
    let tokens_before: u32 = messages.iter().map(|m| m.token_count).sum();

    // Under threshold — no trim needed
    if tokens_before < budget.sliding_window_trigger {
        return Ok(TrimResult {
            kept_messages: messages.to_vec(),
            extracted_facts: Vec::new(),
            turns_dropped: 0,
            tokens_before,
            tokens_after: tokens_before,
        });
    }

    let turns = parse_turns(messages);

    let boundary = match find_trim_boundary(&turns, budget.tail_tokens, budget.min_tail_turns) {
        Some(b) => b,
        None => {
            return Ok(TrimResult {
                kept_messages: messages.to_vec(),
                extracted_facts: Vec::new(),
                turns_dropped: 0,
                tokens_before,
                tokens_after: tokens_before,
            });
        }
    };

    // Split into dropped and kept turns
    let dropped_turns = &turns[1..boundary]; // Skip system prompt (turn 0)
    let kept_turns = &turns[boundary..];

    // Extract memory from dropped turns before removing
    let mut extracted_facts = extract_from_turns(messages, dropped_turns);

    // Extract goal from full messages if not already in store
    if memory_store.get("goal").is_none() {
        if let Some(goal) = extract_goal(messages) {
            extracted_facts.push(goal.clone());
            memory_store.upsert(goal);
        }
    }

    // Upsert extracted facts into memory store
    memory_store.upsert_batch(extracted_facts.clone());
    memory_store.enforce_limits();

    // Collect tail messages from kept turns, filtering out nudges
    let system_prompt = &messages[0];
    let tail_start = kept_turns
        .first()
        .map(|t| t.start)
        .unwrap_or(messages.len());
    let tail_messages: Vec<SessionMessage> = messages[tail_start..]
        .iter()
        .filter(|m| !m.flags.contains(MessageFlags::IS_NUDGE))
        .cloned()
        .collect();

    // Rebuild
    let kept_messages = rebuild_conversation(system_prompt, memory_store, tail_messages, tokenizer);
    let tokens_after: u32 = kept_messages.iter().map(|m| m.token_count).sum();

    Ok(TrimResult {
        kept_messages,
        extracted_facts,
        turns_dropped: dropped_turns.len(),
        tokens_before,
        tokens_after,
    })
}

/// Find the trim boundary: returns the index of the first turn to KEEP.
///
/// Returns None if no trimming is possible (not enough turns to drop).
fn find_trim_boundary(
    turns: &[Turn],
    tail_tokens: u32,
    min_tail_turns: usize,
) -> Option<usize> {
    // Need system + at least 2 content turns to drop anything
    if turns.len() < 3 {
        return None;
    }

    let mut acc_tokens: u32 = 0;
    let mut kept: usize = 0;
    let mut have_user = false;

    // Walk from end to just after system prompt
    for i in (1..turns.len()).rev() {
        acc_tokens += turns[i].tokens;
        kept += 1;
        if turns[i].has_user_request {
            have_user = true;
        }

        if acc_tokens >= tail_tokens && kept >= min_tail_turns && have_user {
            if i > 1 {
                return Some(i);
            }
            return None; // Would need to keep everything
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::SessionBudget;
    use crate::message::{SessionRole, ToolCall, ToolResult};

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

    fn nudge_msg(tokens: u32) -> SessionMessage {
        let mut m = SessionMessage::nudge("Continue.");
        m.token_count = tokens;
        m
    }

    fn test_budget(sliding_trigger: u32, tail_tokens: u32, min_turns: usize) -> SessionBudget {
        SessionBudget {
            context_window: 32768,
            max_output: 4096,
            usable_budget: 20000,
            sliding_window_trigger: sliding_trigger,
            hard_reset_trigger: sliding_trigger + 2000,
            compaction_trigger: sliding_trigger + 5000,
            tail_tokens,
            min_tail_turns: min_turns,
        }
    }

    #[test]
    fn test_sliding_window_under_threshold() {
        let messages = vec![
            sys_msg(50),
            msg(SessionRole::User, "hello", 10),
            msg(SessionRole::Assistant, "hi", 8),
        ];
        let budget = test_budget(5000, 500, 2);
        let mut store = SessionMemoryStore::default();

        let result = sliding_window_trim(&messages, &budget, &mut store, tc()).unwrap();
        assert_eq!(result.turns_dropped, 0);
        assert!(result.extracted_facts.is_empty());
        assert_eq!(result.kept_messages.len(), messages.len());
        assert_eq!(result.tokens_before, result.tokens_after);
    }

    #[test]
    fn test_sliding_window_over_threshold_drops_oldest() {
        // sys(50) + 5 turns of 200 each = 1050 total
        let mut messages = vec![sys_msg(50)];
        for i in 0..5 {
            messages.push(msg(SessionRole::User, &format!("question {i}"), 80));
            messages.push(msg(SessionRole::Assistant, &format!("answer {i}"), 120));
        }
        // Total: 50 + 5*200 = 1050
        let budget = test_budget(500, 300, 1); // trigger at 500, keep 300 tokens tail
        let mut store = SessionMemoryStore::default();

        let result = sliding_window_trim(&messages, &budget, &mut store, tc()).unwrap();
        assert!(result.turns_dropped > 0);
        assert!(result.tokens_after < result.tokens_before);
    }

    #[test]
    fn test_sliding_window_preserves_system_prompt() {
        let mut messages = vec![sys_msg(50)];
        for i in 0..5 {
            messages.push(msg(SessionRole::User, &format!("q{i}"), 100));
            messages.push(msg(SessionRole::Assistant, &format!("a{i}"), 100));
        }
        let budget = test_budget(500, 150, 1);
        let mut store = SessionMemoryStore::default();

        let result = sliding_window_trim(&messages, &budget, &mut store, tc()).unwrap();
        assert_eq!(result.kept_messages[0].role, SessionRole::System);
        assert!(result.kept_messages[0].flags.contains(MessageFlags::PRESERVE));
    }

    #[test]
    fn test_sliding_window_preserves_min_tail_turns() {
        let mut messages = vec![sys_msg(50)];
        for i in 0..5 {
            messages.push(msg(SessionRole::User, &format!("q{i}"), 100));
            messages.push(msg(SessionRole::Assistant, &format!("a{i}"), 100));
        }
        // Request min 3 tail turns with low tail_tokens so the trigger is met easily
        let budget = test_budget(500, 50, 3);
        let mut store = SessionMemoryStore::default();

        let result = sliding_window_trim(&messages, &budget, &mut store, tc()).unwrap();
        // Count user-request turns in the result (excluding system and memory)
        let user_turns: usize = result
            .kept_messages
            .iter()
            .filter(|m| {
                m.role == SessionRole::User
                    && !m.flags.contains(MessageFlags::IS_MEMORY)
                    && !m.flags.contains(MessageFlags::IS_NUDGE)
            })
            .count();
        assert!(user_turns >= 3, "got {user_turns} user turns, expected >= 3");
    }

    #[test]
    fn test_sliding_window_preserves_user_request() {
        let messages = vec![
            sys_msg(50),
            msg(SessionRole::User, "question", 100),
            msg(SessionRole::Assistant, "answer", 100),
            // Nudge-only turns with no real user request
            nudge_msg(50),
            msg(SessionRole::User, "real question", 100),
            msg(SessionRole::Assistant, "real answer", 100),
        ];
        let budget = test_budget(200, 100, 1);
        let mut store = SessionMemoryStore::default();

        let result = sliding_window_trim(&messages, &budget, &mut store, tc()).unwrap();
        // Must have at least one real user message
        let has_user = result.kept_messages.iter().any(|m| {
            m.role == SessionRole::User
                && !m.flags.contains(MessageFlags::IS_MEMORY)
                && !m.flags.contains(MessageFlags::IS_NUDGE)
        });
        assert!(has_user, "must preserve at least one user request");
    }

    #[test]
    fn test_sliding_window_extracts_memory_from_dropped() {
        let messages = vec![
            sys_msg(50),
            msg(SessionRole::User, "find malloc bugs", 100),
            tool_call_msg("file.grep", 20),
            tool_result_msg("file.grep", "src/main.rs:10: malloc(256)", 200),
            msg(SessionRole::Assistant, "Found malloc usage", 50),
            msg(SessionRole::User, "now fix them", 100),
            msg(SessionRole::Assistant, "I will fix them", 100),
        ];
        // Trigger at 300 tokens, keep 200 tail → should drop turn with grep
        let budget = test_budget(300, 200, 1);
        let mut store = SessionMemoryStore::default();

        let result = sliding_window_trim(&messages, &budget, &mut store, tc()).unwrap();
        assert!(!result.extracted_facts.is_empty(), "should extract facts from dropped turns");
        // Memory store should have the goal
        assert!(store.get("goal").is_some(), "goal should be extracted");
    }

    #[test]
    fn test_sliding_window_inserts_memory_at_position_1() {
        let mut messages = vec![sys_msg(50)];
        for i in 0..5 {
            messages.push(msg(SessionRole::User, &format!("q{i}"), 100));
            messages.push(msg(SessionRole::Assistant, &format!("a{i}"), 100));
        }
        let budget = test_budget(500, 150, 1);
        let mut store = SessionMemoryStore::default();

        let result = sliding_window_trim(&messages, &budget, &mut store, tc()).unwrap();
        // If memory was extracted, position 1 should be a memory message
        if result.turns_dropped > 0 && !store.is_empty() {
            assert!(
                result.kept_messages[1].flags.contains(MessageFlags::IS_MEMORY),
                "memory message should be at position 1"
            );
        }
    }

    #[test]
    fn test_sliding_window_never_splits_turn() {
        // Create a multi-message turn: user + tool_call + tool_result + assistant
        let messages = vec![
            sys_msg(50),
            msg(SessionRole::User, "q1", 100),
            msg(SessionRole::Assistant, "a1", 100),
            msg(SessionRole::User, "analyze", 100),
            tool_call_msg("file.grep", 20),
            tool_result_msg("file.grep", "results", 200),
            msg(SessionRole::Assistant, "done", 50),
            msg(SessionRole::User, "next", 100),
            msg(SessionRole::Assistant, "ok", 100),
        ];
        let budget = test_budget(400, 200, 1);
        let mut store = SessionMemoryStore::default();

        let result = sliding_window_trim(&messages, &budget, &mut store, tc()).unwrap();
        // The tool turn (messages 3-6) should be either fully kept or fully dropped
        let has_tool_call = result
            .kept_messages
            .iter()
            .any(|m| !m.tool_calls.is_empty());
        let has_tool_result = result.kept_messages.iter().any(|m| m.tool_result.is_some());
        // If we have a tool call, we must have the result too
        if has_tool_call {
            assert!(has_tool_result, "tool call and result must stay together");
        }
    }

    #[test]
    fn test_sliding_window_drops_nudge_turns() {
        let messages = vec![
            sys_msg(50),
            msg(SessionRole::User, "q1", 100),
            msg(SessionRole::Assistant, "a1", 100),
            msg(SessionRole::User, "q2", 100),
            msg(SessionRole::Assistant, "a2", 100),
            nudge_msg(10), // nudge in the tail
            msg(SessionRole::User, "q3", 100),
            msg(SessionRole::Assistant, "a3", 100),
        ];
        let budget = test_budget(300, 250, 2);
        let mut store = SessionMemoryStore::default();

        let result = sliding_window_trim(&messages, &budget, &mut store, tc()).unwrap();
        // Nudge should be filtered from the tail
        let nudge_count = result
            .kept_messages
            .iter()
            .filter(|m| m.flags.contains(MessageFlags::IS_NUDGE))
            .count();
        assert_eq!(nudge_count, 0, "nudge turns should be dropped from tail");
    }

    #[test]
    fn test_sliding_window_tokens_after_less_than_before() {
        let mut messages = vec![sys_msg(50)];
        for i in 0..10 {
            messages.push(msg(SessionRole::User, &format!("question {i}"), 100));
            messages.push(msg(SessionRole::Assistant, &format!("answer {i}"), 100));
        }
        let budget = test_budget(500, 300, 1);
        let mut store = SessionMemoryStore::default();

        let result = sliding_window_trim(&messages, &budget, &mut store, tc()).unwrap();
        if result.turns_dropped > 0 {
            assert!(
                result.tokens_after < result.tokens_before,
                "tokens_after {} should be < tokens_before {}",
                result.tokens_after,
                result.tokens_before
            );
        }
    }

    #[test]
    fn test_find_trim_boundary_one_turn() {
        let turns = vec![
            Turn {
                start: 0, end: 0, tokens: 50, has_user_request: false,
                has_tool_calls: false, message_count: 1,
            },
            Turn {
                start: 1, end: 2, tokens: 100, has_user_request: true,
                has_tool_calls: false, message_count: 2,
            },
        ];
        assert!(find_trim_boundary(&turns, 6000, 2).is_none());
    }

    #[test]
    fn test_find_trim_boundary_ten_turns() {
        let mut turns = vec![Turn {
            start: 0, end: 0, tokens: 50, has_user_request: false,
            has_tool_calls: false, message_count: 1,
        }];
        for i in 0..10 {
            let start = 1 + i * 2;
            turns.push(Turn {
                start,
                end: start + 1,
                tokens: 1000,
                has_user_request: true,
                has_tool_calls: false,
                message_count: 2,
            });
        }
        // tail_tokens=6000, min_tail_turns=2
        let boundary = find_trim_boundary(&turns, 6000, 2).unwrap();
        // Walking back: need >= 6000 tokens and >= 2 turns with user request
        // Last 6 turns give 6000 tokens
        assert!(boundary > 1, "should drop at least 1 turn");
        assert!(boundary <= turns.len());
        // Verify the kept turns have >= 6000 tokens
        let kept_tokens: u32 = turns[boundary..].iter().map(|t| t.tokens).sum();
        assert!(kept_tokens >= 6000, "kept {kept_tokens} < 6000");
    }

    // === Integration tests ===

    #[test]
    fn test_full_cycle_20_tool_turns() {
        // Create system + 20 tool-use turns
        let mut messages = vec![sys_msg(50)];
        for i in 0..20 {
            messages.push(msg(SessionRole::User, &format!("task {i}"), 50));
            messages.push(tool_call_msg("build", 20));
            messages.push(tool_result_msg(
                "build",
                &format!("Compiling... Finished ({i})"),
                200,
            ));
            messages.push(msg(SessionRole::Assistant, &format!("Build {i} done"), 30));
        }
        // Total: 50 + 20*(50+20+200+30) = 50 + 6000 = 6050
        let budget = test_budget(3000, 1000, 2);
        let mut store = SessionMemoryStore::default();

        let result = sliding_window_trim(&messages, &budget, &mut store, tc()).unwrap();
        assert!(result.turns_dropped > 0, "should trim turns");
        assert!(result.tokens_after < result.tokens_before, "should reduce tokens");
        assert!(!store.is_empty(), "should have memory");
        assert!(store.get("goal").is_some(), "should extract goal");
    }

    #[test]
    fn test_budget_action_cascade() {
        let budget = SessionBudget {
            context_window: 32768,
            max_output: 4096,
            usable_budget: 10000,
            sliding_window_trigger: 5000,
            hard_reset_trigger: 6000,
            compaction_trigger: 8500,
            tail_tokens: 1000,
            min_tail_turns: 2,
        };

        use crate::budget::BudgetAction;
        assert_eq!(budget.action_needed(2000), BudgetAction::None);
        assert_eq!(budget.action_needed(5000), BudgetAction::SlidingWindow);
        assert_eq!(budget.action_needed(6000), BudgetAction::HardReset);
        assert_eq!(budget.action_needed(8500), BudgetAction::Compaction);
    }
}
