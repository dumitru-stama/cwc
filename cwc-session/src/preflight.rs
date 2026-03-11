use cwc_core::traits::TokenCounter;
use serde::{Deserialize, Serialize};

use crate::budget::SessionBudget;
use crate::error::Result;
use crate::memory::{create_memory_message, SessionMemoryStore};
use crate::message::{MessageFlags, Session, SessionMessage, SessionRole};
use crate::reset::hard_reset;

/// Issues found during pre-flight validation.
#[derive(Debug, Clone)]
pub struct PreflightIssue {
    pub kind: PreflightIssueKind,
    pub description: String,
    pub auto_repaired: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreflightIssueKind {
    /// messages[0] is not System role.
    MissingSystemPrompt,
    /// No real user message exists.
    NoUserMessage,
    /// Orphaned tool result (no preceding tool_call with matching call_id).
    OrphanedToolResult,
    /// Token count exceeds usable budget.
    OverBudget,
    /// Same tool called N times consecutively (possible loop).
    ToolLoop,
    /// Too many consecutive tool calls without a user message.
    RunawayToolUse,
    /// Memory message missing or stale.
    StaleMemory,
}

/// Configuration for pre-flight validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreflightConfig {
    pub max_consecutive_same_tool: usize,
    pub max_tool_calls_without_user: usize,
    pub default_system_prompt: String,
}

impl Default for PreflightConfig {
    fn default() -> Self {
        Self {
            max_consecutive_same_tool: 3,
            max_tool_calls_without_user: 15,
            default_system_prompt: "You are a helpful assistant.".to_string(),
        }
    }
}

/// Validate and repair a conversation before sending to the LLM.
///
/// Checks:
/// 1. messages[0] is System role
/// 2. At least 1 real user message exists
/// 3. No orphaned tool results
/// 4. Total tokens < usable budget
/// 5. No tool loops (same tool called too many times)
/// 6. No runaway tool use (too many tool calls without user message)
/// 7. Memory message present and fresh (if store is non-empty)
///
/// Auto-repairs where possible.
pub fn preflight_check(
    session: &mut Session,
    memory_store: &mut SessionMemoryStore,
    budget: &SessionBudget,
    tokenizer: &dyn TokenCounter,
    config: &PreflightConfig,
) -> Result<Vec<PreflightIssue>> {
    let mut issues = Vec::new();

    // 1. Missing system prompt
    check_system_prompt(session, config, &mut issues);

    // 2. No user message
    check_user_message(session, &mut issues);

    // 3. Orphaned tool results
    check_orphaned_tool_results(session, &mut issues);

    // 4. Over budget → hard reset
    check_over_budget(session, memory_store, budget, tokenizer, &mut issues)?;

    // 5. Tool loop detection
    check_tool_loop(session, config, &mut issues);

    // 6. Runaway tool use
    check_runaway_tool_use(session, config, &mut issues);

    // 7. Stale memory
    check_stale_memory(session, memory_store, tokenizer, &mut issues);

    Ok(issues)
}

fn check_system_prompt(
    session: &mut Session,
    config: &PreflightConfig,
    issues: &mut Vec<PreflightIssue>,
) {
    if session.is_empty() || session.messages()[0].role != SessionRole::System {
        let sys = SessionMessage::system(&config.default_system_prompt);
        session.messages_mut().insert(0, sys);
        session.recalculate_total();
        issues.push(PreflightIssue {
            kind: PreflightIssueKind::MissingSystemPrompt,
            description: "Injected default system prompt at position 0".into(),
            auto_repaired: true,
        });
    }
}

fn check_user_message(session: &Session, issues: &mut Vec<PreflightIssue>) {
    let has_real_user = session.messages().iter().any(|m| {
        m.role == SessionRole::User
            && !m.flags.contains(MessageFlags::IS_NUDGE)
            && !m.flags.contains(MessageFlags::IS_MEMORY)
    });
    if !has_real_user {
        issues.push(PreflightIssue {
            kind: PreflightIssueKind::NoUserMessage,
            description: "No real user message found".into(),
            auto_repaired: false,
        });
    }
}

fn check_orphaned_tool_results(session: &mut Session, issues: &mut Vec<PreflightIssue>) {
    // Collect all call_ids from assistant tool_calls
    let mut known_call_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    for msg in session.messages() {
        if msg.role == SessionRole::Assistant {
            for tc in &msg.tool_calls {
                known_call_ids.insert(tc.call_id.clone());
            }
        }
    }

    // Find orphaned tool results
    let orphaned_indices: Vec<usize> = session
        .messages()
        .iter()
        .enumerate()
        .filter(|(_, m)| {
            if let Some(ref tr) = m.tool_result {
                !known_call_ids.contains(&tr.call_id)
            } else {
                false
            }
        })
        .map(|(i, _)| i)
        .collect();

    if !orphaned_indices.is_empty() {
        // Remove orphaned messages (reverse order to preserve indices)
        for &idx in orphaned_indices.iter().rev() {
            session.messages_mut().remove(idx);
        }
        session.recalculate_total();
        issues.push(PreflightIssue {
            kind: PreflightIssueKind::OrphanedToolResult,
            description: format!("Removed {} orphaned tool result(s)", orphaned_indices.len()),
            auto_repaired: true,
        });
    }
}

fn check_over_budget(
    session: &mut Session,
    memory_store: &mut SessionMemoryStore,
    budget: &SessionBudget,
    tokenizer: &dyn TokenCounter,
    issues: &mut Vec<PreflightIssue>,
) -> Result<()> {
    let tokens_before = session.total_tokens();
    if tokens_before >= budget.usable_budget {
        let result = hard_reset(session.messages(), memory_store, budget, tokenizer)?;
        session.replace(result.messages);
        issues.push(PreflightIssue {
            kind: PreflightIssueKind::OverBudget,
            description: format!(
                "Over budget ({} >= {}), forced hard reset → {} tokens",
                tokens_before,
                budget.usable_budget,
                session.total_tokens()
            ),
            auto_repaired: true,
        });
    }
    Ok(())
}

fn check_tool_loop(
    session: &mut Session,
    config: &PreflightConfig,
    issues: &mut Vec<PreflightIssue>,
) {
    if let Some((tool_name, count)) = detect_tool_loop(session.messages(), config.max_consecutive_same_tool) {
        // Inject corrective message
        let mut corrective = SessionMessage::text(
            SessionRole::User,
            "You are repeating. Try a different approach.",
        );
        corrective.token_count = 8; // Approximate
        session.messages_mut().push(corrective);
        session.recalculate_total();
        issues.push(PreflightIssue {
            kind: PreflightIssueKind::ToolLoop,
            description: format!(
                "Tool '{}' called {} consecutive times, injected corrective message",
                tool_name, count
            ),
            auto_repaired: true,
        });
    }
}

fn check_runaway_tool_use(
    session: &Session,
    config: &PreflightConfig,
    issues: &mut Vec<PreflightIssue>,
) {
    let tool_count = count_consecutive_tools(session.messages());
    if tool_count > config.max_tool_calls_without_user {
        issues.push(PreflightIssue {
            kind: PreflightIssueKind::RunawayToolUse,
            description: format!(
                "{} consecutive tool calls without user message (max {})",
                tool_count, config.max_tool_calls_without_user
            ),
            auto_repaired: false,
        });
    }
}

fn check_stale_memory(
    session: &mut Session,
    memory_store: &SessionMemoryStore,
    tokenizer: &dyn TokenCounter,
    issues: &mut Vec<PreflightIssue>,
) {
    if memory_store.is_empty() {
        return;
    }

    // Check if there's an IS_MEMORY message at position 1
    let has_memory = session.messages().len() > 1
        && session.messages()[1]
            .flags
            .contains(MessageFlags::IS_MEMORY);

    if has_memory {
        // Check if it's stale by comparing content
        let fresh = create_memory_message(memory_store, 512, tokenizer);
        if session.messages()[1].content != fresh.content {
            // Replace stale memory
            let mut new_mem = fresh;
            new_mem.token_count = tokenizer.count_tokens(&new_mem.content) + 4;
            session.messages_mut()[1] = new_mem;
            session.recalculate_total();
            issues.push(PreflightIssue {
                kind: PreflightIssueKind::StaleMemory,
                description: "Replaced stale memory message with fresh render".into(),
                auto_repaired: true,
            });
        }
    } else {
        // Insert fresh memory at position 1
        let mut mem = create_memory_message(memory_store, 512, tokenizer);
        if !mem.content.is_empty() {
            mem.token_count = tokenizer.count_tokens(&mem.content) + 4;
            let pos = if session.messages().is_empty() { 0 } else { 1 };
            session.messages_mut().insert(pos, mem);
            session.recalculate_total();
            issues.push(PreflightIssue {
                kind: PreflightIssueKind::StaleMemory,
                description: "Inserted missing memory message at position 1".into(),
                auto_repaired: true,
            });
        }
    }
}

/// Detect if the same tool has been called more than `max_consecutive` times at the end.
/// Returns (tool_name, actual_count) if a loop is detected.
fn detect_tool_loop(messages: &[SessionMessage], max_consecutive: usize) -> Option<(String, usize)> {
    // Extract sequence of tool names from assistant tool_call messages
    let tool_names: Vec<&str> = messages
        .iter()
        .filter(|m| m.role == SessionRole::Assistant && !m.tool_calls.is_empty())
        .filter_map(|m| m.tool_calls.first().map(|tc| tc.tool_name.as_str()))
        .collect();

    if tool_names.is_empty() {
        return None;
    }

    let last = *tool_names.last()?;
    let count = tool_names
        .iter()
        .rev()
        .take_while(|&&name| name == last)
        .count();

    if count > max_consecutive {
        Some((last.to_string(), count))
    } else {
        None
    }
}

/// Count consecutive Tool-role messages from the end without a real User message.
fn count_consecutive_tools(messages: &[SessionMessage]) -> usize {
    let mut count = 0;
    for msg in messages.iter().rev() {
        match msg.role {
            SessionRole::Tool => count += 1,
            SessionRole::Assistant if !msg.tool_calls.is_empty() => {
                // Tool-calling assistant message — part of the tool sequence
            }
            SessionRole::User
                if !msg.flags.contains(MessageFlags::IS_NUDGE)
                    && !msg.flags.contains(MessageFlags::IS_MEMORY) =>
            {
                break;
            }
            _ => {}
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::{FactPriority, MemoryFact, MemorySource};
    use crate::message::{ToolCall, ToolResult};
    use std::sync::Arc;

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
    fn tc_arc() -> Arc<dyn TokenCounter> {
        Arc::new(WordCounter)
    }
    fn tc() -> &'static dyn TokenCounter {
        &WordCounter
    }

    fn test_budget() -> SessionBudget {
        SessionBudget {
            context_window: 32768,
            max_output: 4096,
            usable_budget: 10000,
            sliding_window_trigger: 5000,
            hard_reset_trigger: 6000,
            compaction_trigger: 8500,
            tail_tokens: 2000,
            min_tail_turns: 2,
        }
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

    #[test]
    fn test_preflight_valid_conversation() {
        let mut session = Session::new(tc_arc());
        session.push(SessionMessage::system("You are helpful"));
        session.push(SessionMessage::text(SessionRole::User, "hello"));
        session.push(SessionMessage::text(SessionRole::Assistant, "hi"));

        let mut store = SessionMemoryStore::default();
        let config = PreflightConfig::default();
        let budget = test_budget();

        let issues = preflight_check(&mut session, &mut store, &budget, tc(), &config).unwrap();
        assert!(issues.is_empty(), "valid conversation should have no issues: {issues:?}");
    }

    #[test]
    fn test_preflight_missing_system_prompt() {
        let mut session = Session::new(tc_arc());
        session.push(SessionMessage::text(SessionRole::User, "hello"));

        let mut store = SessionMemoryStore::default();
        let config = PreflightConfig::default();
        let budget = test_budget();

        let issues = preflight_check(&mut session, &mut store, &budget, tc(), &config).unwrap();
        assert!(
            issues.iter().any(|i| i.kind == PreflightIssueKind::MissingSystemPrompt && i.auto_repaired),
            "should detect and repair missing system prompt"
        );
        assert_eq!(session.messages()[0].role, SessionRole::System);
    }

    #[test]
    fn test_preflight_no_user_message() {
        let mut session = Session::new(tc_arc());
        session.push(SessionMessage::system("sys"));
        session.push(SessionMessage::text(SessionRole::Assistant, "hello"));

        let mut store = SessionMemoryStore::default();
        let config = PreflightConfig::default();
        let budget = test_budget();

        let issues = preflight_check(&mut session, &mut store, &budget, tc(), &config).unwrap();
        assert!(
            issues.iter().any(|i| i.kind == PreflightIssueKind::NoUserMessage && !i.auto_repaired),
            "should detect missing user message (not repairable)"
        );
    }

    #[test]
    fn test_preflight_orphaned_tool_result() {
        let mut session = Session::new(tc_arc());
        session.push(SessionMessage::system("sys"));
        session.push(SessionMessage::text(SessionRole::User, "hello"));
        // Orphaned tool result — no matching tool_call
        session.push(SessionMessage::tool_result(ToolResult {
            call_id: "orphan_call".into(),
            tool_name: "grep".into(),
            output: "results".into(),
            is_error: false,
        }));

        let mut store = SessionMemoryStore::default();
        let config = PreflightConfig::default();
        let budget = test_budget();

        let len_before = session.len();
        let issues = preflight_check(&mut session, &mut store, &budget, tc(), &config).unwrap();
        assert!(
            issues.iter().any(|i| i.kind == PreflightIssueKind::OrphanedToolResult && i.auto_repaired),
            "should detect and remove orphaned tool result"
        );
        assert!(session.len() < len_before, "orphaned message should be removed");
    }

    #[test]
    fn test_preflight_over_budget() {
        let mut session = Session::new(tc_arc());
        session.push(SessionMessage::system("sys"));
        // Many large turns that collectively exceed the budget
        for i in 0..10 {
            let mut u = SessionMessage::text(SessionRole::User, format!("question {i}"));
            u.token_count = 500;
            session.messages_mut().push(u);
            let mut a = SessionMessage::text(SessionRole::Assistant, format!("answer {i}"));
            a.token_count = 500;
            session.messages_mut().push(a);
        }
        session.recalculate_total();
        // Total: sys tokens + 10*1000 > 10000

        let mut store = SessionMemoryStore::default();
        let config = PreflightConfig::default();
        let budget = test_budget();
        assert!(session.total_tokens() >= budget.usable_budget);

        let issues = preflight_check(&mut session, &mut store, &budget, tc(), &config).unwrap();
        assert!(
            issues.iter().any(|i| i.kind == PreflightIssueKind::OverBudget && i.auto_repaired),
            "should detect over-budget and hard reset"
        );
        // After hard reset, should be much smaller (system + memory + last 2 turns)
        assert!(
            session.total_tokens() < budget.usable_budget,
            "after reset, tokens {} should be < budget {}",
            session.total_tokens(),
            budget.usable_budget
        );
    }

    #[test]
    fn test_preflight_tool_loop() {
        let mut session = Session::new(tc_arc());
        session.push(SessionMessage::system("sys"));
        session.push(SessionMessage::text(SessionRole::User, "find bugs"));
        // Same tool called 4 times (exceeds default max of 3)
        for _ in 0..4 {
            session.push(SessionMessage::assistant_tool_calls(
                "",
                vec![ToolCall {
                    call_id: "c1".into(),
                    tool_name: "file.grep".into(),
                    arguments: serde_json::json!({"q": "bug"}),
                }],
            ));
            session.push(SessionMessage::tool_result(ToolResult {
                call_id: "c1".into(),
                tool_name: "file.grep".into(),
                output: "results".into(),
                is_error: false,
            }));
        }

        let mut store = SessionMemoryStore::default();
        let config = PreflightConfig::default();
        let budget = test_budget();

        let issues = preflight_check(&mut session, &mut store, &budget, tc(), &config).unwrap();
        assert!(
            issues.iter().any(|i| i.kind == PreflightIssueKind::ToolLoop),
            "should detect tool loop"
        );
    }

    #[test]
    fn test_preflight_runaway_tool_use() {
        let mut session = Session::new(tc_arc());
        session.push(SessionMessage::system("sys"));
        session.push(SessionMessage::text(SessionRole::User, "analyze everything"));
        // 16 consecutive tool calls (exceeds default max of 15)
        for i in 0..16 {
            session.push(SessionMessage::assistant_tool_calls(
                "",
                vec![ToolCall {
                    call_id: format!("c{i}"),
                    tool_name: format!("tool_{}", i % 5),
                    arguments: serde_json::json!({}),
                }],
            ));
            session.push(SessionMessage::tool_result(ToolResult {
                call_id: format!("c{i}"),
                tool_name: format!("tool_{}", i % 5),
                output: "ok".into(),
                is_error: false,
            }));
        }

        let mut store = SessionMemoryStore::default();
        let config = PreflightConfig::default();
        let budget = test_budget();

        let issues = preflight_check(&mut session, &mut store, &budget, tc(), &config).unwrap();
        assert!(
            issues.iter().any(|i| i.kind == PreflightIssueKind::RunawayToolUse),
            "should detect runaway tool use"
        );
    }

    #[test]
    fn test_preflight_stale_memory() {
        let mut session = Session::new(tc_arc());
        session.push(SessionMessage::system("sys"));
        // Insert an outdated memory message
        let mut old_mem = SessionMessage::text(SessionRole::User, "[WORKING_MEMORY]\nGoal: old goal");
        old_mem.flags.insert(MessageFlags::IS_MEMORY);
        session.messages_mut().push(old_mem);
        session.recalculate_total();
        session.push(SessionMessage::text(SessionRole::User, "hello"));

        let mut store = SessionMemoryStore::default();
        store.upsert(make_fact("goal", "new goal", FactPriority::Goal));
        let config = PreflightConfig::default();
        let budget = test_budget();

        let issues = preflight_check(&mut session, &mut store, &budget, tc(), &config).unwrap();
        assert!(
            issues.iter().any(|i| i.kind == PreflightIssueKind::StaleMemory && i.auto_repaired),
            "should detect and replace stale memory"
        );
        assert!(session.messages()[1].content.contains("new goal"));
    }

    #[test]
    fn test_preflight_loop_injects_corrective() {
        let mut session = Session::new(tc_arc());
        session.push(SessionMessage::system("sys"));
        session.push(SessionMessage::text(SessionRole::User, "search"));
        for _ in 0..4 {
            session.push(SessionMessage::assistant_tool_calls(
                "",
                vec![ToolCall {
                    call_id: "c1".into(),
                    tool_name: "file.grep".into(),
                    arguments: serde_json::json!({}),
                }],
            ));
            session.push(SessionMessage::tool_result(ToolResult {
                call_id: "c1".into(),
                tool_name: "file.grep".into(),
                output: "same results".into(),
                is_error: false,
            }));
        }

        let mut store = SessionMemoryStore::default();
        let config = PreflightConfig::default();
        let budget = test_budget();
        let len_before = session.len();

        preflight_check(&mut session, &mut store, &budget, tc(), &config).unwrap();
        assert!(session.len() > len_before, "should inject corrective message");
        let last = session.messages().last().unwrap();
        assert_eq!(last.role, SessionRole::User);
        assert!(last.content.contains("different approach"));
    }

    #[test]
    fn test_preflight_over_budget_reports_pre_reset_tokens() {
        let mut session = Session::new(tc_arc());
        session.push(SessionMessage::system("sys"));
        for i in 0..10 {
            let mut u = SessionMessage::text(SessionRole::User, format!("question {i}"));
            u.token_count = 500;
            session.messages_mut().push(u);
            let mut a = SessionMessage::text(SessionRole::Assistant, format!("answer {i}"));
            a.token_count = 500;
            session.messages_mut().push(a);
        }
        session.recalculate_total();
        let pre_tokens = session.total_tokens();

        let mut store = SessionMemoryStore::default();
        let config = PreflightConfig::default();
        let budget = test_budget();

        let issues = preflight_check(&mut session, &mut store, &budget, tc(), &config).unwrap();
        let over_budget = issues.iter().find(|i| i.kind == PreflightIssueKind::OverBudget).unwrap();
        // Description should contain pre-reset token count
        assert!(
            over_budget.description.contains(&pre_tokens.to_string()),
            "description '{}' should contain pre-reset tokens {}",
            over_budget.description, pre_tokens
        );
    }

    #[test]
    fn test_detect_tool_loop_returns_actual_count() {
        // 8 consecutive calls to the same tool
        let mut messages = vec![
            SessionMessage::system("sys"),
            SessionMessage::text(SessionRole::User, "go"),
        ];
        for _ in 0..8 {
            messages.push(SessionMessage::assistant_tool_calls(
                "",
                vec![ToolCall {
                    call_id: "c1".into(),
                    tool_name: "file.grep".into(),
                    arguments: serde_json::json!({}),
                }],
            ));
            messages.push(SessionMessage::tool_result(ToolResult {
                call_id: "c1".into(),
                tool_name: "file.grep".into(),
                output: "ok".into(),
                is_error: false,
            }));
        }
        let result = detect_tool_loop(&messages, 3);
        let (name, count) = result.expect("should detect loop");
        assert_eq!(name, "file.grep");
        assert_eq!(count, 8, "should report actual count, not threshold");
    }

    #[test]
    fn test_preflight_empty_session_gets_system_prompt() {
        let mut session = Session::new(tc_arc());
        let mut store = SessionMemoryStore::default();
        let config = PreflightConfig::default();
        let budget = test_budget();

        let issues = preflight_check(&mut session, &mut store, &budget, tc(), &config).unwrap();
        assert!(issues.iter().any(|i| i.kind == PreflightIssueKind::MissingSystemPrompt));
        assert!(issues.iter().any(|i| i.kind == PreflightIssueKind::NoUserMessage));
        assert_eq!(session.messages()[0].role, SessionRole::System);
    }
}
