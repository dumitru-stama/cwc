use std::path::PathBuf;
use std::sync::Arc;

use cwc_core::traits::{LlmClient, TokenCounter};
use serde::{Deserialize, Serialize};

use crate::budget::{BudgetAction, SessionBudget};
use crate::compaction::{compact_session, CompactionEngine, CompactionReport};
use crate::compaction::rules::default_rules;
use crate::config::SessionConfig;
use crate::error::Result;
use crate::memory::{extract_goal_text, SessionMemoryStore};
use crate::memory::consolidate::{consolidate_memory, ConsolidationConfig, ConsolidationReport};
use crate::memory::llm_consolidate::{LlmConsolidationConfig, LlmConsolidationReport};
use crate::message::{Session, SessionMessage};
use crate::preflight::{preflight_check, PreflightConfig, PreflightIssue};
use crate::rebuild::MEMORY_RENDER_TOKENS;
use crate::reinforcement::{should_nudge, ReinforcementConfig};
use crate::reset::{hard_reset, ResetResult};
use crate::turn::parse_turns;
use crate::window::sliding_window_trim;

/// Combined configuration for the session manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionManagerConfig {
    pub session: SessionConfig,
    #[serde(default)]
    pub reinforcement: ReinforcementConfig,
    #[serde(default)]
    pub preflight: PreflightConfig,
    #[serde(default = "default_memory_max_bytes")]
    pub memory_max_bytes: usize,
    #[serde(default = "default_memory_max_entries")]
    pub memory_max_entries: usize,
    #[serde(default = "default_artifact_dir")]
    pub artifact_dir: PathBuf,
    #[serde(default)]
    pub consolidation: ConsolidationConfig,
    #[serde(default)]
    pub llm_consolidation: LlmConsolidationConfig,
}

fn default_memory_max_bytes() -> usize {
    2048
}
fn default_memory_max_entries() -> usize {
    30
}
fn default_artifact_dir() -> PathBuf {
    PathBuf::from(".cwc/artifacts")
}

impl Default for SessionManagerConfig {
    fn default() -> Self {
        Self {
            session: SessionConfig::default(),
            reinforcement: ReinforcementConfig::default(),
            preflight: PreflightConfig::default(),
            memory_max_bytes: 2048,
            memory_max_entries: 30,
            artifact_dir: PathBuf::from(".cwc/artifacts"),
            consolidation: ConsolidationConfig::default(),
            llm_consolidation: LlmConsolidationConfig::default(),
        }
    }
}

/// Report of all actions taken during optimization.
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub tokens_saved: u32,
    pub compaction: Option<CompactionReport>,
    pub trim: TrimAction,
    pub preflight_issues: Vec<PreflightIssue>,
    pub nudge_injected: bool,
    pub memory_facts: usize,
    pub consolidation: Option<ConsolidationReport>,
}

/// What trimming action was taken.
#[derive(Debug, Clone)]
pub enum TrimAction {
    SlidingWindow { turns_dropped: usize },
    HardReset,
    None,
}

/// Current budget status.
#[derive(Debug, Clone)]
pub struct BudgetStatus {
    pub total_tokens: u32,
    pub usable_budget: u32,
    pub utilization_percent: f32,
    pub action_needed: BudgetAction,
    pub turns: usize,
    pub memory_facts: usize,
}

/// The main entry point for context window management.
///
/// Takes a conversation and produces an optimized version.
/// Handles the full lifecycle: compact → trim → validate → reinforce → consolidate → output.
/// Optionally, call `llm_consolidate()` after `optimize()` for LLM-enhanced consolidation.
pub struct SessionManager {
    budget: SessionBudget,
    compaction_engine: CompactionEngine,
    memory_store: SessionMemoryStore,
    reinforcement_config: ReinforcementConfig,
    preflight_config: PreflightConfig,
    consolidation_config: ConsolidationConfig,
    llm_consolidation_config: LlmConsolidationConfig,
    tokenizer: Arc<dyn TokenCounter>,
    goal: Option<String>,
    llm: Option<Arc<dyn LlmClient>>,
}

impl SessionManager {
    pub fn new(
        config: SessionManagerConfig,
        tokenizer: Arc<dyn TokenCounter>,
    ) -> Result<Self> {
        let budget = SessionBudget::from_config(&config.session);
        let compaction_engine = CompactionEngine::new(
            default_rules(),
            &config.artifact_dir,
            tokenizer.clone(),
        )?;
        let memory_store =
            SessionMemoryStore::new(config.memory_max_bytes, config.memory_max_entries);

        Ok(Self {
            budget,
            compaction_engine,
            memory_store,
            reinforcement_config: config.reinforcement,
            preflight_config: config.preflight,
            consolidation_config: config.consolidation,
            llm_consolidation_config: config.llm_consolidation,
            tokenizer,
            goal: None,
            llm: None,
        })
    }

    /// Set an optional LLM client for last-resort compaction.
    pub fn set_llm(&mut self, llm: Arc<dyn LlmClient>) {
        self.llm = Some(llm);
    }

    /// Full optimization pipeline:
    ///
    /// 1. Compact tool results that haven't been compacted yet
    /// 2. Check budget thresholds → trim/reset if needed
    /// 3. Pre-flight validation → repair issues
    /// 4. Inject reinforcement nudge if appropriate
    /// 5. Consolidate memory facts (merge related facts into denser summaries)
    /// 6. Return optimized conversation
    pub fn optimize(&mut self, session: &mut Session) -> Result<OptimizationReport> {
        let input_tokens = session.total_tokens();

        // Extract goal from first user message if not set
        if self.goal.is_none() {
            if let Some(goal_text) = extract_goal_text(session.messages(), 200) {
                self.goal = Some(goal_text);
            }
        }

        // 1. Compact tool results
        let compaction_report = compact_session(session, &self.compaction_engine)?;
        let compaction = if compaction_report.messages_compacted > 0 {
            Some(compaction_report)
        } else {
            None
        };

        // 2. Check budget → trim/reset
        let trim = self.apply_trim(session)?;

        // 3. Pre-flight validation
        let preflight_issues = preflight_check(
            session,
            &mut self.memory_store,
            &self.budget,
            &*self.tokenizer,
            &self.preflight_config,
        )?;

        // 4. Reinforcement nudge
        let nudge_injected = self.maybe_inject_nudge(session);

        // 5. Memory consolidation
        let consolidation_report =
            consolidate_memory(&mut self.memory_store, &self.consolidation_config);
        let consolidation = if consolidation_report.facts_consolidated > 0 {
            Some(consolidation_report)
        } else {
            None
        };

        let output_tokens = session.total_tokens();
        let tokens_saved = input_tokens.saturating_sub(output_tokens);

        Ok(OptimizationReport {
            input_tokens,
            output_tokens,
            tokens_saved,
            compaction,
            trim,
            preflight_issues,
            nudge_injected,
            memory_facts: self.memory_store.len(),
            consolidation,
        })
    }

    /// Optimize and return as a new message list (non-mutating input variant).
    pub fn optimize_messages(
        &mut self,
        messages: Vec<SessionMessage>,
    ) -> Result<(Vec<SessionMessage>, OptimizationReport)> {
        let mut session = Session::new(self.tokenizer.clone());
        for msg in messages {
            session.messages_mut().push(msg);
        }
        session.recalculate_total();
        let report = self.optimize(&mut session)?;
        Ok((session.into_messages(), report))
    }

    /// Process a single new message being appended to the session.
    pub fn process_message(
        &mut self,
        session: &mut Session,
        new_message: SessionMessage,
    ) -> Result<OptimizationReport> {
        session.push(new_message);
        self.optimize(session)
    }

    /// Get the current memory store contents.
    pub fn memory(&self) -> &SessionMemoryStore {
        &self.memory_store
    }

    /// Get the extracted goal.
    pub fn goal(&self) -> Option<&str> {
        self.goal.as_deref()
    }

    /// Run LLM-enhanced memory consolidation as an async post-step.
    ///
    /// Call this after `optimize()` when an LlmClient is available and
    /// `llm_consolidation.enabled` is true. Falls back to heuristic
    /// consolidation on LLM failure.
    pub async fn llm_consolidate(&mut self) -> LlmConsolidationReport {
        if !self.llm_consolidation_config.enabled {
            return LlmConsolidationReport::default();
        }

        let llm = match &self.llm {
            Some(llm) => llm.clone(),
            None => {
                return LlmConsolidationReport::default();
            }
        };

        crate::memory::llm_consolidate::llm_consolidate(
            &mut self.memory_store,
            &*llm,
            &self.llm_consolidation_config,
            &self.consolidation_config,
        )
        .await
    }

    /// Re-render the memory message in a message list after the memory store has changed.
    ///
    /// Call this after `llm_consolidate()` when the returned messages need updating.
    /// Looks for an IS_MEMORY message at position 1 and replaces its content with
    /// a fresh render of the memory store. Returns true if a message was updated.
    pub fn refresh_memory_in_messages(&self, messages: &mut [SessionMessage]) -> bool {
        use crate::memory::create_memory_message;
        use crate::message::MessageFlags;

        if self.memory_store.is_empty() || messages.len() < 2 {
            return false;
        }

        if messages[1].flags.contains(MessageFlags::IS_MEMORY) {
            let fresh = create_memory_message(&self.memory_store, MEMORY_RENDER_TOKENS, &*self.tokenizer);
            if messages[1].content != fresh.content {
                let mut new_mem = fresh;
                new_mem.token_count = self.tokenizer.count_tokens(&new_mem.content) + 4;
                messages[1] = new_mem;
                return true;
            }
        }
        false
    }

    /// Force a hard reset (user-triggered).
    pub fn force_reset(&mut self, session: &mut Session) -> Result<ResetResult> {
        let result = hard_reset(
            session.messages(),
            &mut self.memory_store,
            &self.budget,
            &*self.tokenizer,
        )?;
        session.replace(result.messages.clone());
        Ok(result)
    }

    /// Get current budget status.
    pub fn budget_status(&self, session: &Session) -> BudgetStatus {
        let total = session.total_tokens();
        let turns = parse_turns(session.messages()).len().saturating_sub(1); // exclude system turn
        let utilization = if self.budget.usable_budget > 0 {
            total as f32 / self.budget.usable_budget as f32 * 100.0
        } else {
            100.0
        };
        BudgetStatus {
            total_tokens: total,
            usable_budget: self.budget.usable_budget,
            utilization_percent: utilization,
            action_needed: self.budget.action_needed(total),
            turns,
            memory_facts: self.memory_store.len(),
        }
    }

    /// Get a reference to the budget.
    pub fn budget(&self) -> &SessionBudget {
        &self.budget
    }

    fn apply_trim(&mut self, session: &mut Session) -> Result<TrimAction> {
        let tokens = session.total_tokens();
        match self.budget.action_needed(tokens) {
            BudgetAction::None => Ok(TrimAction::None),
            BudgetAction::SlidingWindow => {
                let result = sliding_window_trim(
                    session.messages(),
                    &self.budget,
                    &mut self.memory_store,
                    &*self.tokenizer,
                )?;
                if result.turns_dropped > 0 {
                    session.replace(result.kept_messages);
                    Ok(TrimAction::SlidingWindow {
                        turns_dropped: result.turns_dropped,
                    })
                } else {
                    Ok(TrimAction::None)
                }
            }
            BudgetAction::HardReset => {
                let result = hard_reset(
                    session.messages(),
                    &mut self.memory_store,
                    &self.budget,
                    &*self.tokenizer,
                )?;
                session.replace(result.messages);
                Ok(TrimAction::HardReset)
            }
            BudgetAction::Compaction => {
                // Compaction threshold crossed. Hard reset is instant and free;
                // LLM compaction would require an async call which isn't available
                // in the sync optimize() pipeline. Use hard_reset as fallback.
                let result = hard_reset(
                    session.messages(),
                    &mut self.memory_store,
                    &self.budget,
                    &*self.tokenizer,
                )?;
                session.replace(result.messages);
                Ok(TrimAction::HardReset)
            }
        }
    }

    fn maybe_inject_nudge(&self, session: &mut Session) -> bool {
        if should_nudge(session.messages(), &self.reinforcement_config) {
            let count = crate::reinforcement::tool_results_since_last_nudge(session.messages());
            let mut nudge = crate::reinforcement::create_nudge(
                self.goal.as_deref(),
                count,
                &self.reinforcement_config,
            );
            nudge.token_count = self.tokenizer.count_tokens(&nudge.content) + 4;
            session.messages_mut().push(nudge);
            session.recalculate_total();
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn tc() -> Arc<dyn TokenCounter> {
        Arc::new(WordCounter)
    }

    fn test_config(dir: &std::path::Path) -> SessionManagerConfig {
        SessionManagerConfig {
            artifact_dir: dir.to_path_buf(),
            ..Default::default()
        }
    }

    fn msg(role: SessionRole, content: impl Into<String>, tokens: u32) -> SessionMessage {
        let mut m = SessionMessage::text(role, content);
        m.token_count = tokens;
        m
    }

    #[test]
    fn test_optimize_clean_short_conversation() {
        let dir = tempfile::tempdir().unwrap();
        let mut mgr = SessionManager::new(test_config(dir.path()), tc()).unwrap();
        let mut session = Session::new(tc());
        session.push(SessionMessage::system("You are helpful"));
        session.push(SessionMessage::text(SessionRole::User, "hello"));
        session.push(SessionMessage::text(SessionRole::Assistant, "hi"));

        let report = mgr.optimize(&mut session).unwrap();
        assert_eq!(report.tokens_saved, 0);
        assert!(report.compaction.is_none());
        assert!(matches!(report.trim, TrimAction::None));
    }

    #[test]
    fn test_optimize_compacts_large_tool_results() {
        let dir = tempfile::tempdir().unwrap();
        let mut mgr = SessionManager::new(test_config(dir.path()), tc()).unwrap();
        let mut session = Session::new(tc());
        session.push(SessionMessage::system("sys"));
        session.push(SessionMessage::text(SessionRole::User, "find bugs"));
        session.push(SessionMessage::assistant_tool_calls(
            "",
            vec![ToolCall {
                call_id: "c1".into(),
                tool_name: "shell".into(),
                arguments: serde_json::json!({}),
            }],
        ));
        // Large tool result
        let big: String = (0..500)
            .map(|i| format!("line {i} of output with some content"))
            .collect::<Vec<_>>()
            .join("\n");
        session.push(SessionMessage::tool_result(ToolResult {
            call_id: "c1".into(),
            tool_name: "shell".into(),
            output: big,
            is_error: false,
        }));

        let report = mgr.optimize(&mut session).unwrap();
        assert!(report.compaction.is_some());
        assert!(report.tokens_saved > 0);
    }

    #[test]
    fn test_optimize_over_sliding_window_threshold() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = test_config(dir.path());
        config.session.model = crate::config::ModelProfileConfig::Custom {
            context_window: 4096,
            max_output_tokens: 512,
            effective_fraction: 0.60,
        };
        let mut mgr = SessionManager::new(config, tc()).unwrap();
        // Budget: 4096*0.60 - 512 = 1945; sliding at 50% = 972
        let mut session = Session::new(tc());
        session.push(SessionMessage::system("sys"));
        for i in 0..10 {
            session.messages_mut().push(msg(SessionRole::User, format!("question {i}"), 80));
            session.messages_mut().push(msg(SessionRole::Assistant, format!("answer {i}"), 80));
        }
        session.recalculate_total();
        // Total: ~1600+ tokens, over 972 threshold

        let report = mgr.optimize(&mut session).unwrap();
        match &report.trim {
            TrimAction::SlidingWindow { turns_dropped } => {
                assert!(*turns_dropped > 0);
            }
            TrimAction::HardReset => {} // Also acceptable if it went straight to hard reset
            _ => panic!("expected trimming, got {:?}", report.trim),
        }
    }

    #[test]
    fn test_optimize_over_hard_reset_threshold() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = test_config(dir.path());
        config.session.model = crate::config::ModelProfileConfig::Custom {
            context_window: 4096,
            max_output_tokens: 512,
            effective_fraction: 0.60,
        };
        let mut mgr = SessionManager::new(config, tc()).unwrap();
        // Budget: 1945; hard reset at 60% = 1167
        let mut session = Session::new(tc());
        session.push(SessionMessage::system("sys"));
        for i in 0..20 {
            session.messages_mut().push(msg(SessionRole::User, format!("q{i}"), 80));
            session.messages_mut().push(msg(SessionRole::Assistant, format!("a{i}"), 80));
        }
        session.recalculate_total();
        // Total: ~3200+ tokens, well over hard reset threshold

        let report = mgr.optimize(&mut session).unwrap();
        assert!(matches!(report.trim, TrimAction::HardReset | TrimAction::SlidingWindow { .. }));
        assert!(report.output_tokens < report.input_tokens);
    }

    #[test]
    fn test_optimize_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let mut mgr = SessionManager::new(test_config(dir.path()), tc()).unwrap();
        let mut session = Session::new(tc());
        session.push(SessionMessage::system("sys"));
        session.push(SessionMessage::text(SessionRole::User, "hello"));
        session.push(SessionMessage::text(SessionRole::Assistant, "hi"));

        let _report1 = mgr.optimize(&mut session).unwrap();
        let tokens_after_first = session.total_tokens();
        let report2 = mgr.optimize(&mut session).unwrap();
        let tokens_after_second = session.total_tokens();

        // Should be essentially unchanged
        assert_eq!(tokens_after_first, tokens_after_second);
        assert_eq!(report2.tokens_saved, 0);
    }

    #[test]
    fn test_process_message_tool_result() {
        let dir = tempfile::tempdir().unwrap();
        let mut mgr = SessionManager::new(test_config(dir.path()), tc()).unwrap();
        let mut session = Session::new(tc());
        session.push(SessionMessage::system("sys"));
        session.push(SessionMessage::text(SessionRole::User, "find bugs"));
        session.push(SessionMessage::assistant_tool_calls(
            "",
            vec![ToolCall {
                call_id: "c1".into(),
                tool_name: "shell".into(),
                arguments: serde_json::json!({}),
            }],
        ));

        let big: String = (0..500)
            .map(|i| format!("line {i} content here"))
            .collect::<Vec<_>>()
            .join("\n");
        let tool_msg = SessionMessage::tool_result(ToolResult {
            call_id: "c1".into(),
            tool_name: "shell".into(),
            output: big,
            is_error: false,
        });

        let report = mgr.process_message(&mut session, tool_msg).unwrap();
        assert!(report.compaction.is_some());
    }

    #[test]
    fn test_process_message_user_extracts_goal() {
        let dir = tempfile::tempdir().unwrap();
        let mut mgr = SessionManager::new(test_config(dir.path()), tc()).unwrap();
        let mut session = Session::new(tc());
        session.push(SessionMessage::system("sys"));

        let user_msg = SessionMessage::text(SessionRole::User, "Find all security vulnerabilities");
        mgr.process_message(&mut session, user_msg).unwrap();
        assert!(mgr.goal().is_some());
        assert!(mgr.goal().unwrap().contains("security vulnerabilities"));
    }

    #[test]
    fn test_process_message_threshold_crossed() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = test_config(dir.path());
        config.session.model = crate::config::ModelProfileConfig::Custom {
            context_window: 4096,
            max_output_tokens: 512,
            effective_fraction: 0.60,
        };
        // Budget: 4096*0.60 - 512 = 1945; sliding at 50% = 972
        let mut mgr = SessionManager::new(config, tc()).unwrap();
        let mut session = Session::new(tc());
        session.push(SessionMessage::system("sys"));
        // Use messages_mut to preserve pre-set token counts
        for i in 0..15 {
            session.messages_mut().push(msg(SessionRole::User, format!("q{i}"), 80));
            session.messages_mut().push(msg(SessionRole::Assistant, format!("a{i}"), 80));
        }
        session.recalculate_total();
        // Total: ~5 + 30*80 = 2405, over 972 threshold

        // process_message calls push() which recomputes tokens, but optimize sees the total
        let new_msg = msg(SessionRole::User, "one more question with many words to test", 80);
        let report = mgr.process_message(&mut session, new_msg).unwrap();
        assert!(!matches!(report.trim, TrimAction::None));
    }

    #[test]
    fn test_force_reset() {
        let dir = tempfile::tempdir().unwrap();
        let mut mgr = SessionManager::new(test_config(dir.path()), tc()).unwrap();
        let mut session = Session::new(tc());
        session.push(SessionMessage::system("sys"));
        for i in 0..10 {
            session.push(SessionMessage::text(
                SessionRole::User,
                format!("question {i}"),
            ));
            session.push(SessionMessage::text(
                SessionRole::Assistant,
                format!("answer {i}"),
            ));
        }
        let before = session.total_tokens();

        let result = mgr.force_reset(&mut session).unwrap();
        assert!(result.tokens_after < result.tokens_before);
        assert!(session.total_tokens() < before);
    }

    #[test]
    fn test_budget_status() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = SessionManager::new(test_config(dir.path()), tc()).unwrap();
        let mut session = Session::new(tc());
        session.push(SessionMessage::system("sys"));
        session.push(SessionMessage::text(SessionRole::User, "hello"));

        let status = mgr.budget_status(&session);
        assert!(status.total_tokens > 0);
        assert!(status.utilization_percent < 1.0);
        assert_eq!(status.action_needed, BudgetAction::None);
        assert_eq!(status.turns, 1); // 1 user turn (excluding system)
    }

    #[test]
    fn test_optimize_messages_non_mutating() {
        let dir = tempfile::tempdir().unwrap();
        let mut mgr = SessionManager::new(test_config(dir.path()), tc()).unwrap();
        let messages = vec![
            SessionMessage::system("sys"),
            SessionMessage::text(SessionRole::User, "hello"),
            SessionMessage::text(SessionRole::Assistant, "hi"),
        ];

        let (output, report) = mgr.optimize_messages(messages).unwrap();
        assert!(!output.is_empty());
        assert_eq!(output[0].role, SessionRole::System);
        assert_eq!(report.tokens_saved, 0);
    }

    // === Integration tests ===

    #[test]
    fn test_integration_50_message_optimize() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = test_config(dir.path());
        config.session.model = crate::config::ModelProfileConfig::Custom {
            context_window: 8192,
            max_output_tokens: 2048,
            effective_fraction: 0.60,
        };
        let mut mgr = SessionManager::new(config, tc()).unwrap();
        let mut messages = vec![SessionMessage::system("You are a code reviewer")];
        for i in 0..20 {
            let mut u = SessionMessage::text(SessionRole::User, format!("review function {i}"));
            u.token_count = 30;
            messages.push(u);

            let mut tc_msg = SessionMessage::assistant_tool_calls(
                "",
                vec![ToolCall {
                    call_id: format!("c{i}"),
                    tool_name: "file.grep".into(),
                    arguments: serde_json::json!({"pattern": format!("fn_{i}")}),
                }],
            );
            tc_msg.token_count = 20;
            messages.push(tc_msg);

            let big: String = (0..100)
                .map(|j| format!("line {j} of function {i} with details"))
                .collect::<Vec<_>>()
                .join("\n");
            let mut tr_msg = SessionMessage::tool_result(ToolResult {
                call_id: format!("c{i}"),
                tool_name: "file.grep".into(),
                output: big,
                is_error: false,
            });
            tr_msg.token_count = 500;
            messages.push(tr_msg);

            let mut a = SessionMessage::text(
                SessionRole::Assistant,
                format!("Found issue {i} in the code"),
            );
            a.token_count = 30;
            messages.push(a);
        }

        let (optimized, report) = mgr.optimize_messages(messages).unwrap();
        // Should have compacted large tool results
        assert!(report.compaction.is_some() || report.tokens_saved > 0);
        // Should have some memory extracted
        assert!(!optimized.is_empty());
        assert_eq!(optimized[0].role, SessionRole::System);
    }

    #[test]
    fn test_integration_roundtrip_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let mut mgr = SessionManager::new(test_config(dir.path()), tc()).unwrap();
        let messages = vec![
            SessionMessage::system("sys"),
            SessionMessage::text(SessionRole::User, "hello"),
            SessionMessage::text(SessionRole::Assistant, "hi"),
            SessionMessage::text(SessionRole::User, "how are you"),
            SessionMessage::text(SessionRole::Assistant, "fine"),
        ];

        let (first, _) = mgr.optimize_messages(messages).unwrap();
        let first_tokens: u32 = first.iter().map(|m| m.token_count).sum();

        let (second, report2) = mgr.optimize_messages(first).unwrap();
        let second_tokens: u32 = second.iter().map(|m| m.token_count).sum();

        // Second pass should not change anything meaningful
        assert_eq!(first_tokens, second_tokens);
        assert_eq!(report2.tokens_saved, 0);
    }

    #[test]
    fn test_integration_library_api_one_shot() {
        let input = vec![
            serde_json::json!({"role": "system", "content": "You are helpful"}),
            serde_json::json!({"role": "user", "content": "hello"}),
            serde_json::json!({"role": "assistant", "content": "hi"}),
        ];
        let config = SessionManagerConfig::default();
        let (output, report) = crate::optimize(&input, config).unwrap();
        assert_eq!(output.len(), 3);
        assert_eq!(output[0]["role"], "system");
        assert_eq!(report.tokens_saved, 0);
    }

    // === Edge case tests (audit) ===

    #[test]
    fn test_nudge_includes_goal_without_trimming() {
        // Bug fix: nudge must include goal even when no trim/reset has occurred.
        // The goal is in self.goal, NOT in memory_store until trim happens.
        let dir = tempfile::tempdir().unwrap();
        let mut config = test_config(dir.path());
        config.reinforcement.enabled = true;
        config.reinforcement.nudge_every_n_tool_results = 1;
        config.reinforcement.include_goal = true;
        let mut mgr = SessionManager::new(config, tc()).unwrap();
        let mut session = Session::new(tc());
        session.push(SessionMessage::system("sys"));
        session.push(SessionMessage::text(SessionRole::User, "Find all security bugs"));
        session.push(SessionMessage::assistant_tool_calls(
            "",
            vec![ToolCall {
                call_id: "c1".into(),
                tool_name: "grep".into(),
                arguments: serde_json::json!({}),
            }],
        ));
        session.push(SessionMessage::tool_result(ToolResult {
            call_id: "c1".into(),
            tool_name: "grep".into(),
            output: "found something".into(),
            is_error: false,
        }));

        let report = mgr.optimize(&mut session).unwrap();
        assert!(report.nudge_injected);
        // The nudge should include the goal text
        let last = session.messages().last().unwrap();
        assert!(last.flags.contains(crate::message::MessageFlags::IS_NUDGE));
        assert!(
            last.content.contains("security bugs"),
            "nudge should contain goal text but got: {}",
            last.content
        );
    }

    #[test]
    fn test_optimize_empty_conversation() {
        let dir = tempfile::tempdir().unwrap();
        let mut mgr = SessionManager::new(test_config(dir.path()), tc()).unwrap();
        let (output, report) = mgr.optimize_messages(vec![]).unwrap();
        // Preflight adds a default system prompt to empty sessions
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].role, SessionRole::System);
        assert_eq!(report.input_tokens, 0);
        assert!(report.preflight_issues.iter().any(|i|
            i.kind == crate::preflight::PreflightIssueKind::MissingSystemPrompt
        ));
    }

    #[test]
    fn test_process_message_extracts_goal_via_optimize() {
        // Goal extraction in process_message happens through optimize(),
        // not duplicated in process_message itself
        let dir = tempfile::tempdir().unwrap();
        let mut mgr = SessionManager::new(test_config(dir.path()), tc()).unwrap();
        let mut session = Session::new(tc());
        session.push(SessionMessage::system("sys"));
        let user = SessionMessage::text(SessionRole::User, "Analyze memory leaks");
        mgr.process_message(&mut session, user).unwrap();
        assert!(mgr.goal().is_some());
        assert!(mgr.goal().unwrap().contains("memory leaks"));
    }

    #[test]
    fn test_nudge_not_injected_when_disabled() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = test_config(dir.path());
        config.reinforcement.enabled = false;
        let mut mgr = SessionManager::new(config, tc()).unwrap();
        let mut session = Session::new(tc());
        session.push(SessionMessage::system("sys"));
        session.push(SessionMessage::text(SessionRole::User, "go"));
        session.push(SessionMessage::assistant_tool_calls(
            "",
            vec![ToolCall {
                call_id: "c1".into(),
                tool_name: "test".into(),
                arguments: serde_json::json!({}),
            }],
        ));
        session.push(SessionMessage::tool_result(ToolResult {
            call_id: "c1".into(),
            tool_name: "test".into(),
            output: "ok".into(),
            is_error: false,
        }));

        let report = mgr.optimize(&mut session).unwrap();
        assert!(!report.nudge_injected);
    }

    #[test]
    fn test_apply_trim_compaction_falls_back_to_hard_reset() {
        // When Compaction threshold is crossed but no LLM is set,
        // apply_trim should do a hard reset (not panic)
        let dir = tempfile::tempdir().unwrap();
        let mut config = test_config(dir.path());
        config.session.model = crate::config::ModelProfileConfig::Custom {
            context_window: 4096,
            max_output_tokens: 512,
            effective_fraction: 0.60,
        };
        let mut mgr = SessionManager::new(config, tc()).unwrap();
        // Budget: 1945; compaction at 85% = 1653
        let mut session = Session::new(tc());
        session.push(SessionMessage::system("sys"));
        for i in 0..30 {
            session.messages_mut().push(msg(SessionRole::User, format!("q{i}"), 80));
            session.messages_mut().push(msg(SessionRole::Assistant, format!("a{i}"), 80));
        }
        session.recalculate_total();
        // Total: ~4805 tokens, well over compaction threshold

        let report = mgr.optimize(&mut session).unwrap();
        assert!(matches!(report.trim, TrimAction::HardReset));
        assert!(report.output_tokens < report.input_tokens);
    }

    #[test]
    fn test_session_manager_consolidation_in_optimize() {
        // Populate memory store with enough similar findings, then verify
        // that optimize() consolidates them.
        let dir = tempfile::tempdir().unwrap();
        let mut mgr = SessionManager::new(test_config(dir.path()), tc()).unwrap();

        // Manually inject facts into memory store to simulate accumulated findings
        use crate::memory::types::{MemoryFact, MemorySource, now_millis};
        for i in 0..4 {
            mgr.memory_store.upsert(MemoryFact {
                key: format!("finding:grep_{i}"),
                value: format!(
                    "{} matches in {} files. src/mod_{i}.rs:10: fn_{i}() call with arguments and extra context padding here",
                    i + 1, i + 1
                ),
                source: MemorySource::ToolResult {
                    tool_name: "file.grep".into(),
                    call_id: format!("c{i}"),
                },
                created_at: now_millis(),
                priority: crate::memory::FactPriority::Finding,
            });
        }

        let mut session = Session::new(tc());
        session.push(SessionMessage::system("sys"));
        session.push(SessionMessage::text(SessionRole::User, "hello"));
        session.push(SessionMessage::text(SessionRole::Assistant, "hi"));

        let report = mgr.optimize(&mut session).unwrap();
        // Should have consolidated the 4 grep findings
        assert!(report.consolidation.is_some(), "expected consolidation to fire");
        let cr = report.consolidation.unwrap();
        assert_eq!(cr.facts_consolidated, 1);
        assert_eq!(cr.source_facts_replaced, 4);
        assert!(cr.bytes_saved > 0);
        // The store should now have the consolidated fact
        assert!(mgr.memory().get("consolidated:finding:grep_summary").is_some());
    }

    #[test]
    fn test_session_manager_consolidation_disabled() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = test_config(dir.path());
        config.consolidation.enabled = false;
        let mut mgr = SessionManager::new(config, tc()).unwrap();

        // Add facts that would normally be consolidated
        use crate::memory::types::{MemoryFact, MemorySource, now_millis};
        for i in 0..4 {
            mgr.memory_store.upsert(MemoryFact {
                key: format!("finding:grep_{i}"),
                value: format!(
                    "{} matches in {} files. src/mod_{i}.rs:10: fn_{i}() with padding context",
                    i + 1, i + 1
                ),
                source: MemorySource::ToolResult {
                    tool_name: "file.grep".into(),
                    call_id: format!("c{i}"),
                },
                created_at: now_millis(),
                priority: crate::memory::FactPriority::Finding,
            });
        }

        let mut session = Session::new(tc());
        session.push(SessionMessage::system("sys"));
        session.push(SessionMessage::text(SessionRole::User, "hello"));

        let report = mgr.optimize(&mut session).unwrap();
        // Consolidation disabled — should not fire
        assert!(report.consolidation.is_none());
        // All 4 facts should still be present
        assert_eq!(mgr.memory().len(), 4);
        assert!(mgr.memory().get("finding:grep_0").is_some());
    }
}
