pub mod budget;
pub mod compaction;
pub mod config;
pub mod error;
pub mod eval;
pub mod format;
pub mod llm_compact;
pub mod manager;
pub mod memory;
pub mod message;
pub mod preflight;
pub mod rebuild;
pub mod reinforcement;
pub mod reset;
pub mod turn;
pub mod window;

pub use budget::{BudgetAction, ModelProfile, SessionBudget};
pub use compaction::{
    compact_session, ArtifactStore, CompactionEngine, CompactionReport, CompactionResult,
    CompactionRule, CompactionStrategy,
};
pub use config::{ModelProfileConfig, SessionConfig};
pub use error::{Result, SessionError};
pub use llm_compact::{compaction_schema, llm_compaction};
pub use manager::{BudgetStatus, OptimizationReport, SessionManager, SessionManagerConfig, TrimAction};
pub use memory::{
    consolidate_memory, create_memory_message, extract_from_turns, extract_goal,
    extract_goal_text, format_goal_nudge, llm_consolidate, make_slug, ConsolidationConfig,
    ConsolidationReport, FactPriority, LlmConsolidationConfig, LlmConsolidationReport,
    MemoryFact, MemorySource, SessionMemoryStore,
};
pub use message::{MessageFlags, Session, SessionMessage, SessionRole, ToolCall, ToolResult};
pub use preflight::{preflight_check, PreflightConfig, PreflightIssue, PreflightIssueKind};
pub use rebuild::rebuild_conversation;
pub use reinforcement::{create_nudge, ReinforcementConfig};
pub use reset::{hard_reset, ResetResult};
pub use turn::{is_nudge, parse_turns, Turn};
pub use window::{sliding_window_trim, TrimResult};

/// One-shot convenience function: optimize a conversation in OpenAI format.
///
/// Parses the messages, runs full optimization, returns the result in OpenAI format.
pub fn optimize(
    openai_messages: &[serde_json::Value],
    config: SessionManagerConfig,
) -> Result<(Vec<serde_json::Value>, OptimizationReport)> {
    use cwc_core::traits::TokenCounter;
    use std::sync::Arc;

    // Use a simple word-based tokenizer as default
    struct WordTokenizer;
    impl TokenCounter for WordTokenizer {
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

    let tokenizer: Arc<dyn TokenCounter> = Arc::new(WordTokenizer);
    let messages = format::openai::from_openai(openai_messages, &*tokenizer)?;
    let mut mgr = SessionManager::new(config, tokenizer)?;
    let (optimized, report) = mgr.optimize_messages(messages)?;
    let output = format::openai::to_openai(&optimized);
    Ok((output, report))
}
