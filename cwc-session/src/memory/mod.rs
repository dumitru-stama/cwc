pub mod consolidate;
pub mod extract;
pub mod goal;
pub mod llm_consolidate;
pub mod store;
pub mod types;

pub use consolidate::{consolidate_memory, ConsolidationConfig, ConsolidationReport};
pub use llm_consolidate::{llm_consolidate, LlmConsolidationConfig, LlmConsolidationReport};
pub use extract::{
    extract_conclusion, extract_from_message, extract_from_tool_result, extract_from_turns,
    extract_goal,
};
pub use goal::{create_memory_message, extract_goal_text, format_goal_nudge};
pub use store::SessionMemoryStore;
pub use types::{make_slug, FactPriority, MemoryFact, MemorySource};
