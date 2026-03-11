use std::sync::Arc;

use cwc_cli::compiler::ContextWindowCompiler;
use cwc_core::traits::TokenCounter;

/// Shared application state for all Axum handlers.
pub struct AppState {
    /// The full pipeline compiler (optional — only available if indexes + LLM are configured).
    pub compiler: Option<Arc<ContextWindowCompiler>>,
    /// Tokenizer for session operations (always available).
    pub tokenizer: Arc<dyn TokenCounter>,
}
