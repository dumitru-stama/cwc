use async_trait::async_trait;

use crate::error::Result;
use crate::types::{ChatMessage, Chunk, RetrievalHit, Verdict};

/// Counts tokens in text and truncates to a budget.
pub trait TokenCounter: Send + Sync {
    fn count_tokens(&self, text: &str) -> u32;
    fn truncate_to_tokens(&self, text: &str, max_tokens: u32) -> String;
}

/// Retrieves candidate chunks for a query.
pub trait Retriever: Send + Sync {
    fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<RetrievalHit>>;
}

/// Reranks retrieval hits using a more expensive model.
pub trait Reranker: Send + Sync {
    fn rerank(&self, query: &str, hits: &mut Vec<RetrievalHit>, top_k: usize) -> Result<()>;
}

/// Generates dense embeddings for text.
pub trait Embedder: Send + Sync {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn dim(&self) -> usize;
}

/// Verifies LLM output against source chunks.
pub trait Verifier: Send + Sync {
    fn verify(&self, output: &str, sources: &[Chunk]) -> Result<Verdict>;
}

/// Callback invoked for each streaming token chunk.
pub type StreamCallback = Box<dyn FnMut(&str) + Send>;

/// Calls a local LLM with optional grammar constraints.
#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn generate(&self, prompt: &str, grammar: Option<&str>, max_tokens: u32) -> Result<String>;
    async fn generate_chat(
        &self,
        messages: &[ChatMessage],
        grammar: Option<&str>,
        max_tokens: u32,
    ) -> Result<String>;

    /// Generate a chat completion with streaming. Calls `on_token` for each chunk.
    /// Returns the full accumulated response. Default implementation falls back to non-streaming.
    async fn generate_chat_stream(
        &self,
        messages: &[ChatMessage],
        grammar: Option<&str>,
        max_tokens: u32,
        on_token: StreamCallback,
    ) -> Result<String> {
        let _ = on_token;
        self.generate_chat(messages, grammar, max_tokens).await
    }
}
