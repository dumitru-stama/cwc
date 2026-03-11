use std::sync::{Arc, Mutex};

use std::path::PathBuf;

use cwc_compile::scaffold::{PromptScaffold, ScaffoldConfig};
use cwc_core::cache::{PromptCache, RetrievalCache};
use cwc_core::config::{CompilerMode, CwcConfig};
use cwc_core::error::{CwcError, Result};
use cwc_core::noop::NoopReranker;
use cwc_core::traits::{Embedder, LlmClient, Reranker, Retriever, TokenCounter, Verifier};
use cwc_memory::conversation::ConversationMemory;
use cwc_memory::store::MemoryStore;
use cwc_retrieve::cross_encoder::{CrossEncoderReranker, RerankConfig};
use cwc_verify::complex::{ComplexVerifier, ComplexVerifyConfig};
use cwc_verify::heuristic::{HeuristicVerifier, HeuristicVerifierConfig};

use crate::compiler::ContextWindowCompiler;

/// Builder for constructing a ContextWindowCompiler with flexible component injection.
pub struct CompilerBuilder {
    config: CwcConfig,
    retriever: Option<Arc<dyn Retriever>>,
    reranker: Option<Arc<dyn Reranker>>,
    embedder: Option<Arc<dyn Embedder>>,
    llm: Option<Arc<dyn LlmClient>>,
    verifier: Option<Arc<dyn Verifier>>,
    tokenizer: Option<Arc<dyn TokenCounter>>,
    long_term: Option<Arc<dyn MemoryStore>>,
    max_conversation_turns: usize,
}

impl CompilerBuilder {
    pub fn new(config: CwcConfig) -> Self {
        Self {
            config,
            retriever: None,
            reranker: None,
            embedder: None,
            llm: None,
            verifier: None,
            tokenizer: None,
            long_term: None,
            max_conversation_turns: 10,
        }
    }

    pub fn with_retriever(mut self, r: Arc<dyn Retriever>) -> Self {
        self.retriever = Some(r);
        self
    }

    pub fn with_reranker(mut self, r: Arc<dyn Reranker>) -> Self {
        self.reranker = Some(r);
        self
    }

    pub fn with_embedder(mut self, e: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(e);
        self
    }

    pub fn with_llm(mut self, l: Arc<dyn LlmClient>) -> Self {
        self.llm = Some(l);
        self
    }

    pub fn with_verifier(mut self, v: Arc<dyn Verifier>) -> Self {
        self.verifier = Some(v);
        self
    }

    pub fn with_tokenizer(mut self, t: Arc<dyn TokenCounter>) -> Self {
        self.tokenizer = Some(t);
        self
    }

    pub fn with_memory_store(mut self, m: Arc<dyn MemoryStore>) -> Self {
        self.long_term = Some(m);
        self
    }

    pub fn with_max_conversation_turns(mut self, n: usize) -> Self {
        self.max_conversation_turns = n;
        self
    }

    /// Build the compiler. Requires at minimum: retriever, embedder, llm, tokenizer.
    /// Reranker and verifier default based on config.mode.
    pub fn build(self) -> Result<ContextWindowCompiler> {
        let retriever = self.retriever.ok_or_else(|| {
            CwcError::Config("CompilerBuilder: retriever is required".into())
        })?;

        let embedder = self.embedder.ok_or_else(|| {
            CwcError::Config("CompilerBuilder: embedder is required".into())
        })?;

        let llm = self.llm.ok_or_else(|| {
            CwcError::Config("CompilerBuilder: llm client is required".into())
        })?;

        let tokenizer = self.tokenizer.ok_or_else(|| {
            CwcError::Config("CompilerBuilder: tokenizer is required".into())
        })?;

        // Default reranker based on mode
        let reranker = self.reranker.unwrap_or_else(|| match self.config.mode {
            CompilerMode::Simple => Arc::new(NoopReranker),
            CompilerMode::Complex => {
                // Try to load cross-encoder model for complex mode
                if let Some(ref model_dir) = self.config.model.rerank_model {
                    let model_dir = PathBuf::from(model_dir);
                    let model_path = model_dir.join("model.onnx");
                    let tokenizer_path = model_dir.join("tokenizer.json");
                    match CrossEncoderReranker::load(
                        &model_path,
                        &tokenizer_path,
                        RerankConfig::default(),
                    ) {
                        Ok(ce) => {
                            tracing::info!("loaded cross-encoder reranker from {}", model_dir.display());
                            Arc::new(ce) as Arc<dyn Reranker>
                        }
                        Err(e) => {
                            tracing::warn!(
                                "failed to load cross-encoder from {}: {e}. Falling back to NoopReranker",
                                model_dir.display()
                            );
                            Arc::new(NoopReranker)
                        }
                    }
                } else {
                    tracing::warn!("complex mode but no rerank_model configured. Using NoopReranker");
                    Arc::new(NoopReranker)
                }
            }
        });

        // Default verifier based on mode
        let verifier = self.verifier.unwrap_or_else(|| match self.config.mode {
            CompilerMode::Simple => Arc::new(HeuristicVerifier::new(
                HeuristicVerifierConfig::default(),
            )),
            CompilerMode::Complex => {
                // Complex mode uses ComplexVerifier (heuristic + CoVe + RARR)
                Arc::new(ComplexVerifier::new(
                    Arc::clone(&llm),
                    Arc::clone(&retriever),
                    ComplexVerifyConfig::default(),
                )) as Arc<dyn Verifier>
            }
        });

        let scaffold = PromptScaffold::new(ScaffoldConfig::default());
        let conversation =
            Mutex::new(ConversationMemory::new(self.max_conversation_turns, Arc::clone(&tokenizer)));

        Ok(ContextWindowCompiler {
            retriever,
            reranker,
            embedder,
            llm,
            verifier,
            tokenizer,
            scaffold,
            conversation,
            long_term: self.long_term,
            retrieval_cache: RetrievalCache::new(),
            prompt_cache: PromptCache::new(),
            config: self.config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use cwc_core::noop::NoopVerifier;
    use cwc_core::types::{ChatMessage, RetrievalHit};

    struct FakeTokenCounter;
    impl TokenCounter for FakeTokenCounter {
        fn count_tokens(&self, text: &str) -> u32 {
            text.split_whitespace().count() as u32
        }
        fn truncate_to_tokens(&self, text: &str, max_tokens: u32) -> String {
            text.split_whitespace()
                .take(max_tokens as usize)
                .collect::<Vec<_>>()
                .join(" ")
        }
    }

    struct FakeRetriever;
    impl Retriever for FakeRetriever {
        fn retrieve(&self, _q: &str, _k: usize) -> cwc_core::error::Result<Vec<RetrievalHit>> {
            Ok(vec![])
        }
    }

    struct FakeEmbedder;
    impl Embedder for FakeEmbedder {
        fn embed(&self, _t: &[&str]) -> cwc_core::error::Result<Vec<Vec<f32>>> {
            Ok(vec![vec![0.0; 384]])
        }
        fn dim(&self) -> usize {
            384
        }
    }

    struct FakeLlm;
    #[async_trait]
    impl LlmClient for FakeLlm {
        async fn generate(
            &self,
            _p: &str,
            _g: Option<&str>,
            _m: u32,
        ) -> cwc_core::error::Result<String> {
            Ok("test response".into())
        }
        async fn generate_chat(
            &self,
            _msgs: &[ChatMessage],
            _g: Option<&str>,
            _m: u32,
        ) -> cwc_core::error::Result<String> {
            Ok("test response".into())
        }
    }

    fn test_config() -> CwcConfig {
        CwcConfig::default()
    }

    #[test]
    fn test_builder_missing_retriever() {
        let result = CompilerBuilder::new(test_config())
            .with_embedder(Arc::new(FakeEmbedder))
            .with_llm(Arc::new(FakeLlm))
            .with_tokenizer(Arc::new(FakeTokenCounter))
            .build();
        assert!(result.is_err());
        let err = result.err().unwrap().to_string();
        assert!(err.contains("retriever"), "error should mention retriever: {err}");
    }

    #[test]
    fn test_builder_missing_embedder() {
        let result = CompilerBuilder::new(test_config())
            .with_retriever(Arc::new(FakeRetriever))
            .with_llm(Arc::new(FakeLlm))
            .with_tokenizer(Arc::new(FakeTokenCounter))
            .build();
        assert!(result.is_err());
        let err = result.err().unwrap().to_string();
        assert!(err.contains("embedder"), "error should mention embedder: {err}");
    }

    #[test]
    fn test_builder_missing_llm() {
        let result = CompilerBuilder::new(test_config())
            .with_retriever(Arc::new(FakeRetriever))
            .with_embedder(Arc::new(FakeEmbedder))
            .with_tokenizer(Arc::new(FakeTokenCounter))
            .build();
        assert!(result.is_err());
        let err = result.err().unwrap().to_string();
        assert!(err.contains("llm"), "error should mention llm: {err}");
    }

    #[test]
    fn test_builder_missing_tokenizer() {
        let result = CompilerBuilder::new(test_config())
            .with_retriever(Arc::new(FakeRetriever))
            .with_embedder(Arc::new(FakeEmbedder))
            .with_llm(Arc::new(FakeLlm))
            .build();
        assert!(result.is_err());
        let err = result.err().unwrap().to_string();
        assert!(err.contains("tokenizer"), "error should mention tokenizer: {err}");
    }

    #[test]
    fn test_builder_simple_mode_defaults() {
        let compiler = CompilerBuilder::new(test_config())
            .with_retriever(Arc::new(FakeRetriever))
            .with_embedder(Arc::new(FakeEmbedder))
            .with_llm(Arc::new(FakeLlm))
            .with_tokenizer(Arc::new(FakeTokenCounter))
            .build()
            .unwrap();

        // Should have built successfully with default reranker and verifier
        assert_eq!(compiler.config.mode, CompilerMode::Simple);
    }

    #[test]
    fn test_builder_with_all_components() {
        let compiler = CompilerBuilder::new(test_config())
            .with_retriever(Arc::new(FakeRetriever))
            .with_reranker(Arc::new(NoopReranker))
            .with_embedder(Arc::new(FakeEmbedder))
            .with_llm(Arc::new(FakeLlm))
            .with_verifier(Arc::new(NoopVerifier))
            .with_tokenizer(Arc::new(FakeTokenCounter))
            .with_max_conversation_turns(20)
            .build()
            .unwrap();

        assert!(compiler.long_term.is_none());
    }

    #[test]
    fn test_builder_complex_mode() {
        let mut config = test_config();
        config.mode = CompilerMode::Complex;

        let compiler = CompilerBuilder::new(config)
            .with_retriever(Arc::new(FakeRetriever))
            .with_embedder(Arc::new(FakeEmbedder))
            .with_llm(Arc::new(FakeLlm))
            .with_tokenizer(Arc::new(FakeTokenCounter))
            .build()
            .unwrap();

        assert_eq!(compiler.config.mode, CompilerMode::Complex);
    }

    #[test]
    fn test_builder_complex_mode_with_rerank_model_missing_file() {
        // Complex mode with a rerank_model path that doesn't exist should fall back to NoopReranker
        let mut config = test_config();
        config.mode = CompilerMode::Complex;
        config.model.rerank_model = Some("/nonexistent/model_dir".into());

        let compiler = CompilerBuilder::new(config)
            .with_retriever(Arc::new(FakeRetriever))
            .with_embedder(Arc::new(FakeEmbedder))
            .with_llm(Arc::new(FakeLlm))
            .with_tokenizer(Arc::new(FakeTokenCounter))
            .build()
            .unwrap();

        // Should build successfully (falls back to NoopReranker)
        assert_eq!(compiler.config.mode, CompilerMode::Complex);
    }

    #[test]
    fn test_builder_simple_mode_uses_noop_reranker() {
        // Simple mode should never attempt to load cross-encoder
        let compiler = CompilerBuilder::new(test_config())
            .with_retriever(Arc::new(FakeRetriever))
            .with_embedder(Arc::new(FakeEmbedder))
            .with_llm(Arc::new(FakeLlm))
            .with_tokenizer(Arc::new(FakeTokenCounter))
            .build()
            .unwrap();

        assert_eq!(compiler.config.mode, CompilerMode::Simple);
        // Verify noop behavior: reranking should not change order
        let mut hits = vec![
            cwc_core::types::RetrievalHit {
                chunk: cwc_core::types::Chunk {
                    chunk_id: uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, b"a"),
                    doc_id: uuid::Uuid::nil(),
                    doc_version: 1,
                    source_path: "a.md".into(),
                    section_path: vec![],
                    char_offset: 0,
                    char_len: 5,
                    token_count: 5,
                    text: "doc a".into(),
                    metadata: std::collections::HashMap::new(),
                },
                score_sparse: 0.0,
                score_dense: 0.0,
                score_fused: 0.5,
                score_rerank: 0.0,
            },
        ];
        // NoopReranker just truncates
        cwc_core::traits::Reranker::rerank(
            &NoopReranker,
            "test",
            &mut hits,
            10,
        )
        .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].score_rerank, 0.0);
    }

    #[tokio::test]
    async fn test_compiler_query_with_fakes() {
        let compiler = CompilerBuilder::new(test_config())
            .with_retriever(Arc::new(FakeRetriever))
            .with_embedder(Arc::new(FakeEmbedder))
            .with_llm(Arc::new(FakeLlm))
            .with_tokenizer(Arc::new(FakeTokenCounter))
            .with_verifier(Arc::new(NoopVerifier))
            .build()
            .unwrap();

        let output = compiler.query("What is Rust?", None).await.unwrap();
        assert!(output.verdict.is_pass());
        assert_eq!(output.response.raw_text, "test response");
        assert_eq!(output.attempts, 1);
    }

    #[tokio::test]
    async fn test_compiler_query_populates_timing() {
        let compiler = CompilerBuilder::new(test_config())
            .with_retriever(Arc::new(FakeRetriever))
            .with_embedder(Arc::new(FakeEmbedder))
            .with_llm(Arc::new(FakeLlm))
            .with_tokenizer(Arc::new(FakeTokenCounter))
            .with_verifier(Arc::new(NoopVerifier))
            .build()
            .unwrap();

        let output = compiler.query("test query", None).await.unwrap();
        // total >= sum of individual stages
        assert!(
            output.timing.total_ms
                >= output.timing.retrieval_ms
                    + output.timing.compilation_ms
                    + output.timing.generation_ms
        );
    }

    #[tokio::test]
    async fn test_compiler_conversation_history() {
        let compiler = CompilerBuilder::new(test_config())
            .with_retriever(Arc::new(FakeRetriever))
            .with_embedder(Arc::new(FakeEmbedder))
            .with_llm(Arc::new(FakeLlm))
            .with_tokenizer(Arc::new(FakeTokenCounter))
            .with_verifier(Arc::new(NoopVerifier))
            .build()
            .unwrap();

        assert_eq!(compiler.conversation_len().unwrap(), 0);

        compiler.query("first question", None).await.unwrap();
        assert_eq!(compiler.conversation_len().unwrap(), 2); // user + assistant

        compiler.query("second question", None).await.unwrap();
        assert_eq!(compiler.conversation_len().unwrap(), 4);

        compiler.clear_conversation().unwrap();
        assert_eq!(compiler.conversation_len().unwrap(), 0);
    }

    #[test]
    fn test_compiler_compile_no_llm_call() {
        let compiler = CompilerBuilder::new(test_config())
            .with_retriever(Arc::new(FakeRetriever))
            .with_embedder(Arc::new(FakeEmbedder))
            .with_llm(Arc::new(FakeLlm))
            .with_tokenizer(Arc::new(FakeTokenCounter))
            .with_verifier(Arc::new(NoopVerifier))
            .build()
            .unwrap();

        let (ctx, messages, report) = compiler.compile("What is Rust?", None).unwrap();
        // Compile returns context without LLM call
        assert!(ctx.sources.is_empty()); // FakeRetriever returns no hits
        assert!(!messages.is_empty()); // At minimum system + user messages
        assert_eq!(report.chunks_selected, 0);
    }

    #[test]
    fn test_compiler_ingest_test_corpus() {
        let dir = std::env::temp_dir().join("cwc_test_ingest_corpus");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut config = test_config();
        config.paths.data_dir = dir.join("data").to_string_lossy().into_owned();
        config.paths.index_dir = dir.join("index").to_string_lossy().into_owned();

        let compiler = CompilerBuilder::new(config)
            .with_retriever(Arc::new(FakeRetriever))
            .with_embedder(Arc::new(FakeEmbedder))
            .with_llm(Arc::new(FakeLlm))
            .with_tokenizer(Arc::new(FakeTokenCounter))
            .build()
            .unwrap();

        let corpus_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("tests/corpus");
        let report = compiler.ingest(&[corpus_dir], false).unwrap();

        assert_eq!(report.files_processed, 5); // 5 markdown files in corpus
        assert!(report.chunks_created > 0);
        assert!(report.chunks_path.exists());
        assert!(report.index_path.exists());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_config_validation_endpoint() {
        // Config with default endpoint should be loadable
        let config = CwcConfig::default();
        assert!(!config.model.llm_endpoint.is_empty());

        // Builder doesn't validate endpoint at build time,
        // but config is accessible for inspection
        let compiler = CompilerBuilder::new(config)
            .with_retriever(Arc::new(FakeRetriever))
            .with_embedder(Arc::new(FakeEmbedder))
            .with_llm(Arc::new(FakeLlm))
            .with_tokenizer(Arc::new(FakeTokenCounter))
            .build()
            .unwrap();

        assert_eq!(
            compiler.config().model.llm_endpoint,
            "http://localhost:8080"
        );
    }

    #[tokio::test]
    async fn test_compiler_query_with_memory_store() {
        let mem = Arc::new(cwc_memory::file_memory::FileMemory::new(
            &std::env::temp_dir().join("cwc_test_mem_store.json"),
        ));

        let compiler = CompilerBuilder::new(test_config())
            .with_retriever(Arc::new(FakeRetriever))
            .with_embedder(Arc::new(FakeEmbedder))
            .with_llm(Arc::new(FakeLlm))
            .with_tokenizer(Arc::new(FakeTokenCounter))
            .with_verifier(Arc::new(NoopVerifier))
            .with_memory_store(mem.clone())
            .build()
            .unwrap();

        // Query with memory store configured should work
        let output = compiler.query("I prefer Rust over C++", None).await.unwrap();
        assert!(output.verdict.is_pass());

        let _ = std::fs::remove_file(std::env::temp_dir().join("cwc_test_mem_store.json"));
    }

    #[test]
    fn test_compiler_ingest_single_file() {
        let dir = std::env::temp_dir().join("cwc_test_ingest_single");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut config = test_config();
        config.paths.data_dir = dir.join("data").to_string_lossy().into_owned();
        config.paths.index_dir = dir.join("index").to_string_lossy().into_owned();

        let compiler = CompilerBuilder::new(config)
            .with_retriever(Arc::new(FakeRetriever))
            .with_embedder(Arc::new(FakeEmbedder))
            .with_llm(Arc::new(FakeLlm))
            .with_tokenizer(Arc::new(FakeTokenCounter))
            .build()
            .unwrap();

        let single_file = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("tests/corpus/ownership.md");
        let report = compiler.ingest(&[single_file], false).unwrap();

        assert_eq!(report.files_processed, 1);
        assert!(report.chunks_created > 0);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_compiler_ingest_empty_paths() {
        let dir = std::env::temp_dir().join("cwc_test_ingest_empty");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut config = test_config();
        config.paths.data_dir = dir.join("data").to_string_lossy().into_owned();
        config.paths.index_dir = dir.join("index").to_string_lossy().into_owned();

        let compiler = CompilerBuilder::new(config)
            .with_retriever(Arc::new(FakeRetriever))
            .with_embedder(Arc::new(FakeEmbedder))
            .with_llm(Arc::new(FakeLlm))
            .with_tokenizer(Arc::new(FakeTokenCounter))
            .build()
            .unwrap();

        let report = compiler.ingest(&[], false).unwrap();
        assert_eq!(report.files_processed, 0);
        assert_eq!(report.chunks_created, 0);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn test_compiler_query_retrieval_skipped_for_procedural() {
        // "Write a function" is a procedural query — retrieval should be skipped
        let compiler = CompilerBuilder::new(test_config())
            .with_retriever(Arc::new(FakeRetriever))
            .with_embedder(Arc::new(FakeEmbedder))
            .with_llm(Arc::new(FakeLlm))
            .with_tokenizer(Arc::new(FakeTokenCounter))
            .with_verifier(Arc::new(NoopVerifier))
            .build()
            .unwrap();

        let output = compiler
            .query("Write a function that sorts a list", None)
            .await
            .unwrap();
        // Should succeed even with no retrieval
        assert!(output.verdict.is_pass());
        assert!(output.retrieval_hits.is_empty());
    }

    #[test]
    fn test_builder_max_conversation_turns_default() {
        // Default is 10 turns
        let compiler = CompilerBuilder::new(test_config())
            .with_retriever(Arc::new(FakeRetriever))
            .with_embedder(Arc::new(FakeEmbedder))
            .with_llm(Arc::new(FakeLlm))
            .with_tokenizer(Arc::new(FakeTokenCounter))
            .build()
            .unwrap();

        // Can't directly inspect max_turns, but we can verify the compiler builds
        // with default and conversation starts empty
        assert_eq!(compiler.conversation_len().unwrap(), 0);
    }

    #[test]
    fn test_compile_returns_budget_report() {
        let compiler = CompilerBuilder::new(test_config())
            .with_retriever(Arc::new(FakeRetriever))
            .with_embedder(Arc::new(FakeEmbedder))
            .with_llm(Arc::new(FakeLlm))
            .with_tokenizer(Arc::new(FakeTokenCounter))
            .with_verifier(Arc::new(NoopVerifier))
            .build()
            .unwrap();

        let (_ctx, _msgs, report) = compiler.compile("test", None).unwrap();
        // Budget report should reflect the config
        assert_eq!(report.context_window, test_config().model.context_window);
        assert_eq!(report.chunks_selected, 0);
        assert_eq!(report.chunks_dropped, 0);
    }
}
