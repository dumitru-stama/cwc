pub mod cache;
pub mod config;
pub mod error;
pub mod noop;
pub mod tokenizer;
pub mod traits;
pub mod types;
pub mod validate;

pub use config::CwcConfig;
pub use error::{CwcError, Result};
pub use tokenizer::Tokenizer;

#[cfg(test)]
mod tests {
    use super::types::*;
    use std::collections::HashMap;
    use uuid::Uuid;

    #[test]
    fn test_chunk_json_roundtrip() {
        let chunk = Chunk {
            chunk_id: Uuid::new_v4(),
            doc_id: Uuid::new_v4(),
            doc_version: 2,
            source_path: "docs/intro.md".to_string(),
            section_path: vec!["Chapter 2".to_string(), "Section 2.1".to_string()],
            char_offset: 100,
            char_len: 500,
            token_count: 120,
            text: "This is the chunk text content.".to_string(),
            metadata: HashMap::from([
                ("author".to_string(), "Alice".to_string()),
                ("lang".to_string(), "en".to_string()),
            ]),
        };

        let json = serde_json::to_string(&chunk).unwrap();
        let deserialized: Chunk = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.chunk_id, chunk.chunk_id);
        assert_eq!(deserialized.doc_id, chunk.doc_id);
        assert_eq!(deserialized.doc_version, chunk.doc_version);
        assert_eq!(deserialized.source_path, chunk.source_path);
        assert_eq!(deserialized.section_path, chunk.section_path);
        assert_eq!(deserialized.char_offset, chunk.char_offset);
        assert_eq!(deserialized.char_len, chunk.char_len);
        assert_eq!(deserialized.token_count, chunk.token_count);
        assert_eq!(deserialized.text, chunk.text);
        assert_eq!(deserialized.metadata, chunk.metadata);
    }

    #[test]
    fn test_token_budget_fractions_sum_correctly() {
        let budget = TokenBudget::from_fractions(4096, 0.15, 0.65, 0.10, 0.10);
        let allocated = budget.instruction + budget.sources + budget.memory + budget.output_reserved;
        assert_eq!(allocated + budget.remaining, budget.total);
        assert_eq!(budget.total, 4096);
        assert_eq!(budget.instruction, 614); // floor(4096 * 0.15)
        assert_eq!(budget.sources, 2662); // floor(4096 * 0.65)
        assert_eq!(budget.memory, 409); // floor(4096 * 0.10)
        assert_eq!(budget.output_reserved, 409); // floor(4096 * 0.10)
    }

    #[test]
    fn test_token_budget_remaining_computed() {
        let budget = TokenBudget::from_fractions(1000, 0.20, 0.50, 0.10, 0.10);
        // 200 + 500 + 100 + 100 = 900 allocated, 100 remaining
        assert_eq!(budget.remaining, 1000 - 200 - 500 - 100 - 100);
    }

    #[test]
    fn test_token_budget_zero_total() {
        let budget = TokenBudget::from_fractions(0, 0.15, 0.65, 0.10, 0.10);
        assert_eq!(budget.total, 0);
        assert_eq!(budget.instruction, 0);
        assert_eq!(budget.remaining, 0);
    }

    #[test]
    fn test_retrieval_hit_json_roundtrip() {
        let hit = RetrievalHit {
            chunk: Chunk {
                chunk_id: Uuid::new_v4(),
                doc_id: Uuid::new_v4(),
                doc_version: 1,
                source_path: "test.md".to_string(),
                section_path: vec![],
                char_offset: 0,
                char_len: 5,
                token_count: 2,
                text: "hello".to_string(),
                metadata: HashMap::new(),
            },
            score_sparse: 0.9,
            score_dense: 0.85,
            score_fused: 0.87,
            score_rerank: 0.92,
        };

        let json = serde_json::to_string(&hit).unwrap();
        let de: RetrievalHit = serde_json::from_str(&json).unwrap();
        assert_eq!(hit, de);
    }

    #[test]
    fn test_verdict_variants_serialize() {
        let pass = Verdict::Pass;
        let json = serde_json::to_string(&pass).unwrap();
        let de: Verdict = serde_json::from_str(&json).unwrap();
        assert_eq!(de, Verdict::Pass);

        let fail = Verdict::Fail {
            issues: vec![VerificationIssue {
                kind: IssueKind::MissingCitation,
                description: "no citation".to_string(),
                claim_text: Some("claim".to_string()),
            }],
        };
        let json = serde_json::to_string(&fail).unwrap();
        let de: Verdict = serde_json::from_str(&json).unwrap();
        assert_eq!(de, fail);

        let abstain = Verdict::Abstain {
            reason: "insufficient data".to_string(),
        };
        let json = serde_json::to_string(&abstain).unwrap();
        let de: Verdict = serde_json::from_str(&json).unwrap();
        assert_eq!(de, abstain);
    }

    #[test]
    fn test_compiled_context_json_roundtrip() {
        let ctx = CompiledContext {
            instruction_block: "Answer the question using the sources.".to_string(),
            sources: vec![Chunk {
                chunk_id: Uuid::new_v4(),
                doc_id: Uuid::new_v4(),
                doc_version: 1,
                source_path: "docs/rust.md".to_string(),
                section_path: vec!["Ownership".to_string()],
                char_offset: 0,
                char_len: 42,
                token_count: 10,
                text: "Rust uses ownership for memory management.".to_string(),
                metadata: HashMap::new(),
            }],
            output_schema: Some(serde_json::json!({"type": "object"})),
            budget: TokenBudget::from_fractions(4096, 0.15, 0.65, 0.10, 0.10),
        };

        let json = serde_json::to_string(&ctx).unwrap();
        let de: CompiledContext = serde_json::from_str(&json).unwrap();
        assert_eq!(ctx, de);
    }

    #[test]
    fn test_chat_message_json_roundtrip() {
        let messages = vec![
            ChatMessage {
                role: Role::System,
                content: "You are a helpful assistant.".to_string(),
            },
            ChatMessage {
                role: Role::User,
                content: "Hello".to_string(),
            },
            ChatMessage {
                role: Role::Assistant,
                content: "Hi there!".to_string(),
            },
        ];

        let json = serde_json::to_string(&messages).unwrap();
        let de: Vec<ChatMessage> = serde_json::from_str(&json).unwrap();
        assert_eq!(messages, de);
    }

    #[test]
    fn test_token_budget_fractions_exceed_one() {
        // Fractions sum to 1.5 — remaining should be 0 via saturating_sub
        let budget = TokenBudget::from_fractions(1000, 0.50, 0.50, 0.25, 0.25);
        // 500 + 500 + 250 + 250 = 1500 > 1000
        assert_eq!(budget.remaining, 0);
        assert_eq!(budget.instruction, 500);
        assert_eq!(budget.sources, 500);
        assert_eq!(budget.memory, 250);
        assert_eq!(budget.output_reserved, 250);
    }

    #[test]
    fn test_token_budget_all_zero_fractions() {
        let budget = TokenBudget::from_fractions(4096, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(budget.instruction, 0);
        assert_eq!(budget.sources, 0);
        assert_eq!(budget.memory, 0);
        assert_eq!(budget.output_reserved, 0);
        assert_eq!(budget.remaining, 4096);
    }

    #[test]
    fn test_chunk_empty_fields() {
        let chunk = Chunk {
            chunk_id: Uuid::nil(),
            doc_id: Uuid::nil(),
            doc_version: 0,
            source_path: String::new(),
            section_path: vec![],
            char_offset: 0,
            char_len: 0,
            token_count: 0,
            text: String::new(),
            metadata: HashMap::new(),
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let de: Chunk = serde_json::from_str(&json).unwrap();
        assert_eq!(chunk, de);
    }

    #[test]
    fn test_trait_objects_are_object_safe() {
        // Compile-time verification that all traits can be used as dyn trait objects.
        fn _assert_retriever(_: Box<dyn crate::traits::Retriever>) {}
        fn _assert_reranker(_: Box<dyn crate::traits::Reranker>) {}
        fn _assert_embedder(_: Box<dyn crate::traits::Embedder>) {}
        fn _assert_verifier(_: Box<dyn crate::traits::Verifier>) {}
        fn _assert_llm_client(_: Box<dyn crate::traits::LlmClient>) {}
        fn _assert_token_counter(_: Box<dyn crate::traits::TokenCounter>) {}
    }
}
