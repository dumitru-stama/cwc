use crate::error::Result;
use crate::traits::{Reranker, Verifier};
use crate::types::{Chunk, RetrievalHit, Verdict};

/// A reranker that does nothing — returns hits in their original order,
/// truncated to top_k. This is the simple-mode default.
pub struct NoopReranker;

impl Reranker for NoopReranker {
    fn rerank(&self, _query: &str, hits: &mut Vec<RetrievalHit>, top_k: usize) -> Result<()> {
        hits.truncate(top_k);
        Ok(())
    }
}

/// A verifier that always passes. This is the simple-mode default
/// before heuristic verification is implemented.
pub struct NoopVerifier;

impl Verifier for NoopVerifier {
    fn verify(&self, _output: &str, _sources: &[Chunk]) -> Result<Verdict> {
        Ok(Verdict::Pass)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use uuid::Uuid;

    fn make_hit(score: f32) -> RetrievalHit {
        RetrievalHit {
            chunk: Chunk {
                chunk_id: Uuid::new_v4(),
                doc_id: Uuid::new_v4(),
                doc_version: 1,
                source_path: "test.md".to_string(),
                section_path: vec![],
                char_offset: 0,
                char_len: 5,
                token_count: 3,
                text: "hello".to_string(),
                metadata: HashMap::new(),
            },
            score_sparse: score,
            score_dense: 0.0,
            score_fused: score,
            score_rerank: 0.0,
        }
    }

    #[test]
    fn test_noop_reranker_preserves_order_and_truncates() {
        let reranker = NoopReranker;
        let mut hits = vec![make_hit(1.0), make_hit(0.8), make_hit(0.5), make_hit(0.3)];

        // Store original scores to verify order preservation
        let original_scores: Vec<f32> = hits.iter().map(|h| h.score_sparse).collect();

        reranker.rerank("query", &mut hits, 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert!((hits[0].score_sparse - original_scores[0]).abs() < f32::EPSILON);
        assert!((hits[1].score_sparse - original_scores[1]).abs() < f32::EPSILON);
    }

    #[test]
    fn test_noop_reranker_top_k_larger_than_hits() {
        let reranker = NoopReranker;
        let mut hits = vec![make_hit(1.0), make_hit(0.5)];
        reranker.rerank("query", &mut hits, 10).unwrap();
        assert_eq!(hits.len(), 2); // no change
    }

    #[test]
    fn test_noop_reranker_empty() {
        let reranker = NoopReranker;
        let mut hits: Vec<RetrievalHit> = vec![];
        reranker.rerank("query", &mut hits, 5).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn test_noop_reranker_top_k_zero() {
        let reranker = NoopReranker;
        let mut hits = vec![make_hit(1.0), make_hit(0.5)];
        reranker.rerank("query", &mut hits, 0).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn test_noop_verifier_always_passes() {
        let verifier = NoopVerifier;
        let result = verifier.verify("any output", &[]).unwrap();
        assert!(matches!(result, Verdict::Pass));
    }

    #[test]
    fn test_noop_verifier_passes_with_sources() {
        let verifier = NoopVerifier;
        let chunk = Chunk {
            chunk_id: Uuid::new_v4(),
            doc_id: Uuid::new_v4(),
            doc_version: 1,
            source_path: "test.md".to_string(),
            section_path: vec!["Section 1".to_string()],
            char_offset: 0,
            char_len: 10,
            token_count: 5,
            text: "some source text".to_string(),
            metadata: HashMap::new(),
        };
        let result = verifier.verify("output referencing source", &[chunk]).unwrap();
        assert!(matches!(result, Verdict::Pass));
    }
}
