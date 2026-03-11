use std::collections::HashMap;
use std::sync::Arc;

use cwc_core::config::RetrievalConfig;
use cwc_core::error::Result;
use cwc_core::traits::{Embedder, Reranker, Retriever};
use cwc_core::types::RetrievalHit;
use uuid::Uuid;

use crate::mmr::mmr_select;
use crate::normalize::{min_max_normalize, ScoreField};
use crate::rrf::reciprocal_rank_fusion;

/// Hybrid retriever combining sparse (BM25) and dense (vector) retrieval.
///
/// Pipeline: sparse + dense → normalize → RRF fusion → rerank → MMR diversity → top_k
pub struct HybridRetriever {
    sparse: Arc<dyn Retriever>,
    dense: Arc<dyn Retriever>,
    embedder: Arc<dyn Embedder>,
    reranker: Arc<dyn Reranker>,
    config: RetrievalConfig,
}

impl HybridRetriever {
    pub fn new(
        sparse: Arc<dyn Retriever>,
        dense: Arc<dyn Retriever>,
        embedder: Arc<dyn Embedder>,
        reranker: Arc<dyn Reranker>,
        config: RetrievalConfig,
    ) -> Self {
        Self {
            sparse,
            dense,
            embedder,
            reranker,
            config,
        }
    }
}

impl Retriever for HybridRetriever {
    fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<RetrievalHit>> {
        // 1. Retrieve from both indexes
        let mut sparse_hits = self.sparse.retrieve(query, self.config.sparse_top_k)?;
        let mut dense_hits = self.dense.retrieve(query, self.config.dense_top_k)?;

        // 2. Normalize scores to [0, 1]
        min_max_normalize(&mut sparse_hits, ScoreField::Sparse);
        min_max_normalize(&mut dense_hits, ScoreField::Dense);

        // 3. RRF fusion
        let mut fused = reciprocal_rank_fusion(&[sparse_hits, dense_hits], self.config.rrf_k);

        // 4. Rerank (NoopReranker in simple mode just truncates)
        self.reranker
            .rerank(query, &mut fused, self.config.rerank_top_k)?;

        // 5. MMR diversity selection
        // Get embeddings for all candidate chunks
        let chunk_texts: Vec<&str> = fused.iter().map(|h| h.chunk.text.as_str()).collect();
        if chunk_texts.is_empty() {
            return Ok(vec![]);
        }

        let embeddings_vec = self.embedder.embed(&chunk_texts)?;
        let mut chunk_embeddings: HashMap<Uuid, Vec<f32>> = HashMap::new();
        for (hit, emb) in fused.iter().zip(embeddings_vec.iter()) {
            chunk_embeddings.insert(hit.chunk.chunk_id, emb.clone());
        }

        // Get query embedding
        let query_embeddings = self.embedder.embed(&[query])?;
        let query_embedding = &query_embeddings[0];

        let selected = mmr_select(
            &fused,
            query_embedding,
            &chunk_embeddings,
            self.config.mmr_lambda,
            top_k,
        );

        Ok(selected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cwc_core::noop::NoopReranker;
    use cwc_core::types::Chunk;
    use std::collections::HashMap as StdHashMap;

    /// Mock retriever that returns fixed hits.
    struct MockRetriever {
        hits: Vec<RetrievalHit>,
    }

    impl Retriever for MockRetriever {
        fn retrieve(&self, _query: &str, top_k: usize) -> Result<Vec<RetrievalHit>> {
            Ok(self.hits.iter().take(top_k).cloned().collect())
        }
    }

    /// Mock embedder that returns deterministic embeddings based on chunk text.
    struct MockEmbedder;

    impl Embedder for MockEmbedder {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| {
                    // Hash-based deterministic embedding
                    let hash = t.len() as f32;
                    let contains_ownership = if t.contains("ownership") { 1.0 } else { 0.0 };
                    let contains_borrow = if t.contains("borrow") { 1.0 } else { 0.0 };
                    let contains_trait = if t.contains("trait") { 1.0 } else { 0.0 };
                    vec![hash / 100.0, contains_ownership, contains_borrow, contains_trait]
                })
                .collect())
        }

        fn dim(&self) -> usize {
            4
        }
    }

    fn make_chunk_with(id: Uuid, text: &str) -> Chunk {
        Chunk {
            chunk_id: id,
            doc_id: Uuid::new_v5(&Uuid::NAMESPACE_OID, b"doc"),
            doc_version: 1,
            source_path: "test.md".to_string(),
            section_path: vec![],
            char_offset: 0,
            char_len: text.len(),
            token_count: 5,
            text: text.to_string(),
            metadata: StdHashMap::new(),
        }
    }

    fn make_hit(id: Uuid, text: &str, sparse: f32, dense: f32) -> RetrievalHit {
        RetrievalHit {
            chunk: make_chunk_with(id, text),
            score_sparse: sparse,
            score_dense: dense,
            score_fused: 0.0,
            score_rerank: 0.0,
        }
    }

    #[test]
    fn test_hybrid_combines_sparse_and_dense() {
        let id_a = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"a");
        let id_b = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"b");
        let id_c = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"c");

        let sparse = Arc::new(MockRetriever {
            hits: vec![
                make_hit(id_a, "ownership memory", 5.0, 0.0),
                make_hit(id_b, "borrowing rules", 3.0, 0.0),
            ],
        });
        let dense = Arc::new(MockRetriever {
            hits: vec![
                make_hit(id_b, "borrowing rules", 0.0, 0.9),
                make_hit(id_c, "trait implementation", 0.0, 0.7),
            ],
        });

        let config = RetrievalConfig {
            sparse_top_k: 10,
            dense_top_k: 10,
            rerank_top_k: 10,
            final_top_k: 10,
            mmr_lambda: 0.7,
            rrf_k: 60.0,
            score_threshold: 0.0,
        };

        let retriever = HybridRetriever::new(
            sparse,
            dense,
            Arc::new(MockEmbedder),
            Arc::new(NoopReranker),
            config,
        );

        let results = retriever.retrieve("ownership", 10).unwrap();

        // B should be ranked highest (in both lists)
        assert!(!results.is_empty());
        let b_hit = results.iter().find(|h| h.chunk.chunk_id == id_b).unwrap();
        assert!(b_hit.score_fused > 0.0);

        // All three chunks should appear
        let ids: Vec<Uuid> = results.iter().map(|h| h.chunk.chunk_id).collect();
        assert!(ids.contains(&id_a));
        assert!(ids.contains(&id_b));
        assert!(ids.contains(&id_c));
    }

    #[test]
    fn test_hybrid_sparse_only_match() {
        // Query matching BM25 terms but not semantics
        let id_bm25 = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"bm25");

        let sparse = Arc::new(MockRetriever {
            hits: vec![make_hit(id_bm25, "ownership memory safety", 8.0, 0.0)],
        });
        let dense = Arc::new(MockRetriever { hits: vec![] });

        let config = RetrievalConfig {
            sparse_top_k: 10,
            dense_top_k: 10,
            rerank_top_k: 10,
            final_top_k: 10,
            mmr_lambda: 1.0,
            rrf_k: 60.0,
            score_threshold: 0.0,
        };

        let retriever = HybridRetriever::new(
            sparse,
            dense,
            Arc::new(MockEmbedder),
            Arc::new(NoopReranker),
            config,
        );

        let results = retriever.retrieve("ownership", 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.chunk_id, id_bm25);
        assert!(results[0].score_sparse > 0.0);
    }

    #[test]
    fn test_hybrid_dense_only_match() {
        // Semantic paraphrase query → found via dense
        let id_sem = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"semantic");

        let sparse = Arc::new(MockRetriever { hits: vec![] });
        let dense = Arc::new(MockRetriever {
            hits: vec![make_hit(id_sem, "trait implementation", 0.0, 0.95)],
        });

        let config = RetrievalConfig {
            sparse_top_k: 10,
            dense_top_k: 10,
            rerank_top_k: 10,
            final_top_k: 10,
            mmr_lambda: 1.0,
            rrf_k: 60.0,
            score_threshold: 0.0,
        };

        let retriever = HybridRetriever::new(
            sparse,
            dense,
            Arc::new(MockEmbedder),
            Arc::new(NoopReranker),
            config,
        );

        let results = retriever.retrieve("interfaces in rust", 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.chunk_id, id_sem);
        assert!(results[0].score_dense > 0.0);
    }

    #[test]
    fn test_hybrid_all_scores_populated() {
        let id = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"scored");

        let sparse = Arc::new(MockRetriever {
            hits: vec![make_hit(id, "ownership borrowing", 5.0, 0.0)],
        });
        let dense = Arc::new(MockRetriever {
            hits: vec![make_hit(id, "ownership borrowing", 0.0, 0.8)],
        });

        let config = RetrievalConfig {
            sparse_top_k: 10,
            dense_top_k: 10,
            rerank_top_k: 10,
            final_top_k: 10,
            mmr_lambda: 0.7,
            rrf_k: 60.0,
            score_threshold: 0.0,
        };

        let retriever = HybridRetriever::new(
            sparse,
            dense,
            Arc::new(MockEmbedder),
            Arc::new(NoopReranker),
            config,
        );

        let results = retriever.retrieve("ownership", 5).unwrap();
        assert_eq!(results.len(), 1);
        let hit = &results[0];
        // sparse and dense should be preserved, fused should be > 0
        assert!(hit.score_sparse > 0.0, "sparse score should be populated");
        assert!(hit.score_dense > 0.0, "dense score should be populated");
        assert!(hit.score_fused > 0.0, "fused score should be populated");
    }

    #[test]
    fn test_hybrid_empty_both() {
        let sparse = Arc::new(MockRetriever { hits: vec![] });
        let dense = Arc::new(MockRetriever { hits: vec![] });

        let config = RetrievalConfig::default();
        let retriever = HybridRetriever::new(
            sparse,
            dense,
            Arc::new(MockEmbedder),
            Arc::new(NoopReranker),
            config,
        );

        let results = retriever.retrieve("anything", 5).unwrap();
        assert!(results.is_empty());
    }
}
