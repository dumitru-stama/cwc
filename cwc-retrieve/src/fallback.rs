use std::sync::Arc;

use cwc_core::config::RetrievalConfig;
use cwc_core::error::Result;
use cwc_core::traits::{Embedder, Reranker, Retriever};
use cwc_core::types::RetrievalHit;

use crate::mmr::mmr_select;
use crate::normalize::{min_max_normalize, ScoreField};
use crate::rrf::reciprocal_rank_fusion;

/// Describes which fallback was triggered during retrieval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FallbackBehavior {
    /// Dense retrieval failed, fell back to sparse only.
    SparseOnly,
    /// Reranking failed, using fusion scores.
    SkipRerank,
    /// Embedding failed, skipping dense retrieval.
    SkipDense,
}

/// Result of a retrieval that may include fallback information.
#[derive(Debug)]
pub struct ResilientResult {
    pub hits: Vec<RetrievalHit>,
    pub fallbacks: Vec<FallbackBehavior>,
}

/// A retriever with graceful degradation.
///
/// If dense retrieval fails → fall back to sparse only.
/// If reranking fails → use fusion scores.
/// If embedding fails → skip dense retrieval.
pub struct ResilientRetriever {
    sparse: Arc<dyn Retriever>,
    dense: Arc<dyn Retriever>,
    embedder: Arc<dyn Embedder>,
    reranker: Arc<dyn Reranker>,
    config: RetrievalConfig,
}

impl ResilientRetriever {
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

    /// Retrieve with graceful fallbacks on component failures.
    pub fn retrieve_resilient(&self, query: &str, top_k: usize) -> Result<ResilientResult> {
        let mut fallbacks = Vec::new();

        // Sparse retrieval — must succeed (it's the baseline)
        let mut sparse_hits = self.sparse.retrieve(query, self.config.sparse_top_k)?;
        min_max_normalize(&mut sparse_hits, ScoreField::Sparse);

        // Dense retrieval — fallback to sparse-only on error
        let dense_result = self.dense.retrieve(query, self.config.dense_top_k);
        let dense_hits = match dense_result {
            Ok(mut hits) => {
                min_max_normalize(&mut hits, ScoreField::Dense);
                hits
            }
            Err(e) => {
                tracing::warn!("dense retrieval failed, falling back to sparse only: {e}");
                fallbacks.push(FallbackBehavior::SparseOnly);
                vec![]
            }
        };

        // Special case: if dense failed and we got no dense hits,
        // set fused scores from sparse only
        let mut fused = if dense_hits.is_empty() && fallbacks.contains(&FallbackBehavior::SparseOnly) {
            // Just use sparse hits with fused = sparse score
            for hit in &mut sparse_hits {
                hit.score_fused = hit.score_sparse;
            }
            sparse_hits
        } else {
            reciprocal_rank_fusion(&[sparse_hits, dense_hits], self.config.rrf_k)
        };

        // Rerank — fallback to fusion scores on error
        let rerank_result = self.reranker.rerank(query, &mut fused, self.config.rerank_top_k);
        if let Err(e) = rerank_result {
            tracing::warn!("reranking failed, using fusion scores: {e}");
            fallbacks.push(FallbackBehavior::SkipRerank);
            // Just truncate to rerank_top_k
            fused.truncate(self.config.rerank_top_k);
        }

        // MMR diversity — needs embeddings
        let chunk_texts: Vec<&str> = fused.iter().map(|h| h.chunk.text.as_str()).collect();
        if chunk_texts.is_empty() {
            return Ok(ResilientResult {
                hits: vec![],
                fallbacks,
            });
        }

        let embed_result = self.embedder.embed(&chunk_texts);
        let selected = match embed_result {
            Ok(embeddings_vec) => {
                let mut chunk_embeddings = std::collections::HashMap::new();
                for (hit, emb) in fused.iter().zip(embeddings_vec.iter()) {
                    chunk_embeddings.insert(hit.chunk.chunk_id, emb.clone());
                }

                let query_emb_result = self.embedder.embed(&[query]);
                match query_emb_result {
                    Ok(query_embs) => {
                        mmr_select(&fused, &query_embs[0], &chunk_embeddings, self.config.mmr_lambda, top_k)
                    }
                    Err(e) => {
                        tracing::warn!("query embedding failed, skipping MMR diversity: {e}");
                        fallbacks.push(FallbackBehavior::SkipDense);
                        fused.into_iter().take(top_k).collect()
                    }
                }
            }
            Err(e) => {
                tracing::warn!("chunk embedding failed, skipping MMR diversity: {e}");
                fallbacks.push(FallbackBehavior::SkipDense);
                fused.into_iter().take(top_k).collect()
            }
        };

        Ok(ResilientResult {
            hits: selected,
            fallbacks,
        })
    }
}

/// Also implement the standard Retriever trait (without fallback info).
impl Retriever for ResilientRetriever {
    fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<RetrievalHit>> {
        let result = self.retrieve_resilient(query, top_k)?;
        Ok(result.hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cwc_core::error::CwcError;
    use cwc_core::noop::NoopReranker;
    use cwc_core::types::Chunk;
    use std::collections::HashMap;
    use uuid::Uuid;

    fn make_hit(text: &str, sparse: f32) -> RetrievalHit {
        RetrievalHit {
            chunk: Chunk {
                chunk_id: Uuid::new_v4(),
                doc_id: Uuid::new_v4(),
                doc_version: 1,
                source_path: "test.md".to_string(),
                section_path: vec![],
                char_offset: 0,
                char_len: text.len(),
                token_count: 5,
                text: text.to_string(),
                metadata: HashMap::new(),
            },
            score_sparse: sparse,
            score_dense: 0.0,
            score_fused: 0.0,
            score_rerank: 0.0,
        }
    }

    struct OkRetriever {
        hits: Vec<RetrievalHit>,
    }
    impl Retriever for OkRetriever {
        fn retrieve(&self, _q: &str, k: usize) -> Result<Vec<RetrievalHit>> {
            Ok(self.hits.iter().take(k).cloned().collect())
        }
    }

    struct FailRetriever;
    impl Retriever for FailRetriever {
        fn retrieve(&self, _q: &str, _k: usize) -> Result<Vec<RetrievalHit>> {
            Err(CwcError::Retrieval("dense index unavailable".to_string()))
        }
    }

    struct FailReranker;
    impl Reranker for FailReranker {
        fn rerank(&self, _q: &str, _hits: &mut Vec<RetrievalHit>, _k: usize) -> Result<()> {
            Err(CwcError::Retrieval("reranker timeout".to_string()))
        }
    }

    struct SimpleEmbedder;
    impl Embedder for SimpleEmbedder {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|t| vec![t.len() as f32 / 100.0, 0.5]).collect())
        }
        fn dim(&self) -> usize {
            2
        }
    }

    #[test]
    fn test_fallback_dense_error_returns_sparse_only() {
        let sparse = Arc::new(OkRetriever {
            hits: vec![
                make_hit("rust ownership model", 5.0),
                make_hit("borrowing rules in rust", 3.0),
            ],
        });
        let dense: Arc<dyn Retriever> = Arc::new(FailRetriever);

        let retriever = ResilientRetriever::new(
            sparse,
            dense,
            Arc::new(SimpleEmbedder),
            Arc::new(NoopReranker),
            RetrievalConfig::default(),
        );

        let result = retriever.retrieve_resilient("rust", 10).unwrap();
        assert!(!result.hits.is_empty());
        assert!(result.fallbacks.contains(&FallbackBehavior::SparseOnly));
    }

    #[test]
    fn test_fallback_rerank_error_uses_fusion_scores() {
        let hit1 = make_hit("ownership memory", 5.0);
        let sparse = Arc::new(OkRetriever {
            hits: vec![hit1],
        });
        let dense = Arc::new(OkRetriever { hits: vec![] });

        let retriever = ResilientRetriever::new(
            sparse,
            dense,
            Arc::new(SimpleEmbedder),
            Arc::new(FailReranker),
            RetrievalConfig::default(),
        );

        let result = retriever.retrieve_resilient("ownership", 10).unwrap();
        assert!(!result.hits.is_empty());
        assert!(result.fallbacks.contains(&FallbackBehavior::SkipRerank));
    }

    #[test]
    fn test_no_fallback_when_all_ok() {
        let sparse = Arc::new(OkRetriever {
            hits: vec![make_hit("ownership model", 5.0)],
        });
        let dense = Arc::new(OkRetriever {
            hits: vec![make_hit("borrow checker rules", 0.9)],
        });

        let retriever = ResilientRetriever::new(
            sparse,
            dense,
            Arc::new(SimpleEmbedder),
            Arc::new(NoopReranker),
            RetrievalConfig::default(),
        );

        let result = retriever.retrieve_resilient("rust", 10).unwrap();
        assert!(result.fallbacks.is_empty(), "no fallbacks needed");
    }

    #[test]
    fn test_fallback_empty_sparse_and_dense() {
        // Both retrievers return empty results — should not crash, just return empty
        let sparse = Arc::new(OkRetriever { hits: vec![] });
        let dense = Arc::new(OkRetriever { hits: vec![] });

        let retriever = ResilientRetriever::new(
            sparse,
            dense,
            Arc::new(SimpleEmbedder),
            Arc::new(NoopReranker),
            RetrievalConfig::default(),
        );

        let result = retriever.retrieve_resilient("query", 10).unwrap();
        assert!(result.hits.is_empty());
        assert!(result.fallbacks.is_empty());
    }

    #[test]
    fn test_resilient_retriever_trait_impl() {
        // Test the Retriever trait impl wrapping retrieve_resilient
        let sparse = Arc::new(OkRetriever {
            hits: vec![make_hit("test doc", 3.0)],
        });
        let dense: Arc<dyn Retriever> = Arc::new(FailRetriever);

        let retriever = ResilientRetriever::new(
            sparse,
            dense,
            Arc::new(SimpleEmbedder),
            Arc::new(NoopReranker),
            RetrievalConfig::default(),
        );

        // Use the Retriever trait (not retrieve_resilient) — should still work
        let hits = Retriever::retrieve(&retriever, "test", 10).unwrap();
        assert!(!hits.is_empty());
    }

    #[test]
    fn test_fallback_embed_error_skips_mmr() {
        struct FailEmbedder;
        impl Embedder for FailEmbedder {
            fn embed(&self, _texts: &[&str]) -> Result<Vec<Vec<f32>>> {
                Err(CwcError::Embedding("GPU OOM".to_string()))
            }
            fn dim(&self) -> usize { 2 }
        }

        let sparse = Arc::new(OkRetriever {
            hits: vec![make_hit("doc about rust", 5.0)],
        });
        let dense = Arc::new(OkRetriever {
            hits: vec![make_hit("rust ownership", 0.8)],
        });

        let retriever = ResilientRetriever::new(
            sparse,
            dense,
            Arc::new(FailEmbedder),
            Arc::new(NoopReranker),
            RetrievalConfig::default(),
        );

        let result = retriever.retrieve_resilient("rust", 10).unwrap();
        assert!(!result.hits.is_empty());
        assert!(result.fallbacks.contains(&FallbackBehavior::SkipDense));
    }

    #[test]
    fn test_cache_invalidation_after_ingest() {
        // Tests that RetrievalCache can be cleared (simulating post-ingest invalidation)
        let rc = cwc_core::cache::RetrievalCache::new();
        let hits = vec![make_hit("cached result", 1.0)];
        rc.put("query", &[], hits);
        assert!(rc.get("query", &[]).is_some());

        // Simulate ingest → clear cache
        rc.clear();
        assert!(rc.get("query", &[]).is_none());
    }
}
