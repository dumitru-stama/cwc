use std::sync::Arc;

use cwc_core::traits::{Embedder, Retriever};
use cwc_core::types::RetrievalHit;

use crate::db::ChunkDb;
use crate::memory_index::InMemoryVectorIndex;

/// Dense retriever backed by pgvector (PostgreSQL).
///
/// Bridges the async ChunkDb into the sync Retriever trait by using
/// a tokio runtime handle.
pub struct DenseRetriever {
    db: ChunkDb,
    embedder: Arc<dyn Embedder>,
    handle: tokio::runtime::Handle,
}

impl DenseRetriever {
    pub fn new(
        db: ChunkDb,
        embedder: Arc<dyn Embedder>,
        handle: tokio::runtime::Handle,
    ) -> Self {
        Self {
            db,
            embedder,
            handle,
        }
    }
}

impl Retriever for DenseRetriever {
    fn retrieve(&self, query: &str, top_k: usize) -> cwc_core::Result<Vec<RetrievalHit>> {
        let embs = self.embedder.embed(&[query])?;
        let query_emb = &embs[0];

        let results = self
            .handle
            .block_on(self.db.search_dense(query_emb, top_k))
            .map_err(|e| cwc_core::CwcError::Retrieval(e.to_string()))?;

        Ok(results
            .into_iter()
            .map(|(chunk, score)| RetrievalHit {
                chunk,
                score_sparse: 0.0,
                score_dense: score,
                score_fused: 0.0,
                score_rerank: 0.0,
            })
            .collect())
    }
}

/// Dense retriever backed by an in-memory vector index.
pub struct InMemoryDenseRetriever {
    index: InMemoryVectorIndex,
    embedder: Arc<dyn Embedder>,
}

impl InMemoryDenseRetriever {
    pub fn new(index: InMemoryVectorIndex, embedder: Arc<dyn Embedder>) -> Self {
        Self { index, embedder }
    }
}

impl Retriever for InMemoryDenseRetriever {
    fn retrieve(&self, query: &str, top_k: usize) -> cwc_core::Result<Vec<RetrievalHit>> {
        let embs = self.embedder.embed(&[query])?;
        let query_emb = &embs[0];

        let results = self.index.search(query_emb, top_k);

        Ok(results
            .into_iter()
            .map(|(chunk, score)| RetrievalHit {
                chunk,
                score_sparse: 0.0,
                score_dense: score,
                score_fused: 0.0,
                score_rerank: 0.0,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use uuid::Uuid;

    use cwc_core::types::Chunk;

    /// Mock embedder that returns known embeddings based on text content.
    struct MockEmbedder;

    impl Embedder for MockEmbedder {
        fn embed(&self, texts: &[&str]) -> cwc_core::Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| {
                    if t.contains("ownership") {
                        vec![0.9, 0.1, 0.0, 0.0]
                    } else if t.contains("borrowing") {
                        vec![0.1, 0.9, 0.0, 0.0]
                    } else if t.contains("traits") {
                        vec![0.0, 0.0, 0.9, 0.1]
                    } else if t.contains("error") {
                        vec![0.0, 0.0, 0.1, 0.9]
                    } else {
                        vec![0.25, 0.25, 0.25, 0.25]
                    }
                })
                .collect())
        }

        fn dim(&self) -> usize {
            4
        }
    }

    fn make_chunk(doc_id: Uuid, text: &str, idx: usize) -> Chunk {
        Chunk {
            chunk_id: Uuid::new_v5(
                &Uuid::NAMESPACE_OID,
                format!("{doc_id}-{idx}").as_bytes(),
            ),
            doc_id,
            doc_version: 1,
            source_path: "test.md".to_string(),
            section_path: vec!["Section".to_string()],
            char_offset: idx * 100,
            char_len: text.len(),
            token_count: 10,
            text: text.to_string(),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_in_memory_dense_retriever_trait() {
        let doc_id = Uuid::new_v4();
        let mut index = InMemoryVectorIndex::new(4);

        // Insert chunks with embeddings matching the mock embedder's logic
        index.insert(
            make_chunk(doc_id, "Rust ownership memory management", 0),
            vec![0.9, 0.1, 0.0, 0.0],
        );
        index.insert(
            make_chunk(doc_id, "Borrowing allows references", 1),
            vec![0.1, 0.9, 0.0, 0.0],
        );
        index.insert(
            make_chunk(doc_id, "Traits define shared behavior", 2),
            vec![0.0, 0.0, 0.9, 0.1],
        );

        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder);
        let retriever = InMemoryDenseRetriever::new(index, embedder);

        // Use the Retriever trait
        let results: Vec<RetrievalHit> =
            Retriever::retrieve(&retriever, "ownership", 3).unwrap();
        assert_eq!(results.len(), 3);
        assert!(results[0].chunk.text.contains("ownership"));
    }

    #[test]
    fn test_in_memory_dense_retriever_scores() {
        let doc_id = Uuid::new_v4();
        let mut index = InMemoryVectorIndex::new(4);

        index.insert(
            make_chunk(doc_id, "ownership", 0),
            vec![1.0, 0.0, 0.0, 0.0],
        );
        index.insert(
            make_chunk(doc_id, "borrowing", 1),
            vec![0.0, 1.0, 0.0, 0.0],
        );

        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder);
        let retriever = InMemoryDenseRetriever::new(index, embedder);

        let results = Retriever::retrieve(&retriever, "ownership", 2).unwrap();

        // score_dense should be populated, others zero
        assert!(results[0].score_dense > 0.0);
        assert_eq!(results[0].score_sparse, 0.0);
        assert_eq!(results[0].score_fused, 0.0);
        assert_eq!(results[0].score_rerank, 0.0);
    }

    #[test]
    fn test_in_memory_dense_retriever_empty() {
        let index = InMemoryVectorIndex::new(4);
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder);
        let retriever = InMemoryDenseRetriever::new(index, embedder);

        let results = Retriever::retrieve(&retriever, "ownership", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_sparse_vs_dense_comparison() {
        use crate::sparse::SparseIndex;

        let doc_id = Uuid::new_v4();

        // Build sparse index
        let sparse_idx = SparseIndex::open_in_ram().unwrap();
        let chunks = vec![
            make_chunk(doc_id, "Rust ownership for memory management without garbage collector", 0),
            make_chunk(doc_id, "Borrowing allows references to data without taking ownership", 1),
            make_chunk(doc_id, "Traits define shared behavior similar to interfaces", 2),
            make_chunk(doc_id, "Error handling uses Result and Option types", 3),
        ];
        sparse_idx.index_chunks(&chunks).unwrap();

        // Build in-memory dense index
        let mut dense_idx = InMemoryVectorIndex::new(4);
        dense_idx.insert(chunks[0].clone(), vec![0.9, 0.1, 0.0, 0.0]);
        dense_idx.insert(chunks[1].clone(), vec![0.1, 0.9, 0.0, 0.0]);
        dense_idx.insert(chunks[2].clone(), vec![0.0, 0.0, 0.9, 0.1]);
        dense_idx.insert(chunks[3].clone(), vec![0.0, 0.0, 0.1, 0.9]);

        // Sparse search for "ownership"
        let sparse_results = sparse_idx.search("ownership", 4).unwrap();

        // Dense search for "ownership" (using mock embedder)
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder);
        let dense_retriever = InMemoryDenseRetriever::new(dense_idx, embedder);
        let dense_results = Retriever::retrieve(&dense_retriever, "ownership", 4).unwrap();

        // Both should return results
        assert!(!sparse_results.is_empty(), "sparse should find results");
        assert!(!dense_results.is_empty(), "dense should find results");

        // Both should rank ownership chunk first (for this query)
        assert!(
            sparse_results[0].chunk.text.contains("ownership"),
            "sparse first result should be about ownership"
        );
        assert!(
            dense_results[0].chunk.text.contains("ownership"),
            "dense first result should be about ownership"
        );

        // Sparse uses score_sparse, dense uses score_dense
        assert!(sparse_results[0].score_sparse > 0.0);
        assert_eq!(sparse_results[0].score_dense, 0.0);
        assert!(dense_results[0].score_dense > 0.0);
        assert_eq!(dense_results[0].score_sparse, 0.0);
    }
}
