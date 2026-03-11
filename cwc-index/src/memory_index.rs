use std::path::Path;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use cwc_core::types::Chunk;

use crate::error::{IndexError, Result};

/// In-memory vector index with brute-force cosine similarity search.
///
/// Alternative to pgvector for users who don't want PostgreSQL.
/// Uses exact search (not approximate), which is correct for < 100K chunks.
#[derive(Serialize, Deserialize)]
pub struct InMemoryVectorIndex {
    dim: usize,
    entries: Vec<VectorEntry>,
}

#[derive(Serialize, Deserialize)]
struct VectorEntry {
    chunk: Chunk,
    embedding: Vec<f32>,
}

impl InMemoryVectorIndex {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            entries: Vec::new(),
        }
    }

    /// Insert a chunk with its embedding. If chunk_id already exists, replaces it.
    ///
    /// # Panics
    /// Debug-asserts that `embedding.len() == self.dim`.
    pub fn insert(&mut self, chunk: Chunk, embedding: Vec<f32>) {
        debug_assert_eq!(
            embedding.len(),
            self.dim,
            "embedding dimension {} does not match index dimension {}",
            embedding.len(),
            self.dim
        );
        // Remove existing entry with same chunk_id
        self.entries.retain(|e| e.chunk.chunk_id != chunk.chunk_id);
        self.entries.push(VectorEntry { chunk, embedding });
    }

    /// Batch insert chunks with embeddings.
    pub fn insert_batch(&mut self, items: Vec<(Chunk, Vec<f32>)>) {
        for (chunk, embedding) in items {
            self.insert(chunk, embedding);
        }
    }

    /// Search for the top-k most similar chunks to the query embedding.
    /// Returns (chunk, cosine_similarity) pairs sorted by descending similarity.
    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<(Chunk, f32)> {
        if self.entries.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| (i, cosine_similarity(query_embedding, &e.embedding)))
            .collect();

        // Sort descending by similarity
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(top_k)
            .map(|(i, score)| (self.entries[i].chunk.clone(), score))
            .collect()
    }

    /// Search filtered by document IDs.
    pub fn search_filtered(
        &self,
        query_embedding: &[f32],
        doc_ids: &[Uuid],
        top_k: usize,
    ) -> Vec<(Chunk, f32)> {
        if self.entries.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| doc_ids.contains(&e.chunk.doc_id))
            .map(|(i, e)| (i, cosine_similarity(query_embedding, &e.embedding)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(top_k)
            .map(|(i, score)| (self.entries[i].chunk.clone(), score))
            .collect()
    }

    /// Delete all entries for a document. Returns number deleted.
    pub fn delete_document(&mut self, doc_id: Uuid) -> u32 {
        let before = self.entries.len();
        self.entries.retain(|e| e.chunk.doc_id != doc_id);
        (before - self.entries.len()) as u32
    }

    /// Number of indexed entries.
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Save index to disk using bincode.
    pub fn save(&self, path: &Path) -> Result<()> {
        let bytes =
            bincode::serialize(self).map_err(|e| IndexError::Index(e.to_string()))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load index from disk.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let index: Self =
            bincode::deserialize(&bytes).map_err(|e| IndexError::Index(e.to_string()))?;
        Ok(index)
    }
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

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
    fn test_memory_insert_and_count() {
        let mut idx = InMemoryVectorIndex::new(4);
        let doc_id = Uuid::new_v4();

        idx.insert(make_chunk(doc_id, "Hello", 0), vec![1.0, 0.0, 0.0, 0.0]);
        idx.insert(make_chunk(doc_id, "World", 1), vec![0.0, 1.0, 0.0, 0.0]);

        assert_eq!(idx.count(), 2);
    }

    #[test]
    fn test_memory_100_chunks() {
        let mut idx = InMemoryVectorIndex::new(4);
        let doc_id = Uuid::new_v4();

        for i in 0..100 {
            let emb = vec![i as f32, 0.0, 0.0, 0.0];
            idx.insert(make_chunk(doc_id, &format!("Chunk {i}"), i), emb);
        }

        assert_eq!(idx.count(), 100);
    }

    #[test]
    fn test_memory_search_relevance() {
        let mut idx = InMemoryVectorIndex::new(4);
        let doc_id = Uuid::new_v4();

        // Three chunks in orthogonal directions
        idx.insert(
            make_chunk(doc_id, "ownership", 0),
            vec![1.0, 0.0, 0.0, 0.0],
        );
        idx.insert(
            make_chunk(doc_id, "borrowing", 1),
            vec![0.0, 1.0, 0.0, 0.0],
        );
        idx.insert(
            make_chunk(doc_id, "traits", 2),
            vec![0.0, 0.0, 1.0, 0.0],
        );

        // Query close to ownership direction
        let results = idx.search(&[0.9, 0.1, 0.0, 0.0], 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0.text, "ownership");
    }

    #[test]
    fn test_memory_search_score_range() {
        let mut idx = InMemoryVectorIndex::new(4);
        let doc_id = Uuid::new_v4();

        idx.insert(make_chunk(doc_id, "a", 0), vec![1.0, 0.0, 0.0, 0.0]);
        idx.insert(make_chunk(doc_id, "b", 1), vec![0.0, 1.0, 0.0, 0.0]);
        idx.insert(make_chunk(doc_id, "c", 2), vec![-1.0, 0.0, 0.0, 0.0]);

        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 3);
        for (_, score) in &results {
            assert!(
                *score >= -1.0 && *score <= 1.0,
                "cosine score {score} out of range"
            );
        }
        // First result should have similarity ~1.0
        assert!((results[0].1 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_memory_search_scores_descending() {
        let mut idx = InMemoryVectorIndex::new(4);
        let doc_id = Uuid::new_v4();

        for i in 0..10 {
            let mut emb = vec![0.0; 4];
            emb[i % 4] = 1.0;
            idx.insert(make_chunk(doc_id, &format!("chunk {i}"), i), emb);
        }

        let results = idx.search(&[1.0, 0.5, 0.0, 0.0], 10);
        for w in results.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "scores not descending: {} >= {}",
                w[0].1,
                w[1].1
            );
        }
    }

    #[test]
    fn test_memory_search_top_k() {
        let mut idx = InMemoryVectorIndex::new(4);
        let doc_id = Uuid::new_v4();

        for i in 0..20 {
            idx.insert(
                make_chunk(doc_id, &format!("chunk {i}"), i),
                vec![i as f32, 0.0, 0.0, 0.0],
            );
        }

        let results = idx.search(&[10.0, 0.0, 0.0, 0.0], 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_memory_search_filtered() {
        let mut idx = InMemoryVectorIndex::new(4);
        let doc_a = Uuid::new_v4();
        let doc_b = Uuid::new_v4();

        idx.insert(
            make_chunk(doc_a, "Doc A ownership", 0),
            vec![1.0, 0.0, 0.0, 0.0],
        );
        idx.insert(
            make_chunk(doc_b, "Doc B ownership", 0),
            vec![1.0, 0.0, 0.0, 0.0],
        );

        let results = idx.search_filtered(&[1.0, 0.0, 0.0, 0.0], &[doc_a], 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.doc_id, doc_a);
    }

    #[test]
    fn test_memory_delete_document() {
        let mut idx = InMemoryVectorIndex::new(4);
        let doc_a = Uuid::new_v4();
        let doc_b = Uuid::new_v4();

        idx.insert(
            make_chunk(doc_a, "A1", 0),
            vec![1.0, 0.0, 0.0, 0.0],
        );
        idx.insert(
            make_chunk(doc_a, "A2", 1),
            vec![0.0, 1.0, 0.0, 0.0],
        );
        idx.insert(
            make_chunk(doc_b, "B1", 0),
            vec![0.0, 0.0, 1.0, 0.0],
        );

        assert_eq!(idx.count(), 3);
        let deleted = idx.delete_document(doc_a);
        assert_eq!(deleted, 2);
        assert_eq!(idx.count(), 1);

        // Only doc_b chunk remains
        let results = idx.search(&[0.0, 0.0, 1.0, 0.0], 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.doc_id, doc_b);
    }

    #[test]
    fn test_memory_empty_search() {
        let idx = InMemoryVectorIndex::new(4);
        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_memory_upsert() {
        let mut idx = InMemoryVectorIndex::new(4);
        let doc_id = Uuid::new_v4();

        let chunk = make_chunk(doc_id, "Original text", 0);
        idx.insert(chunk.clone(), vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(idx.count(), 1);

        // Re-insert same chunk_id with different embedding
        let mut updated = chunk;
        updated.doc_version = 2;
        updated.text = "Updated text".to_string();
        idx.insert(updated, vec![0.0, 1.0, 0.0, 0.0]);

        assert_eq!(idx.count(), 1); // No duplicates
        let results = idx.search(&[0.0, 1.0, 0.0, 0.0], 1);
        assert_eq!(results[0].0.text, "Updated text");
    }

    #[test]
    fn test_memory_save_load() {
        let mut idx = InMemoryVectorIndex::new(4);
        let doc_id = Uuid::new_v4();

        idx.insert(
            make_chunk(doc_id, "Saved chunk", 0),
            vec![1.0, 2.0, 3.0, 4.0],
        );
        idx.insert(
            make_chunk(doc_id, "Another chunk", 1),
            vec![5.0, 6.0, 7.0, 8.0],
        );

        let path = std::env::temp_dir().join("cwc_memory_index_test.bin");
        idx.save(&path).unwrap();

        let loaded = InMemoryVectorIndex::load(&path).unwrap();
        assert_eq!(loaded.count(), 2);
        assert_eq!(loaded.dim(), 4);

        let results = loaded.search(&[1.0, 2.0, 3.0, 4.0], 1);
        assert_eq!(results[0].0.text, "Saved chunk");

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_memory_cosine_similarity() {
        // Identical vectors → 1.0
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);

        // Orthogonal → 0.0
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);

        // Opposite → -1.0
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);

        // Zero vector → 0.0
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }

    #[test]
    fn test_memory_batch_insert() {
        let mut idx = InMemoryVectorIndex::new(4);
        let doc_id = Uuid::new_v4();

        let items: Vec<(Chunk, Vec<f32>)> = (0..5)
            .map(|i| {
                let chunk = make_chunk(doc_id, &format!("Batch chunk {i}"), i);
                let emb = vec![i as f32, 0.0, 0.0, 0.0];
                (chunk, emb)
            })
            .collect();

        idx.insert_batch(items);
        assert_eq!(idx.count(), 5);
    }

    #[test]
    fn test_memory_delete_nonexistent() {
        let mut idx = InMemoryVectorIndex::new(4);
        let deleted = idx.delete_document(Uuid::new_v4());
        assert_eq!(deleted, 0);
    }

    #[test]
    fn test_memory_search_top_k_zero() {
        let mut idx = InMemoryVectorIndex::new(4);
        let doc_id = Uuid::new_v4();
        idx.insert(make_chunk(doc_id, "chunk", 0), vec![1.0, 0.0, 0.0, 0.0]);

        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_memory_filtered_empty_doc_ids() {
        let mut idx = InMemoryVectorIndex::new(4);
        let doc_id = Uuid::new_v4();
        idx.insert(make_chunk(doc_id, "chunk", 0), vec![1.0, 0.0, 0.0, 0.0]);

        let results = idx.search_filtered(&[1.0, 0.0, 0.0, 0.0], &[], 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_memory_multiple_documents() {
        let mut idx = InMemoryVectorIndex::new(4);
        let doc_a = Uuid::new_v4();
        let doc_b = Uuid::new_v4();
        let doc_c = Uuid::new_v4();

        idx.insert(make_chunk(doc_a, "A1", 0), vec![1.0, 0.0, 0.0, 0.0]);
        idx.insert(make_chunk(doc_a, "A2", 1), vec![0.9, 0.1, 0.0, 0.0]);
        idx.insert(make_chunk(doc_b, "B1", 0), vec![0.0, 1.0, 0.0, 0.0]);
        idx.insert(make_chunk(doc_c, "C1", 0), vec![0.0, 0.0, 1.0, 0.0]);

        // Unfiltered: all 4 results
        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 10);
        assert_eq!(results.len(), 4);

        // Filter to doc_a + doc_c
        let results = idx.search_filtered(&[1.0, 0.0, 0.0, 0.0], &[doc_a, doc_c], 10);
        assert_eq!(results.len(), 3); // A1, A2, C1
        for (chunk, _) in &results {
            assert!(chunk.doc_id == doc_a || chunk.doc_id == doc_c);
        }
    }

    #[test]
    fn test_memory_large_scale_search() {
        let mut idx = InMemoryVectorIndex::new(384);
        let doc_id = Uuid::new_v4();

        for i in 0..10_000 {
            let mut emb = vec![0.0f32; 384];
            emb[i % 384] = 1.0;
            emb[(i + 1) % 384] = 0.5;
            idx.insert(make_chunk(doc_id, &format!("Chunk {i}"), i), emb);
        }

        assert_eq!(idx.count(), 10_000);

        let mut query = vec![0.0f32; 384];
        query[0] = 1.0;
        query[1] = 0.5;

        let start = std::time::Instant::now();
        let results = idx.search(&query, 10);
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 10);
        assert!(
            elapsed.as_millis() < 500,
            "search over 10K x 384-dim took {}ms, expected < 500ms",
            elapsed.as_millis()
        );
    }

    #[test]
    fn test_memory_save_load_preserves_search() {
        let mut idx = InMemoryVectorIndex::new(4);
        let doc_id = Uuid::new_v4();

        idx.insert(make_chunk(doc_id, "alpha", 0), vec![1.0, 0.0, 0.0, 0.0]);
        idx.insert(make_chunk(doc_id, "beta", 1), vec![0.0, 1.0, 0.0, 0.0]);
        idx.insert(make_chunk(doc_id, "gamma", 2), vec![0.0, 0.0, 1.0, 0.0]);

        let query = [0.8, 0.2, 0.0, 0.0];
        let results_before = idx.search(&query, 3);

        let path = std::env::temp_dir().join("cwc_save_load_search_test.bin");
        idx.save(&path).unwrap();
        let loaded = InMemoryVectorIndex::load(&path).unwrap();
        let results_after = loaded.search(&query, 3);

        assert_eq!(results_before.len(), results_after.len());
        for (before, after) in results_before.iter().zip(results_after.iter()) {
            assert_eq!(before.0.chunk_id, after.0.chunk_id);
            assert!((before.1 - after.1).abs() < 1e-6);
        }

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_memory_load_corrupted_file() {
        let path = std::env::temp_dir().join("cwc_corrupted_index.bin");
        std::fs::write(&path, b"not valid bincode data").unwrap();

        let result = InMemoryVectorIndex::load(&path);
        assert!(result.is_err());

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_memory_load_nonexistent_file() {
        let path = std::env::temp_dir().join("cwc_nonexistent_index_12345.bin");
        let result = InMemoryVectorIndex::load(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_cosine_similarity_different_lengths() {
        // zip truncates to shorter length — verify behavior
        let sim = cosine_similarity(&[1.0, 0.0, 0.0], &[1.0, 0.0]);
        // Only first 2 elements compared: [1,0] · [1,0] / (1*1) = 1.0
        // But norm_a uses all 3 elements of a via zip with b (only 2)
        // Actually zip stops at shorter: so it's [1,0]·[1,0] / sqrt(1)*sqrt(1) = 1.0
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_memory_cosine_empty_vectors() {
        let sim = cosine_similarity(&[], &[]);
        // dot=0, norm_a=0, norm_b=0 → returns 0.0
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_memory_empty_filtered_search() {
        let idx = InMemoryVectorIndex::new(4);
        let results = idx.search_filtered(&[1.0, 0.0, 0.0, 0.0], &[Uuid::new_v4()], 10);
        assert!(results.is_empty());
    }
}
