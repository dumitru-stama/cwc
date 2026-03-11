use uuid::Uuid;

use cwc_core::traits::Embedder;
use cwc_core::types::Chunk;

use crate::cache::{content_hash, EmbeddingCache};
use crate::error::{EmbedError, Result};

/// Embed all chunks, using cache where possible.
///
/// Returns `(chunk_id, embedding)` pairs for all chunks.
pub fn embed_chunks(
    chunks: &[Chunk],
    embedder: &dyn Embedder,
    cache: &EmbeddingCache,
    batch_size: usize,
) -> Result<Vec<(Uuid, Vec<f32>)>> {
    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    // Compute content hashes for all chunks
    let hashes: Vec<String> = chunks.iter().map(|c| content_hash(&c.text)).collect();
    let hash_refs: Vec<&str> = hashes.iter().map(|s| s.as_str()).collect();

    // Batch cache lookup
    let (found, missing) = cache.get_batch(&hash_refs);

    // Build result vector with placeholders
    let mut results: Vec<(Uuid, Vec<f32>)> = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        results.push((chunk.chunk_id, Vec::new()));
    }

    // Fill in cached entries
    for (idx, emb) in &found {
        results[*idx].1 = emb.clone();
    }

    // Embed cache misses in batches
    if !missing.is_empty() {
        let miss_texts: Vec<&str> = missing.iter().map(|&i| chunks[i].text.as_str()).collect();
        let batch_size = batch_size.max(1);

        let mut all_new_embs = Vec::with_capacity(miss_texts.len());

        for batch in miss_texts.chunks(batch_size) {
            let embs = embedder
                .embed(batch)
                .map_err(|e| EmbedError::Embed(e.to_string()))?;
            all_new_embs.extend(embs);
        }

        // Store new embeddings in cache and result
        for (miss_idx, new_emb) in missing.iter().zip(all_new_embs.into_iter()) {
            cache.put(&hashes[*miss_idx], &new_emb)?;
            results[*miss_idx].1 = new_emb;
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// A mock embedder for testing the pipeline without ONNX.
    struct MockEmbedder {
        dim: usize,
        call_count: Mutex<usize>,
    }

    impl MockEmbedder {
        fn new(dim: usize) -> Self {
            Self {
                dim,
                call_count: Mutex::new(0),
            }
        }

        fn calls(&self) -> usize {
            *self.call_count.lock().unwrap()
        }
    }

    impl Embedder for MockEmbedder {
        fn embed(&self, texts: &[&str]) -> cwc_core::Result<Vec<Vec<f32>>> {
            let mut count = self.call_count.lock().unwrap();
            *count += 1;
            Ok(texts
                .iter()
                .map(|t| {
                    // Deterministic: hash-based fake embedding
                    let hash = content_hash(t);
                    let seed = u32::from_le_bytes([
                        hash.as_bytes()[0],
                        hash.as_bytes()[1],
                        hash.as_bytes()[2],
                        hash.as_bytes()[3],
                    ]);
                    (0..self.dim)
                        .map(|i| ((seed as f32 + i as f32) * 0.001).sin())
                        .collect()
                })
                .collect())
        }

        fn dim(&self) -> usize {
            self.dim
        }
    }

    fn make_chunk(text: &str, idx: usize) -> Chunk {
        let doc_id = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"test-doc");
        Chunk {
            chunk_id: Uuid::new_v5(&Uuid::NAMESPACE_OID, format!("chunk-{idx}").as_bytes()),
            doc_id,
            doc_version: 1,
            source_path: "test.md".to_string(),
            section_path: vec!["Test".to_string()],
            char_offset: 0,
            char_len: text.len(),
            token_count: 10,
            text: text.to_string(),
            metadata: HashMap::new(),
        }
    }

    fn test_cache_dir(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("cwc_pipeline_{name}"))
    }

    #[test]
    fn test_embed_chunks_all_miss() {
        let dir = test_cache_dir("all_miss");
        let _ = std::fs::remove_dir_all(&dir);
        let cache = EmbeddingCache::new(&dir);
        let embedder = MockEmbedder::new(4);

        let chunks = vec![
            make_chunk("Hello world", 0),
            make_chunk("Rust programming", 1),
        ];

        let results = embed_chunks(&chunks, &embedder, &cache, 32).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, chunks[0].chunk_id);
        assert_eq!(results[1].0, chunks[1].chunk_id);
        assert_eq!(results[0].1.len(), 4);
        assert_eq!(results[1].1.len(), 4);
        assert!(embedder.calls() > 0);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_embed_chunks_cache_hit() {
        let dir = test_cache_dir("cache_hit");
        let _ = std::fs::remove_dir_all(&dir);
        let cache = EmbeddingCache::new(&dir);
        let embedder = MockEmbedder::new(4);

        let chunks = vec![make_chunk("Hello world", 0)];

        // First call: cache miss
        let _ = embed_chunks(&chunks, &embedder, &cache, 32).unwrap();
        let _calls_after_first = embedder.calls();

        // Second call: cache hit — embedder should NOT be called again
        let embedder2 = MockEmbedder::new(4);
        let results = embed_chunks(&chunks, &embedder2, &cache, 32).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1.len(), 4);
        assert_eq!(embedder2.calls(), 0, "embedder should not be called on cache hit");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_embed_chunks_mixed_cache() {
        let dir = test_cache_dir("mixed_cache");
        let _ = std::fs::remove_dir_all(&dir);
        let cache = EmbeddingCache::new(&dir);
        let embedder = MockEmbedder::new(4);

        // Cache one chunk
        let chunks_v1 = vec![make_chunk("Hello world", 0)];
        embed_chunks(&chunks_v1, &embedder, &cache, 32).unwrap();

        // Now embed two chunks: one cached, one new
        let embedder2 = MockEmbedder::new(4);
        let chunks_v2 = vec![
            make_chunk("Hello world", 0), // cached
            make_chunk("New text", 1),     // new
        ];
        let results = embed_chunks(&chunks_v2, &embedder2, &cache, 32).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].1.len(), 4);
        assert_eq!(results[1].1.len(), 4);
        assert_eq!(embedder2.calls(), 1, "only missing chunks should be embedded");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_embed_chunks_empty() {
        let dir = test_cache_dir("empty");
        let _ = std::fs::remove_dir_all(&dir);
        let cache = EmbeddingCache::new(&dir);
        let embedder = MockEmbedder::new(4);

        let results = embed_chunks(&[], &embedder, &cache, 32).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_embed_chunks_deterministic() {
        let dir = test_cache_dir("deterministic");
        let _ = std::fs::remove_dir_all(&dir);
        let cache = EmbeddingCache::new(&dir);
        let embedder = MockEmbedder::new(4);

        let chunks = vec![make_chunk("Same text", 0)];

        let r1 = embed_chunks(&chunks, &embedder, &cache, 32).unwrap();
        let r2 = embed_chunks(&chunks, &embedder, &cache, 32).unwrap();
        assert_eq!(r1[0].1, r2[0].1, "same text should give same embedding");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_embed_chunks_duplicate_texts_share_cache() {
        let dir = test_cache_dir("dup_texts");
        let _ = std::fs::remove_dir_all(&dir);
        let cache = EmbeddingCache::new(&dir);
        let embedder = MockEmbedder::new(4);

        // Two chunks with identical text but different chunk_ids
        let chunks = vec![
            make_chunk("Same text in both", 0),
            make_chunk("Same text in both", 1),
        ];

        let results = embed_chunks(&chunks, &embedder, &cache, 32).unwrap();
        assert_eq!(results.len(), 2);
        // Both should get the same embedding since content hash is identical
        assert_eq!(results[0].1, results[1].1);
        // But chunk_ids should differ
        assert_ne!(results[0].0, results[1].0);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_embed_chunks_preserves_order() {
        let dir = test_cache_dir("order");
        let _ = std::fs::remove_dir_all(&dir);
        let cache = EmbeddingCache::new(&dir);
        let embedder = MockEmbedder::new(4);

        let chunks = vec![
            make_chunk("Alpha", 0),
            make_chunk("Bravo", 1),
            make_chunk("Charlie", 2),
            make_chunk("Delta", 3),
        ];

        let results = embed_chunks(&chunks, &embedder, &cache, 2).unwrap();
        assert_eq!(results.len(), 4);
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(results[i].0, chunk.chunk_id, "chunk_id mismatch at index {i}");
            assert!(!results[i].1.is_empty(), "embedding empty at index {i}");
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_embed_chunks_small_batch_size() {
        let dir = test_cache_dir("small_batch");
        let _ = std::fs::remove_dir_all(&dir);
        let cache = EmbeddingCache::new(&dir);
        let embedder = MockEmbedder::new(4);

        // batch_size=1 means each text is embedded individually
        let chunks = vec![
            make_chunk("One", 0),
            make_chunk("Two", 1),
            make_chunk("Three", 2),
        ];

        let results = embed_chunks(&chunks, &embedder, &cache, 1).unwrap();
        assert_eq!(results.len(), 3);
        // Each batch of 1 is a separate embed() call
        assert_eq!(embedder.calls(), 3);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_embed_chunks_batch_size_zero_treated_as_one() {
        let dir = test_cache_dir("batch_zero");
        let _ = std::fs::remove_dir_all(&dir);
        let cache = EmbeddingCache::new(&dir);
        let embedder = MockEmbedder::new(4);

        let chunks = vec![make_chunk("Test", 0)];
        // batch_size=0 should be clamped to 1, not panic
        let results = embed_chunks(&chunks, &embedder, &cache, 0).unwrap();
        assert_eq!(results.len(), 1);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
