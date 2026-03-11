use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::error::Result;

/// Content-hash keyed embedding cache on disk.
///
/// Storage layout: `{cache_dir}/{hash[0..2]}/{hash}.bin` containing raw f32 bytes.
pub struct EmbeddingCache {
    cache_dir: PathBuf,
}

impl EmbeddingCache {
    pub fn new(cache_dir: &Path) -> Self {
        Self {
            cache_dir: cache_dir.to_path_buf(),
        }
    }

    /// Get cached embedding by content hash.
    pub fn get(&self, content_hash: &str) -> Option<Vec<f32>> {
        let path = self.hash_path(content_hash);
        let bytes = std::fs::read(&path).ok()?;
        if bytes.len() % 4 != 0 {
            return None;
        }
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Some(floats)
    }

    /// Store embedding keyed by content hash.
    pub fn put(&self, content_hash: &str, embedding: &[f32]) -> Result<()> {
        let path = self.hash_path(content_hash);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let bytes: Vec<u8> = embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        std::fs::write(&path, &bytes)?;
        Ok(())
    }

    /// Batch lookup — returns (found, missing_indices).
    pub fn get_batch(&self, hashes: &[&str]) -> (Vec<(usize, Vec<f32>)>, Vec<usize>) {
        let mut found = Vec::new();
        let mut missing = Vec::new();

        for (i, hash) in hashes.iter().enumerate() {
            match self.get(hash) {
                Some(emb) => found.push((i, emb)),
                None => missing.push(i),
            }
        }

        (found, missing)
    }

    fn hash_path(&self, content_hash: &str) -> PathBuf {
        let prefix = if content_hash.len() >= 2 {
            &content_hash[..2]
        } else {
            content_hash
        };
        self.cache_dir.join(prefix).join(format!("{content_hash}.bin"))
    }
}

/// Compute SHA-256 content hash of text.
pub fn content_hash(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cache_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("cwc_embed_cache_{name}"))
    }

    #[test]
    fn test_cache_put_and_get() {
        let dir = test_cache_dir("put_get");
        let _ = std::fs::remove_dir_all(&dir);
        let cache = EmbeddingCache::new(&dir);

        let hash = content_hash("hello world");
        let embedding = vec![1.0f32, 2.0, 3.0, 4.0];

        cache.put(&hash, &embedding).unwrap();
        let retrieved = cache.get(&hash).unwrap();
        assert_eq!(retrieved, embedding);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_cache_miss() {
        let dir = test_cache_dir("miss");
        let _ = std::fs::remove_dir_all(&dir);
        let cache = EmbeddingCache::new(&dir);

        let result = cache.get("nonexistent_hash");
        assert!(result.is_none());
    }

    #[test]
    fn test_cache_batch_mixed() {
        let dir = test_cache_dir("batch_mixed");
        let _ = std::fs::remove_dir_all(&dir);
        let cache = EmbeddingCache::new(&dir);

        let h1 = content_hash("text one");
        let h2 = content_hash("text two");
        let h3 = content_hash("text three");

        cache.put(&h1, &[1.0, 2.0]).unwrap();
        cache.put(&h3, &[5.0, 6.0]).unwrap();

        let (found, missing) = cache.get_batch(&[&h1, &h2, &h3]);
        assert_eq!(found.len(), 2);
        assert_eq!(missing, vec![1]); // h2 is missing

        // Check found entries
        assert_eq!(found[0].0, 0); // index 0 = h1
        assert_eq!(found[0].1, vec![1.0, 2.0]);
        assert_eq!(found[1].0, 2); // index 2 = h3
        assert_eq!(found[1].1, vec![5.0, 6.0]);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_cache_overwrite() {
        let dir = test_cache_dir("overwrite");
        let _ = std::fs::remove_dir_all(&dir);
        let cache = EmbeddingCache::new(&dir);

        let hash = content_hash("test");
        cache.put(&hash, &[1.0, 2.0]).unwrap();
        cache.put(&hash, &[3.0, 4.0]).unwrap();

        let retrieved = cache.get(&hash).unwrap();
        assert_eq!(retrieved, vec![3.0, 4.0]);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_content_hash_deterministic() {
        let h1 = content_hash("hello world");
        let h2 = content_hash("hello world");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_content_hash_different_texts() {
        let h1 = content_hash("hello");
        let h2 = content_hash("world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_content_hash_empty() {
        let h = content_hash("");
        assert!(!h.is_empty());
        assert_eq!(h.len(), 64); // SHA-256 hex = 64 chars
    }

    #[test]
    fn test_cache_empty_embedding() {
        let dir = test_cache_dir("empty_emb");
        let _ = std::fs::remove_dir_all(&dir);
        let cache = EmbeddingCache::new(&dir);

        let hash = content_hash("empty");
        cache.put(&hash, &[]).unwrap();
        let retrieved = cache.get(&hash).unwrap();
        assert!(retrieved.is_empty());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_cache_large_embedding() {
        let dir = test_cache_dir("large_emb");
        let _ = std::fs::remove_dir_all(&dir);
        let cache = EmbeddingCache::new(&dir);

        let hash = content_hash("large");
        let embedding: Vec<f32> = (0..384).map(|i| i as f32 * 0.001).collect();
        cache.put(&hash, &embedding).unwrap();
        let retrieved = cache.get(&hash).unwrap();
        assert_eq!(retrieved.len(), 384);
        assert_eq!(retrieved, embedding);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_cache_preserves_float_precision() {
        let dir = test_cache_dir("float_precision");
        let _ = std::fs::remove_dir_all(&dir);
        let cache = EmbeddingCache::new(&dir);

        let hash = content_hash("precision");
        let embedding = vec![
            f32::MIN,
            f32::MAX,
            f32::EPSILON,
            0.0,
            -0.0,
            1.234_567_9,
            std::f32::consts::PI,
        ];
        cache.put(&hash, &embedding).unwrap();
        let retrieved = cache.get(&hash).unwrap();
        // f32 roundtrip through le_bytes is exact
        assert_eq!(retrieved, embedding);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_cache_batch_all_missing() {
        let dir = test_cache_dir("batch_all_miss");
        let _ = std::fs::remove_dir_all(&dir);
        let cache = EmbeddingCache::new(&dir);

        let (found, missing) = cache.get_batch(&["a", "b", "c"]);
        assert!(found.is_empty());
        assert_eq!(missing, vec![0, 1, 2]);
    }

    #[test]
    fn test_cache_batch_all_found() {
        let dir = test_cache_dir("batch_all_found");
        let _ = std::fs::remove_dir_all(&dir);
        let cache = EmbeddingCache::new(&dir);

        cache.put("aa", &[1.0]).unwrap();
        cache.put("bb", &[2.0]).unwrap();

        let (found, missing) = cache.get_batch(&["aa", "bb"]);
        assert_eq!(found.len(), 2);
        assert!(missing.is_empty());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_cache_corrupted_file_odd_bytes() {
        let dir = test_cache_dir("corrupted");
        let _ = std::fs::remove_dir_all(&dir);
        let cache = EmbeddingCache::new(&dir);

        // Write a valid entry first to create the directory structure
        let hash = content_hash("corrupt");
        cache.put(&hash, &[1.0]).unwrap();

        // Now overwrite with odd-length bytes (not a multiple of 4)
        let path = cache.hash_path(&hash);
        std::fs::write(&path, [0u8, 1, 2]).unwrap();

        // get() should return None for corrupted files
        assert!(cache.get(&hash).is_none());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_content_hash_unicode() {
        let h1 = content_hash("日本語テスト");
        let h2 = content_hash("日本語テスト");
        let h3 = content_hash("中文测试");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
        assert_eq!(h1.len(), 64);
    }

    #[test]
    fn test_cache_batch_empty_input() {
        let dir = test_cache_dir("batch_empty");
        let _ = std::fs::remove_dir_all(&dir);
        let cache = EmbeddingCache::new(&dir);

        let (found, missing) = cache.get_batch(&[]);
        assert!(found.is_empty());
        assert!(missing.is_empty());
    }
}
