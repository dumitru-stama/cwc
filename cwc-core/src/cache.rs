use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use dashmap::DashMap;

/// A concurrent, TTL-based cache with LRU-ish eviction.
pub struct CacheLayer<K: Hash + Eq, V: Clone> {
    store: DashMap<K, CacheEntry<V>>,
    max_entries: usize,
    ttl: Duration,
    hits: AtomicU64,
    misses: AtomicU64,
}

/// A cached value with creation time and hit count.
#[derive(Clone)]
pub struct CacheEntry<V> {
    pub value: V,
    pub created_at: Instant,
    pub hits: u32,
}

/// Cache usage statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f32,
}

impl<K: Hash + Eq + Clone, V: Clone> CacheLayer<K, V> {
    pub fn new(max_entries: usize, ttl: Duration) -> Self {
        Self {
            store: DashMap::new(),
            max_entries,
            ttl,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Get a value from the cache. Returns `None` if absent or expired.
    pub fn get(&self, key: &K) -> Option<V> {
        if let Some(mut entry) = self.store.get_mut(key) {
            if entry.created_at.elapsed() > self.ttl {
                drop(entry);
                self.store.remove(key);
                self.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            entry.hits += 1;
            let value = entry.value.clone();
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(value)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Insert a value. Evicts expired/oldest entries only when at capacity.
    ///
    /// Note: The capacity check and insert are not atomic, so under concurrent
    /// writes the cache may temporarily exceed `max_entries` by the number of
    /// racing writers. This is acceptable for a best-effort cache.
    pub fn put(&self, key: K, value: V) {
        // Only evict when at capacity AND this is a new key (not an overwrite)
        if self.store.len() >= self.max_entries && !self.store.contains_key(&key) {
            self.evict_expired();
            // If still at capacity after expiring, evict oldest
            if self.store.len() >= self.max_entries {
                self.evict_oldest();
            }
        }

        self.store.insert(
            key,
            CacheEntry {
                value,
                created_at: Instant::now(),
                hits: 0,
            },
        );
    }

    /// Remove a specific key.
    pub fn invalidate(&self, key: &K) {
        self.store.remove(key);
    }

    /// Remove all entries.
    pub fn clear(&self) {
        self.store.clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        CacheStats {
            entries: self.store.len(),
            hits,
            misses,
            hit_rate: if total > 0 {
                hits as f32 / total as f32
            } else {
                0.0
            },
        }
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Remove all entries whose TTL has expired.
    fn evict_expired(&self) {
        let ttl = self.ttl;
        self.store.retain(|_, v| v.created_at.elapsed() <= ttl);
    }

    /// Remove the entry with the oldest creation time.
    fn evict_oldest(&self) {
        let mut oldest_key: Option<K> = None;
        let mut oldest_time = Instant::now();

        for entry in self.store.iter() {
            if entry.value().created_at < oldest_time {
                oldest_time = entry.value().created_at;
                oldest_key = Some(entry.key().clone());
            }
        }

        if let Some(key) = oldest_key {
            self.store.remove(&key);
        }
    }
}

// --- Specific cache types ---

/// Cache for retrieval results, keyed by normalized query + filter hash.
pub struct RetrievalCache {
    cache: CacheLayer<String, Vec<crate::types::RetrievalHit>>,
}

impl RetrievalCache {
    /// Create with default TTL of 5 minutes and 1000 entries.
    pub fn new() -> Self {
        Self {
            cache: CacheLayer::new(1000, Duration::from_secs(300)),
        }
    }

    pub fn with_config(max_entries: usize, ttl: Duration) -> Self {
        Self {
            cache: CacheLayer::new(max_entries, ttl),
        }
    }

    /// Build a normalized cache key from query and optional filter UUIDs.
    pub fn cache_key(query: &str, filters: &[uuid::Uuid]) -> String {
        let normalized = query.trim().to_lowercase();
        // Collapse whitespace
        let normalized: String = normalized.split_whitespace().collect::<Vec<_>>().join(" ");
        if filters.is_empty() {
            normalized
        } else {
            let mut sorted: Vec<String> = filters.iter().map(|u| u.to_string()).collect();
            sorted.sort();
            format!("{normalized}|{}", sorted.join(","))
        }
    }

    pub fn get(&self, query: &str, filters: &[uuid::Uuid]) -> Option<Vec<crate::types::RetrievalHit>> {
        let key = Self::cache_key(query, filters);
        self.cache.get(&key)
    }

    pub fn put(&self, query: &str, filters: &[uuid::Uuid], hits: Vec<crate::types::RetrievalHit>) {
        let key = Self::cache_key(query, filters);
        self.cache.put(key, hits);
    }

    pub fn clear(&self) {
        self.cache.clear();
    }

    pub fn stats(&self) -> CacheStats {
        self.cache.stats()
    }
}

impl Default for RetrievalCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache for compiled prompts, keyed by query + state hash.
pub struct PromptCache {
    cache: CacheLayer<String, crate::types::CompiledContext>,
}

impl PromptCache {
    /// Create with default TTL of 1 minute and 100 entries.
    pub fn new() -> Self {
        Self {
            cache: CacheLayer::new(100, Duration::from_secs(60)),
        }
    }

    pub fn with_config(max_entries: usize, ttl: Duration) -> Self {
        Self {
            cache: CacheLayer::new(max_entries, ttl),
        }
    }

    pub fn get(&self, key: &str) -> Option<crate::types::CompiledContext> {
        self.cache.get(&key.to_string())
    }

    pub fn put(&self, key: String, ctx: crate::types::CompiledContext) {
        self.cache.put(key, ctx);
    }

    pub fn clear(&self) {
        self.cache.clear();
    }

    pub fn stats(&self) -> CacheStats {
        self.cache.stats()
    }
}

impl Default for PromptCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_cache_put_and_get() {
        let cache = CacheLayer::<String, String>::new(100, Duration::from_secs(60));
        cache.put("key1".to_string(), "value1".to_string());
        assert_eq!(cache.get(&"key1".to_string()), Some("value1".to_string()));
    }

    #[test]
    fn test_cache_ttl_expired() {
        let cache = CacheLayer::<String, String>::new(100, Duration::from_millis(50));
        cache.put("key1".to_string(), "value1".to_string());
        assert_eq!(cache.get(&"key1".to_string()), Some("value1".to_string()));
        thread::sleep(Duration::from_millis(60));
        assert_eq!(cache.get(&"key1".to_string()), None);
    }

    #[test]
    fn test_cache_max_entries_eviction() {
        let cache = CacheLayer::<u32, String>::new(3, Duration::from_secs(60));
        cache.put(1, "a".to_string());
        thread::sleep(Duration::from_millis(10));
        cache.put(2, "b".to_string());
        thread::sleep(Duration::from_millis(10));
        cache.put(3, "c".to_string());

        // At capacity, adding a 4th should evict the oldest (1)
        cache.put(4, "d".to_string());
        assert_eq!(cache.len(), 3);
        assert!(cache.get(&1).is_none(), "oldest entry should be evicted");
        assert_eq!(cache.get(&4), Some("d".to_string()));
    }

    #[test]
    fn test_cache_stats() {
        let cache = CacheLayer::<String, u32>::new(100, Duration::from_secs(60));
        cache.put("a".to_string(), 1);
        cache.put("b".to_string(), 2);

        // 2 hits
        cache.get(&"a".to_string());
        cache.get(&"b".to_string());
        // 1 miss
        cache.get(&"c".to_string());

        let stats = cache.stats();
        assert_eq!(stats.entries, 2);
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_cache_invalidate() {
        let cache = CacheLayer::<String, u32>::new(100, Duration::from_secs(60));
        cache.put("a".to_string(), 1);
        assert!(cache.get(&"a".to_string()).is_some());
        cache.invalidate(&"a".to_string());
        assert!(cache.get(&"a".to_string()).is_none());
    }

    #[test]
    fn test_cache_clear() {
        let cache = CacheLayer::<String, u32>::new(100, Duration::from_secs(60));
        cache.put("a".to_string(), 1);
        cache.put("b".to_string(), 2);
        cache.get(&"a".to_string()); // 1 hit
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert_eq!(cache.len(), 0);
        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_cache_empty_stats() {
        let cache = CacheLayer::<String, u32>::new(100, Duration::from_secs(60));
        let stats = cache.stats();
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.hit_rate, 0.0);
    }

    #[test]
    fn test_retrieval_cache_same_query_hit() {
        let rc = RetrievalCache::new();
        let hits = vec![crate::types::RetrievalHit {
            chunk: crate::types::Chunk {
                chunk_id: uuid::Uuid::new_v4(),
                doc_id: uuid::Uuid::new_v4(),
                doc_version: 1,
                source_path: "test.md".to_string(),
                section_path: vec![],
                char_offset: 0,
                char_len: 10,
                token_count: 5,
                text: "hello world".to_string(),
                metadata: std::collections::HashMap::new(),
            },
            score_sparse: 0.5,
            score_dense: 0.5,
            score_fused: 0.5,
            score_rerank: 0.0,
        }];
        rc.put("what is rust", &[], hits.clone());
        assert!(rc.get("what is rust", &[]).is_some());
        assert!(rc.get("something else", &[]).is_none());
    }

    #[test]
    fn test_retrieval_cache_normalized_key() {
        // Case-insensitive, trimmed, collapsed whitespace
        let k1 = RetrievalCache::cache_key("  What  is  Rust  ", &[]);
        let k2 = RetrievalCache::cache_key("what is rust", &[]);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_retrieval_cache_key_with_filters() {
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();
        let k1 = RetrievalCache::cache_key("query", &[id1, id2]);
        let k2 = RetrievalCache::cache_key("query", &[id2, id1]);
        assert_eq!(k1, k2, "filter order should not matter");
    }

    #[test]
    fn test_prompt_cache_put_get() {
        let pc = PromptCache::new();
        let ctx = crate::types::CompiledContext {
            instruction_block: "Answer the question.".to_string(),
            sources: vec![],
            output_schema: None,
            budget: crate::types::TokenBudget::from_fractions(4096, 0.15, 0.65, 0.10, 0.10),
        };
        pc.put("query_hash_123".to_string(), ctx.clone());
        let got = pc.get("query_hash_123").unwrap();
        assert_eq!(got.instruction_block, ctx.instruction_block);
    }

    #[test]
    fn test_cache_is_empty() {
        let cache = CacheLayer::<String, u32>::new(100, Duration::from_secs(60));
        assert!(cache.is_empty());
        cache.put("a".to_string(), 1);
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_cache_overwrite_same_key() {
        let cache = CacheLayer::<String, u32>::new(100, Duration::from_secs(60));
        cache.put("a".to_string(), 1);
        cache.put("a".to_string(), 2);
        assert_eq!(cache.get(&"a".to_string()), Some(2));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_overwrite_at_capacity_no_eviction() {
        // Bug fix: overwriting an existing key at max capacity should NOT evict another entry
        let cache = CacheLayer::<u32, String>::new(2, Duration::from_secs(60));
        cache.put(1, "a".to_string());
        cache.put(2, "b".to_string());
        assert_eq!(cache.len(), 2);

        // Overwrite key 2 — should NOT evict key 1
        cache.put(2, "b_updated".to_string());
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&1), Some("a".to_string()), "key 1 should survive overwrite of key 2");
        assert_eq!(cache.get(&2), Some("b_updated".to_string()));
    }

    #[test]
    fn test_cache_expired_entry_counts_as_miss() {
        let cache = CacheLayer::<String, u32>::new(100, Duration::from_millis(30));
        cache.put("a".to_string(), 1);
        assert!(cache.get(&"a".to_string()).is_some()); // hit
        thread::sleep(Duration::from_millis(40));
        assert!(cache.get(&"a".to_string()).is_none()); // miss (expired)

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_cache_zero_ttl_expires_immediately() {
        let cache = CacheLayer::<String, u32>::new(100, Duration::ZERO);
        cache.put("a".to_string(), 1);
        // With TTL=0, the entry is expired as soon as elapsed > 0
        thread::sleep(Duration::from_millis(1));
        assert!(cache.get(&"a".to_string()).is_none());
    }

    #[test]
    fn test_cache_max_entries_zero() {
        // max_entries=0: every put evicts the only entry, so at most 1 entry at a time
        let cache = CacheLayer::<u32, String>::new(0, Duration::from_secs(60));
        cache.put(1, "a".to_string());
        assert_eq!(cache.len(), 1);
        // Adding a second key evicts the first
        cache.put(2, "b".to_string());
        assert_eq!(cache.len(), 1);
        assert!(cache.get(&1).is_none());
        assert_eq!(cache.get(&2), Some("b".to_string()));
    }

    #[test]
    fn test_retrieval_cache_empty_query() {
        let key1 = RetrievalCache::cache_key("", &[]);
        let key2 = RetrievalCache::cache_key("  ", &[]);
        assert_eq!(key1, key2, "empty and whitespace-only queries should normalize the same");
        assert_eq!(key1, "");
    }

    #[test]
    fn test_cache_get_nonexistent_key() {
        let cache = CacheLayer::<String, u32>::new(100, Duration::from_secs(60));
        assert!(cache.get(&"nonexistent".to_string()).is_none());
        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);
    }
}
