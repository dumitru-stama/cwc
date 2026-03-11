use cwc_core::error::Result;

use crate::longterm::{MemoryCategory, MemoryEntry};

/// Trait for long-term memory storage backends.
pub trait MemoryStore: Send + Sync {
    fn upsert(&self, key: &str, value: &str, category: MemoryCategory) -> Result<()>;
    fn retrieve(&self, query: &str, max_entries: usize) -> Result<Vec<MemoryEntry>>;
    fn list_all(&self) -> Result<Vec<MemoryEntry>>;
    fn delete(&self, key: &str) -> Result<bool>;
    fn by_category(&self, category: MemoryCategory) -> Result<Vec<MemoryEntry>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file_memory::FileMemory;
    use std::sync::Arc;

    /// Test FileMemory through the MemoryStore trait interface.
    #[test]
    fn test_memory_store_trait_via_file_memory() {
        let path = std::env::temp_dir().join("cwc_test_store_trait.json");
        let _ = std::fs::remove_file(&path);

        let store: Arc<dyn MemoryStore> = Arc::new(FileMemory::new(&path));

        store
            .upsert("key1", "value1", MemoryCategory::ProjectFact)
            .unwrap();
        store
            .upsert("key2", "value2", MemoryCategory::UserPreference)
            .unwrap();

        let all = store.list_all().unwrap();
        assert_eq!(all.len(), 2);

        let facts = store.by_category(MemoryCategory::ProjectFact).unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].key, "key1");

        let results = store.retrieve("value1", 10).unwrap();
        assert!(!results.is_empty());

        assert!(store.delete("key1").unwrap());
        assert!(!store.delete("key1").unwrap());

        let remaining = store.list_all().unwrap();
        assert_eq!(remaining.len(), 1);

        let _ = std::fs::remove_file(&path);
    }

    /// Verify MemoryStore is object-safe (can be used as Arc<dyn MemoryStore>).
    #[test]
    fn test_memory_store_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Arc<dyn MemoryStore>>();
    }
}
