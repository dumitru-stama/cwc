use std::path::{Path, PathBuf};

use chrono::Utc;

use cwc_core::error::{CwcError, Result};

use crate::longterm::{MemoryCategory, MemoryEntry};

/// File-backed long-term memory (JSON file, no PostgreSQL needed).
pub struct FileMemory {
    path: PathBuf,
}

impl FileMemory {
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
        }
    }

    /// Load entries from disk, or return empty if file doesn't exist.
    fn load(&self) -> Result<Vec<MemoryEntry>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let data = std::fs::read_to_string(&self.path)?;
        if data.trim().is_empty() {
            return Ok(Vec::new());
        }
        let entries: Vec<MemoryEntry> =
            serde_json::from_str(&data).map_err(|e| CwcError::Config(format!("parse error: {e}")))?;
        Ok(entries)
    }

    /// Save entries to disk.
    fn save(&self, entries: &[MemoryEntry]) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string_pretty(entries)
            .map_err(|e| CwcError::Config(format!("serialize error: {e}")))?;
        std::fs::write(&self.path, data)?;
        Ok(())
    }

    /// Store or update a memory entry.
    pub fn upsert(&self, key: &str, value: &str, category: MemoryCategory) -> Result<()> {
        let mut entries = self.load()?;
        let now = Utc::now();

        if let Some(existing) = entries.iter_mut().find(|e| e.key == key) {
            existing.value = value.to_string();
            existing.category = category;
            existing.updated_at = now;
        } else {
            entries.push(MemoryEntry {
                key: key.to_string(),
                value: value.to_string(),
                category,
                created_at: now,
                updated_at: now,
                access_count: 0,
            });
        }

        self.save(&entries)
    }

    /// Retrieve memories relevant to a query using keyword matching.
    pub fn retrieve(&self, query: &str, max_entries: usize) -> Result<Vec<MemoryEntry>> {
        let mut entries = self.load()?;
        let lower_query = query.to_lowercase();
        let keywords: Vec<&str> = lower_query
            .split_whitespace()
            .filter(|w| w.len() >= 3)
            .collect();

        if keywords.is_empty() {
            // Return most recent
            entries.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
            entries.truncate(max_entries);
            return Ok(entries);
        }

        // Score each entry by number of matching keywords
        let mut scored: Vec<(usize, usize)> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let lower_key = e.key.to_lowercase();
                let lower_value = e.value.to_lowercase();
                let score = keywords
                    .iter()
                    .filter(|k| lower_key.contains(*k) || lower_value.contains(*k))
                    .count();
                (score, i)
            })
            .filter(|(score, _)| *score > 0)
            .collect();

        // Sort by score desc, then recency, then access count
        scored.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then(entries[b.1].updated_at.cmp(&entries[a.1].updated_at))
                .then(entries[b.1].access_count.cmp(&entries[a.1].access_count))
        });

        // Collect top results and increment access counts in-place
        let top_indices: Vec<usize> = scored.iter().take(max_entries).map(|(_, i)| *i).collect();
        for &idx in &top_indices {
            entries[idx].access_count += 1;
        }

        let results: Vec<MemoryEntry> = top_indices.iter().map(|&i| entries[i].clone()).collect();

        // Save back with updated access counts (single load, single save)
        self.save(&entries)?;

        Ok(results)
    }

    /// Get all memories in a category.
    pub fn by_category(&self, category: MemoryCategory) -> Result<Vec<MemoryEntry>> {
        let entries = self.load()?;
        Ok(entries
            .into_iter()
            .filter(|e| e.category == category)
            .collect())
    }

    /// Delete a memory entry. Returns true if it existed.
    pub fn delete(&self, key: &str) -> Result<bool> {
        let mut entries = self.load()?;
        let before = entries.len();
        entries.retain(|e| e.key != key);
        let removed = entries.len() < before;
        if removed {
            self.save(&entries)?;
        }
        Ok(removed)
    }

    /// List all entries.
    pub fn list_all(&self) -> Result<Vec<MemoryEntry>> {
        self.load()
    }
}

impl crate::store::MemoryStore for FileMemory {
    fn upsert(&self, key: &str, value: &str, category: MemoryCategory) -> Result<()> {
        self.upsert(key, value, category)
    }

    fn retrieve(&self, query: &str, max_entries: usize) -> Result<Vec<MemoryEntry>> {
        self.retrieve(query, max_entries)
    }

    fn list_all(&self) -> Result<Vec<MemoryEntry>> {
        self.list_all()
    }

    fn delete(&self, key: &str) -> Result<bool> {
        self.delete(key)
    }

    fn by_category(&self, category: MemoryCategory) -> Result<Vec<MemoryEntry>> {
        self.by_category(category)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_path() -> PathBuf {
        let dir = std::env::temp_dir().join("cwc_test_memory");
        let _ = fs::create_dir_all(&dir);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        dir.join(format!("mem_{}_{id}.json", std::process::id()))
    }

    fn cleanup(path: &Path) {
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_file_memory_save_load_roundtrip() {
        let path = temp_path();
        let mem = FileMemory::new(&path);

        mem.upsert("pref:units", "metric", MemoryCategory::UserPreference)
            .unwrap();
        mem.upsert("fact:lang", "uses Rust", MemoryCategory::ProjectFact)
            .unwrap();

        let entries = mem.list_all().unwrap();
        assert_eq!(entries.len(), 2);

        // Load in a fresh instance
        let mem2 = FileMemory::new(&path);
        let entries2 = mem2.list_all().unwrap();
        assert_eq!(entries2.len(), 2);

        cleanup(&path);
    }

    #[test]
    fn test_file_memory_search_keyword() {
        let path = temp_path();
        let mem = FileMemory::new(&path);

        mem.upsert("pref:output", "JSON format", MemoryCategory::UserPreference)
            .unwrap();
        mem.upsert("fact:db", "uses PostgreSQL", MemoryCategory::ProjectFact)
            .unwrap();
        mem.upsert("pref:units", "metric units", MemoryCategory::UserPreference)
            .unwrap();

        let results = mem.retrieve("output format", 10).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|e| e.key == "pref:output"));

        cleanup(&path);
    }

    #[test]
    fn test_file_memory_upsert_updates() {
        let path = temp_path();
        let mem = FileMemory::new(&path);

        mem.upsert("test:key", "old", MemoryCategory::ProjectFact)
            .unwrap();
        mem.upsert("test:key", "new", MemoryCategory::ProjectFact)
            .unwrap();

        let entries = mem.list_all().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].value, "new");

        cleanup(&path);
    }

    #[test]
    fn test_file_memory_delete() {
        let path = temp_path();
        let mem = FileMemory::new(&path);

        mem.upsert("del:test", "val", MemoryCategory::Correction)
            .unwrap();
        assert!(mem.delete("del:test").unwrap());
        assert!(!mem.delete("del:test").unwrap());
        assert!(mem.list_all().unwrap().is_empty());

        cleanup(&path);
    }

    #[test]
    fn test_file_memory_by_category() {
        let path = temp_path();
        let mem = FileMemory::new(&path);

        mem.upsert("a", "val", MemoryCategory::UserPreference)
            .unwrap();
        mem.upsert("b", "val", MemoryCategory::Correction)
            .unwrap();
        mem.upsert("c", "val", MemoryCategory::UserPreference)
            .unwrap();

        let prefs = mem.by_category(MemoryCategory::UserPreference).unwrap();
        assert_eq!(prefs.len(), 2);
        assert!(prefs.iter().all(|e| e.category == MemoryCategory::UserPreference));

        cleanup(&path);
    }

    #[test]
    fn test_file_memory_empty_file() {
        let path = temp_path();
        let mem = FileMemory::new(&path);
        assert!(mem.list_all().unwrap().is_empty());
        assert!(mem.retrieve("anything", 10).unwrap().is_empty());
        // Don't need cleanup — file was never created
    }

    #[test]
    fn test_file_memory_retrieve_max_zero() {
        let path = temp_path();
        let mem = FileMemory::new(&path);

        mem.upsert("a", "searchable value", MemoryCategory::ProjectFact)
            .unwrap();

        let results = mem.retrieve("searchable", 0).unwrap();
        assert!(results.is_empty());

        cleanup(&path);
    }

    #[test]
    fn test_file_memory_upsert_changes_category() {
        let path = temp_path();
        let mem = FileMemory::new(&path);

        mem.upsert("cat:test", "value", MemoryCategory::ProjectFact)
            .unwrap();
        mem.upsert("cat:test", "value", MemoryCategory::Correction)
            .unwrap();

        let entries = mem.list_all().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].category, MemoryCategory::Correction);

        cleanup(&path);
    }

    #[test]
    fn test_file_memory_retrieve_no_short_keywords() {
        let path = temp_path();
        let mem = FileMemory::new(&path);

        mem.upsert("a", "ab cd", MemoryCategory::ProjectFact).unwrap();
        mem.upsert("b", "xyz", MemoryCategory::ProjectFact).unwrap();

        // All keywords < 3 chars → returns most recent
        let results = mem.retrieve("ab cd", 10).unwrap();
        // "ab" and "cd" are < 3 chars, so keyword list is empty → returns by recency
        assert_eq!(results.len(), 2);

        cleanup(&path);
    }

    #[test]
    fn test_file_memory_access_count_increments() {
        let path = temp_path();
        let mem = FileMemory::new(&path);

        mem.upsert("search:test", "searchable value content", MemoryCategory::ProjectFact)
            .unwrap();

        let _ = mem.retrieve("searchable", 10).unwrap();
        let _ = mem.retrieve("searchable", 10).unwrap();

        let entries = mem.list_all().unwrap();
        let entry = entries.iter().find(|e| e.key == "search:test").unwrap();
        assert!(entry.access_count >= 2);

        cleanup(&path);
    }
}
