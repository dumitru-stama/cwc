use std::path::Path;

use cwc_core::traits::TokenCounter;
use serde::{Deserialize, Serialize};

use crate::error::{Result, SessionError};

use super::types::{FactPriority, MemoryFact};

/// In-memory store for session facts, with persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMemoryStore {
    facts: Vec<MemoryFact>,
    #[serde(default = "default_max_bytes")]
    max_bytes: usize,
    #[serde(default = "default_max_entries")]
    max_entries: usize,
}

fn default_max_bytes() -> usize {
    2048
}

fn default_max_entries() -> usize {
    30
}

impl SessionMemoryStore {
    pub fn new(max_bytes: usize, max_entries: usize) -> Self {
        Self {
            facts: Vec::new(),
            max_bytes,
            max_entries,
        }
    }

    /// Upsert a fact. If key exists, overwrite value and timestamp.
    pub fn upsert(&mut self, fact: MemoryFact) {
        if let Some(existing) = self.facts.iter_mut().find(|f| f.key == fact.key) {
            existing.value = fact.value;
            existing.created_at = fact.created_at;
            existing.priority = fact.priority;
            existing.source = fact.source;
        } else {
            self.facts.push(fact);
        }
    }

    /// Upsert multiple facts.
    pub fn upsert_batch(&mut self, facts: Vec<MemoryFact>) {
        for fact in facts {
            self.upsert(fact);
        }
    }

    /// Get a fact by key.
    pub fn get(&self, key: &str) -> Option<&MemoryFact> {
        self.facts.iter().find(|f| f.key == key)
    }

    /// Remove a fact by key. Returns true if removed.
    pub fn remove(&mut self, key: &str) -> bool {
        let len_before = self.facts.len();
        self.facts.retain(|f| f.key != key);
        self.facts.len() < len_before
    }

    /// Remove multiple facts by key. Returns the number of facts removed.
    pub fn remove_batch(&mut self, keys: &[String]) -> usize {
        let key_set: std::collections::HashSet<&str> =
            keys.iter().map(|k| k.as_str()).collect();
        let len_before = self.facts.len();
        self.facts.retain(|f| !key_set.contains(f.key.as_str()));
        len_before - self.facts.len()
    }

    /// Enforce limits: if over max_entries or max_bytes, evict lowest-priority
    /// oldest entries (but never evict Goal priority).
    pub fn enforce_limits(&mut self) {
        // Sort for eviction: lowest priority first, then oldest first
        // We'll build an eviction order and remove from the end
        while self.needs_eviction() {
            // Find the lowest-priority, oldest non-Goal fact
            let evict_idx = self
                .facts
                .iter()
                .enumerate()
                .filter(|(_, f)| f.priority != FactPriority::Goal)
                .min_by(|(_, a), (_, b)| {
                    a.priority
                        .cmp(&b.priority)
                        .then(a.created_at.cmp(&b.created_at))
                })
                .map(|(i, _)| i);

            match evict_idx {
                Some(idx) => {
                    self.facts.remove(idx);
                }
                None => break, // Only Goal facts remain, can't evict
            }
        }
    }

    /// All facts in insertion order. Use `render()` for priority-sorted output.
    pub fn all(&self) -> &[MemoryFact] {
        &self.facts
    }

    /// Return facts sorted for rendering: priority desc, then recency desc.
    fn sorted_for_render(&self) -> Vec<&MemoryFact> {
        let mut sorted: Vec<&MemoryFact> = self.facts.iter().collect();
        sorted.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then(b.created_at.cmp(&a.created_at))
        });
        sorted
    }

    /// Render as text for injection into the conversation.
    pub fn render(&self, max_tokens: u32, tokenizer: &dyn TokenCounter) -> String {
        if self.facts.is_empty() {
            return String::new();
        }

        let sorted = self.sorted_for_render();
        let mut result = String::from("[WORKING_MEMORY]\n");
        let header_tokens = tokenizer.count_tokens(&result);
        let mut used_tokens = header_tokens;

        for fact in &sorted {
            let line = if fact.priority == FactPriority::Goal {
                format!("Goal: {}\n", fact.value)
            } else {
                format!("{} — {}\n", fact.key, fact.value)
            };

            let line_tokens = tokenizer.count_tokens(&line);
            if used_tokens + line_tokens > max_tokens {
                break;
            }
            result.push_str(&line);
            used_tokens += line_tokens;
        }

        // Remove trailing newline
        if result.ends_with('\n') {
            result.pop();
        }
        result
    }

    /// Serialize to JSON for persistence.
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load from JSON.
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let store: Self = serde_json::from_str(&data).map_err(|e| {
            SessionError::InvalidFormat(format!("memory store: {e}"))
        })?;
        Ok(store)
    }

    /// Current entry count.
    pub fn len(&self) -> usize {
        self.facts.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }

    /// Current rendered byte size (approximate).
    /// Regular facts: "{key} — {value}\n" → key.len() + value.len() + 6 bytes overhead.
    /// Goal facts: "Goal: {value}\n" → value.len() + 7 bytes overhead.
    pub fn rendered_bytes(&self) -> usize {
        self.facts
            .iter()
            .map(|f| {
                if f.priority == FactPriority::Goal {
                    // "Goal: " (6) + value + "\n" (1)
                    f.value.len() + 7
                } else {
                    // key + " — " (5) + value + "\n" (1)
                    f.key.len() + f.value.len() + 6
                }
            })
            .sum()
    }

    fn needs_eviction(&self) -> bool {
        self.facts.len() > self.max_entries || self.rendered_bytes() > self.max_bytes
    }
}

impl Default for SessionMemoryStore {
    fn default() -> Self {
        Self::new(default_max_bytes(), default_max_entries())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::types::{MemorySource, now_millis};

    struct WordCounter;
    impl TokenCounter for WordCounter {
        fn count_tokens(&self, text: &str) -> u32 {
            text.split_whitespace().count() as u32
        }
        fn truncate_to_tokens(&self, text: &str, max: u32) -> String {
            text.split_whitespace()
                .take(max as usize)
                .collect::<Vec<_>>()
                .join(" ")
        }
    }
    fn tc() -> &'static dyn TokenCounter {
        &WordCounter
    }

    fn make_fact(key: &str, value: &str, priority: FactPriority) -> MemoryFact {
        MemoryFact {
            key: key.to_string(),
            value: value.to_string(),
            source: MemorySource::ToolResult {
                tool_name: "test".into(),
                call_id: "c1".into(),
            },
            created_at: now_millis(),
            priority,
        }
    }

    fn make_fact_at(key: &str, value: &str, priority: FactPriority, ts: u64) -> MemoryFact {
        MemoryFact {
            key: key.to_string(),
            value: value.to_string(),
            source: MemorySource::ToolResult {
                tool_name: "test".into(),
                call_id: "c1".into(),
            },
            created_at: ts,
            priority,
        }
    }

    #[test]
    fn test_store_upsert_new() {
        let mut store = SessionMemoryStore::default();
        store.upsert(make_fact("build_result", "success", FactPriority::Status));
        assert_eq!(store.len(), 1);
        assert_eq!(store.get("build_result").unwrap().value, "success");
    }

    #[test]
    fn test_store_upsert_overwrite() {
        let mut store = SessionMemoryStore::default();
        store.upsert(make_fact_at("build_result", "error: E0308", FactPriority::Status, 100));
        store.upsert(make_fact_at("build_result", "success", FactPriority::Status, 200));
        assert_eq!(store.len(), 1);
        assert_eq!(store.get("build_result").unwrap().value, "success");
        assert_eq!(store.get("build_result").unwrap().created_at, 200);
    }

    #[test]
    fn test_store_remove() {
        let mut store = SessionMemoryStore::default();
        store.upsert(make_fact("key1", "val1", FactPriority::Finding));
        assert!(store.remove("key1"));
        assert!(store.is_empty());
        assert!(!store.remove("key1")); // already removed
    }

    #[test]
    fn test_store_enforce_limits_max_entries() {
        let mut store = SessionMemoryStore::new(100000, 3);
        store.upsert(make_fact_at("a", "val", FactPriority::Navigation, 100));
        store.upsert(make_fact_at("b", "val", FactPriority::Finding, 200));
        store.upsert(make_fact_at("c", "val", FactPriority::Finding, 300));
        store.upsert(make_fact_at("d", "val", FactPriority::Finding, 400));
        assert_eq!(store.len(), 4);

        store.enforce_limits();
        assert_eq!(store.len(), 3);
        // Navigation "a" should be evicted first (lowest priority)
        assert!(store.get("a").is_none());
        assert!(store.get("b").is_some());
        assert!(store.get("c").is_some());
        assert!(store.get("d").is_some());
    }

    #[test]
    fn test_store_enforce_limits_evicts_navigation_before_finding() {
        let mut store = SessionMemoryStore::new(100000, 2);
        store.upsert(make_fact_at("nav", "seen file", FactPriority::Navigation, 300));
        store.upsert(make_fact_at("find", "important", FactPriority::Finding, 100));
        store.upsert(make_fact_at("find2", "also important", FactPriority::Finding, 200));

        store.enforce_limits();
        assert_eq!(store.len(), 2);
        // Navigation evicted despite being newer
        assert!(store.get("nav").is_none());
        assert!(store.get("find").is_some());
        assert!(store.get("find2").is_some());
    }

    #[test]
    fn test_store_enforce_limits_never_evicts_goal() {
        let mut store = SessionMemoryStore::new(100000, 2);
        store.upsert(make_fact_at("goal", "find bugs", FactPriority::Goal, 100));
        store.upsert(make_fact_at("a", "val", FactPriority::Finding, 200));
        store.upsert(make_fact_at("b", "val", FactPriority::Finding, 300));
        store.upsert(make_fact_at("c", "val", FactPriority::Finding, 400));

        store.enforce_limits();
        assert_eq!(store.len(), 2);
        assert!(store.get("goal").is_some(), "Goal must never be evicted");
    }

    #[test]
    fn test_store_render_goal_first() {
        let mut store = SessionMemoryStore::default();
        store.upsert(make_fact_at("finding:grep", "3 matches", FactPriority::Finding, 200));
        store.upsert(make_fact_at("goal", "find all bugs", FactPriority::Goal, 100));
        store.upsert(make_fact_at("build_result", "success", FactPriority::Status, 300));

        let rendered = store.render(1000, tc());
        assert!(rendered.starts_with("[WORKING_MEMORY]"));
        // Goal should appear before other entries
        let goal_pos = rendered.find("Goal:").unwrap();
        let finding_pos = rendered.find("finding:grep").unwrap();
        assert!(goal_pos < finding_pos);
    }

    #[test]
    fn test_store_render_respects_max_tokens() {
        let mut store = SessionMemoryStore::default();
        for i in 0..20 {
            store.upsert(make_fact_at(
                &format!("key_{i}"),
                &format!("value with several words for entry {i}"),
                FactPriority::Finding,
                i as u64,
            ));
        }

        let rendered = store.render(15, tc());
        let tokens = tc().count_tokens(&rendered);
        assert!(tokens <= 15, "rendered {tokens} tokens, expected <= 15");
    }

    #[test]
    fn test_store_render_empty() {
        let store = SessionMemoryStore::default();
        let rendered = store.render(1000, tc());
        assert!(rendered.is_empty());
    }

    #[test]
    fn test_store_render_priority_then_recency() {
        let mut store = SessionMemoryStore::default();
        store.upsert(make_fact_at("nav", "seen file", FactPriority::Navigation, 300));
        store.upsert(make_fact_at("find_old", "old finding", FactPriority::Finding, 100));
        store.upsert(make_fact_at("find_new", "new finding", FactPriority::Finding, 200));

        let rendered = store.render(1000, tc());
        // Findings before navigation, newer finding before older
        let find_new_pos = rendered.find("find_new").unwrap();
        let find_old_pos = rendered.find("find_old").unwrap();
        let nav_pos = rendered.find("nav").unwrap();
        assert!(find_new_pos < find_old_pos);
        assert!(find_old_pos < nav_pos);
    }

    #[test]
    fn test_store_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memory.json");

        let mut store = SessionMemoryStore::new(4096, 50);
        store.upsert(make_fact("goal", "find bugs", FactPriority::Goal));
        store.upsert(make_fact("build_result", "success", FactPriority::Status));
        store.upsert(make_fact("finding:grep_malloc", "3 matches", FactPriority::Finding));

        store.save(&path).unwrap();
        let loaded = SessionMemoryStore::load(&path).unwrap();

        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.get("goal").unwrap().value, "find bugs");
        assert_eq!(loaded.get("build_result").unwrap().value, "success");
        assert_eq!(loaded.get("finding:grep_malloc").unwrap().value, "3 matches");
        assert_eq!(loaded.max_bytes, 4096);
        assert_eq!(loaded.max_entries, 50);
    }

    #[test]
    fn test_store_upsert_batch() {
        let mut store = SessionMemoryStore::default();
        let facts = vec![
            make_fact("a", "val_a", FactPriority::Finding),
            make_fact("b", "val_b", FactPriority::Finding),
            make_fact("a", "updated_a", FactPriority::Finding), // overwrites first "a"
        ];
        store.upsert_batch(facts);
        assert_eq!(store.len(), 2);
        assert_eq!(store.get("a").unwrap().value, "updated_a");
    }

    #[test]
    fn test_store_enforce_limits_evicts_oldest_same_priority() {
        let mut store = SessionMemoryStore::new(100000, 2);
        store.upsert(make_fact_at("old", "val", FactPriority::Finding, 100));
        store.upsert(make_fact_at("mid", "val", FactPriority::Finding, 200));
        store.upsert(make_fact_at("new", "val", FactPriority::Finding, 300));

        store.enforce_limits();
        assert_eq!(store.len(), 2);
        // Oldest evicted
        assert!(store.get("old").is_none());
        assert!(store.get("mid").is_some());
        assert!(store.get("new").is_some());
    }

    #[test]
    fn test_store_load_nonexistent() {
        let result = SessionMemoryStore::load(Path::new("/nonexistent/memory.json"));
        assert!(result.is_err());
    }
}
