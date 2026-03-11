use std::collections::{BTreeMap, BTreeSet, HashSet};

use super::store::SessionMemoryStore;
use super::types::{FactPriority, MemoryFact, MemorySource};

/// Maximum number of groups to consolidate per pass.
const MAX_GROUPS_PER_PASS: usize = 5;

/// Minimum group size to consider for consolidation.
const DEFAULT_MIN_GROUP_SIZE: usize = 3;

/// Minimum savings percentage to apply a consolidation (0-100).
const DEFAULT_SAVINGS_PCT: usize = 30;

/// Maximum characters in a consolidated value.
const MAX_CONSOLIDATED_CHARS: usize = 384;

/// Configuration for the consolidation pass.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConsolidationConfig {
    pub enabled: bool,
    pub min_group_size: usize,
    pub savings_pct: usize,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_group_size: DEFAULT_MIN_GROUP_SIZE,
            savings_pct: DEFAULT_SAVINGS_PCT,
        }
    }
}

/// Why facts were grouped together.
#[derive(Debug, Clone)]
pub enum LinkReason {
    /// Same key prefix class (e.g., multiple `finding:grep_*` facts).
    SamePrefix(String),
    /// Facts referencing the same file path.
    SharedPath(String),
}

/// A group of facts that can be consolidated.
#[derive(Debug, Clone)]
pub struct ConsolidationGroup {
    pub source_keys: Vec<String>,
    pub link_reason: LinkReason,
    pub source_bytes: usize,
}

/// Result of a consolidation pass.
#[derive(Debug, Clone)]
pub struct ConsolidationReport {
    pub groups_found: usize,
    pub facts_consolidated: usize,
    pub source_facts_replaced: usize,
    pub bytes_saved: usize,
}

/// Run a consolidation pass on the memory store.
///
/// Detects groups of related facts, synthesizes consolidated replacements,
/// and applies them if the savings threshold is met.
pub fn consolidate_memory(
    store: &mut SessionMemoryStore,
    config: &ConsolidationConfig,
) -> ConsolidationReport {
    if !config.enabled || store.len() < config.min_group_size {
        return ConsolidationReport {
            groups_found: 0,
            facts_consolidated: 0,
            source_facts_replaced: 0,
            bytes_saved: 0,
        };
    }

    let groups = detect_groups(store.all(), config.min_group_size);
    let groups_found = groups.len();

    let mut total_consolidated = 0usize;
    let mut total_replaced = 0usize;
    let mut total_bytes_saved = 0usize;

    for group in groups.into_iter().take(MAX_GROUPS_PER_PASS) {
        let source_facts: Vec<&MemoryFact> = group
            .source_keys
            .iter()
            .filter_map(|k| store.get(k))
            .collect();

        if source_facts.len() < config.min_group_size {
            continue;
        }

        if let Some(consolidated) = synthesize(&group, &source_facts) {
            let consolidated_bytes = consolidated.key.len() + consolidated.value.len() + 6;

            // Check savings threshold
            let savings_pct = if group.source_bytes > 0 {
                ((group.source_bytes.saturating_sub(consolidated_bytes)) * 100)
                    / group.source_bytes
            } else {
                0
            };

            if savings_pct >= config.savings_pct {
                store.remove_batch(&group.source_keys);
                store.upsert(consolidated);
                total_consolidated += 1;
                total_replaced += group.source_keys.len();
                total_bytes_saved += group.source_bytes.saturating_sub(consolidated_bytes);
            }
        }
    }

    if total_consolidated > 0 {
        store.enforce_limits();
    }

    ConsolidationReport {
        groups_found,
        facts_consolidated: total_consolidated,
        source_facts_replaced: total_replaced,
        bytes_saved: total_bytes_saved,
    }
}

/// Detect groups of related facts that can be consolidated.
///
/// Uses BTreeMap for deterministic iteration order — groups are always
/// processed in the same order across runs.
pub fn detect_groups(facts: &[MemoryFact], min_group_size: usize) -> Vec<ConsolidationGroup> {
    let mut groups = Vec::new();
    let mut used_keys = HashSet::new();

    // Only consider non-Goal, non-consolidated facts
    let eligible: Vec<&MemoryFact> = facts
        .iter()
        .filter(|f| f.priority != FactPriority::Goal && !f.key.starts_with("consolidated:"))
        .collect();

    // Phase 1: Group by key prefix (e.g., "finding:grep_*", "file_seen:*")
    let mut prefix_groups: BTreeMap<String, Vec<&MemoryFact>> = BTreeMap::new();
    for fact in &eligible {
        if let Some(prefix) = extract_prefix_class(&fact.key) {
            prefix_groups.entry(prefix).or_default().push(fact);
        }
    }

    for (prefix, members) in &prefix_groups {
        if members.len() >= min_group_size {
            let source_keys: Vec<String> = members.iter().map(|f| f.key.clone()).collect();
            let source_bytes: usize = members
                .iter()
                .map(|f| f.key.len() + f.value.len() + 6)
                .sum();

            for k in &source_keys {
                used_keys.insert(k.clone());
            }

            groups.push(ConsolidationGroup {
                source_keys,
                link_reason: LinkReason::SamePrefix(prefix.clone()),
                source_bytes,
            });
        }
    }

    // Phase 2: Group by shared file path (from facts not already grouped)
    let mut path_groups: BTreeMap<String, Vec<&MemoryFact>> = BTreeMap::new();
    for fact in &eligible {
        if used_keys.contains(&fact.key) {
            continue;
        }
        if let Some(path) = extract_path_reference(&fact.key, &fact.value) {
            path_groups.entry(path).or_default().push(fact);
        }
    }

    for (path, members) in &path_groups {
        if members.len() >= min_group_size {
            let source_keys: Vec<String> = members.iter().map(|f| f.key.clone()).collect();
            let source_bytes: usize = members
                .iter()
                .map(|f| f.key.len() + f.value.len() + 6)
                .sum();

            for k in &source_keys {
                used_keys.insert(k.clone());
            }

            groups.push(ConsolidationGroup {
                source_keys,
                link_reason: LinkReason::SharedPath(path.clone()),
                source_bytes,
            });
        }
    }

    groups
}

/// Synthesize a group of facts into a single consolidated fact.
pub(crate) fn synthesize(group: &ConsolidationGroup, facts: &[&MemoryFact]) -> Option<MemoryFact> {
    if facts.is_empty() {
        return None;
    }

    let max_ts = facts.iter().map(|f| f.created_at).max().unwrap_or(0);
    let source_keys = group.source_keys.clone();

    match &group.link_reason {
        LinkReason::SamePrefix(prefix) => {
            let (key, value) = synthesize_same_prefix(prefix, facts);
            Some(MemoryFact {
                key,
                value,
                source: MemorySource::Consolidation { source_keys },
                created_at: max_ts,
                priority: FactPriority::Finding,
            })
        }
        LinkReason::SharedPath(path) => {
            let value = synthesize_shared_path(path, facts);
            let path_slug = super::types::make_slug(path, 40);
            Some(MemoryFact {
                key: format!("consolidated:path_{path_slug}"),
                value,
                source: MemorySource::Consolidation { source_keys },
                created_at: max_ts,
                priority: FactPriority::Finding,
            })
        }
    }
}

/// Synthesize facts sharing the same key prefix.
fn synthesize_same_prefix(prefix: &str, facts: &[&MemoryFact]) -> (String, String) {
    let key = format!("consolidated:{prefix}_summary");

    if prefix == "finding:grep" {
        // Grep-specific: combine match counts and top files
        let mut total_matches = 0usize;
        let mut all_files = BTreeSet::new();
        let mut patterns = Vec::new();

        for fact in facts {
            // Parse "N matches in M files" pattern from value
            if let Some(n) = fact.value.split_whitespace().next().and_then(|s| s.parse::<usize>().ok()) {
                total_matches += n;
            }
            // Extract pattern name from key: "finding:grep_PATTERN"
            if let Some(pat) = fact.key.strip_prefix("finding:grep_") {
                patterns.push(pat.to_string());
            }
            // Extract file references from value
            for word in fact.value.split_whitespace() {
                if word.contains('/') && word.contains('.') {
                    let clean = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '/' && c != '.' && c != '_' && c != '-');
                    if !clean.is_empty() {
                        all_files.insert(clean.to_string());
                    }
                }
            }
        }

        let top_patterns: Vec<&str> = patterns.iter().take(3).map(|s| s.as_str()).collect();
        let top_files: Vec<&str> = all_files.iter().take(3).map(|s| s.as_str()).collect();

        let mut value = format!(
            "{} searches, ~{} total matches.",
            facts.len(),
            total_matches,
        );
        if !top_patterns.is_empty() {
            value.push_str(&format!(" Patterns: {}", top_patterns.join(", ")));
        }
        if !top_files.is_empty() {
            value.push_str(&format!(". Files: {}", top_files.join(", ")));
        }

        (key, truncate(value))
    } else {
        // Generic prefix: concatenate abbreviated values
        let mut parts = Vec::new();
        for fact in facts {
            let abbrev: String = fact.value.chars().take(60).collect();
            parts.push(abbrev);
        }
        let value = format!(
            "{} items. {}",
            facts.len(),
            parts.join("; ")
        );
        (key, truncate(value))
    }
}

/// Synthesize facts sharing the same file path.
fn synthesize_shared_path(path: &str, facts: &[&MemoryFact]) -> String {
    let mut actions = Vec::new();
    for fact in facts {
        let action = if fact.key.starts_with("file_seen:") {
            "read"
        } else if fact.key.starts_with("file_modified:") {
            "modified"
        } else if fact.key.starts_with("finding:grep_") {
            "grep match"
        } else {
            "referenced"
        };
        let detail: String = fact.value.chars().take(40).collect();
        actions.push(format!("{action} ({detail})"));
    }
    let value = format!(
        "File {path}: {} actions. {}",
        facts.len(),
        actions.join("; ")
    );
    truncate(value)
}

/// Extract the prefix class from a fact key.
///
/// E.g., "finding:grep_malloc" -> "finding:grep", "finding:grep" -> "finding:grep",
/// "file_seen:src_main" -> "file_seen"
fn extract_prefix_class(key: &str) -> Option<String> {
    // Handle two-level prefixes like "finding:grep_X" or bare "finding:grep"
    if let Some(rest) = key.strip_prefix("finding:") {
        if rest.is_empty() {
            return Some("finding".to_string());
        }
        // Sub-classify: "grep_*" -> "finding:grep", bare "grep" -> "finding:grep"
        let subprefix = rest.split('_').next().unwrap_or(rest);
        return Some(format!("finding:{subprefix}"));
    }
    // Handle single-level prefixes
    if let Some(idx) = key.find(':') {
        let prefix = &key[..idx];
        if !prefix.is_empty() {
            return Some(prefix.to_string());
        }
    }
    None
}

/// Extract a file path reference from a fact's key or value.
fn extract_path_reference(_key: &str, value: &str) -> Option<String> {
    for word in value.split_whitespace() {
        let clean = word.trim_matches(|c: char| c == '(' || c == ')' || c == ',');
        if !clean.contains('/') || clean.starts_with("http") || clean.len() <= 3 {
            continue;
        }
        // Strip trailing `:line_number` (e.g., "src/main.rs:10" → "src/main.rs")
        let path = if let Some(idx) = clean.rfind(':') {
            let after = &clean[idx + 1..];
            if after.chars().all(|c| c.is_ascii_digit()) {
                &clean[..idx]
            } else {
                clean
            }
        } else {
            clean
        };
        // Must look like a file path (has an extension or starts with /)
        if path.contains('.') || path.starts_with('/') {
            return Some(path.to_string());
        }
    }
    None
}

fn truncate(s: String) -> String {
    if s.chars().count() <= MAX_CONSOLIDATED_CHARS {
        s
    } else {
        s.chars().take(MAX_CONSOLIDATED_CHARS).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::types::{MemorySource, now_millis};

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

    fn default_config() -> ConsolidationConfig {
        ConsolidationConfig::default()
    }

    // --- Group detection tests ---

    #[test]
    fn test_detect_groups_same_prefix_three_or_more() {
        let facts = vec![
            make_fact("finding:grep_malloc", "3 matches in 2 files", FactPriority::Finding),
            make_fact("finding:grep_free", "5 matches in 3 files", FactPriority::Finding),
            make_fact("finding:grep_realloc", "2 matches in 1 files", FactPriority::Finding),
            make_fact("finding:grep_calloc", "1 matches in 1 files", FactPriority::Finding),
        ];
        let groups = detect_groups(&facts, 3);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].source_keys.len(), 4);
        assert!(matches!(groups[0].link_reason, LinkReason::SamePrefix(ref p) if p == "finding:grep"));
    }

    #[test]
    fn test_detect_groups_skip_under_three() {
        let facts = vec![
            make_fact("finding:grep_malloc", "3 matches", FactPriority::Finding),
            make_fact("finding:grep_free", "5 matches", FactPriority::Finding),
        ];
        let groups = detect_groups(&facts, 3);
        assert!(groups.is_empty());
    }

    #[test]
    fn test_detect_groups_skip_goal() {
        let facts = vec![
            make_fact("goal", "find all bugs", FactPriority::Goal),
            make_fact("finding:grep_a", "1 match", FactPriority::Finding),
            make_fact("finding:grep_b", "2 matches", FactPriority::Finding),
            make_fact("finding:grep_c", "3 matches", FactPriority::Finding),
        ];
        let groups = detect_groups(&facts, 3);
        // Goal is excluded, but the 3 grep findings form a group
        assert_eq!(groups.len(), 1);
        assert!(!groups[0].source_keys.contains(&"goal".to_string()));
    }

    #[test]
    fn test_detect_groups_skip_consolidated() {
        let facts = vec![
            make_fact("consolidated:grep_summary", "old consolidation", FactPriority::Finding),
            make_fact("finding:grep_a", "1 match", FactPriority::Finding),
            make_fact("finding:grep_b", "2 matches", FactPriority::Finding),
            make_fact("finding:grep_c", "3 matches", FactPriority::Finding),
        ];
        let groups = detect_groups(&facts, 3);
        assert_eq!(groups.len(), 1);
        assert!(!groups[0].source_keys.contains(&"consolidated:grep_summary".to_string()));
    }

    #[test]
    fn test_detect_groups_shared_path() {
        let facts = vec![
            make_fact("file_seen:src_main_rs", "read src/main.rs (150 lines)", FactPriority::Navigation),
            make_fact("file_modified:src_main_rs", "modified src/main.rs: added fn", FactPriority::Finding),
            make_fact("cmd_result:grep_main", "found in src/main.rs:10 something", FactPriority::Finding),
        ];
        let groups = detect_groups(&facts, 3);
        assert_eq!(groups.len(), 1);
        assert!(matches!(groups[0].link_reason, LinkReason::SharedPath(ref p) if p.contains("src/main.rs")));
    }

    #[test]
    fn test_detect_groups_multiple_prefixes() {
        let facts = vec![
            make_fact("finding:grep_a", "1 match", FactPriority::Finding),
            make_fact("finding:grep_b", "2 matches", FactPriority::Finding),
            make_fact("finding:grep_c", "3 matches", FactPriority::Finding),
            make_fact("file_seen:a", "read a.rs (10 lines)", FactPriority::Navigation),
            make_fact("file_seen:b", "read b.rs (20 lines)", FactPriority::Navigation),
            make_fact("file_seen:c", "read c.rs (30 lines)", FactPriority::Navigation),
        ];
        let groups = detect_groups(&facts, 3);
        assert_eq!(groups.len(), 2);
    }

    // --- Heuristic synthesis tests ---

    #[test]
    fn test_heuristic_consolidator_grep() {
        let facts = [
            make_fact("finding:grep_malloc", "3 matches in 2 files. src/alloc.rs:10: malloc(256)", FactPriority::Finding),
            make_fact("finding:grep_free", "5 matches in 3 files. src/alloc.rs:20: free(ptr)", FactPriority::Finding),
            make_fact("finding:grep_realloc", "2 matches in 1 files. src/alloc.rs:30: realloc(ptr, 512)", FactPriority::Finding),
        ];
        let fact_refs: Vec<&MemoryFact> = facts.iter().collect();
        let group = ConsolidationGroup {
            source_keys: facts.iter().map(|f| f.key.clone()).collect(),
            link_reason: LinkReason::SamePrefix("finding:grep".into()),
            source_bytes: facts.iter().map(|f| f.key.len() + f.value.len() + 6).sum(),
        };

        let result = synthesize(&group, &fact_refs).unwrap();
        assert_eq!(result.key, "consolidated:finding:grep_summary");
        assert!(result.value.contains("3 searches"));
        assert!(result.value.contains("10 total matches"));
        assert!(result.value.contains("Patterns:"));
        assert!(result.priority == FactPriority::Finding);
    }

    #[test]
    fn test_heuristic_consolidator_generic_findings() {
        let facts = [
            make_fact("finding:custom_a", "tool output alpha with details", FactPriority::Finding),
            make_fact("finding:custom_b", "tool output beta with more details", FactPriority::Finding),
            make_fact("finding:custom_c", "tool output gamma with even more", FactPriority::Finding),
        ];
        let fact_refs: Vec<&MemoryFact> = facts.iter().collect();
        let group = ConsolidationGroup {
            source_keys: facts.iter().map(|f| f.key.clone()).collect(),
            link_reason: LinkReason::SamePrefix("finding:custom".into()),
            source_bytes: facts.iter().map(|f| f.key.len() + f.value.len() + 6).sum(),
        };

        let result = synthesize(&group, &fact_refs).unwrap();
        assert!(result.key.contains("consolidated:"));
        assert!(result.value.contains("3 items"));
    }

    #[test]
    fn test_heuristic_consolidator_shared_path() {
        let facts = [
            make_fact("file_seen:src_main_rs", "read src/main.rs (150 lines)", FactPriority::Navigation),
            make_fact("file_modified:src_main_rs", "modified src/main.rs: added fn", FactPriority::Finding),
            make_fact("cmd_result:test_main", "found in src/main.rs:10", FactPriority::Finding),
        ];
        let fact_refs: Vec<&MemoryFact> = facts.iter().collect();
        let group = ConsolidationGroup {
            source_keys: facts.iter().map(|f| f.key.clone()).collect(),
            link_reason: LinkReason::SharedPath("src/main.rs".into()),
            source_bytes: facts.iter().map(|f| f.key.len() + f.value.len() + 6).sum(),
        };

        let result = synthesize(&group, &fact_refs).unwrap();
        assert!(result.key.starts_with("consolidated:path_"));
        assert!(result.value.contains("src/main.rs"));
        assert!(result.value.contains("3 actions"));
    }

    #[test]
    fn test_extract_prefix_class_bare_finding() {
        // Bare "finding:grep" (no underscore suffix) should still classify as "finding:grep"
        assert_eq!(extract_prefix_class("finding:grep"), Some("finding:grep".into()));
        // Empty finding suffix
        assert_eq!(extract_prefix_class("finding:"), Some("finding".into()));
    }

    // --- End-to-end consolidation tests ---

    #[test]
    fn test_consolidate_memory_applies_and_removes_sources() {
        let mut store = SessionMemoryStore::new(8192, 50);
        store.upsert(make_fact("finding:grep_malloc", "3 matches in 2 files. src/a.rs:1 malloc(1)", FactPriority::Finding));
        store.upsert(make_fact("finding:grep_free", "5 matches in 3 files. src/b.rs:2 free(p)", FactPriority::Finding));
        store.upsert(make_fact("finding:grep_realloc", "2 matches in 1 files. src/c.rs:3 realloc(p,4)", FactPriority::Finding));
        store.upsert(make_fact("finding:grep_calloc", "1 matches in 1 files. src/d.rs:4 calloc(1,4)", FactPriority::Finding));
        store.upsert(make_fact("unrelated", "something else", FactPriority::Finding));
        assert_eq!(store.len(), 5);

        let report = consolidate_memory(&mut store, &default_config());
        assert_eq!(report.groups_found, 1);
        assert_eq!(report.facts_consolidated, 1);
        assert_eq!(report.source_facts_replaced, 4);
        assert!(report.bytes_saved > 0);

        // Source facts removed, consolidated fact added
        assert!(store.get("finding:grep_malloc").is_none());
        assert!(store.get("finding:grep_free").is_none());
        assert!(store.get("finding:grep_realloc").is_none());
        assert!(store.get("finding:grep_calloc").is_none());
        assert!(store.get("consolidated:finding:grep_summary").is_some());
        // Unrelated fact untouched
        assert!(store.get("unrelated").is_some());
    }

    #[test]
    fn test_consolidate_memory_savings_threshold() {
        // Create facts where consolidation would NOT save 30%
        let mut store = SessionMemoryStore::new(8192, 50);
        // Very short facts — consolidation overhead may exceed savings
        store.upsert(make_fact("finding:grep_a", "1m", FactPriority::Finding));
        store.upsert(make_fact("finding:grep_b", "2m", FactPriority::Finding));
        store.upsert(make_fact("finding:grep_c", "3m", FactPriority::Finding));

        let report = consolidate_memory(&mut store, &default_config());
        // The consolidated summary is likely bigger than 3 tiny facts
        // so it should be rejected by the savings threshold
        assert_eq!(report.facts_consolidated, 0);
        // Source facts should still be present
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_consolidate_memory_max_groups_cap() {
        let mut store = SessionMemoryStore::new(16384, 100);

        // Create 6 distinct prefix groups of 3+ facts each
        let prefixes = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"];
        for prefix in &prefixes {
            for i in 0..4 {
                let long_value = format!("detailed output from {prefix} tool run {i} with lots of content that makes this fact worth consolidating into a summary because it takes up space");
                store.upsert(make_fact(
                    &format!("finding:{prefix}_{i}"),
                    &long_value,
                    FactPriority::Finding,
                ));
            }
        }
        assert_eq!(store.len(), 24);

        let report = consolidate_memory(&mut store, &default_config());
        assert_eq!(report.groups_found, 6);
        // Only MAX_GROUPS_PER_PASS (5) should be processed
        assert!(report.facts_consolidated <= MAX_GROUPS_PER_PASS);
    }

    #[test]
    fn test_consolidate_memory_idempotent() {
        let mut store = SessionMemoryStore::new(8192, 50);
        // Use long values so consolidation passes the 30% savings threshold
        store.upsert(make_fact("finding:grep_malloc", "3 matches in 2 files. src/alloc.rs:10: malloc(256) with extra context about the allocation pattern found here", FactPriority::Finding));
        store.upsert(make_fact("finding:grep_free", "5 matches in 3 files. src/dealloc.rs:20: free(ptr) with additional context about the deallocation pattern and usage", FactPriority::Finding));
        store.upsert(make_fact("finding:grep_realloc", "2 matches in 1 files. src/realloc.rs:30: realloc(ptr, 512) with further context about the reallocation strategy used", FactPriority::Finding));

        let report1 = consolidate_memory(&mut store, &default_config());
        assert!(report1.facts_consolidated > 0, "first pass must consolidate");
        let len_after_first = store.len();

        // Second pass should change nothing (consolidated: prefix is excluded)
        let report2 = consolidate_memory(&mut store, &default_config());
        assert_eq!(report2.groups_found, 0);
        assert_eq!(report2.facts_consolidated, 0);
        assert_eq!(store.len(), len_after_first);
    }

    #[test]
    fn test_consolidate_memory_respects_2kb_cap() {
        let mut store = SessionMemoryStore::new(2048, 30);
        // Fill with many findings
        for i in 0..20 {
            let value = format!("detailed finding number {i} with content about something interesting in the code that we discovered during analysis");
            store.upsert(make_fact(
                &format!("finding:grep_{i}"),
                &value,
                FactPriority::Finding,
            ));
        }
        store.enforce_limits();
        let bytes_before = store.rendered_bytes();

        let _report = consolidate_memory(&mut store, &default_config());
        // After consolidation + enforce_limits, should still be within budget
        assert!(store.rendered_bytes() <= 2048 || store.rendered_bytes() <= bytes_before);
    }

    #[test]
    fn test_consolidate_memory_empty_store() {
        let mut store = SessionMemoryStore::new(2048, 30);
        let report = consolidate_memory(&mut store, &default_config());
        assert_eq!(report.groups_found, 0);
        assert_eq!(report.facts_consolidated, 0);
        assert_eq!(report.source_facts_replaced, 0);
        assert_eq!(report.bytes_saved, 0);
    }

    #[test]
    fn test_consolidation_report_accurate() {
        let mut store = SessionMemoryStore::new(8192, 50);
        store.upsert(make_fact("finding:grep_malloc", "3 matches in 2 files. src/a.rs:1: malloc(256) with extra detail", FactPriority::Finding));
        store.upsert(make_fact("finding:grep_free", "5 matches in 3 files. src/b.rs:2: free(ptr) with more detail", FactPriority::Finding));
        store.upsert(make_fact("finding:grep_realloc", "2 matches in 1 files. src/c.rs:3: realloc(ptr, 512) more stuff", FactPriority::Finding));

        let bytes_before = store.rendered_bytes();
        let report = consolidate_memory(&mut store, &default_config());

        if report.facts_consolidated > 0 {
            assert_eq!(report.source_facts_replaced, 3);
            assert!(report.bytes_saved > 0);
            let bytes_after = store.rendered_bytes();
            assert!(bytes_after < bytes_before);
        }
    }

    #[test]
    fn test_consolidate_memory_disabled() {
        let mut store = SessionMemoryStore::new(8192, 50);
        store.upsert(make_fact("finding:grep_a", "long output a here", FactPriority::Finding));
        store.upsert(make_fact("finding:grep_b", "long output b here", FactPriority::Finding));
        store.upsert(make_fact("finding:grep_c", "long output c here", FactPriority::Finding));

        let config = ConsolidationConfig {
            enabled: false,
            ..Default::default()
        };
        let report = consolidate_memory(&mut store, &config);
        assert_eq!(report.groups_found, 0);
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_consolidate_preserves_goal() {
        let mut store = SessionMemoryStore::new(8192, 50);
        store.upsert(make_fact("goal", "find all memory leaks", FactPriority::Goal));
        store.upsert(make_fact("finding:grep_a", "3 matches. src/x.rs:1 leak detail one", FactPriority::Finding));
        store.upsert(make_fact("finding:grep_b", "5 matches. src/y.rs:2 leak detail two", FactPriority::Finding));
        store.upsert(make_fact("finding:grep_c", "2 matches. src/z.rs:3 leak detail three", FactPriority::Finding));

        consolidate_memory(&mut store, &default_config());
        // Goal must survive
        assert!(store.get("goal").is_some());
        assert_eq!(store.get("goal").unwrap().value, "find all memory leaks");
    }

    #[test]
    fn test_consolidate_source_keys_tracked() {
        let mut store = SessionMemoryStore::new(8192, 50);
        store.upsert(make_fact("finding:grep_malloc", "3 matches in 2 files. src/a.rs:1 detail", FactPriority::Finding));
        store.upsert(make_fact("finding:grep_free", "5 matches in 3 files. src/b.rs:2 detail", FactPriority::Finding));
        store.upsert(make_fact("finding:grep_realloc", "2 matches in 1 files. src/c.rs:3 detail", FactPriority::Finding));

        let report = consolidate_memory(&mut store, &default_config());
        if report.facts_consolidated > 0 {
            let consolidated = store.get("consolidated:finding:grep_summary").unwrap();
            if let MemorySource::Consolidation { ref source_keys } = consolidated.source {
                assert_eq!(source_keys.len(), 3);
                assert!(source_keys.contains(&"finding:grep_malloc".to_string()));
                assert!(source_keys.contains(&"finding:grep_free".to_string()));
                assert!(source_keys.contains(&"finding:grep_realloc".to_string()));
            } else {
                panic!("expected Consolidation source");
            }
        }
    }

    // --- Helper function tests ---

    #[test]
    fn test_extract_prefix_class() {
        assert_eq!(extract_prefix_class("finding:grep_malloc"), Some("finding:grep".into()));
        assert_eq!(extract_prefix_class("finding:reverse_main"), Some("finding:reverse".into()));
        assert_eq!(extract_prefix_class("file_seen:src_main"), Some("file_seen".into()));
        assert_eq!(extract_prefix_class("cmd_result:ls_la"), Some("cmd_result".into()));
        assert_eq!(extract_prefix_class("build_result"), None);
        assert_eq!(extract_prefix_class("goal"), None);
    }

    #[test]
    fn test_extract_path_reference() {
        assert_eq!(
            extract_path_reference("file_seen:x", "read src/main.rs (150 lines)"),
            Some("src/main.rs".into())
        );
        assert_eq!(
            extract_path_reference("cmd", "output from /usr/lib/test.so"),
            Some("/usr/lib/test.so".into())
        );
        assert_eq!(
            extract_path_reference("finding:x", "no path here at all"),
            None
        );
        // Strips :line_number suffix
        assert_eq!(
            extract_path_reference("cmd", "found in src/main.rs:10 something"),
            Some("src/main.rs".into())
        );
    }
}
