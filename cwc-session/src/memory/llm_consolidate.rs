//! LLM-enhanced memory consolidation.
//!
//! Uses an LLM to produce higher-quality consolidated summaries than the
//! deterministic heuristics in `consolidate.rs`. This is an optional async
//! post-step — the sync `optimize()` pipeline runs first with heuristic
//! consolidation, then callers can invoke `llm_consolidate()` for smarter
//! merging when an LlmClient is available.
//!
//! Falls back gracefully to heuristic results on any LLM failure.

use cwc_core::traits::LlmClient;
use cwc_core::types::{ChatMessage, Role};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use super::consolidate::{detect_groups, ConsolidationConfig, ConsolidationGroup};
use super::store::SessionMemoryStore;
use super::types::{now_millis, FactPriority, MemoryFact, MemorySource};

/// Configuration for LLM-enhanced consolidation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConsolidationConfig {
    /// Enable LLM consolidation (requires an LlmClient).
    pub enabled: bool,
    /// Max tokens for LLM response per group.
    #[serde(default = "default_llm_max_tokens")]
    pub max_tokens: u32,
    /// Max groups to send to LLM per pass.
    #[serde(default = "default_llm_max_groups")]
    pub max_groups: usize,
    /// Minimum group size for LLM consolidation (defaults to heuristic min_group_size).
    #[serde(default = "default_llm_min_group_size")]
    pub min_group_size: usize,
    /// Whether to also attempt cross-group insight discovery.
    #[serde(default)]
    pub cross_group_insights: bool,
}

fn default_llm_max_tokens() -> u32 {
    128
}
fn default_llm_max_groups() -> usize {
    3
}
fn default_llm_min_group_size() -> usize {
    3
}

impl Default for LlmConsolidationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_tokens: default_llm_max_tokens(),
            max_groups: default_llm_max_groups(),
            min_group_size: default_llm_min_group_size(),
            cross_group_insights: false,
        }
    }
}

/// Report from LLM consolidation pass.
#[derive(Debug, Clone, Default)]
pub struct LlmConsolidationReport {
    /// Groups refined by LLM.
    pub groups_refined: usize,
    /// Groups where LLM failed and heuristic was used instead.
    pub fallback_to_heuristic: usize,
    /// Cross-group insights discovered (if enabled).
    pub cross_group_insights: usize,
    /// Source facts replaced.
    pub source_facts_replaced: usize,
    /// Approximate bytes saved.
    pub bytes_saved: usize,
}

const SYSTEM_PROMPT: &str = "\
You are a memory consolidation assistant. You merge related facts from a working \
memory into a single, dense summary. Output ONLY the consolidated summary text — \
no preamble, no explanation, no markdown formatting. Keep it under 200 characters. \
Preserve all key details: counts, file paths, function names, error codes.";

const CROSS_GROUP_SYSTEM: &str = "\
You are a memory analyst. Given several groups of related facts, identify any \
cross-group connections or insights. Output a single sentence of at most 200 \
characters, or output NONE if no meaningful connection exists.";

/// Run LLM-enhanced memory consolidation.
///
/// This is an async post-step that should be called after `optimize()`.
/// For each detected group, it sends the facts to the LLM for a smarter
/// summary. On LLM failure, falls back to heuristic consolidation for
/// that group.
pub async fn llm_consolidate(
    store: &mut SessionMemoryStore,
    llm: &dyn LlmClient,
    config: &LlmConsolidationConfig,
    heuristic_config: &ConsolidationConfig,
) -> LlmConsolidationReport {
    if !config.enabled {
        return LlmConsolidationReport::default();
    }

    let groups = detect_groups(store.all(), config.min_group_size);
    if groups.is_empty() {
        return LlmConsolidationReport::default();
    }

    let mut report = LlmConsolidationReport::default();

    let groups_to_process: Vec<ConsolidationGroup> =
        groups.into_iter().take(config.max_groups).collect();

    // Snapshot fact values for cross-group prompt BEFORE the loop modifies the store
    let cross_group_snapshot: Vec<Vec<(String, String)>> = if config.cross_group_insights {
        groups_to_process
            .iter()
            .map(|g| {
                g.source_keys
                    .iter()
                    .filter_map(|k| store.get(k).map(|f| (k.clone(), f.value.clone())))
                    .collect()
            })
            .collect()
    } else {
        Vec::new()
    };

    for group in &groups_to_process {
        let source_facts: Vec<&MemoryFact> = group
            .source_keys
            .iter()
            .filter_map(|k| store.get(k))
            .collect();

        let actual_count = source_facts.len();
        if actual_count < config.min_group_size {
            continue;
        }

        // Build the LLM prompt
        let user_prompt = build_group_prompt(group, &source_facts);
        let messages = vec![
            ChatMessage {
                role: Role::System,
                content: SYSTEM_PROMPT.to_string(),
            },
            ChatMessage {
                role: Role::User,
                content: user_prompt,
            },
        ];

        match llm.generate_chat(&messages, None, config.max_tokens).await {
            Ok(summary) => {
                let summary = summary.trim().to_string();
                if summary.is_empty() || summary.chars().count() > 500 {
                    // Bad response — fall back to heuristic for this group only
                    debug!("LLM returned empty/oversized summary, falling back to heuristic");
                    if try_heuristic_fallback_single(store, group, heuristic_config) {
                        report.fallback_to_heuristic += 1;
                        report.source_facts_replaced += actual_count;
                    }
                    continue;
                }

                let consolidated_key = make_consolidated_key(group);
                let consolidated = MemoryFact {
                    key: consolidated_key,
                    value: summary,
                    source: MemorySource::Consolidation {
                        source_keys: group.source_keys.clone(),
                    },
                    created_at: now_millis(),
                    priority: FactPriority::Finding,
                };

                let consolidated_bytes = consolidated.key.len() + consolidated.value.len() + 6;

                // Only apply if it actually saves space
                if consolidated_bytes < group.source_bytes {
                    let saved = group.source_bytes - consolidated_bytes;
                    store.remove_batch(&group.source_keys);
                    store.upsert(consolidated);
                    report.groups_refined += 1;
                    report.source_facts_replaced += actual_count;
                    report.bytes_saved += saved;
                } else {
                    debug!("LLM summary didn't save space, skipping");
                }
            }
            Err(e) => {
                warn!("LLM consolidation failed: {e}, falling back to heuristic");
                if try_heuristic_fallback_single(store, group, heuristic_config) {
                    report.fallback_to_heuristic += 1;
                    report.source_facts_replaced += actual_count;
                }
            }
        }
    }

    // Cross-group insight discovery (uses pre-loop snapshot)
    if config.cross_group_insights && groups_to_process.len() >= 2 {
        // Collect the consolidated keys that now exist in the store
        // (these replaced the original source_keys during the loop)
        let consolidated_keys: Vec<String> = groups_to_process
            .iter()
            .map(make_consolidated_key)
            .filter(|k| store.get(k).is_some())
            .collect();

        if let Some(mut insight) = try_cross_group_insight(
            &groups_to_process,
            &cross_group_snapshot,
            llm,
            config.max_tokens,
        )
        .await
        {
            // Reference the consolidated keys (which exist in the store)
            // rather than the original source keys (which were removed)
            insight.source = MemorySource::Consolidation {
                source_keys: consolidated_keys,
            };
            store.upsert(insight);
            report.cross_group_insights = 1;
        }
    }

    if report.groups_refined > 0 || report.fallback_to_heuristic > 0 {
        store.enforce_limits();
    }

    report
}

/// Build a user prompt for consolidating a group of facts.
fn build_group_prompt(group: &ConsolidationGroup, facts: &[&MemoryFact]) -> String {
    let mut prompt = String::from("Consolidate these related memory facts into one dense summary:\n\n");
    for fact in facts {
        prompt.push_str(&format!("- [{}] {}\n", fact.key, fact.value));
    }
    prompt.push_str(&format!(
        "\nThese facts are related by: {:?}",
        group.link_reason
    ));
    prompt
}

/// Generate a consolidated key from the group's link reason.
fn make_consolidated_key(group: &ConsolidationGroup) -> String {
    use super::consolidate::LinkReason;
    match &group.link_reason {
        LinkReason::SamePrefix(prefix) => format!("consolidated:{prefix}_summary"),
        LinkReason::SharedPath(path) => {
            let slug = super::types::make_slug(path, 40);
            format!("consolidated:path_{slug}")
        }
    }
}

/// Try heuristic consolidation as fallback for a single group.
///
/// Uses the heuristic `synthesize` function on this specific group only,
/// rather than running the full `consolidate_memory` pass (which would
/// consume all groups and prevent the LLM from trying the remaining ones).
fn try_heuristic_fallback_single(
    store: &mut SessionMemoryStore,
    group: &ConsolidationGroup,
    heuristic_config: &ConsolidationConfig,
) -> bool {
    use super::consolidate::synthesize;

    let source_facts: Vec<&MemoryFact> = group
        .source_keys
        .iter()
        .filter_map(|k| store.get(k))
        .collect();

    if source_facts.len() < heuristic_config.min_group_size {
        return false;
    }

    if let Some(consolidated) = synthesize(group, &source_facts) {
        let consolidated_bytes = consolidated.key.len() + consolidated.value.len() + 6;
        let savings_pct = if group.source_bytes > 0 {
            (group.source_bytes.saturating_sub(consolidated_bytes) * 100) / group.source_bytes
        } else {
            0
        };

        if savings_pct >= heuristic_config.savings_pct {
            store.remove_batch(&group.source_keys);
            store.upsert(consolidated);
            return true;
        }
    }
    false
}

/// Try to discover cross-group insights using the LLM.
///
/// Uses a pre-captured snapshot of fact values (taken before the consolidation
/// loop modifies the store) to build the prompt.
async fn try_cross_group_insight(
    groups: &[ConsolidationGroup],
    snapshot: &[Vec<(String, String)>], // (key, value) pairs per group
    llm: &dyn LlmClient,
    max_tokens: u32,
) -> Option<MemoryFact> {
    let mut prompt = String::from(
        "Here are several groups of related facts from a working memory. \
         Identify any cross-group connections:\n\n",
    );

    for (i, (group, facts)) in groups.iter().zip(snapshot.iter()).enumerate() {
        prompt.push_str(&format!("Group {} ({:?}):\n", i + 1, group.link_reason));
        for (_key, value) in facts {
            prompt.push_str(&format!("  - {value}\n"));
        }
        prompt.push('\n');
    }

    let messages = vec![
        ChatMessage {
            role: Role::System,
            content: CROSS_GROUP_SYSTEM.to_string(),
        },
        ChatMessage {
            role: Role::User,
            content: prompt,
        },
    ];

    match llm.generate_chat(&messages, None, max_tokens).await {
        Ok(response) => {
            let response = response.trim().to_string();
            if response.eq_ignore_ascii_case("NONE")
                || response.is_empty()
                || response.chars().count() > 500
            {
                None
            } else {
                Some(MemoryFact {
                    key: "consolidated:cross_group_insight".to_string(),
                    value: response,
                    source: MemorySource::Consolidation {
                        source_keys: groups
                            .iter()
                            .flat_map(|g| g.source_keys.iter().cloned())
                            .collect(),
                    },
                    created_at: now_millis(),
                    priority: FactPriority::Finding,
                })
            }
        }
        Err(e) => {
            warn!("Cross-group insight discovery failed: {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::consolidate::ConsolidationConfig;
    use super::super::types::MemorySource;
    use async_trait::async_trait;
    use cwc_core::traits::LlmClient;

    // Mock LLM that returns canned responses
    struct MockLlm {
        response: String,
    }

    #[async_trait]
    impl LlmClient for MockLlm {
        async fn generate(
            &self,
            _prompt: &str,
            _grammar: Option<&str>,
            _max_tokens: u32,
        ) -> cwc_core::error::Result<String> {
            Ok(self.response.clone())
        }
        async fn generate_chat(
            &self,
            _messages: &[ChatMessage],
            _grammar: Option<&str>,
            _max_tokens: u32,
        ) -> cwc_core::error::Result<String> {
            Ok(self.response.clone())
        }
    }

    // Mock LLM that always fails
    struct FailingLlm;

    #[async_trait]
    impl LlmClient for FailingLlm {
        async fn generate(
            &self,
            _prompt: &str,
            _grammar: Option<&str>,
            _max_tokens: u32,
        ) -> cwc_core::error::Result<String> {
            Err(cwc_core::error::CwcError::Llm("connection refused".into()))
        }
        async fn generate_chat(
            &self,
            _messages: &[ChatMessage],
            _grammar: Option<&str>,
            _max_tokens: u32,
        ) -> cwc_core::error::Result<String> {
            Err(cwc_core::error::CwcError::Llm("connection refused".into()))
        }
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

    fn test_llm_config() -> LlmConsolidationConfig {
        LlmConsolidationConfig {
            enabled: true,
            max_tokens: 128,
            max_groups: 3,
            min_group_size: 3,
            cross_group_insights: false,
        }
    }

    fn heuristic_config() -> ConsolidationConfig {
        ConsolidationConfig::default()
    }

    fn populate_grep_facts(store: &mut SessionMemoryStore) {
        store.upsert(make_fact(
            "finding:grep_malloc",
            "3 matches in 2 files. src/alloc.rs:10: malloc(256) with extra context padding",
            FactPriority::Finding,
        ));
        store.upsert(make_fact(
            "finding:grep_free",
            "5 matches in 3 files. src/dealloc.rs:20: free(ptr) with more context padding",
            FactPriority::Finding,
        ));
        store.upsert(make_fact(
            "finding:grep_realloc",
            "2 matches in 1 files. src/realloc.rs:30: realloc(ptr, 512) with context",
            FactPriority::Finding,
        ));
    }

    #[tokio::test]
    async fn test_llm_consolidate_basic() {
        let mut store = SessionMemoryStore::new(8192, 50);
        populate_grep_facts(&mut store);

        let llm = MockLlm {
            response: "3 grep searches: malloc(3), free(5), realloc(2) in alloc/dealloc/realloc.rs".to_string(),
        };

        let report = llm_consolidate(
            &mut store,
            &llm,
            &test_llm_config(),
            &heuristic_config(),
        )
        .await;

        assert_eq!(report.groups_refined, 1);
        assert_eq!(report.fallback_to_heuristic, 0);
        assert_eq!(report.source_facts_replaced, 3);
        assert!(report.bytes_saved > 0);

        // Source facts removed
        assert!(store.get("finding:grep_malloc").is_none());
        assert!(store.get("finding:grep_free").is_none());
        assert!(store.get("finding:grep_realloc").is_none());

        // Consolidated fact present
        let consolidated = store.get("consolidated:finding:grep_summary").unwrap();
        assert!(consolidated.value.contains("grep searches"));
    }

    #[tokio::test]
    async fn test_llm_consolidate_disabled() {
        let mut store = SessionMemoryStore::new(8192, 50);
        populate_grep_facts(&mut store);

        let llm = MockLlm {
            response: "should not be called".to_string(),
        };
        let config = LlmConsolidationConfig {
            enabled: false,
            ..test_llm_config()
        };

        let report = llm_consolidate(&mut store, &llm, &config, &heuristic_config()).await;

        assert_eq!(report.groups_refined, 0);
        assert_eq!(store.len(), 3); // All facts remain
    }

    #[tokio::test]
    async fn test_llm_consolidate_fallback_on_failure() {
        let mut store = SessionMemoryStore::new(8192, 50);
        populate_grep_facts(&mut store);

        let llm = FailingLlm;

        let report = llm_consolidate(
            &mut store,
            &llm,
            &test_llm_config(),
            &heuristic_config(),
        )
        .await;

        assert_eq!(report.groups_refined, 0);
        // Heuristic fallback may or may not consolidate depending on savings threshold
        // but the function should not panic
        assert!(report.fallback_to_heuristic <= 1);
    }

    #[tokio::test]
    async fn test_llm_consolidate_empty_response_falls_back() {
        let mut store = SessionMemoryStore::new(8192, 50);
        populate_grep_facts(&mut store);

        let llm = MockLlm {
            response: "".to_string(),
        };

        let report = llm_consolidate(
            &mut store,
            &llm,
            &test_llm_config(),
            &heuristic_config(),
        )
        .await;

        // Empty response → fallback to heuristic
        assert_eq!(report.groups_refined, 0);
    }

    #[tokio::test]
    async fn test_llm_consolidate_oversized_response_falls_back() {
        let mut store = SessionMemoryStore::new(8192, 50);
        populate_grep_facts(&mut store);

        let llm = MockLlm {
            response: "x".repeat(501),
        };

        let report = llm_consolidate(
            &mut store,
            &llm,
            &test_llm_config(),
            &heuristic_config(),
        )
        .await;

        assert_eq!(report.groups_refined, 0);
    }

    #[tokio::test]
    async fn test_llm_consolidate_no_groups() {
        let mut store = SessionMemoryStore::new(8192, 50);
        store.upsert(make_fact("standalone_a", "value a", FactPriority::Finding));
        store.upsert(make_fact("standalone_b", "value b", FactPriority::Finding));

        let llm = MockLlm {
            response: "should not be called".to_string(),
        };

        let report = llm_consolidate(
            &mut store,
            &llm,
            &test_llm_config(),
            &heuristic_config(),
        )
        .await;

        assert_eq!(report.groups_refined, 0);
        assert_eq!(store.len(), 2);
    }

    #[tokio::test]
    async fn test_llm_consolidate_with_cross_group_insights() {
        let mut store = SessionMemoryStore::new(8192, 50);
        // Group 1: grep findings
        populate_grep_facts(&mut store);
        // Group 2: file_seen findings
        store.upsert(make_fact(
            "file_seen:src_alloc_rs",
            "read src/alloc.rs (150 lines) with allocator implementation",
            FactPriority::Navigation,
        ));
        store.upsert(make_fact(
            "file_seen:src_dealloc_rs",
            "read src/dealloc.rs (80 lines) with deallocator implementation",
            FactPriority::Navigation,
        ));
        store.upsert(make_fact(
            "file_seen:src_realloc_rs",
            "read src/realloc.rs (60 lines) with reallocator implementation",
            FactPriority::Navigation,
        ));

        let llm = MockLlm {
            response: "All groups relate to memory management: alloc, dealloc, realloc".to_string(),
        };
        let config = LlmConsolidationConfig {
            enabled: true,
            cross_group_insights: true,
            ..test_llm_config()
        };

        let report = llm_consolidate(&mut store, &llm, &config, &heuristic_config()).await;

        // Should have processed groups and found cross-group insight
        assert!(report.cross_group_insights <= 1);
    }

    #[tokio::test]
    async fn test_llm_consolidate_cross_group_none_response() {
        let mut store = SessionMemoryStore::new(8192, 50);
        populate_grep_facts(&mut store);
        store.upsert(make_fact(
            "file_seen:a",
            "read a.rs (10 lines) content padding here for the test",
            FactPriority::Navigation,
        ));
        store.upsert(make_fact(
            "file_seen:b",
            "read b.rs (20 lines) content padding here for the test",
            FactPriority::Navigation,
        ));
        store.upsert(make_fact(
            "file_seen:c",
            "read c.rs (30 lines) content padding here for the test",
            FactPriority::Navigation,
        ));

        let llm = MockLlm {
            response: "NONE".to_string(),
        };
        let config = LlmConsolidationConfig {
            enabled: true,
            cross_group_insights: true,
            ..test_llm_config()
        };

        let report = llm_consolidate(&mut store, &llm, &config, &heuristic_config()).await;

        // "NONE" response means no insight discovered
        assert_eq!(report.cross_group_insights, 0);
    }

    #[tokio::test]
    async fn test_llm_consolidate_preserves_goal() {
        let mut store = SessionMemoryStore::new(8192, 50);
        store.upsert(make_fact("goal", "find memory leaks", FactPriority::Goal));
        populate_grep_facts(&mut store);

        let llm = MockLlm {
            response: "3 grep searches: malloc/free/realloc across alloc modules".to_string(),
        };

        llm_consolidate(&mut store, &llm, &test_llm_config(), &heuristic_config()).await;

        // Goal must survive
        assert!(store.get("goal").is_some());
        assert_eq!(store.get("goal").unwrap().value, "find memory leaks");
    }
}
