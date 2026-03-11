use std::sync::LazyLock;

use regex::Regex;

use crate::longterm::MemoryCategory;

/// A candidate memory extracted from a conversation exchange.
#[derive(Debug, Clone)]
pub struct PendingMemory {
    pub key: String,
    pub value: String,
    pub category: MemoryCategory,
    pub confidence: f32,
}

static ALWAYS_NEVER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(always|never)\s+(.{3,60})").expect("always/never regex")
});

static CORRECTION_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:no,?\s+)?(?:actually|it'?s|it is|that'?s|that is),?\s+(.{3,80})")
        .expect("correction regex")
});

static PREFER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:i\s+prefer|i\s+want|please\s+use|i\s+always\s+use)\s+(.{3,60})")
        .expect("preference regex")
});

static DECISION_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:we\s+decided|let'?s\s+(?:go\s+with|use)|the\s+decision\s+is)\s+(.{3,80})")
        .expect("decision regex")
});

/// Extract memorable facts from a user query (no LLM call, deterministic).
pub fn extract_memories(query: &str, _response: &str) -> Vec<PendingMemory> {
    let mut results = Vec::new();

    // "always X" / "never Y" → UserPreference
    for cap in ALWAYS_NEVER_RE.captures_iter(query) {
        let keyword = cap[1].to_lowercase();
        let rest = cap[2].trim().trim_end_matches('.').to_string();
        results.push(PendingMemory {
            key: format!("pref:{}", slugify(&rest)),
            value: format!("{keyword} {rest}"),
            category: MemoryCategory::UserPreference,
            confidence: 0.8,
        });
    }

    // Corrections: "no, it's X" / "actually, it's X"
    if query.to_lowercase().starts_with("no,")
        || query.to_lowercase().starts_with("no ")
        || query.to_lowercase().contains("actually")
    {
        for cap in CORRECTION_RE.captures_iter(query) {
            let fact = cap[1].trim().trim_end_matches('.').to_string();
            if fact.split_whitespace().count() >= 2 {
                results.push(PendingMemory {
                    key: format!("correction:{}", slugify(&fact)),
                    value: fact,
                    category: MemoryCategory::Correction,
                    confidence: 0.7,
                });
            }
        }
    }

    // Preferences: "I prefer X" / "use X" / "please use X"
    for cap in PREFER_RE.captures_iter(query) {
        let pref = cap[1].trim().trim_end_matches('.').to_string();
        if pref.split_whitespace().count() >= 1 {
            // Avoid duplicating with always/never
            let slug = slugify(&pref);
            if !results.iter().any(|r| r.key.contains(&slug)) {
                results.push(PendingMemory {
                    key: format!("pref:{slug}"),
                    value: pref,
                    category: MemoryCategory::UserPreference,
                    confidence: 0.6,
                });
            }
        }
    }

    // Decisions: "we decided to use X" / "let's go with X"
    for cap in DECISION_RE.captures_iter(query) {
        let decision = cap[1].trim().trim_end_matches('.').to_string();
        if decision.split_whitespace().count() >= 2 {
            results.push(PendingMemory {
                key: format!("decision:{}", slugify(&decision)),
                value: decision,
                category: MemoryCategory::PriorDecision,
                confidence: 0.7,
            });
        }
    }

    results
}

/// Convert text to a simple slug for use as a key.
fn slugify(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect::<String>()
        .split('_')
        .filter(|s| !s.is_empty())
        .collect::<Vec<&str>>()
        .join("_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_always_preference() {
        let mems = extract_memories("Always use JSON output format", "ok");
        assert!(!mems.is_empty());
        assert_eq!(mems[0].category, MemoryCategory::UserPreference);
        assert!(mems[0].value.contains("use JSON output format"));
    }

    #[test]
    fn test_extract_never_preference() {
        let mems = extract_memories("Never include XML in responses", "ok");
        assert!(!mems.is_empty());
        assert_eq!(mems[0].category, MemoryCategory::UserPreference);
        assert!(mems[0].value.contains("never"));
    }

    #[test]
    fn test_extract_correction() {
        let mems = extract_memories("No, it's 5 not 3", "ok");
        assert!(!mems.is_empty());
        let correction = mems.iter().find(|m| m.category == MemoryCategory::Correction);
        assert!(correction.is_some());
    }

    #[test]
    fn test_extract_correction_actually() {
        let mems = extract_memories("Actually, the API returns JSON not XML", "ok");
        assert!(!mems.is_empty());
        assert!(mems.iter().any(|m| m.category == MemoryCategory::Correction));
    }

    #[test]
    fn test_extract_decision() {
        let mems = extract_memories("We decided to use pgvector over FAISS", "ok");
        assert!(!mems.is_empty());
        let decision = mems.iter().find(|m| m.category == MemoryCategory::PriorDecision);
        assert!(decision.is_some());
        assert!(decision.unwrap().value.contains("pgvector"));
    }

    #[test]
    fn test_extract_nothing_from_plain_query() {
        let mems = extract_memories("What is ownership in Rust?", "Ownership is...");
        assert!(mems.is_empty());
    }

    #[test]
    fn test_extract_preference_use() {
        let mems = extract_memories("Please use metric units", "ok");
        assert!(!mems.is_empty());
        assert!(mems.iter().any(|m| m.category == MemoryCategory::UserPreference));
    }

    #[test]
    fn test_slugify() {
        assert_eq!(slugify("use JSON format!"), "use_json_format");
        assert_eq!(slugify("  hello  world  "), "hello_world");
    }

    #[test]
    fn test_extract_no_false_positive_bare_use() {
        // "How do I use HashMap?" should NOT extract a preference
        let mems = extract_memories("How do I use HashMap in Rust?", "You can...");
        let prefs: Vec<_> = mems.iter().filter(|m| m.category == MemoryCategory::UserPreference).collect();
        assert!(prefs.is_empty(), "bare 'use' in a question should not trigger preference: {prefs:?}");
    }

    #[test]
    fn test_extract_lets_go_with_decision() {
        let mems = extract_memories("Let's go with PostgreSQL for the database", "ok");
        assert!(!mems.is_empty());
        let decision = mems.iter().find(|m| m.category == MemoryCategory::PriorDecision);
        assert!(decision.is_some());
        assert!(decision.unwrap().value.contains("PostgreSQL"));
    }

    #[test]
    fn test_extract_multi_pattern_single_query() {
        let mems = extract_memories("Always use JSON. We decided to use pgvector", "ok");
        assert!(mems.iter().any(|m| m.category == MemoryCategory::UserPreference));
        assert!(mems.iter().any(|m| m.category == MemoryCategory::PriorDecision));
    }

    #[test]
    fn test_extract_i_prefer_pattern() {
        let mems = extract_memories("I prefer snake_case naming", "ok");
        assert!(!mems.is_empty());
        assert!(mems.iter().any(|m| m.category == MemoryCategory::UserPreference));
        assert!(mems.iter().any(|m| m.value.contains("snake_case")));
    }

    #[test]
    fn test_extract_short_fragments_ignored() {
        // Fragments shorter than 3 chars should be rejected by the regex {3,} minimum
        let mems = extract_memories("Always do", "ok");
        // "do" is only 2 chars, but "always" regex captures "do" which is at the min boundary
        // The regex is `.{3,60}` so "do" (2 chars) doesn't match
        assert!(mems.is_empty());
    }
}
