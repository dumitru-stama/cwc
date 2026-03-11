use serde::{Deserialize, Serialize};

/// A fact extracted from the conversation for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFact {
    /// Deterministic key, e.g., "finding:grep_malloc", "build_result".
    pub key: String,
    /// Compact fact text.
    pub value: String,
    /// Where this fact came from.
    pub source: MemorySource,
    /// Unix timestamp in milliseconds.
    pub created_at: u64,
    /// Eviction priority.
    pub priority: FactPriority,
}

/// Priority levels for memory facts. Higher priority = evicted last.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FactPriority {
    /// File/resource seen — evicted first when memory is tight.
    Navigation = 0,
    /// Build/test status — overwritten on each new result.
    Status = 1,
    /// Key finding from tool use — evicted last among non-goal.
    Finding = 2,
    /// Goal — never evicted.
    Goal = 3,
}

/// Where a memory fact originated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemorySource {
    UserMessage { message_index: usize },
    ToolResult { tool_name: String, call_id: String },
    AssistantConclusion { message_index: usize },
    Consolidation { source_keys: Vec<String> },
}

/// Generate a deterministic slug from text.
///
/// Lowercase, replace non-alphanumeric with `_`, collapse runs, truncate to max_len chars.
pub fn make_slug(text: &str, max_len: usize) -> String {
    let mut slug = String::with_capacity(text.len().min(max_len));
    let mut last_was_underscore = false;

    for c in text.chars() {
        if c.is_ascii_alphanumeric() {
            slug.push(c.to_ascii_lowercase());
            last_was_underscore = false;
        } else if !last_was_underscore && !slug.is_empty() {
            slug.push('_');
            last_was_underscore = true;
        }
        if slug.len() >= max_len {
            break;
        }
    }

    // Trim trailing underscore
    if slug.ends_with('_') {
        slug.pop();
    }
    slug
}

/// Current time in unix milliseconds.
pub fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slug_basic() {
        assert_eq!(make_slug("file.grep malloc", 40), "file_grep_malloc");
    }

    #[test]
    fn test_slug_special_chars() {
        assert_eq!(make_slug("hello@world#2024!", 40), "hello_world_2024");
    }

    #[test]
    fn test_slug_collapse_runs() {
        assert_eq!(make_slug("a---b...c   d", 40), "a_b_c_d");
    }

    #[test]
    fn test_slug_truncate() {
        let long = "a".repeat(100);
        let slug = make_slug(&long, 40);
        assert!(slug.len() <= 40);
    }

    #[test]
    fn test_slug_empty() {
        assert_eq!(make_slug("", 40), "");
    }

    #[test]
    fn test_slug_only_special() {
        assert_eq!(make_slug("@#$%", 40), "");
    }

    #[test]
    fn test_slug_leading_special() {
        // Leading specials are skipped until first alphanumeric
        assert_eq!(make_slug("...hello", 40), "hello");
    }

    #[test]
    fn test_fact_priority_ordering() {
        assert!(FactPriority::Navigation < FactPriority::Status);
        assert!(FactPriority::Status < FactPriority::Finding);
        assert!(FactPriority::Finding < FactPriority::Goal);
    }

    #[test]
    fn test_memory_fact_serde_roundtrip() {
        let fact = MemoryFact {
            key: "build_result".into(),
            value: "success".into(),
            source: MemorySource::ToolResult {
                tool_name: "build".into(),
                call_id: "c1".into(),
            },
            created_at: 1234567890,
            priority: FactPriority::Status,
        };
        let json = serde_json::to_string(&fact).unwrap();
        let back: MemoryFact = serde_json::from_str(&json).unwrap();
        assert_eq!(back.key, "build_result");
        assert_eq!(back.priority, FactPriority::Status);
    }

    #[test]
    fn test_now_millis_nonzero() {
        assert!(now_millis() > 0);
    }

    #[test]
    fn test_memory_source_consolidation_serde_roundtrip() {
        let fact = MemoryFact {
            key: "consolidated:grep_summary".into(),
            value: "5 searches found 20 matches".into(),
            source: MemorySource::Consolidation {
                source_keys: vec![
                    "finding:grep_malloc".into(),
                    "finding:grep_free".into(),
                    "finding:grep_realloc".into(),
                ],
            },
            created_at: 9999999,
            priority: FactPriority::Finding,
        };
        let json = serde_json::to_string(&fact).unwrap();
        let back: MemoryFact = serde_json::from_str(&json).unwrap();
        assert_eq!(back.key, "consolidated:grep_summary");
        if let MemorySource::Consolidation { source_keys } = &back.source {
            assert_eq!(source_keys.len(), 3);
            assert_eq!(source_keys[0], "finding:grep_malloc");
        } else {
            panic!("expected Consolidation source");
        }
    }
}
