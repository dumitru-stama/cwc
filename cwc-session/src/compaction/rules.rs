use serde::{Deserialize, Serialize};

/// Per-tool compaction configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionRule {
    /// Tool name pattern (exact match or glob, e.g., "file.*", "build").
    pub tool_pattern: String,
    /// Maximum tokens to keep inline. 0 = always compact.
    pub max_inline_tokens: u32,
    /// Number of "top items" to include in summary (for TopItems strategy).
    pub summary_items: usize,
    /// Strategy for generating the summary.
    pub strategy: CompactionStrategy,
}

/// How to generate a compact summary of tool output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompactionStrategy {
    /// Keep first N lines that fit in the token budget, append "... (M more lines)".
    HeadTruncate,
    /// Extract structured items (matches, functions, errors) and show top N.
    TopItems,
    /// Keep first lines + last M lines (for build output where errors appear at end).
    HeadTail { tail_lines: usize },
    /// Custom extractor name (registered at runtime).
    Custom(String),
    /// Pass through unchanged (for tools with already-small output).
    PassThrough,
}

/// Match a tool name against a rule pattern.
///
/// Patterns:
/// - Exact match: `"build"` matches `"build"` only
/// - Glob prefix: `"file.*"` matches `"file.grep"`, `"file.read"`, etc.
/// - Wildcard: `"*"` matches anything
pub fn pattern_matches(pattern: &str, tool_name: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix(".*") {
        // Glob: "file.*" matches "file.grep", "file.read_range", etc.
        tool_name.starts_with(prefix)
            && tool_name.as_bytes().get(prefix.len()) == Some(&b'.')
    } else {
        // Exact match
        pattern == tool_name
    }
}

/// Default compaction rules for common tool patterns.
///
/// Rules are ordered most-specific-first. The first matching rule wins.
pub fn default_rules() -> Vec<CompactionRule> {
    vec![
        CompactionRule {
            tool_pattern: "file.write".into(),
            max_inline_tokens: 200,
            summary_items: 0,
            strategy: CompactionStrategy::PassThrough,
        },
        CompactionRule {
            tool_pattern: "file.grep".into(),
            max_inline_tokens: 300,
            summary_items: 5,
            strategy: CompactionStrategy::TopItems,
        },
        CompactionRule {
            tool_pattern: "file.read".into(),
            max_inline_tokens: 1500,
            summary_items: 0,
            strategy: CompactionStrategy::HeadTruncate,
        },
        CompactionRule {
            tool_pattern: "file.*".into(),
            max_inline_tokens: 500,
            summary_items: 0,
            strategy: CompactionStrategy::HeadTruncate,
        },
        CompactionRule {
            tool_pattern: "build".into(),
            max_inline_tokens: 500,
            summary_items: 5,
            strategy: CompactionStrategy::HeadTail { tail_lines: 20 },
        },
        CompactionRule {
            tool_pattern: "test.*".into(),
            max_inline_tokens: 500,
            summary_items: 5,
            strategy: CompactionStrategy::TopItems,
        },
        CompactionRule {
            tool_pattern: "shell".into(),
            max_inline_tokens: 500,
            summary_items: 0,
            strategy: CompactionStrategy::HeadTruncate,
        },
        // Catch-all: any tool not matched above
        CompactionRule {
            tool_pattern: "*".into(),
            max_inline_tokens: 500,
            summary_items: 0,
            strategy: CompactionStrategy::HeadTruncate,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_matches_exact() {
        assert!(pattern_matches("build", "build"));
        assert!(!pattern_matches("build", "build.test"));
        assert!(!pattern_matches("build", "rebuild"));
    }

    #[test]
    fn test_pattern_matches_glob() {
        assert!(pattern_matches("file.*", "file.grep"));
        assert!(pattern_matches("file.*", "file.read"));
        assert!(pattern_matches("file.*", "file.read_range"));
        assert!(!pattern_matches("file.*", "file"));
        assert!(!pattern_matches("file.*", "filetype"));
    }

    #[test]
    fn test_pattern_matches_wildcard() {
        assert!(pattern_matches("*", "anything"));
        assert!(pattern_matches("*", "file.grep"));
        assert!(pattern_matches("*", ""));
    }

    #[test]
    fn test_pattern_matches_test_glob() {
        assert!(pattern_matches("test.*", "test.run"));
        assert!(pattern_matches("test.*", "test.coverage"));
        assert!(!pattern_matches("test.*", "test"));
        assert!(!pattern_matches("test.*", "testing"));
    }

    #[test]
    fn test_default_rules_order() {
        let rules = default_rules();
        // file.write should come before file.* glob
        let write_idx = rules
            .iter()
            .position(|r| r.tool_pattern == "file.write")
            .unwrap();
        let glob_idx = rules
            .iter()
            .position(|r| r.tool_pattern == "file.*")
            .unwrap();
        assert!(write_idx < glob_idx, "exact rules before glob rules");
    }

    #[test]
    fn test_default_rules_known_tools() {
        let rules = default_rules();
        // file.grep should match TopItems
        let grep = rules.iter().find(|r| r.tool_pattern == "file.grep").unwrap();
        assert!(matches!(grep.strategy, CompactionStrategy::TopItems));
        assert_eq!(grep.summary_items, 5);

        // build should match HeadTail
        let build = rules.iter().find(|r| r.tool_pattern == "build").unwrap();
        assert!(matches!(build.strategy, CompactionStrategy::HeadTail { tail_lines: 20 }));
    }

    #[test]
    fn test_default_rules_wildcard_last() {
        let rules = default_rules();
        let last = rules.last().unwrap();
        assert_eq!(last.tool_pattern, "*");
    }

    #[test]
    fn test_default_rules_unknown_tool_matches_wildcard() {
        let rules = default_rules();
        let tool = "custom.analyzer";
        // Walk rules, first match wins
        let matched = rules.iter().find(|r| pattern_matches(&r.tool_pattern, tool));
        assert!(matched.is_some());
        assert_eq!(matched.unwrap().tool_pattern, "*");
    }

    #[test]
    fn test_first_matching_rule_wins() {
        let rules = default_rules();
        // "file.grep" should match "file.grep" (exact) before "file.*" (glob)
        let matched = rules
            .iter()
            .find(|r| pattern_matches(&r.tool_pattern, "file.grep"))
            .unwrap();
        assert_eq!(matched.tool_pattern, "file.grep");
        assert!(matches!(matched.strategy, CompactionStrategy::TopItems));
    }

    #[test]
    fn test_pattern_matches_dot_star_prefix() {
        // ".*" means tools starting with "." followed by something
        assert!(pattern_matches(".*", ".hidden"));
        assert!(!pattern_matches(".*", "visible"));
        assert!(!pattern_matches(".*", ""));
    }

    #[test]
    fn test_pattern_matches_empty_tool_name() {
        assert!(pattern_matches("*", ""));
        assert!(pattern_matches("", ""));
        assert!(!pattern_matches("build", ""));
        assert!(!pattern_matches("file.*", ""));
    }

    #[test]
    fn test_compaction_rule_toml_roundtrip() {
        let rule = CompactionRule {
            tool_pattern: "file.grep".into(),
            max_inline_tokens: 300,
            summary_items: 5,
            strategy: CompactionStrategy::TopItems,
        };
        let toml_str = toml::to_string(&rule).unwrap();
        let back: CompactionRule = toml::from_str(&toml_str).unwrap();
        assert_eq!(back.tool_pattern, "file.grep");
        assert_eq!(back.max_inline_tokens, 300);
        assert!(matches!(back.strategy, CompactionStrategy::TopItems));
    }

    #[test]
    fn test_compaction_rule_serde_roundtrip() {
        let rule = CompactionRule {
            tool_pattern: "build".into(),
            max_inline_tokens: 500,
            summary_items: 5,
            strategy: CompactionStrategy::HeadTail { tail_lines: 20 },
        };
        let json = serde_json::to_string(&rule).unwrap();
        let back: CompactionRule = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tool_pattern, "build");
        assert_eq!(back.max_inline_tokens, 500);
        assert!(matches!(back.strategy, CompactionStrategy::HeadTail { tail_lines: 20 }));
    }
}
