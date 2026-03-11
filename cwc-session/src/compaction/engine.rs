use std::path::{Path, PathBuf};
use std::sync::Arc;

use cwc_core::traits::TokenCounter;

use crate::error::Result;

use super::artifact::ArtifactStore;
use super::rules::{pattern_matches, CompactionRule, CompactionStrategy};
use super::summary::{summarize_head_tail, summarize_head_truncate, summarize_top_items};

/// Result of compacting a tool result.
#[derive(Debug, Clone)]
pub struct CompactionResult {
    /// The compact inline text to keep in the conversation.
    pub inline_text: String,
    /// Token count of the inline text.
    pub inline_tokens: u32,
    /// If the full output was stored as an artifact, its path.
    pub artifact_path: Option<PathBuf>,
    /// Original token count before compaction.
    pub original_tokens: u32,
    /// Whether any compaction was applied.
    pub was_compacted: bool,
}

/// Engine that compacts tool results based on configurable rules.
pub struct CompactionEngine {
    rules: Vec<CompactionRule>,
    artifact_store: ArtifactStore,
    tokenizer: Arc<dyn TokenCounter>,
}

impl CompactionEngine {
    pub fn new(
        rules: Vec<CompactionRule>,
        artifact_dir: &Path,
        tokenizer: Arc<dyn TokenCounter>,
    ) -> Result<Self> {
        let artifact_store = ArtifactStore::new(artifact_dir)?;
        Ok(Self {
            rules,
            artifact_store,
            tokenizer,
        })
    }

    /// Compact a tool result based on matching rules.
    ///
    /// If the output is under the rule's `max_inline_tokens`, it is returned
    /// unchanged. Otherwise, a summary is generated and the full output is
    /// stored as an artifact.
    pub fn compact_tool_result(
        &self,
        tool_name: &str,
        call_id: &str,
        output: &str,
    ) -> Result<CompactionResult> {
        let original_tokens = self.tokenizer.count_tokens(output) + 4; // +4 role overhead
        let rule = self.find_rule(tool_name);

        // Check if compaction is needed
        if original_tokens <= rule.max_inline_tokens {
            return Ok(CompactionResult {
                inline_text: output.to_string(),
                inline_tokens: original_tokens,
                artifact_path: None,
                original_tokens,
                was_compacted: false,
            });
        }

        // PassThrough: return as-is regardless of size
        if matches!(rule.strategy, CompactionStrategy::PassThrough) {
            return Ok(CompactionResult {
                inline_text: output.to_string(),
                inline_tokens: original_tokens,
                artifact_path: None,
                original_tokens,
                was_compacted: false,
            });
        }

        // Store full output as artifact
        let artifact_path = self.artifact_store.store(tool_name, call_id, output)?;

        // Generate summary
        let summary = self.generate_summary(tool_name, output, rule);
        let artifact_ref = format!(
            "\n[Full output: {}]",
            artifact_path.display()
        );
        let inline_text = format!("{summary}{artifact_ref}");
        let inline_tokens = self.tokenizer.count_tokens(&inline_text) + 4;

        Ok(CompactionResult {
            inline_text,
            inline_tokens,
            artifact_path: Some(artifact_path),
            original_tokens,
            was_compacted: true,
        })
    }

    /// Compact a long text message (assistant or user).
    ///
    /// Uses simple head-truncation with a token budget. No artifact storage.
    pub fn compact_text(&self, text: &str, max_tokens: u32) -> CompactionResult {
        let original_tokens = self.tokenizer.count_tokens(text) + 4;
        if original_tokens <= max_tokens {
            return CompactionResult {
                inline_text: text.to_string(),
                inline_tokens: original_tokens,
                artifact_path: None,
                original_tokens,
                was_compacted: false,
            };
        }

        let summary = summarize_head_truncate(text, max_tokens.saturating_sub(4), &*self.tokenizer);
        let inline_tokens = self.tokenizer.count_tokens(&summary) + 4;
        CompactionResult {
            inline_text: summary,
            inline_tokens,
            artifact_path: None,
            original_tokens,
            was_compacted: true,
        }
    }

    /// Find the first matching rule for a tool name.
    fn find_rule(&self, tool_name: &str) -> &CompactionRule {
        self.rules
            .iter()
            .find(|r| pattern_matches(&r.tool_pattern, tool_name))
            .unwrap_or_else(|| {
                // Should never happen if rules include a wildcard catch-all,
                // but fall back to a sensible default.
                static FALLBACK: CompactionRule = CompactionRule {
                    tool_pattern: String::new(),
                    max_inline_tokens: 500,
                    summary_items: 0,
                    strategy: CompactionStrategy::HeadTruncate,
                };
                &FALLBACK
            })
    }

    fn generate_summary(&self, tool_name: &str, output: &str, rule: &CompactionRule) -> String {
        match &rule.strategy {
            CompactionStrategy::HeadTruncate => {
                summarize_head_truncate(output, rule.max_inline_tokens, &*self.tokenizer)
            }
            CompactionStrategy::TopItems => summarize_top_items(
                tool_name,
                output,
                rule.summary_items,
                rule.max_inline_tokens,
                &*self.tokenizer,
            ),
            CompactionStrategy::HeadTail { tail_lines } => {
                summarize_head_tail(output, rule.max_inline_tokens, *tail_lines, &*self.tokenizer)
            }
            CompactionStrategy::PassThrough => output.to_string(),
            CompactionStrategy::Custom(_) => {
                // Custom extractors not yet implemented — fall back to HeadTruncate
                summarize_head_truncate(output, rule.max_inline_tokens, &*self.tokenizer)
            }
        }
    }

    /// Access the artifact store.
    pub fn artifact_store(&self) -> &ArtifactStore {
        &self.artifact_store
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::rules::default_rules;

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

    fn make_engine(dir: &Path) -> CompactionEngine {
        CompactionEngine::new(default_rules(), dir, Arc::new(WordCounter)).unwrap()
    }

    #[test]
    fn test_engine_under_limit_no_compaction() {
        let dir = tempfile::tempdir().unwrap();
        let engine = make_engine(dir.path());

        let result = engine
            .compact_tool_result("file.grep", "c1", "short output")
            .unwrap();
        assert!(!result.was_compacted);
        assert!(result.artifact_path.is_none());
        assert_eq!(result.inline_text, "short output");
    }

    #[test]
    fn test_engine_over_limit_compacted() {
        let dir = tempfile::tempdir().unwrap();
        let engine = make_engine(dir.path());

        // Generate output that exceeds file.grep's 300-token limit
        let lines: Vec<String> = (0..200)
            .map(|i| format!("src/file{i}.rs:{i}: let x = allocate({i});"))
            .collect();
        let output = lines.join("\n");

        let result = engine
            .compact_tool_result("file.grep", "c1", &output)
            .unwrap();
        assert!(result.was_compacted);
        assert!(result.artifact_path.is_some());
        assert!(result.inline_text.contains("Found 200 matches"));
        assert!(result.inline_text.contains("[Full output:"));
        assert!(result.inline_tokens < result.original_tokens);
    }

    #[test]
    fn test_engine_passthrough_not_compacted() {
        let dir = tempfile::tempdir().unwrap();
        let engine = make_engine(dir.path());

        // file.write uses PassThrough — never compacted
        let long_output = "x ".repeat(1000);
        let result = engine
            .compact_tool_result("file.write", "c1", &long_output)
            .unwrap();
        assert!(!result.was_compacted);
        assert!(result.artifact_path.is_none());
    }

    #[test]
    fn test_engine_compact_text() {
        let dir = tempfile::tempdir().unwrap();
        let engine = make_engine(dir.path());

        let long_text = (0..100)
            .map(|i| format!("line {i} with some words"))
            .collect::<Vec<_>>()
            .join("\n");

        let result = engine.compact_text(&long_text, 20);
        assert!(result.was_compacted);
        assert!(result.inline_tokens <= 20);
        assert!(result.inline_text.contains("more lines"));
    }

    #[test]
    fn test_engine_compact_text_under_limit() {
        let dir = tempfile::tempdir().unwrap();
        let engine = make_engine(dir.path());

        let result = engine.compact_text("short message", 100);
        assert!(!result.was_compacted);
        assert_eq!(result.inline_text, "short message");
    }

    #[test]
    fn test_engine_unknown_tool_matches_wildcard() {
        let dir = tempfile::tempdir().unwrap();
        let engine = make_engine(dir.path());

        // Generate long output for an unknown tool
        let output: String = (0..500).map(|i| format!("line {i} content")).collect::<Vec<_>>().join("\n");
        let result = engine
            .compact_tool_result("custom.tool", "c1", &output)
            .unwrap();
        // Wildcard rule: max_inline=500, HeadTruncate
        assert!(result.was_compacted);
        assert!(result.artifact_path.is_some());
    }

    #[test]
    fn test_engine_artifact_retrievable() {
        let dir = tempfile::tempdir().unwrap();
        let engine = make_engine(dir.path());

        let lines: Vec<String> = (0..200)
            .map(|i| format!("src/file{i}.rs:{i}: content"))
            .collect();
        let output = lines.join("\n");

        let result = engine
            .compact_tool_result("file.grep", "c1", &output)
            .unwrap();
        assert!(result.was_compacted);

        // Full output should be retrievable from artifact
        let artifact_path = result.artifact_path.unwrap();
        let full = engine.artifact_store().retrieve(&artifact_path).unwrap();
        assert_eq!(full, output);
    }

    #[test]
    fn test_engine_empty_output_not_compacted() {
        let dir = tempfile::tempdir().unwrap();
        let engine = make_engine(dir.path());

        let result = engine.compact_tool_result("shell", "c1", "").unwrap();
        assert!(!result.was_compacted);
        assert_eq!(result.inline_text, "");
    }

    #[test]
    fn test_engine_empty_rules_uses_fallback() {
        let dir = tempfile::tempdir().unwrap();
        // Empty rules — should use the static FALLBACK (500 tokens, HeadTruncate)
        let engine = CompactionEngine::new(vec![], dir.path(), Arc::new(WordCounter)).unwrap();

        let output: String = (0..500).map(|i| format!("line {i} content")).collect::<Vec<_>>().join("\n");
        let result = engine.compact_tool_result("anything", "c1", &output).unwrap();
        assert!(result.was_compacted);
        assert!(result.artifact_path.is_some());
    }

    #[test]
    fn test_engine_max_inline_zero_always_compacts() {
        let dir = tempfile::tempdir().unwrap();
        let rules = vec![CompactionRule {
            tool_pattern: "*".into(),
            max_inline_tokens: 0,
            summary_items: 0,
            strategy: CompactionStrategy::HeadTruncate,
        }];
        let engine = CompactionEngine::new(rules, dir.path(), Arc::new(WordCounter)).unwrap();

        // Even short output should be compacted when max_inline_tokens=0
        let result = engine.compact_tool_result("tool", "c1", "hello world").unwrap();
        assert!(result.was_compacted);
        assert!(result.artifact_path.is_some());
    }

    #[test]
    fn test_engine_preserves_call_id_in_artifact_name() {
        let dir = tempfile::tempdir().unwrap();
        let engine = make_engine(dir.path());

        let output: String = (0..500).map(|i| format!("line {i} content")).collect::<Vec<_>>().join("\n");
        let result = engine
            .compact_tool_result("shell", "call_abc_123", &output)
            .unwrap();
        let path = result.artifact_path.unwrap();
        let filename = path.file_name().unwrap().to_str().unwrap();
        assert!(
            filename.contains("call_abc_123"),
            "filename should contain call_id: {filename}"
        );
    }
}
