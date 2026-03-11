pub mod artifact;
pub mod engine;
pub mod rules;
pub mod summary;

pub use artifact::ArtifactStore;
pub use engine::{CompactionEngine, CompactionResult};
pub use rules::{CompactionRule, CompactionStrategy};

use crate::error::Result;
use crate::message::{MessageFlags, Session, SessionRole};

/// Report of a session compaction pass.
#[derive(Debug, Clone)]
pub struct CompactionReport {
    /// Number of messages that were compacted.
    pub messages_compacted: usize,
    /// Total tokens across processed messages before compaction.
    pub tokens_before: u32,
    /// Total tokens across processed messages after compaction.
    pub tokens_after: u32,
    /// Tokens saved (tokens_before - tokens_after).
    pub tokens_saved: u32,
    /// Number of artifact files created.
    pub artifacts_created: usize,
}

/// Compact all tool results in a session that haven't been compacted yet.
///
/// Modifies messages in-place:
/// - Tool result messages over the inline limit get their content replaced with a summary
/// - The IS_COMPACTED flag is set on processed messages
/// - Token counts are updated on modified messages
///
/// Returns a report of what was changed.
pub fn compact_session(
    session: &mut Session,
    engine: &CompactionEngine,
) -> Result<CompactionReport> {
    let mut messages_compacted = 0;
    let mut tokens_before: u32 = 0;
    let mut tokens_after: u32 = 0;
    let mut artifacts_created = 0;

    let messages = session.messages_mut();
    for msg in messages.iter_mut() {
        // Skip non-tool messages
        if msg.role != SessionRole::Tool {
            continue;
        }
        // Skip already-compacted messages
        if msg.flags.contains(MessageFlags::IS_COMPACTED) {
            continue;
        }
        // Skip preserved messages
        if msg.flags.contains(MessageFlags::PRESERVE) {
            continue;
        }

        let tool_name = msg
            .tool_result
            .as_ref()
            .map(|tr| tr.tool_name.as_str())
            .unwrap_or("unknown");
        let call_id = msg
            .tool_result
            .as_ref()
            .map(|tr| tr.call_id.as_str())
            .unwrap_or("unknown");

        let result = engine.compact_tool_result(tool_name, call_id, &msg.content)?;

        tokens_before += result.original_tokens;

        if result.was_compacted {
            msg.content = result.inline_text.clone();
            // Keep tool_result.output in sync with content to avoid
            // data inconsistency when serialized. Full output is in the artifact.
            if let Some(ref mut tr) = msg.tool_result {
                tr.output = result.inline_text;
            }
            msg.token_count = result.inline_tokens;
            msg.flags.insert(MessageFlags::IS_COMPACTED);
            tokens_after += result.inline_tokens;
            messages_compacted += 1;
            if result.artifact_path.is_some() {
                artifacts_created += 1;
            }
        } else {
            tokens_after += result.original_tokens;
        }
    }

    // Recalculate session total after modifying message token counts in-place
    session.recalculate_total();

    let tokens_saved = tokens_before.saturating_sub(tokens_after);
    Ok(CompactionReport {
        messages_compacted,
        tokens_before,
        tokens_after,
        tokens_saved,
        artifacts_created,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{SessionMessage, ToolCall, ToolResult};
    use std::sync::Arc;

    use cwc_core::traits::TokenCounter;

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

    fn tc() -> Arc<dyn TokenCounter> {
        Arc::new(WordCounter)
    }

    fn make_engine(dir: &std::path::Path) -> CompactionEngine {
        CompactionEngine::new(rules::default_rules(), dir, tc()).unwrap()
    }

    fn make_tool_result(tool_name: &str, call_id: &str, output: &str) -> SessionMessage {
        SessionMessage::tool_result(ToolResult {
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            output: output.into(),
            is_error: false,
        })
    }

    #[test]
    fn test_compact_session_processes_tool_messages() {
        let dir = tempfile::tempdir().unwrap();
        let engine = make_engine(dir.path());

        let mut session = Session::new(tc());
        session.push(SessionMessage::system("You are helpful"));
        session.push(SessionMessage::text(SessionRole::User, "find bugs"));
        session.push(SessionMessage::assistant_tool_calls(
            "",
            vec![ToolCall {
                call_id: "c1".into(),
                tool_name: "file.grep".into(),
                arguments: serde_json::json!({"pattern": "bug"}),
            }],
        ));
        // Create a large tool result (>300 tokens for file.grep)
        let big_output: String = (0..200)
            .map(|i| format!("src/mod{i}.rs:{i}: let bug = {i};"))
            .collect::<Vec<_>>()
            .join("\n");
        session.push(make_tool_result("file.grep", "c1", &big_output));
        session.push(SessionMessage::text(SessionRole::Assistant, "Found bugs"));

        let total_before = session.total_tokens();
        let report = compact_session(&mut session, &engine).unwrap();

        assert_eq!(report.messages_compacted, 1);
        assert!(report.tokens_saved > 0);
        assert_eq!(report.artifacts_created, 1);
        assert!(session.total_tokens() < total_before);
    }

    #[test]
    fn test_compact_session_sets_is_compacted_flag() {
        let dir = tempfile::tempdir().unwrap();
        let engine = make_engine(dir.path());

        let mut session = Session::new(tc());
        let big: String = (0..500).map(|i| format!("line {i} content")).collect::<Vec<_>>().join("\n");
        session.push(make_tool_result("shell", "c1", &big));

        compact_session(&mut session, &engine).unwrap();

        assert!(session.messages()[0].flags.contains(MessageFlags::IS_COMPACTED));
    }

    #[test]
    fn test_compact_session_skips_already_compacted() {
        let dir = tempfile::tempdir().unwrap();
        let engine = make_engine(dir.path());

        let mut session = Session::new(tc());
        let big: String = (0..500).map(|i| format!("line {i} content")).collect::<Vec<_>>().join("\n");
        session.push(make_tool_result("shell", "c1", &big));

        // First compaction
        let report1 = compact_session(&mut session, &engine).unwrap();
        assert_eq!(report1.messages_compacted, 1);

        // Second compaction — should skip
        let report2 = compact_session(&mut session, &engine).unwrap();
        assert_eq!(report2.messages_compacted, 0);
        assert_eq!(report2.tokens_saved, 0);
    }

    #[test]
    fn test_compact_session_does_not_touch_non_tool() {
        let dir = tempfile::tempdir().unwrap();
        let engine = make_engine(dir.path());

        let mut session = Session::new(tc());
        let long_text = "word ".repeat(1000);
        session.push(SessionMessage::system("system prompt"));
        session.push(SessionMessage::text(SessionRole::User, &long_text));
        session.push(SessionMessage::text(SessionRole::Assistant, &long_text));

        let report = compact_session(&mut session, &engine).unwrap();
        assert_eq!(report.messages_compacted, 0);
        assert_eq!(report.tokens_saved, 0);
        assert_eq!(report.artifacts_created, 0);
    }

    #[test]
    fn test_compact_session_tokens_saved_accurate() {
        let dir = tempfile::tempdir().unwrap();
        let engine = make_engine(dir.path());

        let mut session = Session::new(tc());
        // Multi-line output (HeadTruncate works on lines)
        let big: String = (0..500)
            .map(|i| format!("output line {i} with some content here"))
            .collect::<Vec<_>>()
            .join("\n");
        session.push(make_tool_result("shell", "c1", &big));

        let report = compact_session(&mut session, &engine).unwrap();
        assert!(
            report.tokens_before >= report.tokens_after,
            "before={} after={}",
            report.tokens_before,
            report.tokens_after
        );
        assert_eq!(report.tokens_saved, report.tokens_before - report.tokens_after);
    }

    #[test]
    fn test_compact_session_preserves_call_id_linkage() {
        let dir = tempfile::tempdir().unwrap();
        let engine = make_engine(dir.path());

        let mut session = Session::new(tc());
        session.push(SessionMessage::assistant_tool_calls(
            "",
            vec![ToolCall {
                call_id: "c1".into(),
                tool_name: "shell".into(),
                arguments: serde_json::json!({}),
            }],
        ));
        let big: String = (0..500).map(|i| format!("line {i} content")).collect::<Vec<_>>().join("\n");
        session.push(make_tool_result("shell", "c1", &big));

        compact_session(&mut session, &engine).unwrap();

        // Tool result should still have its call_id
        let tool_msg = &session.messages()[1];
        let tr = tool_msg.tool_result.as_ref().unwrap();
        assert_eq!(tr.call_id, "c1");
        assert_eq!(tr.tool_name, "shell");
    }

    #[test]
    fn test_compact_session_summary_includes_artifact_reference() {
        let dir = tempfile::tempdir().unwrap();
        let engine = make_engine(dir.path());

        let mut session = Session::new(tc());
        let big: String = (0..500).map(|i| format!("line {i} content")).collect::<Vec<_>>().join("\n");
        session.push(make_tool_result("shell", "c1", &big));

        compact_session(&mut session, &engine).unwrap();

        let content = &session.messages()[0].content;
        assert!(content.contains("[Full output:"), "should reference artifact: {content}");
    }

    #[test]
    fn test_compact_session_token_count_updated() {
        let dir = tempfile::tempdir().unwrap();
        let engine = make_engine(dir.path());

        let mut session = Session::new(tc());
        let big: String = (0..500).map(|i| format!("line {i} content")).collect::<Vec<_>>().join("\n");
        session.push(make_tool_result("shell", "c1", &big));

        let before_msg_tokens = session.messages()[0].token_count;
        compact_session(&mut session, &engine).unwrap();
        let after_msg_tokens = session.messages()[0].token_count;

        assert!(after_msg_tokens < before_msg_tokens);
        // Session total should match sum of all message token counts
        let sum: u32 = session.messages().iter().map(|m| m.token_count).sum();
        assert_eq!(session.total_tokens(), sum);
    }

    #[test]
    fn test_compact_session_small_tool_not_compacted() {
        let dir = tempfile::tempdir().unwrap();
        let engine = make_engine(dir.path());

        let mut session = Session::new(tc());
        session.push(make_tool_result("file.grep", "c1", "one match"));

        let report = compact_session(&mut session, &engine).unwrap();
        assert_eq!(report.messages_compacted, 0);
        assert!(!session.messages()[0].flags.contains(MessageFlags::IS_COMPACTED));
    }

    #[test]
    fn test_compact_session_report_counts() {
        let dir = tempfile::tempdir().unwrap();
        let engine = make_engine(dir.path());

        let mut session = Session::new(tc());
        let big: String = (0..500).map(|i| format!("line {i} content")).collect::<Vec<_>>().join("\n");
        session.push(make_tool_result("shell", "c1", &big));
        session.push(make_tool_result("file.grep", "c2", "small"));
        session.push(make_tool_result("shell", "c3", &big));

        let report = compact_session(&mut session, &engine).unwrap();
        // 2 big tool results compacted, 1 small not
        assert_eq!(report.messages_compacted, 2);
        assert_eq!(report.artifacts_created, 2);
    }

    #[test]
    fn test_compact_session_syncs_tool_result_output() {
        let dir = tempfile::tempdir().unwrap();
        let engine = make_engine(dir.path());

        let mut session = Session::new(tc());
        let big: String = (0..500).map(|i| format!("line {i} content")).collect::<Vec<_>>().join("\n");
        session.push(make_tool_result("shell", "c1", &big));

        compact_session(&mut session, &engine).unwrap();

        let msg = &session.messages()[0];
        let tr = msg.tool_result.as_ref().unwrap();
        // After compaction, tool_result.output must match content
        assert_eq!(tr.output, msg.content);
        // Both should contain the summary, not the original
        assert!(msg.content.contains("[Full output:"));
        assert!(tr.output.contains("[Full output:"));
    }

    #[test]
    fn test_compact_session_skips_preserved() {
        let dir = tempfile::tempdir().unwrap();
        let engine = make_engine(dir.path());

        let mut session = Session::new(tc());
        let big: String = (0..500).map(|i| format!("line {i} content")).collect::<Vec<_>>().join("\n");
        let mut msg = make_tool_result("shell", "c1", &big);
        msg.flags.insert(MessageFlags::PRESERVE);
        session.push(msg);

        let report = compact_session(&mut session, &engine).unwrap();
        assert_eq!(report.messages_compacted, 0);
    }
}
