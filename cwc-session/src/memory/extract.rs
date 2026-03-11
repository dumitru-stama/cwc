use crate::message::{MessageFlags, SessionMessage, SessionRole};
use crate::turn::Turn;

use super::types::{make_slug, now_millis, FactPriority, MemoryFact, MemorySource};

/// Tools whose output is too low-value to remember.
const NAVIGATIONAL_TOOLS: &[&str] = &["ls", "cd", "pwd", "which", "dir"];

/// Maximum characters for different extraction categories.
const MAX_CHARS_FINDING: usize = 256;
const MAX_CHARS_SHORT: usize = 128;
const MAX_CHARS_STATUS: usize = 256;
const MAX_CHARS_REVERSE: usize = 512;

/// Extract memory facts from a single message.
pub fn extract_from_message(msg: &SessionMessage, index: usize) -> Vec<MemoryFact> {
    let mut facts = Vec::new();

    match msg.role {
        SessionRole::Tool => {
            if let Some(ref tr) = msg.tool_result {
                if let Some(fact) = extract_from_tool_result(
                    &tr.tool_name,
                    &tr.call_id,
                    &msg.content,
                    &serde_json::Value::Null, // arguments not stored on tool result msg
                ) {
                    facts.push(fact);
                }
            }
        }
        SessionRole::Assistant => {
            if let Some(fact) = extract_conclusion(msg, index) {
                facts.push(fact);
            }
        }
        _ => {}
    }

    facts
}

/// Extract a memory fact from a tool result. Dispatches on tool name.
///
/// Returns `None` for navigational tools (ls, cd, pwd, which).
pub fn extract_from_tool_result(
    tool_name: &str,
    call_id: &str,
    output: &str,
    arguments: &serde_json::Value,
) -> Option<MemoryFact> {
    // Skip navigational tools
    if is_navigational(tool_name) {
        return None;
    }

    let source = MemorySource::ToolResult {
        tool_name: tool_name.to_string(),
        call_id: call_id.to_string(),
    };
    let ts = now_millis();

    // Dispatch based on tool name patterns
    if matches_any(tool_name, &["file.grep", "grep", "search"]) {
        Some(extract_grep_fact(tool_name, output, arguments, source, ts))
    } else if matches_any(tool_name, &["file.read", "read", "cat"]) {
        Some(extract_read_fact(output, arguments, source, ts))
    } else if matches_any(tool_name, &["file.write", "file.edit", "write", "edit"]) {
        Some(extract_write_fact(output, arguments, source, ts))
    } else if matches_any(tool_name, &["build", "cargo build", "make"]) {
        Some(extract_build_fact(output, source, ts))
    } else if matches_any(tool_name, &["test", "cargo test"]) || tool_name.starts_with("test.") {
        Some(extract_test_fact(tool_name, output, source, ts))
    } else if matches_any(tool_name, &["decompile", "disassemble"]) {
        Some(extract_reverse_fact(output, arguments, source, ts))
    } else if matches_any(tool_name, &["shell", "exec", "bash"]) {
        Some(extract_shell_fact(output, arguments, source, ts))
    } else {
        // Unknown tool — generic finding
        Some(extract_generic_fact(tool_name, output, source, ts))
    }
}

/// Extract the goal from the first real user message.
///
/// "Real" = SessionRole::User and not flagged IS_NUDGE or IS_MEMORY.
pub fn extract_goal(messages: &[SessionMessage]) -> Option<MemoryFact> {
    for (i, msg) in messages.iter().enumerate() {
        if msg.role == SessionRole::User
            && !msg.flags.contains(MessageFlags::IS_NUDGE)
            && !msg.flags.contains(MessageFlags::IS_MEMORY)
        {
            let trimmed = msg.content.trim();
            let goal_text = truncate_chars(trimmed, MAX_CHARS_FINDING);
            if goal_text.is_empty() {
                continue;
            }
            return Some(MemoryFact {
                key: "goal".to_string(),
                value: goal_text,
                source: MemorySource::UserMessage { message_index: i },
                created_at: now_millis(),
                priority: FactPriority::Goal,
            });
        }
    }
    None
}

/// Extract a conclusion from an assistant's message.
///
/// Only extracts from text-only assistant messages (no tool calls).
pub fn extract_conclusion(msg: &SessionMessage, index: usize) -> Option<MemoryFact> {
    if msg.role != SessionRole::Assistant {
        return None;
    }
    // Only text-only assistant messages (not tool call messages)
    if !msg.tool_calls.is_empty() {
        return None;
    }
    if msg.content.trim().is_empty() {
        return None;
    }
    Some(MemoryFact {
        key: "conclusion".to_string(),
        value: truncate_chars(&msg.content, MAX_CHARS_FINDING),
        source: MemorySource::AssistantConclusion {
            message_index: index,
        },
        created_at: now_millis(),
        priority: FactPriority::Finding,
    })
}

/// Extract all memory facts from messages in the given turns.
///
/// Deduplicates by key — later entries overwrite earlier ones.
/// Correlates tool_call arguments with tool_result messages by call_id.
pub fn extract_from_turns(
    messages: &[SessionMessage],
    turns: &[Turn],
) -> Vec<MemoryFact> {
    // Build a lookup of call_id → arguments from assistant tool_call messages
    let mut args_by_call_id: std::collections::HashMap<&str, &serde_json::Value> =
        std::collections::HashMap::new();
    for turn in turns {
        let start = turn.start;
        let end = (turn.end + 1).min(messages.len());
        for msg in messages.iter().take(end).skip(start) {
            if msg.role == SessionRole::Assistant {
                for tc in &msg.tool_calls {
                    args_by_call_id.insert(&tc.call_id, &tc.arguments);
                }
            }
        }
    }

    let mut facts_by_key: Vec<(String, MemoryFact)> = Vec::new();

    for turn in turns {
        let start = turn.start;
        let end = (turn.end + 1).min(messages.len());
        for (i, msg) in messages.iter().enumerate().take(end).skip(start) {
            let extracted = extract_from_message_with_args(msg, i, &args_by_call_id);
            for fact in extracted {
                // Overwrite existing key
                if let Some(pos) = facts_by_key.iter().position(|(k, _)| *k == fact.key) {
                    facts_by_key[pos] = (fact.key.clone(), fact);
                } else {
                    facts_by_key.push((fact.key.clone(), fact));
                }
            }
        }
    }

    facts_by_key.into_iter().map(|(_, f)| f).collect()
}

/// Like `extract_from_message` but with a call_id → arguments lookup for tool results.
fn extract_from_message_with_args(
    msg: &SessionMessage,
    index: usize,
    args_by_call_id: &std::collections::HashMap<&str, &serde_json::Value>,
) -> Vec<MemoryFact> {
    let mut facts = Vec::new();

    match msg.role {
        SessionRole::Tool => {
            if let Some(ref tr) = msg.tool_result {
                let arguments = args_by_call_id
                    .get(tr.call_id.as_str())
                    .copied()
                    .unwrap_or(&serde_json::Value::Null);
                if let Some(fact) = extract_from_tool_result(
                    &tr.tool_name,
                    &tr.call_id,
                    &msg.content,
                    arguments,
                ) {
                    facts.push(fact);
                }
            }
        }
        SessionRole::Assistant => {
            if let Some(fact) = extract_conclusion(msg, index) {
                facts.push(fact);
            }
        }
        _ => {}
    }

    facts
}

// --- Helper functions ---

fn is_navigational(tool_name: &str) -> bool {
    NAVIGATIONAL_TOOLS.contains(&tool_name)
}

fn matches_any(tool_name: &str, patterns: &[&str]) -> bool {
    patterns.contains(&tool_name)
}

fn truncate_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        text.to_string()
    } else {
        text.chars().take(max).collect()
    }
}

fn extract_grep_fact(
    _tool_name: &str,
    output: &str,
    arguments: &serde_json::Value,
    source: MemorySource,
    ts: u64,
) -> MemoryFact {
    let query = arguments
        .get("pattern")
        .or_else(|| arguments.get("query"))
        .or_else(|| arguments.get("q"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let query_slug = make_slug(query, 40);

    // Count matches and unique files, extract top 3
    let lines: Vec<&str> = output.lines().collect();
    let match_count = lines.len();
    // Count unique files (text before first colon in grep-style lines)
    let mut files = std::collections::HashSet::new();
    for line in &lines {
        if let Some(idx) = line.find(':') {
            files.insert(&line[..idx]);
        }
    }
    let file_count = files.len();
    let top3: Vec<&str> = lines.iter().take(3).copied().collect();
    let mut value = if file_count > 0 {
        format!("{match_count} matches in {file_count} files.")
    } else {
        format!("{match_count} matches.")
    };
    for line in &top3 {
        let preview: String = line.chars().take(80).collect();
        value.push_str(&format!(" {preview}"));
    }

    MemoryFact {
        key: format!("finding:grep_{query_slug}"),
        value: truncate_chars(&value, MAX_CHARS_FINDING),
        source,
        created_at: ts,
        priority: FactPriority::Finding,
    }
}

fn extract_read_fact(
    output: &str,
    arguments: &serde_json::Value,
    source: MemorySource,
    ts: u64,
) -> MemoryFact {
    let path = arguments
        .get("path")
        .or_else(|| arguments.get("file"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    let line_count = output.lines().count();
    let value = format!("read {path} ({line_count} lines)");

    MemoryFact {
        key: format!("file_seen:{}", make_slug(path, 40)),
        value: truncate_chars(&value, MAX_CHARS_SHORT),
        source,
        created_at: ts,
        priority: FactPriority::Navigation,
    }
}

fn extract_write_fact(
    output: &str,
    arguments: &serde_json::Value,
    source: MemorySource,
    ts: u64,
) -> MemoryFact {
    let path = arguments
        .get("path")
        .or_else(|| arguments.get("file"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    let value = format!("modified {path}: {}", truncate_chars(output, 80));

    MemoryFact {
        key: format!("file_modified:{}", make_slug(path, 40)),
        value: truncate_chars(&value, MAX_CHARS_SHORT),
        source,
        created_at: ts,
        priority: FactPriority::Finding,
    }
}

fn extract_build_fact(
    output: &str,
    source: MemorySource,
    ts: u64,
) -> MemoryFact {
    // Check for errors
    let has_error = output.lines().any(|l| {
        let t = l.trim();
        t.starts_with("error") || t.contains("FAILED")
    });

    let value = if has_error {
        let first_error = output
            .lines()
            .find(|l| l.trim().starts_with("error"))
            .unwrap_or("unknown error");
        format!("error: {}", truncate_chars(first_error.trim(), 200))
    } else {
        "success".to_string()
    };

    MemoryFact {
        key: "build_result".to_string(),
        value: truncate_chars(&value, MAX_CHARS_STATUS),
        source,
        created_at: ts,
        priority: FactPriority::Status,
    }
}

fn extract_test_fact(
    tool_name: &str,
    output: &str,
    source: MemorySource,
    ts: u64,
) -> MemoryFact {
    // Extract suite name from tool name or use "default"
    let suite = if tool_name.contains('.') {
        tool_name.split('.').next_back().unwrap_or("default")
    } else {
        "default"
    };

    // Count passed/failed from Rust test output
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut first_failure: Option<String> = None;

    for line in output.lines() {
        let t = line.trim();
        if t.starts_with("test ") && t.contains(" ... ") {
            if t.ends_with("ok") {
                passed += 1;
            } else if t.contains("FAILED") {
                failed += 1;
                if first_failure.is_none() {
                    // Extract test name
                    if let Some(rest) = t.strip_prefix("test ") {
                        if let Some(idx) = rest.find(" ... ") {
                            first_failure = Some(rest[..idx].to_string());
                        }
                    }
                }
            }
        }
    }

    let mut value = format!("{passed} passed, {failed} failed");
    if let Some(ref fail) = first_failure {
        value.push_str(&format!(": {fail}"));
    }

    MemoryFact {
        key: format!("test_result:{}", make_slug(suite, 40)),
        value: truncate_chars(&value, MAX_CHARS_STATUS),
        source,
        created_at: ts,
        priority: FactPriority::Status,
    }
}

fn extract_reverse_fact(
    output: &str,
    arguments: &serde_json::Value,
    source: MemorySource,
    ts: u64,
) -> MemoryFact {
    let fn_name = arguments
        .get("function")
        .or_else(|| arguments.get("name"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    // Take signature + first 3 lines
    let lines: Vec<&str> = output.lines().take(4).collect();
    let value = lines.join("\n");

    MemoryFact {
        key: format!("finding:reverse_{}", make_slug(fn_name, 40)),
        value: truncate_chars(&value, MAX_CHARS_REVERSE),
        source,
        created_at: ts,
        priority: FactPriority::Finding,
    }
}

fn extract_shell_fact(
    output: &str,
    arguments: &serde_json::Value,
    source: MemorySource,
    ts: u64,
) -> MemoryFact {
    let cmd = arguments
        .get("command")
        .or_else(|| arguments.get("cmd"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let cmd_slug = make_slug(cmd, 40);

    let first_3: Vec<&str> = output.lines().take(3).collect();
    let value = first_3.join("\n");

    MemoryFact {
        key: format!("cmd_result:{cmd_slug}"),
        value: truncate_chars(&value, MAX_CHARS_FINDING),
        source,
        created_at: ts,
        priority: FactPriority::Finding,
    }
}

fn extract_generic_fact(
    tool_name: &str,
    output: &str,
    source: MemorySource,
    ts: u64,
) -> MemoryFact {
    MemoryFact {
        key: format!("finding:{}", make_slug(tool_name, 40)),
        value: truncate_chars(output, MAX_CHARS_FINDING),
        source,
        created_at: ts,
        priority: FactPriority::Finding,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{ToolCall, ToolResult};

    #[test]
    fn test_extract_grep_output() {
        let output = "src/main.rs:10: malloc(256)\nsrc/alloc.rs:20: malloc(128)\nsrc/parser.rs:45: malloc(64)";
        let args = serde_json::json!({"pattern": "malloc"});
        let fact = extract_from_tool_result("file.grep", "c1", output, &args).unwrap();
        assert_eq!(fact.key, "finding:grep_malloc");
        assert!(fact.value.contains("3 matches in 3 files"));
        assert!(fact.value.contains("src/main.rs:10"));
        assert_eq!(fact.priority, FactPriority::Finding);
    }

    #[test]
    fn test_extract_file_read() {
        let output = "line1\nline2\nline3";
        let args = serde_json::json!({"path": "src/parser.rs"});
        let fact = extract_from_tool_result("file.read", "c1", output, &args).unwrap();
        assert_eq!(fact.key, "file_seen:src_parser_rs");
        assert!(fact.value.contains("src/parser.rs"));
        assert!(fact.value.contains("3 lines"));
        assert_eq!(fact.priority, FactPriority::Navigation);
    }

    #[test]
    fn test_extract_file_write() {
        let output = "File written successfully";
        let args = serde_json::json!({"path": "src/main.rs"});
        let fact = extract_from_tool_result("file.write", "c1", output, &args).unwrap();
        assert_eq!(fact.key, "file_modified:src_main_rs");
        assert!(fact.value.contains("modified src/main.rs"));
        assert_eq!(fact.priority, FactPriority::Finding);
    }

    #[test]
    fn test_extract_build_success() {
        let output = "  Compiling foo v0.1.0\n  Finished dev [unoptimized] in 1.2s";
        let fact = extract_from_tool_result("build", "c1", output, &serde_json::Value::Null).unwrap();
        assert_eq!(fact.key, "build_result");
        assert_eq!(fact.value, "success");
        assert_eq!(fact.priority, FactPriority::Status);
    }

    #[test]
    fn test_extract_build_failure() {
        let output = "  Compiling foo v0.1.0\nerror[E0308]: mismatched types\n  --> src/main.rs:5:5";
        let fact = extract_from_tool_result("build", "c1", output, &serde_json::Value::Null).unwrap();
        assert_eq!(fact.key, "build_result");
        assert!(fact.value.starts_with("error:"));
        assert!(fact.value.contains("E0308"));
        assert_eq!(fact.priority, FactPriority::Status);
    }

    #[test]
    fn test_extract_test_run() {
        let output = "test test_one ... ok\ntest test_two ... ok\ntest test_three ... FAILED\ntest result: FAILED. 2 passed; 1 failed";
        let fact = extract_from_tool_result("test", "c1", output, &serde_json::Value::Null).unwrap();
        assert_eq!(fact.key, "test_result:default");
        assert!(fact.value.contains("2 passed"));
        assert!(fact.value.contains("1 failed"));
        assert!(fact.value.contains("test_three"));
        assert_eq!(fact.priority, FactPriority::Status);
    }

    #[test]
    fn test_extract_test_with_suite() {
        let output = "test test_a ... ok";
        let fact = extract_from_tool_result("test.integration", "c1", output, &serde_json::Value::Null).unwrap();
        assert_eq!(fact.key, "test_result:integration");
    }

    #[test]
    fn test_extract_navigational_skip() {
        assert!(extract_from_tool_result("ls", "c1", "file1\nfile2", &serde_json::Value::Null).is_none());
        assert!(extract_from_tool_result("cd", "c1", "/home", &serde_json::Value::Null).is_none());
        assert!(extract_from_tool_result("pwd", "c1", "/home/user", &serde_json::Value::Null).is_none());
        assert!(extract_from_tool_result("which", "c1", "/usr/bin/cargo", &serde_json::Value::Null).is_none());
    }

    #[test]
    fn test_extract_unknown_tool() {
        let output = "some interesting output from a custom tool";
        let fact = extract_from_tool_result("custom.analyze", "c1", output, &serde_json::Value::Null).unwrap();
        assert_eq!(fact.key, "finding:custom_analyze");
        assert_eq!(fact.value, output);
        assert_eq!(fact.priority, FactPriority::Finding);
    }

    #[test]
    fn test_extract_unknown_tool_truncated() {
        let output = "x".repeat(500);
        let fact = extract_from_tool_result("custom.tool", "c1", &output, &serde_json::Value::Null).unwrap();
        assert!(fact.value.len() <= MAX_CHARS_FINDING);
    }

    #[test]
    fn test_extract_goal_first_user_message() {
        let messages = vec![
            SessionMessage::system("system prompt"),
            SessionMessage::text(SessionRole::User, "Find all malloc calls and fix the leaks"),
            SessionMessage::text(SessionRole::Assistant, "I'll help with that"),
        ];
        let goal = extract_goal(&messages).unwrap();
        assert_eq!(goal.key, "goal");
        assert_eq!(goal.value, "Find all malloc calls and fix the leaks");
        assert_eq!(goal.priority, FactPriority::Goal);
    }

    #[test]
    fn test_extract_goal_skips_nudge_and_system() {
        let messages = vec![
            SessionMessage::system("system prompt"),
            SessionMessage::nudge("continue working"),
            SessionMessage::text(SessionRole::User, "the real request"),
        ];
        let goal = extract_goal(&messages).unwrap();
        assert_eq!(goal.value, "the real request");
    }

    #[test]
    fn test_extract_goal_truncates() {
        let long_msg = "x".repeat(500);
        let messages = vec![
            SessionMessage::text(SessionRole::User, &long_msg),
        ];
        let goal = extract_goal(&messages).unwrap();
        assert!(goal.value.len() <= MAX_CHARS_FINDING);
    }

    #[test]
    fn test_extract_goal_empty_conversation() {
        let messages: Vec<SessionMessage> = vec![];
        assert!(extract_goal(&messages).is_none());
    }

    #[test]
    fn test_extract_goal_no_user_messages() {
        let messages = vec![
            SessionMessage::system("system prompt"),
            SessionMessage::text(SessionRole::Assistant, "hello"),
        ];
        assert!(extract_goal(&messages).is_none());
    }

    #[test]
    fn test_extract_conclusion() {
        let msg = SessionMessage::text(SessionRole::Assistant, "The analysis shows 3 memory leaks in parser.rs");
        let fact = extract_conclusion(&msg, 5).unwrap();
        assert_eq!(fact.key, "conclusion");
        assert!(fact.value.contains("3 memory leaks"));
        assert_eq!(fact.priority, FactPriority::Finding);
        if let MemorySource::AssistantConclusion { message_index } = fact.source {
            assert_eq!(message_index, 5);
        } else {
            panic!("expected AssistantConclusion source");
        }
    }

    #[test]
    fn test_extract_conclusion_skips_tool_call_messages() {
        let msg = SessionMessage::assistant_tool_calls(
            "Let me check",
            vec![ToolCall {
                call_id: "c1".into(),
                tool_name: "grep".into(),
                arguments: serde_json::json!({}),
            }],
        );
        assert!(extract_conclusion(&msg, 0).is_none());
    }

    #[test]
    fn test_extract_conclusion_skips_empty() {
        let msg = SessionMessage::text(SessionRole::Assistant, "");
        assert!(extract_conclusion(&msg, 0).is_none());

        let msg2 = SessionMessage::text(SessionRole::Assistant, "   ");
        assert!(extract_conclusion(&msg2, 0).is_none());
    }

    #[test]
    fn test_extract_from_turns_deduplicates() {
        let messages = vec![
            SessionMessage::tool_result(ToolResult {
                call_id: "c1".into(),
                tool_name: "build".into(),
                output: "error[E0308]: mismatched types".into(),
                is_error: false,
            }),
            SessionMessage::text(SessionRole::Assistant, "fixing..."),
            SessionMessage::tool_result(ToolResult {
                call_id: "c2".into(),
                tool_name: "build".into(),
                output: "  Compiling foo\n  Finished dev".into(),
                is_error: false,
            }),
        ];
        let turns = vec![
            Turn { start: 0, end: 1, tokens: 100, has_user_request: false, has_tool_calls: true, message_count: 2 },
            Turn { start: 2, end: 2, tokens: 50, has_user_request: false, has_tool_calls: true, message_count: 1 },
        ];

        let facts = extract_from_turns(&messages, &turns);
        // "build_result" key should appear only once (the later success overwrites the error)
        let build_facts: Vec<_> = facts.iter().filter(|f| f.key == "build_result").collect();
        assert_eq!(build_facts.len(), 1);
        assert_eq!(build_facts[0].value, "success");
    }

    #[test]
    fn test_extract_from_turns_later_overwrites_earlier() {
        let messages = vec![
            SessionMessage::tool_result(ToolResult {
                call_id: "c1".into(),
                tool_name: "build".into(),
                output: "error[E0001]: first error".into(),
                is_error: false,
            }),
            SessionMessage::tool_result(ToolResult {
                call_id: "c2".into(),
                tool_name: "build".into(),
                output: "error[E0002]: second error".into(),
                is_error: false,
            }),
        ];
        let turns = vec![
            Turn { start: 0, end: 1, tokens: 100, has_user_request: false, has_tool_calls: true, message_count: 2 },
        ];

        let facts = extract_from_turns(&messages, &turns);
        let build = facts.iter().find(|f| f.key == "build_result").unwrap();
        assert!(build.value.contains("E0002"), "later build should overwrite: {}", build.value);
    }

    #[test]
    fn test_extract_shell_fact() {
        let output = "total 42\ndrwxr-xr-x  5 user user  4096 Jan  1 00:00 src\n-rw-r--r--  1 user user   256 Jan  1 00:00 Cargo.toml\nmore lines";
        let args = serde_json::json!({"command": "ls -la"});
        let fact = extract_from_tool_result("shell", "c1", output, &args).unwrap();
        assert_eq!(fact.key, "cmd_result:ls_la");
        // Should have first 3 lines
        assert!(fact.value.contains("total 42"));
        assert!(fact.value.contains("src"));
        assert!(fact.value.contains("Cargo.toml"));
        // Should NOT have 4th line
        assert!(!fact.value.contains("more lines"));
    }

    #[test]
    fn test_extract_from_message_tool_role() {
        let msg = SessionMessage::tool_result(ToolResult {
            call_id: "c1".into(),
            tool_name: "build".into(),
            output: "  Compiling foo\n  Finished".into(),
            is_error: false,
        });
        let facts = extract_from_message(&msg, 3);
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].key, "build_result");
    }

    #[test]
    fn test_extract_from_message_user_role_no_facts() {
        let msg = SessionMessage::text(SessionRole::User, "hello");
        let facts = extract_from_message(&msg, 0);
        assert!(facts.is_empty());
    }

    #[test]
    fn test_extract_decompile_fact() {
        let output = "int main(int argc, char **argv) {\n  int x = 0;\n  printf(\"hello\");\n  return x;\n}";
        let args = serde_json::json!({"function": "main"});
        let fact = extract_from_tool_result("decompile", "c1", output, &args).unwrap();
        assert_eq!(fact.key, "finding:reverse_main");
        assert!(fact.value.contains("int main"));
        assert_eq!(fact.priority, FactPriority::Finding);
    }

    #[test]
    fn test_key_overwrite_same_grep_query() {
        // Simulating running grep twice with same query
        let args = serde_json::json!({"pattern": "malloc"});
        let fact1 = extract_from_tool_result("file.grep", "c1", "file.rs:1: malloc(10)", &args).unwrap();
        let fact2 = extract_from_tool_result("file.grep", "c2", "file.rs:1: malloc(10)\nfile.rs:2: malloc(20)", &args).unwrap();
        // Same key
        assert_eq!(fact1.key, fact2.key);
        // But different values
        assert!(fact1.value.contains("1 match"));
        assert!(fact2.value.contains("2 match"));
    }

    #[test]
    fn test_extract_goal_trims_whitespace() {
        let messages = vec![
            SessionMessage::text(SessionRole::User, "  \n  Find the bug  \n  "),
        ];
        let goal = extract_goal(&messages).unwrap();
        assert_eq!(goal.value, "Find the bug");
    }

    #[test]
    fn test_extract_goal_skips_whitespace_only_user() {
        let messages = vec![
            SessionMessage::text(SessionRole::User, "   \n\t  "),
            SessionMessage::text(SessionRole::User, "real request"),
        ];
        let goal = extract_goal(&messages).unwrap();
        assert_eq!(goal.value, "real request");
    }

    #[test]
    fn test_extract_goal_skips_memory_flagged() {
        let mut mem_msg = SessionMessage::text(SessionRole::User, "[WORKING_MEMORY] old context");
        mem_msg.flags.insert(MessageFlags::IS_MEMORY);
        let messages = vec![
            mem_msg,
            SessionMessage::text(SessionRole::User, "actual question"),
        ];
        let goal = extract_goal(&messages).unwrap();
        assert_eq!(goal.value, "actual question");
    }

    #[test]
    fn test_extract_from_turns_passes_arguments() {
        // Assistant makes a tool call with arguments, then we get the tool result
        let messages = vec![
            SessionMessage::assistant_tool_calls(
                "Let me search",
                vec![ToolCall {
                    call_id: "c1".into(),
                    tool_name: "file.grep".into(),
                    arguments: serde_json::json!({"pattern": "malloc"}),
                }],
            ),
            SessionMessage::tool_result(ToolResult {
                call_id: "c1".into(),
                tool_name: "file.grep".into(),
                output: "src/main.rs:10: malloc(256)\nsrc/alloc.rs:20: malloc(128)".into(),
                is_error: false,
            }),
        ];
        let turns = vec![
            Turn { start: 0, end: 1, tokens: 100, has_user_request: false, has_tool_calls: true, message_count: 2 },
        ];

        let facts = extract_from_turns(&messages, &turns);
        let grep_fact = facts.iter().find(|f| f.key.starts_with("finding:grep_")).unwrap();
        // Should have used the "malloc" pattern from arguments
        assert_eq!(grep_fact.key, "finding:grep_malloc");
        assert!(grep_fact.value.contains("2 matches in 2 files"));
    }

    #[test]
    fn test_grep_no_colon_lines() {
        // Output without grep-style file:line format
        let output = "match 1\nmatch 2";
        let args = serde_json::json!({"pattern": "something"});
        let fact = extract_from_tool_result("file.grep", "c1", output, &args).unwrap();
        // 0 files since no lines contain ':'
        assert!(fact.value.starts_with("2 matches."));
    }

    #[test]
    fn test_grep_same_file_multiple_lines() {
        let output = "src/main.rs:10: malloc(1)\nsrc/main.rs:20: malloc(2)\nsrc/main.rs:30: malloc(3)";
        let args = serde_json::json!({"pattern": "malloc"});
        let fact = extract_from_tool_result("file.grep", "c1", output, &args).unwrap();
        // 3 matches but only 1 unique file
        assert!(fact.value.contains("3 matches in 1 files"), "got: {}", fact.value);
    }

    #[test]
    fn test_extract_test_all_passing() {
        let output = "test test_one ... ok\ntest test_two ... ok\ntest result: ok. 2 passed; 0 failed";
        let fact = extract_from_tool_result("test", "c1", output, &serde_json::Value::Null).unwrap();
        assert!(fact.value.contains("2 passed"));
        assert!(fact.value.contains("0 failed"));
        // No first_failure suffix
        assert!(!fact.value.contains(':'));
    }
}
