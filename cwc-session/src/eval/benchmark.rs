use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Result, SessionError};
use crate::message::{SessionMessage, SessionRole, ToolCall, ToolResult};

/// A benchmark conversation with ground-truth annotations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConversation {
    pub name: String,
    pub description: String,
    pub messages: Vec<SessionMessage>,
    /// Ground-truth facts that should be in memory after processing.
    pub expected_facts: Vec<ExpectedFact>,
    /// Turn indices where hallucination is injected.
    pub hallucination_turns: Vec<usize>,
    /// Pairs of turn indices with identical tool calls.
    pub duplicate_turns: Vec<(usize, usize)>,
    /// The original goal text.
    pub goal: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedFact {
    pub key_pattern: String,
    pub value_contains: String,
}

/// 10-turn simple bug fix conversation.
pub fn benchmark_short_coding() -> BenchmarkConversation {
    let mut messages = vec![
        SessionMessage::system("You are a code reviewer. Find and fix bugs."),
        SessionMessage::text(SessionRole::User, "Find the null pointer bug in parser.rs"),
    ];

    // Turn 1: grep for null
    add_tool_turn(
        &mut messages,
        "c1",
        "file.grep",
        serde_json::json!({"pattern": "unwrap()", "path": "src/parser.rs"}),
        "src/parser.rs:42: let val = map.get(key).unwrap();\nsrc/parser.rs:87: let node = tree.root().unwrap();",
    );
    messages.push(SessionMessage::text(
        SessionRole::Assistant,
        "Found two unwrap() calls. Let me read the context around line 42.",
    ));

    // Turn 2: read the file
    add_tool_turn(
        &mut messages,
        "c2",
        "file.read",
        serde_json::json!({"path": "src/parser.rs", "line": 40, "count": 10}),
        "40: fn parse_value(map: &HashMap<String, Value>, key: &str) -> Value {\n41:     // BUG: panics if key is missing\n42:     let val = map.get(key).unwrap();\n43:     val.clone()\n44: }",
    );
    messages.push(SessionMessage::text(
        SessionRole::Assistant,
        "Found the bug at line 42. `map.get(key).unwrap()` panics when key is missing. Should use `get(key).cloned().unwrap_or_default()`.",
    ));

    // Turn 3: check the other unwrap
    add_tool_turn(
        &mut messages,
        "c3",
        "file.read",
        serde_json::json!({"path": "src/parser.rs", "line": 85, "count": 10}),
        "85: fn get_root(tree: &Tree) -> Node {\n86:     // This unwrap is safe: tree always has a root\n87:     let node = tree.root().unwrap();\n88:     node\n89: }",
    );
    messages.push(SessionMessage::text(
        SessionRole::Assistant,
        "The unwrap at line 87 is safe — trees always have a root. Only the line 42 bug needs fixing.",
    ));

    BenchmarkConversation {
        name: "short_coding".into(),
        description: "10-turn simple bug fix: find and analyze null pointer bug".into(),
        messages,
        expected_facts: vec![
            ExpectedFact {
                key_pattern: "goal".into(),
                value_contains: "null pointer".into(),
            },
            ExpectedFact {
                key_pattern: "finding:*".into(),
                value_contains: "unwrap".into(),
            },
        ],
        hallucination_turns: vec![],
        duplicate_turns: vec![],
        goal: "Find the null pointer bug in parser.rs".into(),
    }
}

/// 40-turn codebase exploration conversation.
pub fn benchmark_long_exploration() -> BenchmarkConversation {
    let mut messages = vec![
        SessionMessage::system("You are a senior engineer analyzing a codebase."),
        SessionMessage::text(
            SessionRole::User,
            "Analyze the architecture of this Rust project and identify potential issues",
        ),
    ];

    let files = [
        ("src/main.rs", "mod server;\nmod config;\nmod db;\n\nfn main() { server::run(); }"),
        ("src/server.rs", "use crate::db;\npub fn run() { let pool = db::connect(); handle_requests(pool); }"),
        ("src/config.rs", "pub struct Config { pub db_url: String, pub port: u16 }\nimpl Config { pub fn load() -> Self { todo!() } }"),
        ("src/db.rs", "pub fn connect() -> Pool { Pool::new(\"hardcoded_url\") }"),
    ];

    for (i, (path, content)) in files.iter().enumerate() {
        let idx = i + 1;
        add_tool_turn(
            &mut messages,
            &format!("c{idx}"),
            "file.read",
            serde_json::json!({"path": path}),
            content,
        );
        messages.push(SessionMessage::text(
            SessionRole::Assistant,
            format!("Read {path}. Continuing analysis."),
        ));
    }

    // Add more exploration turns with grep
    let patterns = [
        ("todo!()", "src/config.rs:3: pub fn load() -> Self { todo!() }"),
        ("unwrap()", "src/server.rs:5: pool.get().unwrap()"),
        ("hardcoded", "src/db.rs:1: Pool::new(\"hardcoded_url\")"),
        ("panic", "No results found"),
        ("unsafe", "No results found"),
        ("clone()", "src/main.rs:10: let cfg = config.clone();"),
    ];

    for (i, (pat, result)) in patterns.iter().enumerate() {
        let idx = i + 5;
        add_tool_turn(
            &mut messages,
            &format!("c{idx}"),
            "file.grep",
            serde_json::json!({"pattern": pat}),
            result,
        );
        messages.push(SessionMessage::text(
            SessionRole::Assistant,
            format!("Searched for '{pat}'. Noted findings."),
        ));
    }

    // Fill remaining turns with deeper analysis
    for i in 11..20 {
        add_tool_turn(
            &mut messages,
            &format!("c{i}"),
            "file.read",
            serde_json::json!({"path": format!("src/module_{}.rs", i - 10)}),
            &format!("// Module {} content\nfn process() {{ /* implementation */ }}", i - 10),
        );
        messages.push(SessionMessage::text(
            SessionRole::Assistant,
            format!("Analyzed module {}. No critical issues.", i - 10),
        ));
    }

    BenchmarkConversation {
        name: "long_exploration".into(),
        description: "40-turn codebase analysis with multiple file reads and greps".into(),
        messages,
        expected_facts: vec![
            ExpectedFact {
                key_pattern: "goal".into(),
                value_contains: "architecture".into(),
            },
            ExpectedFact {
                key_pattern: "finding:*".into(),
                value_contains: "hardcoded".into(),
            },
            ExpectedFact {
                key_pattern: "finding:*".into(),
                value_contains: "todo".into(),
            },
        ],
        hallucination_turns: vec![],
        duplicate_turns: vec![],
        goal: "Analyze the architecture of this Rust project and identify potential issues".into(),
    }
}

/// 30-turn tool-heavy conversation.
pub fn benchmark_tool_heavy() -> BenchmarkConversation {
    let mut messages = vec![
        SessionMessage::system("You are a CI/CD engineer fixing build failures."),
        SessionMessage::text(SessionRole::User, "Fix the failing build in the main branch"),
    ];

    // Alternate grep → read → build cycles
    for cycle in 0..10 {
        let base = cycle * 3;

        add_tool_turn(
            &mut messages,
            &format!("c{}", base + 1),
            "file.grep",
            serde_json::json!({"pattern": format!("error_{cycle}"), "path": "src/"}),
            &format!("src/lib.rs:{}: type error_{cycle}", 10 + cycle * 5),
        );

        add_tool_turn(
            &mut messages,
            &format!("c{}", base + 2),
            "file.read",
            serde_json::json!({"path": "src/lib.rs", "line": 10 + cycle * 5}),
            &format!("fn broken_{cycle}() -> u32 {{ \"not_a_number\" }}"),
        );

        add_tool_turn(
            &mut messages,
            &format!("c{}", base + 3),
            "shell",
            serde_json::json!({"command": "cargo build"}),
            &format!("error[E0308]: mismatched types in broken_{cycle}"),
        );

        messages.push(SessionMessage::text(
            SessionRole::Assistant,
            format!("Found and analyzed error_{cycle}. The function broken_{cycle} returns a string instead of u32."),
        ));
    }

    BenchmarkConversation {
        name: "tool_heavy".into(),
        description: "30-turn build fix with many grep/read/build cycles".into(),
        messages,
        expected_facts: vec![
            ExpectedFact {
                key_pattern: "goal".into(),
                value_contains: "build".into(),
            },
        ],
        hallucination_turns: vec![],
        duplicate_turns: vec![],
        goal: "Fix the failing build in the main branch".into(),
    }
}

/// 60-turn conversation designed to trigger all 3 layers.
pub fn benchmark_context_pressure() -> BenchmarkConversation {
    let mut messages = vec![
        SessionMessage::system("You are analyzing a large codebase for security vulnerabilities."),
        SessionMessage::text(
            SessionRole::User,
            "Audit all files in src/ for SQL injection, XSS, and command injection",
        ),
    ];

    // Generate 25 heavy tool turns (each with large output) to push context
    for i in 0..25 {
        let big_output: String = (0..100)
            .map(|j| format!("src/handler_{i}.rs:{j}: let query = format!(\"SELECT * FROM users WHERE id = {{}}\", input);"))
            .collect::<Vec<_>>()
            .join("\n");

        add_tool_turn(
            &mut messages,
            &format!("c{}", i + 1),
            "file.grep",
            serde_json::json!({"pattern": "format!(\"SELECT", "path": format!("src/handler_{i}.rs")}),
            &big_output,
        );

        // Set high token count to simulate pressure
        if let Some(tr) = messages.iter_mut().rev().find(|m| m.role == SessionRole::Tool) {
            tr.token_count = 500;
        }

        messages.push(SessionMessage::text(
            SessionRole::Assistant,
            format!("Found potential SQL injection in handler_{i}.rs. Continuing audit."),
        ));
    }

    BenchmarkConversation {
        name: "context_pressure".into(),
        description: "60-turn security audit designed to trigger sliding window, hard reset, and compaction".into(),
        messages,
        expected_facts: vec![
            ExpectedFact {
                key_pattern: "goal".into(),
                value_contains: "SQL injection".into(),
            },
        ],
        hallucination_turns: vec![],
        duplicate_turns: vec![],
        goal: "Audit all files in src/ for SQL injection, XSS, and command injection".into(),
    }
}

/// Conversation with planted hallucinations for detection testing.
pub fn benchmark_hallucination() -> BenchmarkConversation {
    let mut messages = vec![
        SessionMessage::system("You are a precise code reviewer. Only report facts you verify."),
        SessionMessage::text(SessionRole::User, "Find memory leaks in allocator.rs"),
    ];

    // Normal turns
    add_tool_turn(
        &mut messages,
        "c1",
        "file.read",
        serde_json::json!({"path": "src/allocator.rs"}),
        "fn allocate(size: usize) -> *mut u8 {\n    let ptr = unsafe { alloc(Layout::from_size_align(size, 8).unwrap()) };\n    ptr\n}\nfn deallocate(ptr: *mut u8, size: usize) {\n    unsafe { dealloc(ptr, Layout::from_size_align(size, 8).unwrap()) };\n}",
    );
    messages.push(SessionMessage::text(
        SessionRole::Assistant,
        "Read allocator.rs. The allocate/deallocate pair looks correct, but let me check for missing deallocations.",
    ));

    add_tool_turn(
        &mut messages,
        "c2",
        "file.grep",
        serde_json::json!({"pattern": "allocate(", "path": "src/"}),
        "src/allocator.rs:1: fn allocate\nsrc/main.rs:15: let p = allocate(1024);\nsrc/main.rs:20: let q = allocate(2048);",
    );

    // Turn 3: hallucination — model claims to find a leak that doesn't exist
    messages.push(SessionMessage::text(
        SessionRole::Assistant,
        "Found a critical memory leak! In main.rs line 25, there's a `allocate(4096)` call that is never freed. This is a severe vulnerability.",
    ));

    // More normal work
    add_tool_turn(
        &mut messages,
        "c3",
        "file.read",
        serde_json::json!({"path": "src/main.rs", "line": 14, "count": 15}),
        "14: fn process() {\n15:     let p = allocate(1024);\n16:     use_buffer(p, 1024);\n17:     deallocate(p, 1024);\n18:\n19:     let q = allocate(2048);\n20:     use_buffer(q, 2048);\n21:     deallocate(q, 2048);\n22: }",
    );
    messages.push(SessionMessage::text(
        SessionRole::Assistant,
        "After reading the actual code, both allocations at lines 15 and 19 are properly freed. No memory leaks found.",
    ));

    BenchmarkConversation {
        name: "hallucination".into(),
        description: "Includes planted hallucination at turn 3 (false claim about line 25)".into(),
        messages,
        expected_facts: vec![
            ExpectedFact {
                key_pattern: "goal".into(),
                value_contains: "memory leak".into(),
            },
        ],
        hallucination_turns: vec![3], // Turn index 3 (0-based from messages after system+user)
        duplicate_turns: vec![],
        goal: "Find memory leaks in allocator.rs".into(),
    }
}

/// Conversation where model loops on same file reads.
pub fn benchmark_repetition() -> BenchmarkConversation {
    let mut messages = vec![
        SessionMessage::system("You are debugging a performance issue."),
        SessionMessage::text(SessionRole::User, "Find why the server is slow"),
    ];

    // First read of main.rs
    add_tool_turn(
        &mut messages,
        "c1",
        "file.read",
        serde_json::json!({"path": "src/main.rs"}),
        "fn main() { let server = Server::new(); server.run(); }",
    );
    messages.push(SessionMessage::text(
        SessionRole::Assistant,
        "Read main.rs. Let me check the server module.",
    ));

    // Read server.rs
    add_tool_turn(
        &mut messages,
        "c2",
        "file.read",
        serde_json::json!({"path": "src/server.rs"}),
        "impl Server { fn run(&self) { loop { self.handle_request(); } } }",
    );
    messages.push(SessionMessage::text(
        SessionRole::Assistant,
        "Let me re-read main.rs for more context.",
    ));

    // DUPLICATE: re-read main.rs with same args
    add_tool_turn(
        &mut messages,
        "c3",
        "file.read",
        serde_json::json!({"path": "src/main.rs"}),
        "fn main() { let server = Server::new(); server.run(); }",
    );
    messages.push(SessionMessage::text(
        SessionRole::Assistant,
        "Let me grep for slow operations.",
    ));

    // Grep
    add_tool_turn(
        &mut messages,
        "c4",
        "file.grep",
        serde_json::json!({"pattern": "sleep|block|wait"}),
        "src/server.rs:10: std::thread::sleep(Duration::from_secs(1));",
    );
    messages.push(SessionMessage::text(
        SessionRole::Assistant,
        "Found it! There's a `thread::sleep(1s)` in server.rs. Let me re-check.",
    ));

    // DUPLICATE: same grep again
    add_tool_turn(
        &mut messages,
        "c5",
        "file.grep",
        serde_json::json!({"pattern": "sleep|block|wait"}),
        "src/server.rs:10: std::thread::sleep(Duration::from_secs(1));",
    );
    messages.push(SessionMessage::text(
        SessionRole::Assistant,
        "Confirmed: the performance bottleneck is the sleep(1s) in the request handler.",
    ));

    BenchmarkConversation {
        name: "repetition".into(),
        description: "Model loops on same file reads and grep calls".into(),
        messages,
        expected_facts: vec![
            ExpectedFact {
                key_pattern: "goal".into(),
                value_contains: "slow".into(),
            },
        ],
        hallucination_turns: vec![],
        duplicate_turns: vec![(0, 2), (3, 4)], // main.rs read, grep dups
        goal: "Find why the server is slow".into(),
    }
}

/// Load a benchmark from a JSON file.
pub fn load_benchmark(path: &Path) -> Result<BenchmarkConversation> {
    let content = std::fs::read_to_string(path)?;
    serde_json::from_str(&content).map_err(|e| {
        SessionError::InvalidFormat(format!("failed to parse benchmark {}: {e}", path.display()))
    })
}

/// Helper: add a tool_call + tool_result pair to messages.
fn add_tool_turn(
    messages: &mut Vec<SessionMessage>,
    call_id: &str,
    tool_name: &str,
    arguments: serde_json::Value,
    output: &str,
) {
    messages.push(SessionMessage::assistant_tool_calls(
        "",
        vec![ToolCall {
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            arguments,
        }],
    ));
    messages.push(SessionMessage::tool_result(ToolResult {
        call_id: call_id.into(),
        tool_name: tool_name.into(),
        output: output.into(),
        is_error: false,
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_short_coding() {
        let bench = benchmark_short_coding();
        assert_eq!(bench.name, "short_coding");
        assert!(bench.messages.len() >= 8); // system + user + 3*(call+result+assistant)
        assert!(!bench.expected_facts.is_empty());
        assert!(!bench.goal.is_empty());
    }

    #[test]
    fn test_benchmark_long_exploration() {
        let bench = benchmark_long_exploration();
        assert_eq!(bench.name, "long_exploration");
        assert!(bench.messages.len() >= 30);
        assert!(!bench.expected_facts.is_empty());
    }

    #[test]
    fn test_benchmark_tool_heavy() {
        let bench = benchmark_tool_heavy();
        assert_eq!(bench.name, "tool_heavy");
        // 10 cycles * (3 tool turns * 2 messages each + 1 assistant) + system + user
        let tool_msgs: usize = bench
            .messages
            .iter()
            .filter(|m| m.role == SessionRole::Tool)
            .count();
        assert!(tool_msgs >= 20);
    }

    #[test]
    fn test_benchmark_context_pressure() {
        let bench = benchmark_context_pressure();
        assert_eq!(bench.name, "context_pressure");
        // Should have high total token count from large tool results
        let total_tokens: u32 = bench.messages.iter().map(|m| m.token_count).sum();
        assert!(total_tokens > 1000, "context pressure benchmark should have many tokens");
    }

    #[test]
    fn test_load_benchmark_from_json() {
        let bench = benchmark_short_coding();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bench.json");
        let json = serde_json::to_string_pretty(&bench).unwrap();
        std::fs::write(&path, &json).unwrap();

        let loaded = load_benchmark(&path).unwrap();
        assert_eq!(loaded.name, bench.name);
        assert_eq!(loaded.messages.len(), bench.messages.len());
    }
}
