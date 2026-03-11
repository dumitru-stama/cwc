use cwc_core::traits::TokenCounter;

/// Generate a summary using HeadTruncate strategy.
///
/// Keeps the first N lines that fit within `max_tokens`, then appends
/// "... (M more lines)" if truncated.
pub fn summarize_head_truncate(
    output: &str,
    max_tokens: u32,
    tokenizer: &dyn TokenCounter,
) -> String {
    if output.is_empty() {
        return String::new();
    }

    let total_tokens = tokenizer.count_tokens(output);
    if total_tokens <= max_tokens {
        return output.to_string();
    }

    let lines: Vec<&str> = output.lines().collect();
    let total_lines = lines.len();
    // Reserve budget for the truncation notice
    let reserve = 10;
    let budget = max_tokens.saturating_sub(reserve);

    // Single-line or very long first line: fall back to token-level truncation
    if total_lines <= 1 {
        // Use at least half the budget for content, rest for the notice
        let content_budget = budget.max(max_tokens / 2);
        let truncated = tokenizer.truncate_to_tokens(output, content_budget);
        let remaining_tokens = total_tokens.saturating_sub(tokenizer.count_tokens(&truncated));
        if remaining_tokens > 0 {
            return format!("{truncated}\n... ({remaining_tokens} more tokens)");
        }
        return truncated;
    }

    let mut kept = Vec::new();
    let mut used_tokens: u32 = 0;

    for line in &lines {
        let line_tokens = tokenizer.count_tokens(line);
        if used_tokens + line_tokens > budget && !kept.is_empty() {
            break;
        }
        kept.push(*line);
        used_tokens += line_tokens;
    }

    let remaining = total_lines - kept.len();
    let mut result = kept.join("\n");
    if remaining > 0 {
        result.push_str(&format!("\n... ({remaining} more lines)"));
    }
    result
}

/// Generate a summary using TopItems strategy.
///
/// Extracts structured items from tool output based on tool name,
/// shows top N items with aggregate counts.
pub fn summarize_top_items(
    tool_name: &str,
    output: &str,
    max_items: usize,
    max_tokens: u32,
    tokenizer: &dyn TokenCounter,
) -> String {
    if output.is_empty() {
        return String::new();
    }

    let total_tokens = tokenizer.count_tokens(output);
    if total_tokens <= max_tokens {
        return output.to_string();
    }

    if tool_name.contains("grep") {
        summarize_grep_items(output, max_items)
    } else if tool_name.contains("test") {
        summarize_test_items(output, max_items)
    } else if tool_name.contains("build") {
        summarize_build_items(output, max_items)
    } else {
        // Fallback: treat as generic line items
        summarize_generic_items(output, max_items)
    }
}

/// Generate a summary using HeadTail strategy.
///
/// Keeps the first lines that fit + last `tail_lines` lines, with "..." separator.
/// If output is shorter than head+tail, returns unchanged.
pub fn summarize_head_tail(
    output: &str,
    max_tokens: u32,
    tail_lines: usize,
    tokenizer: &dyn TokenCounter,
) -> String {
    if output.is_empty() {
        return String::new();
    }

    let total_tokens = tokenizer.count_tokens(output);
    if total_tokens <= max_tokens {
        return output.to_string();
    }

    let lines: Vec<&str> = output.lines().collect();
    if lines.len() <= tail_lines + 1 {
        return output.to_string();
    }

    let tail_start = lines.len().saturating_sub(tail_lines);
    let tail: Vec<&str> = lines[tail_start..].to_vec();
    let tail_text = tail.join("\n");
    let tail_tokens = tokenizer.count_tokens(&tail_text);

    // separator + tail annotation
    let separator_reserve = 10;
    let head_budget = max_tokens.saturating_sub(tail_tokens).saturating_sub(separator_reserve);

    let mut head = Vec::new();
    let mut used: u32 = 0;
    for line in &lines[..tail_start] {
        let lt = tokenizer.count_tokens(line);
        if used + lt > head_budget && !head.is_empty() {
            break;
        }
        head.push(*line);
        used += lt;
    }

    let skipped = lines.len() - head.len() - tail.len();
    let mut result = head.join("\n");
    result.push_str(&format!("\n... ({skipped} lines omitted)\n"));
    result.push_str(&tail_text);
    result
}

/// Extract grep-style matches: "file:line: content" or "file:line-content".
pub struct ItemExtractor;

impl ItemExtractor {
    /// Extract grep items as (location, content) pairs.
    pub fn extract_grep_items(output: &str) -> Vec<(String, String)> {
        let mut items = Vec::new();
        for line in output.lines() {
            // Pattern: "path:linenum: content" or "path:linenum-content"
            if let Some((loc, content)) = split_grep_line(line) {
                items.push((loc.to_string(), content.to_string()));
            }
        }
        items
    }

    /// Extract test results as (test_name, passed) pairs.
    ///
    /// Recognizes Rust test output: `test test_name ... ok` / `test test_name ... FAILED`
    pub fn extract_test_items(output: &str) -> Vec<(String, bool)> {
        let mut items = Vec::new();
        for line in output.lines() {
            let trimmed = line.trim();
            // Rust test format: "test <name> ... ok|FAILED"
            if trimmed.starts_with("test ") && trimmed.contains(" ... ") {
                let rest = &trimmed[5..];
                if let Some(idx) = rest.find(" ... ") {
                    let name = &rest[..idx];
                    let status = rest[idx + 5..].trim();
                    let passed = status == "ok";
                    items.push((name.to_string(), passed));
                }
            } else if trimmed.ends_with("... ok") {
                let name = trimmed.trim_end_matches("... ok").trim();
                items.push((name.to_string(), true));
            } else if trimmed.ends_with("... FAILED") {
                let name = trimmed.trim_end_matches("... FAILED").trim();
                items.push((name.to_string(), false));
            }
        }
        items
    }

    /// Extract build errors/warnings as (severity, message) pairs.
    pub fn extract_build_items(output: &str) -> Vec<(String, String)> {
        let mut items = Vec::new();
        for line in output.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("error") || trimmed.starts_with("warning") {
                let severity = if trimmed.starts_with("error") {
                    "error"
                } else {
                    "warning"
                };
                items.push((severity.to_string(), trimmed.to_string()));
            }
        }
        items
    }
}

/// Split a grep-style line into (location, content).
fn split_grep_line(line: &str) -> Option<(&str, &str)> {
    // Try "path:num: content" pattern
    // Find the second colon (or colon after digits)
    let bytes = line.as_bytes();
    let mut colon_count = 0;
    for (i, &b) in bytes.iter().enumerate() {
        if b == b':' {
            colon_count += 1;
            if colon_count >= 2 {
                let loc = &line[..i];
                let content = line[i + 1..].trim_start();
                return Some((loc, content));
            }
        }
    }
    // Fallback: single colon split
    if let Some(idx) = line.find(':') {
        let loc = &line[..idx];
        let content = line[idx + 1..].trim_start();
        if !loc.is_empty() {
            return Some((loc, content));
        }
    }
    None
}

fn summarize_grep_items(output: &str, max_items: usize) -> String {
    let items = ItemExtractor::extract_grep_items(output);
    let total = items.len();
    if total == 0 {
        let line_count = output.lines().count();
        return format!("({line_count} lines of output)");
    }

    let mut result = format!("Found {total} matches.");
    let show = max_items.min(total);
    for (loc, content) in items.iter().take(show) {
        let preview: String = content.chars().take(80).collect();
        result.push_str(&format!("\n  {loc}: {preview}"));
    }
    if total > show {
        result.push_str(&format!("\n  ... ({} more matches)", total - show));
    }
    result
}

fn summarize_test_items(output: &str, max_items: usize) -> String {
    let items = ItemExtractor::extract_test_items(output);
    let passed = items.iter().filter(|(_, p)| *p).count();
    let failed = items.iter().filter(|(_, p)| !*p).count();
    let total = items.len();

    let mut result = format!("{total} tests: {passed} passed, {failed} failed.");
    // Show failed tests first
    let failures: Vec<_> = items.iter().filter(|(_, p)| !*p).collect();
    let show = max_items.min(failures.len());
    for (name, _) in failures.iter().take(show) {
        result.push_str(&format!("\n  FAILED: {name}"));
    }
    if failures.len() > show {
        result.push_str(&format!("\n  ... ({} more failures)", failures.len() - show));
    }
    result
}

fn summarize_build_items(output: &str, max_items: usize) -> String {
    let items = ItemExtractor::extract_build_items(output);
    let errors = items.iter().filter(|(s, _)| s == "error").count();
    let warnings = items.iter().filter(|(s, _)| s == "warning").count();

    let status = if errors > 0 { "failed" } else { "succeeded" };
    let mut result = format!("Build {status}. {errors} errors, {warnings} warnings.");

    // Show errors first, then warnings
    let show_items: Vec<_> = items
        .iter()
        .filter(|(s, _)| s == "error")
        .chain(items.iter().filter(|(s, _)| s == "warning"))
        .take(max_items)
        .collect();
    for (_, msg) in &show_items {
        let preview: String = msg.chars().take(100).collect();
        result.push_str(&format!("\n  {preview}"));
    }
    if items.len() > show_items.len() {
        result.push_str(&format!("\n  ... ({} more)", items.len() - show_items.len()));
    }
    result
}

fn summarize_generic_items(output: &str, max_items: usize) -> String {
    let lines: Vec<&str> = output.lines().collect();
    let total = lines.len();
    let show = max_items.min(total);
    let mut result = String::new();
    for line in lines.iter().take(show) {
        if !result.is_empty() {
            result.push('\n');
        }
        result.push_str(line);
    }
    if total > show {
        result.push_str(&format!("\n... ({} more lines)", total - show));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // --- HeadTruncate tests ---

    #[test]
    fn test_head_truncate_under_limit() {
        let output = "line one\nline two\nline three";
        let result = summarize_head_truncate(output, 100, tc());
        assert_eq!(result, output);
    }

    #[test]
    fn test_head_truncate_over_limit() {
        let output = "line one\nline two\nline three\nline four\nline five\nline six\nline seven\nline eight\nline nine\nline ten";
        // Each line is 2 words. Budget = 8 words - 10 reserve = ~0 but we keep at least 1 line
        let result = summarize_head_truncate(output, 8, tc());
        assert!(result.contains("more lines"), "got: {result}");
        // Should keep some lines and truncate
        assert!(result.lines().count() < 10);
    }

    #[test]
    fn test_head_truncate_empty() {
        let result = summarize_head_truncate("", 100, tc());
        assert_eq!(result, "");
    }

    #[test]
    fn test_head_truncate_single_long_line_kept() {
        // A single line that exceeds budget — we still keep it (at least first line)
        let output = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10";
        let result = summarize_head_truncate(output, 5, tc());
        // Single line — no truncation notice since there's nothing remaining
        assert!(result.contains("word1"));
    }

    // --- TopItems tests ---

    #[test]
    fn test_top_items_grep() {
        let output = "src/main.rs:10: let x = malloc(256);\nsrc/main.rs:20: free(x);\nsrc/util.rs:5: malloc(128);\nsrc/util.rs:15: malloc(64);\nsrc/lib.rs:100: malloc(512);";
        let result = summarize_top_items("file.grep", output, 3, 10, tc());
        assert!(result.contains("Found 5 matches"));
        // Should show top 3
        assert!(result.contains("src/main.rs:10"));
        assert!(result.contains("2 more matches"));
    }

    #[test]
    fn test_top_items_test() {
        let output = "test test_one ... ok\ntest test_two ... ok\ntest test_three ... FAILED\ntest test_four ... FAILED\ntest test_five ... ok";
        let result = summarize_top_items("test.run", output, 5, 5, tc());
        assert!(result.contains("5 tests: 3 passed, 2 failed"));
        assert!(result.contains("FAILED: test_three"));
        assert!(result.contains("FAILED: test_four"));
    }

    #[test]
    fn test_top_items_build() {
        let output = "  Compiling foo v0.1.0\nerror[E0308]: mismatched types\n  --> src/main.rs:5:5\nwarning: unused variable: `x`\n  --> src/main.rs:3:9\nwarning: unused import\nerror: aborting due to previous error";
        let result = summarize_top_items("build", output, 5, 5, tc());
        assert!(result.contains("Build failed"));
        assert!(result.contains("errors"));
        assert!(result.contains("warnings"));
    }

    #[test]
    fn test_top_items_under_limit() {
        let output = "one match";
        let result = summarize_top_items("file.grep", output, 5, 100, tc());
        // Under token limit — returned as-is
        assert_eq!(result, output);
    }

    // --- HeadTail tests ---

    #[test]
    fn test_head_tail_under_limit() {
        let output = "line1\nline2\nline3";
        let result = summarize_head_tail(output, 100, 2, tc());
        assert_eq!(result, output);
    }

    #[test]
    fn test_head_tail_keeps_head_and_tail() {
        let mut lines = Vec::new();
        for i in 0..30 {
            lines.push(format!("line {i}"));
        }
        let output = lines.join("\n");
        let result = summarize_head_tail(&output, 20, 5, tc());
        // Should have head lines + "..." + last 5 lines
        assert!(result.contains("lines omitted"), "got: {result}");
        assert!(result.contains("line 29")); // last line
        assert!(result.contains("line 25")); // tail start
        assert!(result.contains("line 0"));  // first line
    }

    #[test]
    fn test_head_tail_short_output() {
        // Output shorter than tail_lines → returned unchanged
        let output = "line1\nline2";
        let result = summarize_head_tail(output, 5, 5, tc());
        assert_eq!(result, output);
    }

    // --- ItemExtractor tests ---

    #[test]
    fn test_extract_grep_items() {
        let output = "src/main.rs:10: let x = 5;\nsrc/lib.rs:20: fn foo() {}";
        let items = ItemExtractor::extract_grep_items(output);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].0, "src/main.rs:10");
        assert_eq!(items[0].1, "let x = 5;");
    }

    #[test]
    fn test_extract_test_items() {
        let output = "test test_one ... ok\ntest test_two ... FAILED\ntest test_three ... ok";
        let items = ItemExtractor::extract_test_items(output);
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].0, "test_one");
        assert!(items[0].1);
        assert_eq!(items[1].0, "test_two");
        assert!(!items[1].1);
    }

    #[test]
    fn test_extract_build_items() {
        let output = "error[E0308]: mismatched types\nwarning: unused variable\nerror: aborting";
        let items = ItemExtractor::extract_build_items(output);
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].0, "error");
        assert_eq!(items[1].0, "warning");
        assert_eq!(items[2].0, "error");
    }

    #[test]
    fn test_extract_grep_items_empty() {
        let items = ItemExtractor::extract_grep_items("no matches here");
        // "no matches here" has one colon-less line → no items
        assert!(items.is_empty());
    }

    #[test]
    fn test_head_tail_zero_tail_lines() {
        let output = (0..20).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n");
        let result = summarize_head_tail(&output, 15, 0, tc());
        // With 0 tail lines, should show head + omitted count but no tail
        assert!(result.contains("line 0"));
        assert!(result.contains("lines omitted"));
        assert!(!result.contains("line 19")); // tail not shown
    }

    #[test]
    fn test_top_items_grep_zero_items() {
        let output = "src/a.rs:1: x\nsrc/b.rs:2: y\nsrc/c.rs:3: z";
        let result = summarize_top_items("file.grep", output, 0, 5, tc());
        // max_items=0 → shows count but no individual items
        assert!(result.contains("Found 3 matches"));
        assert!(result.contains("3 more matches"));
    }

    #[test]
    fn test_top_items_test_zero_items() {
        let output = "test test_a ... ok\ntest test_b ... FAILED";
        let result = summarize_top_items("test.run", output, 0, 2, tc());
        assert!(result.contains("2 tests: 1 passed, 1 failed"));
        // No individual failures shown since max_items=0
        assert!(!result.contains("FAILED: test_b"));
    }

    #[test]
    fn test_head_truncate_preserves_content_when_fits() {
        let output = "short output";
        let result = summarize_head_truncate(output, 1000, tc());
        assert_eq!(result, "short output");
    }
}
