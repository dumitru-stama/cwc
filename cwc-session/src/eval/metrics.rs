use serde::{Deserialize, Serialize};

use crate::manager::OptimizationReport;
use crate::message::{SessionMessage, SessionRole, ToolCall};

/// Metrics for evaluating session management quality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetrics {
    pub turn_survival: TurnSurvival,
    pub goal_retention: GoalRetention,
    pub repetition: RepetitionMetrics,
    pub efficiency: EfficiencyMetrics,
    pub memory_quality: MemoryQuality,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnSurvival {
    /// Turn index where first hallucination was detected.
    /// None = no hallucination (full survival).
    pub first_hallucination_turn: Option<usize>,
    /// Total productive turns.
    pub productive_turns: usize,
    /// Total turns in the session.
    pub total_turns: usize,
    /// Survival rate = productive_turns / total_turns.
    pub survival_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalRetention {
    /// After how many turns was the model probed for its goal.
    pub probe_turn: usize,
    /// Did the model's response match the original goal?
    pub retained: bool,
    /// Similarity score (word overlap) between stated goal and original.
    pub similarity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepetitionMetrics {
    /// Number of times the model called the same tool with same arguments.
    pub duplicate_tool_calls: usize,
    /// Number of times the model re-read a file it already read.
    pub file_re_reads: usize,
    /// Number of times the model restated a finding it already made.
    pub restated_findings: usize,
    /// Repetition rate = duplicates / total_tool_calls.
    pub repetition_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    /// Average tokens per turn (lower = more efficient).
    pub avg_tokens_per_turn: f32,
    /// Peak token usage as fraction of usable budget.
    pub peak_utilization: f32,
    /// Number of sliding window trims triggered.
    pub sliding_window_count: usize,
    /// Number of hard resets triggered.
    pub hard_reset_count: usize,
    /// Total tokens saved by compaction.
    pub compaction_tokens_saved: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryQuality {
    /// Facts extracted by the system.
    pub extracted_facts: usize,
    /// Facts that match ground-truth annotations.
    pub correct_facts: usize,
    /// Facts missed (in ground truth but not extracted).
    pub missed_facts: usize,
    /// Precision = correct / extracted.
    pub precision: f32,
    /// Recall = correct / (correct + missed).
    pub recall: f32,
}

/// Compute TurnSurvival from annotated hallucination turns.
pub fn compute_turn_survival(
    total_turns: usize,
    hallucination_turns: &[usize],
) -> TurnSurvival {
    let first_hallucination_turn = hallucination_turns.iter().min().copied();
    let productive_turns = match first_hallucination_turn {
        Some(t) => t.min(total_turns),
        None => total_turns,
    };
    let survival_rate = if total_turns == 0 {
        1.0
    } else {
        productive_turns as f32 / total_turns as f32
    };
    TurnSurvival {
        first_hallucination_turn,
        productive_turns,
        total_turns,
        survival_rate,
    }
}

/// Compute GoalRetention by comparing stated goal to original using word overlap.
pub fn compute_goal_retention(
    original_goal: &str,
    stated_goal: &str,
    probe_turn: usize,
) -> GoalRetention {
    let similarity = word_overlap_similarity(original_goal, stated_goal);
    GoalRetention {
        probe_turn,
        retained: similarity >= 0.5,
        similarity,
    }
}

/// Compute RepetitionMetrics from a list of assistant messages.
pub fn compute_repetition(messages: &[SessionMessage]) -> RepetitionMetrics {
    let mut seen_tool_calls: Vec<(String, String)> = Vec::new(); // (name, args_hash)
    let mut seen_file_reads: Vec<String> = Vec::new();
    let mut duplicate_tool_calls = 0usize;
    let mut file_re_reads = 0usize;
    let mut total_tool_calls = 0usize;

    for msg in messages {
        if msg.role != SessionRole::Assistant {
            continue;
        }
        for tc in &msg.tool_calls {
            total_tool_calls += 1;
            let args_str = tc.arguments.to_string();
            let key = (tc.tool_name.clone(), args_str);

            if seen_tool_calls.contains(&key) {
                duplicate_tool_calls += 1;
                if is_file_read_tool(&tc.tool_name) {
                    if let Some(path) = extract_file_path(tc) {
                        if seen_file_reads.contains(&path) {
                            file_re_reads += 1;
                        }
                    }
                }
            } else {
                seen_tool_calls.push(key);
                if is_file_read_tool(&tc.tool_name) {
                    if let Some(path) = extract_file_path(tc) {
                        if !seen_file_reads.contains(&path) {
                            seen_file_reads.push(path);
                        }
                    }
                }
            }
        }
    }

    let repetition_rate = if total_tool_calls == 0 {
        0.0
    } else {
        duplicate_tool_calls as f32 / total_tool_calls as f32
    };

    RepetitionMetrics {
        duplicate_tool_calls,
        file_re_reads,
        restated_findings: 0, // Would require NLP; stubbed
        repetition_rate,
    }
}

/// Compute EfficiencyMetrics from a sequence of optimization reports.
pub fn compute_efficiency(
    reports: &[OptimizationReport],
    usable_budget: u32,
) -> EfficiencyMetrics {
    let mut total_tokens = 0u64;
    let mut peak_tokens = 0u32;
    let mut sliding_window_count = 0;
    let mut hard_reset_count = 0;
    let mut compaction_tokens_saved = 0u32;

    for report in reports {
        total_tokens += report.output_tokens as u64;
        if report.output_tokens > peak_tokens {
            peak_tokens = report.output_tokens;
        }
        match &report.trim {
            crate::manager::TrimAction::SlidingWindow { .. } => sliding_window_count += 1,
            crate::manager::TrimAction::HardReset => hard_reset_count += 1,
            _ => {}
        }
        compaction_tokens_saved += report.tokens_saved;
    }

    let avg_tokens_per_turn = if reports.is_empty() {
        0.0
    } else {
        total_tokens as f32 / reports.len() as f32
    };
    let peak_utilization = if usable_budget == 0 {
        0.0
    } else {
        peak_tokens as f32 / usable_budget as f32
    };

    EfficiencyMetrics {
        avg_tokens_per_turn,
        peak_utilization,
        sliding_window_count,
        hard_reset_count,
        compaction_tokens_saved,
    }
}

/// Compute MemoryQuality against ground-truth expected facts.
///
/// Uses 1-to-1 matching: each expected fact matches at most one extracted fact,
/// and each extracted fact matches at most one expected fact. This prevents
/// precision from exceeding 1.0 when one extracted fact matches multiple patterns.
pub fn compute_memory_quality(
    extracted_keys: &[String],
    extracted_values: &[String],
    expected: &[(String, String)], // (key_pattern, value_contains)
) -> MemoryQuality {
    let extracted_facts = extracted_keys.len();
    let mut correct_facts = 0;
    let mut matched_expected = vec![false; expected.len()];
    let mut matched_extracted = vec![false; extracted_facts];

    for (i, (key_pat, val_substr)) in expected.iter().enumerate() {
        for (j, (ek, ev)) in extracted_keys.iter().zip(extracted_values.iter()).enumerate() {
            if !matched_expected[i]
                && !matched_extracted[j]
                && key_matches(ek, key_pat)
                && ev.contains(val_substr.as_str())
            {
                correct_facts += 1;
                matched_expected[i] = true;
                matched_extracted[j] = true;
            }
        }
    }

    let missed_facts = matched_expected.iter().filter(|&&m| !m).count();
    let precision = if extracted_facts == 0 {
        1.0 // No false positives if nothing extracted
    } else {
        correct_facts as f32 / extracted_facts as f32
    };
    let total_relevant = correct_facts + missed_facts;
    let recall = if total_relevant == 0 {
        1.0
    } else {
        correct_facts as f32 / total_relevant as f32
    };

    MemoryQuality {
        extracted_facts,
        correct_facts,
        missed_facts,
        precision,
        recall,
    }
}

/// Word overlap similarity between two strings (Jaccard on word sets).
fn word_overlap_similarity(a: &str, b: &str) -> f32 {
    let a_lower: std::collections::HashSet<String> =
        a.split_whitespace().map(|w| w.to_lowercase()).collect();
    let b_lower: std::collections::HashSet<String> =
        b.split_whitespace().map(|w| w.to_lowercase()).collect();

    if a_lower.is_empty() && b_lower.is_empty() {
        return 1.0;
    }
    if a_lower.is_empty() || b_lower.is_empty() {
        return 0.0;
    }

    let intersection = a_lower.intersection(&b_lower).count();
    let union = a_lower.union(&b_lower).count();

    if union == 0 {
        1.0
    } else {
        intersection as f32 / union as f32
    }
}

fn is_file_read_tool(name: &str) -> bool {
    matches!(name, "file.read" | "file.read_range" | "Read" | "cat")
}

fn extract_file_path(tc: &ToolCall) -> Option<String> {
    tc.arguments
        .get("path")
        .or_else(|| tc.arguments.get("file_path"))
        .or_else(|| tc.arguments.get("file"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Check if an extracted key matches a pattern.
/// Pattern can be a glob (* wildcards) or substring.
fn key_matches(key: &str, pattern: &str) -> bool {
    if pattern.contains('*') {
        glob_match(key, pattern)
    } else {
        key.contains(pattern)
    }
}

/// Simple glob matching supporting only '*' wildcard.
fn glob_match(text: &str, pattern: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return text == pattern;
    }
    let mut pos = 0;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        match text[pos..].find(part) {
            Some(found) => {
                if i == 0 && found != 0 {
                    return false; // First part must be at start
                }
                pos += found + part.len();
            }
            None => return false,
        }
    }
    // If pattern doesn't end with *, text must end exactly
    if !pattern.ends_with('*') {
        return text.ends_with(parts.last().unwrap_or(&""));
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manager::TrimAction;
    use crate::message::{SessionMessage, SessionRole, ToolCall};

    #[test]
    fn test_turn_survival_with_hallucination() {
        let survival = compute_turn_survival(20, &[12, 15, 18]);
        assert_eq!(survival.first_hallucination_turn, Some(12));
        assert_eq!(survival.productive_turns, 12);
        assert_eq!(survival.total_turns, 20);
        assert!((survival.survival_rate - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_turn_survival_hallucination_index_exceeds_total() {
        // Hallucination at index 3 but only 1 user turn — should clamp
        let survival = compute_turn_survival(1, &[3]);
        assert_eq!(survival.productive_turns, 1); // clamped to total
        assert!(survival.survival_rate <= 1.0, "rate was {}", survival.survival_rate);
    }

    #[test]
    fn test_turn_survival_no_hallucination() {
        let survival = compute_turn_survival(15, &[]);
        assert_eq!(survival.first_hallucination_turn, None);
        assert_eq!(survival.productive_turns, 15);
        assert!((survival.survival_rate - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_goal_retention_exact_match() {
        let retention = compute_goal_retention(
            "find security vulnerabilities",
            "find security vulnerabilities",
            10,
        );
        assert!((retention.similarity - 1.0).abs() < 0.001);
        assert!(retention.retained);
        assert_eq!(retention.probe_turn, 10);
    }

    #[test]
    fn test_goal_retention_partial_match() {
        let retention = compute_goal_retention(
            "find security vulnerabilities in the parser",
            "analyze security issues in the codebase",
            5,
        );
        assert!(retention.similarity > 0.0);
        assert!(retention.similarity < 1.0);
    }

    #[test]
    fn test_goal_retention_no_match() {
        let retention = compute_goal_retention(
            "find security vulnerabilities",
            "the weather is sunny today",
            8,
        );
        assert!(retention.similarity < 0.5);
        assert!(!retention.retained);
    }

    #[test]
    fn test_repetition_duplicate_tool_calls() {
        let messages = vec![
            SessionMessage::assistant_tool_calls(
                "",
                vec![ToolCall {
                    call_id: "c1".into(),
                    tool_name: "file.grep".into(),
                    arguments: serde_json::json!({"pattern": "bug"}),
                }],
            ),
            SessionMessage::text(SessionRole::User, "continue"),
            SessionMessage::assistant_tool_calls(
                "",
                vec![ToolCall {
                    call_id: "c2".into(),
                    tool_name: "file.grep".into(),
                    arguments: serde_json::json!({"pattern": "bug"}),
                }],
            ),
        ];
        let rep = compute_repetition(&messages);
        assert_eq!(rep.duplicate_tool_calls, 1);
        assert!((rep.repetition_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_repetition_file_re_reads() {
        let messages = vec![
            SessionMessage::assistant_tool_calls(
                "",
                vec![ToolCall {
                    call_id: "c1".into(),
                    tool_name: "file.read".into(),
                    arguments: serde_json::json!({"path": "src/main.rs"}),
                }],
            ),
            SessionMessage::text(SessionRole::User, "continue"),
            SessionMessage::assistant_tool_calls(
                "",
                vec![ToolCall {
                    call_id: "c2".into(),
                    tool_name: "file.read".into(),
                    arguments: serde_json::json!({"path": "src/main.rs"}),
                }],
            ),
        ];
        let rep = compute_repetition(&messages);
        assert_eq!(rep.duplicate_tool_calls, 1);
        assert_eq!(rep.file_re_reads, 1);
    }

    #[test]
    fn test_repetition_no_duplicates() {
        let messages = vec![
            SessionMessage::assistant_tool_calls(
                "",
                vec![ToolCall {
                    call_id: "c1".into(),
                    tool_name: "file.grep".into(),
                    arguments: serde_json::json!({"pattern": "bug"}),
                }],
            ),
            SessionMessage::assistant_tool_calls(
                "",
                vec![ToolCall {
                    call_id: "c2".into(),
                    tool_name: "file.grep".into(),
                    arguments: serde_json::json!({"pattern": "error"}),
                }],
            ),
        ];
        let rep = compute_repetition(&messages);
        assert_eq!(rep.duplicate_tool_calls, 0);
        assert!((rep.repetition_rate - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_efficiency_from_reports() {
        let reports = vec![
            OptimizationReport {
                input_tokens: 1000,
                output_tokens: 800,
                tokens_saved: 200,
                compaction: None,
                trim: TrimAction::None,
                preflight_issues: vec![],
                nudge_injected: false,
                memory_facts: 0,
                consolidation: None,
            },
            OptimizationReport {
                input_tokens: 2000,
                output_tokens: 1500,
                tokens_saved: 500,
                compaction: None,
                trim: TrimAction::SlidingWindow { turns_dropped: 3 },
                preflight_issues: vec![],
                nudge_injected: true,
                memory_facts: 2,
                consolidation: None,
            },
            OptimizationReport {
                input_tokens: 3000,
                output_tokens: 500,
                tokens_saved: 2500,
                compaction: None,
                trim: TrimAction::HardReset,
                preflight_issues: vec![],
                nudge_injected: false,
                memory_facts: 5,
                consolidation: None,
            },
        ];
        let eff = compute_efficiency(&reports, 10000);
        assert!((eff.avg_tokens_per_turn - (2800.0 / 3.0)).abs() < 1.0);
        assert!((eff.peak_utilization - 0.15).abs() < 0.01);
        assert_eq!(eff.sliding_window_count, 1);
        assert_eq!(eff.hard_reset_count, 1);
        assert_eq!(eff.compaction_tokens_saved, 3200);
    }

    #[test]
    fn test_memory_quality_perfect() {
        let keys = vec!["goal".into(), "finding:parser_bug".into()];
        let values = vec!["find bugs".into(), "null pointer in parser".into()];
        let expected = vec![
            ("goal".into(), "bugs".into()),
            ("finding:*".into(), "parser".into()),
        ];
        let mq = compute_memory_quality(&keys, &values, &expected);
        assert_eq!(mq.correct_facts, 2);
        assert_eq!(mq.missed_facts, 0);
        assert!((mq.precision - 1.0).abs() < 0.001);
        assert!((mq.recall - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_memory_quality_partial() {
        let keys = vec!["goal".into(), "extra_fact".into()];
        let values = vec!["find bugs".into(), "irrelevant".into()];
        let expected = vec![
            ("goal".into(), "bugs".into()),
            ("finding:*".into(), "parser".into()),
        ];
        let mq = compute_memory_quality(&keys, &values, &expected);
        assert_eq!(mq.correct_facts, 1);
        assert_eq!(mq.missed_facts, 1);
        assert!((mq.precision - 0.5).abs() < 0.01);
        assert!((mq.recall - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_memory_quality_precision_never_exceeds_1() {
        // One extracted fact matches two expected patterns — should not double-count
        let keys = vec!["finding:all_issues".into()];
        let values = vec!["unwrap parser hardcoded todo".into()];
        let expected = vec![
            ("finding:*".into(), "unwrap".into()),
            ("finding:*".into(), "parser".into()),
            ("finding:*".into(), "hardcoded".into()),
        ];
        let mq = compute_memory_quality(&keys, &values, &expected);
        // Only 1 extracted fact, so at most 1 can match (1-to-1)
        assert_eq!(mq.correct_facts, 1);
        assert!(mq.precision <= 1.0, "precision was {}", mq.precision);
        assert_eq!(mq.missed_facts, 2);
    }

    #[test]
    fn test_glob_match() {
        assert!(glob_match("finding:parser_bug", "finding:*"));
        assert!(glob_match("finding:parser_bug", "*parser*"));
        assert!(!glob_match("goal", "finding:*"));
        assert!(glob_match("goal", "goal"));
    }

    #[test]
    fn test_word_overlap_similarity() {
        let sim = word_overlap_similarity("hello world", "hello world");
        assert!((sim - 1.0).abs() < 0.001);

        let sim = word_overlap_similarity("hello world", "goodbye moon");
        assert!((sim - 0.0).abs() < 0.001);

        let sim = word_overlap_similarity("find security bugs", "find all bugs");
        assert!(sim > 0.0);
        assert!(sim < 1.0);
    }
}
