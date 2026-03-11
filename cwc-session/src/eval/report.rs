use super::runner::{AggregateSessionMetrics, SessionComparisonReport, SessionEvalReport};

/// Format session evaluation results as human-readable text.
pub fn format_session_report(report: &SessionEvalReport) -> String {
    let mut out = String::new();

    out.push_str(&format!(
        "=== Session Evaluation Report ===\nModel: {}\n\n",
        report.model_profile
    ));

    for (name, metrics) in &report.benchmarks {
        out.push_str(&format!("--- {} ---\n", name));
        out.push_str(&format!(
            "  Turn survival:     {:.0}% ({}/{} productive)\n",
            metrics.turn_survival.survival_rate * 100.0,
            metrics.turn_survival.productive_turns,
            metrics.turn_survival.total_turns,
        ));
        if let Some(t) = metrics.turn_survival.first_hallucination_turn {
            out.push_str(&format!("  First halluc:      turn {}\n", t));
        }
        out.push_str(&format!(
            "  Goal retention:    {:.0}% ({})\n",
            metrics.goal_retention.similarity * 100.0,
            if metrics.goal_retention.retained {
                "retained"
            } else {
                "lost"
            },
        ));
        out.push_str(&format!(
            "  Repetition rate:   {:.1}% ({} dups, {} re-reads)\n",
            metrics.repetition.repetition_rate * 100.0,
            metrics.repetition.duplicate_tool_calls,
            metrics.repetition.file_re_reads,
        ));
        out.push_str(&format!(
            "  Avg tokens/turn:   {:.0}\n",
            metrics.efficiency.avg_tokens_per_turn,
        ));
        out.push_str(&format!(
            "  Peak utilization:  {:.1}%\n",
            metrics.efficiency.peak_utilization * 100.0,
        ));
        out.push_str(&format!(
            "  Trims:             {} sliding, {} reset\n",
            metrics.efficiency.sliding_window_count, metrics.efficiency.hard_reset_count,
        ));
        out.push_str(&format!(
            "  Memory:            P={:.0}% R={:.0}% ({}/{} correct)\n\n",
            metrics.memory_quality.precision * 100.0,
            metrics.memory_quality.recall * 100.0,
            metrics.memory_quality.correct_facts,
            metrics.memory_quality.extracted_facts,
        ));
    }

    out.push_str("=== Aggregate ===\n");
    format_aggregate(&mut out, &report.aggregate);
    out
}

/// Format A/B comparison report.
pub fn format_session_comparison(report: &SessionComparisonReport) -> String {
    let mut out = String::new();

    out.push_str(&format!(
        "=== A/B Comparison ===\nConfig A: {}\nConfig B: {}\n\n",
        report.config_a_name, report.config_b_name
    ));

    for (name, metrics_a, metrics_b) in &report.per_benchmark {
        out.push_str(&format!("--- {} ---\n", name));
        out.push_str(&format!(
            "  Survival:    A={:.0}%  B={:.0}%\n",
            metrics_a.turn_survival.survival_rate * 100.0,
            metrics_b.turn_survival.survival_rate * 100.0,
        ));
        out.push_str(&format!(
            "  Goal sim:    A={:.0}%  B={:.0}%\n",
            metrics_a.goal_retention.similarity * 100.0,
            metrics_b.goal_retention.similarity * 100.0,
        ));
        out.push_str(&format!(
            "  Repetition:  A={:.1}%  B={:.1}%\n",
            metrics_a.repetition.repetition_rate * 100.0,
            metrics_b.repetition.repetition_rate * 100.0,
        ));
        out.push_str(&format!(
            "  Tokens/turn: A={:.0}   B={:.0}\n\n",
            metrics_a.efficiency.avg_tokens_per_turn, metrics_b.efficiency.avg_tokens_per_turn,
        ));
    }

    out.push_str("=== Metric Deltas (B - A) ===\n");
    for delta in &report.improvements {
        let arrow = if delta.improved { "+" } else { "-" };
        out.push_str(&format!(
            "  {:<20} A={:.3}  B={:.3}  delta={}{:.3}\n",
            delta.metric,
            delta.value_a,
            delta.value_b,
            arrow,
            delta.delta.abs(),
        ));
    }

    out
}

fn format_aggregate(out: &mut String, agg: &AggregateSessionMetrics) {
    out.push_str(&format!(
        "  Avg survival rate:       {:.1}%\n",
        agg.avg_survival_rate * 100.0
    ));
    out.push_str(&format!(
        "  Avg goal retention:      {:.1}%\n",
        agg.avg_goal_retention_similarity * 100.0
    ));
    out.push_str(&format!(
        "  Avg repetition rate:     {:.1}%\n",
        agg.avg_repetition_rate * 100.0
    ));
    out.push_str(&format!(
        "  Avg tokens/turn:         {:.0}\n",
        agg.avg_tokens_per_turn
    ));
    out.push_str(&format!(
        "  Total sliding windows:   {}\n",
        agg.total_sliding_windows
    ));
    out.push_str(&format!(
        "  Total hard resets:       {}\n",
        agg.total_hard_resets
    ));
    out.push_str(&format!(
        "  Memory precision:        {:.1}%\n",
        agg.memory_precision * 100.0
    ));
    out.push_str(&format!(
        "  Memory recall:           {:.1}%\n",
        agg.memory_recall * 100.0
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::metrics::*;

    fn sample_metrics() -> SessionMetrics {
        SessionMetrics {
            turn_survival: TurnSurvival {
                first_hallucination_turn: None,
                productive_turns: 10,
                total_turns: 10,
                survival_rate: 1.0,
            },
            goal_retention: GoalRetention {
                probe_turn: 10,
                retained: true,
                similarity: 0.85,
            },
            repetition: RepetitionMetrics {
                duplicate_tool_calls: 1,
                file_re_reads: 0,
                restated_findings: 0,
                repetition_rate: 0.1,
            },
            efficiency: EfficiencyMetrics {
                avg_tokens_per_turn: 500.0,
                peak_utilization: 0.45,
                sliding_window_count: 1,
                hard_reset_count: 0,
                compaction_tokens_saved: 2000,
            },
            memory_quality: MemoryQuality {
                extracted_facts: 5,
                correct_facts: 4,
                missed_facts: 1,
                precision: 0.8,
                recall: 0.8,
            },
        }
    }

    #[test]
    fn test_format_session_report_includes_all_sections() {
        let report = SessionEvalReport {
            model_profile: "local_large".into(),
            benchmarks: vec![("test_bench".into(), sample_metrics())],
            aggregate: AggregateSessionMetrics {
                avg_survival_rate: 1.0,
                avg_goal_retention_similarity: 0.85,
                avg_repetition_rate: 0.1,
                avg_tokens_per_turn: 500.0,
                total_sliding_windows: 1,
                total_hard_resets: 0,
                memory_precision: 0.8,
                memory_recall: 0.8,
            },
        };
        let text = format_session_report(&report);
        assert!(text.contains("Session Evaluation Report"));
        assert!(text.contains("test_bench"));
        assert!(text.contains("Turn survival"));
        assert!(text.contains("Goal retention"));
        assert!(text.contains("Repetition rate"));
        assert!(text.contains("Aggregate"));
        assert!(text.contains("Memory precision"));
    }

    #[test]
    fn test_format_session_comparison_shows_deltas() {
        use crate::eval::runner::MetricDelta;
        let report = SessionComparisonReport {
            config_a_name: "managed".into(),
            config_b_name: "unmanaged".into(),
            per_benchmark: vec![(
                "test_bench".into(),
                sample_metrics(),
                sample_metrics(),
            )],
            improvements: vec![
                MetricDelta {
                    metric: "survival_rate".into(),
                    value_a: 0.9,
                    value_b: 1.0,
                    delta: 0.1,
                    improved: true,
                },
                MetricDelta {
                    metric: "repetition_rate".into(),
                    value_a: 0.2,
                    value_b: 0.1,
                    delta: -0.1,
                    improved: true,
                },
            ],
        };
        let text = format_session_comparison(&report);
        assert!(text.contains("A/B Comparison"));
        assert!(text.contains("managed"));
        assert!(text.contains("unmanaged"));
        assert!(text.contains("survival_rate"));
        assert!(text.contains("Metric Deltas"));
    }

    #[test]
    fn test_format_session_report_empty_benchmarks() {
        let report = SessionEvalReport {
            model_profile: "test".into(),
            benchmarks: vec![],
            aggregate: AggregateSessionMetrics {
                avg_survival_rate: 0.0,
                avg_goal_retention_similarity: 0.0,
                avg_repetition_rate: 0.0,
                avg_tokens_per_turn: 0.0,
                total_sliding_windows: 0,
                total_hard_resets: 0,
                memory_precision: 0.0,
                memory_recall: 0.0,
            },
        };
        let text = format_session_report(&report);
        assert!(text.contains("Session Evaluation Report"));
        assert!(text.contains("Aggregate"));
    }
}
