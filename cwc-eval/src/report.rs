use std::fmt::Write;

use crate::runner::{BenchmarkReport, ComparisonReport};

/// Format a benchmark report as a human-readable string.
pub fn format_report(report: &BenchmarkReport) -> String {
    let mut out = String::new();

    writeln!(out, "=== CWC Benchmark Report ===").ok();
    writeln!(
        out,
        "Dataset: {} ({} queries)",
        report.dataset_name,
        report.generation.result_count
    )
    .ok();
    writeln!(out).ok();

    writeln!(out, "Retrieval:").ok();
    writeln!(out, "  Recall@k:          {:.2}", report.retrieval.recall_at_k).ok();
    writeln!(out, "  Precision@k:       {:.2}", report.retrieval.precision_at_k).ok();
    writeln!(out, "  MRR:               {:.2}", report.retrieval.mrr).ok();
    writeln!(out, "  nDCG@k:            {:.2}", report.retrieval.ndcg_at_k).ok();
    writeln!(out, "  Hit Rate:          {:.2}", report.retrieval.hit_rate).ok();
    writeln!(out).ok();

    writeln!(out, "Generation:").ok();
    writeln!(out, "  Citation Rate:     {:.2}", report.generation.citation_rate).ok();
    writeln!(
        out,
        "  Citation Validity: {:.2}",
        report.generation.citation_validity
    )
    .ok();
    writeln!(
        out,
        "  Abstention Acc:    {:.2}",
        report.generation.abstention_accuracy
    )
    .ok();
    writeln!(
        out,
        "  Schema Compliance: {:.2}",
        report.generation.schema_compliance
    )
    .ok();
    writeln!(
        out,
        "  Answer Relevance:  {:.2}",
        report.generation.answer_relevance
    )
    .ok();
    writeln!(out).ok();

    if report.adversarial.query_count > 0 {
        writeln!(out, "Adversarial:").ok();
        writeln!(
            out,
            "  Injection Resist:  {:.2}",
            report.adversarial.injection_resistance
        )
        .ok();
        writeln!(
            out,
            "  Delimiter Escape:  {:.2}",
            report.adversarial.delimiter_escape
        )
        .ok();
        writeln!(
            out,
            "  Override Resist:   {:.2}",
            report.adversarial.instruction_override
        )
        .ok();
        writeln!(out).ok();
    }

    if report.timing.query_count > 0 {
        writeln!(out, "Timing:").ok();
        writeln!(
            out,
            "  Avg Retrieval:     {:.0}ms",
            report.timing.avg_retrieval_ms
        )
        .ok();
        writeln!(
            out,
            "  Avg Generation:    {:.0}ms",
            report.timing.avg_generation_ms
        )
        .ok();
        writeln!(
            out,
            "  Avg Total:         {:.0}ms",
            report.timing.avg_total_ms
        )
        .ok();
        writeln!(
            out,
            "  p95 Total:         {:.0}ms",
            report.timing.p95_total_ms
        )
        .ok();
    }

    out
}

/// Format a comparison report.
pub fn format_comparison(report: &ComparisonReport) -> String {
    let mut out = String::new();

    writeln!(out, "=== CWC A/B Comparison ===").ok();
    writeln!(
        out,
        "Dataset: {} vs {}",
        report.report_a.dataset_name, report.report_b.dataset_name
    )
    .ok();
    writeln!(out).ok();

    writeln!(
        out,
        "{:<22} {:>8} {:>8} {:>8} Status",
        "Metric", "A", "B", "Delta"
    )
    .ok();
    writeln!(out, "{}", "-".repeat(60)).ok();

    for d in &report.improvements {
        let status = if d.delta.abs() < 0.001 {
            "  "
        } else if d.improved {
            "+"
        } else {
            "-"
        };
        writeln!(
            out,
            "{:<22} {:>8.3} {:>8.3} {:>+8.3} {}",
            d.name, d.value_a, d.value_b, d.delta, status
        )
        .ok();
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adversarial::AdversarialMetrics;
    use crate::generation_metrics::GenerationMetrics;
    use crate::retrieval_metrics::RetrievalMetrics;
    use crate::runner::{AggregatedTiming, MetricDelta};

    fn sample_report() -> BenchmarkReport {
        BenchmarkReport {
            dataset_name: "rust_knowledge".into(),
            retrieval: RetrievalMetrics {
                recall_at_k: 0.82,
                precision_at_k: 0.45,
                mrr: 0.71,
                ndcg_at_k: 0.75,
                hit_rate: 0.94,
                query_count: 50,
            },
            generation: GenerationMetrics {
                citation_rate: 0.78,
                citation_validity: 0.95,
                abstention_accuracy: 0.80,
                schema_compliance: 1.0,
                answer_relevance: 0.85,
                verbosity_ratio: 2.5,
                result_count: 50,
            },
            adversarial: AdversarialMetrics {
                injection_resistance: 0.90,
                delimiter_escape: 0.95,
                instruction_override: 0.88,
                query_count: 10,
            },
            timing: AggregatedTiming {
                avg_retrieval_ms: 45.0,
                avg_compilation_ms: 12.0,
                avg_generation_ms: 2300.0,
                avg_verification_ms: 8.0,
                avg_total_ms: 2500.0,
                p95_total_ms: 3800.0,
                query_count: 50,
            },
            per_query: vec![],
        }
    }

    #[test]
    fn test_format_report_contains_sections() {
        let report = sample_report();
        let formatted = format_report(&report);

        assert!(formatted.contains("CWC Benchmark Report"));
        assert!(formatted.contains("rust_knowledge"));
        assert!(formatted.contains("50 queries"));
        assert!(formatted.contains("Retrieval:"));
        assert!(formatted.contains("Generation:"));
        assert!(formatted.contains("Adversarial:"));
        assert!(formatted.contains("Timing:"));
        assert!(formatted.contains("0.82")); // recall
        assert!(formatted.contains("0.78")); // citation rate
        assert!(formatted.contains("0.90")); // injection resistance
        assert!(formatted.contains("2500")); // avg total
    }

    #[test]
    fn test_format_report_no_adversarial() {
        let mut report = sample_report();
        report.adversarial = AdversarialMetrics::default();
        let formatted = format_report(&report);
        assert!(!formatted.contains("Adversarial:"));
    }

    #[test]
    fn test_format_report_no_timing() {
        let mut report = sample_report();
        report.timing = AggregatedTiming::default(); // query_count = 0
        let formatted = format_report(&report);
        assert!(!formatted.contains("Timing:"));
    }

    #[test]
    fn test_format_comparison_regression_shows_minus() {
        let report_a = sample_report();
        let report_b = sample_report();

        let comparison = ComparisonReport {
            report_a,
            report_b,
            improvements: vec![MetricDelta {
                name: "recall_at_k".into(),
                value_a: 0.90,
                value_b: 0.70,
                delta: -0.20,
                improved: false,
            }],
        };

        let formatted = format_comparison(&comparison);
        assert!(formatted.contains("-"));
        assert!(formatted.contains("-0.200"));
    }

    #[test]
    fn test_format_comparison_no_change_shows_spaces() {
        let report_a = sample_report();
        let report_b = sample_report();

        let comparison = ComparisonReport {
            report_a,
            report_b,
            improvements: vec![MetricDelta {
                name: "recall_at_k".into(),
                value_a: 0.82,
                value_b: 0.82,
                delta: 0.0,
                improved: false,
            }],
        };

        let formatted = format_comparison(&comparison);
        // Delta near zero → status should be "  " (spaces, not + or -)
        assert!(formatted.contains("+0.000   ")); // +0.000 followed by spaces
    }

    #[test]
    fn test_format_comparison() {
        let report_a = sample_report();
        let mut report_b = sample_report();
        report_b.retrieval.recall_at_k = 0.90;

        let comparison = ComparisonReport {
            report_a: report_a.clone(),
            report_b: report_b.clone(),
            improvements: vec![MetricDelta {
                name: "recall_at_k".into(),
                value_a: 0.82,
                value_b: 0.90,
                delta: 0.08,
                improved: true,
            }],
        };

        let formatted = format_comparison(&comparison);
        assert!(formatted.contains("A/B Comparison"));
        assert!(formatted.contains("recall_at_k"));
        assert!(formatted.contains("+"));
    }
}
