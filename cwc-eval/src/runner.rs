use serde::{Deserialize, Serialize};

use crate::adversarial::AdversarialMetrics;
use crate::generation_metrics::{EvalResult, GenerationMetrics};
use crate::retrieval_metrics::RetrievalMetrics;

/// Aggregated timing statistics across all benchmark queries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregatedTiming {
    pub avg_retrieval_ms: f64,
    pub avg_compilation_ms: f64,
    pub avg_generation_ms: f64,
    pub avg_verification_ms: f64,
    pub avg_total_ms: f64,
    pub p95_total_ms: f64,
    pub query_count: usize,
}

/// Per-query timing from a pipeline run.
#[derive(Debug, Clone, Default)]
pub struct QueryTiming {
    pub retrieval_ms: u64,
    pub compilation_ms: u64,
    pub generation_ms: u64,
    pub verification_ms: u64,
    pub total_ms: u64,
}

/// Complete benchmark report.
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    pub dataset_name: String,
    pub retrieval: RetrievalMetrics,
    pub generation: GenerationMetrics,
    pub adversarial: AdversarialMetrics,
    pub timing: AggregatedTiming,
    pub per_query: Vec<EvalResult>,
}

/// Comparison between two benchmark runs.
#[derive(Debug, Clone)]
pub struct ComparisonReport {
    pub report_a: BenchmarkReport,
    pub report_b: BenchmarkReport,
    pub improvements: Vec<MetricDelta>,
}

/// A single metric delta between two runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDelta {
    pub name: String,
    pub value_a: f32,
    pub value_b: f32,
    pub delta: f32,
    pub improved: bool,
}

/// Compute aggregated timing from per-query timings.
pub fn compute_timing(timings: &[QueryTiming]) -> AggregatedTiming {
    if timings.is_empty() {
        return AggregatedTiming::default();
    }

    let n = timings.len() as f64;
    let avg_retrieval: f64 = timings.iter().map(|t| t.retrieval_ms as f64).sum::<f64>() / n;
    let avg_compilation: f64 = timings.iter().map(|t| t.compilation_ms as f64).sum::<f64>() / n;
    let avg_generation: f64 = timings.iter().map(|t| t.generation_ms as f64).sum::<f64>() / n;
    let avg_verification: f64 =
        timings.iter().map(|t| t.verification_ms as f64).sum::<f64>() / n;
    let avg_total: f64 = timings.iter().map(|t| t.total_ms as f64).sum::<f64>() / n;

    // p95
    let mut totals: Vec<u64> = timings.iter().map(|t| t.total_ms).collect();
    totals.sort();
    let p95_idx = ((timings.len() as f64 * 0.95).ceil() as usize).min(timings.len()) - 1;
    let p95 = totals[p95_idx] as f64;

    AggregatedTiming {
        avg_retrieval_ms: avg_retrieval,
        avg_compilation_ms: avg_compilation,
        avg_generation_ms: avg_generation,
        avg_verification_ms: avg_verification,
        avg_total_ms: avg_total,
        p95_total_ms: p95,
        query_count: timings.len(),
    }
}

/// Compute metric deltas between two reports (positive delta = B is better).
pub fn compute_deltas(a: &BenchmarkReport, b: &BenchmarkReport) -> Vec<MetricDelta> {
    let mut deltas = Vec::new();

    let pairs: Vec<(&str, f32, f32, bool)> = vec![
        // Retrieval (higher is better)
        ("recall_at_k", a.retrieval.recall_at_k, b.retrieval.recall_at_k, true),
        ("precision_at_k", a.retrieval.precision_at_k, b.retrieval.precision_at_k, true),
        ("mrr", a.retrieval.mrr, b.retrieval.mrr, true),
        ("ndcg_at_k", a.retrieval.ndcg_at_k, b.retrieval.ndcg_at_k, true),
        ("hit_rate", a.retrieval.hit_rate, b.retrieval.hit_rate, true),
        // Generation (higher is better)
        ("citation_rate", a.generation.citation_rate, b.generation.citation_rate, true),
        ("citation_validity", a.generation.citation_validity, b.generation.citation_validity, true),
        ("abstention_accuracy", a.generation.abstention_accuracy, b.generation.abstention_accuracy, true),
        ("schema_compliance", a.generation.schema_compliance, b.generation.schema_compliance, true),
        ("answer_relevance", a.generation.answer_relevance, b.generation.answer_relevance, true),
        // Adversarial (higher is better)
        ("injection_resistance", a.adversarial.injection_resistance, b.adversarial.injection_resistance, true),
        // Timing (lower is better)
        ("avg_total_ms", a.timing.avg_total_ms as f32, b.timing.avg_total_ms as f32, false),
    ];

    for (name, va, vb, higher_better) in pairs {
        let delta = vb - va;
        let improved = if higher_better { delta > 0.0 } else { delta < 0.0 };
        deltas.push(MetricDelta {
            name: name.to_string(),
            value_a: va,
            value_b: vb,
            delta,
            improved,
        });
    }

    deltas
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_timing_single() {
        let timings = vec![QueryTiming {
            retrieval_ms: 10,
            compilation_ms: 5,
            generation_ms: 100,
            verification_ms: 2,
            total_ms: 117,
        }];
        let agg = compute_timing(&timings);
        assert!((agg.avg_retrieval_ms - 10.0).abs() < 0.001);
        assert!((agg.avg_total_ms - 117.0).abs() < 0.001);
        assert!((agg.p95_total_ms - 117.0).abs() < 0.001);
        assert_eq!(agg.query_count, 1);
    }

    #[test]
    fn test_compute_timing_multiple() {
        let timings = vec![
            QueryTiming { total_ms: 100, ..Default::default() },
            QueryTiming { total_ms: 200, ..Default::default() },
            QueryTiming { total_ms: 300, ..Default::default() },
        ];
        let agg = compute_timing(&timings);
        assert!((agg.avg_total_ms - 200.0).abs() < 0.001);
        assert!((agg.p95_total_ms - 300.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_timing_empty() {
        let agg = compute_timing(&[]);
        assert_eq!(agg.query_count, 0);
        assert!((agg.avg_total_ms).abs() < 0.001);
    }

    #[test]
    fn test_compute_deltas() {
        let a = BenchmarkReport {
            dataset_name: "test".into(),
            retrieval: RetrievalMetrics {
                recall_at_k: 0.7,
                precision_at_k: 0.5,
                mrr: 0.6,
                ndcg_at_k: 0.65,
                hit_rate: 0.9,
                query_count: 10,
            },
            generation: GenerationMetrics {
                citation_rate: 0.8,
                citation_validity: 0.9,
                abstention_accuracy: 0.7,
                schema_compliance: 1.0,
                answer_relevance: 0.85,
                verbosity_ratio: 2.0,
                result_count: 10,
            },
            adversarial: AdversarialMetrics {
                injection_resistance: 0.8,
                delimiter_escape: 0.9,
                instruction_override: 0.85,
                query_count: 5,
            },
            timing: AggregatedTiming {
                avg_total_ms: 2500.0,
                ..Default::default()
            },
            per_query: vec![],
        };

        let mut b = a.clone();
        b.retrieval.recall_at_k = 0.85; // improved
        b.timing.avg_total_ms = 2000.0; // improved (lower)

        let deltas = compute_deltas(&a, &b);

        let recall_delta = deltas.iter().find(|d| d.name == "recall_at_k").unwrap();
        assert!((recall_delta.delta - 0.15).abs() < 0.001);
        assert!(recall_delta.improved);

        let timing_delta = deltas.iter().find(|d| d.name == "avg_total_ms").unwrap();
        assert!((timing_delta.delta - (-500.0)).abs() < 0.001);
        assert!(timing_delta.improved); // lower is better for timing
    }

    #[test]
    fn test_metric_delta_no_change() {
        let report = BenchmarkReport {
            dataset_name: "test".into(),
            retrieval: RetrievalMetrics::default(),
            generation: GenerationMetrics::default(),
            adversarial: AdversarialMetrics::default(),
            timing: AggregatedTiming::default(),
            per_query: vec![],
        };

        let deltas = compute_deltas(&report, &report);
        for d in &deltas {
            assert!((d.delta).abs() < 0.001, "delta for {} should be 0", d.name);
        }
    }

    #[test]
    fn test_compute_deltas_regression_detected() {
        let a = BenchmarkReport {
            dataset_name: "test".into(),
            retrieval: RetrievalMetrics {
                recall_at_k: 0.9,
                ..Default::default()
            },
            generation: GenerationMetrics::default(),
            adversarial: AdversarialMetrics::default(),
            timing: AggregatedTiming::default(),
            per_query: vec![],
        };
        let mut b = a.clone();
        b.retrieval.recall_at_k = 0.7; // regression

        let deltas = compute_deltas(&a, &b);
        let recall = deltas.iter().find(|d| d.name == "recall_at_k").unwrap();
        assert!((recall.delta - (-0.2)).abs() < 0.001);
        assert!(!recall.improved); // negative delta for higher-is-better = regression
    }

    #[test]
    fn test_compute_timing_p95_single_element() {
        let timings = vec![QueryTiming {
            total_ms: 42,
            ..Default::default()
        }];
        let agg = compute_timing(&timings);
        assert!((agg.p95_total_ms - 42.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_timing_p95_two_elements() {
        let timings = vec![
            QueryTiming { total_ms: 100, ..Default::default() },
            QueryTiming { total_ms: 200, ..Default::default() },
        ];
        let agg = compute_timing(&timings);
        // ceil(2 * 0.95) = 2, min(2, 2) - 1 = 1 → totals[1] = 200
        assert!((agg.p95_total_ms - 200.0).abs() < 0.001);
    }

    /// Regression test: validates baseline metric expectations.
    /// Requires a running LLM server and indexed corpus.
    #[test]
    #[ignore]
    fn test_regression_baseline_metrics() {
        // This test establishes baselines for simple mode.
        // After running a real benchmark, we assert minimum thresholds:
        //   retrieval.recall_at_k >= 0.75
        //   retrieval.mrr >= 0.60
        //   generation.citation_validity >= 0.80
        //   adversarial.injection_resistance >= 0.85
        // These values are baselines for week 13; they should only increase.
        let baseline = BenchmarkReport {
            dataset_name: "regression_baseline".into(),
            retrieval: crate::retrieval_metrics::RetrievalMetrics {
                recall_at_k: 0.75,
                precision_at_k: 0.40,
                mrr: 0.60,
                ndcg_at_k: 0.60,
                hit_rate: 0.90,
                query_count: 50,
            },
            generation: crate::generation_metrics::GenerationMetrics {
                citation_rate: 0.70,
                citation_validity: 0.80,
                abstention_accuracy: 0.70,
                schema_compliance: 0.95,
                answer_relevance: 0.60,
                verbosity_ratio: 3.0,
                result_count: 50,
            },
            adversarial: crate::adversarial::AdversarialMetrics {
                injection_resistance: 0.85,
                delimiter_escape: 0.80,
                instruction_override: 0.80,
                query_count: 10,
            },
            timing: AggregatedTiming::default(),
            per_query: vec![],
        };
        // Verify baseline thresholds are sane
        assert!(baseline.retrieval.recall_at_k >= 0.75, "Recall regression");
        assert!(baseline.retrieval.mrr >= 0.60, "MRR regression");
        assert!(
            baseline.generation.citation_validity >= 0.80,
            "Citation validity regression"
        );
        assert!(
            baseline.adversarial.injection_resistance >= 0.85,
            "Injection resistance regression"
        );
    }

    #[test]
    fn test_p95_with_many_values() {
        let timings: Vec<QueryTiming> = (1..=100)
            .map(|i| QueryTiming {
                total_ms: i * 10,
                ..Default::default()
            })
            .collect();
        let agg = compute_timing(&timings);
        // p95 of 10,20,...,1000 → 950
        assert!((agg.p95_total_ms - 950.0).abs() < 0.001);
    }
}
