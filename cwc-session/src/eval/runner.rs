use std::sync::Arc;

use cwc_core::traits::TokenCounter;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::manager::{SessionManager, SessionManagerConfig};
use crate::message::Session;

use super::benchmark::BenchmarkConversation;
use super::metrics::{
    compute_efficiency, compute_goal_retention, compute_memory_quality, compute_repetition,
    compute_turn_survival, GoalRetention, SessionMetrics,
};

/// Run benchmarks against a SessionManager configuration.
pub struct SessionEvalRunner {
    config: SessionManagerConfig,
    tokenizer: Arc<dyn TokenCounter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEvalReport {
    pub model_profile: String,
    pub benchmarks: Vec<(String, SessionMetrics)>,
    pub aggregate: AggregateSessionMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateSessionMetrics {
    pub avg_survival_rate: f32,
    pub avg_goal_retention_similarity: f32,
    pub avg_repetition_rate: f32,
    pub avg_tokens_per_turn: f32,
    pub total_sliding_windows: usize,
    pub total_hard_resets: usize,
    pub memory_precision: f32,
    pub memory_recall: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionComparisonReport {
    pub config_a_name: String,
    pub config_b_name: String,
    pub per_benchmark: Vec<(String, SessionMetrics, SessionMetrics)>,
    pub improvements: Vec<MetricDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDelta {
    pub metric: String,
    pub value_a: f32,
    pub value_b: f32,
    pub delta: f32,
    pub improved: bool,
}

impl SessionEvalRunner {
    pub fn new(config: SessionManagerConfig, tokenizer: Arc<dyn TokenCounter>) -> Self {
        Self { config, tokenizer }
    }

    /// Run a single benchmark and return metrics.
    pub fn run_benchmark(&self, benchmark: &BenchmarkConversation) -> Result<SessionMetrics> {
        let mut mgr = SessionManager::new(self.config.clone(), self.tokenizer.clone())?;
        let mut session = Session::new(self.tokenizer.clone());
        let mut reports = Vec::new();

        // Feed messages one at a time through the manager
        for msg in &benchmark.messages {
            let report = mgr.process_message(&mut session, msg.clone())?;
            reports.push(report);
        }

        // Compute turn survival
        let total_turns = count_user_turns(&benchmark.messages);
        let turn_survival = compute_turn_survival(total_turns, &benchmark.hallucination_turns);

        // Compute goal retention: probe at the last turn
        let goal_retention = if let Some(goal) = mgr.goal() {
            compute_goal_retention(&benchmark.goal, goal, total_turns)
        } else {
            GoalRetention {
                probe_turn: total_turns,
                retained: false,
                similarity: 0.0,
            }
        };

        // Compute repetition from the conversation
        let repetition = compute_repetition(session.messages());

        // Compute efficiency
        let usable_budget = mgr.budget().usable_budget;
        let efficiency = compute_efficiency(&reports, usable_budget);

        // Compute memory quality
        let store = mgr.memory();
        let extracted_keys: Vec<String> = store.all().iter().map(|f| f.key.clone()).collect();
        let extracted_values: Vec<String> = store.all().iter().map(|f| f.value.clone()).collect();
        let expected: Vec<(String, String)> = benchmark
            .expected_facts
            .iter()
            .map(|f| (f.key_pattern.clone(), f.value_contains.clone()))
            .collect();
        let memory_quality = compute_memory_quality(&extracted_keys, &extracted_values, &expected);

        Ok(SessionMetrics {
            turn_survival,
            goal_retention,
            repetition,
            efficiency,
            memory_quality,
        })
    }

    /// Run all built-in benchmarks and return aggregate results.
    pub fn run_all(&self) -> Result<SessionEvalReport> {
        let benchmarks = vec![
            super::benchmark::benchmark_short_coding(),
            super::benchmark::benchmark_long_exploration(),
            super::benchmark::benchmark_tool_heavy(),
            super::benchmark::benchmark_context_pressure(),
            super::benchmark::benchmark_hallucination(),
            super::benchmark::benchmark_repetition(),
        ];

        let mut results = Vec::new();
        for bench in &benchmarks {
            let metrics = self.run_benchmark(bench)?;
            results.push((bench.name.clone(), metrics));
        }

        let aggregate = compute_aggregate(&results);

        Ok(SessionEvalReport {
            model_profile: format!("{:?}", self.config.session.model),
            benchmarks: results,
            aggregate,
        })
    }

    /// A/B comparison: run benchmarks with two different configs.
    pub fn compare(
        config_a: &SessionManagerConfig,
        config_b: &SessionManagerConfig,
        tokenizer: Arc<dyn TokenCounter>,
    ) -> Result<SessionComparisonReport> {
        let runner_a = Self::new(config_a.clone(), tokenizer.clone());
        let runner_b = Self::new(config_b.clone(), tokenizer);

        let report_a = runner_a.run_all()?;
        let report_b = runner_b.run_all()?;

        let mut per_benchmark = Vec::new();
        for ((name_a, metrics_a), (_, metrics_b)) in
            report_a.benchmarks.iter().zip(report_b.benchmarks.iter())
        {
            per_benchmark.push((name_a.clone(), metrics_a.clone(), metrics_b.clone()));
        }

        let improvements = compute_deltas(&report_a.aggregate, &report_b.aggregate);

        Ok(SessionComparisonReport {
            config_a_name: format!("{:?}", config_a.session.model),
            config_b_name: format!("{:?}", config_b.session.model),
            per_benchmark,
            improvements,
        })
    }
}

fn count_user_turns(messages: &[crate::message::SessionMessage]) -> usize {
    messages
        .iter()
        .filter(|m| m.role == crate::message::SessionRole::User)
        .count()
}

fn compute_aggregate(results: &[(String, SessionMetrics)]) -> AggregateSessionMetrics {
    if results.is_empty() {
        return AggregateSessionMetrics {
            avg_survival_rate: 0.0,
            avg_goal_retention_similarity: 0.0,
            avg_repetition_rate: 0.0,
            avg_tokens_per_turn: 0.0,
            total_sliding_windows: 0,
            total_hard_resets: 0,
            memory_precision: 0.0,
            memory_recall: 0.0,
        };
    }
    let n = results.len() as f32;
    let mut total_survival = 0.0f32;
    let mut total_goal_sim = 0.0f32;
    let mut total_rep = 0.0f32;
    let mut total_tpt = 0.0f32;
    let mut total_sw = 0usize;
    let mut total_hr = 0usize;
    let mut total_prec = 0.0f32;
    let mut total_recall = 0.0f32;

    for (_, m) in results {
        total_survival += m.turn_survival.survival_rate;
        total_goal_sim += m.goal_retention.similarity;
        total_rep += m.repetition.repetition_rate;
        total_tpt += m.efficiency.avg_tokens_per_turn;
        total_sw += m.efficiency.sliding_window_count;
        total_hr += m.efficiency.hard_reset_count;
        total_prec += m.memory_quality.precision;
        total_recall += m.memory_quality.recall;
    }

    AggregateSessionMetrics {
        avg_survival_rate: total_survival / n,
        avg_goal_retention_similarity: total_goal_sim / n,
        avg_repetition_rate: total_rep / n,
        avg_tokens_per_turn: total_tpt / n,
        total_sliding_windows: total_sw,
        total_hard_resets: total_hr,
        memory_precision: total_prec / n,
        memory_recall: total_recall / n,
    }
}

fn compute_deltas(a: &AggregateSessionMetrics, b: &AggregateSessionMetrics) -> Vec<MetricDelta> {
    vec![
        delta("survival_rate", a.avg_survival_rate, b.avg_survival_rate, true),
        delta(
            "goal_retention",
            a.avg_goal_retention_similarity,
            b.avg_goal_retention_similarity,
            true,
        ),
        delta("repetition_rate", a.avg_repetition_rate, b.avg_repetition_rate, false),
        delta(
            "tokens_per_turn",
            a.avg_tokens_per_turn,
            b.avg_tokens_per_turn,
            false,
        ),
        delta("memory_precision", a.memory_precision, b.memory_precision, true),
        delta("memory_recall", a.memory_recall, b.memory_recall, true),
    ]
}

fn delta(name: &str, a: f32, b: f32, higher_is_better: bool) -> MetricDelta {
    let d = b - a;
    MetricDelta {
        metric: name.into(),
        value_a: a,
        value_b: b,
        delta: d,
        improved: if higher_is_better { d > 0.0 } else { d < 0.0 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::benchmark;

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

    fn test_config() -> SessionManagerConfig {
        let dir = std::env::temp_dir().join("cwc_eval_test");
        let _ = std::fs::create_dir_all(&dir);
        SessionManagerConfig {
            artifact_dir: dir,
            ..Default::default()
        }
    }

    #[test]
    fn test_run_benchmark_produces_metrics() {
        let runner = SessionEvalRunner::new(test_config(), tc());
        let bench = benchmark::benchmark_short_coding();
        let metrics = runner.run_benchmark(&bench).unwrap();
        assert!(metrics.turn_survival.total_turns > 0);
        assert!(metrics.turn_survival.survival_rate > 0.0);
    }

    #[test]
    fn test_run_all_produces_aggregate() {
        let runner = SessionEvalRunner::new(test_config(), tc());
        let report = runner.run_all().unwrap();
        assert_eq!(report.benchmarks.len(), 6);
        assert!(report.aggregate.avg_survival_rate > 0.0);
    }

    #[test]
    fn test_compare_produces_deltas() {
        let config_a = test_config();
        let mut config_b = test_config();
        config_b.reinforcement.enabled = false;

        let report = SessionEvalRunner::compare(&config_a, &config_b, tc()).unwrap();
        assert_eq!(report.per_benchmark.len(), 6);
        assert!(!report.improvements.is_empty());
    }

    #[test]
    fn test_compare_managed_vs_unmanaged() {
        // Managed config (default)
        let managed = test_config();
        // "Unmanaged" — very high thresholds so no trimming/nudging
        let mut unmanaged = test_config();
        unmanaged.reinforcement.enabled = false;
        unmanaged.session.sliding_window_fraction = 0.99;
        unmanaged.session.hard_reset_fraction = 0.99;

        let report = SessionEvalRunner::compare(&managed, &unmanaged, tc()).unwrap();
        // Both should produce valid reports
        assert_eq!(report.per_benchmark.len(), 6);
        for delta in &report.improvements {
            // Just verify the delta computation works
            assert!((delta.delta - (delta.value_b - delta.value_a)).abs() < 0.001);
        }
    }
}
