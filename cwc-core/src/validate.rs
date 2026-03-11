use std::fmt;
use std::path::Path;

use crate::config::{CompilerMode, CwcConfig};

/// Severity of a configuration issue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Error => write!(f, "ERROR"),
            Self::Warning => write!(f, "WARNING"),
        }
    }
}

/// A single configuration warning or error.
#[derive(Debug, Clone)]
pub struct ConfigWarning {
    pub field: String,
    pub message: String,
    pub severity: Severity,
}

impl fmt::Display for ConfigWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.severity, self.field, self.message)
    }
}

/// Validate configuration and return any issues found.
///
/// Errors should abort startup. Warnings should be logged.
/// This function only checks local/static conditions (no network calls).
pub fn validate_config(config: &CwcConfig) -> Vec<ConfigWarning> {
    let mut warnings = Vec::new();

    // Context window minimum
    if config.model.context_window < 2048 {
        warnings.push(ConfigWarning {
            field: "model.context_window".to_string(),
            message: format!(
                "context window {} is below minimum useful size of 2048",
                config.model.context_window
            ),
            severity: Severity::Error,
        });
    }

    // Max output tokens should be less than context window
    if config.model.max_output_tokens >= config.model.context_window {
        warnings.push(ConfigWarning {
            field: "model.max_output_tokens".to_string(),
            message: format!(
                "max_output_tokens ({}) >= context_window ({})",
                config.model.max_output_tokens, config.model.context_window
            ),
            severity: Severity::Error,
        });
    }

    // Budget fractions: check for negative or NaN values
    for (name, val) in [
        ("budget.instruction_fraction", config.budget.instruction_fraction),
        ("budget.sources_fraction", config.budget.sources_fraction),
        ("budget.memory_fraction", config.budget.memory_fraction),
        ("budget.output_fraction", config.budget.output_fraction),
    ] {
        if val.is_nan() {
            warnings.push(ConfigWarning {
                field: name.to_string(),
                message: "fraction is NaN".to_string(),
                severity: Severity::Error,
            });
        } else if val < 0.0 {
            warnings.push(ConfigWarning {
                field: name.to_string(),
                message: format!("fraction {val} is negative"),
                severity: Severity::Error,
            });
        }
    }

    // Budget fractions sum check
    let frac_sum = config.budget.instruction_fraction
        + config.budget.sources_fraction
        + config.budget.memory_fraction
        + config.budget.output_fraction;
    if frac_sum > 1.0 + f32::EPSILON {
        warnings.push(ConfigWarning {
            field: "budget.*_fraction".to_string(),
            message: format!("budget fractions sum to {frac_sum:.2}, exceeding 1.0"),
            severity: Severity::Error,
        });
    }
    if frac_sum < 0.9 {
        warnings.push(ConfigWarning {
            field: "budget.*_fraction".to_string(),
            message: format!(
                "budget fractions sum to {frac_sum:.2}, leaving {:.0}% unused",
                (1.0 - frac_sum) * 100.0
            ),
            severity: Severity::Warning,
        });
    }

    // Top-k values
    for (name, val) in [
        ("retrieval.dense_top_k", config.retrieval.dense_top_k),
        ("retrieval.sparse_top_k", config.retrieval.sparse_top_k),
        ("retrieval.rerank_top_k", config.retrieval.rerank_top_k),
        ("retrieval.final_top_k", config.retrieval.final_top_k),
    ] {
        if val == 0 {
            warnings.push(ConfigWarning {
                field: name.to_string(),
                message: "top_k of 0 means no results will be returned".to_string(),
                severity: Severity::Error,
            });
        }
        if val > 1000 {
            warnings.push(ConfigWarning {
                field: name.to_string(),
                message: format!("top_k of {val} is unusually large, may cause performance issues"),
                severity: Severity::Warning,
            });
        }
    }

    // MMR lambda range
    if !(0.0..=1.0).contains(&config.retrieval.mmr_lambda) {
        warnings.push(ConfigWarning {
            field: "retrieval.mmr_lambda".to_string(),
            message: format!(
                "mmr_lambda {} is outside [0.0, 1.0]",
                config.retrieval.mmr_lambda
            ),
            severity: Severity::Error,
        });
    }

    // RRF k must be positive (used as denominator in 1/(k+rank))
    if config.retrieval.rrf_k <= 0.0 || config.retrieval.rrf_k.is_nan() {
        warnings.push(ConfigWarning {
            field: "retrieval.rrf_k".to_string(),
            message: format!(
                "rrf_k {} must be positive (used as denominator in fusion scoring)",
                config.retrieval.rrf_k
            ),
            severity: Severity::Error,
        });
    }

    // Score threshold sanity
    if config.retrieval.score_threshold < 0.0 || config.retrieval.score_threshold.is_nan() {
        warnings.push(ConfigWarning {
            field: "retrieval.score_threshold".to_string(),
            message: format!(
                "score_threshold {} is negative or NaN",
                config.retrieval.score_threshold
            ),
            severity: Severity::Error,
        });
    }

    // Embedding model file check
    let model_path = Path::new(&config.model.embedding_model);
    if !model_path.exists() {
        warnings.push(ConfigWarning {
            field: "model.embedding_model".to_string(),
            message: format!("embedding model file not found: {}", config.model.embedding_model),
            severity: Severity::Warning,
        });
    }

    // Reranker model check (required for complex mode)
    if config.mode == CompilerMode::Complex {
        match &config.model.rerank_model {
            None => {
                warnings.push(ConfigWarning {
                    field: "model.rerank_model".to_string(),
                    message: "complex mode requires a rerank_model but none configured".to_string(),
                    severity: Severity::Warning,
                });
            }
            Some(path) if !Path::new(path).exists() => {
                warnings.push(ConfigWarning {
                    field: "model.rerank_model".to_string(),
                    message: format!("rerank model file not found: {path}"),
                    severity: Severity::Warning,
                });
            }
            _ => {}
        }
    }

    // Index directory check (read-only — validation should not have side effects)
    let index_dir = Path::new(&config.paths.index_dir);
    if !index_dir.exists() {
        warnings.push(ConfigWarning {
            field: "paths.index_dir".to_string(),
            message: format!(
                "index directory '{}' does not exist",
                config.paths.index_dir
            ),
            severity: Severity::Warning,
        });
    }

    // Data directory check
    let data_dir = Path::new(&config.paths.data_dir);
    if !data_dir.exists() {
        warnings.push(ConfigWarning {
            field: "paths.data_dir".to_string(),
            message: format!(
                "data directory '{}' does not exist",
                config.paths.data_dir
            ),
            severity: Severity::Warning,
        });
    }

    // LLM endpoint format check
    if !config.model.llm_endpoint.starts_with("http://")
        && !config.model.llm_endpoint.starts_with("https://")
    {
        warnings.push(ConfigWarning {
            field: "model.llm_endpoint".to_string(),
            message: format!(
                "endpoint '{}' doesn't start with http:// or https://",
                config.model.llm_endpoint
            ),
            severity: Severity::Error,
        });
    }

    warnings
}

/// Check if any errors exist in the warnings list.
pub fn has_errors(warnings: &[ConfigWarning]) -> bool {
    warnings.iter().any(|w| w.severity == Severity::Error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;

    #[test]
    fn test_validate_valid_config_no_errors() {
        let config = CwcConfig::default();
        let warnings = validate_config(&config);
        let errors: Vec<_> = warnings
            .iter()
            .filter(|w| w.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty(), "default config should have no errors: {errors:?}");
    }

    #[test]
    fn test_validate_missing_model_file() {
        let mut config = CwcConfig::default();
        config.model.embedding_model = "/nonexistent/model.onnx".to_string();
        let warnings = validate_config(&config);
        assert!(warnings
            .iter()
            .any(|w| w.field == "model.embedding_model" && w.severity == Severity::Warning));
    }

    #[test]
    fn test_validate_budget_fractions_exceed_one() {
        let mut config = CwcConfig::default();
        config.budget.instruction_fraction = 0.50;
        config.budget.sources_fraction = 0.50;
        config.budget.memory_fraction = 0.25;
        config.budget.output_fraction = 0.25;
        let warnings = validate_config(&config);
        assert!(warnings
            .iter()
            .any(|w| w.field.contains("fraction") && w.severity == Severity::Error));
    }

    #[test]
    fn test_validate_context_window_too_small() {
        let mut config = CwcConfig::default();
        config.model.context_window = 512;
        let warnings = validate_config(&config);
        assert!(warnings
            .iter()
            .any(|w| w.field == "model.context_window" && w.severity == Severity::Error));
    }

    #[test]
    fn test_validate_zero_top_k() {
        let mut config = CwcConfig::default();
        config.retrieval.final_top_k = 0;
        let warnings = validate_config(&config);
        assert!(warnings
            .iter()
            .any(|w| w.field == "retrieval.final_top_k" && w.severity == Severity::Error));
    }

    #[test]
    fn test_validate_large_top_k_warning() {
        let mut config = CwcConfig::default();
        config.retrieval.dense_top_k = 5000;
        let warnings = validate_config(&config);
        assert!(warnings
            .iter()
            .any(|w| w.field == "retrieval.dense_top_k" && w.severity == Severity::Warning));
    }

    #[test]
    fn test_validate_invalid_llm_endpoint() {
        let mut config = CwcConfig::default();
        config.model.llm_endpoint = "localhost:8080".to_string();
        let warnings = validate_config(&config);
        assert!(warnings
            .iter()
            .any(|w| w.field == "model.llm_endpoint" && w.severity == Severity::Error));
    }

    #[test]
    fn test_validate_max_output_exceeds_context() {
        let mut config = CwcConfig::default();
        config.model.max_output_tokens = 8192;
        config.model.context_window = 4096;
        let warnings = validate_config(&config);
        assert!(warnings
            .iter()
            .any(|w| w.field == "model.max_output_tokens" && w.severity == Severity::Error));
    }

    #[test]
    fn test_validate_complex_mode_no_reranker() {
        let config = CwcConfig {
            mode: CompilerMode::Complex,
            ..Default::default()
        };
        let warnings = validate_config(&config);
        assert!(warnings
            .iter()
            .any(|w| w.field == "model.rerank_model" && w.severity == Severity::Warning));
    }

    #[test]
    fn test_validate_mmr_lambda_out_of_range() {
        let mut config = CwcConfig::default();
        config.retrieval.mmr_lambda = 1.5;
        let warnings = validate_config(&config);
        assert!(warnings
            .iter()
            .any(|w| w.field == "retrieval.mmr_lambda" && w.severity == Severity::Error));
    }

    #[test]
    fn test_has_errors_true() {
        let warnings = vec![ConfigWarning {
            field: "test".to_string(),
            message: "bad".to_string(),
            severity: Severity::Error,
        }];
        assert!(has_errors(&warnings));
    }

    #[test]
    fn test_has_errors_false() {
        let warnings = vec![ConfigWarning {
            field: "test".to_string(),
            message: "ok".to_string(),
            severity: Severity::Warning,
        }];
        assert!(!has_errors(&warnings));
    }

    #[test]
    fn test_validate_low_budget_sum_warning() {
        let mut config = CwcConfig::default();
        config.budget.instruction_fraction = 0.10;
        config.budget.sources_fraction = 0.30;
        config.budget.memory_fraction = 0.05;
        config.budget.output_fraction = 0.05;
        let warnings = validate_config(&config);
        assert!(warnings
            .iter()
            .any(|w| w.field.contains("fraction") && w.severity == Severity::Warning));
    }

    #[test]
    fn test_validate_negative_budget_fraction() {
        let mut config = CwcConfig::default();
        config.budget.sources_fraction = -0.5;
        let warnings = validate_config(&config);
        assert!(warnings
            .iter()
            .any(|w| w.field == "budget.sources_fraction" && w.severity == Severity::Error));
    }

    #[test]
    fn test_validate_nonexistent_dirs_warning() {
        let mut config = CwcConfig::default();
        config.paths.index_dir = "/nonexistent/path/cwc_index_test_12345".to_string();
        config.paths.data_dir = "/nonexistent/path/cwc_data_test_12345".to_string();
        let warnings = validate_config(&config);
        assert!(warnings
            .iter()
            .any(|w| w.field == "paths.index_dir" && w.severity == Severity::Warning));
        assert!(warnings
            .iter()
            .any(|w| w.field == "paths.data_dir" && w.severity == Severity::Warning));
    }

    #[test]
    fn test_validate_context_window_boundary_2048() {
        // Exactly 2048 should NOT trigger an error (minimum useful size)
        let mut config = CwcConfig::default();
        config.model.context_window = 2048;
        config.model.max_output_tokens = 512;
        let warnings = validate_config(&config);
        assert!(!warnings
            .iter()
            .any(|w| w.field == "model.context_window" && w.severity == Severity::Error));
    }

    #[test]
    fn test_validate_max_output_equals_context() {
        // max_output == context_window should be an error (>= check)
        let mut config = CwcConfig::default();
        config.model.context_window = 4096;
        config.model.max_output_tokens = 4096;
        let warnings = validate_config(&config);
        assert!(warnings
            .iter()
            .any(|w| w.field == "model.max_output_tokens" && w.severity == Severity::Error));
    }

    #[test]
    fn test_validate_nan_mmr_lambda() {
        let mut config = CwcConfig::default();
        config.retrieval.mmr_lambda = f32::NAN;
        let warnings = validate_config(&config);
        // NaN is outside [0.0, 1.0] — .contains() returns false for NaN
        assert!(warnings
            .iter()
            .any(|w| w.field == "retrieval.mmr_lambda" && w.severity == Severity::Error));
    }

    #[test]
    fn test_validate_display_format() {
        let w = ConfigWarning {
            field: "test.field".to_string(),
            message: "something wrong".to_string(),
            severity: Severity::Error,
        };
        assert_eq!(format!("{w}"), "[ERROR] test.field: something wrong");
    }

    #[test]
    fn test_validate_has_errors_empty() {
        assert!(!has_errors(&[]));
    }

    #[test]
    fn test_validate_nan_budget_fraction() {
        let mut config = CwcConfig::default();
        config.budget.instruction_fraction = f32::NAN;
        let warnings = validate_config(&config);
        assert!(warnings
            .iter()
            .any(|w| w.field == "budget.instruction_fraction"
                && w.severity == Severity::Error
                && w.message.contains("NaN")));
    }

    #[test]
    fn test_validate_rrf_k_zero() {
        let mut config = CwcConfig::default();
        config.retrieval.rrf_k = 0.0;
        let warnings = validate_config(&config);
        assert!(warnings
            .iter()
            .any(|w| w.field == "retrieval.rrf_k" && w.severity == Severity::Error));
    }

    #[test]
    fn test_validate_rrf_k_negative() {
        let mut config = CwcConfig::default();
        config.retrieval.rrf_k = -10.0;
        let warnings = validate_config(&config);
        assert!(warnings
            .iter()
            .any(|w| w.field == "retrieval.rrf_k" && w.severity == Severity::Error));
    }

    #[test]
    fn test_validate_score_threshold_negative() {
        let mut config = CwcConfig::default();
        config.retrieval.score_threshold = -0.5;
        let warnings = validate_config(&config);
        assert!(warnings
            .iter()
            .any(|w| w.field == "retrieval.score_threshold" && w.severity == Severity::Error));
    }

    #[test]
    fn test_validate_rrf_k_nan() {
        let mut config = CwcConfig::default();
        config.retrieval.rrf_k = f32::NAN;
        let warnings = validate_config(&config);
        assert!(warnings
            .iter()
            .any(|w| w.field == "retrieval.rrf_k" && w.severity == Severity::Error));
    }
}
