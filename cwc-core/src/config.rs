use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::error::{CwcError, Result};

/// Top-level configuration for the context window compiler.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CwcConfig {
    #[serde(default)]
    pub mode: CompilerMode,
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub retrieval: RetrievalConfig,
    #[serde(default)]
    pub budget: BudgetConfig,
    #[serde(default)]
    pub verify: VerifyConfig,
    #[serde(default)]
    pub paths: PathConfig,
}

impl CwcConfig {
    /// Load configuration from a TOML file, falling back to defaults for
    /// any missing fields.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let contents = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            CwcError::Config(format!("failed to read config file {:?}: {}", path.as_ref(), e))
        })?;
        let config: CwcConfig = toml::from_str(&contents)?;
        Ok(config)
    }
}

/// Pipeline mode: Simple (heuristic) or Complex (neural rerank + CoVe).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CompilerMode {
    #[default]
    Simple,
    Complex,
}

/// Model endpoint and parameter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default = "default_llm_endpoint")]
    pub llm_endpoint: String,
    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,
    #[serde(default)]
    pub rerank_model: Option<String>,
    #[serde(default = "default_context_window")]
    pub context_window: u32,
    #[serde(default = "default_max_output_tokens")]
    pub max_output_tokens: u32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            llm_endpoint: default_llm_endpoint(),
            embedding_model: default_embedding_model(),
            rerank_model: None,
            context_window: default_context_window(),
            max_output_tokens: default_max_output_tokens(),
        }
    }
}

fn default_llm_endpoint() -> String {
    "http://localhost:8080".to_string()
}

fn default_embedding_model() -> String {
    "models/bge-small-en-v1.5.onnx".to_string()
}

fn default_context_window() -> u32 {
    4096
}

fn default_max_output_tokens() -> u32 {
    1024
}

/// Retrieval pipeline tuning parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    #[serde(default = "default_top_k")]
    pub dense_top_k: usize,
    #[serde(default = "default_top_k")]
    pub sparse_top_k: usize,
    #[serde(default = "default_rerank_top_k")]
    pub rerank_top_k: usize,
    #[serde(default = "default_final_top_k")]
    pub final_top_k: usize,
    #[serde(default = "default_mmr_lambda")]
    pub mmr_lambda: f32,
    #[serde(default = "default_score_threshold")]
    pub score_threshold: f32,
    #[serde(default = "default_rrf_k")]
    pub rrf_k: f32,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            dense_top_k: default_top_k(),
            sparse_top_k: default_top_k(),
            rerank_top_k: default_rerank_top_k(),
            final_top_k: default_final_top_k(),
            mmr_lambda: default_mmr_lambda(),
            score_threshold: default_score_threshold(),
            rrf_k: default_rrf_k(),
        }
    }
}

fn default_top_k() -> usize {
    50
}

fn default_rerank_top_k() -> usize {
    20
}

fn default_final_top_k() -> usize {
    10
}

fn default_mmr_lambda() -> f32 {
    0.7
}

fn default_score_threshold() -> f32 {
    0.0
}

fn default_rrf_k() -> f32 {
    60.0
}

/// Token budget fractional allocations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConfig {
    #[serde(default = "default_instruction_fraction")]
    pub instruction_fraction: f32,
    #[serde(default = "default_sources_fraction")]
    pub sources_fraction: f32,
    #[serde(default = "default_memory_fraction")]
    pub memory_fraction: f32,
    #[serde(default = "default_output_fraction")]
    pub output_fraction: f32,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            instruction_fraction: default_instruction_fraction(),
            sources_fraction: default_sources_fraction(),
            memory_fraction: default_memory_fraction(),
            output_fraction: default_output_fraction(),
        }
    }
}

fn default_instruction_fraction() -> f32 {
    0.15
}

fn default_sources_fraction() -> f32 {
    0.65
}

fn default_memory_fraction() -> f32 {
    0.10
}

fn default_output_fraction() -> f32 {
    0.10
}

/// Verification settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyConfig {
    #[serde(default = "default_true")]
    pub check_citations: bool,
    #[serde(default = "default_true")]
    pub check_schema: bool,
    #[serde(default = "default_true")]
    pub check_abstention: bool,
}

impl Default for VerifyConfig {
    fn default() -> Self {
        Self {
            check_citations: true,
            check_schema: true,
            check_abstention: true,
        }
    }
}

fn default_true() -> bool {
    true
}

/// File system paths used by the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathConfig {
    #[serde(default = "default_data_dir")]
    pub data_dir: String,
    #[serde(default = "default_index_dir")]
    pub index_dir: String,
    #[serde(default = "default_models_dir")]
    pub models_dir: String,
}

impl Default for PathConfig {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
            index_dir: default_index_dir(),
            models_dir: default_models_dir(),
        }
    }
}

fn default_data_dir() -> String {
    "data".to_string()
}

fn default_index_dir() -> String {
    "index".to_string()
}

fn default_models_dir() -> String {
    "models".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_loads_from_toml_with_defaults() {
        let toml_str = r#"
# mode omitted — should default to Simple

[model]
llm_endpoint = "http://localhost:9090"
context_window = 8192
"#;
        let config: CwcConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.mode, CompilerMode::Simple);
        assert_eq!(config.model.llm_endpoint, "http://localhost:9090");
        assert_eq!(config.model.context_window, 8192);
        assert_eq!(config.model.max_output_tokens, 1024); // default
        assert_eq!(config.retrieval.dense_top_k, 50); // default
        assert!((config.budget.sources_fraction - 0.65).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_default() {
        let config = CwcConfig::default();
        assert_eq!(config.mode, CompilerMode::Simple);
        assert_eq!(config.model.context_window, 4096);
        assert_eq!(config.retrieval.final_top_k, 10);
        assert!((config.budget.instruction_fraction - 0.15).abs() < f32::EPSILON);
        assert!((config.budget.sources_fraction - 0.65).abs() < f32::EPSILON);
        assert!((config.budget.memory_fraction - 0.10).abs() < f32::EPSILON);
        assert!((config.budget.output_fraction - 0.10).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_complex_mode() {
        let toml_str = r#"
mode = "complex"

[model]
rerank_model = "models/cross-encoder"
"#;
        let config: CwcConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.mode, CompilerMode::Complex);
        assert_eq!(
            config.model.rerank_model.as_deref(),
            Some("models/cross-encoder")
        );
    }

    #[test]
    fn test_config_load_missing_file() {
        let result = CwcConfig::load("/nonexistent/path/cwc.toml");
        assert!(result.is_err());
    }

    #[test]
    fn test_config_load_from_file() {
        let dir = std::env::temp_dir().join("cwc_test_config");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cwc.toml");
        std::fs::write(
            &path,
            r#"
mode = "simple"

[model]
llm_endpoint = "http://127.0.0.1:5000"
context_window = 16384
max_output_tokens = 2048

[budget]
instruction_fraction = 0.20
sources_fraction = 0.60
"#,
        )
        .unwrap();

        let config = CwcConfig::load(&path).unwrap();
        assert_eq!(config.mode, CompilerMode::Simple);
        assert_eq!(config.model.llm_endpoint, "http://127.0.0.1:5000");
        assert_eq!(config.model.context_window, 16384);
        assert_eq!(config.model.max_output_tokens, 2048);
        assert!((config.budget.instruction_fraction - 0.20).abs() < f32::EPSILON);
        assert!((config.budget.sources_fraction - 0.60).abs() < f32::EPSILON);
        // Unspecified fields get defaults
        assert!((config.budget.memory_fraction - 0.10).abs() < f32::EPSILON);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_config_toml_roundtrip() {
        let config = CwcConfig::default();
        let toml_str = toml::to_string(&config).unwrap();
        let de: CwcConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(de.mode, config.mode);
        assert_eq!(de.model.context_window, config.model.context_window);
        assert_eq!(de.retrieval.final_top_k, config.retrieval.final_top_k);
        assert!(
            (de.budget.sources_fraction - config.budget.sources_fraction).abs() < f32::EPSILON
        );
    }

    #[test]
    fn test_config_all_fields_from_toml() {
        let toml_str = r#"
mode = "complex"

[model]
llm_endpoint = "http://gpu-box:8080"
embedding_model = "models/e5-large.onnx"
rerank_model = "models/ce-large"
context_window = 32768
max_output_tokens = 4096

[retrieval]
dense_top_k = 100
sparse_top_k = 100
rerank_top_k = 30
final_top_k = 15
mmr_lambda = 0.5
score_threshold = 0.1

[budget]
instruction_fraction = 0.10
sources_fraction = 0.70
memory_fraction = 0.05
output_fraction = 0.15

[verify]
check_citations = true
check_schema = false
check_abstention = true

[paths]
data_dir = "/data/corpus"
index_dir = "/data/index"
models_dir = "/opt/models"
"#;
        let config: CwcConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.mode, CompilerMode::Complex);
        assert_eq!(config.model.llm_endpoint, "http://gpu-box:8080");
        assert_eq!(config.model.embedding_model, "models/e5-large.onnx");
        assert_eq!(config.model.rerank_model.as_deref(), Some("models/ce-large"));
        assert_eq!(config.model.context_window, 32768);
        assert_eq!(config.model.max_output_tokens, 4096);
        assert_eq!(config.retrieval.dense_top_k, 100);
        assert_eq!(config.retrieval.sparse_top_k, 100);
        assert_eq!(config.retrieval.rerank_top_k, 30);
        assert_eq!(config.retrieval.final_top_k, 15);
        assert!((config.retrieval.mmr_lambda - 0.5).abs() < f32::EPSILON);
        assert!((config.retrieval.score_threshold - 0.1).abs() < f32::EPSILON);
        assert!((config.budget.instruction_fraction - 0.10).abs() < f32::EPSILON);
        assert!((config.budget.sources_fraction - 0.70).abs() < f32::EPSILON);
        assert!((config.budget.memory_fraction - 0.05).abs() < f32::EPSILON);
        assert!((config.budget.output_fraction - 0.15).abs() < f32::EPSILON);
        assert!(config.verify.check_citations);
        assert!(!config.verify.check_schema);
        assert!(config.verify.check_abstention);
        assert_eq!(config.paths.data_dir, "/data/corpus");
        assert_eq!(config.paths.index_dir, "/data/index");
        assert_eq!(config.paths.models_dir, "/opt/models");
    }

    #[test]
    fn test_config_empty_toml() {
        // Completely empty TOML should produce all defaults
        let config: CwcConfig = toml::from_str("").unwrap();
        assert_eq!(config.mode, CompilerMode::Simple);
        assert_eq!(config.model.llm_endpoint, "http://localhost:8080");
        assert_eq!(config.model.context_window, 4096);
    }

    #[test]
    fn test_config_invalid_mode() {
        let toml_str = r#"mode = "turbo""#;
        let result: std::result::Result<CwcConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }
}
