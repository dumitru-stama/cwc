use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Configuration for the embedding model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedConfig {
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    #[serde(default = "default_dim")]
    pub dim: usize,
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,
    #[serde(default = "default_true")]
    pub normalize: bool,
    #[serde(default)]
    pub query_prefix: Option<String>,
    #[serde(default)]
    pub doc_prefix: Option<String>,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_num_threads")]
    pub num_threads: usize,
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/bge-small-en-v1.5/model.onnx"),
            tokenizer_path: PathBuf::from("models/bge-small-en-v1.5/tokenizer.json"),
            dim: default_dim(),
            max_seq_len: default_max_seq_len(),
            normalize: true,
            query_prefix: None,
            doc_prefix: None,
            batch_size: default_batch_size(),
            num_threads: default_num_threads(),
        }
    }
}

fn default_dim() -> usize {
    384
}

fn default_max_seq_len() -> usize {
    512
}

fn default_true() -> bool {
    true
}

fn default_batch_size() -> usize {
    32
}

fn default_num_threads() -> usize {
    4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_config_default() {
        let config = EmbedConfig::default();
        assert_eq!(config.dim, 384);
        assert_eq!(config.max_seq_len, 512);
        assert!(config.normalize);
        assert!(config.query_prefix.is_none());
        assert!(config.doc_prefix.is_none());
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.num_threads, 4);
    }

    #[test]
    fn test_embed_config_serde_roundtrip() {
        let config = EmbedConfig {
            model_path: PathBuf::from("models/e5-small/model.onnx"),
            tokenizer_path: PathBuf::from("models/e5-small/tokenizer.json"),
            dim: 768,
            max_seq_len: 256,
            normalize: false,
            query_prefix: Some("query: ".to_string()),
            doc_prefix: Some("passage: ".to_string()),
            batch_size: 16,
            num_threads: 8,
        };
        let json = serde_json::to_string(&config).unwrap();
        let de: EmbedConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(de.dim, 768);
        assert_eq!(de.max_seq_len, 256);
        assert!(!de.normalize);
        assert_eq!(de.query_prefix.as_deref(), Some("query: "));
        assert_eq!(de.doc_prefix.as_deref(), Some("passage: "));
        assert_eq!(de.batch_size, 16);
        assert_eq!(de.num_threads, 8);
    }

    #[test]
    fn test_embed_config_deserialize_with_defaults() {
        let json = r#"{"model_path": "m.onnx", "tokenizer_path": "t.json"}"#;
        let config: EmbedConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.dim, 384);
        assert_eq!(config.max_seq_len, 512);
        assert!(config.normalize);
        assert_eq!(config.batch_size, 32);
    }
}
