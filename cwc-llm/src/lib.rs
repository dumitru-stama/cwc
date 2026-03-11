pub mod grammar;
pub mod llamacpp;
pub mod parse;
pub mod vllm;

use serde::{Deserialize, Serialize};

/// Generation parameters shared across backends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub max_tokens: u32,
    pub repeat_penalty: f32,
    pub stop: Vec<String>,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            temperature: 0.1,
            top_p: 0.9,
            top_k: 40,
            max_tokens: 1024,
            repeat_penalty: 1.1,
            stop: vec![],
        }
    }
}

/// Supported LLM backend types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    LlamaCpp,
    Vllm,
}

impl std::str::FromStr for Backend {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "llamacpp" | "llama.cpp" | "llama" => Ok(Self::LlamaCpp),
            "vllm" => Ok(Self::Vllm),
            other => Err(format!("unknown backend: {other}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_params_default() {
        let params = GenerateParams::default();
        assert!((params.temperature - 0.1).abs() < f32::EPSILON);
        assert_eq!(params.max_tokens, 1024);
        assert!(params.stop.is_empty());
    }

    #[test]
    fn test_backend_parse() {
        assert_eq!("llamacpp".parse::<Backend>().unwrap(), Backend::LlamaCpp);
        assert_eq!("llama.cpp".parse::<Backend>().unwrap(), Backend::LlamaCpp);
        assert_eq!("vllm".parse::<Backend>().unwrap(), Backend::Vllm);
        assert!("unknown".parse::<Backend>().is_err());
    }
}
