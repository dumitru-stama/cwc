use thiserror::Error;

#[derive(Debug, Error)]
pub enum CwcError {
    #[error("config error: {0}")]
    Config(String),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("retrieval error: {0}")]
    Retrieval(String),

    #[error("embedding error: {0}")]
    Embedding(String),

    #[error("llm error: {0}")]
    Llm(String),

    #[error("verification error: {0}")]
    Verification(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("toml parse error: {0}")]
    Toml(#[from] toml::de::Error),
}

pub type Result<T> = std::result::Result<T, CwcError>;
