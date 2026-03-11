use thiserror::Error;

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("invalid message format: {0}")]
    InvalidFormat(String),

    #[error("missing required field: {0}")]
    MissingField(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("llm error: {0}")]
    Llm(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, SessionError>;
