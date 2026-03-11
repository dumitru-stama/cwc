use thiserror::Error;

#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("ort error: {0}")]
    Ort(String),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("embed error: {0}")]
    Embed(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

impl From<ort::Error> for EmbedError {
    fn from(e: ort::Error) -> Self {
        Self::Ort(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, EmbedError>;
