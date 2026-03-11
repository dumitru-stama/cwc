use thiserror::Error;

#[derive(Debug, Error)]
pub enum IngestError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("loader error: {0}")]
    Loader(String),

    #[error("glob pattern error: {0}")]
    Glob(#[from] glob::PatternError),

    #[error("glob match error: {0}")]
    GlobIter(#[from] glob::GlobError),

    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] cwc_core::CwcError),
}

pub type Result<T> = std::result::Result<T, IngestError>;
