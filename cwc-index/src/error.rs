use thiserror::Error;

#[derive(Debug, Error)]
pub enum IndexError {
    #[error("tantivy error: {0}")]
    Tantivy(#[from] tantivy::TantivyError),

    #[error("tantivy query parse error: {0}")]
    QueryParse(#[from] tantivy::query::QueryParserError),

    #[error("tantivy open directory error: {0}")]
    OpenDirectory(#[from] tantivy::directory::error::OpenDirectoryError),

    #[error("tantivy open read error: {0}")]
    OpenRead(#[from] tantivy::directory::error::OpenReadError),

    #[error("index error: {0}")]
    Index(String),

    #[error("database error: {0}")]
    Database(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, IndexError>;
