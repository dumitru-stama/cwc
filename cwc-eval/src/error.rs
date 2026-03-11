use thiserror::Error;

#[derive(Debug, Error)]
pub enum EvalError {
    #[error("I/O error: {0}")]
    Io(String),

    #[error("parse error: {0}")]
    Parse(String),

    #[error("evaluation error: {0}")]
    Eval(String),
}

impl From<std::io::Error> for EvalError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, EvalError>;
