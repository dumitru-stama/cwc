pub mod cache;
pub mod config;
pub mod error;
pub mod onnx;
pub mod pipeline;

pub use cache::{content_hash, EmbeddingCache};
pub use config::EmbedConfig;
pub use error::{EmbedError, Result};
pub use onnx::{cosine_similarity, OnnxEmbedder};
pub use pipeline::embed_chunks;
