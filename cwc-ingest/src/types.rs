use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;

/// A raw document loaded from disk, before chunking.
#[derive(Debug, Clone)]
pub struct RawDocument {
    pub doc_id: Uuid,
    pub source_path: PathBuf,
    pub doc_version: u32,
    pub content: String,
    pub sections: Vec<Section>,
    pub metadata: HashMap<String, String>,
}

/// A structural section within a document.
#[derive(Debug, Clone, PartialEq)]
pub struct Section {
    pub path: Vec<String>,
    pub text: String,
    pub char_offset: usize,
    pub char_len: usize,
}
