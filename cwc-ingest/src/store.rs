use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::Path;

use cwc_core::types::Chunk;

use crate::error::Result;

const CHUNKS_FILE: &str = "chunks.jsonl";

/// Save chunks to a JSONL file in the given directory.
pub fn save_chunks(chunks: &[Chunk], dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(CHUNKS_FILE);
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);

    for chunk in chunks {
        let line = serde_json::to_string(chunk)?;
        writeln!(writer, "{line}")?;
    }

    writer.flush()?;
    Ok(())
}

/// Load chunks from a JSONL file in the given directory.
pub fn load_chunks(dir: &Path) -> Result<Vec<Chunk>> {
    let path = dir.join(CHUNKS_FILE);
    if !path.exists() {
        return Ok(Vec::new());
    }

    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut chunks = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let chunk: Chunk = serde_json::from_str(&line)?;
        chunks.push(chunk);
    }

    Ok(chunks)
}

/// Upsert chunks: merge new chunks with existing ones by chunk_id.
/// If a chunk with the same chunk_id exists but has a different doc_version,
/// the new one replaces it.
pub fn upsert_chunks(new_chunks: &[Chunk], dir: &Path) -> Result<()> {
    let mut existing = load_chunks(dir)?;

    // Index existing by chunk_id
    let mut by_id: HashMap<uuid::Uuid, usize> = HashMap::new();
    for (i, chunk) in existing.iter().enumerate() {
        by_id.insert(chunk.chunk_id, i);
    }

    for new in new_chunks {
        if let Some(&idx) = by_id.get(&new.chunk_id) {
            // Replace if version differs
            if existing[idx].doc_version != new.doc_version {
                existing[idx] = new.clone();
            }
        } else {
            by_id.insert(new.chunk_id, existing.len());
            existing.push(new.clone());
        }
    }

    save_chunks(&existing, dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use uuid::Uuid;

    fn make_chunk(id: Uuid, version: u32, text: &str) -> Chunk {
        Chunk {
            chunk_id: id,
            doc_id: Uuid::new_v4(),
            doc_version: version,
            source_path: "test.md".to_string(),
            section_path: vec!["Section".to_string()],
            char_offset: 0,
            char_len: text.len(),
            token_count: 5,
            text: text.to_string(),
            metadata: HashMap::new(),
        }
    }

    fn test_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("cwc_store_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_store_roundtrip() {
        let dir = test_dir("roundtrip");
        let chunks = vec![
            make_chunk(Uuid::new_v4(), 1, "chunk one"),
            make_chunk(Uuid::new_v4(), 1, "chunk two"),
        ];

        save_chunks(&chunks, &dir).unwrap();
        let loaded = load_chunks(&dir).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].text, chunks[0].text);
        assert_eq!(loaded[1].text, chunks[1].text);
        assert_eq!(loaded[0].chunk_id, chunks[0].chunk_id);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_store_load_empty() {
        let dir = test_dir("load_empty");
        let loaded = load_chunks(&dir).unwrap();
        assert!(loaded.is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_store_upsert_new() {
        let dir = test_dir("upsert_new");
        let c1 = make_chunk(Uuid::new_v4(), 1, "original");
        save_chunks(std::slice::from_ref(&c1), &dir).unwrap();

        let c2 = make_chunk(Uuid::new_v4(), 1, "new chunk");
        upsert_chunks(std::slice::from_ref(&c2), &dir).unwrap();

        let loaded = load_chunks(&dir).unwrap();
        assert_eq!(loaded.len(), 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_store_upsert_replaces_on_version_change() {
        let dir = test_dir("upsert_replace");
        let id = Uuid::new_v4();
        let c1 = make_chunk(id, 1, "version one");
        save_chunks(&[c1], &dir).unwrap();

        let c2 = make_chunk(id, 2, "version two");
        upsert_chunks(&[c2], &dir).unwrap();

        let loaded = load_chunks(&dir).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].text, "version two");
        assert_eq!(loaded[0].doc_version, 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_store_upsert_same_version_no_change() {
        let dir = test_dir("upsert_same_ver");
        let id = Uuid::new_v4();
        let c1 = make_chunk(id, 1, "original text");
        save_chunks(&[c1], &dir).unwrap();

        let c2 = make_chunk(id, 1, "different text same version");
        upsert_chunks(&[c2], &dir).unwrap();

        let loaded = load_chunks(&dir).unwrap();
        assert_eq!(loaded.len(), 1);
        // Should keep original since version didn't change
        assert_eq!(loaded[0].text, "original text");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
