use std::collections::HashMap;
use uuid::Uuid;

use cwc_core::types::Chunk;
use cwc_core::Tokenizer;

use crate::error::Result;
use crate::types::RawDocument;

/// Namespace for deterministic chunk ID generation.
const CHUNK_NS: Uuid = Uuid::from_bytes([
    0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30,
    0xc8,
]);

/// Generate a deterministic chunk ID from doc_id, char_offset, and char_len.
pub fn chunk_id(doc_id: &Uuid, char_offset: usize, char_len: usize) -> Uuid {
    let mut data = doc_id.as_bytes().to_vec();
    data.extend_from_slice(&(char_offset as u64).to_le_bytes());
    data.extend_from_slice(&(char_len as u64).to_le_bytes());
    Uuid::new_v5(&CHUNK_NS, &data)
}

/// Chunking strategy trait.
pub trait Chunker: Send + Sync {
    fn chunk(&self, doc: &RawDocument, tokenizer: &Tokenizer) -> Result<Vec<Chunk>>;
}

/// Recursive separator-based chunker. Splits on paragraph boundaries first,
/// then lines, then sentences, then token boundaries.
pub struct RecursiveChunker {
    pub max_chunk_tokens: u32,
    pub overlap_tokens: u32,
}

impl RecursiveChunker {
    pub fn new(max_chunk_tokens: u32, overlap_tokens: u32) -> Self {
        Self {
            max_chunk_tokens,
            overlap_tokens,
        }
    }
}

impl Chunker for RecursiveChunker {
    fn chunk(&self, doc: &RawDocument, tokenizer: &Tokenizer) -> Result<Vec<Chunk>> {
        let mut chunks = Vec::new();

        for section in &doc.sections {
            if section.text.is_empty() {
                continue;
            }
            let section_chunks = chunk_section(
                &section.text,
                section.char_offset,
                &section.path,
                doc,
                tokenizer,
                self.max_chunk_tokens,
                self.overlap_tokens,
            );
            chunks.extend(section_chunks);
        }

        Ok(chunks)
    }
}

fn chunk_section(
    text: &str,
    base_offset: usize,
    section_path: &[String],
    doc: &RawDocument,
    tokenizer: &Tokenizer,
    max_tokens: u32,
    overlap_tokens: u32,
) -> Vec<Chunk> {
    // Reduce fragment budget when overlap is used, so that
    // overlap_prefix + fragment stays within max_tokens.
    let frag_max = if overlap_tokens > 0 {
        max_tokens.saturating_sub(overlap_tokens)
    } else {
        max_tokens
    };
    let fragments = split_recursive(text, tokenizer, frag_max);
    let mut chunks = Vec::new();
    let mut overlap_prefix = String::new();

    for frag in &fragments {
        let chunk_text = if overlap_prefix.is_empty() {
            frag.text.clone()
        } else {
            format!("{}{}", overlap_prefix, frag.text)
        };

        let char_offset = base_offset + frag.offset;
        let char_len = chunk_text.len();
        let token_count = tokenizer.count_tokens(&chunk_text);
        let cid = chunk_id(&doc.doc_id, char_offset, char_len);

        chunks.push(Chunk {
            chunk_id: cid,
            doc_id: doc.doc_id,
            doc_version: doc.doc_version,
            source_path: doc.source_path.to_string_lossy().to_string(),
            section_path: section_path.to_vec(),
            char_offset,
            char_len,
            token_count,
            text: chunk_text,
            metadata: HashMap::new(),
        });

        // Compute overlap for next chunk
        if overlap_tokens > 0 {
            overlap_prefix = get_tail_tokens(&frag.text, tokenizer, overlap_tokens);
        }
    }

    chunks
}

/// Get the last `n` tokens from text as a string.
fn get_tail_tokens(text: &str, tokenizer: &Tokenizer, n: u32) -> String {
    let total = tokenizer.count_tokens(text);
    if total <= n {
        return text.to_string();
    }
    // Skip (total - n) tokens from the start
    let skip = total - n;
    let prefix = tokenizer.truncate_to_tokens(text, skip);
    text[prefix.len()..].to_string()
}

struct Fragment {
    text: String,
    offset: usize,
}

fn split_recursive(text: &str, tokenizer: &Tokenizer, max_tokens: u32) -> Vec<Fragment> {
    if tokenizer.count_tokens(text) <= max_tokens {
        return vec![Fragment {
            text: text.to_string(),
            offset: 0,
        }];
    }

    // Try splitting on paragraphs
    let parts = split_on_separator(text, "\n\n");
    if parts.len() > 1 {
        return merge_and_recurse(parts, tokenizer, max_tokens);
    }

    // Try splitting on lines
    let parts = split_on_separator(text, "\n");
    if parts.len() > 1 {
        return merge_and_recurse(parts, tokenizer, max_tokens);
    }

    // Try splitting on sentences
    let parts = split_on_separator(text, ". ");
    if parts.len() > 1 {
        return merge_and_recurse(parts, tokenizer, max_tokens);
    }

    // Last resort: token boundary split
    split_on_tokens(text, tokenizer, max_tokens)
}

fn split_on_separator(text: &str, sep: &str) -> Vec<Fragment> {
    let mut frags = Vec::new();
    let mut off = 0;

    for (i, part) in text.split(sep).enumerate() {
        if i > 0 {
            off += sep.len();
        }
        let frag_text = if i == 0 {
            part.to_string()
        } else {
            format!("{sep}{part}")
        };
        if !frag_text.is_empty() {
            let frag_offset = if i == 0 { off } else { off - sep.len() };
            frags.push(Fragment {
                text: frag_text,
                offset: frag_offset,
            });
        }
        off += part.len();
    }

    frags
}

fn merge_and_recurse(
    parts: Vec<Fragment>,
    tokenizer: &Tokenizer,
    max_tokens: u32,
) -> Vec<Fragment> {
    let mut result = Vec::new();
    let mut current_text = String::new();
    let mut current_offset = 0;

    for frag in parts {
        let merged = if current_text.is_empty() {
            frag.text.clone()
        } else {
            format!("{}{}", current_text, frag.text)
        };

        if tokenizer.count_tokens(&merged) <= max_tokens {
            if current_text.is_empty() {
                current_offset = frag.offset;
            }
            current_text = merged;
        } else {
            // Flush current
            if !current_text.is_empty() {
                if tokenizer.count_tokens(&current_text) <= max_tokens {
                    result.push(Fragment {
                        text: current_text,
                        offset: current_offset,
                    });
                } else {
                    let sub = split_recursive(&current_text, tokenizer, max_tokens);
                    for mut s in sub {
                        s.offset += current_offset;
                        result.push(s);
                    }
                }
            }
            current_text = frag.text;
            current_offset = frag.offset;
        }
    }

    // Flush remaining
    if !current_text.is_empty() {
        if tokenizer.count_tokens(&current_text) <= max_tokens {
            result.push(Fragment {
                text: current_text,
                offset: current_offset,
            });
        } else {
            let sub = split_recursive(&current_text, tokenizer, max_tokens);
            for mut s in sub {
                s.offset += current_offset;
                result.push(s);
            }
        }
    }

    result
}

fn split_on_tokens(text: &str, tokenizer: &Tokenizer, max_tokens: u32) -> Vec<Fragment> {
    let mut result = Vec::new();
    let mut remaining = text;
    let mut offset = 0;

    while !remaining.is_empty() {
        let truncated = tokenizer.truncate_to_tokens(remaining, max_tokens);
        if truncated.is_empty() {
            // Safety: if truncate returns empty but remaining is not empty,
            // force at least one byte to avoid infinite loop
            let boundary = remaining
                .char_indices()
                .nth(1)
                .map(|(i, _)| i)
                .unwrap_or(remaining.len());
            result.push(Fragment {
                text: remaining[..boundary].to_string(),
                offset,
            });
            offset += boundary;
            remaining = &remaining[boundary..];
        } else {
            let len = truncated.len();
            result.push(Fragment {
                text: truncated,
                offset,
            });
            offset += len;
            remaining = &remaining[len..];
        }
    }

    result
}

/// Token-based chunker. Fixed chunk size with overlap, no structure awareness.
pub struct TokenChunker {
    pub chunk_size_tokens: u32,
    pub overlap_tokens: u32,
}

impl TokenChunker {
    pub fn new(chunk_size_tokens: u32, overlap_tokens: u32) -> Self {
        Self {
            chunk_size_tokens,
            overlap_tokens,
        }
    }
}

impl Chunker for TokenChunker {
    fn chunk(&self, doc: &RawDocument, tokenizer: &Tokenizer) -> Result<Vec<Chunk>> {
        let mut chunks = Vec::new();
        let full_text = &doc.content;

        if full_text.is_empty() {
            return Ok(chunks);
        }

        let mut remaining = full_text.as_str();
        let mut offset = 0;
        let mut overlap_prefix = String::new();

        while !remaining.is_empty() {
            let chunk_text = if overlap_prefix.is_empty() {
                let t = tokenizer.truncate_to_tokens(remaining, self.chunk_size_tokens);
                if t.is_empty() && !remaining.is_empty() {
                    // Force at least one character
                    let boundary = remaining
                        .char_indices()
                        .nth(1)
                        .map(|(i, _)| i)
                        .unwrap_or(remaining.len());
                    remaining[..boundary].to_string()
                } else {
                    t
                }
            } else {
                // Budget for overlap
                let budget = self.chunk_size_tokens.saturating_sub(
                    tokenizer.count_tokens(&overlap_prefix),
                );
                let new_part = tokenizer.truncate_to_tokens(remaining, budget);
                if new_part.is_empty() && !remaining.is_empty() {
                    let boundary = remaining
                        .char_indices()
                        .nth(1)
                        .map(|(i, _)| i)
                        .unwrap_or(remaining.len());
                    format!("{}{}", overlap_prefix, &remaining[..boundary])
                } else {
                    format!("{}{}", overlap_prefix, new_part)
                }
            };

            let char_len = chunk_text.len();
            let token_count = tokenizer.count_tokens(&chunk_text);
            let cid = chunk_id(&doc.doc_id, offset, char_len);

            chunks.push(Chunk {
                chunk_id: cid,
                doc_id: doc.doc_id,
                doc_version: doc.doc_version,
                source_path: doc.source_path.to_string_lossy().to_string(),
                section_path: vec![],
                char_offset: offset,
                char_len,
                token_count,
                text: chunk_text.clone(),
                metadata: HashMap::new(),
            });

            // Advance past what we consumed (excluding overlap prefix)
            let consumed = if overlap_prefix.is_empty() {
                chunk_text.len()
            } else {
                chunk_text.len() - overlap_prefix.len()
            };
            if consumed == 0 {
                break;
            }
            offset += consumed;
            remaining = &remaining[consumed.min(remaining.len())..];

            // Compute overlap for next
            if self.overlap_tokens > 0 && !remaining.is_empty() {
                overlap_prefix = get_tail_tokens(&chunk_text, tokenizer, self.overlap_tokens);
            } else {
                overlap_prefix = String::new();
            }
        }

        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Section;
    use std::path::PathBuf;

    fn make_doc(content: &str, sections: Vec<Section>) -> RawDocument {
        RawDocument {
            doc_id: Uuid::new_v5(&CHUNK_NS, b"test-doc"),
            source_path: PathBuf::from("test.md"),
            doc_version: 1,
            content: content.to_string(),
            sections,
            metadata: HashMap::new(),
        }
    }

    fn tokenizer() -> Tokenizer {
        Tokenizer::default_tokenizer().unwrap()
    }

    #[test]
    fn test_recursive_chunker_paragraph_split() {
        let text = (0..20)
            .map(|i| format!("Paragraph {i} with some extra text to use tokens."))
            .collect::<Vec<_>>()
            .join("\n\n");

        let doc = make_doc(
            &text,
            vec![Section {
                path: vec!["Root".to_string()],
                text: text.clone(),
                char_offset: 0,
                char_len: text.len(),
            }],
        );

        let tok = tokenizer();
        let chunker = RecursiveChunker::new(50, 0);
        let chunks = chunker.chunk(&doc, &tok).unwrap();

        // Each chunk should not exceed max_tokens
        for c in &chunks {
            assert!(
                c.token_count <= 50,
                "chunk has {} tokens, max is 50",
                c.token_count
            );
        }
        assert!(chunks.len() > 1);
        // All chunks should have the section path
        for c in &chunks {
            assert_eq!(c.section_path, vec!["Root"]);
        }
    }

    #[test]
    fn test_recursive_chunker_section_boundaries() {
        let s1 = "Section one content with enough text.";
        let s2 = "Section two with different content here.";
        let content = format!("{s1}\n\n{s2}");

        let doc = make_doc(
            &content,
            vec![
                Section {
                    path: vec!["S1".to_string()],
                    text: s1.to_string(),
                    char_offset: 0,
                    char_len: s1.len(),
                },
                Section {
                    path: vec!["S2".to_string()],
                    text: s2.to_string(),
                    char_offset: s1.len() + 2,
                    char_len: s2.len(),
                },
            ],
        );

        let tok = tokenizer();
        let chunker = RecursiveChunker::new(512, 0);
        let chunks = chunker.chunk(&doc, &tok).unwrap();

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].section_path, vec!["S1"]);
        assert_eq!(chunks[1].section_path, vec!["S2"]);
    }

    #[test]
    fn test_recursive_chunker_overlap() {
        // Create enough text to split into multiple chunks
        let text = (0..30)
            .map(|i| format!("Sentence number {i} has some words."))
            .collect::<Vec<_>>()
            .join(" ");

        let doc = make_doc(
            &text,
            vec![Section {
                path: vec![],
                text: text.clone(),
                char_offset: 0,
                char_len: text.len(),
            }],
        );

        let tok = tokenizer();
        let chunker = RecursiveChunker::new(40, 10);
        let chunks = chunker.chunk(&doc, &tok).unwrap();

        assert!(chunks.len() >= 2, "expected at least 2 chunks");
        // Verify overlap: tail of chunk[0] should appear at start of chunk[1]
        if chunks.len() >= 2 {
            let tail_0 = get_tail_tokens(&chunks[0].text, &tok, 10);
            assert!(
                chunks[1].text.starts_with(&tail_0),
                "chunk[1] should start with overlap from chunk[0]\ntail_0: {tail_0:?}\nchunk[1] start: {:?}",
                &chunks[1].text[..tail_0.len().min(chunks[1].text.len())]
            );
        }
    }

    #[test]
    fn test_recursive_chunker_long_line() {
        // Single long line that exceeds max tokens — should split on sentences then tokens
        let text = (0..100)
            .map(|i| format!("Word{i}"))
            .collect::<Vec<_>>()
            .join(" ");

        let doc = make_doc(
            &text,
            vec![Section {
                path: vec![],
                text: text.clone(),
                char_offset: 0,
                char_len: text.len(),
            }],
        );

        let tok = tokenizer();
        let chunker = RecursiveChunker::new(20, 0);
        let chunks = chunker.chunk(&doc, &tok).unwrap();

        for c in &chunks {
            assert!(
                c.token_count <= 20,
                "chunk exceeded max: {} tokens",
                c.token_count
            );
        }
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_token_chunker_fixed_size() {
        let text = "word ".repeat(200);
        let doc = make_doc(
            &text,
            vec![Section {
                path: vec![],
                text: text.clone(),
                char_offset: 0,
                char_len: text.len(),
            }],
        );

        let tok = tokenizer();
        let chunker = TokenChunker::new(50, 0);
        let chunks = chunker.chunk(&doc, &tok).unwrap();

        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(c.token_count <= 50, "token_count {} > 50", c.token_count);
        }
    }

    #[test]
    fn test_chunk_id_determinism() {
        let doc_id = Uuid::new_v4();
        let id1 = chunk_id(&doc_id, 100, 500);
        let id2 = chunk_id(&doc_id, 100, 500);
        assert_eq!(id1, id2);

        // Different offset → different id
        let id3 = chunk_id(&doc_id, 101, 500);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_recursive_chunker_empty_section() {
        let doc = make_doc(
            "",
            vec![Section {
                path: vec![],
                text: String::new(),
                char_offset: 0,
                char_len: 0,
            }],
        );

        let tok = tokenizer();
        let chunker = RecursiveChunker::new(512, 0);
        let chunks = chunker.chunk(&doc, &tok).unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_recursive_chunker_overlap_within_budget() {
        // Verify that overlap + fragment stays within max_chunk_tokens
        let text = (0..50)
            .map(|i| format!("Sentence number {i} has some words here."))
            .collect::<Vec<_>>()
            .join(" ");

        let doc = make_doc(
            &text,
            vec![Section {
                path: vec![],
                text: text.clone(),
                char_offset: 0,
                char_len: text.len(),
            }],
        );

        let tok = tokenizer();
        let max = 60;
        let overlap = 15;
        let chunker = RecursiveChunker::new(max, overlap);
        let chunks = chunker.chunk(&doc, &tok).unwrap();

        assert!(chunks.len() >= 2, "expected multiple chunks");
        for (i, c) in chunks.iter().enumerate() {
            assert!(
                c.token_count <= max,
                "chunk {i} has {} tokens, max is {max}",
                c.token_count
            );
        }
    }

    #[test]
    fn test_recursive_chunker_single_chunk() {
        let text = "Short text.";
        let doc = make_doc(
            text,
            vec![Section {
                path: vec!["Only".to_string()],
                text: text.to_string(),
                char_offset: 0,
                char_len: text.len(),
            }],
        );

        let tok = tokenizer();
        let chunker = RecursiveChunker::new(512, 0);
        let chunks = chunker.chunk(&doc, &tok).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Short text.");
        assert_eq!(chunks[0].section_path, vec!["Only"]);
    }

    #[test]
    fn test_token_chunker_with_overlap() {
        let text = "word ".repeat(200);
        let doc = make_doc(
            &text,
            vec![Section {
                path: vec![],
                text: text.clone(),
                char_offset: 0,
                char_len: text.len(),
            }],
        );

        let tok = tokenizer();
        let max = 50;
        let overlap = 10;
        let chunker = TokenChunker::new(max, overlap);
        let chunks = chunker.chunk(&doc, &tok).unwrap();

        assert!(chunks.len() > 1);
        for (i, c) in chunks.iter().enumerate() {
            assert!(
                c.token_count <= max,
                "chunk {i} has {} tokens, max is {max}",
                c.token_count
            );
        }

        // Verify overlap: tail of chunk[0] should appear at start of chunk[1]
        if chunks.len() >= 2 {
            let tail_0 = get_tail_tokens(&chunks[0].text, &tok, overlap);
            assert!(
                chunks[1].text.starts_with(&tail_0),
                "chunk[1] should start with overlap from chunk[0]"
            );
        }
    }

    #[test]
    fn test_chunk_id_different_doc() {
        let id_a = Uuid::new_v5(&CHUNK_NS, b"doc-a");
        let id_b = Uuid::new_v5(&CHUNK_NS, b"doc-b");
        // Same offset/len, different doc → different chunk_id
        let cid_a = chunk_id(&id_a, 0, 100);
        let cid_b = chunk_id(&id_b, 0, 100);
        assert_ne!(cid_a, cid_b);
    }

    #[test]
    fn test_end_to_end_load_and_chunk() {
        use crate::loader::load_file;
        use std::fs;

        let dir = std::env::temp_dir().join("cwc_chunker_e2e");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let path = dir.join("test.md");
        fs::write(
            &path,
            "# Rust Ownership\n\nRust uses ownership rules.\n\n## Borrowing\n\nYou can borrow values.\n",
        )
        .unwrap();

        let doc = load_file(&path).unwrap();
        let tok = tokenizer();
        let chunker = RecursiveChunker::new(512, 0);
        let chunks = chunker.chunk(&doc, &tok).unwrap();

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].section_path, vec!["Rust Ownership"]);
        assert_eq!(
            chunks[1].section_path,
            vec!["Rust Ownership", "Borrowing"]
        );
        assert!(chunks[0].text.contains("ownership rules"));
        assert!(chunks[1].text.contains("borrow values"));

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_end_to_end_empty_file_zero_chunks() {
        use crate::loader::load_file;
        use std::fs;

        let dir = std::env::temp_dir().join("cwc_chunker_empty_e2e");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let path = dir.join("empty.txt");
        fs::write(&path, "").unwrap();

        let doc = load_file(&path).unwrap();
        let tok = tokenizer();
        let chunker = RecursiveChunker::new(512, 0);
        let chunks = chunker.chunk(&doc, &tok).unwrap();
        assert!(chunks.is_empty(), "empty file should produce zero chunks");

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_token_chunker_empty() {
        let doc = make_doc(
            "",
            vec![Section {
                path: vec![],
                text: String::new(),
                char_offset: 0,
                char_len: 0,
            }],
        );

        let tok = tokenizer();
        let chunker = TokenChunker::new(50, 0);
        let chunks = chunker.chunk(&doc, &tok).unwrap();
        assert!(chunks.is_empty());
    }
}
