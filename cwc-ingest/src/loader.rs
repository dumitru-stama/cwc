use std::collections::HashMap;
use std::path::{Path, PathBuf};
use uuid::Uuid;

use crate::error::{IngestError, Result};
use crate::markdown;
use crate::types::{RawDocument, Section};

/// Detect whether a file is binary by checking the first 8KB for null bytes.
fn is_binary(path: &Path) -> Result<bool> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut buf = [0u8; 8192];
    let n = file.read(&mut buf)?;
    Ok(buf[..n].contains(&0))
}

/// Determine file format from extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    PlainText,
    Markdown,
    Json,
    Jsonl,
    Code,
}

fn detect_format(path: &Path) -> FileFormat {
    match path.extension().and_then(|e| e.to_str()) {
        Some("md" | "markdown") => FileFormat::Markdown,
        Some("json") => FileFormat::Json,
        Some("jsonl" | "ndjson") => FileFormat::Jsonl,
        Some("rs" | "py" | "c" | "cpp" | "h" | "go" | "js" | "ts" | "java" | "rb" | "sh"
        | "zig" | "asm" | "toml" | "yaml" | "yml") => FileFormat::Code,
        _ => FileFormat::PlainText,
    }
}

fn language_from_ext(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_string())
}

/// Load a single file into a RawDocument.
pub fn load_file(path: impl AsRef<Path>) -> Result<RawDocument> {
    let path = path.as_ref();
    if is_binary(path)? {
        return Err(IngestError::Loader(format!(
            "binary file skipped: {}",
            path.display()
        )));
    }

    let content = std::fs::read_to_string(path)?;
    let format = detect_format(path);
    let doc_id = Uuid::new_v5(&Uuid::NAMESPACE_URL, path.to_string_lossy().as_bytes());
    let mut metadata = HashMap::new();
    metadata.insert("format".to_string(), format!("{format:?}"));

    if format == FileFormat::Code {
        if let Some(lang) = language_from_ext(path) {
            metadata.insert("language".to_string(), lang);
        }
    }

    let sections = match format {
        FileFormat::Markdown => {
            let (front_matter, sections) = markdown::parse_markdown(&content);
            if let Some(fm) = front_matter {
                metadata.insert("front_matter".to_string(), fm);
            }
            sections
        }
        FileFormat::Json => parse_json_sections(&content)?,
        FileFormat::Jsonl => parse_jsonl_sections(&content),
        _ => {
            // Plain text / code: single section
            vec![Section {
                path: vec![],
                text: content.clone(),
                char_offset: 0,
                char_len: content.len(),
            }]
        }
    };

    Ok(RawDocument {
        doc_id,
        source_path: path.to_path_buf(),
        doc_version: 1,
        content,
        sections,
        metadata,
    })
}

fn parse_json_sections(content: &str) -> Result<Vec<Section>> {
    let value: serde_json::Value = serde_json::from_str(content)?;
    match value {
        serde_json::Value::Array(arr) => {
            let mut sections = Vec::new();
            // Each element becomes a section. We serialize each back for text.
            // Track char offsets approximately (re-serialized, not original offsets).
            let mut offset = 0;
            for (i, elem) in arr.iter().enumerate() {
                let text = serde_json::to_string_pretty(elem)?;
                let len = text.len();
                sections.push(Section {
                    path: vec![format!("[{i}]")],
                    text,
                    char_offset: offset,
                    char_len: len,
                });
                offset += len;
            }
            Ok(sections)
        }
        _ => {
            // Single object: one section
            Ok(vec![Section {
                path: vec![],
                text: content.to_string(),
                char_offset: 0,
                char_len: content.len(),
            }])
        }
    }
}

fn parse_jsonl_sections(content: &str) -> Vec<Section> {
    let mut sections = Vec::new();
    let mut offset = 0;
    for (i, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            offset += line.len() + 1; // +1 for newline
            continue;
        }
        let len = line.len();
        sections.push(Section {
            path: vec![format!("line_{i}")],
            text: line.to_string(),
            char_offset: offset,
            char_len: len,
        });
        offset += len + 1;
    }
    sections
}

/// Load all matching files from a directory.
pub fn load_directory(
    dir: impl AsRef<Path>,
    recursive: bool,
    glob_filter: Option<&str>,
) -> Result<Vec<RawDocument>> {
    let dir = dir.as_ref();
    let paths = collect_paths(dir, recursive, glob_filter)?;

    let mut docs = Vec::new();
    for path in paths {
        match load_file(&path) {
            Ok(doc) => docs.push(doc),
            Err(IngestError::Loader(msg)) => {
                tracing::debug!("skipping {}: {}", path.display(), msg);
            }
            Err(e) => return Err(e),
        }
    }
    Ok(docs)
}

fn collect_paths(
    dir: &Path,
    recursive: bool,
    glob_filter: Option<&str>,
) -> Result<Vec<PathBuf>> {
    if let Some(pattern) = glob_filter {
        let full_pattern = if recursive {
            format!("{}/**/{}", dir.display(), pattern)
        } else {
            format!("{}/{}", dir.display(), pattern)
        };
        let paths: std::result::Result<Vec<_>, _> = glob::glob(&full_pattern)?.collect();
        let mut paths = paths?;
        paths.sort();
        Ok(paths)
    } else {
        let mut paths = Vec::new();
        walk_dir(dir, recursive, &mut paths)?;
        paths.sort();
        Ok(paths)
    }
}

fn walk_dir(dir: &Path, recursive: bool, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let ft = entry.file_type()?;
        if ft.is_file() {
            out.push(entry.path());
        } else if ft.is_dir() && recursive {
            walk_dir(&entry.path(), true, out)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cwc_loader_{name}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_loader_plain_text() {
        let dir = test_dir("plain_text");
        let path = dir.join("hello.txt");
        fs::write(&path, "Hello, world!\nSecond line.").unwrap();

        let doc = load_file(&path).unwrap();
        assert_eq!(doc.sections.len(), 1);
        assert_eq!(doc.sections[0].text, "Hello, world!\nSecond line.");
        assert!(doc.sections[0].path.is_empty());
        assert_eq!(doc.metadata.get("format").unwrap(), "PlainText");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_loader_json_array() {
        let dir = test_dir("json_array");
        let path = dir.join("data.json");
        fs::write(&path, r#"[{"a":1},{"b":2},{"c":3}]"#).unwrap();

        let doc = load_file(&path).unwrap();
        assert_eq!(doc.sections.len(), 3);
        assert_eq!(doc.sections[0].path, vec!["[0]"]);
        assert_eq!(doc.sections[1].path, vec!["[1]"]);
        assert_eq!(doc.sections[2].path, vec!["[2]"]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_loader_json_single_object() {
        let dir = test_dir("json_single");
        let path = dir.join("single.json");
        fs::write(&path, r#"{"key":"value"}"#).unwrap();

        let doc = load_file(&path).unwrap();
        assert_eq!(doc.sections.len(), 1);
        assert!(doc.sections[0].path.is_empty());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_loader_binary_file_skipped() {
        let dir = test_dir("binary_skip");
        let path = dir.join("binary.bin");
        fs::write(&path, b"\x00\x01\x02\x03binary data").unwrap();

        let result = load_file(&path);
        assert!(result.is_err());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_loader_empty_file() {
        let dir = test_dir("empty_file");
        let path = dir.join("empty.txt");
        fs::write(&path, "").unwrap();

        let doc = load_file(&path).unwrap();
        assert_eq!(doc.sections.len(), 1);
        assert_eq!(doc.sections[0].text, "");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_load_directory_with_glob() {
        let dir = test_dir("dir_glob");
        fs::write(dir.join("a.txt"), "file a").unwrap();
        fs::write(dir.join("b.md"), "# B").unwrap();
        fs::write(dir.join("c.txt"), "file c").unwrap();

        let docs = load_directory(&dir, false, Some("*.txt")).unwrap();
        assert_eq!(docs.len(), 2);
        // Should be sorted
        let names: Vec<_> = docs
            .iter()
            .map(|d| d.source_path.file_name().unwrap().to_str().unwrap())
            .collect();
        assert_eq!(names, vec!["a.txt", "c.txt"]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_load_directory_recursive() {
        let dir = test_dir("dir_recursive");
        let sub = dir.join("sub");
        fs::create_dir_all(&sub).unwrap();
        fs::write(dir.join("top.txt"), "top").unwrap();
        fs::write(sub.join("nested.txt"), "nested").unwrap();

        let docs = load_directory(&dir, true, None).unwrap();
        assert_eq!(docs.len(), 2);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_load_directory_skips_binary() {
        let dir = test_dir("dir_skips_binary");
        fs::write(dir.join("good.txt"), "text").unwrap();
        fs::write(dir.join("bad.bin"), b"\x00binary").unwrap();

        let docs = load_directory(&dir, false, None).unwrap();
        assert_eq!(docs.len(), 1);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_loader_jsonl() {
        let dir = test_dir("jsonl");
        let path = dir.join("data.jsonl");
        fs::write(&path, "{\"a\":1}\n{\"b\":2}\n{\"c\":3}\n").unwrap();

        let doc = load_file(&path).unwrap();
        assert_eq!(doc.sections.len(), 3);
        assert_eq!(doc.sections[0].path, vec!["line_0"]);
        assert_eq!(doc.sections[1].path, vec!["line_1"]);
        assert_eq!(doc.sections[2].path, vec!["line_2"]);
        assert!(doc.sections[0].text.contains("\"a\":1"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_load_directory_non_recursive_skips_subdirs() {
        let dir = test_dir("non_recursive");
        let sub = dir.join("sub");
        fs::create_dir_all(&sub).unwrap();
        fs::write(dir.join("top.txt"), "top").unwrap();
        fs::write(sub.join("nested.txt"), "nested").unwrap();

        let docs = load_directory(&dir, false, None).unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(
            docs[0].source_path.file_name().unwrap().to_str().unwrap(),
            "top.txt"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_loader_markdown_no_language_metadata() {
        let dir = test_dir("md_no_lang");
        let path = dir.join("doc.md");
        fs::write(&path, "# Title\n\nBody.\n").unwrap();

        let doc = load_file(&path).unwrap();
        assert_eq!(doc.metadata.get("format").unwrap(), "Markdown");
        // language should NOT be set for markdown files
        assert!(!doc.metadata.contains_key("language"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_loader_code_file_has_language() {
        let dir = test_dir("code_lang");
        let path = dir.join("main.rs");
        fs::write(&path, "fn main() {}").unwrap();

        let doc = load_file(&path).unwrap();
        assert_eq!(doc.metadata.get("language").unwrap(), "rs");
        assert_eq!(doc.metadata.get("format").unwrap(), "Code");
        fs::remove_dir_all(&dir).unwrap();
    }
}
