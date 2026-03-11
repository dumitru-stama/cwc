use std::io::BufRead;
use std::path::Path;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{EvalError, Result};

/// A collection of evaluation queries with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalDataset {
    pub name: String,
    pub queries: Vec<EvalQuery>,
}

/// A single evaluation query with ground-truth annotations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalQuery {
    pub query: String,
    pub expected_answer: Option<String>,
    pub relevant_chunk_ids: Vec<Uuid>,
    pub expected_citations: Vec<String>,
    pub category: QueryCategory,
    pub difficulty: Difficulty,
}

/// Category of a query (affects which metrics apply).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryCategory {
    Factual,
    MultiHop,
    Comparison,
    Procedural,
    Unanswerable,
    Adversarial,
}

/// Difficulty level for a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
}

impl EvalDataset {
    /// Load a dataset from a JSONL file. Each line is one EvalQuery.
    /// First line may optionally be a metadata line with `{"name": "..."}`.
    pub fn load_jsonl(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| EvalError::Io(format!("failed to open {}: {e}", path.display())))?;
        let reader = std::io::BufReader::new(file);

        let mut name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unnamed")
            .to_string();
        let mut queries = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line
                .map_err(|e| EvalError::Io(format!("read error at line {}: {e}", line_num + 1)))?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Try parsing as metadata first (only if no queries yet)
            if queries.is_empty() {
                if let Ok(meta) = serde_json::from_str::<DatasetMeta>(trimmed) {
                    if let Some(n) = meta.name {
                        if meta.query.is_none() {
                            name = n;
                            continue;
                        }
                    }
                }
            }

            let query: EvalQuery = serde_json::from_str(trimmed).map_err(|e| {
                EvalError::Parse(format!("line {}: {e}", line_num + 1))
            })?;
            queries.push(query);
        }

        if queries.is_empty() {
            return Err(EvalError::Parse("dataset contains no queries".into()));
        }

        Ok(Self { name, queries })
    }

    /// Load from a JSON array (single file, not JSONL).
    pub fn load_json(path: &Path, name: &str) -> Result<Self> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| EvalError::Io(format!("failed to read {}: {e}", path.display())))?;
        let queries: Vec<EvalQuery> = serde_json::from_str(&data)
            .map_err(|e| EvalError::Parse(format!("JSON parse error: {e}")))?;
        if queries.is_empty() {
            return Err(EvalError::Parse("dataset contains no queries".into()));
        }
        Ok(Self {
            name: name.to_string(),
            queries,
        })
    }

    /// Create a dataset from in-memory queries.
    pub fn new(name: &str, queries: Vec<EvalQuery>) -> Self {
        Self {
            name: name.to_string(),
            queries,
        }
    }

    /// Save dataset to JSONL format.
    pub fn save_jsonl(&self, path: &Path) -> Result<()> {
        let mut lines = Vec::with_capacity(self.queries.len() + 1);
        lines.push(
            serde_json::to_string(&DatasetMeta {
                name: Some(self.name.clone()),
                query: None,
            })
            .map_err(|e| EvalError::Parse(format!("serialize error: {e}")))?,
        );
        for query in &self.queries {
            lines.push(
                serde_json::to_string(query)
                    .map_err(|e| EvalError::Parse(format!("serialize error: {e}")))?,
            );
        }
        let mut content = lines.join("\n");
        content.push('\n');
        std::fs::write(path, content)
            .map_err(|e| EvalError::Io(format!("write error: {e}")))?;
        Ok(())
    }

    /// Filter queries by category.
    pub fn filter_category(&self, category: QueryCategory) -> Vec<&EvalQuery> {
        self.queries.iter().filter(|q| q.category == category).collect()
    }

    /// Number of queries.
    pub fn len(&self) -> usize {
        self.queries.len()
    }

    /// Whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.queries.is_empty()
    }
}

/// Helper for parsing the optional metadata line.
#[derive(Deserialize, Serialize)]
struct DatasetMeta {
    name: Option<String>,
    #[serde(skip_serializing)]
    query: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("cwc_eval_{}_{}", name, std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    fn sample_query(category: QueryCategory) -> EvalQuery {
        EvalQuery {
            query: "What is ownership in Rust?".into(),
            expected_answer: Some("Ownership is Rust's memory management system.".into()),
            relevant_chunk_ids: vec![Uuid::nil()],
            expected_citations: vec!["[S1]".into()],
            category,
            difficulty: Difficulty::Easy,
        }
    }

    #[test]
    fn test_dataset_load_jsonl() {
        let dir = temp_dir("test_jsonl");
        let path = dir.join("test.jsonl");

        let q1 = sample_query(QueryCategory::Factual);
        let q2 = EvalQuery {
            query: "Compare Rust and C++".into(),
            expected_answer: None,
            relevant_chunk_ids: vec![],
            expected_citations: vec![],
            category: QueryCategory::Comparison,
            difficulty: Difficulty::Medium,
        };

        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, r#"{{"name": "test_dataset"}}"#).unwrap();
        writeln!(f, "{}", serde_json::to_string(&q1).unwrap()).unwrap();
        writeln!(f, "{}", serde_json::to_string(&q2).unwrap()).unwrap();

        let ds = EvalDataset::load_jsonl(&path).unwrap();
        assert_eq!(ds.name, "test_dataset");
        assert_eq!(ds.queries.len(), 2);
        assert_eq!(ds.queries[0].category, QueryCategory::Factual);
        assert_eq!(ds.queries[1].category, QueryCategory::Comparison);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_dataset_load_jsonl_no_metadata() {
        let dir = temp_dir("test_nometa");
        let path = dir.join("queries.jsonl");

        let q = sample_query(QueryCategory::Factual);
        std::fs::write(&path, serde_json::to_string(&q).unwrap()).unwrap();

        let ds = EvalDataset::load_jsonl(&path).unwrap();
        assert_eq!(ds.name, "queries"); // from filename
        assert_eq!(ds.queries.len(), 1);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_dataset_load_empty_file() {
        let dir = temp_dir("test_empty");
        let path = dir.join("empty.jsonl");
        std::fs::write(&path, "").unwrap();

        let result = EvalDataset::load_jsonl(&path);
        assert!(result.is_err());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_dataset_load_jsonl_bad_line() {
        let dir = temp_dir("test_bad");
        let path = dir.join("bad.jsonl");
        std::fs::write(&path, "not json at all\n").unwrap();

        let result = EvalDataset::load_jsonl(&path);
        assert!(result.is_err());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_dataset_save_and_reload() {
        let dir = temp_dir("test_roundtrip");
        let path = dir.join("roundtrip.jsonl");

        let ds = EvalDataset::new(
            "roundtrip_test",
            vec![
                sample_query(QueryCategory::Factual),
                sample_query(QueryCategory::Unanswerable),
            ],
        );
        ds.save_jsonl(&path).unwrap();

        let loaded = EvalDataset::load_jsonl(&path).unwrap();
        assert_eq!(loaded.name, "roundtrip_test");
        assert_eq!(loaded.queries.len(), 2);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_dataset_filter_category() {
        let ds = EvalDataset::new(
            "filter_test",
            vec![
                sample_query(QueryCategory::Factual),
                sample_query(QueryCategory::Comparison),
                sample_query(QueryCategory::Factual),
            ],
        );

        let factual = ds.filter_category(QueryCategory::Factual);
        assert_eq!(factual.len(), 2);

        let comparison = ds.filter_category(QueryCategory::Comparison);
        assert_eq!(comparison.len(), 1);

        let procedural = ds.filter_category(QueryCategory::Procedural);
        assert!(procedural.is_empty());
    }

    #[test]
    fn test_dataset_load_jsonl_skips_blank_lines() {
        let dir = temp_dir("test_blank");
        let path = dir.join("blanks.jsonl");

        let q = sample_query(QueryCategory::Factual);
        let line = serde_json::to_string(&q).unwrap();
        std::fs::write(&path, format!("\n{line}\n\n{line}\n")).unwrap();

        let ds = EvalDataset::load_jsonl(&path).unwrap();
        assert_eq!(ds.queries.len(), 2);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_query_category_serde() {
        let q = sample_query(QueryCategory::MultiHop);
        let json = serde_json::to_string(&q).unwrap();
        assert!(json.contains("multi_hop"));
        let back: EvalQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(back.category, QueryCategory::MultiHop);
    }

    #[test]
    fn test_dataset_load_json() {
        let dir = temp_dir("test_json");
        let path = dir.join("queries.json");

        let q = sample_query(QueryCategory::Factual);
        let queries = vec![q];
        std::fs::write(&path, serde_json::to_string(&queries).unwrap()).unwrap();

        let ds = EvalDataset::load_json(&path, "json_test").unwrap();
        assert_eq!(ds.name, "json_test");
        assert_eq!(ds.queries.len(), 1);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_dataset_load_json_empty_array() {
        let dir = temp_dir("test_json_empty");
        let path = dir.join("empty.json");
        std::fs::write(&path, "[]").unwrap();

        let result = EvalDataset::load_json(&path, "empty");
        assert!(result.is_err());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_dataset_metadata_line_with_query_field() {
        // If the first line looks like metadata but has a "query" field,
        // it should be treated as a query, not metadata
        let dir = temp_dir("test_meta_query");
        let path = dir.join("tricky.jsonl");

        let q = sample_query(QueryCategory::Factual);
        let line = serde_json::to_string(&q).unwrap();
        // First line is a valid query (has "query" field) → not metadata
        std::fs::write(&path, format!("{line}\n")).unwrap();

        let ds = EvalDataset::load_jsonl(&path).unwrap();
        assert_eq!(ds.name, "tricky"); // from filename, not metadata
        assert_eq!(ds.queries.len(), 1);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_dataset_load_nonexistent_file() {
        let result = EvalDataset::load_jsonl(std::path::Path::new("/nonexistent/path/data.jsonl"));
        assert!(result.is_err());
    }

    #[test]
    fn test_dataset_len_and_is_empty() {
        let ds = EvalDataset::new("empty", vec![]);
        assert!(ds.is_empty());
        assert_eq!(ds.len(), 0);

        let ds2 = EvalDataset::new("one", vec![sample_query(QueryCategory::Factual)]);
        assert!(!ds2.is_empty());
        assert_eq!(ds2.len(), 1);
    }
}
