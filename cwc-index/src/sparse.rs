use std::path::Path;

use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy};
use uuid::Uuid;

use cwc_core::types::{Chunk, RetrievalHit};

use crate::error::{IndexError, Result};

/// BM25 full-text search index backed by tantivy.
pub struct SparseIndex {
    index: Index,
    reader: IndexReader,
    // Field handles cached for performance
    f_chunk_id: Field,
    f_doc_id: Field,
    f_text: Field,
    f_section_path: Field,
    f_source_path: Field,
    f_chunk_json: Field,
}

impl SparseIndex {
    /// Create or open an index at the given directory.
    pub fn open_or_create(dir: &Path) -> Result<Self> {
        let schema = Self::build_schema();
        let fields = Self::resolve_fields(&schema)?;

        std::fs::create_dir_all(dir)?;
        let dir_obj = tantivy::directory::MmapDirectory::open(dir)?;

        let index = if Index::exists(&dir_obj)? {
            Index::open(dir_obj)?
        } else {
            Index::create_in_dir(dir, schema.clone())?
        };

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;

        Ok(Self {
            index,
            reader,
            f_chunk_id: fields.0,
            f_doc_id: fields.1,
            f_text: fields.2,
            f_section_path: fields.3,
            f_source_path: fields.4,
            f_chunk_json: fields.5,
        })
    }

    /// Create a RAM-based index (for testing).
    pub fn open_in_ram() -> Result<Self> {
        let schema = Self::build_schema();
        let fields = Self::resolve_fields(&schema)?;

        let index = Index::create_in_ram(schema);
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;

        Ok(Self {
            index,
            reader,
            f_chunk_id: fields.0,
            f_doc_id: fields.1,
            f_text: fields.2,
            f_section_path: fields.3,
            f_source_path: fields.4,
            f_chunk_json: fields.5,
        })
    }

    fn resolve_fields(schema: &Schema) -> Result<(Field, Field, Field, Field, Field, Field)> {
        let get = |name: &str| -> Result<Field> {
            schema.get_field(name).map_err(|_| IndexError::Index(format!("missing schema field: {name}")))
        };
        Ok((get("chunk_id")?, get("doc_id")?, get("text")?, get("section_path")?, get("source_path")?, get("chunk_json")?))
    }

    fn build_schema() -> Schema {
        let mut builder = Schema::builder();
        builder.add_text_field("chunk_id", STRING | STORED);
        builder.add_text_field("doc_id", STRING | STORED);
        builder.add_text_field("text", TEXT | STORED);
        builder.add_text_field("section_path", STRING | STORED);
        builder.add_text_field("source_path", STRING | STORED);
        // Store full chunk as JSON for reconstruction
        builder.add_text_field("chunk_json", STORED);
        builder.build()
    }

    fn writer(&self) -> Result<IndexWriter> {
        // 50MB heap for indexing
        let writer = self.index.writer(50_000_000)?;
        Ok(writer)
    }

    /// Index a batch of chunks. Idempotent — deletes existing docs with same chunk_id first.
    pub fn index_chunks(&self, chunks: &[Chunk]) -> Result<u32> {
        let mut writer = self.writer()?;
        let mut count = 0u32;

        for chunk in chunks {
            let chunk_id_str = chunk.chunk_id.to_string();
            // Delete existing doc with same chunk_id
            let term = tantivy::Term::from_field_text(self.f_chunk_id, &chunk_id_str);
            writer.delete_term(term);

            let section_path_str = chunk.section_path.join(" > ");
            let chunk_json =
                serde_json::to_string(chunk).map_err(|e| IndexError::Index(e.to_string()))?;

            let mut doc = TantivyDocument::new();
            doc.add_text(self.f_chunk_id, &chunk_id_str);
            doc.add_text(self.f_doc_id, chunk.doc_id.to_string());
            doc.add_text(self.f_text, &chunk.text);
            doc.add_text(self.f_section_path, &section_path_str);
            doc.add_text(self.f_source_path, &chunk.source_path);
            doc.add_text(self.f_chunk_json, &chunk_json);

            writer.add_document(doc)?;
            count += 1;
        }

        writer.commit()?;
        self.reader.reload()?;
        Ok(count)
    }

    /// Delete all chunks belonging to a document.
    pub fn delete_document(&self, doc_id: Uuid) -> Result<u32> {
        let mut writer = self.writer()?;
        let term = tantivy::Term::from_field_text(self.f_doc_id, &doc_id.to_string());

        // Count before delete
        let before = self.count()?;
        writer.delete_term(term);
        writer.commit()?;
        self.reader.reload()?;
        let after = self.count()?;

        Ok((before - after) as u32)
    }

    /// Search with BM25 scoring.
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<RetrievalHit>> {
        self.search_inner(query, None, top_k)
    }

    /// Search with document filter.
    pub fn search_filtered(
        &self,
        query: &str,
        doc_ids: &[Uuid],
        top_k: usize,
    ) -> Result<Vec<RetrievalHit>> {
        let filter: Vec<String> = doc_ids.iter().map(|id| id.to_string()).collect();
        self.search_inner(query, Some(&filter), top_k)
    }

    fn search_inner(
        &self,
        query_str: &str,
        doc_id_filter: Option<&[String]>,
        top_k: usize,
    ) -> Result<Vec<RetrievalHit>> {
        let searcher = self.reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.f_text]);

        let query = query_parser.parse_query(query_str).unwrap_or_else(|_| {
            // Fallback: treat as bag-of-words OR query
            let terms: Vec<String> = query_str
                .split_whitespace()
                .map(|w| w.to_string())
                .collect();
            let fallback = terms.join(" OR ");
            query_parser
                .parse_query(&fallback)
                .unwrap_or_else(|_| Box::new(tantivy::query::AllQuery))
        });

        let top_docs = searcher.search(&query, &TopDocs::with_limit(top_k * 2))?;

        let mut hits = Vec::new();
        for (score, doc_addr) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_addr)?;

            // Apply doc_id filter if specified
            if let Some(filter) = doc_id_filter {
                let doc_id_val = doc
                    .get_first(self.f_doc_id)
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if !filter.iter().any(|f| f == doc_id_val) {
                    continue;
                }
            }

            let chunk_json = doc
                .get_first(self.f_chunk_json)
                .and_then(|v| v.as_str())
                .unwrap_or("{}");

            let chunk: Chunk = serde_json::from_str(chunk_json)
                .map_err(|e| IndexError::Index(format!("failed to deserialize chunk: {e}")))?;

            hits.push(RetrievalHit {
                chunk,
                score_sparse: score,
                score_dense: 0.0,
                score_fused: 0.0,
                score_rerank: 0.0,
            });

            if hits.len() >= top_k {
                break;
            }
        }

        Ok(hits)
    }

    /// Number of indexed chunks.
    pub fn count(&self) -> Result<u64> {
        let searcher = self.reader.searcher();
        Ok(searcher.num_docs())
    }
}

/// Retriever trait implementation for sparse BM25 search.
pub struct SparseRetriever {
    index: SparseIndex,
}

impl SparseRetriever {
    pub fn new(index: SparseIndex) -> Self {
        Self { index }
    }
}

impl cwc_core::traits::Retriever for SparseRetriever {
    fn retrieve(&self, query: &str, top_k: usize) -> cwc_core::Result<Vec<RetrievalHit>> {
        self.index
            .search(query, top_k)
            .map_err(|e| cwc_core::CwcError::Retrieval(e.to_string()))
    }
}

/// Full index rebuild from chunks.
pub fn build_index(chunks: &[Chunk], index_dir: &Path) -> Result<u32> {
    // Remove existing index
    if index_dir.exists() {
        std::fs::remove_dir_all(index_dir)?;
    }
    let index = SparseIndex::open_or_create(index_dir)?;
    index.index_chunks(chunks)
}

/// Incremental index update: delete old doc chunks, add new ones.
pub fn update_index(new_chunks: &[Chunk], index_dir: &Path) -> Result<u32> {
    let index = SparseIndex::open_or_create(index_dir)?;

    // Collect unique doc_ids to delete
    let doc_ids: Vec<Uuid> = new_chunks
        .iter()
        .map(|c| c.doc_id)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    for doc_id in &doc_ids {
        index.delete_document(*doc_id)?;
    }

    index.index_chunks(new_chunks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::Instant;

    fn make_chunk(doc_id: Uuid, text: &str, idx: usize) -> Chunk {
        Chunk {
            chunk_id: Uuid::new_v5(&Uuid::NAMESPACE_OID, format!("{doc_id}-{idx}").as_bytes()),
            doc_id,
            doc_version: 1,
            source_path: "test.md".to_string(),
            section_path: vec!["Section".to_string()],
            char_offset: idx * 100,
            char_len: text.len(),
            token_count: 10,
            text: text.to_string(),
            metadata: HashMap::new(),
        }
    }

    fn make_test_corpus() -> (Uuid, Vec<Chunk>) {
        let doc_id = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"test-doc");
        let chunks = vec![
            make_chunk(doc_id, "Rust uses ownership for memory management without a garbage collector", 0),
            make_chunk(doc_id, "Borrowing allows references to data without taking ownership", 1),
            make_chunk(doc_id, "Lifetimes ensure references are valid for the right duration", 2),
            make_chunk(doc_id, "Traits define shared behavior similar to interfaces in other languages", 3),
            make_chunk(doc_id, "Error handling in Rust uses Result and Option types instead of exceptions", 4),
            make_chunk(doc_id, "The context window compiler processes documents into token-budgeted bundles", 5),
            make_chunk(doc_id, "BM25 is a ranking function used in information retrieval systems", 6),
            make_chunk(doc_id, "Vector embeddings capture semantic meaning of text for similarity search", 7),
            make_chunk(doc_id, "Cargo is Rust's build system and package manager", 8),
            make_chunk(doc_id, "Pattern matching with match expressions is exhaustive in Rust", 9),
        ];
        (doc_id, chunks)
    }

    #[test]
    fn test_sparse_index_count() {
        let idx = SparseIndex::open_in_ram().unwrap();
        let (_, chunks) = make_test_corpus();
        idx.index_chunks(&chunks).unwrap();
        assert_eq!(idx.count().unwrap(), 10);
    }

    #[test]
    fn test_sparse_index_100_chunks() {
        let idx = SparseIndex::open_in_ram().unwrap();
        let doc_id = Uuid::new_v4();
        let chunks: Vec<Chunk> = (0..100)
            .map(|i| make_chunk(doc_id, &format!("Chunk number {i} with some text content"), i))
            .collect();
        idx.index_chunks(&chunks).unwrap();
        assert_eq!(idx.count().unwrap(), 100);
    }

    #[test]
    fn test_sparse_search_exact_term() {
        let idx = SparseIndex::open_in_ram().unwrap();
        let (_, chunks) = make_test_corpus();
        idx.index_chunks(&chunks).unwrap();

        let results = idx.search("garbage collector", 5).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].chunk.text.contains("garbage collector"));
    }

    #[test]
    fn test_sparse_search_no_results() {
        let idx = SparseIndex::open_in_ram().unwrap();
        let (_, chunks) = make_test_corpus();
        idx.index_chunks(&chunks).unwrap();

        let results = idx.search("xylophone quantum neutrino", 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_sparse_bm25_relevance() {
        let idx = SparseIndex::open_in_ram().unwrap();
        let (_, chunks) = make_test_corpus();
        idx.index_chunks(&chunks).unwrap();

        let results = idx.search("ownership memory", 10).unwrap();
        assert!(!results.is_empty());
        // The chunk about ownership should rank first
        assert!(
            results[0].chunk.text.contains("ownership"),
            "expected ownership chunk first, got: {}",
            results[0].chunk.text
        );
        // Scores should be descending
        for w in results.windows(2) {
            assert!(w[0].score_sparse >= w[1].score_sparse);
        }
    }

    #[test]
    fn test_sparse_phrase_query() {
        let idx = SparseIndex::open_in_ram().unwrap();
        let (_, chunks) = make_test_corpus();
        idx.index_chunks(&chunks).unwrap();

        let results = idx.search("\"context window\"", 5).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].chunk.text.contains("context window"));
    }

    #[test]
    fn test_sparse_document_filter() {
        let idx = SparseIndex::open_in_ram().unwrap();
        let doc_a = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"doc-a");
        let doc_b = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"doc-b");

        let chunks = vec![
            make_chunk(doc_a, "Rust ownership model for memory safety", 0),
            make_chunk(doc_b, "Rust ownership and borrowing rules", 1),
        ];
        idx.index_chunks(&chunks).unwrap();

        let results = idx.search_filtered("ownership", &[doc_a], 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.doc_id, doc_a);
    }

    #[test]
    fn test_sparse_reindex_idempotent() {
        let idx = SparseIndex::open_in_ram().unwrap();
        let (_, chunks) = make_test_corpus();

        idx.index_chunks(&chunks).unwrap();
        assert_eq!(idx.count().unwrap(), 10);

        // Re-index same chunks
        idx.index_chunks(&chunks).unwrap();
        assert_eq!(idx.count().unwrap(), 10);
    }

    #[test]
    fn test_sparse_delete_document() {
        let idx = SparseIndex::open_in_ram().unwrap();
        let doc_a = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"doc-a");
        let doc_b = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"doc-b");

        let chunks = vec![
            make_chunk(doc_a, "Doc A chunk one", 0),
            make_chunk(doc_a, "Doc A chunk two", 1),
            make_chunk(doc_b, "Doc B chunk one", 0),
        ];
        idx.index_chunks(&chunks).unwrap();
        assert_eq!(idx.count().unwrap(), 3);

        let deleted = idx.delete_document(doc_a).unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(idx.count().unwrap(), 1);
    }

    #[test]
    fn test_sparse_incremental_update() {
        let idx = SparseIndex::open_in_ram().unwrap();
        let doc_id = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"update-doc");

        let chunks_v1 = vec![make_chunk(doc_id, "Old content about Rust", 0)];
        idx.index_chunks(&chunks_v1).unwrap();

        let results = idx.search("Old content", 5).unwrap();
        assert!(!results.is_empty());

        // Delete and re-add with new content
        idx.delete_document(doc_id).unwrap();
        let chunks_v2 = vec![make_chunk(doc_id, "New updated content about Rust", 0)];
        idx.index_chunks(&chunks_v2).unwrap();

        let results = idx.search("New updated", 5).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].chunk.text.contains("New updated"));

        // Old-only term should not match new content
        let results = idx.search("\"Old content\"", 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_sparse_index_persist() {
        let dir = std::env::temp_dir().join("cwc_sparse_persist");
        let _ = std::fs::remove_dir_all(&dir);

        let (_, chunks) = make_test_corpus();

        // Create and index
        {
            let idx = SparseIndex::open_or_create(&dir).unwrap();
            idx.index_chunks(&chunks).unwrap();
            assert_eq!(idx.count().unwrap(), 10);
        }

        // Reopen and verify
        {
            let idx = SparseIndex::open_or_create(&dir).unwrap();
            assert_eq!(idx.count().unwrap(), 10);
            let results = idx.search("ownership", 5).unwrap();
            assert!(!results.is_empty());
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_sparse_large_batch_performance() {
        let idx = SparseIndex::open_in_ram().unwrap();
        let doc_id = Uuid::new_v4();
        let chunks: Vec<Chunk> = (0..10_000)
            .map(|i| {
                make_chunk(
                    doc_id,
                    &format!("Document chunk {i} with various words about topic {}", i % 50),
                    i,
                )
            })
            .collect();

        idx.index_chunks(&chunks).unwrap();
        assert_eq!(idx.count().unwrap(), 10_000);

        let start = Instant::now();
        let results = idx.search("topic document words", 10).unwrap();
        let elapsed = start.elapsed();

        assert!(!results.is_empty());
        assert!(
            elapsed.as_millis() < 100,
            "search took {}ms, expected < 100ms",
            elapsed.as_millis()
        );
    }

    #[test]
    fn test_sparse_retriever_trait() {
        use cwc_core::traits::Retriever;

        let idx = SparseIndex::open_in_ram().unwrap();
        let (_, chunks) = make_test_corpus();
        idx.index_chunks(&chunks).unwrap();

        let retriever = SparseRetriever::new(idx);
        let results = retriever.retrieve("ownership", 5).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].chunk.text.contains("ownership"));
        // score_sparse should be populated, others zero
        assert!(results[0].score_sparse > 0.0);
        assert_eq!(results[0].score_dense, 0.0);
        assert_eq!(results[0].score_fused, 0.0);
        assert_eq!(results[0].score_rerank, 0.0);
    }

    #[test]
    fn test_sparse_search_query_fallback() {
        let idx = SparseIndex::open_in_ram().unwrap();
        let (_, chunks) = make_test_corpus();
        idx.index_chunks(&chunks).unwrap();

        // Invalid query syntax should fall back to bag-of-words
        let results = idx.search("ownership AND OR memory (invalid", 5).unwrap();
        // Should not panic, may return results
        let _ = results;
    }
}
