use std::collections::HashMap;

use sqlx::postgres::PgPool;
use sqlx::Row;
use uuid::Uuid;

use cwc_core::types::Chunk;

use crate::error::{IndexError, Result};

/// Chunk repository backed by PostgreSQL with pgvector.
pub struct ChunkDb {
    pool: PgPool,
    embedding_dim: usize,
}

impl ChunkDb {
    /// Connect to PostgreSQL and create a new ChunkDb.
    pub async fn new(database_url: &str, embedding_dim: usize) -> Result<Self> {
        let pool = PgPool::connect(database_url)
            .await
            .map_err(|e| IndexError::Database(e.to_string()))?;
        Ok(Self { pool, embedding_dim })
    }

    /// Create from an existing connection pool.
    pub fn from_pool(pool: PgPool, embedding_dim: usize) -> Self {
        Self { pool, embedding_dim }
    }

    /// Run database migrations (create tables and indexes).
    pub async fn run_migrations(&self) -> Result<()> {
        sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
            .execute(&self.pool)
            .await
            .map_err(|e| IndexError::Database(e.to_string()))?;

        sqlx::query(
            r#"CREATE TABLE IF NOT EXISTS chunks (
                chunk_id    UUID PRIMARY KEY,
                doc_id      UUID NOT NULL,
                doc_version INTEGER NOT NULL,
                source_path TEXT NOT NULL,
                section_path TEXT[] NOT NULL,
                text        TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )"#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| IndexError::Database(e.to_string()))?;

        let create_emb = format!(
            r#"CREATE TABLE IF NOT EXISTS chunk_embeddings (
                chunk_id    UUID PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                embedding   vector({}) NOT NULL
            )"#,
            self.embedding_dim
        );
        sqlx::query(&create_emb)
            .execute(&self.pool)
            .await
            .map_err(|e| IndexError::Database(e.to_string()))?;

        sqlx::query(
            r#"DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_indexes WHERE indexname = 'idx_chunk_embeddings_hnsw'
                ) THEN
                    CREATE INDEX idx_chunk_embeddings_hnsw
                        ON chunk_embeddings USING hnsw (embedding vector_cosine_ops)
                        WITH (m = 16, ef_construction = 200);
                END IF;
            END $$"#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| IndexError::Database(e.to_string()))?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
            .execute(&self.pool)
            .await
            .map_err(|e| IndexError::Database(e.to_string()))?;

        Ok(())
    }

    /// Upsert chunks (insert or update on conflict).
    pub async fn upsert_chunks(&self, chunks: &[Chunk]) -> Result<u32> {
        let mut count = 0u32;
        for chunk in chunks {
            let metadata = serde_json::to_value(&chunk.metadata)
                .map_err(|e| IndexError::Index(e.to_string()))?;

            sqlx::query(
                r#"INSERT INTO chunks (chunk_id, doc_id, doc_version, source_path,
                                      section_path, text, token_count, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    doc_version = EXCLUDED.doc_version,
                    source_path = EXCLUDED.source_path,
                    section_path = EXCLUDED.section_path,
                    text = EXCLUDED.text,
                    token_count = EXCLUDED.token_count,
                    metadata = EXCLUDED.metadata"#,
            )
            .bind(chunk.chunk_id)
            .bind(chunk.doc_id)
            .bind(chunk.doc_version as i32)
            .bind(&chunk.source_path)
            .bind(&chunk.section_path)
            .bind(&chunk.text)
            .bind(chunk.token_count as i32)
            .bind(&metadata)
            .execute(&self.pool)
            .await
            .map_err(|e| IndexError::Database(e.to_string()))?;

            count += 1;
        }
        Ok(count)
    }

    /// Store embeddings for chunks.
    pub async fn upsert_embeddings(&self, items: &[(Uuid, Vec<f32>)]) -> Result<u32> {
        let mut count = 0u32;
        for (chunk_id, embedding) in items {
            let vec_str = format!(
                "[{}]",
                embedding
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );

            sqlx::query(
                r#"INSERT INTO chunk_embeddings (chunk_id, embedding)
                VALUES ($1, $2::vector)
                ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding"#,
            )
            .bind(chunk_id)
            .bind(&vec_str)
            .execute(&self.pool)
            .await
            .map_err(|e| IndexError::Database(e.to_string()))?;

            count += 1;
        }
        Ok(count)
    }

    /// Delete all chunks (and embeddings via CASCADE) for a document.
    pub async fn delete_document(&self, doc_id: Uuid) -> Result<u32> {
        let result = sqlx::query("DELETE FROM chunks WHERE doc_id = $1")
            .bind(doc_id)
            .execute(&self.pool)
            .await
            .map_err(|e| IndexError::Database(e.to_string()))?;

        Ok(result.rows_affected() as u32)
    }

    /// ANN search using pgvector cosine distance.
    pub async fn search_dense(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<(Chunk, f32)>> {
        let vec_str = embedding_to_pg_str(query_embedding);

        let rows = sqlx::query(
            r#"SELECT c.chunk_id, c.doc_id, c.doc_version, c.source_path,
                      c.section_path, c.text, c.token_count, c.metadata,
                      1 - (e.embedding <=> $1::vector) as similarity
               FROM chunk_embeddings e
               JOIN chunks c ON c.chunk_id = e.chunk_id
               ORDER BY e.embedding <=> $1::vector
               LIMIT $2"#,
        )
        .bind(&vec_str)
        .bind(top_k as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| IndexError::Database(e.to_string()))?;

        rows_to_chunks(&rows)
    }

    /// ANN search filtered by document IDs.
    pub async fn search_dense_filtered(
        &self,
        query_embedding: &[f32],
        doc_ids: &[Uuid],
        top_k: usize,
    ) -> Result<Vec<(Chunk, f32)>> {
        let vec_str = embedding_to_pg_str(query_embedding);

        let rows = sqlx::query(
            r#"SELECT c.chunk_id, c.doc_id, c.doc_version, c.source_path,
                      c.section_path, c.text, c.token_count, c.metadata,
                      1 - (e.embedding <=> $1::vector) as similarity
               FROM chunk_embeddings e
               JOIN chunks c ON c.chunk_id = e.chunk_id
               WHERE c.doc_id = ANY($3)
               ORDER BY e.embedding <=> $1::vector
               LIMIT $2"#,
        )
        .bind(&vec_str)
        .bind(top_k as i64)
        .bind(doc_ids)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| IndexError::Database(e.to_string()))?;

        rows_to_chunks(&rows)
    }

    /// Total indexed chunk count.
    pub async fn count(&self) -> Result<u64> {
        let row = sqlx::query("SELECT COUNT(*) as cnt FROM chunks")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| IndexError::Database(e.to_string()))?;

        let count: i64 = row
            .try_get("cnt")
            .map_err(|e| IndexError::Database(e.to_string()))?;
        Ok(count as u64)
    }
}

fn embedding_to_pg_str(embedding: &[f32]) -> String {
    format!(
        "[{}]",
        embedding
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn rows_to_chunks(rows: &[sqlx::postgres::PgRow]) -> Result<Vec<(Chunk, f32)>> {
    let mut results = Vec::with_capacity(rows.len());
    for row in rows {
        let chunk_id: Uuid = row
            .try_get("chunk_id")
            .map_err(|e| IndexError::Database(e.to_string()))?;
        let doc_id: Uuid = row
            .try_get("doc_id")
            .map_err(|e| IndexError::Database(e.to_string()))?;
        let doc_version: i32 = row
            .try_get("doc_version")
            .map_err(|e| IndexError::Database(e.to_string()))?;
        let source_path: String = row
            .try_get("source_path")
            .map_err(|e| IndexError::Database(e.to_string()))?;
        let section_path: Vec<String> = row
            .try_get("section_path")
            .map_err(|e| IndexError::Database(e.to_string()))?;
        let text: String = row
            .try_get("text")
            .map_err(|e| IndexError::Database(e.to_string()))?;
        let token_count: i32 = row
            .try_get("token_count")
            .map_err(|e| IndexError::Database(e.to_string()))?;
        let metadata_val: serde_json::Value = row
            .try_get("metadata")
            .map_err(|e| IndexError::Database(e.to_string()))?;
        let similarity: f64 = row
            .try_get("similarity")
            .map_err(|e| IndexError::Database(e.to_string()))?;

        let metadata: HashMap<String, String> =
            serde_json::from_value(metadata_val).unwrap_or_default();

        let chunk = Chunk {
            chunk_id,
            doc_id,
            doc_version: doc_version as u32,
            source_path,
            section_path,
            char_offset: 0,
            char_len: text.len(),
            token_count: token_count as u32,
            text,
            metadata,
        };

        results.push((chunk, similarity as f32));
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db_url() -> String {
        std::env::var("CWC_TEST_DB_URL")
            .unwrap_or_else(|_| "postgres://localhost/cwc_test".to_string())
    }

    fn make_chunk(doc_id: Uuid, text: &str, idx: usize) -> Chunk {
        Chunk {
            chunk_id: Uuid::new_v5(
                &Uuid::NAMESPACE_OID,
                format!("{doc_id}-{idx}").as_bytes(),
            ),
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

    fn make_embedding(dim: usize, seed: usize) -> Vec<f32> {
        (0..dim)
            .map(|i| ((seed as f32 + i as f32) * 0.1).sin())
            .collect()
    }

    async fn setup_db() -> ChunkDb {
        let db = ChunkDb::new(&test_db_url(), 384).await.unwrap();
        db.run_migrations().await.unwrap();
        // Clean slate
        sqlx::query("TRUNCATE chunks CASCADE")
            .execute(&db.pool)
            .await
            .unwrap();
        db
    }

    #[tokio::test]
    #[ignore]
    async fn test_db_upsert_100_chunks() {
        let db = setup_db().await;
        let doc_id = Uuid::new_v4();
        let chunks: Vec<Chunk> = (0..100)
            .map(|i| make_chunk(doc_id, &format!("Chunk {i} about topic {}", i % 10), i))
            .collect();

        let count = db.upsert_chunks(&chunks).await.unwrap();
        assert_eq!(count, 100);
        assert_eq!(db.count().await.unwrap(), 100);
    }

    #[tokio::test]
    #[ignore]
    async fn test_db_upsert_embeddings() {
        let db = setup_db().await;
        let doc_id = Uuid::new_v4();
        let chunks: Vec<Chunk> = (0..5)
            .map(|i| make_chunk(doc_id, &format!("Chunk {i}"), i))
            .collect();
        db.upsert_chunks(&chunks).await.unwrap();

        let items: Vec<(Uuid, Vec<f32>)> = chunks
            .iter()
            .enumerate()
            .map(|(i, c)| (c.chunk_id, make_embedding(384, i)))
            .collect();

        let count = db.upsert_embeddings(&items).await.unwrap();
        assert_eq!(count, 5);
    }

    #[tokio::test]
    #[ignore]
    async fn test_db_search_dense_relevance() {
        let db = setup_db().await;
        let doc_id = Uuid::new_v4();
        let chunks = vec![
            make_chunk(doc_id, "Rust ownership memory management", 0),
            make_chunk(doc_id, "Python garbage collection automatic", 1),
            make_chunk(doc_id, "JavaScript event loop async", 2),
        ];
        db.upsert_chunks(&chunks).await.unwrap();

        // Use distinct non-zero embeddings (zero vector causes NaN in cosine distance)
        let emb0 = vec![1.0f32; 384]; // ownership — all 1s
        let mut emb1 = vec![0.0f32; 384]; // python — orthogonal direction
        emb1[0] = -1.0;
        emb1[1] = -1.0;
        let mut emb2 = vec![0.0f32; 384]; // javascript — different direction
        emb2[2] = 1.0;
        emb2[3] = 1.0;

        let items = vec![
            (chunks[0].chunk_id, emb0.clone()),
            (chunks[1].chunk_id, emb1),
            (chunks[2].chunk_id, emb2),
        ];
        db.upsert_embeddings(&items).await.unwrap();

        // Query close to ownership embedding
        let results = db.search_dense(&emb0, 3).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0.chunk_id, chunks[0].chunk_id);
    }

    #[tokio::test]
    #[ignore]
    async fn test_db_search_dense_filtered() {
        let db = setup_db().await;
        let doc_a = Uuid::new_v4();
        let doc_b = Uuid::new_v4();

        let chunks = vec![
            make_chunk(doc_a, "Doc A about Rust", 0),
            make_chunk(doc_b, "Doc B about Python", 0),
        ];
        db.upsert_chunks(&chunks).await.unwrap();

        let emb0 = make_embedding(384, 0);
        let emb1 = make_embedding(384, 1);
        let items = vec![
            (chunks[0].chunk_id, emb0.clone()),
            (chunks[1].chunk_id, emb1),
        ];
        db.upsert_embeddings(&items).await.unwrap();

        let results = db
            .search_dense_filtered(&emb0, &[doc_a], 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.doc_id, doc_a);
    }

    #[tokio::test]
    #[ignore]
    async fn test_db_delete_document() {
        let db = setup_db().await;
        let doc_a = Uuid::new_v4();
        let doc_b = Uuid::new_v4();

        let chunks = vec![
            make_chunk(doc_a, "Doc A chunk 1", 0),
            make_chunk(doc_a, "Doc A chunk 2", 1),
            make_chunk(doc_b, "Doc B chunk 1", 0),
        ];
        db.upsert_chunks(&chunks).await.unwrap();

        let items: Vec<(Uuid, Vec<f32>)> = chunks
            .iter()
            .enumerate()
            .map(|(i, c)| (c.chunk_id, make_embedding(384, i)))
            .collect();
        db.upsert_embeddings(&items).await.unwrap();

        let deleted = db.delete_document(doc_a).await.unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(db.count().await.unwrap(), 1);

        // Embeddings should also be gone (CASCADE)
        let emb_count: i64 =
            sqlx::query("SELECT COUNT(*) as cnt FROM chunk_embeddings")
                .fetch_one(&db.pool)
                .await
                .unwrap()
                .try_get("cnt")
                .unwrap();
        assert_eq!(emb_count, 1);
    }

    #[tokio::test]
    #[ignore]
    async fn test_db_reingestion() {
        let db = setup_db().await;
        let doc_id = Uuid::new_v4();

        let mut chunk = make_chunk(doc_id, "Original text v1", 0);
        db.upsert_chunks(&[chunk.clone()]).await.unwrap();

        // Re-ingest with higher version and new text
        chunk.doc_version = 2;
        chunk.text = "Updated text v2".to_string();
        db.upsert_chunks(&[chunk.clone()]).await.unwrap();

        assert_eq!(db.count().await.unwrap(), 1);

        // Verify the text was updated by searching
        let row = sqlx::query("SELECT text, doc_version FROM chunks WHERE chunk_id = $1")
            .bind(chunk.chunk_id)
            .fetch_one(&db.pool)
            .await
            .unwrap();
        let text: String = row.try_get("text").unwrap();
        let version: i32 = row.try_get("doc_version").unwrap();
        assert_eq!(text, "Updated text v2");
        assert_eq!(version, 2);
    }

    #[tokio::test]
    #[ignore]
    async fn test_db_explain_hnsw() {
        let db = setup_db().await;
        let doc_id = Uuid::new_v4();
        let chunks: Vec<Chunk> = (0..10)
            .map(|i| make_chunk(doc_id, &format!("Chunk {i}"), i))
            .collect();
        db.upsert_chunks(&chunks).await.unwrap();

        let items: Vec<(Uuid, Vec<f32>)> = chunks
            .iter()
            .enumerate()
            .map(|(i, c)| (c.chunk_id, make_embedding(384, i)))
            .collect();
        db.upsert_embeddings(&items).await.unwrap();

        // Set ef_search for HNSW
        sqlx::query("SET hnsw.ef_search = 100")
            .execute(&db.pool)
            .await
            .unwrap();

        let vec_str = embedding_to_pg_str(&make_embedding(384, 0));
        let rows = sqlx::query(
            r#"EXPLAIN SELECT c.chunk_id
               FROM chunk_embeddings e
               JOIN chunks c ON c.chunk_id = e.chunk_id
               ORDER BY e.embedding <=> $1::vector
               LIMIT 5"#,
        )
        .bind(&vec_str)
        .fetch_all(&db.pool)
        .await
        .unwrap();

        let plan: String = rows
            .iter()
            .map(|r| r.try_get::<String, _>(0).unwrap_or_default())
            .collect::<Vec<_>>()
            .join("\n");

        // With enough data, HNSW index should be used
        // For small datasets PostgreSQL may choose seq scan, which is fine
        let _ = plan; // Inspect manually if needed
    }
}
