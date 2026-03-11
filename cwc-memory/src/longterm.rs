use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;

use cwc_core::error::{CwcError, Result};

/// Category of a long-term memory entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryCategory {
    UserPreference,
    ProjectFact,
    PriorDecision,
    Correction,
}

impl MemoryCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::UserPreference => "user_preference",
            Self::ProjectFact => "project_fact",
            Self::PriorDecision => "prior_decision",
            Self::Correction => "correction",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "user_preference" => Some(Self::UserPreference),
            "project_fact" => Some(Self::ProjectFact),
            "prior_decision" => Some(Self::PriorDecision),
            "correction" => Some(Self::Correction),
            _ => None,
        }
    }
}

/// A single long-term memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub key: String,
    pub value: String,
    pub category: MemoryCategory,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub access_count: u32,
}

/// Escape SIMILAR TO metacharacters in a keyword.
fn escape_similar_to(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '%' | '_' | '|' | '*' | '+' | '?' | '{' | '}' | '(' | ')' | '[' | ']' => {
                out.push('\\');
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

/// PostgreSQL-backed long-term memory store.
pub struct LongTermMemory {
    db: PgPool,
}

impl LongTermMemory {
    pub async fn new(db: PgPool) -> Result<Self> {
        // Create table if it doesn't exist
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS long_term_memory (
                key          TEXT PRIMARY KEY,
                value        TEXT NOT NULL,
                category     TEXT NOT NULL,
                created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                access_count INTEGER NOT NULL DEFAULT 0
            )",
        )
        .execute(&db)
        .await
        .map_err(|e| CwcError::Config(format!("failed to create memory table: {e}")))?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_ltm_category ON long_term_memory(category)",
        )
        .execute(&db)
        .await
        .map_err(|e| CwcError::Config(format!("failed to create index: {e}")))?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_ltm_updated ON long_term_memory(updated_at DESC)",
        )
        .execute(&db)
        .await
        .map_err(|e| CwcError::Config(format!("failed to create index: {e}")))?;

        Ok(Self { db })
    }

    /// Store or update a memory entry.
    pub async fn upsert(&self, key: &str, value: &str, category: MemoryCategory) -> Result<()> {
        sqlx::query(
            "INSERT INTO long_term_memory (key, value, category)
             VALUES ($1, $2, $3)
             ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value,
                    category = EXCLUDED.category,
                    updated_at = now()",
        )
        .bind(key)
        .bind(value)
        .bind(category.as_str())
        .execute(&self.db)
        .await
        .map_err(|e| CwcError::Config(format!("upsert failed: {e}")))?;
        Ok(())
    }

    /// Retrieve memories relevant to a query.
    /// Uses keyword matching + recency + access frequency.
    pub async fn retrieve(
        &self,
        query: &str,
        max_entries: usize,
    ) -> Result<Vec<MemoryEntry>> {
        // Simple keyword-based retrieval: match any word from the query in key or value
        let keywords: Vec<String> = query
            .split_whitespace()
            .filter(|w| w.len() >= 3)
            .map(|w| escape_similar_to(&w.to_lowercase()))
            .collect();

        if keywords.is_empty() {
            return self.list_recent(max_entries).await;
        }

        // Build a query that matches any keyword, scored by recency + access_count
        let pattern = keywords.join("|");
        let rows = sqlx::query_as::<_, MemoryRow>(
            "SELECT key, value, category, created_at, updated_at, access_count
             FROM long_term_memory
             WHERE LOWER(key) SIMILAR TO $1 OR LOWER(value) SIMILAR TO $1
             ORDER BY access_count DESC, updated_at DESC
             LIMIT $2",
        )
        .bind(format!("%({pattern})%"))
        .bind(max_entries as i64)
        .fetch_all(&self.db)
        .await
        .map_err(|e| CwcError::Config(format!("retrieve failed: {e}")))?;

        // Increment access counts for returned entries
        for row in &rows {
            let _ = sqlx::query(
                "UPDATE long_term_memory SET access_count = access_count + 1 WHERE key = $1",
            )
            .bind(&row.key)
            .execute(&self.db)
            .await;
        }

        Ok(rows.into_iter().map(|r| r.into_entry()).collect())
    }

    /// Get all memories in a category.
    pub async fn by_category(&self, category: MemoryCategory) -> Result<Vec<MemoryEntry>> {
        let rows = sqlx::query_as::<_, MemoryRow>(
            "SELECT key, value, category, created_at, updated_at, access_count
             FROM long_term_memory
             WHERE category = $1
             ORDER BY updated_at DESC",
        )
        .bind(category.as_str())
        .fetch_all(&self.db)
        .await
        .map_err(|e| CwcError::Config(format!("by_category failed: {e}")))?;

        Ok(rows.into_iter().map(|r| r.into_entry()).collect())
    }

    /// Delete a memory entry. Returns true if it existed.
    pub async fn delete(&self, key: &str) -> Result<bool> {
        let result = sqlx::query("DELETE FROM long_term_memory WHERE key = $1")
            .bind(key)
            .execute(&self.db)
            .await
            .map_err(|e| CwcError::Config(format!("delete failed: {e}")))?;
        Ok(result.rows_affected() > 0)
    }

    /// List all entries.
    pub async fn list_all(&self) -> Result<Vec<MemoryEntry>> {
        let rows = sqlx::query_as::<_, MemoryRow>(
            "SELECT key, value, category, created_at, updated_at, access_count
             FROM long_term_memory
             ORDER BY updated_at DESC",
        )
        .fetch_all(&self.db)
        .await
        .map_err(|e| CwcError::Config(format!("list_all failed: {e}")))?;

        Ok(rows.into_iter().map(|r| r.into_entry()).collect())
    }

    async fn list_recent(&self, max_entries: usize) -> Result<Vec<MemoryEntry>> {
        let rows = sqlx::query_as::<_, MemoryRow>(
            "SELECT key, value, category, created_at, updated_at, access_count
             FROM long_term_memory
             ORDER BY updated_at DESC
             LIMIT $1",
        )
        .bind(max_entries as i64)
        .fetch_all(&self.db)
        .await
        .map_err(|e| CwcError::Config(format!("list_recent failed: {e}")))?;

        Ok(rows.into_iter().map(|r| r.into_entry()).collect())
    }
}

#[derive(sqlx::FromRow)]
struct MemoryRow {
    key: String,
    value: String,
    category: String,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    access_count: i32,
}

impl MemoryRow {
    fn into_entry(self) -> MemoryEntry {
        MemoryEntry {
            key: self.key,
            value: self.value,
            category: MemoryCategory::parse(&self.category)
                .unwrap_or(MemoryCategory::ProjectFact),
            created_at: self.created_at,
            updated_at: self.updated_at,
            access_count: self.access_count as u32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // All PG-backed tests are ignored — require running PostgreSQL with CWC_TEST_DB_URL
    #[tokio::test]
    #[ignore]
    async fn test_longterm_upsert_retrieve() {
        let url = std::env::var("CWC_TEST_DB_URL")
            .unwrap_or_else(|_| "postgres://localhost/cwc_test".to_string());
        let pool = PgPool::connect(&url).await.unwrap();
        let mem = LongTermMemory::new(pool).await.unwrap();

        mem.upsert("pref:units", "metric", MemoryCategory::UserPreference)
            .await
            .unwrap();

        let results = mem.retrieve("units", 10).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].key, "pref:units");
        assert_eq!(results[0].value, "metric");

        // Cleanup
        mem.delete("pref:units").await.unwrap();
    }

    #[tokio::test]
    #[ignore]
    async fn test_longterm_upsert_updates_existing() {
        let url = std::env::var("CWC_TEST_DB_URL")
            .unwrap_or_else(|_| "postgres://localhost/cwc_test".to_string());
        let pool = PgPool::connect(&url).await.unwrap();
        let mem = LongTermMemory::new(pool).await.unwrap();

        mem.upsert("test:update", "old_value", MemoryCategory::ProjectFact)
            .await
            .unwrap();
        mem.upsert("test:update", "new_value", MemoryCategory::ProjectFact)
            .await
            .unwrap();

        let all = mem.list_all().await.unwrap();
        let entry = all.iter().find(|e| e.key == "test:update").unwrap();
        assert_eq!(entry.value, "new_value");

        mem.delete("test:update").await.unwrap();
    }

    #[tokio::test]
    #[ignore]
    async fn test_longterm_by_category() {
        let url = std::env::var("CWC_TEST_DB_URL")
            .unwrap_or_else(|_| "postgres://localhost/cwc_test".to_string());
        let pool = PgPool::connect(&url).await.unwrap();
        let mem = LongTermMemory::new(pool).await.unwrap();

        mem.upsert("cat:a", "val", MemoryCategory::Correction)
            .await
            .unwrap();
        mem.upsert("cat:b", "val", MemoryCategory::UserPreference)
            .await
            .unwrap();

        let corrections = mem.by_category(MemoryCategory::Correction).await.unwrap();
        assert!(corrections.iter().any(|e| e.key == "cat:a"));
        assert!(!corrections.iter().any(|e| e.key == "cat:b"));

        mem.delete("cat:a").await.unwrap();
        mem.delete("cat:b").await.unwrap();
    }

    #[tokio::test]
    #[ignore]
    async fn test_longterm_delete() {
        let url = std::env::var("CWC_TEST_DB_URL")
            .unwrap_or_else(|_| "postgres://localhost/cwc_test".to_string());
        let pool = PgPool::connect(&url).await.unwrap();
        let mem = LongTermMemory::new(pool).await.unwrap();

        mem.upsert("del:test", "val", MemoryCategory::ProjectFact)
            .await
            .unwrap();
        assert!(mem.delete("del:test").await.unwrap());
        assert!(!mem.delete("del:test").await.unwrap()); // Already deleted
    }

    #[tokio::test]
    #[ignore]
    async fn test_longterm_access_count() {
        let url = std::env::var("CWC_TEST_DB_URL")
            .unwrap_or_else(|_| "postgres://localhost/cwc_test".to_string());
        let pool = PgPool::connect(&url).await.unwrap();
        let mem = LongTermMemory::new(pool).await.unwrap();

        mem.upsert("access:test", "val", MemoryCategory::ProjectFact)
            .await
            .unwrap();

        // Retrieve twice
        let _ = mem.retrieve("access test", 10).await.unwrap();
        let _ = mem.retrieve("access test", 10).await.unwrap();

        let all = mem.list_all().await.unwrap();
        let entry = all.iter().find(|e| e.key == "access:test").unwrap();
        assert!(entry.access_count >= 2);

        mem.delete("access:test").await.unwrap();
    }

    #[test]
    fn test_memory_category_roundtrip() {
        for cat in [
            MemoryCategory::UserPreference,
            MemoryCategory::ProjectFact,
            MemoryCategory::PriorDecision,
            MemoryCategory::Correction,
        ] {
            let s = cat.as_str();
            let parsed = MemoryCategory::parse(s).unwrap();
            assert_eq!(parsed, cat);
        }
    }

    #[test]
    fn test_memory_category_unknown() {
        assert!(MemoryCategory::parse("unknown").is_none());
    }

    #[test]
    fn test_memory_category_serde_roundtrip() {
        for cat in [
            MemoryCategory::UserPreference,
            MemoryCategory::ProjectFact,
            MemoryCategory::PriorDecision,
            MemoryCategory::Correction,
        ] {
            let json = serde_json::to_string(&cat).unwrap();
            let back: MemoryCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(back, cat);
        }
    }

    #[test]
    fn test_memory_entry_serde_roundtrip() {
        use chrono::Utc;
        let entry = MemoryEntry {
            key: "test:key".to_string(),
            value: "test value".to_string(),
            category: MemoryCategory::UserPreference,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            access_count: 5,
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: MemoryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back.key, entry.key);
        assert_eq!(back.value, entry.value);
        assert_eq!(back.category, entry.category);
        assert_eq!(back.access_count, entry.access_count);
    }

    #[test]
    fn test_escape_similar_to_plain() {
        assert_eq!(escape_similar_to("hello"), "hello");
    }

    #[test]
    fn test_escape_similar_to_metacharacters() {
        assert_eq!(escape_similar_to("c++"), r"c\+\+");
        assert_eq!(escape_similar_to("a%b"), r"a\%b");
        assert_eq!(escape_similar_to("(test)"), r"\(test\)");
        assert_eq!(escape_similar_to("a|b"), r"a\|b");
        assert_eq!(escape_similar_to("a_b"), r"a\_b");
    }
}
