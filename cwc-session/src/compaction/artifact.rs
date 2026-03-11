use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use crate::error::{Result, SessionError};

/// Store full tool output as artifact files for later retrieval.
pub struct ArtifactStore {
    base_dir: PathBuf,
}

impl ArtifactStore {
    /// Create a new artifact store. Creates the base directory if it doesn't exist.
    pub fn new(base_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(base_dir)?;
        Ok(Self {
            base_dir: base_dir.to_path_buf(),
        })
    }

    /// Store content and return the artifact path.
    ///
    /// Filename: `{tool_name}_{call_id}_{timestamp_ms}.txt`
    pub fn store(
        &self,
        tool_name: &str,
        call_id: &str,
        content: &str,
    ) -> Result<PathBuf> {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis();
        // Sanitize tool_name and call_id for filesystem safety
        let safe_tool = sanitize_filename(tool_name);
        let safe_call = sanitize_filename(call_id);
        let filename = format!("{safe_tool}_{safe_call}_{timestamp}.txt");
        let path = self.base_dir.join(&filename);
        std::fs::write(&path, content)?;
        Ok(path)
    }

    /// Retrieve an artifact by path.
    pub fn retrieve(&self, path: &Path) -> Result<String> {
        std::fs::read_to_string(path).map_err(SessionError::Io)
    }

    /// Clean up artifacts older than `max_age`. Returns the number of files removed.
    pub fn cleanup(&self, max_age: Duration) -> Result<usize> {
        let mut removed = 0;
        let now = SystemTime::now();
        let entries = std::fs::read_dir(&self.base_dir)?;
        for entry in entries {
            let entry = entry?;
            let metadata = entry.metadata()?;
            if let Ok(modified) = metadata.modified() {
                if let Ok(age) = now.duration_since(modified) {
                    if age > max_age {
                        std::fs::remove_file(entry.path())?;
                        removed += 1;
                    }
                }
            }
        }
        Ok(removed)
    }

    /// Base directory path.
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }
}

/// Replace filesystem-unsafe characters with underscores.
fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_artifact_store_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let store = ArtifactStore::new(dir.path()).unwrap();

        let content = "line 1\nline 2\nline 3";
        let path = store.store("file.grep", "call_123", content).unwrap();

        assert!(path.exists());
        let retrieved = store.retrieve(&path).unwrap();
        assert_eq!(retrieved, content);
    }

    #[test]
    fn test_artifact_store_unique_filenames() {
        let dir = tempfile::tempdir().unwrap();
        let store = ArtifactStore::new(dir.path()).unwrap();

        let p1 = store.store("tool", "c1", "content1").unwrap();
        let p2 = store.store("tool", "c2", "content2").unwrap();
        // Different call_ids → different filenames
        assert_ne!(p1, p2);
        assert!(p1.exists());
        assert!(p2.exists());
    }

    #[test]
    fn test_artifact_store_cleanup() {
        let dir = tempfile::tempdir().unwrap();
        let store = ArtifactStore::new(dir.path()).unwrap();

        store.store("tool", "c1", "old content").unwrap();
        store.store("tool", "c2", "more content").unwrap();

        // Cleanup with 0 duration removes everything
        let removed = store.cleanup(Duration::ZERO).unwrap();
        assert_eq!(removed, 2);
    }

    #[test]
    fn test_artifact_store_cleanup_keeps_recent() {
        let dir = tempfile::tempdir().unwrap();
        let store = ArtifactStore::new(dir.path()).unwrap();

        store.store("tool", "c1", "recent").unwrap();

        // Cleanup with very large max_age keeps everything
        let removed = store.cleanup(Duration::from_secs(3600)).unwrap();
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_artifact_store_retrieve_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let store = ArtifactStore::new(dir.path()).unwrap();
        let result = store.retrieve(Path::new("/nonexistent/file.txt"));
        assert!(result.is_err());
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("file.grep"), "file.grep");
        assert_eq!(sanitize_filename("call/123"), "call_123");
        assert_eq!(sanitize_filename("a b c"), "a_b_c");
        assert_eq!(sanitize_filename("tool:name"), "tool_name");
    }

    #[test]
    fn test_artifact_creates_base_dir() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("deep").join("nested").join("artifacts");
        assert!(!nested.exists());
        let _store = ArtifactStore::new(&nested).unwrap();
        assert!(nested.exists());
    }
}
