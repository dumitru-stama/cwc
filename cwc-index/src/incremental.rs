use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::error::{IndexError, Result};

/// Tracks file state for incremental change detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestEntry {
    /// SHA-256 hex hash of file content.
    pub hash: String,
    /// Last modified time as seconds since epoch.
    pub mtime: i64,
    /// Document ID assigned during ingestion.
    pub doc_id: Uuid,
}

/// On-disk manifest mapping file paths to their ingestion state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Manifest {
    pub entries: HashMap<String, ManifestEntry>,
}

impl Manifest {
    /// Load a manifest from a JSON file. Returns empty manifest if file doesn't exist.
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let contents = std::fs::read_to_string(path)
            .map_err(|e| IndexError::Index(e.to_string()))?;
        let manifest: Manifest = serde_json::from_str(&contents)
            .map_err(|e| IndexError::Index(format!("manifest parse error: {e}")))?;
        Ok(manifest)
    }

    /// Save the manifest to a JSON file.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| IndexError::Index(e.to_string()))?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| IndexError::Index(format!("manifest serialize error: {e}")))?;
        std::fs::write(path, json)
            .map_err(|e| IndexError::Index(e.to_string()))?;
        Ok(())
    }
}

/// The set of changes detected between a manifest and the filesystem.
#[derive(Debug, Clone, Default)]
pub struct ChangeSet {
    pub added: Vec<PathBuf>,
    pub modified: Vec<PathBuf>,
    pub deleted: Vec<PathBuf>,
    pub unchanged: usize,
}

impl ChangeSet {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.modified.is_empty() && self.deleted.is_empty()
    }

    pub fn total_changes(&self) -> usize {
        self.added.len() + self.modified.len() + self.deleted.len()
    }
}

/// Report of what was updated during incremental processing.
#[derive(Debug, Clone, Default)]
pub struct UpdateReport {
    pub chunks_added: u32,
    pub chunks_updated: u32,
    pub chunks_deleted: u32,
    pub embeddings_computed: u32,
    pub duration_ms: u64,
}

/// Detect changes between the current filesystem state and a saved manifest.
///
/// `paths` are the input file paths to check against the manifest.
/// Files in the manifest but not in `paths` are considered deleted.
pub fn detect_changes(manifest: &Manifest, paths: &[PathBuf]) -> Result<ChangeSet> {
    let mut changeset = ChangeSet::default();
    let mut seen = std::collections::HashSet::new();

    for path in paths {
        let canonical = path
            .canonicalize()
            .unwrap_or_else(|_| path.clone());
        let key = canonical.to_string_lossy().to_string();
        seen.insert(key.clone());

        if let Some(entry) = manifest.entries.get(&key) {
            // Check mtime first (fast path)
            let mtime = file_mtime(&canonical)?;
            if mtime == entry.mtime {
                changeset.unchanged += 1;
                continue;
            }

            // Mtime changed — check content hash
            let hash = file_hash(&canonical)?;
            if hash == entry.hash {
                // Content identical, just mtime drift
                changeset.unchanged += 1;
            } else {
                changeset.modified.push(canonical);
            }
        } else {
            changeset.added.push(canonical);
        }
    }

    // Files in manifest but not in the input paths → deleted
    for key in manifest.entries.keys() {
        if !seen.contains(key) {
            changeset.deleted.push(PathBuf::from(key));
        }
    }

    Ok(changeset)
}

/// Create or update manifest entries for the given paths.
/// Returns the updated manifest.
pub fn update_manifest(
    manifest: &mut Manifest,
    changeset: &ChangeSet,
    doc_ids: &HashMap<PathBuf, Uuid>,
) -> Result<()> {
    // Add new entries
    for path in &changeset.added {
        let key = path.to_string_lossy().to_string();
        let hash = file_hash(path)?;
        let mtime = file_mtime(path)?;
        let doc_id = doc_ids.get(path).copied().unwrap_or_else(Uuid::new_v4);
        manifest.entries.insert(
            key,
            ManifestEntry {
                hash,
                mtime,
                doc_id,
            },
        );
    }

    // Update modified entries
    for path in &changeset.modified {
        let key = path.to_string_lossy().to_string();
        let hash = file_hash(path)?;
        let mtime = file_mtime(path)?;
        let doc_id = manifest
            .entries
            .get(&key)
            .map(|e| e.doc_id)
            .unwrap_or_else(Uuid::new_v4);
        manifest.entries.insert(
            key,
            ManifestEntry {
                hash,
                mtime,
                doc_id,
            },
        );
    }

    // Remove deleted entries
    for path in &changeset.deleted {
        let key = path.to_string_lossy().to_string();
        manifest.entries.remove(&key);
    }

    Ok(())
}

/// Compute SHA-256 hex hash of a file's content.
pub fn file_hash(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| IndexError::Index(format!("cannot read {}: {e}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| IndexError::Index(e.to_string()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Get file modification time as milliseconds since Unix epoch.
fn file_mtime(path: &Path) -> Result<i64> {
    let meta = std::fs::metadata(path)
        .map_err(|e| IndexError::Index(format!("cannot stat {}: {e}", path.display())))?;
    let mtime = meta
        .modified()
        .map_err(|e| IndexError::Index(format!("mtime error: {e}")))?;
    let duration = mtime
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    Ok(duration.as_millis() as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cwc_incr_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_file(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path.canonicalize().unwrap()
    }

    #[test]
    fn test_manifest_roundtrip() {
        let dir = test_dir("manifest_rt");
        let manifest_path = dir.join("manifest.json");

        let mut manifest = Manifest::default();
        let doc_id = Uuid::new_v4();
        manifest.entries.insert(
            "/some/path.md".to_string(),
            ManifestEntry {
                hash: "abcd1234".to_string(),
                mtime: 1700000000,
                doc_id,
            },
        );

        manifest.save(&manifest_path).unwrap();
        let loaded = Manifest::load(&manifest_path).unwrap();
        assert_eq!(loaded.entries.len(), 1);
        assert_eq!(loaded.entries["/some/path.md"].hash, "abcd1234");
        assert_eq!(loaded.entries["/some/path.md"].doc_id, doc_id);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_manifest_load_missing_returns_empty() {
        let m = Manifest::load(Path::new("/nonexistent/manifest.json")).unwrap();
        assert!(m.entries.is_empty());
    }

    #[test]
    fn test_detect_changes_new_file() {
        let dir = test_dir("detect_new");
        let file = write_file(&dir, "new.md", "hello world");

        let manifest = Manifest::default();
        let changes = detect_changes(&manifest, &[file]).unwrap();

        assert_eq!(changes.added.len(), 1);
        assert!(changes.modified.is_empty());
        assert!(changes.deleted.is_empty());
        assert_eq!(changes.unchanged, 0);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_detect_changes_modified_file() {
        let dir = test_dir("detect_mod");
        let file = write_file(&dir, "doc.md", "original content");
        let hash = file_hash(&file).unwrap();
        let mtime = file_mtime(&file).unwrap();

        let mut manifest = Manifest::default();
        manifest.entries.insert(
            file.to_string_lossy().to_string(),
            ManifestEntry {
                hash,
                mtime: mtime - 100, // old mtime
                doc_id: Uuid::new_v4(),
            },
        );

        // Rewrite with different content
        std::fs::write(&file, "modified content").unwrap();

        let changes = detect_changes(&manifest, &[file]).unwrap();
        assert_eq!(changes.modified.len(), 1);
        assert!(changes.added.is_empty());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_detect_changes_deleted_file() {
        let dir = test_dir("detect_del");

        let mut manifest = Manifest::default();
        manifest.entries.insert(
            "/tmp/cwc_incr_detect_del/deleted.md".to_string(),
            ManifestEntry {
                hash: "abc".to_string(),
                mtime: 1700000000,
                doc_id: Uuid::new_v4(),
            },
        );

        // No paths match the manifest entry
        let changes = detect_changes(&manifest, &[]).unwrap();
        assert_eq!(changes.deleted.len(), 1);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_detect_changes_unchanged_file() {
        let dir = test_dir("detect_unchanged");
        let file = write_file(&dir, "stable.md", "stable content");
        let hash = file_hash(&file).unwrap();
        let mtime = file_mtime(&file).unwrap();

        let mut manifest = Manifest::default();
        manifest.entries.insert(
            file.to_string_lossy().to_string(),
            ManifestEntry {
                hash,
                mtime,
                doc_id: Uuid::new_v4(),
            },
        );

        let changes = detect_changes(&manifest, &[file]).unwrap();
        assert_eq!(changes.unchanged, 1);
        assert!(changes.added.is_empty());
        assert!(changes.modified.is_empty());
        assert!(changes.deleted.is_empty());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_update_manifest_adds_entries() {
        let dir = test_dir("update_add");
        let file = write_file(&dir, "new.md", "new content");

        let mut manifest = Manifest::default();
        let changeset = ChangeSet {
            added: vec![file.clone()],
            ..Default::default()
        };
        let doc_id = Uuid::new_v4();
        let mut doc_ids = HashMap::new();
        doc_ids.insert(file.clone(), doc_id);

        update_manifest(&mut manifest, &changeset, &doc_ids).unwrap();

        let key = file.to_string_lossy().to_string();
        assert!(manifest.entries.contains_key(&key));
        assert_eq!(manifest.entries[&key].doc_id, doc_id);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_update_manifest_removes_deleted() {
        let dir = test_dir("update_del");
        let deleted_path = PathBuf::from("/tmp/cwc_incr_update_del/gone.md");

        let mut manifest = Manifest::default();
        manifest.entries.insert(
            deleted_path.to_string_lossy().to_string(),
            ManifestEntry {
                hash: "old".to_string(),
                mtime: 0,
                doc_id: Uuid::new_v4(),
            },
        );

        let changeset = ChangeSet {
            deleted: vec![deleted_path.clone()],
            ..Default::default()
        };

        update_manifest(&mut manifest, &changeset, &HashMap::new()).unwrap();
        assert!(manifest.entries.is_empty());

        std::fs::remove_dir_all(&dir).unwrap_or(());
    }

    #[test]
    fn test_file_hash_deterministic() {
        let dir = test_dir("hash_determ");
        let file = write_file(&dir, "test.txt", "hello world");
        let h1 = file_hash(&file).unwrap();
        let h2 = file_hash(&file).unwrap();
        assert_eq!(h1, h2);
        assert!(!h1.is_empty());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_changeset_is_empty() {
        let cs = ChangeSet::default();
        assert!(cs.is_empty());
        assert_eq!(cs.total_changes(), 0);
    }

    #[test]
    fn test_changeset_total_changes() {
        let cs = ChangeSet {
            added: vec![PathBuf::from("a")],
            modified: vec![PathBuf::from("b"), PathBuf::from("c")],
            deleted: vec![PathBuf::from("d")],
            unchanged: 5,
        };
        assert_eq!(cs.total_changes(), 4);
        assert!(!cs.is_empty());
    }

    #[test]
    fn test_update_manifest_modified_preserves_doc_id() {
        let dir = test_dir("update_mod_docid");
        let file = write_file(&dir, "doc.md", "original content");
        let key = file.to_string_lossy().to_string();
        let original_doc_id = Uuid::new_v4();

        let mut manifest = Manifest::default();
        manifest.entries.insert(
            key.clone(),
            ManifestEntry {
                hash: "oldhash".to_string(),
                mtime: 0,
                doc_id: original_doc_id,
            },
        );

        // Rewrite file
        std::fs::write(&file, "modified content").unwrap();

        let changeset = ChangeSet {
            modified: vec![file.clone()],
            ..Default::default()
        };

        update_manifest(&mut manifest, &changeset, &HashMap::new()).unwrap();

        // Doc ID should be preserved for modified files
        assert_eq!(manifest.entries[&key].doc_id, original_doc_id);
        assert_ne!(manifest.entries[&key].hash, "oldhash");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_file_hash_empty_file() {
        let dir = test_dir("hash_empty");
        let file = write_file(&dir, "empty.txt", "");
        let h1 = file_hash(&file).unwrap();
        let h2 = file_hash(&file).unwrap();
        assert_eq!(h1, h2);
        // SHA-256 of empty input is the well-known hash
        assert_eq!(h1, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_manifest_save_creates_parent_dirs() {
        let dir = test_dir("manifest_parents");
        let deep_path = dir.join("a/b/c/manifest.json");
        let manifest = Manifest::default();
        manifest.save(&deep_path).unwrap();
        assert!(deep_path.exists());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_detect_changes_mixed() {
        // Test all 4 states simultaneously: added + modified + deleted + unchanged
        let dir = test_dir("detect_mixed");
        let unchanged_file = write_file(&dir, "unchanged.md", "same content");
        let modified_file = write_file(&dir, "modified.md", "old content");
        let added_file = write_file(&dir, "added.md", "new file");

        let unchanged_hash = file_hash(&unchanged_file).unwrap();
        let unchanged_mtime = file_mtime(&unchanged_file).unwrap();

        let mut manifest = Manifest::default();
        // unchanged: same hash + mtime
        manifest.entries.insert(
            unchanged_file.to_string_lossy().to_string(),
            ManifestEntry { hash: unchanged_hash, mtime: unchanged_mtime, doc_id: Uuid::new_v4() },
        );
        // modified: different mtime (triggers hash check)
        manifest.entries.insert(
            modified_file.to_string_lossy().to_string(),
            ManifestEntry { hash: "stale_hash".to_string(), mtime: 0, doc_id: Uuid::new_v4() },
        );
        // deleted: in manifest but not in paths
        manifest.entries.insert(
            "/tmp/cwc_incr_detect_mixed/deleted.md".to_string(),
            ManifestEntry { hash: "x".to_string(), mtime: 0, doc_id: Uuid::new_v4() },
        );
        // added: in paths but not in manifest (added_file)

        let changes = detect_changes(&manifest, &[unchanged_file, modified_file, added_file]).unwrap();
        assert_eq!(changes.unchanged, 1);
        assert_eq!(changes.modified.len(), 1);
        assert_eq!(changes.added.len(), 1);
        assert_eq!(changes.deleted.len(), 1);
        assert_eq!(changes.total_changes(), 3);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_detect_changes_mtime_drift_same_content() {
        // File mtime changed but content is identical → should be counted as unchanged
        let dir = test_dir("detect_mtime_drift");
        let file = write_file(&dir, "doc.md", "same content");
        let hash = file_hash(&file).unwrap();

        let mut manifest = Manifest::default();
        manifest.entries.insert(
            file.to_string_lossy().to_string(),
            ManifestEntry {
                hash,
                mtime: 0, // different mtime from actual
                doc_id: Uuid::new_v4(),
            },
        );

        let changes = detect_changes(&manifest, &[file]).unwrap();
        assert_eq!(changes.unchanged, 1, "same content should be unchanged despite mtime drift");
        assert!(changes.modified.is_empty());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_file_hash_error_nonexistent() {
        let result = file_hash(Path::new("/nonexistent/file.txt"));
        assert!(result.is_err());
    }
}
