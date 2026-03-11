use cwc_core::types::RetrievalHit;
use uuid::Uuid;

/// Assign sequential source IDs [S1], [S2], ... based on final order.
/// Returns mapping from chunk_id -> source_id for citation tracking.
pub fn assign_source_ids(chunks: &[RetrievalHit]) -> Vec<(Uuid, String)> {
    chunks
        .iter()
        .enumerate()
        .map(|(i, hit)| (hit.chunk.chunk_id, format!("[S{}]", i + 1)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_hit;

    #[test]
    fn test_source_ids_sequential() {
        let chunks = vec![
            make_hit(0, 10, 0.9),
            make_hit(1, 10, 0.8),
            make_hit(2, 10, 0.7),
        ];
        let ids = assign_source_ids(&chunks);
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0].1, "[S1]");
        assert_eq!(ids[1].1, "[S2]");
        assert_eq!(ids[2].1, "[S3]");
    }

    #[test]
    fn test_source_ids_match_chunk_ids() {
        let chunks = vec![
            make_hit(0, 10, 0.9),
            make_hit(1, 10, 0.8),
        ];
        let ids = assign_source_ids(&chunks);
        assert_eq!(ids[0].0, chunks[0].chunk.chunk_id);
        assert_eq!(ids[1].0, chunks[1].chunk.chunk_id);
    }

    #[test]
    fn test_source_ids_empty() {
        let ids = assign_source_ids(&[]);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_source_ids_single() {
        let chunks = vec![make_hit(0, 10, 0.9)];
        let ids = assign_source_ids(&chunks);
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0].1, "[S1]");
    }
}
