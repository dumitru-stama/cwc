use cwc_core::types::RetrievalHit;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReorderStrategy {
    /// Keep retrieval ranking order (baseline).
    RankOrder,
    /// Best evidence at start and end, weaker in middle.
    #[default]
    EdgePlacement,
    /// Reverse: put strongest at end (recency bias models).
    StrongestLast,
}

/// Reorder selected chunks to optimize for positional effects.
///
/// **EdgePlacement algorithm:**
/// 1. Sort by score (descending)
/// 2. Interleave: best -> start, second best -> end, third -> start, fourth -> end, ...
/// 3. Result: strongest evidence at edges, weakest in the middle
pub fn reorder_chunks(mut chunks: Vec<RetrievalHit>, strategy: ReorderStrategy) -> Vec<RetrievalHit> {
    match strategy {
        ReorderStrategy::RankOrder => chunks,
        ReorderStrategy::StrongestLast => {
            chunks.sort_by(|a, b| a.score_fused.partial_cmp(&b.score_fused).unwrap_or(std::cmp::Ordering::Equal));
            chunks
        }
        ReorderStrategy::EdgePlacement => edge_placement(chunks),
    }
}

fn edge_placement(mut chunks: Vec<RetrievalHit>) -> Vec<RetrievalHit> {
    if chunks.len() <= 2 {
        return chunks;
    }

    // Sort descending by fused score
    chunks.sort_by(|a, b| {
        b.score_fused
            .partial_cmp(&a.score_fused)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n = chunks.len();
    let mut result = vec![None; n];
    let mut left = 0;
    let mut right = n - 1;

    for (i, chunk) in chunks.into_iter().enumerate() {
        if i % 2 == 0 {
            result[left] = Some(chunk);
            left += 1;
        } else {
            result[right] = Some(chunk);
            right = right.saturating_sub(1);
        }
    }

    result.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_hit;

    #[test]
    fn test_rank_order_preserves_input() {
        let chunks = vec![
            make_hit(0, 10, 0.5),
            make_hit(1, 10, 0.9),
            make_hit(2, 10, 0.3),
        ];
        let result = reorder_chunks(chunks.clone(), ReorderStrategy::RankOrder);
        assert_eq!(result.len(), 3);
        for (a, b) in result.iter().zip(chunks.iter()) {
            assert_eq!(a.chunk.chunk_id, b.chunk.chunk_id);
        }
    }

    #[test]
    fn test_edge_placement_strongest_at_edges() {
        let chunks = vec![
            make_hit(0, 10, 0.9),
            make_hit(1, 10, 0.8),
            make_hit(2, 10, 0.7),
            make_hit(3, 10, 0.6),
            make_hit(4, 10, 0.5),
        ];
        let result = reorder_chunks(chunks, ReorderStrategy::EdgePlacement);
        assert_eq!(result.len(), 5);

        // First and last should have the highest scores
        let first_score = result.first().unwrap().score_fused;
        let last_score = result.last().unwrap().score_fused;
        let middle_scores: Vec<f32> = result[1..result.len() - 1]
            .iter()
            .map(|h| h.score_fused)
            .collect();

        assert!(first_score >= *middle_scores.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap());
        assert!(last_score >= *middle_scores.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap());
    }

    #[test]
    fn test_edge_placement_middle_has_lowest_scores() {
        let chunks = vec![
            make_hit(0, 10, 0.9),
            make_hit(1, 10, 0.8),
            make_hit(2, 10, 0.7),
            make_hit(3, 10, 0.6),
            make_hit(4, 10, 0.5),
        ];
        let result = reorder_chunks(chunks, ReorderStrategy::EdgePlacement);

        // The very middle element should have the lowest score
        let mid = result.len() / 2;
        let middle_score = result[mid].score_fused;
        let edge_min = result.first().unwrap().score_fused.min(result.last().unwrap().score_fused);
        assert!(middle_score <= edge_min);
    }

    #[test]
    fn test_edge_placement_odd_count() {
        let chunks = vec![
            make_hit(0, 10, 0.9),
            make_hit(1, 10, 0.7),
            make_hit(2, 10, 0.5),
        ];
        let result = reorder_chunks(chunks, ReorderStrategy::EdgePlacement);
        assert_eq!(result.len(), 3);
        // Best (0.9) at start, second best (0.7) at end, weakest (0.5) in middle
        assert_eq!(result[0].score_fused, 0.9);
        assert_eq!(result[2].score_fused, 0.7);
        assert_eq!(result[1].score_fused, 0.5);
    }

    #[test]
    fn test_edge_placement_single_chunk_noop() {
        let chunks = vec![make_hit(0, 10, 0.9)];
        let result = reorder_chunks(chunks, ReorderStrategy::EdgePlacement);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].score_fused, 0.9);
    }

    #[test]
    fn test_edge_placement_two_chunks() {
        let chunks = vec![make_hit(0, 10, 0.9), make_hit(1, 10, 0.5)];
        let result = reorder_chunks(chunks, ReorderStrategy::EdgePlacement);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_strongest_last() {
        let chunks = vec![
            make_hit(0, 10, 0.9),
            make_hit(1, 10, 0.5),
            make_hit(2, 10, 0.7),
        ];
        let result = reorder_chunks(chunks, ReorderStrategy::StrongestLast);
        assert_eq!(result.last().unwrap().score_fused, 0.9);
        assert_eq!(result.first().unwrap().score_fused, 0.5);
    }

    #[test]
    fn test_edge_placement_even_count() {
        let chunks = vec![
            make_hit(0, 10, 0.9),
            make_hit(1, 10, 0.8),
            make_hit(2, 10, 0.7),
            make_hit(3, 10, 0.6),
        ];
        let result = reorder_chunks(chunks, ReorderStrategy::EdgePlacement);
        assert_eq!(result.len(), 4);
        // After sort desc: [0.9, 0.8, 0.7, 0.6]
        // i=0 (0.9) → pos 0, i=1 (0.8) → pos 3, i=2 (0.7) → pos 1, i=3 (0.6) → pos 2
        // Result: [0.9, 0.7, 0.6, 0.8]
        assert_eq!(result[0].score_fused, 0.9);
        assert_eq!(result[3].score_fused, 0.8);
        // Middle elements have lower scores than both edges
        assert!(result[1].score_fused < result[0].score_fused);
        assert!(result[2].score_fused < result[3].score_fused);
    }

    #[test]
    fn test_edge_placement_equal_scores() {
        let chunks = vec![
            make_hit(0, 10, 0.5),
            make_hit(1, 10, 0.5),
            make_hit(2, 10, 0.5),
        ];
        // All equal scores — should not crash, all elements present
        let result = reorder_chunks(chunks, ReorderStrategy::EdgePlacement);
        assert_eq!(result.len(), 3);
        // All scores are 0.5
        for h in &result {
            assert_eq!(h.score_fused, 0.5);
        }
    }

    #[test]
    fn test_reorder_empty_all_strategies() {
        for strategy in [
            ReorderStrategy::RankOrder,
            ReorderStrategy::EdgePlacement,
            ReorderStrategy::StrongestLast,
        ] {
            let result = reorder_chunks(vec![], strategy);
            assert!(result.is_empty());
        }
    }

    #[test]
    fn test_edge_placement_preserves_all_chunks() {
        // Verify no chunks are lost or duplicated
        let chunks: Vec<_> = (0..8)
            .map(|i| make_hit(i, 10, 0.9 - i as f32 * 0.1))
            .collect();
        let ids_before: Vec<_> = chunks.iter().map(|h| h.chunk.chunk_id).collect();
        let result = reorder_chunks(chunks, ReorderStrategy::EdgePlacement);
        let mut ids_after: Vec<_> = result.iter().map(|h| h.chunk.chunk_id).collect();
        let mut ids_sorted = ids_before.clone();
        ids_sorted.sort();
        ids_after.sort();
        assert_eq!(ids_sorted, ids_after);
    }
}
