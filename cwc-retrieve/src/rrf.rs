use std::collections::HashMap;
use uuid::Uuid;

use cwc_core::types::RetrievalHit;

/// Fuse multiple ranked lists using Reciprocal Rank Fusion.
///
/// score(d) = sum over lists: 1 / (k + rank(d))
/// where k is a constant (default 60) and rank is 1-based.
///
/// Merges by `chunk_id`. The best individual scores (score_sparse, score_dense)
/// are preserved from the input lists. `score_fused` is set to the RRF score.
pub fn reciprocal_rank_fusion(lists: &[Vec<RetrievalHit>], k: f32) -> Vec<RetrievalHit> {
    // Map chunk_id → (accumulated RRF score, best hit)
    let mut merged: HashMap<Uuid, (f32, RetrievalHit)> = HashMap::new();

    for list in lists {
        for (rank_0, hit) in list.iter().enumerate() {
            let rank = (rank_0 + 1) as f32; // 1-based
            let rrf_score = 1.0 / (k + rank);

            merged
                .entry(hit.chunk.chunk_id)
                .and_modify(|(acc_score, best_hit)| {
                    *acc_score += rrf_score;
                    // Preserve the best individual scores
                    if hit.score_sparse > best_hit.score_sparse {
                        best_hit.score_sparse = hit.score_sparse;
                    }
                    if hit.score_dense > best_hit.score_dense {
                        best_hit.score_dense = hit.score_dense;
                    }
                    if hit.score_rerank > best_hit.score_rerank {
                        best_hit.score_rerank = hit.score_rerank;
                    }
                })
                .or_insert_with(|| (rrf_score, hit.clone()));
        }
    }

    let mut results: Vec<RetrievalHit> = merged
        .into_values()
        .map(|(rrf_score, mut hit)| {
            hit.score_fused = rrf_score;
            hit
        })
        .collect();

    // Sort by fused score descending
    results.sort_by(|a, b| {
        b.score_fused
            .partial_cmp(&a.score_fused)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use cwc_core::types::Chunk;
    use std::collections::HashMap as StdHashMap;

    fn make_hit_with_id(id: Uuid, sparse: f32, dense: f32) -> RetrievalHit {
        RetrievalHit {
            chunk: Chunk {
                chunk_id: id,
                doc_id: Uuid::new_v4(),
                doc_version: 1,
                source_path: "test.md".to_string(),
                section_path: vec![],
                char_offset: 0,
                char_len: 5,
                token_count: 3,
                text: format!("chunk {id}"),
                metadata: StdHashMap::new(),
            },
            score_sparse: sparse,
            score_dense: dense,
            score_fused: 0.0,
            score_rerank: 0.0,
        }
    }

    #[test]
    fn test_rrf_overlapping_items_sum_correctly() {
        let id_a = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"a");
        let id_b = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"b");
        let id_c = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"c");

        let k = 60.0;
        // List 1: A rank 1, B rank 2
        let list1 = vec![
            make_hit_with_id(id_a, 5.0, 0.0),
            make_hit_with_id(id_b, 3.0, 0.0),
        ];
        // List 2: B rank 1, C rank 2
        let list2 = vec![
            make_hit_with_id(id_b, 0.0, 0.9),
            make_hit_with_id(id_c, 0.0, 0.7),
        ];

        let results = reciprocal_rank_fusion(&[list1, list2], k);

        // Expected RRF scores:
        // A: 1/(60+1) = 0.01639
        // B: 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252
        // C: 1/(60+2) = 0.01613
        let b_hit = results.iter().find(|h| h.chunk.chunk_id == id_b).unwrap();
        let a_hit = results.iter().find(|h| h.chunk.chunk_id == id_a).unwrap();
        let c_hit = results.iter().find(|h| h.chunk.chunk_id == id_c).unwrap();

        let expected_b = 1.0 / (k + 2.0) + 1.0 / (k + 1.0);
        let expected_a = 1.0 / (k + 1.0);
        let expected_c = 1.0 / (k + 2.0);

        assert!((b_hit.score_fused - expected_b).abs() < 1e-6);
        assert!((a_hit.score_fused - expected_a).abs() < 1e-6);
        assert!((c_hit.score_fused - expected_c).abs() < 1e-6);
    }

    #[test]
    fn test_rrf_item_in_both_ranked_higher() {
        let id_both = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"both");
        let id_only = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"only");

        let k = 60.0;
        // "both" appears in both lists, "only" in one
        let list1 = vec![
            make_hit_with_id(id_only, 10.0, 0.0),
            make_hit_with_id(id_both, 5.0, 0.0),
        ];
        let list2 = vec![make_hit_with_id(id_both, 0.0, 0.9)];

        let results = reciprocal_rank_fusion(&[list1, list2], k);

        // "both" gets score from both lists; "only" from one
        let both_hit = results.iter().find(|h| h.chunk.chunk_id == id_both).unwrap();
        let only_hit = results.iter().find(|h| h.chunk.chunk_id == id_only).unwrap();
        assert!(both_hit.score_fused > only_hit.score_fused);
    }

    #[test]
    fn test_rrf_k_parameter_affects_distribution() {
        let id_a = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"a");
        let id_b = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"b");

        let list_k1 = vec![
            make_hit_with_id(id_a, 1.0, 0.0),
            make_hit_with_id(id_b, 0.5, 0.0),
        ];
        let list_k60 = vec![
            make_hit_with_id(id_a, 1.0, 0.0),
            make_hit_with_id(id_b, 0.5, 0.0),
        ];

        // With k=1: scores are 1/2 and 1/3, ratio = 1.5
        let results_k1 = reciprocal_rank_fusion(&[list_k1], 1.0);
        let ratio_k1 = results_k1[0].score_fused / results_k1[1].score_fused;

        // With k=60: scores are 1/61 and 1/62, ratio ≈ 1.016
        let results_k60 = reciprocal_rank_fusion(&[list_k60], 60.0);
        let ratio_k60 = results_k60[0].score_fused / results_k60[1].score_fused;

        // Higher k makes scores more uniform (lower ratio)
        assert!(ratio_k1 > ratio_k60);
    }

    #[test]
    fn test_rrf_preserves_best_individual_scores() {
        let id = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"shared");

        let list1 = vec![make_hit_with_id(id, 8.0, 0.0)];
        let list2 = vec![make_hit_with_id(id, 0.0, 0.95)];

        let results = reciprocal_rank_fusion(&[list1, list2], 60.0);
        assert_eq!(results.len(), 1);
        assert!((results[0].score_sparse - 8.0).abs() < f32::EPSILON);
        assert!((results[0].score_dense - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rrf_empty_lists() {
        let results = reciprocal_rank_fusion(&[], 60.0);
        assert!(results.is_empty());

        let results = reciprocal_rank_fusion(&[vec![], vec![]], 60.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_rrf_single_list() {
        let id = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"single");
        let list = vec![make_hit_with_id(id, 5.0, 0.0)];
        let results = reciprocal_rank_fusion(&[list], 60.0);
        assert_eq!(results.len(), 1);
        let expected = 1.0 / (60.0 + 1.0);
        assert!((results[0].score_fused - expected).abs() < 1e-6);
    }

    #[test]
    fn test_rrf_three_lists() {
        let id_a = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"a");
        let id_b = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"b");

        let k = 60.0;
        let list1 = vec![make_hit_with_id(id_a, 5.0, 0.0)];
        let list2 = vec![make_hit_with_id(id_a, 0.0, 0.9)];
        let list3 = vec![
            make_hit_with_id(id_b, 0.0, 0.0),
            make_hit_with_id(id_a, 0.0, 0.0),
        ];

        let results = reciprocal_rank_fusion(&[list1, list2, list3], k);
        // A appears in all 3 lists: ranks 1, 1, 2
        let a_hit = results.iter().find(|h| h.chunk.chunk_id == id_a).unwrap();
        let expected = 1.0 / (k + 1.0) + 1.0 / (k + 1.0) + 1.0 / (k + 2.0);
        assert!((a_hit.score_fused - expected).abs() < 1e-6);

        // B appears in 1 list at rank 1
        let b_hit = results.iter().find(|h| h.chunk.chunk_id == id_b).unwrap();
        let expected_b = 1.0 / (k + 1.0);
        assert!((b_hit.score_fused - expected_b).abs() < 1e-6);

        // A should rank higher than B
        assert!(a_hit.score_fused > b_hit.score_fused);
    }

    #[test]
    fn test_rrf_sorted_descending() {
        let ids: Vec<Uuid> = (0..5)
            .map(|i| Uuid::new_v5(&Uuid::NAMESPACE_OID, format!("id{i}").as_bytes()))
            .collect();

        // Items at different ranks
        let list: Vec<RetrievalHit> = ids
            .iter()
            .map(|id| make_hit_with_id(*id, 1.0, 0.0))
            .collect();

        let results = reciprocal_rank_fusion(&[list], 60.0);
        for w in results.windows(2) {
            assert!(
                w[0].score_fused >= w[1].score_fused,
                "results should be sorted descending"
            );
        }
    }

    #[test]
    fn test_rrf_empty_sparse_valid_dense() {
        let id = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"dense-only");
        let sparse: Vec<RetrievalHit> = vec![];
        let dense = vec![make_hit_with_id(id, 0.0, 0.85)];

        let results = reciprocal_rank_fusion(&[sparse, dense], 60.0);
        assert_eq!(results.len(), 1);
        assert!((results[0].score_dense - 0.85).abs() < f32::EPSILON);
        assert!(results[0].score_fused > 0.0);
    }
}
