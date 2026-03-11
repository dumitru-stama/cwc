use uuid::Uuid;

/// Aggregated retrieval quality metrics over a set of queries.
#[derive(Debug, Clone, Default)]
pub struct RetrievalMetrics {
    /// Fraction of relevant chunks that appear in the top-k results.
    pub recall_at_k: f32,
    /// Fraction of top-k results that are relevant.
    pub precision_at_k: f32,
    /// Mean Reciprocal Rank: average of 1/rank of the first relevant result.
    pub mrr: f32,
    /// Normalized Discounted Cumulative Gain at k.
    pub ndcg_at_k: f32,
    /// Fraction of queries with at least one relevant result in top-k.
    pub hit_rate: f32,
    /// Number of queries evaluated.
    pub query_count: usize,
}

/// Compute retrieval metrics over (retrieved_ids, ground_truth_ids) pairs.
///
/// `results` is a list of (retrieved chunk IDs in rank order, relevant chunk IDs).
/// `k` is the cutoff for top-k evaluation.
pub fn compute_retrieval_metrics(
    results: &[(Vec<Uuid>, Vec<Uuid>)],
    k: usize,
) -> RetrievalMetrics {
    if results.is_empty() || k == 0 {
        return RetrievalMetrics {
            query_count: results.len(),
            ..Default::default()
        };
    }

    let mut total_recall = 0.0f64;
    let mut total_precision = 0.0f64;
    let mut total_rr = 0.0f64;
    let mut total_ndcg = 0.0f64;
    let mut total_hits = 0usize;

    for (retrieved, relevant) in results {
        let top_k: Vec<&Uuid> = retrieved.iter().take(k).collect();
        let relevant_set: std::collections::HashSet<&Uuid> = relevant.iter().collect();

        if relevant_set.is_empty() {
            // No ground truth — skip this query for recall/precision/mrr
            // but still count for hit_rate (it should be 0)
            continue;
        }

        // Recall@k: how many relevant were retrieved
        let relevant_found = top_k.iter().filter(|id| relevant_set.contains(*id)).count();
        total_recall += relevant_found as f64 / relevant_set.len() as f64;

        // Precision@k: how many retrieved are relevant
        total_precision += relevant_found as f64 / top_k.len().max(1) as f64;

        // MRR: 1/rank of first relevant
        let first_relevant_rank = top_k
            .iter()
            .position(|id| relevant_set.contains(id))
            .map(|pos| pos + 1);
        if let Some(rank) = first_relevant_rank {
            total_rr += 1.0 / rank as f64;
            total_hits += 1;
        }

        // nDCG@k
        total_ndcg += ndcg(&top_k, &relevant_set, k);
    }

    // Count queries that have ground truth for averaging
    let queries_with_gt = results
        .iter()
        .filter(|(_, rel)| !rel.is_empty())
        .count();
    let n = queries_with_gt.max(1) as f64;

    RetrievalMetrics {
        recall_at_k: (total_recall / n) as f32,
        precision_at_k: (total_precision / n) as f32,
        mrr: (total_rr / n) as f32,
        ndcg_at_k: (total_ndcg / n) as f32,
        hit_rate: total_hits as f32 / queries_with_gt.max(1) as f32,
        query_count: results.len(),
    }
}

/// Compute nDCG for a single query.
fn ndcg(
    retrieved: &[&Uuid],
    relevant: &std::collections::HashSet<&Uuid>,
    k: usize,
) -> f64 {
    // DCG: sum of 1/log2(rank+1) for relevant items
    let dcg: f64 = retrieved
        .iter()
        .take(k)
        .enumerate()
        .filter(|(_, id)| relevant.contains(*id))
        .map(|(i, _)| 1.0 / (i as f64 + 2.0).log2())
        .sum();

    // Ideal DCG: all relevant items at top positions
    let ideal_count = relevant.len().min(k);
    let idcg: f64 = (0..ideal_count)
        .map(|i| 1.0 / (i as f64 + 2.0).log2())
        .sum();

    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uuid(n: u8) -> Uuid {
        Uuid::from_bytes([n, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    }

    #[test]
    fn test_perfect_recall() {
        let results = vec![(
            vec![uuid(1), uuid(2), uuid(3)],
            vec![uuid(1), uuid(2), uuid(3)],
        )];
        let m = compute_retrieval_metrics(&results, 10);
        assert!((m.recall_at_k - 1.0).abs() < 0.001);
        assert!((m.precision_at_k - 1.0).abs() < 0.001);
        assert!((m.mrr - 1.0).abs() < 0.001);
        assert!((m.ndcg_at_k - 1.0).abs() < 0.001);
        assert!((m.hit_rate - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_no_relevant_results() {
        let results = vec![(
            vec![uuid(1), uuid(2), uuid(3)],
            vec![uuid(4), uuid(5)],
        )];
        let m = compute_retrieval_metrics(&results, 3);
        assert!((m.recall_at_k).abs() < 0.001);
        assert!((m.precision_at_k).abs() < 0.001);
        assert!((m.mrr).abs() < 0.001);
        assert!((m.hit_rate).abs() < 0.001);
    }

    #[test]
    fn test_relevant_at_position_3() {
        let results = vec![(
            vec![uuid(10), uuid(20), uuid(1)],
            vec![uuid(1)],
        )];
        let m = compute_retrieval_metrics(&results, 10);
        assert!((m.recall_at_k - 1.0).abs() < 0.001);
        // MRR = 1/3
        assert!((m.mrr - 1.0 / 3.0).abs() < 0.001);
        // Precision = 1/3
        assert!((m.precision_at_k - 1.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_empty_results() {
        let m = compute_retrieval_metrics(&[], 10);
        assert_eq!(m.query_count, 0);
        assert!((m.recall_at_k).abs() < 0.001);
    }

    #[test]
    fn test_k_zero() {
        let results = vec![(vec![uuid(1)], vec![uuid(1)])];
        let m = compute_retrieval_metrics(&results, 0);
        assert!((m.recall_at_k).abs() < 0.001);
    }

    #[test]
    fn test_multiple_queries_averaging() {
        let results = vec![
            // Query 1: perfect recall
            (vec![uuid(1), uuid(2)], vec![uuid(1), uuid(2)]),
            // Query 2: no relevant found
            (vec![uuid(3), uuid(4)], vec![uuid(5)]),
        ];
        let m = compute_retrieval_metrics(&results, 10);
        // Recall: (1.0 + 0.0) / 2 = 0.5
        assert!((m.recall_at_k - 0.5).abs() < 0.001);
        // Hit rate: 1 out of 2 = 0.5
        assert!((m.hit_rate - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_partial_recall() {
        // 2 of 3 relevant found
        let results = vec![(
            vec![uuid(1), uuid(2), uuid(10)],
            vec![uuid(1), uuid(2), uuid(3)],
        )];
        let m = compute_retrieval_metrics(&results, 3);
        assert!((m.recall_at_k - 2.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_ndcg_ideal_order() {
        // All relevant at top positions → nDCG = 1.0
        let results = vec![(
            vec![uuid(1), uuid(2), uuid(3), uuid(10)],
            vec![uuid(1), uuid(2), uuid(3)],
        )];
        let m = compute_retrieval_metrics(&results, 10);
        assert!((m.ndcg_at_k - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_ndcg_worst_order() {
        // Single relevant item at last position (k=3) vs ideal (position 1)
        let results = vec![(
            vec![uuid(10), uuid(20), uuid(1)],
            vec![uuid(1)],
        )];
        let m = compute_retrieval_metrics(&results, 3);
        // DCG = 1/log2(4) = 0.5, IDCG = 1/log2(2) = 1.0, nDCG = 0.5
        assert!((m.ndcg_at_k - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_empty_ground_truth_skipped() {
        // Queries with no ground truth should not affect recall/precision
        let results = vec![
            (vec![uuid(1)], vec![]), // no ground truth
            (vec![uuid(1)], vec![uuid(1)]), // perfect
        ];
        let m = compute_retrieval_metrics(&results, 10);
        // Only 1 query has ground truth, so recall = 1.0
        assert!((m.recall_at_k - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_hit_rate_ignores_empty_ground_truth() {
        // 1 query with empty ground truth + 1 query with a hit
        // hit_rate should be 1.0 (1 hit out of 1 evaluable query)
        let results = vec![
            (vec![uuid(1)], vec![]),       // no ground truth — skipped
            (vec![uuid(1)], vec![uuid(1)]), // hit
        ];
        let m = compute_retrieval_metrics(&results, 10);
        assert!((m.hit_rate - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_all_empty_ground_truth() {
        let results = vec![
            (vec![uuid(1)], vec![]),
            (vec![uuid(2)], vec![]),
        ];
        let m = compute_retrieval_metrics(&results, 10);
        assert_eq!(m.query_count, 2);
        // All ground truth empty: metrics should be 0 (no data to evaluate)
        assert!((m.recall_at_k).abs() < 0.001);
        assert!((m.hit_rate).abs() < 0.001);
    }

    #[test]
    fn test_duplicate_relevant_ids() {
        // If ground truth has duplicates, HashSet deduplication means we
        // check against unique IDs only
        let results = vec![(
            vec![uuid(1), uuid(2)],
            vec![uuid(1), uuid(1), uuid(1)], // 3 copies of same ID
        )];
        let m = compute_retrieval_metrics(&results, 10);
        // Only 1 unique relevant ID, and it's retrieved → recall = 1.0
        assert!((m.recall_at_k - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_precision_with_k_larger_than_retrieved() {
        // k=10 but only 2 retrieved, 1 relevant
        let results = vec![(
            vec![uuid(1), uuid(2)],
            vec![uuid(1)],
        )];
        let m = compute_retrieval_metrics(&results, 10);
        // Precision = 1/2 (1 relevant out of 2 retrieved, not 1/10)
        assert!((m.precision_at_k - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_k_smaller_than_retrieved() {
        let results = vec![(
            vec![uuid(10), uuid(20), uuid(30), uuid(1)],
            vec![uuid(1)],
        )];
        // k=2: only first 2 items considered, uuid(1) not in top-2
        let m = compute_retrieval_metrics(&results, 2);
        assert!((m.recall_at_k).abs() < 0.001);
        assert!((m.mrr).abs() < 0.001);
    }
}
