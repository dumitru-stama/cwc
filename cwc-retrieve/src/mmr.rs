use std::collections::HashMap;
use uuid::Uuid;

use cwc_core::types::RetrievalHit;

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Select top_k items balancing relevance and diversity using MMR.
///
/// MMR(d) = lambda * sim(d, query) - (1-lambda) * max_sim(d, selected)
///
/// - `hits`: candidate retrieval results (already fused/scored)
/// - `query_embedding`: the query vector
/// - `chunk_embeddings`: map from chunk_id to its embedding vector
/// - `lambda`: trade-off (0.0 = max diversity, 1.0 = max relevance)
/// - `top_k`: number of items to select
///
/// Chunks without an embedding in `chunk_embeddings` are skipped.
pub fn mmr_select(
    hits: &[RetrievalHit],
    query_embedding: &[f32],
    chunk_embeddings: &HashMap<Uuid, Vec<f32>>,
    lambda: f32,
    top_k: usize,
) -> Vec<RetrievalHit> {
    if hits.is_empty() || top_k == 0 {
        return vec![];
    }

    // Filter to only hits that have embeddings
    let candidates: Vec<&RetrievalHit> = hits
        .iter()
        .filter(|h| chunk_embeddings.contains_key(&h.chunk.chunk_id))
        .collect();

    if candidates.is_empty() {
        return vec![];
    }

    let mut selected: Vec<RetrievalHit> = Vec::with_capacity(top_k);
    let mut selected_embeddings: Vec<Vec<f32>> = Vec::with_capacity(top_k);
    let mut remaining: Vec<&RetrievalHit> = candidates;

    while selected.len() < top_k && !remaining.is_empty() {
        let mut best_idx = 0;
        let mut best_mmr = f32::NEG_INFINITY;

        for (i, hit) in remaining.iter().enumerate() {
            let emb = &chunk_embeddings[&hit.chunk.chunk_id];
            let relevance = cosine_similarity(emb, query_embedding);

            let max_sim_to_selected = if selected_embeddings.is_empty() {
                0.0
            } else {
                selected_embeddings
                    .iter()
                    .map(|s| cosine_similarity(emb, s))
                    .fold(f32::NEG_INFINITY, f32::max)
            };

            let mmr = lambda * relevance - (1.0 - lambda) * max_sim_to_selected;

            if mmr > best_mmr {
                best_mmr = mmr;
                best_idx = i;
            }
        }

        let chosen = remaining.remove(best_idx);
        let emb = chunk_embeddings[&chosen.chunk.chunk_id].clone();
        selected.push(chosen.clone());
        selected_embeddings.push(emb);
    }

    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    use cwc_core::types::Chunk;
    use std::collections::HashMap as StdHashMap;

    fn make_hit_with_id(id: Uuid, text: &str, fused: f32) -> RetrievalHit {
        RetrievalHit {
            chunk: Chunk {
                chunk_id: id,
                doc_id: Uuid::new_v4(),
                doc_version: 1,
                source_path: "test.md".to_string(),
                section_path: vec![],
                char_offset: 0,
                char_len: text.len(),
                token_count: 5,
                text: text.to_string(),
                metadata: StdHashMap::new(),
            },
            score_sparse: 0.0,
            score_dense: 0.0,
            score_fused: fused,
            score_rerank: 0.0,
        }
    }

    #[test]
    fn test_mmr_lambda_one_pure_relevance() {
        // lambda=1.0 → diversity term is 0, so it's purely relevance-based
        let id_a = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"a");
        let id_b = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"b");
        let id_c = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"c");

        let query = vec![1.0, 0.0, 0.0];
        // A is most similar to query, B moderately, C least
        let mut embeddings = HashMap::new();
        embeddings.insert(id_a, vec![0.9, 0.1, 0.0]);
        embeddings.insert(id_b, vec![0.5, 0.5, 0.0]);
        embeddings.insert(id_c, vec![0.0, 0.0, 1.0]);

        let hits = vec![
            make_hit_with_id(id_a, "chunk a", 0.9),
            make_hit_with_id(id_b, "chunk b", 0.5),
            make_hit_with_id(id_c, "chunk c", 0.1),
        ];

        let selected = mmr_select(&hits, &query, &embeddings, 1.0, 3);
        assert_eq!(selected.len(), 3);
        // With lambda=1.0, order should follow cosine similarity to query
        assert_eq!(selected[0].chunk.chunk_id, id_a);
        assert_eq!(selected[1].chunk.chunk_id, id_b);
        assert_eq!(selected[2].chunk.chunk_id, id_c);
    }

    #[test]
    fn test_mmr_lambda_zero_max_diversity() {
        // lambda=0.0 → picks most diverse items
        let id_a = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"a");
        let id_b = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"b");
        let id_c = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"c");

        let query = vec![1.0, 0.0, 0.0];
        // A and B are very similar; C is orthogonal
        let mut embeddings = HashMap::new();
        embeddings.insert(id_a, vec![1.0, 0.0, 0.0]);
        embeddings.insert(id_b, vec![0.99, 0.01, 0.0]);
        embeddings.insert(id_c, vec![0.0, 0.0, 1.0]);

        let hits = vec![
            make_hit_with_id(id_a, "chunk a", 0.9),
            make_hit_with_id(id_b, "chunk b", 0.8),
            make_hit_with_id(id_c, "chunk c", 0.5),
        ];

        let selected = mmr_select(&hits, &query, &embeddings, 0.0, 2);
        assert_eq!(selected.len(), 2);
        // First pick: least max-sim-to-selected (all start at 0), but
        // with lambda=0, MMR = -max_sim_selected. First pick is arbitrary
        // (all have max_sim=0 initially). After first pick, the most diverse
        // from the selected should come next.
        // The two selected should include C (the orthogonal one)
        let ids: Vec<Uuid> = selected.iter().map(|h| h.chunk.chunk_id).collect();
        assert!(ids.contains(&id_c), "diverse item C should be selected");
    }

    #[test]
    fn test_mmr_redundant_chunks_deduplication() {
        // Near-duplicate chunks: MMR should prefer diverse ones over near-dups
        let id_a = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"a");
        let id_b = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"b"); // near-dup of A
        let id_c = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"c"); // different topic

        let query = vec![0.9, 0.1, 0.0];
        let mut embeddings = HashMap::new();
        embeddings.insert(id_a, vec![0.95, 0.05, 0.0]); // very close to query
        embeddings.insert(id_b, vec![0.9, 0.3, 0.0]); // similar direction, near-dup area
        embeddings.insert(id_c, vec![0.1, 0.95, 0.0]); // very different direction

        let hits = vec![
            make_hit_with_id(id_a, "ownership memory safety", 0.9),
            make_hit_with_id(id_b, "ownership memory management", 0.85),
            make_hit_with_id(id_c, "borrowing and references", 0.7),
        ];

        // With lambda=0.5, MMR should pick A first (most relevant),
        // then C (more diverse than B which is near-dup of A)
        let selected = mmr_select(&hits, &query, &embeddings, 0.5, 2);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].chunk.chunk_id, id_a);
        // B is a near-dup of A, so C should be preferred for diversity
        assert_eq!(selected[1].chunk.chunk_id, id_c);
    }

    #[test]
    fn test_mmr_empty_inputs() {
        let query = vec![1.0, 0.0];
        let embeddings = HashMap::new();
        let hits: Vec<RetrievalHit> = vec![];

        let selected = mmr_select(&hits, &query, &embeddings, 0.7, 5);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_mmr_top_k_zero() {
        let id = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"a");
        let query = vec![1.0, 0.0];
        let mut embeddings = HashMap::new();
        embeddings.insert(id, vec![1.0, 0.0]);
        let hits = vec![make_hit_with_id(id, "test", 0.9)];

        let selected = mmr_select(&hits, &query, &embeddings, 0.7, 0);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_mmr_top_k_larger_than_candidates() {
        let id_a = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"a");
        let id_b = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"b");

        let query = vec![1.0, 0.0];
        let mut embeddings = HashMap::new();
        embeddings.insert(id_a, vec![1.0, 0.0]);
        embeddings.insert(id_b, vec![0.0, 1.0]);
        let hits = vec![
            make_hit_with_id(id_a, "a", 0.9),
            make_hit_with_id(id_b, "b", 0.5),
        ];

        let selected = mmr_select(&hits, &query, &embeddings, 0.7, 10);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_mmr_missing_embedding_skipped() {
        let id_a = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"a");
        let id_b = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"b");

        let query = vec![1.0, 0.0];
        let mut embeddings = HashMap::new();
        embeddings.insert(id_a, vec![1.0, 0.0]);
        // id_b has no embedding

        let hits = vec![
            make_hit_with_id(id_a, "a", 0.9),
            make_hit_with_id(id_b, "b", 0.95),
        ];

        let selected = mmr_select(&hits, &query, &embeddings, 0.7, 2);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].chunk.chunk_id, id_a);
    }

    #[test]
    fn test_mmr_single_candidate() {
        let id = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"single");
        let query = vec![1.0, 0.0];
        let mut embeddings = HashMap::new();
        embeddings.insert(id, vec![0.8, 0.2]);
        let hits = vec![make_hit_with_id(id, "single chunk", 0.9)];

        let selected = mmr_select(&hits, &query, &embeddings, 0.7, 1);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].chunk.chunk_id, id);
    }

    #[test]
    fn test_mmr_all_embeddings_missing() {
        let id = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"no-emb");
        let query = vec![1.0, 0.0];
        let embeddings = HashMap::new(); // no embeddings at all
        let hits = vec![make_hit_with_id(id, "chunk", 0.9)];

        let selected = mmr_select(&hits, &query, &embeddings, 0.7, 5);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_cosine_similarity_unit() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]) - 0.0).abs() < 1e-6);
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) - -1.0).abs() < 1e-6);
        assert!((cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]) - 0.0).abs() < 1e-6);
    }
}
