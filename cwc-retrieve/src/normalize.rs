use cwc_core::types::RetrievalHit;

/// Which score field to normalize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreField {
    Sparse,
    Dense,
    Fused,
    Rerank,
}

fn get_score(hit: &RetrievalHit, field: ScoreField) -> f32 {
    match field {
        ScoreField::Sparse => hit.score_sparse,
        ScoreField::Dense => hit.score_dense,
        ScoreField::Fused => hit.score_fused,
        ScoreField::Rerank => hit.score_rerank,
    }
}

fn set_score(hit: &mut RetrievalHit, field: ScoreField, value: f32) {
    match field {
        ScoreField::Sparse => hit.score_sparse = value,
        ScoreField::Dense => hit.score_dense = value,
        ScoreField::Fused => hit.score_fused = value,
        ScoreField::Rerank => hit.score_rerank = value,
    }
}

/// Min-max normalization of a score field to [0, 1].
/// Preserves relative order. If all scores are equal, sets all to 1.0.
pub fn min_max_normalize(hits: &mut [RetrievalHit], field: ScoreField) {
    if hits.is_empty() {
        return;
    }

    let min = hits
        .iter()
        .map(|h| get_score(h, field))
        .fold(f32::INFINITY, f32::min);
    let max = hits
        .iter()
        .map(|h| get_score(h, field))
        .fold(f32::NEG_INFINITY, f32::max);

    let range = max - min;
    if range < f32::EPSILON {
        // All scores identical — set to 1.0
        for hit in hits.iter_mut() {
            set_score(hit, field, 1.0);
        }
        return;
    }

    for hit in hits.iter_mut() {
        let raw = get_score(hit, field);
        set_score(hit, field, (raw - min) / range);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cwc_core::types::Chunk;
    use std::collections::HashMap;
    use uuid::Uuid;

    fn make_hit(sparse: f32, dense: f32) -> RetrievalHit {
        RetrievalHit {
            chunk: Chunk {
                chunk_id: Uuid::new_v4(),
                doc_id: Uuid::new_v4(),
                doc_version: 1,
                source_path: "test.md".to_string(),
                section_path: vec![],
                char_offset: 0,
                char_len: 5,
                token_count: 3,
                text: "test".to_string(),
                metadata: HashMap::new(),
            },
            score_sparse: sparse,
            score_dense: dense,
            score_fused: 0.0,
            score_rerank: 0.0,
        }
    }

    #[test]
    fn test_normalize_output_in_range() {
        let mut hits = vec![
            make_hit(10.0, 0.0),
            make_hit(5.0, 0.0),
            make_hit(1.0, 0.0),
            make_hit(20.0, 0.0),
        ];
        min_max_normalize(&mut hits, ScoreField::Sparse);
        for hit in &hits {
            assert!(hit.score_sparse >= 0.0);
            assert!(hit.score_sparse <= 1.0);
        }
        // Max should be 1.0, min should be 0.0
        assert!((hits[3].score_sparse - 1.0).abs() < f32::EPSILON); // was 20.0
        assert!((hits[2].score_sparse - 0.0).abs() < f32::EPSILON); // was 1.0
    }

    #[test]
    fn test_normalize_preserves_relative_order() {
        let mut hits = vec![
            make_hit(15.0, 0.0),
            make_hit(5.0, 0.0),
            make_hit(25.0, 0.0),
        ];
        min_max_normalize(&mut hits, ScoreField::Sparse);
        assert!(hits[2].score_sparse > hits[0].score_sparse);
        assert!(hits[0].score_sparse > hits[1].score_sparse);
    }

    #[test]
    fn test_normalize_all_equal() {
        let mut hits = vec![
            make_hit(5.0, 0.0),
            make_hit(5.0, 0.0),
            make_hit(5.0, 0.0),
        ];
        min_max_normalize(&mut hits, ScoreField::Sparse);
        for hit in &hits {
            assert!((hit.score_sparse - 1.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_normalize_empty() {
        let mut hits: Vec<RetrievalHit> = vec![];
        min_max_normalize(&mut hits, ScoreField::Sparse);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_normalize_single_element() {
        let mut hits = vec![make_hit(7.0, 0.0)];
        min_max_normalize(&mut hits, ScoreField::Sparse);
        assert!((hits[0].score_sparse - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_normalize_negative_scores() {
        let mut hits = vec![
            make_hit(-5.0, 0.0),
            make_hit(0.0, 0.0),
            make_hit(5.0, 0.0),
        ];
        min_max_normalize(&mut hits, ScoreField::Sparse);
        assert!((hits[0].score_sparse - 0.0).abs() < f32::EPSILON);
        assert!((hits[1].score_sparse - 0.5).abs() < f32::EPSILON);
        assert!((hits[2].score_sparse - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_normalize_two_elements() {
        let mut hits = vec![make_hit(3.0, 0.0), make_hit(7.0, 0.0)];
        min_max_normalize(&mut hits, ScoreField::Sparse);
        assert!((hits[0].score_sparse - 0.0).abs() < f32::EPSILON);
        assert!((hits[1].score_sparse - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_normalize_does_not_affect_other_fields() {
        let mut hits = vec![make_hit(10.0, 0.5), make_hit(20.0, 0.9)];
        min_max_normalize(&mut hits, ScoreField::Sparse);
        // Dense should be untouched
        assert!((hits[0].score_dense - 0.5).abs() < f32::EPSILON);
        assert!((hits[1].score_dense - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_normalize_dense_field() {
        let mut hits = vec![
            make_hit(0.0, 0.2),
            make_hit(0.0, 0.8),
            make_hit(0.0, 0.5),
        ];
        min_max_normalize(&mut hits, ScoreField::Dense);
        assert!((hits[0].score_dense - 0.0).abs() < f32::EPSILON);
        assert!((hits[1].score_dense - 1.0).abs() < f32::EPSILON);
        assert!((hits[2].score_dense - 0.5).abs() < f32::EPSILON);
    }
}
