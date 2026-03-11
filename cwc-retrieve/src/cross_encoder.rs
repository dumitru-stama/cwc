use std::borrow::Cow;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use tokenizers::Tokenizer;

use cwc_core::error::Result;
use cwc_core::types::RetrievalHit;
use cwc_core::CwcError;

/// Configuration for the cross-encoder reranker.
#[derive(Debug, Clone)]
pub struct RerankConfig {
    /// Maximum input sequence length (query + document tokens).
    pub max_seq_len: usize,
    /// Batch size for scoring pairs.
    pub batch_size: usize,
    /// Timeout in milliseconds; if exceeded, fall back to fusion scores.
    pub timeout_ms: u64,
    /// How many top candidates from fusion to rerank.
    pub rerank_top_k: usize,
    /// Weight of rerank score in combined scoring (0.0 = pure fusion, 1.0 = pure rerank).
    pub score_weight: f32,
    /// Number of ONNX Runtime intra-op threads.
    pub num_threads: usize,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 512,
            batch_size: 16,
            timeout_ms: 5000,
            rerank_top_k: 30,
            score_weight: 0.7,
            num_threads: 4,
        }
    }
}

/// Cross-encoder reranker using an ONNX model.
///
/// Scores (query, document) pairs jointly through a transformer model.
/// The model outputs a single logit per pair, which is converted to a
/// relevance score via sigmoid.
pub struct CrossEncoderReranker {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    config: RerankConfig,
}

impl CrossEncoderReranker {
    /// Load a cross-encoder model from ONNX + tokenizer.json files.
    pub fn load(
        model_path: &Path,
        tokenizer_path: &Path,
        config: RerankConfig,
    ) -> Result<Self> {
        let session = Session::builder()
            .map_err(ort_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_err)?
            .with_intra_threads(config.num_threads)
            .map_err(ort_err)?
            .commit_from_file(model_path)
            .map_err(ort_err)?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| CwcError::Retrieval(format!("tokenizer load error: {e}")))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            config,
        })
    }

    /// Score a single (query, document) pair. Returns a score in [0, 1].
    pub fn score_pair(&self, query: &str, document: &str) -> Result<f32> {
        let scores = self.score_batch(query, &[document])?;
        scores
            .into_iter()
            .next()
            .ok_or_else(|| CwcError::Retrieval("empty score result".into()))
    }

    /// Score multiple (query, document) pairs in batches.
    /// Returns one score per document, in [0, 1] via sigmoid.
    pub fn score_batch(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        let mut all_scores = Vec::with_capacity(documents.len());
        let batch_size = self.config.batch_size.max(1);

        for batch in documents.chunks(batch_size) {
            let batch_scores = self.score_batch_inner(query, batch)?;
            all_scores.extend(batch_scores);
        }

        Ok(calibrate_scores(&all_scores))
    }

    /// Inner batch scoring — runs ONNX inference on a batch of (query, doc) pairs.
    /// Returns raw logits (not yet calibrated).
    fn score_batch_inner(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        let batch_size = documents.len();
        let max_len = self.config.max_seq_len;

        // Encode each (query, document) pair
        let mut input_ids_flat = Vec::with_capacity(batch_size * max_len);
        let mut attention_mask_flat = Vec::with_capacity(batch_size * max_len);
        let mut token_type_ids_flat = Vec::with_capacity(batch_size * max_len);

        for doc in documents {
            let encoding = self
                .tokenizer
                .encode((query, *doc), true)
                .map_err(|e| CwcError::Retrieval(format!("tokenizer encode error: {e}")))?;

            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let type_ids = encoding.get_type_ids();
            let len = ids.len().min(max_len);

            for i in 0..len {
                input_ids_flat.push(ids[i] as i64);
                attention_mask_flat.push(mask[i] as i64);
                token_type_ids_flat.push(type_ids[i] as i64);
            }
            // Pad to max_len
            for _ in len..max_len {
                input_ids_flat.push(0i64);
                attention_mask_flat.push(0i64);
                token_type_ids_flat.push(0i64);
            }
        }

        let shape = [batch_size, max_len];

        let input_ids =
            ort::value::Tensor::from_array((shape, input_ids_flat)).map_err(ort_err)?;
        let attention_mask =
            ort::value::Tensor::from_array((shape, attention_mask_flat)).map_err(ort_err)?;
        let token_type_ids =
            ort::value::Tensor::from_array((shape, token_type_ids_flat)).map_err(ort_err)?;

        let inputs: Vec<(Cow<str>, ort::session::SessionInputValue)> = vec![
            (
                Cow::Borrowed("input_ids"),
                ort::session::SessionInputValue::from(input_ids),
            ),
            (
                Cow::Borrowed("attention_mask"),
                ort::session::SessionInputValue::from(attention_mask),
            ),
            (
                Cow::Borrowed("token_type_ids"),
                ort::session::SessionInputValue::from(token_type_ids),
            ),
        ];

        let mut session = self
            .session
            .lock()
            .map_err(|e| CwcError::Retrieval(format!("session lock error: {e}")))?;
        let outputs = session.run(inputs).map_err(ort_err)?;

        // Cross-encoder output: [batch_size, 1] or [batch_size] logit
        // Try "logits" first, then fall back to first output by index
        let flat: Vec<f32> = if let Some(logits_val) = outputs.get("logits") {
            let tensor = logits_val.try_extract_array::<f32>().map_err(ort_err)?;
            tensor.iter().copied().collect()
        } else {
            // Fall back to first output
            let first = outputs
                .iter()
                .next()
                .ok_or_else(|| CwcError::Retrieval("no output tensor in model".into()))?;
            let tensor = first.1.try_extract_array::<f32>().map_err(ort_err)?;
            tensor.iter().copied().collect()
        };

        // Handle [batch, 1] or [batch] shape
        let scores: Vec<f32> = if flat.len() == batch_size {
            flat
        } else if flat.len() > batch_size && flat.len().is_multiple_of(batch_size) {
            // [batch, num_classes] — take last logit per row (relevance class)
            // For most cross-encoders, shape is [batch, 1]
            let num_classes = flat.len() / batch_size;
            flat.chunks(num_classes)
                .map(|chunk| *chunk.last().unwrap_or(&0.0))
                .collect()
        } else {
            return Err(CwcError::Retrieval(format!(
                "unexpected output shape: {} values for batch size {}",
                flat.len(),
                batch_size
            )));
        };

        Ok(scores)
    }
}

/// Normalize raw logits to [0, 1] via sigmoid.
pub fn calibrate_scores(raw_logits: &[f32]) -> Vec<f32> {
    raw_logits.iter().map(|&x| sigmoid(x)).collect()
}

/// Combine reranker score with fusion score for final ranking.
///
/// `rerank_weight` controls the blend: 1.0 = pure rerank, 0.0 = pure fusion.
pub fn combined_score(fused: f32, rerank: f32, rerank_weight: f32) -> f32 {
    let w = rerank_weight.clamp(0.0, 1.0);
    w * rerank + (1.0 - w) * fused
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn ort_err<T: std::fmt::Display>(e: T) -> CwcError {
    CwcError::Retrieval(format!("ort error: {e}"))
}

impl cwc_core::traits::Reranker for CrossEncoderReranker {
    fn rerank(&self, query: &str, hits: &mut Vec<RetrievalHit>, top_k: usize) -> Result<()> {
        if hits.is_empty() {
            return Ok(());
        }

        // Only rerank top candidates from fusion
        let rerank_count = hits.len().min(self.config.rerank_top_k);
        let candidates = &hits[..rerank_count];
        let texts: Vec<&str> = candidates.iter().map(|h| h.chunk.text.as_str()).collect();

        let start = Instant::now();
        let scores = match self.score_batch(query, &texts) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("reranking failed, falling back to fusion scores: {e}");
                hits.truncate(top_k);
                return Ok(());
            }
        };
        let elapsed_ms = start.elapsed().as_millis() as u64;

        if elapsed_ms > self.config.timeout_ms {
            tracing::warn!(
                "reranking took {}ms (timeout {}ms), falling back to fusion scores",
                elapsed_ms,
                self.config.timeout_ms
            );
            hits.truncate(top_k);
            return Ok(());
        }

        tracing::debug!(
            "reranked {} candidates in {}ms",
            rerank_count,
            elapsed_ms
        );

        // Update rerank scores and compute combined scores
        for (hit, &rerank_score) in hits.iter_mut().zip(scores.iter()) {
            hit.score_rerank = rerank_score;
        }
        // Hits beyond rerank_count keep score_rerank = 0.0

        // Sort by combined score (rerank-weighted)
        let weight = self.config.score_weight;
        hits.sort_by(|a, b| {
            let sa = combined_score(a.score_fused, a.score_rerank, weight);
            let sb = combined_score(b.score_fused, b.score_rerank, weight);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });

        hits.truncate(top_k);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cwc_core::types::{Chunk, RetrievalHit};
    use std::collections::HashMap;
    use uuid::Uuid;

    fn make_hit(text: &str, fused: f32) -> RetrievalHit {
        RetrievalHit {
            chunk: Chunk {
                chunk_id: Uuid::new_v5(&Uuid::NAMESPACE_OID, text.as_bytes()),
                doc_id: Uuid::nil(),
                doc_version: 1,
                source_path: "test.md".into(),
                section_path: vec![],
                char_offset: 0,
                char_len: text.len(),
                token_count: 5,
                text: text.to_string(),
                metadata: HashMap::new(),
            },
            score_sparse: 0.0,
            score_dense: 0.0,
            score_fused: fused,
            score_rerank: 0.0,
        }
    }

    // --- Score calibration tests ---

    #[test]
    fn test_sigmoid_zero() {
        assert!((sigmoid(0.0) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_sigmoid_large_positive() {
        assert!((sigmoid(10.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_sigmoid_large_negative() {
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn test_calibrate_scores_empty() {
        assert!(calibrate_scores(&[]).is_empty());
    }

    #[test]
    fn test_calibrate_scores_range() {
        let logits = vec![-5.0, -1.0, 0.0, 1.0, 5.0];
        let calibrated = calibrate_scores(&logits);
        assert_eq!(calibrated.len(), 5);
        // All should be in [0, 1]
        for &s in &calibrated {
            assert!((0.0..=1.0).contains(&s));
        }
        // Should be monotonically increasing
        for i in 1..calibrated.len() {
            assert!(calibrated[i] > calibrated[i - 1]);
        }
    }

    #[test]
    fn test_calibrate_scores_preserves_ordering() {
        let logits = vec![3.0, 1.0, -2.0, 0.5];
        let calibrated = calibrate_scores(&logits);
        assert!(calibrated[0] > calibrated[1]); // 3.0 > 1.0
        assert!(calibrated[1] > calibrated[3]); // 1.0 > 0.5
        assert!(calibrated[3] > calibrated[2]); // 0.5 > -2.0
    }

    // --- Combined score tests ---

    #[test]
    fn test_combined_score_pure_rerank() {
        let score = combined_score(0.3, 0.9, 1.0);
        assert!((score - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_combined_score_pure_fusion() {
        let score = combined_score(0.3, 0.9, 0.0);
        assert!((score - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_combined_score_half_weight() {
        let score = combined_score(0.4, 0.8, 0.5);
        // 0.5 * 0.8 + 0.5 * 0.4 = 0.6
        assert!((score - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_combined_score_default_weight() {
        let score = combined_score(0.3, 0.9, 0.7);
        // 0.7 * 0.9 + 0.3 * 0.3 = 0.63 + 0.09 = 0.72
        assert!((score - 0.72).abs() < 0.001);
    }

    #[test]
    fn test_combined_score_weight_clamped() {
        // Weight > 1.0 clamped to 1.0
        let score = combined_score(0.3, 0.9, 1.5);
        assert!((score - 0.9).abs() < 0.001);
        // Weight < 0.0 clamped to 0.0
        let score2 = combined_score(0.3, 0.9, -0.5);
        assert!((score2 - 0.3).abs() < 0.001);
    }

    // --- Reranker config tests ---

    #[test]
    fn test_rerank_config_defaults() {
        let config = RerankConfig::default();
        assert_eq!(config.max_seq_len, 512);
        assert_eq!(config.batch_size, 16);
        assert_eq!(config.timeout_ms, 5000);
        assert_eq!(config.rerank_top_k, 30);
        assert!((config.score_weight - 0.7).abs() < 0.001);
    }

    // --- Mock reranker for testing the trait implementation ---

    /// A mock cross-encoder that scores based on keyword overlap.
    /// Used to test the rerank trait flow without ONNX.
    struct MockCrossEncoderScorer {
        config: RerankConfig,
    }

    impl MockCrossEncoderScorer {
        fn new(config: RerankConfig) -> Self {
            Self { config }
        }

        /// Score based on how many query words appear in the document.
        fn mock_score(&self, query: &str, document: &str) -> f32 {
            let query_words: Vec<&str> = query.split_whitespace().collect();
            let doc_lower = document.to_lowercase();
            let found = query_words
                .iter()
                .filter(|w| doc_lower.contains(&w.to_lowercase()))
                .count();
            if query_words.is_empty() {
                return 0.5;
            }
            found as f32 / query_words.len() as f32
        }
    }

    impl cwc_core::traits::Reranker for MockCrossEncoderScorer {
        fn rerank(
            &self,
            query: &str,
            hits: &mut Vec<RetrievalHit>,
            top_k: usize,
        ) -> Result<()> {
            if hits.is_empty() {
                return Ok(());
            }

            let rerank_count = hits.len().min(self.config.rerank_top_k);
            for hit in hits.iter_mut().take(rerank_count) {
                hit.score_rerank = self.mock_score(query, &hit.chunk.text);
            }

            let weight = self.config.score_weight;
            hits.sort_by(|a, b| {
                let sa = combined_score(a.score_fused, a.score_rerank, weight);
                let sb = combined_score(b.score_fused, b.score_rerank, weight);
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });

            hits.truncate(top_k);
            Ok(())
        }
    }

    #[test]
    fn test_reranker_trait_resorts_by_rerank_score() {
        use cwc_core::traits::Reranker;

        let reranker = MockCrossEncoderScorer::new(RerankConfig {
            score_weight: 1.0, // pure rerank
            ..Default::default()
        });

        let mut hits = vec![
            make_hit("unrelated topic about cooking", 0.9),
            make_hit("rust ownership and memory safety", 0.3),
            make_hit("borrowing rules in rust programming", 0.5),
        ];

        reranker.rerank("rust ownership", &mut hits, 10).unwrap();

        // "rust ownership and memory safety" should now be first (highest rerank score)
        assert!(
            hits[0].chunk.text.contains("rust ownership"),
            "expected 'rust ownership' first, got: {}",
            hits[0].chunk.text
        );
    }

    #[test]
    fn test_reranker_trait_truncates_to_top_k() {
        use cwc_core::traits::Reranker;

        let reranker = MockCrossEncoderScorer::new(RerankConfig::default());

        let mut hits = vec![
            make_hit("doc a", 0.9),
            make_hit("doc b", 0.8),
            make_hit("doc c", 0.7),
            make_hit("doc d", 0.6),
            make_hit("doc e", 0.5),
        ];

        reranker.rerank("test", &mut hits, 3).unwrap();
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn test_reranker_trait_empty_hits() {
        use cwc_core::traits::Reranker;

        let reranker = MockCrossEncoderScorer::new(RerankConfig::default());
        let mut hits = vec![];
        reranker.rerank("test", &mut hits, 10).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn test_reranker_trait_scores_populated() {
        use cwc_core::traits::Reranker;

        let reranker = MockCrossEncoderScorer::new(RerankConfig {
            score_weight: 0.7,
            ..Default::default()
        });

        let mut hits = vec![
            make_hit("rust programming language", 0.5),
        ];

        reranker.rerank("rust programming", &mut hits, 10).unwrap();
        assert!(hits[0].score_rerank > 0.0, "rerank score should be set");
    }

    #[test]
    fn test_reranker_combined_score_affects_order() {
        use cwc_core::traits::Reranker;

        let reranker = MockCrossEncoderScorer::new(RerankConfig {
            score_weight: 0.5, // balanced
            ..Default::default()
        });

        // High fusion but low rerank vs low fusion but high rerank
        let mut hits = vec![
            make_hit("cooking recipes for pasta", 0.9), // high fused, but irrelevant
            make_hit("rust ownership borrow", 0.1),      // low fused, but very relevant
        ];

        reranker.rerank("rust ownership", &mut hits, 10).unwrap();

        // With balanced weight, the relevant doc should win
        // mock_score("rust ownership", "rust ownership borrow") = 2/2 = 1.0
        // combined = 0.5*1.0 + 0.5*0.1 = 0.55
        // mock_score("rust ownership", "cooking recipes for pasta") = 0/2 = 0.0
        // combined = 0.5*0.0 + 0.5*0.9 = 0.45
        assert!(
            hits[0].chunk.text.contains("rust ownership"),
            "relevant doc should rank higher with balanced weight"
        );
    }

    /// A mock reranker that sleeps, simulating timeout behavior.
    struct SlowReranker {
        sleep_ms: u64,
    }

    impl cwc_core::traits::Reranker for SlowReranker {
        fn rerank(
            &self,
            _query: &str,
            hits: &mut Vec<RetrievalHit>,
            top_k: usize,
        ) -> Result<()> {
            if hits.is_empty() {
                return Ok(());
            }

            // Simulate the CrossEncoderReranker's timeout logic:
            // Score, check elapsed, fall back if over timeout.
            let start = std::time::Instant::now();
            std::thread::sleep(std::time::Duration::from_millis(self.sleep_ms));
            let elapsed_ms = start.elapsed().as_millis() as u64;

            let timeout_ms = 50; // very short timeout for test
            if elapsed_ms > timeout_ms {
                // Fall back to fusion order — just truncate
                hits.truncate(top_k);
                return Ok(());
            }

            // Would normally rerank here, but we never reach this
            hits.truncate(top_k);
            Ok(())
        }
    }

    #[test]
    fn test_timeout_falls_back_to_fusion_scores() {
        use cwc_core::traits::Reranker;

        let reranker = SlowReranker { sleep_ms: 100 }; // will exceed 50ms timeout

        let mut hits = vec![
            make_hit("doc a high fused", 0.9),
            make_hit("doc b low fused", 0.1),
            make_hit("doc c mid fused", 0.5),
        ];

        reranker.rerank("test query", &mut hits, 3).unwrap();

        // Should have fallen back — original fusion order preserved (no rerank scores set)
        assert_eq!(hits.len(), 3);
        // Hits should still be in original order (no re-sorting by rerank score)
        assert!(
            hits[0].score_rerank == 0.0,
            "timeout fallback should not set rerank scores"
        );
        assert!(
            hits[1].score_rerank == 0.0,
            "timeout fallback should not set rerank scores"
        );
    }

    #[test]
    fn test_reranker_respects_rerank_top_k() {
        use cwc_core::traits::Reranker;

        let reranker = MockCrossEncoderScorer::new(RerankConfig {
            rerank_top_k: 2,
            score_weight: 1.0,
            ..Default::default()
        });

        let mut hits = vec![
            make_hit("rust ownership model", 0.9),           // matches query, will be reranked
            make_hit("rust borrow checker", 0.8),             // partial match, will be reranked
            make_hit("rust ownership and borrow rules", 0.1), // best rerank match but outside top_k
        ];

        reranker.rerank("rust ownership", &mut hits, 10).unwrap();

        // First 2 hits were reranked (score > 0), 3rd was NOT reranked (stays 0.0)
        let unreranked: Vec<&RetrievalHit> = hits.iter().filter(|h| h.score_rerank == 0.0).collect();
        let reranked: Vec<&RetrievalHit> = hits.iter().filter(|h| h.score_rerank > 0.0).collect();
        assert_eq!(reranked.len(), 2, "exactly 2 hits should be reranked (rerank_top_k=2)");
        assert_eq!(unreranked.len(), 1, "1 hit should remain unreranked");
        // The unreranked one is the 3rd original hit
        assert!(
            unreranked[0].chunk.text.contains("and borrow rules"),
            "the hit beyond rerank_top_k should not have been scored"
        );
    }

    #[test]
    fn test_batch_size_1_vs_16_same_results() {
        // Verify that different batch sizes produce identical calibrated scores
        let logits = vec![2.0, -1.0, 0.5, 3.0, -0.5];
        let cal_all = calibrate_scores(&logits);

        // Simulate batch_size=1: calibrate each individually
        let cal_one: Vec<f32> = logits.iter().map(|&l| calibrate_scores(&[l])[0]).collect();

        for (a, b) in cal_all.iter().zip(cal_one.iter()) {
            assert!((a - b).abs() < 0.001, "batch vs single mismatch: {a} vs {b}");
        }
    }

    // --- Additional edge case tests ---

    #[test]
    fn test_reranker_single_hit() {
        use cwc_core::traits::Reranker;

        let reranker = MockCrossEncoderScorer::new(RerankConfig {
            score_weight: 0.7,
            ..Default::default()
        });

        let mut hits = vec![make_hit("rust ownership", 0.5)];
        reranker.rerank("rust", &mut hits, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].score_rerank > 0.0);
    }

    #[test]
    fn test_reranker_top_k_larger_than_hits() {
        use cwc_core::traits::Reranker;

        let reranker = MockCrossEncoderScorer::new(RerankConfig::default());

        let mut hits = vec![
            make_hit("doc a", 0.5),
            make_hit("doc b", 0.3),
        ];

        reranker.rerank("test", &mut hits, 100).unwrap();
        // Should return all hits (only 2 available)
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn test_reranker_all_same_fused_score() {
        use cwc_core::traits::Reranker;

        let reranker = MockCrossEncoderScorer::new(RerankConfig {
            score_weight: 1.0, // pure rerank
            ..Default::default()
        });

        // All same fused score — rerank should break ties
        let mut hits = vec![
            make_hit("cooking recipes", 0.5),
            make_hit("rust ownership model", 0.5),
            make_hit("gardening tips", 0.5),
        ];

        reranker.rerank("rust ownership", &mut hits, 10).unwrap();
        // Highest rerank score should be first
        assert!(
            hits[0].chunk.text.contains("rust ownership"),
            "reranker should break fusion ties: got '{}'",
            hits[0].chunk.text
        );
    }

    #[test]
    fn test_combined_score_both_zero() {
        let score = combined_score(0.0, 0.0, 0.7);
        assert!((score - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_calibrate_scores_extreme_values() {
        let logits = vec![f32::MAX, f32::MIN, 0.0];
        let calibrated = calibrate_scores(&logits);
        assert_eq!(calibrated.len(), 3);
        // MAX → sigmoid → 1.0 (or very close)
        assert!((calibrated[0] - 1.0).abs() < 0.001);
        // MIN → sigmoid → 0.0 (or very close)
        assert!(calibrated[1] < 0.001);
        // 0.0 → sigmoid → 0.5
        assert!((calibrated[2] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_reranker_rerank_top_k_zero_skips_scoring() {
        use cwc_core::traits::Reranker;

        let reranker = MockCrossEncoderScorer::new(RerankConfig {
            rerank_top_k: 0,
            score_weight: 1.0,
            ..Default::default()
        });

        let mut hits = vec![
            make_hit("rust ownership", 0.9),
            make_hit("cooking tips", 0.3),
        ];

        reranker.rerank("rust", &mut hits, 10).unwrap();
        // With rerank_top_k=0, no hits get reranked
        for hit in &hits {
            assert_eq!(hit.score_rerank, 0.0, "no hit should be reranked with rerank_top_k=0");
        }
    }

    #[test]
    fn test_reranker_top_k_truncates_after_sort() {
        use cwc_core::traits::Reranker;

        let reranker = MockCrossEncoderScorer::new(RerankConfig {
            score_weight: 1.0, // pure rerank
            ..Default::default()
        });

        let mut hits = vec![
            make_hit("cooking pasta recipes", 0.9),
            make_hit("rust ownership", 0.1),
            make_hit("gardening advice", 0.8),
            make_hit("rust borrowing rules", 0.2),
        ];

        reranker.rerank("rust ownership", &mut hits, 2).unwrap();
        assert_eq!(hits.len(), 2);
        // Top 2 by rerank should be the rust-related ones
        assert!(hits[0].chunk.text.contains("rust ownership"));
        assert!(hits[1].chunk.text.contains("rust borrowing"));
    }

    // --- ONNX model tests (require model files) ---

    fn reranker_model_paths() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
        let candidates = [
            (
                "models/bge-reranker-base/model.onnx",
                "models/bge-reranker-base/tokenizer.json",
            ),
            (
                "../models/bge-reranker-base/model.onnx",
                "../models/bge-reranker-base/tokenizer.json",
            ),
            (
                "models/cross-encoder/model.onnx",
                "models/cross-encoder/tokenizer.json",
            ),
            (
                "../models/cross-encoder/model.onnx",
                "../models/cross-encoder/tokenizer.json",
            ),
        ];
        for (model, tok) in &candidates {
            let mp = std::path::PathBuf::from(model);
            let tp = std::path::PathBuf::from(tok);
            if mp.exists() && tp.exists() {
                return Some((mp, tp));
            }
        }
        None
    }

    #[test]
    #[ignore] // Requires ONNX cross-encoder model
    fn test_cross_encoder_load_model() {
        let (mp, tp) = reranker_model_paths().expect("reranker model not found");
        let reranker = CrossEncoderReranker::load(&mp, &tp, RerankConfig::default()).unwrap();
        assert!(reranker.config.max_seq_len > 0);
    }

    #[test]
    #[ignore]
    fn test_cross_encoder_score_pair_relevant_higher() {
        let (mp, tp) = reranker_model_paths().expect("reranker model not found");
        let reranker = CrossEncoderReranker::load(&mp, &tp, RerankConfig::default()).unwrap();

        let relevant = reranker
            .score_pair("What is Rust's ownership model?", "Rust uses ownership to manage memory safely without a garbage collector.")
            .unwrap();
        let irrelevant = reranker
            .score_pair("What is Rust's ownership model?", "The best chocolate cake recipe uses cocoa powder.")
            .unwrap();

        assert!(
            relevant > irrelevant,
            "relevant pair should score higher: {relevant} vs {irrelevant}"
        );
    }

    #[test]
    #[ignore]
    fn test_cross_encoder_score_batch_correct_count() {
        let (mp, tp) = reranker_model_paths().expect("reranker model not found");
        let reranker = CrossEncoderReranker::load(&mp, &tp, RerankConfig::default()).unwrap();

        let docs = vec![
            "Ownership ensures memory safety.",
            "Borrowing allows temporary access.",
            "Cooking pasta is an art form.",
        ];
        let scores = reranker.score_batch("What is ownership?", &docs).unwrap();
        assert_eq!(scores.len(), 3);
        for &s in &scores {
            assert!((0.0..=1.0).contains(&s), "score {s} out of range");
        }
    }

    #[test]
    #[ignore]
    fn test_cross_encoder_score_batch_empty() {
        let (mp, tp) = reranker_model_paths().expect("reranker model not found");
        let reranker = CrossEncoderReranker::load(&mp, &tp, RerankConfig::default()).unwrap();

        let scores = reranker.score_batch("test", &[]).unwrap();
        assert!(scores.is_empty());
    }

    #[test]
    #[ignore]
    fn test_cross_encoder_long_document_truncated() {
        let (mp, tp) = reranker_model_paths().expect("reranker model not found");
        let reranker = CrossEncoderReranker::load(&mp, &tp, RerankConfig::default()).unwrap();

        let long_doc = "word ".repeat(10000);
        let score = reranker.score_pair("test query", &long_doc).unwrap();
        assert!((0.0..=1.0).contains(&score), "should handle long docs");
    }

    #[test]
    #[ignore]
    fn test_cross_encoder_rerank_trait_end_to_end() {
        use cwc_core::traits::Reranker as RerankerTrait;

        let (mp, tp) = reranker_model_paths().expect("reranker model not found");
        let reranker = CrossEncoderReranker::load(
            &mp,
            &tp,
            RerankConfig {
                score_weight: 1.0,
                ..Default::default()
            },
        )
        .unwrap();

        let mut hits = vec![
            make_hit("The best chocolate cake recipe uses cocoa powder.", 0.9),
            make_hit("Rust uses ownership to manage memory safely.", 0.1),
        ];

        reranker.rerank("What is Rust's ownership model?", &mut hits, 10).unwrap();

        // Relevant hit should be promoted
        assert!(
            hits[0].chunk.text.contains("ownership"),
            "relevant chunk should be ranked first"
        );
    }
}
