use std::path::Path;
use std::sync::Mutex;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use tokenizers::Tokenizer;

use crate::config::EmbedConfig;
use crate::error::{EmbedError, Result};

/// ONNX Runtime-based embedding model (E5, BGE family).
pub struct OnnxEmbedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    dim: usize,
    max_seq_len: usize,
    normalize: bool,
    query_prefix: Option<String>,
    doc_prefix: Option<String>,
    batch_size: usize,
}

impl OnnxEmbedder {
    /// Load model from ONNX file + tokenizer.json.
    pub fn load(
        model_path: &Path,
        tokenizer_path: &Path,
        config: &EmbedConfig,
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
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            dim: config.dim,
            max_seq_len: config.max_seq_len,
            normalize: config.normalize,
            query_prefix: config.query_prefix.clone(),
            doc_prefix: config.doc_prefix.clone(),
            batch_size: config.batch_size,
        })
    }

    /// Embed a single document text (applies doc_prefix).
    pub fn embed_document(&self, text: &str) -> Result<Vec<f32>> {
        let prefixed = match &self.doc_prefix {
            Some(p) => format!("{p}{text}"),
            None => text.to_string(),
        };
        self.embed_single(&prefixed)
    }

    /// Embed a query text (applies query_prefix).
    pub fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let prefixed = match &self.query_prefix {
            Some(p) => format!("{p}{query}"),
            None => query.to_string(),
        };
        self.embed_single(&prefixed)
    }

    /// Batch embed documents.
    pub fn embed_documents_batch(
        &self,
        texts: &[&str],
        batch_size: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let mut all_embeddings = Vec::with_capacity(texts.len());
        let batch_size = batch_size.max(1);

        for batch in texts.chunks(batch_size) {
            let prefixed: Vec<String> = batch
                .iter()
                .map(|t| match &self.doc_prefix {
                    Some(p) => format!("{p}{t}"),
                    None => t.to_string(),
                })
                .collect();
            let refs: Vec<&str> = prefixed.iter().map(|s| s.as_str()).collect();
            let mut batch_embs = self.embed_batch(&refs)?;
            all_embeddings.append(&mut batch_embs);
        }

        Ok(all_embeddings)
    }

    fn embed_single(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text])?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| EmbedError::Embed("empty embedding result".to_string()))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        let batch_size = encodings.len();

        // Truncate and pad to max_seq_len
        let mut input_ids_flat = Vec::with_capacity(batch_size * self.max_seq_len);
        let mut attention_mask_flat = Vec::with_capacity(batch_size * self.max_seq_len);
        let mut token_type_ids_flat = Vec::with_capacity(batch_size * self.max_seq_len);

        for enc in &encodings {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let type_ids = enc.get_type_ids();
            let len = ids.len().min(self.max_seq_len);

            for i in 0..len {
                input_ids_flat.push(ids[i] as i64);
                attention_mask_flat.push(mask[i] as i64);
                token_type_ids_flat.push(type_ids[i] as i64);
            }
            for _ in len..self.max_seq_len {
                input_ids_flat.push(0i64);
                attention_mask_flat.push(0i64);
                token_type_ids_flat.push(0i64);
            }
        }

        let shape = [batch_size, self.max_seq_len];

        let input_ids =
            ort::value::Tensor::from_array((shape, input_ids_flat)).map_err(ort_err)?;
        let attention_mask =
            ort::value::Tensor::from_array((shape, attention_mask_flat)).map_err(ort_err)?;
        let token_type_ids =
            ort::value::Tensor::from_array((shape, token_type_ids_flat)).map_err(ort_err)?;

        use std::borrow::Cow;
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

        let mut session = self.session.lock().map_err(|e| EmbedError::Embed(e.to_string()))?;
        let outputs = session.run(inputs).map_err(ort_err)?;

        // Try pooled outputs first, then fall back to mean pooling
        let output_names = ["sentence_embedding", "pooler_output"];
        for name in &output_names {
            if let Some(val) = outputs.get(*name) {
                let tensor = val.try_extract_array::<f32>().map_err(ort_err)?;
                let array: ndarray::ArrayView2<f32> = tensor
                    .into_dimensionality()
                    .map_err(|e| EmbedError::Embed(format!("shape error: {e}")))?;
                return Ok(self.extract_embeddings_view(array));
            }
        }

        // Fall back to last_hidden_state and do mean pooling
        let hidden_val = outputs
            .get("last_hidden_state")
            .ok_or_else(|| {
                EmbedError::Embed("no recognized output tensor in model".to_string())
            })?;
        let hidden = hidden_val.try_extract_array::<f32>().map_err(ort_err)?;
        let hidden: ndarray::ArrayView3<f32> = hidden
            .into_dimensionality()
            .map_err(|e| EmbedError::Embed(format!("shape error: {e}")))?;

        let seq_len = hidden.shape()[1];
        let dim = hidden.shape()[2];

        let mut pooled = vec![0.0f32; batch_size * dim];
        for b in 0..batch_size {
            let enc = &encodings[b];
            let mask = enc.get_attention_mask();
            let valid_len = mask.iter().take(seq_len).filter(|&&m| m == 1).count();
            if valid_len == 0 {
                continue;
            }
            for s in 0..seq_len.min(mask.len()) {
                if mask[s] == 1 {
                    for d in 0..dim {
                        pooled[b * dim + d] += hidden[[b, s, d]];
                    }
                }
            }
            for d in 0..dim {
                pooled[b * dim + d] /= valid_len as f32;
            }
        }

        let pooled_array = ndarray::Array2::from_shape_vec((batch_size, dim), pooled)
            .map_err(|e| EmbedError::Embed(e.to_string()))?;
        Ok(self.extract_embeddings_view(pooled_array.view()))
    }

    fn extract_embeddings_view(&self, tensor: ndarray::ArrayView2<f32>) -> Vec<Vec<f32>> {
        let batch_size = tensor.shape()[0];
        let mut results = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let row = tensor.row(b);
            let mut vec: Vec<f32> = row.to_vec();

            if self.normalize {
                let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for v in &mut vec {
                        *v /= norm;
                    }
                }
            }

            results.push(vec);
        }

        results
    }
}

/// Convert any ort::Error<T> to EmbedError via Display.
fn ort_err<T: std::fmt::Display>(e: T) -> EmbedError {
    EmbedError::Ort(e.to_string())
}

impl cwc_core::traits::Embedder for OnnxEmbedder {
    fn embed(&self, texts: &[&str]) -> cwc_core::Result<Vec<Vec<f32>>> {
        self.embed_documents_batch(texts, self.batch_size)
            .map_err(|e| cwc_core::CwcError::Embedding(e.to_string()))
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model_paths() -> Option<(PathBuf, PathBuf)> {
        let candidates = [
            (
                "models/bge-small-en-v1.5/model.onnx",
                "models/bge-small-en-v1.5/tokenizer.json",
            ),
            (
                "../models/bge-small-en-v1.5/model.onnx",
                "../models/bge-small-en-v1.5/tokenizer.json",
            ),
        ];
        for (model, tok) in &candidates {
            let mp = PathBuf::from(model);
            let tp = PathBuf::from(tok);
            if mp.exists() && tp.exists() {
                return Some((mp, tp));
            }
        }
        None
    }

    fn load_test_embedder() -> Option<OnnxEmbedder> {
        let (mp, tp) = model_paths()?;
        let config = EmbedConfig {
            model_path: mp.clone(),
            tokenizer_path: tp.clone(),
            dim: 384,
            max_seq_len: 512,
            normalize: true,
            query_prefix: None,
            doc_prefix: None,
            batch_size: 32,
            num_threads: 4,
        };
        OnnxEmbedder::load(&mp, &tp, &config).ok()
    }

    #[test]
    #[ignore] // Requires ONNX model files
    fn test_onnx_load_model() {
        let embedder = load_test_embedder().expect("model not found");
        assert_eq!(embedder.dim, 384);
    }

    #[test]
    #[ignore]
    fn test_onnx_single_embedding_dimension() {
        let embedder = load_test_embedder().expect("model not found");
        let emb = embedder.embed_document("Hello world").unwrap();
        assert_eq!(emb.len(), 384);
    }

    #[test]
    #[ignore]
    fn test_onnx_embedding_normalized() {
        let embedder = load_test_embedder().expect("model not found");
        let emb = embedder.embed_document("Hello world").unwrap();
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "expected L2 norm ~1.0, got {norm}"
        );
    }

    #[test]
    #[ignore]
    fn test_onnx_batch_embedding() {
        let embedder = load_test_embedder().expect("model not found");
        let texts = vec!["Hello world", "Rust programming", "Machine learning"];
        let embs = embedder.embed_documents_batch(&texts, 2).unwrap();
        assert_eq!(embs.len(), 3);
        for emb in &embs {
            assert_eq!(emb.len(), 384);
        }
    }

    #[test]
    #[ignore]
    fn test_onnx_query_prefix() {
        let (mp, tp) = model_paths().expect("model not found");
        let config = EmbedConfig {
            model_path: mp.clone(),
            tokenizer_path: tp.clone(),
            dim: 384,
            max_seq_len: 512,
            normalize: true,
            query_prefix: Some("query: ".to_string()),
            doc_prefix: Some("passage: ".to_string()),
            batch_size: 32,
            num_threads: 4,
        };
        let embedder = OnnxEmbedder::load(&mp, &tp, &config).unwrap();

        let q_emb = embedder.embed_query("test").unwrap();
        let d_emb = embedder.embed_document("test").unwrap();

        let sim = cosine_similarity(&q_emb, &d_emb);
        assert!(
            sim < 1.0,
            "query and doc embeddings should differ due to prefix"
        );
    }

    #[test]
    #[ignore]
    fn test_onnx_semantic_similarity() {
        let embedder = load_test_embedder().expect("model not found");
        let e1 = embedder
            .embed_document("Rust uses ownership for memory safety")
            .unwrap();
        let e2 = embedder
            .embed_document("Memory management in Rust without garbage collection")
            .unwrap();
        let e3 = embedder
            .embed_document("How to bake chocolate chip cookies")
            .unwrap();

        let sim_related = cosine_similarity(&e1, &e2);
        let sim_unrelated = cosine_similarity(&e1, &e3);
        assert!(
            sim_related > sim_unrelated,
            "related texts should be more similar: {sim_related} vs {sim_unrelated}"
        );
    }

    #[test]
    #[ignore]
    fn test_onnx_determinism() {
        let embedder = load_test_embedder().expect("model not found");
        let e1 = embedder.embed_document("Deterministic test").unwrap();
        let e2 = embedder.embed_document("Deterministic test").unwrap();
        assert_eq!(e1, e2, "same text should produce same embedding");
    }

    #[test]
    #[ignore]
    fn test_onnx_max_seq_len_truncation() {
        let embedder = load_test_embedder().expect("model not found");
        let long_text = "word ".repeat(10000);
        let emb = embedder.embed_document(&long_text).unwrap();
        assert_eq!(
            emb.len(),
            384,
            "long text should still produce valid embedding"
        );
    }

    #[test]
    #[ignore]
    fn test_onnx_empty_text() {
        let embedder = load_test_embedder().expect("model not found");
        let emb = embedder.embed_document("").unwrap();
        assert_eq!(
            emb.len(),
            384,
            "empty text should produce valid embedding"
        );
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        // zip truncates to shortest — this should not panic
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        // Only first 2 elements compared: dot=1.0, norm_a=1.0, norm_b=1.0
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_normalized_vectors() {
        // For L2-normalized vectors, cosine similarity = dot product
        let a = vec![0.6, 0.8, 0.0]; // norm = 1.0
        let b = vec![0.8, 0.6, 0.0]; // norm = 1.0
        let sim = cosine_similarity(&a, &b);
        let expected = 0.6 * 0.8 + 0.8 * 0.6; // 0.96
        assert!((sim - expected).abs() < 1e-5);
    }
}
