use std::sync::LazyLock;

use regex::Regex;

use crate::dataset::{EvalQuery, QueryCategory};

static CITATION_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[S\d+\]").expect("citation regex"));

/// Aggregated generation quality metrics.
#[derive(Debug, Clone, Default)]
pub struct GenerationMetrics {
    /// Fraction of factual sentences that contain at least one citation.
    pub citation_rate: f32,
    /// Fraction of citations that reference valid source IDs.
    pub citation_validity: f32,
    /// Fraction of unanswerable queries correctly abstained.
    pub abstention_accuracy: f32,
    /// Fraction of outputs that matched the expected schema.
    pub schema_compliance: f32,
    /// Heuristic answer relevance: fraction of query terms found in answer.
    pub answer_relevance: f32,
    /// Average ratio of output tokens to query tokens.
    pub verbosity_ratio: f32,
    /// Number of results evaluated.
    pub result_count: usize,
}

/// A single evaluation result pairing a query with pipeline output.
#[derive(Debug, Clone)]
pub struct EvalResult {
    pub query: EvalQuery,
    pub response_text: String,
    pub citations_found: Vec<String>,
    pub is_abstention: bool,
    pub schema_valid: bool,
    pub source_ids_available: Vec<String>,
    pub response_token_count: u32,
    pub query_token_count: u32,
}

/// Compute generation metrics from a set of evaluation results.
pub fn compute_generation_metrics(results: &[EvalResult]) -> GenerationMetrics {
    if results.is_empty() {
        return GenerationMetrics::default();
    }

    let mut total_citation_rate = 0.0f64;
    let mut citation_rate_count = 0usize;
    let mut total_citation_validity = 0.0f64;
    let mut citation_validity_count = 0usize;
    let mut abstention_correct = 0usize;
    let mut abstention_total = 0usize;
    let mut schema_valid = 0usize;
    let mut total_relevance = 0.0f64;
    let mut total_verbosity = 0.0f64;

    for result in results {
        // Citation rate: for factual/multihop/comparison queries
        if is_citation_applicable(&result.query.category) {
            let rate = sentence_citation_rate(&result.response_text);
            total_citation_rate += rate as f64;
            citation_rate_count += 1;
        }

        // Citation validity: all citations should reference valid source IDs
        if !result.citations_found.is_empty() {
            let valid = result
                .citations_found
                .iter()
                .filter(|c| result.source_ids_available.contains(c))
                .count();
            total_citation_validity += valid as f64 / result.citations_found.len() as f64;
            citation_validity_count += 1;
        }

        // Abstention accuracy
        if result.query.category == QueryCategory::Unanswerable {
            abstention_total += 1;
            if result.is_abstention {
                abstention_correct += 1;
            }
        }

        // Schema compliance
        if result.schema_valid {
            schema_valid += 1;
        }

        // Answer relevance
        total_relevance += answer_relevance_score(&result.query.query, &result.response_text) as f64;

        // Verbosity ratio
        if result.query_token_count > 0 {
            total_verbosity +=
                result.response_token_count as f64 / result.query_token_count as f64;
        }
    }

    let n = results.len() as f64;

    GenerationMetrics {
        citation_rate: if citation_rate_count > 0 {
            (total_citation_rate / citation_rate_count as f64) as f32
        } else {
            0.0
        },
        citation_validity: if citation_validity_count > 0 {
            (total_citation_validity / citation_validity_count as f64) as f32
        } else {
            1.0 // no citations to be invalid
        },
        abstention_accuracy: if abstention_total > 0 {
            abstention_correct as f32 / abstention_total as f32
        } else {
            1.0 // no unanswerable queries
        },
        schema_compliance: schema_valid as f32 / results.len().max(1) as f32,
        answer_relevance: (total_relevance / n) as f32,
        verbosity_ratio: (total_verbosity / n) as f32,
        result_count: results.len(),
    }
}

/// Whether citation metrics apply to this category.
fn is_citation_applicable(category: &QueryCategory) -> bool {
    matches!(
        category,
        QueryCategory::Factual | QueryCategory::MultiHop | QueryCategory::Comparison
    )
}

/// Fraction of non-trivial sentences that contain a citation marker [S\d+].
fn sentence_citation_rate(text: &str) -> f32 {
    let sentences: Vec<&str> = text
        .split(['.', '!', '?'])
        .map(|s| s.trim())
        .filter(|s| s.split_whitespace().count() >= 3) // non-trivial
        .collect();

    if sentences.is_empty() {
        return 0.0;
    }

    let with_citation = sentences
        .iter()
        .filter(|s| CITATION_RE.is_match(s))
        .count();

    with_citation as f32 / sentences.len() as f32
}

/// Heuristic relevance: fraction of significant query words found in the answer.
fn answer_relevance_score(query: &str, answer: &str) -> f32 {
    let stop_words: std::collections::HashSet<&str> = [
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall",
        "can", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "about",
        "what", "how", "why", "when", "where", "which", "who", "whom", "this", "that", "these",
        "those", "it", "its", "and", "or", "but", "not", "no", "if", "then", "than", "so",
    ]
    .into_iter()
    .collect();

    let query_words: Vec<String> = query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 3 && !stop_words.contains(w))
        .map(|w| w.to_string())
        .collect();

    if query_words.is_empty() {
        return 1.0; // no meaningful words to check
    }

    let answer_lower = answer.to_lowercase();
    let found = query_words
        .iter()
        .filter(|w| answer_lower.contains(w.as_str()))
        .count();

    found as f32 / query_words.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Difficulty;

    fn make_query(category: QueryCategory) -> EvalQuery {
        EvalQuery {
            query: "What is ownership in Rust?".into(),
            expected_answer: None,
            relevant_chunk_ids: vec![],
            expected_citations: vec![],
            category,
            difficulty: Difficulty::Easy,
        }
    }

    fn make_result(
        category: QueryCategory,
        response: &str,
        citations: Vec<&str>,
        source_ids: Vec<&str>,
        is_abstention: bool,
    ) -> EvalResult {
        EvalResult {
            query: make_query(category),
            response_text: response.into(),
            citations_found: citations.into_iter().map(String::from).collect(),
            is_abstention,
            schema_valid: true,
            source_ids_available: source_ids.into_iter().map(String::from).collect(),
            response_token_count: response.split_whitespace().count() as u32,
            query_token_count: 5,
        }
    }

    #[test]
    fn test_all_citations_valid() {
        let results = vec![make_result(
            QueryCategory::Factual,
            "Ownership is key [S1]. It ensures safety [S2].",
            vec!["[S1]", "[S2]"],
            vec!["[S1]", "[S2]", "[S3]"],
            false,
        )];
        let m = compute_generation_metrics(&results);
        assert!((m.citation_validity - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_partial_citation_validity() {
        let results = vec![make_result(
            QueryCategory::Factual,
            "Ownership [S1]. Safety [S99].",
            vec!["[S1]", "[S99]"],
            vec!["[S1]", "[S2]"],
            false,
        )];
        let m = compute_generation_metrics(&results);
        assert!((m.citation_validity - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_correct_abstention() {
        let results = vec![
            make_result(QueryCategory::Unanswerable, "INSUFFICIENT_EVIDENCE", vec![], vec![], true),
            make_result(QueryCategory::Unanswerable, "I think...", vec![], vec![], false),
        ];
        let m = compute_generation_metrics(&results);
        assert!((m.abstention_accuracy - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_all_abstentions_correct() {
        let results = vec![
            make_result(QueryCategory::Unanswerable, "", vec![], vec![], true),
        ];
        let m = compute_generation_metrics(&results);
        assert!((m.abstention_accuracy - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_no_unanswerable_queries() {
        let results = vec![make_result(
            QueryCategory::Factual,
            "Answer [S1].",
            vec!["[S1]"],
            vec!["[S1]"],
            false,
        )];
        let m = compute_generation_metrics(&results);
        // No unanswerable queries → abstention_accuracy = 1.0
        assert!((m.abstention_accuracy - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_citation_rate_all_cited() {
        let results = vec![make_result(
            QueryCategory::Factual,
            "Ownership ensures safety [S1]. Each value has one owner [S2]. This prevents bugs [S1].",
            vec!["[S1]", "[S2]"],
            vec!["[S1]", "[S2]"],
            false,
        )];
        let m = compute_generation_metrics(&results);
        assert!((m.citation_rate - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_citation_rate_none_cited() {
        let results = vec![make_result(
            QueryCategory::Factual,
            "Ownership ensures safety. Each value has one owner. This prevents bugs.",
            vec![],
            vec!["[S1]"],
            false,
        )];
        let m = compute_generation_metrics(&results);
        assert!((m.citation_rate).abs() < 0.001);
    }

    #[test]
    fn test_citation_rate_not_applied_to_procedural() {
        let results = vec![make_result(
            QueryCategory::Procedural,
            "To sort a list, use the sort method.",
            vec![],
            vec![],
            false,
        )];
        let m = compute_generation_metrics(&results);
        // Procedural queries don't affect citation rate
        assert!((m.citation_rate).abs() < 0.001);
    }

    #[test]
    fn test_schema_compliance() {
        let mut r1 = make_result(QueryCategory::Factual, "ok", vec![], vec![], false);
        r1.schema_valid = true;
        let mut r2 = make_result(QueryCategory::Factual, "bad", vec![], vec![], false);
        r2.schema_valid = false;

        let m = compute_generation_metrics(&[r1, r2]);
        assert!((m.schema_compliance - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_answer_relevance_all_terms() {
        let score = answer_relevance_score(
            "What is ownership in Rust?",
            "Ownership in Rust is the core memory safety feature.",
        );
        assert!(score > 0.5); // "ownership" and "rust" should be found
    }

    #[test]
    fn test_answer_relevance_no_terms() {
        let score = answer_relevance_score(
            "What is ownership in Rust?",
            "The sky is blue and water is wet.",
        );
        assert!(score < 0.5);
    }

    #[test]
    fn test_empty_results() {
        let m = compute_generation_metrics(&[]);
        assert_eq!(m.result_count, 0);
    }

    #[test]
    fn test_verbosity_ratio() {
        let results = vec![EvalResult {
            query: make_query(QueryCategory::Factual),
            response_text: "This is a ten word response to the user query given.".into(),
            citations_found: vec![],
            is_abstention: false,
            schema_valid: true,
            source_ids_available: vec![],
            response_token_count: 10,
            query_token_count: 5,
        }];
        let m = compute_generation_metrics(&results);
        assert!((m.verbosity_ratio - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_citation_rate_multihop_and_comparison() {
        // MultiHop and Comparison categories should also count for citation rate
        let r1 = make_result(
            QueryCategory::MultiHop,
            "Fact one is important [S1]. Fact two also matters [S2].",
            vec!["[S1]", "[S2]"],
            vec!["[S1]", "[S2]"],
            false,
        );
        let r2 = make_result(
            QueryCategory::Comparison,
            "A is better than B [S1]. But B has advantages [S2].",
            vec!["[S1]", "[S2]"],
            vec!["[S1]", "[S2]"],
            false,
        );
        let m = compute_generation_metrics(&[r1, r2]);
        assert!((m.citation_rate - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_verbosity_ratio_zero_query_tokens() {
        // Zero query tokens should not cause division by zero
        let results = vec![EvalResult {
            query: make_query(QueryCategory::Factual),
            response_text: "Some answer text here.".into(),
            citations_found: vec![],
            is_abstention: false,
            schema_valid: true,
            source_ids_available: vec![],
            response_token_count: 10,
            query_token_count: 0, // edge case
        }];
        let m = compute_generation_metrics(&results);
        // Should not panic; verbosity should be 0 (skipped)
        assert!((m.verbosity_ratio).abs() < 0.001);
    }

    #[test]
    fn test_answer_relevance_empty_query() {
        // Query with only stop words → no meaningful words → relevance = 1.0
        let score = answer_relevance_score("What is the?", "Anything here.");
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_answer_relevance_empty_answer() {
        let score = answer_relevance_score("What is ownership in Rust?", "");
        assert!((score).abs() < 0.001);
    }

    #[test]
    fn test_no_citations_means_full_validity() {
        // When there are no citations at all, validity defaults to 1.0
        let results = vec![make_result(
            QueryCategory::Factual,
            "No citations anywhere in this text.",
            vec![],
            vec!["[S1]"],
            false,
        )];
        let m = compute_generation_metrics(&results);
        assert!((m.citation_validity - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_sentence_citation_rate_short_sentences_ignored() {
        // Short sentences (< 3 words) should be ignored
        let rate = sentence_citation_rate("Yes. Ownership ensures safety [S1]. No.");
        // "Yes" and "No" are < 3 words, only "Ownership ensures safety [S1]" counts
        assert!((rate - 1.0).abs() < 0.001);
    }
}
