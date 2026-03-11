use cwc_core::error::Result;
use cwc_core::types::{Chunk, IssueKind, Verdict, VerificationIssue};

use crate::abstention::AbstentionDetector;
use crate::citations::{self, CitationVerifier};
use crate::schema::SchemaVerifier;

/// Configuration for the heuristic verifier.
pub struct HeuristicVerifierConfig {
    /// Fraction of factual sentences that can be uncited before failing (0.0-1.0).
    pub coverage_fail_threshold: f32,
    /// Expected output schema (None = free-form text).
    pub schema: Option<serde_json::Value>,
}

impl Default for HeuristicVerifierConfig {
    fn default() -> Self {
        Self {
            coverage_fail_threshold: 0.3,
            schema: None,
        }
    }
}

pub struct HeuristicVerifier {
    config: HeuristicVerifierConfig,
}

impl HeuristicVerifier {
    pub fn new(config: HeuristicVerifierConfig) -> Self {
        Self { config }
    }

    /// Build the set of valid source IDs from chunks (S1, S2, ...).
    fn valid_source_ids(sources: &[Chunk]) -> std::collections::HashSet<String> {
        (1..=sources.len())
            .map(|i| format!("S{i}"))
            .collect()
    }
}

impl cwc_core::traits::Verifier for HeuristicVerifier {
    fn verify(&self, output: &str, sources: &[Chunk]) -> Result<Verdict> {
        // 1. Check abstention first
        if AbstentionDetector::is_abstention(output) {
            let missing = AbstentionDetector::extract_missing_info(output);
            let reason = if missing.is_empty() {
                "model indicated insufficient evidence".to_string()
            } else {
                format!("insufficient evidence: {}", missing.join("; "))
            };
            return Ok(Verdict::Abstain { reason });
        }

        let mut all_issues = Vec::new();

        // 2. Parse as JSON and validate schema if expected
        if let Some(schema) = &self.config.schema {
            match serde_json::from_str::<serde_json::Value>(output) {
                Ok(parsed) => {
                    let schema_issues = SchemaVerifier::validate(&parsed, schema);
                    all_issues.extend(schema_issues);
                }
                Err(e) => {
                    all_issues.push(VerificationIssue {
                        kind: IssueKind::SchemaViolation,
                        description: format!("output is not valid JSON: {e}"),
                        claim_text: None,
                    });
                }
            }
        }

        // 3. Extract and validate citations
        let valid_ids = Self::valid_source_ids(sources);
        let citations = CitationVerifier::extract_citations(output);
        let validity_issues = CitationVerifier::check_validity(&citations, &valid_ids);
        all_issues.extend(validity_issues);

        // 4. Check citation coverage
        let coverage_issues = CitationVerifier::check_coverage(output, &citations);

        // Only fail on coverage if more than threshold of factual sentences are uncited
        if !coverage_issues.is_empty() {
            // Count total factual-looking sentences (uncited = coverage_issues, cited = rest)
            // We use the coverage issues count vs a rough total
            // Since coverage_issues only flags factual sentences without citations,
            // we check if the ratio is above threshold
            let total_factual = count_factual_sentences(output);
            let uncited = coverage_issues.len();

            if total_factual > 0
                && (uncited as f32 / total_factual as f32) > self.config.coverage_fail_threshold
            {
                all_issues.extend(coverage_issues);
            }
        }

        // 5. Return verdict
        if all_issues.is_empty() {
            Ok(Verdict::Pass)
        } else {
            Ok(Verdict::Fail {
                issues: all_issues,
            })
        }
    }
}

/// Count the total number of factual-looking sentences (both cited and uncited).
/// Reuses the same heuristics as `CitationVerifier::check_coverage` to ensure
/// the denominator and numerator use identical criteria.
fn count_factual_sentences(text: &str) -> usize {
    citations::split_sentences(text)
        .iter()
        .filter(|(_, _, s)| {
            if s.split_whitespace().count() < 8 {
                return false;
            }
            if s.trim_end().ends_with('?') {
                return false;
            }
            if citations::is_opinion_sentence(s) {
                return false;
            }
            citations::looks_factual(s)
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cwc_core::traits::Verifier;
    use uuid::Uuid;
    use std::collections::HashMap;

    fn make_chunks(n: usize) -> Vec<Chunk> {
        (0..n)
            .map(|i| Chunk {
                chunk_id: Uuid::new_v4(),
                doc_id: Uuid::new_v4(),
                doc_version: 1,
                source_path: format!("doc{i}.md"),
                section_path: vec![],
                char_offset: 0,
                char_len: 100,
                token_count: 50,
                text: format!("chunk {i} content"),
                metadata: HashMap::new(),
            })
            .collect()
    }

    #[test]
    fn test_heuristic_verifier_pass() {
        let verifier = HeuristicVerifier::new(HeuristicVerifierConfig::default());
        let sources = make_chunks(3);
        let output = "Rust ensures memory safety through ownership [S1]. The borrow checker prevents data races [S2].";
        let verdict = verifier.verify(output, &sources).unwrap();
        assert!(verdict.is_pass());
    }

    #[test]
    fn test_heuristic_verifier_abstention() {
        let verifier = HeuristicVerifier::new(HeuristicVerifierConfig::default());
        let sources = make_chunks(3);
        let output = "INSUFFICIENT_EVIDENCE: The sources don't contain information about quantum computing.";
        let verdict = verifier.verify(output, &sources).unwrap();
        assert!(verdict.is_abstain());
    }

    #[test]
    fn test_heuristic_verifier_invalid_citations() {
        let verifier = HeuristicVerifier::new(HeuristicVerifierConfig::default());
        let sources = make_chunks(2); // S1, S2 are valid
        let output = "The result is significant [S5]. More details in [S9].";
        let verdict = verifier.verify(output, &sources).unwrap();
        assert!(verdict.is_fail());
        let issues = verdict.issues();
        assert!(issues.iter().any(|i| i.kind == IssueKind::InvalidCitationId));
    }

    #[test]
    fn test_heuristic_verifier_schema_violation() {
        let schema = serde_json::json!({
            "type": "object",
            "required": ["answer"],
            "properties": { "answer": { "type": "string" } }
        });
        let verifier = HeuristicVerifier::new(HeuristicVerifierConfig {
            schema: Some(schema),
            ..Default::default()
        });
        let sources = make_chunks(2);
        let output = r#"{"wrong_field": "value"}"#;
        let verdict = verifier.verify(output, &sources).unwrap();
        assert!(verdict.is_fail());
        assert!(verdict.issues().iter().any(|i| i.kind == IssueKind::SchemaViolation));
    }

    #[test]
    fn test_heuristic_verifier_invalid_json_with_schema() {
        let schema = serde_json::json!({
            "type": "object",
            "required": ["answer"],
            "properties": { "answer": { "type": "string" } }
        });
        let verifier = HeuristicVerifier::new(HeuristicVerifierConfig {
            schema: Some(schema),
            ..Default::default()
        });
        let sources = make_chunks(2);
        let output = "not json at all";
        let verdict = verifier.verify(output, &sources).unwrap();
        assert!(verdict.is_fail());
        assert!(verdict.issues().iter().any(|i| i.description.contains("not valid JSON")));
    }

    #[test]
    fn test_heuristic_verifier_coverage_below_threshold() {
        let verifier = HeuristicVerifier::new(HeuristicVerifierConfig::default());
        let sources = make_chunks(3);
        // 4 factual sentences: 3 cited, 1 not → 25% uncited → pass
        let output = "The system uses 4GB of RAM for processing heavy workloads [S1]. \
                       It runs on 8 cores for parallel computation tasks [S2]. \
                       The throughput reaches 100 requests per second under load [S3]. \
                       The latency is about 50 milliseconds per request on average.";
        let verdict = verifier.verify(output, &sources).unwrap();
        // 1 out of 4 factual = 25% < 30% threshold → pass
        assert!(verdict.is_pass());
    }

    #[test]
    fn test_heuristic_verifier_coverage_above_threshold() {
        let verifier = HeuristicVerifier::new(HeuristicVerifierConfig::default());
        let sources = make_chunks(3);
        // All factual, none cited → 100% uncited > 30% → fail
        let output = "The system uses 4GB of RAM for processing heavy workloads. \
                       It runs on 8 cores for parallel computation tasks. \
                       The throughput reaches 100 requests per second under load.";
        let verdict = verifier.verify(output, &sources).unwrap();
        assert!(verdict.is_fail());
        assert!(verdict.issues().iter().any(|i| i.kind == IssueKind::MissingCitation));
    }

    #[test]
    fn test_heuristic_verifier_no_sources() {
        let verifier = HeuristicVerifier::new(HeuristicVerifierConfig::default());
        let sources: Vec<Chunk> = vec![];
        // With no sources, ANY citation is invalid
        let output = "The answer is clear [S1].";
        let verdict = verifier.verify(output, &sources).unwrap();
        assert!(verdict.is_fail());
    }

    #[test]
    fn test_heuristic_verifier_empty_output() {
        let verifier = HeuristicVerifier::new(HeuristicVerifierConfig::default());
        let sources = make_chunks(2);
        let verdict = verifier.verify("", &sources).unwrap();
        assert!(verdict.is_pass()); // Empty output, no issues to find
    }

    #[test]
    fn test_heuristic_verifier_schema_and_citations_combined() {
        // Both schema violation and invalid citation → both reported
        let schema = serde_json::json!({
            "type": "object",
            "required": ["answer"],
            "properties": { "answer": { "type": "string" } }
        });
        let verifier = HeuristicVerifier::new(HeuristicVerifierConfig {
            schema: Some(schema),
            ..Default::default()
        });
        let sources = make_chunks(1);
        let output = r#"{"wrong": "value [S9]"}"#;
        let verdict = verifier.verify(output, &sources).unwrap();
        assert!(verdict.is_fail());
        let issues = verdict.issues();
        assert!(issues.iter().any(|i| i.kind == IssueKind::SchemaViolation));
        assert!(issues.iter().any(|i| i.kind == IssueKind::InvalidCitationId));
    }

    #[test]
    fn test_valid_source_ids() {
        let chunks = make_chunks(3);
        let ids = HeuristicVerifier::valid_source_ids(&chunks);
        assert!(ids.contains("S1"));
        assert!(ids.contains("S2"));
        assert!(ids.contains("S3"));
        assert!(!ids.contains("S0"));
        assert!(!ids.contains("S4"));
    }
}
