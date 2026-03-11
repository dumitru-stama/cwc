use std::sync::Arc;

use cwc_core::error::Result;
use cwc_core::traits::{LlmClient, Retriever, Verifier};
use cwc_core::types::{Chunk, IssueKind, Verdict, VerificationIssue};
use tokio::runtime::Handle;

use crate::claims::ClaimExtractor;
use crate::cove::{CoveConfig, CoveVerifier};
use crate::heuristic::{HeuristicVerifier, HeuristicVerifierConfig};
use crate::rarr::RarrReviser;

/// Configuration for the complex verifier.
pub struct ComplexVerifyConfig {
    /// Whether to run CoVe verification (default true).
    pub run_cove: bool,
    /// Whether to run RARR attribution (default true).
    pub run_rarr: bool,
    /// Minimum consistency ratio to pass CoVe (default 0.7).
    pub cove_threshold: f32,
    /// Maximum total LLM calls for verification (default 10).
    pub max_llm_calls: usize,
    /// CoVe-specific configuration.
    pub cove_config: CoveConfig,
    /// Heuristic verifier configuration.
    pub heuristic_config: HeuristicVerifierConfig,
}

impl Default for ComplexVerifyConfig {
    fn default() -> Self {
        Self {
            run_cove: true,
            run_rarr: true,
            cove_threshold: 0.7,
            max_llm_calls: 10,
            cove_config: CoveConfig::default(),
            heuristic_config: HeuristicVerifierConfig::default(),
        }
    }
}

/// Metrics about the verification process.
pub struct VerificationMetrics {
    /// Percentage of planted hallucinations caught.
    pub hallucination_detection_rate: f32,
    /// Percentage of correct claims flagged as issues.
    pub false_positive_rate: f32,
    /// Average LLM calls per query.
    pub verification_cost: f32,
    /// Verification latency in milliseconds.
    pub verification_latency_ms: u64,
}

pub struct ComplexVerifier {
    heuristic: HeuristicVerifier,
    cove: CoveVerifier,
    rarr: RarrReviser,
    config: ComplexVerifyConfig,
}

impl ComplexVerifier {
    pub fn new(
        llm: Arc<dyn LlmClient>,
        retriever: Arc<dyn Retriever>,
        config: ComplexVerifyConfig,
    ) -> Self {
        let heuristic = HeuristicVerifier::new(HeuristicVerifierConfig {
            coverage_fail_threshold: config.heuristic_config.coverage_fail_threshold,
            schema: config.heuristic_config.schema.clone(),
        });
        let cove = CoveVerifier::new(
            Arc::clone(&llm),
            CoveConfig {
                max_questions: config.cove_config.max_questions,
                temperature: config.cove_config.temperature,
                grammar: config.cove_config.grammar.clone(),
            },
        );
        let rarr = RarrReviser::new(llm, retriever);

        Self {
            heuristic,
            cove,
            rarr,
            config,
        }
    }
}

impl Verifier for ComplexVerifier {
    fn verify(&self, output: &str, sources: &[Chunk]) -> Result<Verdict> {
        // 1. Heuristic checks (fast, always run)
        let heuristic_verdict = self.heuristic.verify(output, sources)?;

        // If heuristic fails, return immediately — no point running expensive checks
        if heuristic_verdict.is_fail() {
            return Ok(heuristic_verdict);
        }

        // If heuristic says abstain, respect that
        if heuristic_verdict.is_abstain() {
            return Ok(heuristic_verdict);
        }

        let mut remaining_budget = self.config.max_llm_calls;
        let mut all_issues: Vec<VerificationIssue> = Vec::new();

        // 2. CoVe verification (if enabled and budget allows)
        if self.config.run_cove && remaining_budget >= 2 {
            let cove_result = tokio::task::block_in_place(|| {
                Handle::current().block_on(
                    self.cove.verify_cove(output, sources, &mut remaining_budget),
                )
            })?;

            // Check consistency ratio against threshold
            if !cove_result.claims.is_empty() {
                let consistent_count = cove_result
                    .claims
                    .iter()
                    .filter(|c| c.consistent)
                    .count();
                let ratio = consistent_count as f32 / cove_result.claims.len() as f32;

                if ratio < self.config.cove_threshold {
                    // Add CoVe issues
                    for cv in &cove_result.claims {
                        if !cv.consistent {
                            all_issues.push(VerificationIssue {
                                kind: IssueKind::UnsupportedClaim,
                                description: format!(
                                    "CoVe: claim inconsistent with independent verification (confidence: {:.2})",
                                    cv.confidence
                                ),
                                claim_text: Some(cv.claim.text.clone()),
                            });
                        }
                    }
                }
            }
        }

        // 3. RARR attribution (if enabled, CoVe found issues, and budget allows)
        if self.config.run_rarr && !all_issues.is_empty() && remaining_budget > 0 {
            let claims = ClaimExtractor::extract_claims(output);

            let rarr_result = tokio::task::block_in_place(|| {
                Handle::current().block_on(
                    self.rarr
                        .attribute(output, &claims, sources, &mut remaining_budget),
                )
            })?;

            // Add issues for removed claims
            for removed in &rarr_result.removed_claims {
                all_issues.push(VerificationIssue {
                    kind: IssueKind::UnsupportedClaim,
                    description: "RARR: claim removed due to lack of supporting evidence"
                        .to_string(),
                    claim_text: Some(removed.clone()),
                });
            }
        }

        // 4. Return final verdict
        if all_issues.is_empty() {
            Ok(Verdict::Pass)
        } else {
            Ok(Verdict::Fail {
                issues: all_issues,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use cwc_core::types::{ChatMessage, RetrievalHit};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use uuid::Uuid;

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
                text: format!(
                    "Rust uses a borrow checker at compile time for memory safety. Source chunk {i}."
                ),
                metadata: HashMap::new(),
            })
            .collect()
    }

    struct FakeRetriever;
    impl Retriever for FakeRetriever {
        fn retrieve(&self, _q: &str, _k: usize) -> Result<Vec<RetrievalHit>> {
            Ok(vec![])
        }
    }

    /// LLM that returns consistent CoVe answers (claims supported).
    struct ConsistentLlm;
    #[async_trait]
    impl LlmClient for ConsistentLlm {
        async fn generate(
            &self,
            prompt: &str,
            _grammar: Option<&str>,
            _max_tokens: u32,
        ) -> Result<String> {
            if prompt.contains("Generate a focused verification question") {
                Ok("Does Rust use a borrow checker at compile time?".to_string())
            } else if prompt.contains("Answer this question") {
                Ok("Yes, Rust uses a borrow checker at compile time for memory safety."
                    .to_string())
            } else if prompt.contains("SUPPORTS, CONTRADICTS, or NEUTRAL") {
                Ok("SUPPORTS".to_string())
            } else {
                Ok("consistent answer".to_string())
            }
        }
        async fn generate_chat(
            &self,
            _msgs: &[ChatMessage],
            _g: Option<&str>,
            _m: u32,
        ) -> Result<String> {
            Ok(String::new())
        }
    }

    /// LLM that produces contradicting CoVe answers (hallucination detected).
    struct HallucinationDetectingLlm;
    #[async_trait]
    impl LlmClient for HallucinationDetectingLlm {
        async fn generate(
            &self,
            prompt: &str,
            _grammar: Option<&str>,
            _max_tokens: u32,
        ) -> Result<String> {
            if prompt.contains("Generate a focused verification question") {
                Ok("Is this claim accurate based on the sources?".to_string())
            } else if prompt.contains("Answer this question") {
                Ok("There is not enough information in the sources to verify this claim."
                    .to_string())
            } else if prompt.contains("SUPPORTS, CONTRADICTS, or NEUTRAL") {
                Ok("NEUTRAL".to_string())
            } else {
                Ok(String::new())
            }
        }
        async fn generate_chat(
            &self,
            _msgs: &[ChatMessage],
            _g: Option<&str>,
            _m: u32,
        ) -> Result<String> {
            Ok(String::new())
        }
    }

    /// Counting LLM for budget tests.
    struct CountingLlm {
        count: AtomicUsize,
    }
    impl CountingLlm {
        fn new() -> Self {
            Self {
                count: AtomicUsize::new(0),
            }
        }
        fn count(&self) -> usize {
            self.count.load(Ordering::SeqCst)
        }
    }
    #[async_trait]
    impl LlmClient for CountingLlm {
        async fn generate(
            &self,
            prompt: &str,
            _grammar: Option<&str>,
            _max_tokens: u32,
        ) -> Result<String> {
            self.count.fetch_add(1, Ordering::SeqCst);
            if prompt.contains("Generate a focused verification question") {
                Ok("Is this claim correct?".to_string())
            } else if prompt.contains("Answer this question") {
                Ok("Yes, the borrow checker ensures memory safety at compile time.".to_string())
            } else {
                Ok("SUPPORTS".to_string())
            }
        }
        async fn generate_chat(
            &self,
            _msgs: &[ChatMessage],
            _g: Option<&str>,
            _m: u32,
        ) -> Result<String> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(String::new())
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_complex_verifier_runs_heuristic_first() {
        // Output with invalid citations should fail at heuristic stage
        let verifier = ComplexVerifier::new(
            Arc::new(ConsistentLlm),
            Arc::new(FakeRetriever),
            ComplexVerifyConfig::default(),
        );
        let sources = make_chunks(2);
        let output = "The result is significant [S9]. More details in [S10].";
        let verdict = verifier.verify(output, &sources).unwrap();
        assert!(verdict.is_fail());
        assert!(verdict
            .issues()
            .iter()
            .any(|i| i.kind == IssueKind::InvalidCitationId));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_complex_verifier_heuristic_fail_skips_cove() {
        let counting_llm = Arc::new(CountingLlm::new());
        let verifier = ComplexVerifier::new(
            counting_llm.clone(),
            Arc::new(FakeRetriever),
            ComplexVerifyConfig::default(),
        );
        let sources = make_chunks(2);
        // Invalid citation → heuristic fails → no CoVe calls
        let output = "The answer is very clear [S99].";
        let _ = verifier.verify(output, &sources);
        assert_eq!(counting_llm.count(), 0, "no LLM calls when heuristic fails");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_complex_verifier_cove_passes_consistent() {
        let verifier = ComplexVerifier::new(
            Arc::new(ConsistentLlm),
            Arc::new(FakeRetriever),
            ComplexVerifyConfig::default(),
        );
        let sources = make_chunks(3);
        let output =
            "Rust uses a borrow checker at compile time for memory safety [S1].";
        let verdict = verifier.verify(output, &sources).unwrap();
        assert!(verdict.is_pass());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_complex_verifier_detects_hallucination() {
        let verifier = ComplexVerifier::new(
            Arc::new(HallucinationDetectingLlm),
            Arc::new(FakeRetriever),
            ComplexVerifyConfig {
                cove_threshold: 0.7,
                run_rarr: false, // disable RARR to test CoVe only
                ..Default::default()
            },
        );
        let sources = make_chunks(3);
        // This claim should be flagged by CoVe (LLM says "not enough information")
        let output = "Rust uses a borrow checker at compile time for memory safety [S1].";
        let verdict = verifier.verify(output, &sources).unwrap();
        assert!(verdict.is_fail());
        assert!(verdict
            .issues()
            .iter()
            .any(|i| i.kind == IssueKind::UnsupportedClaim));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_complex_verifier_correct_claim_not_flagged() {
        let verifier = ComplexVerifier::new(
            Arc::new(ConsistentLlm),
            Arc::new(FakeRetriever),
            ComplexVerifyConfig::default(),
        );
        let sources = make_chunks(3);
        let output =
            "Rust uses a borrow checker at compile time for memory safety [S1].";
        let verdict = verifier.verify(output, &sources).unwrap();
        // Consistent LLM → should pass (no false positive)
        assert!(verdict.is_pass());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_complex_verifier_budget_respected() {
        let counting_llm = Arc::new(CountingLlm::new());
        let verifier = ComplexVerifier::new(
            counting_llm.clone(),
            Arc::new(FakeRetriever),
            ComplexVerifyConfig {
                max_llm_calls: 4,
                ..Default::default()
            },
        );
        let sources = make_chunks(3);
        let output = "Rust was created in 2010 by Graydon Hoare at Mozilla Research [S1]. \
                       The borrow checker ensures memory safety without garbage collection [S2]. \
                       Rust uses LLVM for code generation and optimization in release builds [S3]. \
                       The ownership system prevents data races at compile time automatically [S1].";
        let _ = verifier.verify(output, &sources);
        assert!(
            counting_llm.count() <= 4,
            "LLM calls ({}) should not exceed budget (4)",
            counting_llm.count()
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_complex_verifier_budget_exhausted_graceful() {
        let verifier = ComplexVerifier::new(
            Arc::new(ConsistentLlm),
            Arc::new(FakeRetriever),
            ComplexVerifyConfig {
                max_llm_calls: 0, // no budget at all
                ..Default::default()
            },
        );
        let sources = make_chunks(3);
        let output =
            "Rust uses a borrow checker at compile time for memory safety [S1].";
        // Should not crash with zero budget
        let verdict = verifier.verify(output, &sources).unwrap();
        // With zero budget, CoVe is skipped → heuristic pass → overall pass
        assert!(verdict.is_pass());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_complex_verifier_abstention_respected() {
        let verifier = ComplexVerifier::new(
            Arc::new(ConsistentLlm),
            Arc::new(FakeRetriever),
            ComplexVerifyConfig::default(),
        );
        let sources = make_chunks(2);
        let output =
            "INSUFFICIENT_EVIDENCE: The sources don't contain information about quantum computing.";
        let verdict = verifier.verify(output, &sources).unwrap();
        assert!(verdict.is_abstain());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_complex_verifier_cove_disabled() {
        let counting_llm = Arc::new(CountingLlm::new());
        let verifier = ComplexVerifier::new(
            counting_llm.clone(),
            Arc::new(FakeRetriever),
            ComplexVerifyConfig {
                run_cove: false,
                run_rarr: false,
                ..Default::default()
            },
        );
        let sources = make_chunks(3);
        let output =
            "Rust uses a borrow checker at compile time for memory safety [S1].";
        let verdict = verifier.verify(output, &sources).unwrap();
        assert!(verdict.is_pass());
        assert_eq!(counting_llm.count(), 0, "no LLM calls when CoVe disabled");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_complex_verifier_max_questions_limits_claims() {
        let counting_llm = Arc::new(CountingLlm::new());
        let verifier = ComplexVerifier::new(
            counting_llm.clone(),
            Arc::new(FakeRetriever),
            ComplexVerifyConfig {
                max_llm_calls: 20, // plenty of budget
                cove_config: CoveConfig {
                    max_questions: 1, // only verify 1 claim
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        let sources = make_chunks(3);
        // 4 factual claims but max_questions = 1
        let output = "Rust was created in 2010 by Graydon Hoare at Mozilla Research [S1]. \
                       The borrow checker ensures memory safety without garbage collection [S2]. \
                       Rust uses LLVM for code generation and optimization in release builds [S3]. \
                       The ownership system prevents data races at compile time automatically [S1].";
        let _ = verifier.verify(output, &sources);
        // max_questions=1 → only 1 claim verified → 2 LLM calls (question + answer)
        assert_eq!(counting_llm.count(), 2, "max_questions=1 should limit to 2 LLM calls");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_complex_verifier_empty_output() {
        let verifier = ComplexVerifier::new(
            Arc::new(ConsistentLlm),
            Arc::new(FakeRetriever),
            ComplexVerifyConfig::default(),
        );
        let sources = make_chunks(2);
        let verdict = verifier.verify("", &sources).unwrap();
        assert!(verdict.is_pass());
    }
}
