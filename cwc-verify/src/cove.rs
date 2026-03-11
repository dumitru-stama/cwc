use std::sync::Arc;

use cwc_core::error::Result;
use cwc_core::traits::LlmClient;
use cwc_core::types::{Chunk, Verdict, VerificationIssue, IssueKind};

use crate::claims::{Claim, ClaimExtractor};

/// Configuration for the CoVe verifier.
pub struct CoveConfig {
    /// Max verification questions per response (default 5).
    pub max_questions: usize,
    /// Temperature for verification LLM calls (low: 0.0-0.1).
    pub temperature: f32,
    /// Optional grammar constraint for verification output.
    pub grammar: Option<String>,
}

impl Default for CoveConfig {
    fn default() -> Self {
        Self {
            max_questions: 5,
            temperature: 0.0,
            grammar: None,
        }
    }
}

/// Result of CoVe verification.
pub struct CoveResult {
    pub claims: Vec<ClaimVerification>,
    pub overall: Verdict,
}

/// Verification result for a single claim.
pub struct ClaimVerification {
    pub claim: Claim,
    pub question: String,
    pub independent_answer: String,
    pub consistent: bool,
    pub confidence: f32,
}

pub struct CoveVerifier {
    llm: Arc<dyn LlmClient>,
    config: CoveConfig,
}

impl CoveVerifier {
    pub fn new(
        llm: Arc<dyn LlmClient>,
        config: CoveConfig,
    ) -> Self {
        Self {
            llm,
            config,
        }
    }

    /// Full CoVe pipeline:
    /// 1. Extract claims from draft response
    /// 2. Generate verification questions for factual claims
    /// 3. Answer each question independently (fresh LLM call, no draft context)
    /// 4. Compare verification answers to original claims
    /// 5. Flag inconsistencies
    pub async fn verify_cove(
        &self,
        draft: &str,
        sources: &[Chunk],
        remaining_budget: &mut usize,
    ) -> Result<CoveResult> {
        let factual_claims = ClaimExtractor::extract_factual_claims(draft);

        // Limit to max_questions
        let claims_to_verify: Vec<Claim> = factual_claims
            .into_iter()
            .take(self.config.max_questions)
            .collect();

        let source_context = format_sources(sources);
        let mut verifications = Vec::new();

        for claim in claims_to_verify {
            // Need at least 2 calls: question gen + independent answer
            if *remaining_budget < 2 {
                tracing::warn!("CoVe budget exhausted after {} verifications", verifications.len());
                break;
            }

            // Step 2: Generate verification question
            let question = self.generate_question(&claim).await?;
            *remaining_budget -= 1;

            // Step 3: Answer independently (without seeing the draft)
            let independent_answer = self
                .answer_independently(&question, &source_context)
                .await?;
            *remaining_budget -= 1;

            // Step 4: Check consistency
            let (consistent, confidence) =
                check_consistency(&claim.text, &independent_answer);

            verifications.push(ClaimVerification {
                claim,
                question,
                independent_answer,
                consistent,
                confidence,
            });
        }

        // Build overall verdict
        let overall = build_verdict(&verifications);

        Ok(CoveResult {
            claims: verifications,
            overall,
        })
    }

    /// Generate a verification question for a factual claim.
    async fn generate_question(&self, claim: &Claim) -> Result<String> {
        let prompt = format!(
            "Given this factual claim: \"{}\"\n\
             Generate a focused verification question that can be answered from source documents.\n\
             Output ONLY the question, nothing else.\n\
             Question:",
            claim.text
        );

        let answer = self
            .llm
            .generate(&prompt, self.config.grammar.as_deref(), 128)
            .await?;

        Ok(answer.trim().to_string())
    }

    /// Answer a verification question independently using only the sources.
    async fn answer_independently(
        &self,
        question: &str,
        source_context: &str,
    ) -> Result<String> {
        let prompt = format!(
            "{source_context}\n\n\
             Answer this question using ONLY the provided sources. \
             If the sources don't contain enough information, say \"not enough information\".\n\n\
             Question: {question}\n\
             Answer:",
        );

        let answer = self
            .llm
            .generate(&prompt, self.config.grammar.as_deref(), 256)
            .await?;

        Ok(answer.trim().to_string())
    }
}

/// Format source chunks for inclusion in a verification prompt.
fn format_sources(sources: &[Chunk]) -> String {
    if sources.is_empty() {
        return String::from("[SOURCES]\nNo sources provided.");
    }

    let mut out = String::from("[SOURCES]\n");
    for (i, chunk) in sources.iter().enumerate() {
        out.push_str(&format!("[S{}] {}\n\n", i + 1, chunk.text));
    }
    out
}

/// Check if an independent answer is consistent with the original claim.
///
/// Returns (consistent, confidence).
/// Uses simple heuristics: word overlap, contradiction markers, "not enough info".
fn check_consistency(claim_text: &str, answer: &str) -> (bool, f32) {
    let answer_lower = answer.to_lowercase();

    // If the answer says there isn't enough information, that's inconsistent
    let insufficient_markers = [
        "not enough information",
        "insufficient information",
        "cannot determine",
        "cannot be determined",
        "no information",
        "not mentioned",
        "not found in the sources",
        "the sources do not",
        "the sources don't",
    ];

    for marker in &insufficient_markers {
        if answer_lower.contains(marker) {
            return (false, 0.8);
        }
    }

    // Check for strong contradiction markers (unambiguously negative)
    let strong_contradiction_markers = [
        "incorrect",
        "not true",
        "false",
        "wrong",
        "contradicts",
        "on the contrary",
    ];

    // Weak markers that are only contradictions when paired with negation words
    let weak_contradiction_markers = ["actually", "in fact", "however"];
    let negation_words = ["not", "no", "never", "neither", "nor", "doesn't", "don't", "isn't", "aren't", "wasn't", "weren't"];

    let has_strong_contradiction = strong_contradiction_markers
        .iter()
        .any(|m| answer_lower.contains(m));

    let has_weak_with_negation = weak_contradiction_markers
        .iter()
        .any(|m| answer_lower.contains(m))
        && negation_words.iter().any(|n| {
            answer_lower.split_whitespace().any(|w| {
                let stripped = w.trim_end_matches(|c: char| c.is_ascii_punctuation());
                stripped == *n
            })
        });

    if has_strong_contradiction || has_weak_with_negation {
        // Check if the contradiction references terms from the claim
        let claim_lower = claim_text.to_lowercase();
        let claim_words: std::collections::HashSet<String> = claim_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .map(|w| w.to_string())
            .collect();

        let answer_words: std::collections::HashSet<String> = answer_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .map(|w| w.to_string())
            .collect();

        let overlap: usize = claim_words.intersection(&answer_words).count();

        if overlap > 0 {
            return (false, 0.7);
        }
    }

    // Positive: check word overlap between claim and answer
    let claim_lower = claim_text.to_lowercase();
    let claim_words: std::collections::HashSet<String> = claim_lower
        .split_whitespace()
        .filter(|w| w.len() > 3)
        .map(|w| w.to_string())
        .collect();

    let answer_words: std::collections::HashSet<String> = answer_lower
        .split_whitespace()
        .filter(|w| w.len() > 3)
        .map(|w| w.to_string())
        .collect();

    if claim_words.is_empty() {
        return (true, 0.5);
    }

    let overlap = claim_words.intersection(&answer_words).count();
    let overlap_ratio = overlap as f32 / claim_words.len() as f32;

    if overlap_ratio >= 0.3 {
        (true, 0.5 + overlap_ratio * 0.5)
    } else {
        // Low overlap — answer doesn't seem to address the claim
        (false, 0.4)
    }
}

/// Build a Verdict from the list of claim verifications.
fn build_verdict(verifications: &[ClaimVerification]) -> Verdict {
    if verifications.is_empty() {
        return Verdict::Pass;
    }

    let issues: Vec<VerificationIssue> = verifications
        .iter()
        .filter(|v| !v.consistent)
        .map(|v| VerificationIssue {
            kind: IssueKind::UnsupportedClaim,
            description: format!(
                "claim inconsistent with independent verification (confidence: {:.2}): \"{}\"",
                v.confidence, v.claim.text
            ),
            claim_text: Some(v.claim.text.clone()),
        })
        .collect();

    if issues.is_empty() {
        Verdict::Pass
    } else {
        Verdict::Fail { issues }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use cwc_core::types::ChatMessage;
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
                text: format!("Rust uses a borrow checker at compile time for memory safety. Source chunk {i}."),
                metadata: HashMap::new(),
            })
            .collect()
    }

    /// Mock LLM that returns consistent answers (supports the claims).
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
                Ok("Does Rust use a borrow checker at compile time for memory safety?".to_string())
            } else if prompt.contains("Answer this question") {
                Ok("Yes, Rust uses a borrow checker at compile time to ensure memory safety without garbage collection.".to_string())
            } else {
                Ok("unknown prompt type".to_string())
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

    /// Mock LLM that returns contradicting answers.
    struct ContradictingLlm;
    #[async_trait]
    impl LlmClient for ContradictingLlm {
        async fn generate(
            &self,
            prompt: &str,
            _grammar: Option<&str>,
            _max_tokens: u32,
        ) -> Result<String> {
            if prompt.contains("Generate a focused verification question") {
                Ok("Does Rust use a borrow checker at compile time?".to_string())
            } else if prompt.contains("Answer this question") {
                Ok("This is not true. The borrow checker is actually wrong about this claim.".to_string())
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

    /// Mock LLM that says "not enough information".
    struct InsufficientInfoLlm;
    #[async_trait]
    impl LlmClient for InsufficientInfoLlm {
        async fn generate(
            &self,
            prompt: &str,
            _grammar: Option<&str>,
            _max_tokens: u32,
        ) -> Result<String> {
            if prompt.contains("Generate a focused verification question") {
                Ok("Does Rust use LLVM for code generation?".to_string())
            } else if prompt.contains("Answer this question") {
                Ok("There is not enough information in the sources to determine this.".to_string())
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

    /// Mock LLM that counts calls.
    struct CountingLlm {
        call_count: AtomicUsize,
    }
    impl CountingLlm {
        fn new() -> Self {
            Self {
                call_count: AtomicUsize::new(0),
            }
        }
        fn count(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
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
            self.call_count.fetch_add(1, Ordering::SeqCst);
            if prompt.contains("Generate a focused verification question") {
                Ok("Is this claim true based on the sources?".to_string())
            } else {
                Ok("Yes, the sources confirm this claim about the borrow checker.".to_string())
            }
        }
        async fn generate_chat(
            &self,
            _msgs: &[ChatMessage],
            _g: Option<&str>,
            _m: u32,
        ) -> Result<String> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Ok(String::new())
        }
    }

    #[tokio::test]
    async fn test_cove_question_generation() {
        let verifier = CoveVerifier::new(
            Arc::new(ConsistentLlm),
            CoveConfig::default(),
        );
        let sources = make_chunks(2);
        let draft = "Rust uses a borrow checker at compile time for memory safety [S1].";
        let mut budget = 10;
        let result = verifier.verify_cove(draft, &sources, &mut budget).await.unwrap();

        // Should have generated questions for factual claims
        assert!(!result.claims.is_empty());
        for cv in &result.claims {
            assert!(!cv.question.is_empty(), "question should not be empty");
        }
    }

    #[tokio::test]
    async fn test_cove_independent_answer_uses_sources() {
        let verifier = CoveVerifier::new(
            Arc::new(ConsistentLlm),
            CoveConfig::default(),
        );
        let sources = make_chunks(2);
        let draft = "Rust uses a borrow checker at compile time for memory safety [S1].";
        let mut budget = 10;
        let result = verifier.verify_cove(draft, &sources, &mut budget).await.unwrap();

        assert!(!result.claims.is_empty());
        for cv in &result.claims {
            assert!(!cv.independent_answer.is_empty(), "answer should not be empty");
        }
    }

    #[tokio::test]
    async fn test_cove_consistent_claim_passes() {
        let verifier = CoveVerifier::new(
            Arc::new(ConsistentLlm),
            CoveConfig::default(),
        );
        let sources = make_chunks(2);
        let draft = "Rust uses a borrow checker at compile time for memory safety [S1].";
        let mut budget = 10;
        let result = verifier.verify_cove(draft, &sources, &mut budget).await.unwrap();

        assert!(!result.claims.is_empty());
        assert!(result.claims[0].consistent, "consistent answer should mark claim as consistent");
        assert!(result.overall.is_pass());
    }

    #[tokio::test]
    async fn test_cove_contradicting_answer_fails() {
        let verifier = CoveVerifier::new(
            Arc::new(ContradictingLlm),
            CoveConfig::default(),
        );
        let sources = make_chunks(2);
        let draft = "Rust uses a borrow checker at compile time for memory safety [S1].";
        let mut budget = 10;
        let result = verifier.verify_cove(draft, &sources, &mut budget).await.unwrap();

        assert!(!result.claims.is_empty());
        assert!(!result.claims[0].consistent, "contradicting answer should mark claim as inconsistent");
        assert!(result.overall.is_fail());
        assert!(result.overall.issues().iter().any(|i| i.kind == IssueKind::UnsupportedClaim));
    }

    #[tokio::test]
    async fn test_cove_insufficient_info_inconsistent() {
        let verifier = CoveVerifier::new(
            Arc::new(InsufficientInfoLlm),
            CoveConfig::default(),
        );
        let sources = make_chunks(2);
        let draft = "Rust uses LLVM for code generation and optimization tasks.";
        let mut budget = 10;
        let result = verifier.verify_cove(draft, &sources, &mut budget).await.unwrap();

        assert!(!result.claims.is_empty());
        assert!(!result.claims[0].consistent, "'not enough information' should be inconsistent");
    }

    #[tokio::test]
    async fn test_cove_budget_limits_calls() {
        let counting_llm = Arc::new(CountingLlm::new());
        let verifier = CoveVerifier::new(
            counting_llm.clone(),
            CoveConfig {
                max_questions: 10, // high limit
                ..Default::default()
            },
        );
        let sources = make_chunks(3);
        // Draft with multiple factual claims
        let draft = "Rust was created in 2010 by Graydon Hoare at Mozilla Research. \
                      The borrow checker ensures memory safety without garbage collection. \
                      Rust uses LLVM for code generation and optimization in release builds. \
                      The ownership system prevents data races at compile time automatically.";
        let mut budget = 4; // Only enough for 2 claims (2 calls each)
        let result = verifier.verify_cove(draft, &sources, &mut budget).await.unwrap();

        // Should have verified at most 2 claims (4 budget / 2 calls per claim)
        assert!(result.claims.len() <= 2);
        assert_eq!(counting_llm.count(), result.claims.len() * 2);
        assert_eq!(budget, 0);
    }

    #[tokio::test]
    async fn test_cove_no_factual_claims_passes() {
        let verifier = CoveVerifier::new(
            Arc::new(ConsistentLlm),
            CoveConfig::default(),
        );
        let sources = make_chunks(2);
        let draft = "I think this is great. Maybe it works well. Perhaps it could be better.";
        let mut budget = 10;
        let result = verifier.verify_cove(draft, &sources, &mut budget).await.unwrap();

        assert!(result.claims.is_empty());
        assert!(result.overall.is_pass());
        assert_eq!(budget, 10); // No budget consumed
    }

    #[tokio::test]
    async fn test_cove_empty_draft_passes() {
        let verifier = CoveVerifier::new(
            Arc::new(ConsistentLlm),
            CoveConfig::default(),
        );
        let sources = make_chunks(1);
        let mut budget = 10;
        let result = verifier.verify_cove("", &sources, &mut budget).await.unwrap();

        assert!(result.claims.is_empty());
        assert!(result.overall.is_pass());
    }

    #[test]
    fn test_check_consistency_matching_words() {
        let (consistent, confidence) = check_consistency(
            "Rust uses a borrow checker at compile time",
            "Yes, Rust does use a borrow checker at compile time for safety",
        );
        assert!(consistent);
        assert!(confidence > 0.5);
    }

    #[test]
    fn test_check_consistency_contradiction() {
        let (consistent, _) = check_consistency(
            "Rust uses a borrow checker at compile time",
            "This is not true, Rust does not actually use a borrow checker",
        );
        assert!(!consistent);
    }

    #[test]
    fn test_check_consistency_not_enough_info() {
        let (consistent, confidence) = check_consistency(
            "Rust was created in 2010",
            "There is not enough information to verify this.",
        );
        assert!(!consistent);
        assert!(confidence > 0.5);
    }

    #[test]
    fn test_check_consistency_low_overlap() {
        let (consistent, _) = check_consistency(
            "Rust uses LLVM for code generation",
            "Python is an interpreted language used for data science",
        );
        assert!(!consistent);
    }

    #[test]
    fn test_format_sources_empty() {
        let sources = format_sources(&[]);
        assert!(sources.contains("No sources provided"));
    }

    #[test]
    fn test_format_sources_numbered() {
        let chunks = make_chunks(3);
        let formatted = format_sources(&chunks);
        assert!(formatted.contains("[S1]"));
        assert!(formatted.contains("[S2]"));
        assert!(formatted.contains("[S3]"));
    }

    #[test]
    fn test_build_verdict_all_consistent() {
        let verifications = vec![ClaimVerification {
            claim: Claim {
                text: "test claim".to_string(),
                sentence_index: 0,
                is_factual: true,
                cited_sources: vec![],
            },
            question: "Is this true?".to_string(),
            independent_answer: "Yes".to_string(),
            consistent: true,
            confidence: 0.9,
        }];
        let verdict = build_verdict(&verifications);
        assert!(verdict.is_pass());
    }

    #[test]
    fn test_build_verdict_inconsistent() {
        let verifications = vec![ClaimVerification {
            claim: Claim {
                text: "false claim".to_string(),
                sentence_index: 0,
                is_factual: true,
                cited_sources: vec![],
            },
            question: "Is this true?".to_string(),
            independent_answer: "No".to_string(),
            consistent: false,
            confidence: 0.7,
        }];
        let verdict = build_verdict(&verifications);
        assert!(verdict.is_fail());
        assert!(verdict.issues().iter().any(|i| i.kind == IssueKind::UnsupportedClaim));
    }

    #[test]
    fn test_build_verdict_empty() {
        let verdict = build_verdict(&[]);
        assert!(verdict.is_pass());
    }

    #[test]
    fn test_check_consistency_confirming_actually_not_false_positive() {
        // "actually" used in a confirming context should NOT be flagged as contradiction
        let (consistent, _) = check_consistency(
            "Rust uses a borrow checker at compile time",
            "Yes, Rust actually does use a borrow checker at compile time for memory safety.",
        );
        assert!(consistent, "confirming 'actually' should not be a false positive");
    }

    #[test]
    fn test_check_consistency_however_confirming_not_false_positive() {
        // "however" used without negation should not be a contradiction
        let (consistent, _) = check_consistency(
            "Rust ensures memory safety without garbage collection",
            "Rust ensures memory safety; however, it does this through its ownership system.",
        );
        assert!(consistent, "confirming 'however' without negation should not be a false positive");
    }

    #[tokio::test]
    async fn test_cove_empty_answer_from_llm() {
        /// LLM that returns empty strings for answers
        struct EmptyAnswerLlm;
        #[async_trait]
        impl LlmClient for EmptyAnswerLlm {
            async fn generate(
                &self,
                prompt: &str,
                _grammar: Option<&str>,
                _max_tokens: u32,
            ) -> Result<String> {
                if prompt.contains("Generate a focused verification question") {
                    Ok("Is this claim true?".to_string())
                } else {
                    Ok("".to_string()) // empty answer
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

        let verifier = CoveVerifier::new(Arc::new(EmptyAnswerLlm), CoveConfig::default());
        let sources = make_chunks(1);
        let draft = "Rust uses a borrow checker at compile time for memory safety.";
        let mut budget = 10;
        let result = verifier.verify_cove(draft, &sources, &mut budget).await.unwrap();

        // Empty answer has zero word overlap → low overlap → inconsistent
        assert!(!result.claims.is_empty());
        assert!(!result.claims[0].consistent, "empty answer should be inconsistent (low overlap)");
    }

    #[test]
    fn test_check_consistency_negation_with_trailing_punctuation() {
        // "not," with trailing comma should still be detected as negation
        let (consistent, _) = check_consistency(
            "Rust uses a borrow checker at compile time",
            "However, that's not, strictly speaking, how the borrow checker works in Rust.",
        );
        assert!(!consistent, "negation word with trailing punctuation should be detected");
    }
}
