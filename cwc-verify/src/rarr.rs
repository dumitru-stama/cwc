use std::sync::Arc;

use cwc_core::error::Result;
use cwc_core::traits::{LlmClient, Retriever};
use cwc_core::types::Chunk;

use crate::claims::Claim;

/// Result of RARR post-hoc attribution.
pub struct RarrResult {
    /// The revised text with added/removed citations.
    pub revised_text: String,
    /// Claims that received new citations: (claim_text, added_citation).
    pub added_citations: Vec<(String, String)>,
    /// Claims removed because no supporting evidence was found.
    pub removed_claims: Vec<String>,
    /// Additionally retrieved chunks used for attribution.
    pub new_sources: Vec<Chunk>,
}

pub struct RarrReviser {
    llm: Arc<dyn LlmClient>,
    retriever: Arc<dyn Retriever>,
}

impl RarrReviser {
    pub fn new(llm: Arc<dyn LlmClient>, retriever: Arc<dyn Retriever>) -> Self {
        Self { llm, retriever }
    }

    /// For each uncited factual claim, retrieve supporting evidence
    /// and either add citations or flag the claim for removal.
    ///
    /// `remaining_budget` tracks the number of LLM calls left.
    pub async fn attribute(
        &self,
        draft: &str,
        claims: &[Claim],
        existing_sources: &[Chunk],
        remaining_budget: &mut usize,
    ) -> Result<RarrResult> {
        let uncited: Vec<&Claim> = claims
            .iter()
            .filter(|c| c.is_factual && c.cited_sources.is_empty())
            .collect();

        let mut revised_text = draft.to_string();
        let mut added_citations = Vec::new();
        let mut removed_claims = Vec::new();
        let mut new_sources = Vec::new();

        // Track the next source ID to assign
        let mut next_source_id = existing_sources.len() + 1;

        for claim in &uncited {
            if *remaining_budget == 0 {
                tracing::warn!("RARR budget exhausted, skipping remaining claims");
                break;
            }

            // Step 1: Retrieve evidence for this claim
            let retrieved = self.retriever.retrieve(&claim.text, 3)?;

            if retrieved.is_empty() {
                // No evidence found — flag for removal
                removed_claims.push(claim.text.clone());
                // Remove the claim sentence from the revised text
                revised_text = revised_text.replace(&claim.text, "");
                continue;
            }

            // Step 2: Check entailment via LLM
            let evidence_text = retrieved
                .iter()
                .map(|h| h.chunk.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");

            let entailment = self
                .check_entailment(&claim.text, &evidence_text)
                .await?;
            *remaining_budget -= 1;

            match entailment {
                Entailment::Supported => {
                    // Add citation to the claim in the revised text
                    let citation = format!("[S{}]", next_source_id);
                    revised_text = revised_text.replace(
                        &claim.text,
                        &format!("{} {}", claim.text, citation),
                    );
                    added_citations.push((claim.text.clone(), citation));
                    // Add the best supporting chunk as a new source
                    new_sources.push(retrieved[0].chunk.clone());
                    next_source_id += 1;
                }
                Entailment::Contradicted => {
                    // Remove the contradicted claim
                    removed_claims.push(claim.text.clone());
                    revised_text = revised_text.replace(&claim.text, "");
                }
                Entailment::Neutral => {
                    // No clear evidence — flag for removal
                    removed_claims.push(claim.text.clone());
                    revised_text = revised_text.replace(&claim.text, "");
                }
            }
        }

        // Clean up multiple spaces from removed claims
        while revised_text.contains("  ") {
            revised_text = revised_text.replace("  ", " ");
        }
        let revised_text = revised_text.trim().to_string();

        Ok(RarrResult {
            revised_text,
            added_citations,
            removed_claims,
            new_sources,
        })
    }

    /// Check if the evidence supports, contradicts, or is neutral to the claim.
    async fn check_entailment(&self, claim: &str, evidence: &str) -> Result<Entailment> {
        let prompt = format!(
            "Evidence:\n{evidence}\n\n\
             Claim: \"{claim}\"\n\n\
             Does the evidence support, contradict, or neither support nor contradict the claim?\n\
             Answer with exactly one word: SUPPORTS, CONTRADICTS, or NEUTRAL"
        );

        let answer = self.llm.generate(&prompt, None, 16).await?;
        let answer_lower = answer.trim().to_lowercase();

        if answer_lower.contains("support") {
            Ok(Entailment::Supported)
        } else if answer_lower.contains("contradict") {
            Ok(Entailment::Contradicted)
        } else {
            Ok(Entailment::Neutral)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Entailment {
    Supported,
    Contradicted,
    Neutral,
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use cwc_core::types::{ChatMessage, RetrievalHit};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use uuid::Uuid;

    fn make_chunk(text: &str) -> Chunk {
        Chunk {
            chunk_id: Uuid::new_v4(),
            doc_id: Uuid::new_v4(),
            doc_version: 1,
            source_path: "doc.md".to_string(),
            section_path: vec![],
            char_offset: 0,
            char_len: text.len(),
            token_count: text.split_whitespace().count() as u32,
            text: text.to_string(),
            metadata: HashMap::new(),
        }
    }

    fn make_hit(text: &str) -> RetrievalHit {
        RetrievalHit {
            chunk: make_chunk(text),
            score_sparse: 0.5,
            score_dense: 0.5,
            score_fused: 0.5,
            score_rerank: 0.0,
        }
    }

    /// Retriever that returns supporting evidence.
    struct SupportingRetriever;
    impl Retriever for SupportingRetriever {
        fn retrieve(&self, _q: &str, _k: usize) -> Result<Vec<RetrievalHit>> {
            Ok(vec![make_hit(
                "Rust was created in 2010 by Graydon Hoare at Mozilla Research.",
            )])
        }
    }

    /// Retriever that returns nothing.
    struct EmptyRetriever;
    impl Retriever for EmptyRetriever {
        fn retrieve(&self, _q: &str, _k: usize) -> Result<Vec<RetrievalHit>> {
            Ok(vec![])
        }
    }

    /// LLM that says "SUPPORTS" for entailment checks.
    struct SupportsLlm;
    #[async_trait]
    impl LlmClient for SupportsLlm {
        async fn generate(
            &self,
            _prompt: &str,
            _grammar: Option<&str>,
            _max_tokens: u32,
        ) -> Result<String> {
            Ok("SUPPORTS".to_string())
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

    /// LLM that says "CONTRADICTS".
    struct ContradictLlm;
    #[async_trait]
    impl LlmClient for ContradictLlm {
        async fn generate(
            &self,
            _prompt: &str,
            _grammar: Option<&str>,
            _max_tokens: u32,
        ) -> Result<String> {
            Ok("CONTRADICTS".to_string())
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

    /// LLM that says "NEUTRAL".
    struct NeutralLlm;
    #[async_trait]
    impl LlmClient for NeutralLlm {
        async fn generate(
            &self,
            _prompt: &str,
            _grammar: Option<&str>,
            _max_tokens: u32,
        ) -> Result<String> {
            Ok("NEUTRAL".to_string())
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
            _prompt: &str,
            _grammar: Option<&str>,
            _max_tokens: u32,
        ) -> Result<String> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok("SUPPORTS".to_string())
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

    #[tokio::test]
    async fn test_rarr_adds_citation_for_supported_claim() {
        let reviser = RarrReviser::new(
            Arc::new(SupportsLlm),
            Arc::new(SupportingRetriever),
        );
        let existing_sources = vec![make_chunk("existing source text")];
        let claims = vec![Claim {
            text: "Rust was created in 2010 by Graydon Hoare at Mozilla Research.".to_string(),
            sentence_index: 0,
            is_factual: true,
            cited_sources: vec![], // uncited
        }];
        let draft = "Rust was created in 2010 by Graydon Hoare at Mozilla Research.";
        let mut budget = 5;

        let result = reviser
            .attribute(draft, &claims, &existing_sources, &mut budget)
            .await
            .unwrap();

        assert_eq!(result.added_citations.len(), 1);
        assert!(result.added_citations[0].1.contains("[S2]")); // S2 because S1 is existing
        assert!(result.revised_text.contains("[S2]"));
        assert!(result.removed_claims.is_empty());
        assert_eq!(result.new_sources.len(), 1);
    }

    #[tokio::test]
    async fn test_rarr_removes_claim_no_evidence() {
        let reviser = RarrReviser::new(
            Arc::new(SupportsLlm),
            Arc::new(EmptyRetriever), // no evidence found
        );
        let claims = vec![Claim {
            text: "The moon is made of cheese.".to_string(),
            sentence_index: 0,
            is_factual: true,
            cited_sources: vec![],
        }];
        let draft = "The moon is made of cheese. This is well known.";
        let mut budget = 5;

        let result = reviser.attribute(draft, &claims, &[], &mut budget).await.unwrap();

        assert_eq!(result.removed_claims.len(), 1);
        assert_eq!(result.removed_claims[0], "The moon is made of cheese.");
        assert!(!result.revised_text.contains("cheese"));
        assert_eq!(budget, 5); // No LLM call needed when no evidence
    }

    #[tokio::test]
    async fn test_rarr_removes_contradicted_claim() {
        let reviser = RarrReviser::new(
            Arc::new(ContradictLlm),
            Arc::new(SupportingRetriever),
        );
        let claims = vec![Claim {
            text: "Rust was created in 2020 by the Linux Foundation.".to_string(),
            sentence_index: 0,
            is_factual: true,
            cited_sources: vec![],
        }];
        let draft = "Rust was created in 2020 by the Linux Foundation. It is a great language.";
        let mut budget = 5;

        let result = reviser.attribute(draft, &claims, &[], &mut budget).await.unwrap();

        assert_eq!(result.removed_claims.len(), 1);
        assert!(!result.revised_text.contains("2020"));
        assert!(result.revised_text.contains("great language"));
    }

    #[tokio::test]
    async fn test_rarr_skips_cited_claims() {
        let counting_llm = Arc::new(CountingLlm::new());
        let reviser = RarrReviser::new(
            counting_llm.clone(),
            Arc::new(SupportingRetriever),
        );
        let claims = vec![Claim {
            text: "Rust ensures memory safety [S1].".to_string(),
            sentence_index: 0,
            is_factual: true,
            cited_sources: vec!["S1".to_string()], // already cited
        }];
        let draft = "Rust ensures memory safety [S1].";
        let mut budget = 5;

        let result = reviser.attribute(draft, &claims, &[], &mut budget).await.unwrap();

        assert!(result.added_citations.is_empty());
        assert!(result.removed_claims.is_empty());
        assert_eq!(counting_llm.count(), 0); // No LLM calls for cited claims
        assert_eq!(budget, 5);
    }

    #[tokio::test]
    async fn test_rarr_budget_exhaustion() {
        let counting_llm = Arc::new(CountingLlm::new());
        let reviser = RarrReviser::new(
            counting_llm.clone(),
            Arc::new(SupportingRetriever),
        );
        let claims = vec![
            Claim {
                text: "Rust was created in 2010 by Graydon Hoare at Mozilla Research.".to_string(),
                sentence_index: 0,
                is_factual: true,
                cited_sources: vec![],
            },
            Claim {
                text: "The borrow checker ensures memory safety without garbage collection.".to_string(),
                sentence_index: 1,
                is_factual: true,
                cited_sources: vec![],
            },
            Claim {
                text: "Rust uses LLVM for code generation and optimization in release builds.".to_string(),
                sentence_index: 2,
                is_factual: true,
                cited_sources: vec![],
            },
        ];
        let draft = "Rust was created in 2010 by Graydon Hoare at Mozilla Research. \
                      The borrow checker ensures memory safety without garbage collection. \
                      Rust uses LLVM for code generation and optimization in release builds.";
        let mut budget = 1; // Only enough for 1 entailment check

        let result = reviser.attribute(draft, &claims, &[], &mut budget).await.unwrap();

        assert_eq!(counting_llm.count(), 1);
        assert_eq!(budget, 0);
        // Only first claim processed, rest skipped
        assert_eq!(result.added_citations.len(), 1);
    }

    #[tokio::test]
    async fn test_rarr_empty_claims() {
        let reviser = RarrReviser::new(
            Arc::new(SupportsLlm),
            Arc::new(SupportingRetriever),
        );
        let draft = "Some text.";
        let mut budget = 5;

        let result = reviser.attribute(draft, &[], &[], &mut budget).await.unwrap();

        assert!(result.added_citations.is_empty());
        assert!(result.removed_claims.is_empty());
        assert_eq!(result.revised_text, "Some text.");
        assert_eq!(budget, 5);
    }

    #[tokio::test]
    async fn test_rarr_neutral_removes_claim() {
        let reviser = RarrReviser::new(
            Arc::new(NeutralLlm),
            Arc::new(SupportingRetriever),
        );
        let claims = vec![Claim {
            text: "The system has 4GB of RAM for processing heavy workloads.".to_string(),
            sentence_index: 0,
            is_factual: true,
            cited_sources: vec![],
        }];
        let draft = "The system has 4GB of RAM for processing heavy workloads. Other info here.";
        let mut budget = 5;

        let result = reviser.attribute(draft, &claims, &[], &mut budget).await.unwrap();

        assert_eq!(result.removed_claims.len(), 1);
        assert!(!result.revised_text.contains("4GB"));
    }

    #[tokio::test]
    async fn test_rarr_mixed_claims_some_supported_some_not() {
        let reviser = RarrReviser::new(
            Arc::new(SupportsLlm),
            Arc::new(SupportingRetriever),
        );
        // Two uncited factual claims, one non-factual (skipped)
        let claims = vec![
            Claim {
                text: "Rust was created in 2010 by Graydon Hoare at Mozilla Research.".to_string(),
                sentence_index: 0,
                is_factual: true,
                cited_sources: vec![],
            },
            Claim {
                text: "I think it's wonderful.".to_string(),
                sentence_index: 1,
                is_factual: false,
                cited_sources: vec![],
            },
            Claim {
                text: "The borrow checker ensures memory safety without garbage collection.".to_string(),
                sentence_index: 2,
                is_factual: true,
                cited_sources: vec![],
            },
        ];
        let draft = "Rust was created in 2010 by Graydon Hoare at Mozilla Research. I think it's wonderful. The borrow checker ensures memory safety without garbage collection.";
        let mut budget = 10;

        let result = reviser.attribute(draft, &claims, &[], &mut budget).await.unwrap();

        // Both factual uncited claims get citations, non-factual skipped
        assert_eq!(result.added_citations.len(), 2);
        assert!(result.revised_text.contains("[S1]"));
        assert!(result.revised_text.contains("[S2]"));
        assert!(result.revised_text.contains("I think it's wonderful")); // untouched
    }

    #[tokio::test]
    async fn test_rarr_duplicate_claim_text_double_replacement() {
        // Bug 4: String::replace replaces ALL occurrences — if the same claim text
        // appears twice, both get citations. This test documents the behavior.
        let reviser = RarrReviser::new(
            Arc::new(SupportsLlm),
            Arc::new(SupportingRetriever),
        );
        let claims = vec![Claim {
            text: "Rust is fast.".to_string(),
            sentence_index: 0,
            is_factual: true,
            cited_sources: vec![],
        }];
        let draft = "Rust is fast. Really, Rust is fast.";
        let mut budget = 5;

        let result = reviser.attribute(draft, &claims, &[], &mut budget).await.unwrap();

        // Both occurrences get replaced (known limitation)
        let citation_count = result.revised_text.matches("[S1]").count();
        assert_eq!(citation_count, 2, "String::replace affects all occurrences (known behavior)");
    }

    #[tokio::test]
    async fn test_rarr_skips_non_factual_claims() {
        let counting_llm = Arc::new(CountingLlm::new());
        let reviser = RarrReviser::new(
            counting_llm.clone(),
            Arc::new(SupportingRetriever),
        );
        let claims = vec![Claim {
            text: "I think this is a great approach.".to_string(),
            sentence_index: 0,
            is_factual: false, // not factual
            cited_sources: vec![],
        }];
        let draft = "I think this is a great approach.";
        let mut budget = 5;

        let result = reviser.attribute(draft, &claims, &[], &mut budget).await.unwrap();

        assert_eq!(counting_llm.count(), 0);
        assert!(result.added_citations.is_empty());
        assert!(result.removed_claims.is_empty());
    }
}
