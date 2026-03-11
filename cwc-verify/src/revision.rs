use std::sync::Arc;

use cwc_core::error::Result;
use cwc_core::traits::{LlmClient, Verifier};
use cwc_core::types::{ChatMessage, Chunk, Role, Verdict};

/// Result of a revision attempt.
pub struct RevisionResult {
    pub output: String,
    pub verdict: Verdict,
    pub attempts: usize,
}

pub struct RevisionLoop {
    llm: Arc<dyn LlmClient>,
    verifier: Arc<dyn Verifier>,
    max_attempts: usize,
}

impl RevisionLoop {
    pub fn new(
        llm: Arc<dyn LlmClient>,
        verifier: Arc<dyn Verifier>,
        max_attempts: usize,
    ) -> Self {
        Self {
            llm,
            verifier,
            max_attempts,
        }
    }

    /// Build a revision prompt from the failed verdict.
    pub fn build_revision_prompt(verdict: &Verdict) -> String {
        let mut prompt = String::from("Your previous response had the following issues:\n");

        for issue in verdict.issues() {
            prompt.push_str(&format!("- {}\n", issue.description));
        }

        prompt.push_str(
            "\nPlease revise your response to fix these issues.\n\
             Keep all other content unchanged. Output the complete revised response.",
        );

        prompt
    }

    /// Attempt to fix a failed verification by asking the LLM to revise.
    pub async fn revise(
        &self,
        original_messages: &[ChatMessage],
        draft: &str,
        verdict: &Verdict,
        sources: &[Chunk],
        grammar: Option<&str>,
        max_tokens: u32,
    ) -> Result<RevisionResult> {
        let mut best_output = draft.to_string();
        let mut best_verdict = verdict.clone();
        let mut attempts = 0;

        for _ in 0..self.max_attempts {
            attempts += 1;

            let revision_prompt = Self::build_revision_prompt(&best_verdict);

            // Build messages: original context + draft as assistant + revision request
            let mut messages = original_messages.to_vec();
            messages.push(ChatMessage {
                role: Role::Assistant,
                content: best_output.clone(),
            });
            messages.push(ChatMessage {
                role: Role::User,
                content: revision_prompt,
            });

            let revised = self
                .llm
                .generate_chat(&messages, grammar, max_tokens)
                .await?;

            let new_verdict = self.verifier.verify(&revised, sources)?;

            best_output = revised;
            best_verdict = new_verdict;

            if best_verdict.is_pass() || best_verdict.is_abstain() {
                break;
            }
        }

        Ok(RevisionResult {
            output: best_output,
            verdict: best_verdict,
            attempts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_revision_prompt_includes_issues() {
        use cwc_core::types::{IssueKind, VerificationIssue};

        let verdict = Verdict::Fail {
            issues: vec![
                VerificationIssue {
                    kind: IssueKind::InvalidCitationId,
                    description: "[S7] is not a valid source ID. Valid: [S1], [S2]".to_string(),
                    claim_text: None,
                },
                VerificationIssue {
                    kind: IssueKind::SchemaViolation,
                    description: "missing required field: confidence".to_string(),
                    claim_text: None,
                },
            ],
        };

        let prompt = RevisionLoop::build_revision_prompt(&verdict);
        assert!(prompt.contains("[S7] is not a valid source ID"));
        assert!(prompt.contains("missing required field: confidence"));
        assert!(prompt.contains("Please revise your response"));
    }

    #[test]
    fn test_revision_prompt_empty_on_pass() {
        let verdict = Verdict::Pass;
        let prompt = RevisionLoop::build_revision_prompt(&verdict);
        // No issues listed, but the structure is still there
        assert!(prompt.contains("Please revise"));
        assert!(!prompt.contains("- ")); // no issue bullets
    }

    #[tokio::test]
    async fn test_revision_loop_max_attempts() {
        use cwc_core::types::{IssueKind, VerificationIssue};
        use std::collections::HashMap;
        use uuid::Uuid;

        // Mock LLM that always returns the same bad output
        struct MockLlm;
        #[async_trait::async_trait]
        impl LlmClient for MockLlm {
            async fn generate(
                &self, _prompt: &str, _grammar: Option<&str>, _max_tokens: u32,
            ) -> Result<String> {
                Ok("bad output [S99]".to_string())
            }
            async fn generate_chat(
                &self, _messages: &[ChatMessage], _grammar: Option<&str>, _max_tokens: u32,
            ) -> Result<String> {
                Ok("bad output [S99]".to_string())
            }
        }

        // Mock verifier that always fails
        struct MockVerifier;
        impl Verifier for MockVerifier {
            fn verify(&self, _output: &str, _sources: &[Chunk]) -> Result<Verdict> {
                Ok(Verdict::Fail {
                    issues: vec![VerificationIssue {
                        kind: IssueKind::InvalidCitationId,
                        description: "still bad".to_string(),
                        claim_text: None,
                    }],
                })
            }
        }

        let revision_loop = RevisionLoop::new(
            Arc::new(MockLlm),
            Arc::new(MockVerifier),
            1, // max 1 attempt
        );

        let sources: Vec<Chunk> = vec![Chunk {
            chunk_id: Uuid::new_v4(),
            doc_id: Uuid::new_v4(),
            doc_version: 1,
            source_path: "doc.md".to_string(),
            section_path: vec![],
            char_offset: 0,
            char_len: 100,
            token_count: 50,
            text: "content".to_string(),
            metadata: HashMap::new(),
        }];

        let original_messages = vec![ChatMessage {
            role: Role::User,
            content: "test question".to_string(),
        }];

        let verdict = Verdict::Fail {
            issues: vec![VerificationIssue {
                kind: IssueKind::InvalidCitationId,
                description: "bad citation".to_string(),
                claim_text: None,
            }],
        };

        let result = revision_loop
            .revise(&original_messages, "draft", &verdict, &sources, None, 512)
            .await
            .unwrap();

        assert_eq!(result.attempts, 1);
        assert!(result.verdict.is_fail());
    }

    #[tokio::test]
    async fn test_revision_loop_succeeds_on_retry() {
        use cwc_core::types::{IssueKind, VerificationIssue};
        use std::collections::HashMap;
        use uuid::Uuid;

        // Mock LLM that returns good output
        struct MockLlm;
        #[async_trait::async_trait]
        impl LlmClient for MockLlm {
            async fn generate(
                &self, _prompt: &str, _grammar: Option<&str>, _max_tokens: u32,
            ) -> Result<String> {
                Ok("good output [S1]".to_string())
            }
            async fn generate_chat(
                &self, _messages: &[ChatMessage], _grammar: Option<&str>, _max_tokens: u32,
            ) -> Result<String> {
                Ok("good output [S1]".to_string())
            }
        }

        // Mock verifier that always passes
        struct MockVerifier;
        impl Verifier for MockVerifier {
            fn verify(&self, _output: &str, _sources: &[Chunk]) -> Result<Verdict> {
                Ok(Verdict::Pass)
            }
        }

        let revision_loop = RevisionLoop::new(
            Arc::new(MockLlm),
            Arc::new(MockVerifier),
            1,
        );

        let sources: Vec<Chunk> = vec![Chunk {
            chunk_id: Uuid::new_v4(),
            doc_id: Uuid::new_v4(),
            doc_version: 1,
            source_path: "doc.md".to_string(),
            section_path: vec![],
            char_offset: 0,
            char_len: 100,
            token_count: 50,
            text: "content".to_string(),
            metadata: HashMap::new(),
        }];

        let original_messages = vec![ChatMessage {
            role: Role::User,
            content: "test question".to_string(),
        }];

        let verdict = Verdict::Fail {
            issues: vec![VerificationIssue {
                kind: IssueKind::InvalidCitationId,
                description: "bad citation".to_string(),
                claim_text: None,
            }],
        };

        let result = revision_loop
            .revise(&original_messages, "draft", &verdict, &sources, None, 512)
            .await
            .unwrap();

        assert_eq!(result.attempts, 1);
        assert!(result.verdict.is_pass());
        assert_eq!(result.output, "good output [S1]");
    }

    #[test]
    fn test_revision_prompt_coverage_issues() {
        use cwc_core::types::{IssueKind, VerificationIssue};

        let verdict = Verdict::Fail {
            issues: vec![
                VerificationIssue {
                    kind: IssueKind::MissingCitation,
                    description: "factual sentence lacks citation".to_string(),
                    claim_text: Some("The system has 4GB RAM.".to_string()),
                },
                VerificationIssue {
                    kind: IssueKind::MissingCitation,
                    description: "factual sentence lacks citation".to_string(),
                    claim_text: Some("It runs on 8 cores.".to_string()),
                },
            ],
        };

        let prompt = RevisionLoop::build_revision_prompt(&verdict);
        // Should list both issues
        let dash_count = prompt.matches("- factual sentence lacks citation").count();
        assert_eq!(dash_count, 2);
    }
}
