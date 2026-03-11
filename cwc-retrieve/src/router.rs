use cwc_core::config::RetrievalConfig;

/// Whether retrieval is needed and how much.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetrievalDecision {
    /// Purely procedural task, no retrieval needed.
    Skip,
    /// Standard retrieval.
    Retrieve { top_k: usize },
    /// Low-confidence query, cast a wider net.
    RetrieveMore { top_k: usize },
}

/// Command-like patterns that suggest no retrieval is needed.
const SKIP_PREFIXES: &[&str] = &[
    "reformat",
    "convert",
    "translate",
    "fix the grammar",
    "fix grammar",
    "summarize this",
    "rewrite",
    "list the",
    "sort",
    "count",
];

/// Question words that suggest retrieval is needed.
const QUESTION_WORDS: &[&str] = &[
    "what",
    "how",
    "why",
    "when",
    "where",
    "which",
    "who",
    "explain",
    "describe",
    "compare",
    "define",
];

/// Classify whether retrieval is needed for a query.
///
/// Simple heuristic router:
/// - Very short queries (< 3 words) that look like commands → Skip
/// - Queries starting with command-like prefixes → Skip
/// - Queries with question words or topic references → Retrieve
/// - Ambiguous queries → RetrieveMore (wider net)
/// - Default → Retrieve
pub fn classify_retrieval_need(query: &str, config: &RetrievalConfig) -> RetrievalDecision {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return RetrievalDecision::Skip;
    }

    let lower = trimmed.to_lowercase();
    let word_count = trimmed.split_whitespace().count();

    // Very short queries that look like commands
    if word_count < 3 {
        // Check if it looks like a command
        let is_command = SKIP_PREFIXES.iter().any(|p| lower.starts_with(p));
        if is_command {
            return RetrievalDecision::Skip;
        }
        // Very short non-command: ambiguous, cast wider net
        if word_count == 1 {
            return RetrievalDecision::RetrieveMore {
                top_k: config.final_top_k * 2,
            };
        }
    }

    // Command-like prefixes → Skip
    if SKIP_PREFIXES.iter().any(|p| lower.starts_with(p)) {
        return RetrievalDecision::Skip;
    }

    // Question words → Retrieve
    let first_word = lower.split_whitespace().next().unwrap_or("");
    if QUESTION_WORDS.contains(&first_word) {
        return RetrievalDecision::Retrieve {
            top_k: config.final_top_k,
        };
    }

    // Contains "?" → probably a question
    if trimmed.contains('?') {
        return RetrievalDecision::Retrieve {
            top_k: config.final_top_k,
        };
    }

    // Default: retrieve
    RetrievalDecision::Retrieve {
        top_k: config.final_top_k,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> RetrievalConfig {
        RetrievalConfig::default()
    }

    #[test]
    fn test_router_skip_reformat() {
        let config = default_config();
        let decision = classify_retrieval_need("reformat this text", &config);
        assert_eq!(decision, RetrievalDecision::Skip);
    }

    #[test]
    fn test_router_skip_convert() {
        let config = default_config();
        let decision = classify_retrieval_need("convert this to JSON", &config);
        assert_eq!(decision, RetrievalDecision::Skip);
    }

    #[test]
    fn test_router_retrieve_question() {
        let config = default_config();
        let decision = classify_retrieval_need("what is a context window?", &config);
        assert_eq!(
            decision,
            RetrievalDecision::Retrieve {
                top_k: config.final_top_k
            }
        );
    }

    #[test]
    fn test_router_retrieve_how() {
        let config = default_config();
        let decision = classify_retrieval_need("how does Rust handle memory safety", &config);
        assert_eq!(
            decision,
            RetrievalDecision::Retrieve {
                top_k: config.final_top_k
            }
        );
    }

    #[test]
    fn test_router_retrieve_explain() {
        let config = default_config();
        let decision = classify_retrieval_need("explain the borrowing rules in Rust", &config);
        assert_eq!(
            decision,
            RetrievalDecision::Retrieve {
                top_k: config.final_top_k
            }
        );
    }

    #[test]
    fn test_router_skip_translate() {
        let config = default_config();
        let decision = classify_retrieval_need("translate this to French", &config);
        assert_eq!(decision, RetrievalDecision::Skip);
    }

    #[test]
    fn test_router_empty_query() {
        let config = default_config();
        let decision = classify_retrieval_need("", &config);
        assert_eq!(decision, RetrievalDecision::Skip);
    }

    #[test]
    fn test_router_single_word_retrieves_more() {
        let config = default_config();
        let decision = classify_retrieval_need("ownership", &config);
        assert_eq!(
            decision,
            RetrievalDecision::RetrieveMore {
                top_k: config.final_top_k * 2
            }
        );
    }

    #[test]
    fn test_router_question_mark_triggers_retrieve() {
        let config = default_config();
        let decision = classify_retrieval_need("context window?", &config);
        assert_eq!(
            decision,
            RetrievalDecision::Retrieve {
                top_k: config.final_top_k
            }
        );
    }

    #[test]
    fn test_router_whitespace_only() {
        let config = default_config();
        let decision = classify_retrieval_need("   ", &config);
        assert_eq!(decision, RetrievalDecision::Skip);
    }

    #[test]
    fn test_router_case_insensitive() {
        let config = default_config();
        // "WHAT" should still match question word "what"
        let decision = classify_retrieval_need("WHAT is ownership?", &config);
        assert_eq!(
            decision,
            RetrievalDecision::Retrieve {
                top_k: config.final_top_k
            }
        );
    }

    #[test]
    fn test_router_rewrite_skip() {
        let config = default_config();
        let decision = classify_retrieval_need("rewrite this paragraph more concisely", &config);
        assert_eq!(decision, RetrievalDecision::Skip);
    }

    #[test]
    fn test_router_two_word_non_command() {
        let config = default_config();
        // Two words, not a command prefix → Retrieve (default)
        let decision = classify_retrieval_need("rust ownership", &config);
        assert_eq!(
            decision,
            RetrievalDecision::Retrieve {
                top_k: config.final_top_k
            }
        );
    }

    #[test]
    fn test_router_default_retrieve() {
        let config = default_config();
        let decision =
            classify_retrieval_need("tell me about Rust error handling patterns", &config);
        assert_eq!(
            decision,
            RetrievalDecision::Retrieve {
                top_k: config.final_top_k
            }
        );
    }
}
