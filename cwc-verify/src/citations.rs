use std::collections::HashSet;
use std::sync::LazyLock;

use cwc_core::types::{IssueKind, VerificationIssue};
use regex::Regex;

static CITATION_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[S(\d+)\]").expect("citation regex"));

/// A citation reference found in text.
#[derive(Debug, Clone)]
pub struct CitationRef {
    pub id: String,
    pub char_offset: usize,
    pub sentence: String,
}

pub struct CitationVerifier;

impl CitationVerifier {
    /// Extract all citation references from text.
    pub fn extract_citations(text: &str) -> Vec<CitationRef> {
        let sentences = split_sentences(text);
        let mut refs = Vec::new();

        for cap in CITATION_RE.captures_iter(text) {
            let m = cap.get(0).expect("group 0 always exists on a regex match");
            let id = format!("S{}", &cap[1]);
            let offset = m.start();

            // Find which sentence contains this citation
            let sentence = sentences
                .iter()
                .find(|(start, end, _)| offset >= *start && offset < *end)
                .map(|(_, _, s)| s.clone())
                .unwrap_or_default();

            refs.push(CitationRef {
                id,
                char_offset: offset,
                sentence,
            });
        }
        refs
    }

    /// Check that all citations reference valid source IDs.
    pub fn check_validity(
        citations: &[CitationRef],
        valid_ids: &HashSet<String>,
    ) -> Vec<VerificationIssue> {
        let mut issues = Vec::new();
        let mut seen_invalid = HashSet::new();

        for cref in citations {
            if !valid_ids.contains(&cref.id) && seen_invalid.insert(cref.id.clone()) {
                issues.push(VerificationIssue {
                    kind: IssueKind::InvalidCitationId,
                    description: format!(
                        "[{}] is not a valid source ID. Valid: {}",
                        cref.id,
                        format_valid_ids(valid_ids),
                    ),
                    claim_text: Some(cref.sentence.clone()),
                });
            }
        }
        issues
    }

    /// Check citation coverage: factual-looking sentences should have citations.
    pub fn check_coverage(
        text: &str,
        citations: &[CitationRef],
    ) -> Vec<VerificationIssue> {
        let sentences = split_sentences(text);
        let cited_offsets: HashSet<usize> = citations.iter().map(|c| c.char_offset).collect();
        let mut issues = Vec::new();

        for (start, end, sentence) in &sentences {
            // Skip short sentences (< 8 words)
            if sentence.split_whitespace().count() < 8 {
                continue;
            }
            // Skip questions
            if sentence.trim_end().ends_with('?') {
                continue;
            }
            // Skip self-referential / opinion sentences
            if is_opinion_sentence(sentence) {
                continue;
            }
            // Check if this sentence has any citation
            let has_citation = cited_offsets.iter().any(|o| *o >= *start && *o < *end);
            if has_citation {
                continue;
            }
            // Flag if it looks factual
            if looks_factual(sentence) {
                issues.push(VerificationIssue {
                    kind: IssueKind::MissingCitation,
                    description: "factual sentence lacks citation".to_string(),
                    claim_text: Some(sentence.clone()),
                });
            }
        }
        issues
    }
}

/// Split text into sentences, returning (start_offset, end_offset, text).
pub(crate) fn split_sentences(text: &str) -> Vec<(usize, usize, String)> {
    static SENTENCE_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"[^.!?\n]+[.!?\n]?").expect("sentence regex"));

    SENTENCE_RE
        .find_iter(text)
        .map(|m| (m.start(), m.end(), m.as_str().trim().to_string()))
        .filter(|(_, _, s)| !s.is_empty())
        .collect()
}

pub(crate) fn is_opinion_sentence(s: &str) -> bool {
    let lower = s.to_lowercase();
    let opinion_markers = [
        "i think",
        "i believe",
        "in my opinion",
        "it seems",
        "based on the sources",
        "based on the provided",
        "from the sources",
        "the sources suggest",
        "the sources indicate",
    ];
    opinion_markers.iter().any(|m| lower.contains(m))
}

pub(crate) fn looks_factual(s: &str) -> bool {
    static NUMERIC_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\d+").expect("numeric regex"));

    let lower = s.to_lowercase();

    // Numeric values
    if NUMERIC_RE.is_match(s) {
        return true;
    }

    // Evidential phrases
    let factual_markers = [
        "according to",
        "studies show",
        "research indicates",
        "research shows",
        "data shows",
        "statistics show",
        "was founded",
        "was established",
        "was created",
        "is defined as",
        "is known as",
    ];
    factual_markers.iter().any(|m| lower.contains(m))
}

fn format_valid_ids(ids: &HashSet<String>) -> String {
    let mut sorted: Vec<&String> = ids.iter().collect();
    sorted.sort();
    let formatted: Vec<String> = sorted.iter().map(|id| format!("[{id}]")).collect();
    formatted.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_citations_basic() {
        let refs = CitationVerifier::extract_citations("Answer [S1] and also [S3].");
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].id, "S1");
        assert_eq!(refs[1].id, "S3");
    }

    #[test]
    fn test_extract_citations_empty() {
        let refs = CitationVerifier::extract_citations("No citations here.");
        assert!(refs.is_empty());
    }

    #[test]
    fn test_extract_citations_malformed_not_extracted() {
        let refs = CitationVerifier::extract_citations("See [SX] and [S] and [S-1].");
        assert!(refs.is_empty());
    }

    #[test]
    fn test_check_validity_all_valid() {
        let refs = CitationVerifier::extract_citations("Uses [S1] and [S2].");
        let valid: HashSet<String> = ["S1", "S2", "S3"].iter().map(|s| s.to_string()).collect();
        let issues = CitationVerifier::check_validity(&refs, &valid);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_check_validity_invalid_id() {
        let refs = CitationVerifier::extract_citations("Uses [S5].");
        let valid: HashSet<String> = ["S1", "S2"].iter().map(|s| s.to_string()).collect();
        let issues = CitationVerifier::check_validity(&refs, &valid);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].kind, IssueKind::InvalidCitationId);
        assert!(issues[0].description.contains("S5"));
    }

    #[test]
    fn test_check_coverage_numeric_without_citation() {
        let text = "The system uses 4GB of RAM and runs on 8 cores for processing tasks.";
        let refs = CitationVerifier::extract_citations(text);
        let issues = CitationVerifier::check_coverage(text, &refs);
        assert!(!issues.is_empty());
        assert_eq!(issues[0].kind, IssueKind::MissingCitation);
    }

    #[test]
    fn test_check_coverage_opinion_no_issue() {
        let text = "I think this is correct based on the sources provided in this context.";
        let refs = CitationVerifier::extract_citations(text);
        let issues = CitationVerifier::check_coverage(text, &refs);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_check_coverage_short_sentence_no_issue() {
        let text = "Yes.";
        let refs = CitationVerifier::extract_citations(text);
        let issues = CitationVerifier::check_coverage(text, &refs);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_check_coverage_cited_no_issue() {
        let text = "The system uses 4GB of RAM [S1] for running the heavy processing tasks.";
        let refs = CitationVerifier::extract_citations(text);
        let issues = CitationVerifier::check_coverage(text, &refs);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_check_coverage_question_no_issue() {
        let text = "What is the speed of light at 300000 km per second exactly?";
        let refs = CitationVerifier::extract_citations(text);
        let issues = CitationVerifier::check_coverage(text, &refs);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_check_validity_deduplicates_invalid() {
        let refs = CitationVerifier::extract_citations("[S9] and [S9] again.");
        let valid: HashSet<String> = ["S1"].iter().map(|s| s.to_string()).collect();
        let issues = CitationVerifier::check_validity(&refs, &valid);
        assert_eq!(issues.len(), 1); // Only one issue for S9
    }

    #[test]
    fn test_extract_citations_sentence_association() {
        let text = "First sentence [S1]. Second sentence [S2].";
        let refs = CitationVerifier::extract_citations(text);
        assert_eq!(refs.len(), 2);
        assert!(refs[0].sentence.contains("First"));
        assert!(refs[1].sentence.contains("Second"));
    }

    #[test]
    fn test_extract_citations_high_number() {
        let refs = CitationVerifier::extract_citations("See [S99] and [S100].");
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].id, "S99");
        assert_eq!(refs[1].id, "S100");
    }

    #[test]
    fn test_check_validity_empty_citations() {
        let valid: HashSet<String> = ["S1"].iter().map(|s| s.to_string()).collect();
        let issues = CitationVerifier::check_validity(&[], &valid);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_check_validity_empty_valid_ids() {
        let refs = CitationVerifier::extract_citations("Uses [S1].");
        let valid: HashSet<String> = HashSet::new();
        let issues = CitationVerifier::check_validity(&refs, &valid);
        assert_eq!(issues.len(), 1);
    }

    #[test]
    fn test_check_coverage_evidential_phrase_without_citation() {
        let text = "According to recent research the system performs well under heavy load conditions.";
        let refs = CitationVerifier::extract_citations(text);
        let issues = CitationVerifier::check_coverage(text, &refs);
        assert!(!issues.is_empty());
        assert_eq!(issues[0].kind, IssueKind::MissingCitation);
    }

    #[test]
    fn test_check_coverage_multi_sentence_mixed() {
        // One factual uncited, one cited, one opinion → only the uncited factual flagged
        let text = "The system was founded in 2020 by a team of engineers at a major company. \
                     The architecture uses 16 cores for computation [S1]. \
                     I think the design is elegant and well thought out for this use case.";
        let refs = CitationVerifier::extract_citations(text);
        let issues = CitationVerifier::check_coverage(text, &refs);
        // First sentence: "was founded" + "2020" → factual, no citation → flagged
        // Second: cited → not flagged
        // Third: opinion → not flagged
        assert_eq!(issues.len(), 1);
        assert!(issues[0].claim_text.as_ref().unwrap().contains("founded"));
    }

    #[test]
    fn test_split_sentences_newline() {
        let sentences = split_sentences("Line one.\nLine two.");
        assert_eq!(sentences.len(), 2);
    }

    #[test]
    fn test_check_coverage_no_factual_no_issue() {
        // All non-factual, non-numeric, no evidential phrases
        let text = "The weather seems nice outside right now in this part of the country.";
        let refs = CitationVerifier::extract_citations(text);
        let issues = CitationVerifier::check_coverage(text, &refs);
        assert!(issues.is_empty());
    }
}
