use std::sync::LazyLock;

use regex::Regex;

static MISSING_INFO_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:don'?t|do not|doesn'?t|does not|cannot|can'?t) (?:have|contain|include|provide|find|cover|address|answer|determine)\b[^.!?\n]*")
        .expect("missing info regex")
});

pub struct AbstentionDetector;

impl AbstentionDetector {
    /// Check if the response indicates insufficient evidence.
    pub fn is_abstention(text: &str) -> bool {
        // Exact keyword
        if text.contains("INSUFFICIENT_EVIDENCE") {
            return true;
        }

        let lower = text.to_lowercase();

        let patterns = [
            "i don't have enough information",
            "i do not have enough information",
            "the sources don't contain",
            "the sources do not contain",
            "i cannot answer this based on the provided sources",
            "i cannot answer this based on the sources",
            "not enough information in the provided sources",
            "the provided sources do not",
            "insufficient information",
            "unable to determine from the provided",
            "unable to answer based on",
            "cannot be determined from the sources",
            "the sources do not provide enough",
            "the sources don't provide enough",
        ];

        patterns.iter().any(|p| lower.contains(p))
    }

    /// Extract what information the model says is missing.
    pub fn extract_missing_info(text: &str) -> Vec<String> {
        MISSING_INFO_RE
            .find_iter(text)
            .map(|m| m.as_str().trim().to_string())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abstention_exact_keyword() {
        assert!(AbstentionDetector::is_abstention("INSUFFICIENT_EVIDENCE"));
        assert!(AbstentionDetector::is_abstention(
            "The answer is INSUFFICIENT_EVIDENCE based on what I found."
        ));
    }

    #[test]
    fn test_abstention_fuzzy_patterns() {
        assert!(AbstentionDetector::is_abstention(
            "I don't have enough information to answer this question."
        ));
        assert!(AbstentionDetector::is_abstention(
            "The sources don't contain relevant data for this query."
        ));
        assert!(AbstentionDetector::is_abstention(
            "I cannot answer this based on the provided sources."
        ));
        assert!(AbstentionDetector::is_abstention(
            "There is insufficient information in the documents."
        ));
    }

    #[test]
    fn test_abstention_normal_answer() {
        assert!(!AbstentionDetector::is_abstention(
            "Rust's ownership system ensures memory safety without garbage collection."
        ));
        assert!(!AbstentionDetector::is_abstention(
            "The answer is 42 according to the sources [S1]."
        ));
    }

    #[test]
    fn test_abstention_case_insensitive() {
        assert!(AbstentionDetector::is_abstention(
            "I DON'T HAVE ENOUGH INFORMATION to answer."
        ));
    }

    #[test]
    fn test_extract_missing_info() {
        let text =
            "The sources don't contain any information about performance benchmarks. I cannot find details about memory usage.";
        let missing = AbstentionDetector::extract_missing_info(text);
        assert!(!missing.is_empty());
    }

    #[test]
    fn test_extract_missing_info_none() {
        let text = "Rust is a systems programming language [S1].";
        let missing = AbstentionDetector::extract_missing_info(text);
        assert!(missing.is_empty());
    }

    #[test]
    fn test_abstention_keyword_embedded_in_json() {
        // JSON output containing the keyword should still be detected
        let text = r#"{"answer": "INSUFFICIENT_EVIDENCE", "confidence": 0.0}"#;
        assert!(AbstentionDetector::is_abstention(text));
    }

    #[test]
    fn test_abstention_partial_match_not_triggered() {
        // "insufficient" alone in a normal sentence shouldn't trigger
        // (it actually does because "insufficient information" is a pattern)
        // but "insufficient budget" should NOT trigger
        assert!(!AbstentionDetector::is_abstention(
            "The budget was insufficient for the project completion timeline."
        ));
    }

    #[test]
    fn test_abstention_empty_input() {
        assert!(!AbstentionDetector::is_abstention(""));
    }

    #[test]
    fn test_extract_missing_info_multiple() {
        let text = "The sources don't contain details about performance. \
                     They also don't provide information about memory usage.";
        let missing = AbstentionDetector::extract_missing_info(text);
        assert!(missing.len() >= 2);
    }
}
