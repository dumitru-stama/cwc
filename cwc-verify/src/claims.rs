use std::sync::LazyLock;

use regex::Regex;

/// An atomic claim extracted from a response.
#[derive(Debug, Clone)]
pub struct Claim {
    /// The claim text (usually one sentence).
    pub text: String,
    /// Index of the sentence in the original response.
    pub sentence_index: usize,
    /// Whether this claim looks factual (verifiable).
    pub is_factual: bool,
    /// Source citations found in this claim (e.g. ["S1", "S3"]).
    pub cited_sources: Vec<String>,
}

static NUMERIC_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\d+").expect("numeric regex"));

static DATE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b").expect("date regex"));

static MEASUREMENT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\d+\s*(?:gb|mb|kb|ms|seconds?|minutes?|hours?|bytes?|ghz|mhz|km|cm|mm|kg|lbs?|%)")
        .expect("measurement regex")
});

static PROPER_NOUN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b").expect("proper noun regex"));

static CITATION_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[S(\d+)\]").expect("citation regex"));

/// Non-factual sentence starters / markers.
const NON_FACTUAL_PREFIXES: &[&str] = &[
    "i think",
    "i believe",
    "perhaps",
    "maybe",
    "in my opinion",
    "it seems",
    "it appears",
    "i would say",
    "arguably",
    "it's possible",
    "it is possible",
];

/// Factual verb patterns (copula + likely factual content).
const FACTUAL_VERB_PATTERNS: &[&str] = &[
    " is ", " are ", " was ", " were ",
    " has ", " have ", " had ",
    " uses ", " used ", " provides ", " requires ",
    " ensures ", " prevents ", " enables ",
];

pub struct ClaimExtractor;

impl ClaimExtractor {
    /// Break a response into atomic factual claims.
    /// Uses sentence splitting + heuristic filtering (drop opinions, meta-text, questions).
    pub fn extract_claims(text: &str) -> Vec<Claim> {
        let sentences = split_into_sentences(text);
        let mut claims = Vec::new();

        for (idx, sentence) in sentences.iter().enumerate() {
            let trimmed = sentence.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Skip very short sentences
            if trimmed.split_whitespace().count() < 4 {
                continue;
            }

            let is_factual = is_factual_claim(trimmed);
            let cited_sources = extract_claim_citations(trimmed);

            claims.push(Claim {
                text: trimmed.to_string(),
                sentence_index: idx,
                is_factual,
                cited_sources,
            });
        }

        claims
    }

    /// Extract only the factual claims (for verification).
    pub fn extract_factual_claims(text: &str) -> Vec<Claim> {
        Self::extract_claims(text)
            .into_iter()
            .filter(|c| c.is_factual)
            .collect()
    }

    /// Extract factual claims that lack citations (candidates for RARR).
    pub fn extract_uncited_factual_claims(text: &str) -> Vec<Claim> {
        Self::extract_claims(text)
            .into_iter()
            .filter(|c| c.is_factual && c.cited_sources.is_empty())
            .collect()
    }
}

/// Split text into sentences. Reuses the citations module's sentence splitter.
fn split_into_sentences(text: &str) -> Vec<String> {
    crate::citations::split_sentences(text)
        .into_iter()
        .map(|(_, _, s)| s)
        .collect()
}

/// Determine if a sentence looks like a factual claim.
fn is_factual_claim(sentence: &str) -> bool {
    let lower = sentence.to_lowercase();

    // Questions are not factual claims
    if sentence.trim_end().ends_with('?') {
        return false;
    }

    // Non-factual prefixes
    for prefix in NON_FACTUAL_PREFIXES {
        if lower.starts_with(prefix) {
            return false;
        }
    }

    // Strip citation markers before numeric checks so [S1] doesn't inflate factual detection
    let stripped = CITATION_RE.replace_all(sentence, "");

    // Check for factual indicators
    if NUMERIC_RE.is_match(&stripped) {
        return true;
    }
    if DATE_RE.is_match(&stripped) {
        return true;
    }
    if MEASUREMENT_RE.is_match(&stripped) {
        return true;
    }
    if PROPER_NOUN_RE.is_match(sentence) {
        return true;
    }

    // Factual verb patterns
    for pattern in FACTUAL_VERB_PATTERNS {
        if lower.contains(pattern) {
            return true;
        }
    }

    false
}

/// Extract citation IDs from a claim sentence.
fn extract_claim_citations(sentence: &str) -> Vec<String> {
    CITATION_RE
        .captures_iter(sentence)
        .map(|cap| format!("S{}", &cap[1]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_claims_factual_and_opinion() {
        let text = "Rust uses LLVM. I think it's great.";
        let _claims = ClaimExtractor::extract_claims(text);
        // "Rust uses LLVM." is short (3 words) — below threshold
        // "I think it's great." is also short
        // Use longer sentences:
        let text2 = "Rust uses LLVM for code generation and optimization. I think it's a great programming language overall.";
        let claims2 = ClaimExtractor::extract_claims(text2);
        assert!(claims2.len() >= 2);
        let factual = claims2.iter().find(|c| c.text.contains("LLVM"));
        assert!(factual.is_some());
        assert!(factual.unwrap().is_factual, "LLVM sentence should be factual (proper noun)");
        let opinion = claims2.iter().find(|c| c.text.contains("I think"));
        assert!(opinion.is_some());
        assert!(!opinion.unwrap().is_factual, "opinion should not be factual");
    }

    #[test]
    fn test_extract_claims_question_not_factual() {
        let text = "What is the speed of light in a vacuum?";
        let claims = ClaimExtractor::extract_claims(text);
        for claim in &claims {
            assert!(!claim.is_factual, "questions should not be factual");
        }
    }

    #[test]
    fn test_extract_claims_numbers_are_factual() {
        let text = "The system processes 1000 requests per second under normal conditions.";
        let claims = ClaimExtractor::extract_claims(text);
        assert!(!claims.is_empty());
        assert!(claims[0].is_factual, "sentence with numbers should be factual");
    }

    #[test]
    fn test_extract_claims_with_citations() {
        let text = "Rust ensures memory safety through ownership [S1]. The borrow checker prevents data races [S2].";
        let claims = ClaimExtractor::extract_claims(text);
        let cited: Vec<&Claim> = claims.iter().filter(|c| !c.cited_sources.is_empty()).collect();
        assert!(cited.len() >= 2);
        assert!(cited[0].cited_sources.contains(&"S1".to_string()));
    }

    #[test]
    fn test_extract_factual_claims_filters() {
        let text = "Rust was created in 2010 by Graydon Hoare at Mozilla. I think it's wonderful. What do you think?";
        let factual = ClaimExtractor::extract_factual_claims(text);
        // Only the first sentence is factual
        assert_eq!(factual.len(), 1);
        assert!(factual[0].text.contains("2010"));
    }

    #[test]
    fn test_extract_uncited_factual_claims() {
        let text = "Rust was released in 2015 as a stable language for public use. The borrow checker is compile-time [S1].";
        let uncited = ClaimExtractor::extract_uncited_factual_claims(text);
        assert_eq!(uncited.len(), 1);
        assert!(uncited[0].text.contains("2015"));
    }

    #[test]
    fn test_is_factual_with_proper_nouns() {
        assert!(is_factual_claim("The Rust Programming Language was developed at Mozilla Research."));
    }

    #[test]
    fn test_is_factual_with_measurements() {
        assert!(is_factual_claim("The response time was approximately 50ms under load."));
    }

    #[test]
    fn test_is_factual_with_verb_patterns() {
        assert!(is_factual_claim("The ownership system ensures memory safety without garbage collection."));
    }

    #[test]
    fn test_not_factual_maybe_prefix() {
        assert!(!is_factual_claim("Maybe the system could handle more load in the future."));
    }

    #[test]
    fn test_not_factual_question() {
        assert!(!is_factual_claim("Is Rust faster than C++ in all cases?"));
    }

    #[test]
    fn test_empty_text_no_claims() {
        let claims = ClaimExtractor::extract_claims("");
        assert!(claims.is_empty());
    }

    #[test]
    fn test_short_sentences_filtered() {
        let claims = ClaimExtractor::extract_claims("Yes. No. Ok.");
        assert!(claims.is_empty());
    }

    #[test]
    fn test_extract_claim_citations_multiple() {
        let cites = extract_claim_citations("According to [S1] and [S3], the result is clear.");
        assert_eq!(cites, vec!["S1", "S3"]);
    }

    #[test]
    fn test_extract_claim_citations_none() {
        let cites = extract_claim_citations("The system runs on Linux.");
        assert!(cites.is_empty());
    }

    #[test]
    fn test_sentence_indices_sequential() {
        let text = "First claim is here. Second claim follows. Third claim ends.";
        let claims = ClaimExtractor::extract_claims(text);
        for (i, claim) in claims.iter().enumerate() {
            assert_eq!(claim.sentence_index, i);
        }
    }

    #[test]
    fn test_perhaps_not_factual() {
        assert!(!is_factual_claim("Perhaps the approach could be improved with better algorithms."));
    }

    #[test]
    fn test_date_is_factual() {
        assert!(is_factual_claim("The project started on 01/15/2024 with the initial commit."));
    }

    #[test]
    fn test_citation_numbers_dont_inflate_factual() {
        // "[S1]" contains digits, but those shouldn't make a non-factual sentence factual
        assert!(!is_factual_claim("Perhaps it could work better overall [S1]."));
    }

    #[test]
    fn test_non_factual_prefix_mid_sentence_not_filtered() {
        // "I think" at start → not factual. But "I think" in the middle shouldn't trigger.
        assert!(is_factual_claim("The system uses what I think is a borrow checker for memory safety."));
    }
}
