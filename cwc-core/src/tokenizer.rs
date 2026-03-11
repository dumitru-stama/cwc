use tiktoken_rs::CoreBPE;

use crate::error::{CwcError, Result};
use crate::traits::TokenCounter;

/// Tokenizer wrapper for fast token counting and truncation.
pub struct Tokenizer {
    bpe: CoreBPE,
}

impl TokenCounter for Tokenizer {
    fn count_tokens(&self, text: &str) -> u32 {
        Tokenizer::count_tokens(self, text)
    }

    fn truncate_to_tokens(&self, text: &str, max_tokens: u32) -> String {
        Tokenizer::truncate_to_tokens(self, text, max_tokens)
    }
}

impl Tokenizer {
    /// Create a tokenizer for the given model name.
    ///
    /// Supported model names follow tiktoken conventions:
    /// "gpt-4", "gpt-3.5-turbo", "cl100k_base", etc.
    /// Falls back to cl100k_base if the model is unrecognized.
    pub fn new(model: &str) -> Result<Self> {
        let bpe = tiktoken_rs::get_bpe_from_model(model)
            .or_else(|_| tiktoken_rs::cl100k_base())
            .map_err(|e| CwcError::Tokenizer(format!("failed to load tokenizer: {e}")))?;
        Ok(Self { bpe })
    }

    /// Create a tokenizer using the cl100k_base encoding (default).
    pub fn default_tokenizer() -> Result<Self> {
        let bpe = tiktoken_rs::cl100k_base()
            .map_err(|e| CwcError::Tokenizer(format!("failed to load cl100k_base: {e}")))?;
        Ok(Self { bpe })
    }

    /// Count the number of tokens in a text string.
    pub fn count_tokens(&self, text: &str) -> u32 {
        self.bpe.encode_ordinary(text).len() as u32
    }

    /// Truncate text to fit within a maximum token budget.
    ///
    /// Returns the longest prefix of the text that uses at most `max_tokens`
    /// tokens. The result is valid UTF-8 and does not split multi-byte characters.
    pub fn truncate_to_tokens(&self, text: &str, max_tokens: u32) -> String {
        let tokens = self.bpe.encode_ordinary(text);
        if tokens.len() <= max_tokens as usize {
            return text.to_string();
        }
        let truncated_tokens = &tokens[..max_tokens as usize];
        match self.bpe.decode(truncated_tokens.to_vec()) {
            Ok(s) => s,
            Err(_) => {
                // Fallback: try with one fewer token until we get valid output
                for n in (0..max_tokens as usize).rev() {
                    if let Ok(s) = self.bpe.decode(tokens[..n].to_vec()) {
                        return s;
                    }
                }
                String::new()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_count_tokens_plausible() {
        let tok = Tokenizer::default_tokenizer().unwrap();
        // "hello world" is typically 2 tokens
        let count = tok.count_tokens("hello world");
        assert!((2..=4).contains(&count), "expected 2-4 tokens, got {count}");

        // Empty string is 0 tokens
        assert_eq!(tok.count_tokens(""), 0);

        // Longer text should have more tokens
        let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(10);
        let long_count = tok.count_tokens(&long_text);
        assert!(long_count > 10, "long text should have many tokens");
    }

    #[test]
    fn test_tokenizer_truncate_never_exceeds_limit() {
        let tok = Tokenizer::default_tokenizer().unwrap();
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(100);

        for max in [1, 5, 10, 50] {
            let truncated = tok.truncate_to_tokens(&text, max);
            let count = tok.count_tokens(&truncated);
            assert!(
                count <= max,
                "truncate_to_tokens({max}) produced {count} tokens"
            );
        }
    }

    #[test]
    fn test_tokenizer_truncate_returns_full_text_if_under_budget() {
        let tok = Tokenizer::default_tokenizer().unwrap();
        let text = "hello world";
        let truncated = tok.truncate_to_tokens(text, 100);
        assert_eq!(truncated, text);
    }

    #[test]
    fn test_tokenizer_truncate_empty() {
        let tok = Tokenizer::default_tokenizer().unwrap();
        let truncated = tok.truncate_to_tokens("", 10);
        assert_eq!(truncated, "");
    }

    #[test]
    fn test_tokenizer_new_with_model() {
        let tok = Tokenizer::new("gpt-4").unwrap();
        let count = tok.count_tokens("hello");
        assert!(count >= 1);
    }

    #[test]
    fn test_tokenizer_new_fallback() {
        // Unknown model should fall back to cl100k_base
        let tok = Tokenizer::new("unknown-model-xyz").unwrap();
        let count = tok.count_tokens("hello");
        assert!(count >= 1);
    }

    #[test]
    fn test_tokenizer_truncate_to_zero_tokens() {
        let tok = Tokenizer::default_tokenizer().unwrap();
        let truncated = tok.truncate_to_tokens("hello world", 0);
        assert_eq!(tok.count_tokens(&truncated), 0);
        assert!(truncated.is_empty());
    }

    #[test]
    fn test_tokenizer_unicode_truncation() {
        let tok = Tokenizer::default_tokenizer().unwrap();
        // Mix of ASCII, CJK, and emoji
        let text = "Hello 世界! 🦀 Rust is great. こんにちは。";
        let full_count = tok.count_tokens(text);
        assert!(full_count > 5);

        // Truncate to half — should produce valid UTF-8
        let half = full_count / 2;
        let truncated = tok.truncate_to_tokens(text, half);
        let trunc_count = tok.count_tokens(&truncated);
        assert!(trunc_count <= half);
        // Verify the result is valid UTF-8 (it is, since it's a String)
        assert!(truncated.len() <= text.len());
    }

    #[test]
    fn test_tokenizer_count_single_token() {
        let tok = Tokenizer::default_tokenizer().unwrap();
        // "a" should be exactly 1 token
        assert_eq!(tok.count_tokens("a"), 1);
    }

    #[test]
    fn test_tokenizer_truncate_exact_boundary() {
        let tok = Tokenizer::default_tokenizer().unwrap();
        let text = "hello world";
        let exact_count = tok.count_tokens(text);
        // Truncating to exactly the token count should return the full text
        let truncated = tok.truncate_to_tokens(text, exact_count);
        assert_eq!(truncated, text);
    }
}
