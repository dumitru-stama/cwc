use std::sync::Arc;

use cwc_core::config::BudgetConfig;
use cwc_core::traits::TokenCounter;
use cwc_core::types::{RetrievalHit, TokenBudget};

const SAFETY_MARGIN: u32 = 256;

pub struct BudgetAllocator {
    tokenizer: Arc<dyn TokenCounter>,
    config: BudgetConfig,
}

impl BudgetAllocator {
    pub fn new(tokenizer: Arc<dyn TokenCounter>, config: BudgetConfig) -> Self {
        Self { tokenizer, config }
    }

    /// Compute the token budget breakdown for a given context window.
    ///
    /// Budget computation:
    ///   available = context_window - max_output_tokens
    ///   sources = available - instruction_tokens - memory_tokens - safety_margin(256)
    ///
    /// The `instruction_tokens` and `memory_tokens` are actual measured counts,
    /// not fractions.
    pub fn allocate(
        &self,
        context_window: u32,
        max_output_tokens: u32,
        instruction_tokens: u32,
        memory_tokens: u32,
    ) -> TokenBudget {
        let available = context_window.saturating_sub(max_output_tokens);
        let sources = available
            .saturating_sub(instruction_tokens)
            .saturating_sub(memory_tokens)
            .saturating_sub(SAFETY_MARGIN);
        let allocated = instruction_tokens + sources + memory_tokens + max_output_tokens;
        let remaining = context_window.saturating_sub(allocated);

        TokenBudget {
            total: context_window,
            instruction: instruction_tokens,
            sources,
            memory: memory_tokens,
            output_reserved: max_output_tokens,
            remaining,
        }
    }

    /// Select chunks that fit within the sources budget.
    /// Returns (selected chunks, actual token count used, number of truncated chunks).
    ///
    /// - Walk ranked hits in order
    /// - If a chunk fits, include it
    /// - If a chunk doesn't fit, skip it and continue to the next
    /// - After the walk, if budget remains, truncate the highest-scored skipped
    ///   chunk to fill remaining budget
    pub fn select_chunks(
        &self,
        chunks: &[RetrievalHit],
        budget: &TokenBudget,
    ) -> (Vec<RetrievalHit>, u32, usize) {
        let sources_budget = budget.sources;
        let mut selected = Vec::new();
        let mut used: u32 = 0;
        let mut first_skipped: Option<&RetrievalHit> = None;

        for hit in chunks {
            let chunk_tokens = hit.chunk.token_count;

            if used + chunk_tokens <= sources_budget {
                selected.push(hit.clone());
                used += chunk_tokens;
            } else if first_skipped.is_none() {
                // Remember the first (highest-scored) skipped chunk for potential truncation
                first_skipped = Some(hit);
            }
        }

        // Try to fill remaining budget by truncating the best skipped chunk
        let mut truncated_count = 0;
        if let Some(skipped) = first_skipped {
            let remaining = sources_budget.saturating_sub(used);
            if remaining > 0 {
                let truncated_text =
                    self.tokenizer.truncate_to_tokens(&skipped.chunk.text, remaining);
                let actual_tokens = self.tokenizer.count_tokens(&truncated_text);
                if actual_tokens > 0 {
                    let mut truncated_hit = skipped.clone();
                    truncated_hit.chunk.text = truncated_text;
                    truncated_hit.chunk.token_count = actual_tokens;
                    selected.push(truncated_hit);
                    used += actual_tokens;
                    truncated_count = 1;
                }
            }
        }

        (selected, used, truncated_count)
    }

    /// Access the underlying BudgetConfig.
    pub fn config(&self) -> &BudgetConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_hit;

    struct FakeTokenCounter;

    impl TokenCounter for FakeTokenCounter {
        fn count_tokens(&self, text: &str) -> u32 {
            // Simple: 1 token per word
            if text.is_empty() {
                0
            } else {
                text.split_whitespace().count() as u32
            }
        }

        fn truncate_to_tokens(&self, text: &str, max_tokens: u32) -> String {
            let words: Vec<&str> = text.split_whitespace().collect();
            words[..words.len().min(max_tokens as usize)].join(" ")
        }
    }

    fn make_allocator() -> BudgetAllocator {
        BudgetAllocator::new(Arc::new(FakeTokenCounter), BudgetConfig::default())
    }

    #[test]
    fn test_budget_allocation_sums_within_context_window() {
        let alloc = make_allocator();
        let budget = alloc.allocate(4096, 1024, 200, 100);
        // available = 4096 - 1024 = 3072
        // sources = 3072 - 200 - 100 - 256 = 2516
        assert_eq!(budget.total, 4096);
        assert_eq!(budget.instruction, 200);
        assert_eq!(budget.memory, 100);
        assert_eq!(budget.output_reserved, 1024);
        assert_eq!(budget.sources, 2516);
        let allocated = budget.instruction + budget.sources + budget.memory + budget.output_reserved;
        assert!(allocated <= budget.total);
    }

    #[test]
    fn test_budget_allocation_uses_actual_measured_tokens() {
        let alloc = make_allocator();
        let budget = alloc.allocate(4096, 1024, 500, 300);
        assert_eq!(budget.instruction, 500);
        assert_eq!(budget.memory, 300);
        // sources = (4096 - 1024) - 500 - 300 - 256 = 2016
        assert_eq!(budget.sources, 2016);
    }

    #[test]
    fn test_chunk_selection_total_within_budget() {
        let alloc = make_allocator();
        let budget = alloc.allocate(4096, 1024, 200, 100);

        let chunks: Vec<RetrievalHit> = (0..50)
            .map(|i| make_hit(i, 100, 0.9 - i as f32 * 0.01))
            .collect();

        let (selected, used, _) = alloc.select_chunks(&chunks, &budget);
        assert!(used <= budget.sources);
        assert!(!selected.is_empty());
        let total: u32 = selected.iter().map(|h| h.chunk.token_count).sum();
        assert_eq!(total, used);
    }

    #[test]
    fn test_chunk_selection_highest_scored_first() {
        let alloc = make_allocator();
        let budget = TokenBudget {
            total: 4096,
            instruction: 200,
            sources: 300,
            memory: 100,
            output_reserved: 1024,
            remaining: 0,
        };

        let chunks = vec![
            make_hit(0, 100, 0.9),
            make_hit(1, 100, 0.8),
            make_hit(2, 100, 0.7),
            make_hit(3, 100, 0.6),
        ];

        let (selected, _, _) = alloc.select_chunks(&chunks, &budget);
        // Budget is 300, each chunk is 100 tokens, so we get exactly 3
        assert_eq!(selected.len(), 3);
        assert_eq!(selected[0].score_fused, 0.9);
        assert_eq!(selected[1].score_fused, 0.8);
        assert_eq!(selected[2].score_fused, 0.7);
    }

    #[test]
    fn test_chunk_selection_skip_large_take_smaller() {
        // Plan test: "chunk too large for remaining budget → skipped (not truncated if others fit)"
        let alloc = make_allocator();
        let budget = TokenBudget {
            total: 4096,
            instruction: 0,
            sources: 140,
            memory: 0,
            output_reserved: 0,
            remaining: 0,
        };

        // Chunk 0 (100) fits, chunk 1 (200) is too large → skipped,
        // chunk 2 (30) fits after chunk 0.
        let chunks = vec![
            make_hit(0, 100, 0.9),
            make_hit(1, 200, 0.8),
            make_hit(2, 30, 0.7),
        ];

        let (selected, used, truncated) = alloc.select_chunks(&chunks, &budget);
        // Should select chunk 0 (100) and chunk 2 (30), skip chunk 1
        assert_eq!(selected.len(), 3); // 100 + 30 + truncated(10 from budget 140-130=10)
        assert_eq!(selected[0].chunk.chunk_id, chunks[0].chunk.chunk_id);
        assert_eq!(selected[1].chunk.chunk_id, chunks[2].chunk.chunk_id);
        // The skipped chunk 1 gets truncated to fill remaining 10 tokens
        assert_eq!(selected[2].chunk.chunk_id, chunks[1].chunk.chunk_id);
        assert!(selected[2].chunk.token_count <= 10);
        assert!(used <= 140);
        assert_eq!(truncated, 1);
    }

    #[test]
    fn test_chunk_selection_skip_no_truncation_needed() {
        // Budget exactly filled by smaller chunks, large chunk just skipped
        let alloc = make_allocator();
        let budget = TokenBudget {
            total: 4096,
            instruction: 0,
            sources: 130,
            memory: 0,
            output_reserved: 0,
            remaining: 0,
        };

        let chunks = vec![
            make_hit(0, 100, 0.9),
            make_hit(1, 200, 0.8),
            make_hit(2, 30, 0.7),
        ];

        let (selected, used, truncated) = alloc.select_chunks(&chunks, &budget);
        // 100 + 30 = 130, exact fit. Chunk 1 skipped, no truncation.
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].chunk.chunk_id, chunks[0].chunk.chunk_id);
        assert_eq!(selected[1].chunk.chunk_id, chunks[2].chunk.chunk_id);
        assert_eq!(used, 130);
        assert_eq!(truncated, 0);
    }

    #[test]
    fn test_chunk_truncation_last_fills_remaining() {
        // Plan test: "Chunk truncation: last chunk truncated to fill remaining budget"
        let alloc = make_allocator();
        let budget = TokenBudget {
            total: 4096,
            instruction: 0,
            sources: 150,
            memory: 0,
            output_reserved: 0,
            remaining: 0,
        };

        let chunks = vec![
            make_hit(0, 100, 0.9),
            make_hit(1, 100, 0.8),
        ];

        let (selected, used, truncated) = alloc.select_chunks(&chunks, &budget);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].chunk.token_count, 100);
        // Second chunk truncated to fit remaining 50 tokens
        assert!(selected[1].chunk.token_count <= 50);
        assert!(used <= 150);
        assert_eq!(truncated, 1);
    }

    #[test]
    fn test_chunk_truncation_single_oversized() {
        // Only chunk is larger than budget — truncated to fit
        let alloc = make_allocator();
        let budget = TokenBudget {
            total: 1000,
            instruction: 0,
            sources: 25,
            memory: 0,
            output_reserved: 0,
            remaining: 0,
        };

        let mut hit = make_hit(0, 30, 0.9);
        hit.chunk.text = "word ".repeat(30).trim().to_string();
        hit.chunk.token_count = 30;

        let (selected, used, truncated) = alloc.select_chunks(&[hit], &budget);
        assert_eq!(selected.len(), 1);
        assert!(used <= 25);
        assert!(selected[0].chunk.token_count <= 25);
        assert_eq!(truncated, 1);
    }

    #[test]
    fn test_zero_chunks_empty_result() {
        let alloc = make_allocator();
        let budget = alloc.allocate(4096, 1024, 200, 100);
        let (selected, used, truncated) = alloc.select_chunks(&[], &budget);
        assert!(selected.is_empty());
        assert_eq!(used, 0);
        assert_eq!(truncated, 0);
    }

    #[test]
    fn test_single_chunk_selected() {
        let alloc = make_allocator();
        let budget = TokenBudget {
            total: 4096,
            instruction: 0,
            sources: 500,
            memory: 0,
            output_reserved: 0,
            remaining: 0,
        };

        let chunks = vec![make_hit(0, 100, 0.9)];
        let (selected, used, truncated) = alloc.select_chunks(&chunks, &budget);
        assert_eq!(selected.len(), 1);
        assert_eq!(used, 100);
        assert_eq!(truncated, 0);
    }

    #[test]
    fn test_large_context_window_no_overflow() {
        let alloc = make_allocator();
        let budget = alloc.allocate(131072, 4096, 500, 200);
        // available = 131072 - 4096 = 126976
        // sources = 126976 - 500 - 200 - 256 = 126020
        assert_eq!(budget.sources, 126020);
        assert_eq!(budget.total, 131072);
        let allocated = budget.instruction + budget.sources + budget.memory + budget.output_reserved;
        assert!(allocated <= budget.total);
    }

    #[test]
    fn test_budget_allocation_instruction_exceeds_available() {
        // instruction + memory + output > context_window → sources = 0
        let alloc = make_allocator();
        let budget = alloc.allocate(1000, 800, 300, 200);
        // available = 1000 - 800 = 200
        // sources = 200 - 300 → saturating_sub → 0
        assert_eq!(budget.sources, 0);
    }

    #[test]
    fn test_chunk_selection_all_too_large() {
        // All chunks are larger than budget — best one gets truncated
        let alloc = make_allocator();
        let budget = TokenBudget {
            total: 4096,
            instruction: 0,
            sources: 20,
            memory: 0,
            output_reserved: 0,
            remaining: 0,
        };

        let mut c0 = make_hit(0, 100, 0.9);
        c0.chunk.text = "word ".repeat(100).trim().to_string();
        c0.chunk.token_count = 100;

        let mut c1 = make_hit(1, 200, 0.8);
        c1.chunk.text = "word ".repeat(200).trim().to_string();
        c1.chunk.token_count = 200;

        let (selected, used, truncated) = alloc.select_chunks(&[c0, c1], &budget);
        assert_eq!(selected.len(), 1);
        assert!(used <= 20);
        assert_eq!(truncated, 1);
        // The truncated chunk should be chunk 0 (highest scored skipped)
        assert_eq!(selected[0].score_fused, 0.9);
    }

    #[test]
    fn test_chunk_selection_exact_budget_fill() {
        let alloc = make_allocator();
        let budget = TokenBudget {
            total: 4096,
            instruction: 0,
            sources: 300,
            memory: 0,
            output_reserved: 0,
            remaining: 0,
        };

        let chunks = vec![
            make_hit(0, 100, 0.9),
            make_hit(1, 100, 0.8),
            make_hit(2, 100, 0.7),
        ];

        let (selected, used, truncated) = alloc.select_chunks(&chunks, &budget);
        assert_eq!(selected.len(), 3);
        assert_eq!(used, 300);
        assert_eq!(truncated, 0);
    }

    #[test]
    fn test_chunk_selection_zero_budget() {
        let alloc = make_allocator();
        let budget = TokenBudget {
            total: 4096,
            instruction: 0,
            sources: 0,
            memory: 0,
            output_reserved: 0,
            remaining: 0,
        };

        let chunks = vec![make_hit(0, 100, 0.9)];
        let (selected, used, truncated) = alloc.select_chunks(&chunks, &budget);
        assert!(selected.is_empty());
        assert_eq!(used, 0);
        assert_eq!(truncated, 0);
    }
}
