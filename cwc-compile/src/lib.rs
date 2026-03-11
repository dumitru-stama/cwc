pub mod budget;
pub mod escape;
pub mod reorder;
pub mod report;
pub mod scaffold;
pub mod schema;
pub mod source_ids;

use std::sync::Arc;

use cwc_core::config::BudgetConfig;
use cwc_core::traits::TokenCounter;
use cwc_core::types::RetrievalHit;
use uuid::Uuid;

use budget::BudgetAllocator;
use reorder::{ReorderStrategy, reorder_chunks};
use report::BudgetReport;
use source_ids::assign_source_ids;

/// Parameters for `compile_sources`.
pub struct CompileParams<'a> {
    pub hits: Vec<RetrievalHit>,
    pub context_window: u32,
    pub max_output_tokens: u32,
    pub instruction_text: &'a str,
    pub memory_text: &'a str,
    pub tokenizer: Arc<dyn TokenCounter>,
    pub config: &'a BudgetConfig,
    pub strategy: ReorderStrategy,
}

/// Compile sources: select chunks within budget, reorder, assign source IDs,
/// and produce a budget report.
pub fn compile_sources(
    params: CompileParams<'_>,
) -> (Vec<RetrievalHit>, BudgetReport, Vec<(Uuid, String)>) {
    let CompileParams {
        hits,
        context_window,
        max_output_tokens,
        instruction_text,
        memory_text,
        tokenizer,
        config,
        strategy,
    } = params;

    let instruction_tokens = tokenizer.count_tokens(instruction_text);
    let memory_tokens = tokenizer.count_tokens(memory_text);

    let allocator = BudgetAllocator::new(Arc::clone(&tokenizer), config.clone());
    let budget = allocator.allocate(context_window, max_output_tokens, instruction_tokens, memory_tokens);

    let total_input = hits.len();
    let (selected, used, chunks_truncated) = allocator.select_chunks(&hits, &budget);

    let report = BudgetReport::new(&budget, used, total_input, selected.len(), chunks_truncated);
    report.log();

    let reordered = reorder_chunks(selected, strategy);
    let source_ids = assign_source_ids(&reordered);

    (reordered, report, source_ids)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use cwc_core::types::{Chunk, RetrievalHit};
    use std::collections::HashMap;
    use uuid::Uuid;

    /// Test helper: create a RetrievalHit with deterministic chunk_id.
    pub fn make_hit(index: u32, token_count: u32, score: f32) -> RetrievalHit {
        let chunk_id = Uuid::from_u128(index as u128);
        RetrievalHit {
            chunk: Chunk {
                chunk_id,
                doc_id: Uuid::nil(),
                doc_version: 1,
                source_path: format!("doc_{index}.md"),
                section_path: vec![],
                char_offset: 0,
                char_len: (token_count * 5) as usize,
                token_count,
                text: format!("Content of chunk {index}. ").repeat(token_count as usize / 2 + 1),
                metadata: HashMap::new(),
            },
            score_sparse: score,
            score_dense: score,
            score_fused: score,
            score_rerank: score,
        }
    }

    struct FakeTokenCounter;

    impl TokenCounter for FakeTokenCounter {
        fn count_tokens(&self, text: &str) -> u32 {
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

    fn fake_tc() -> Arc<dyn TokenCounter> {
        Arc::new(FakeTokenCounter)
    }

    #[test]
    fn test_compile_sources_integration() {
        let hits: Vec<RetrievalHit> = (0..50)
            .map(|i| make_hit(i, 100, 0.9 - i as f32 * 0.01))
            .collect();

        let config = BudgetConfig::default();
        let (selected, report, source_ids) = compile_sources(CompileParams {
            hits,
            context_window: 4096,
            max_output_tokens: 1024,
            instruction_text: "Answer using sources.",
            memory_text: "Previous context here",
            tokenizer: fake_tc(),
            config: &config,
            strategy: ReorderStrategy::EdgePlacement,
        });

        assert!(!selected.is_empty());
        assert_eq!(selected.len(), source_ids.len());
        assert!(report.sources_used <= report.sources_budget);
        assert!(report.utilization > 0.0);
        assert_eq!(source_ids[0].1, "[S1]");
    }

    #[test]
    fn test_compile_sources_zero_chunks() {
        let config = BudgetConfig::default();
        let (selected, report, source_ids) = compile_sources(CompileParams {
            hits: vec![],
            context_window: 4096,
            max_output_tokens: 1024,
            instruction_text: "Instructions",
            memory_text: "",
            tokenizer: fake_tc(),
            config: &config,
            strategy: ReorderStrategy::RankOrder,
        });

        assert!(selected.is_empty());
        assert!(source_ids.is_empty());
        assert_eq!(report.chunks_selected, 0);
        assert_eq!(report.chunks_dropped, 0);
        assert_eq!(report.sources_used, 0);
    }

    #[test]
    fn test_compile_sources_single_chunk() {
        let config = BudgetConfig::default();
        let (selected, _report, source_ids) = compile_sources(CompileParams {
            hits: vec![make_hit(0, 50, 0.9)],
            context_window: 4096,
            max_output_tokens: 1024,
            instruction_text: "Inst",
            memory_text: "Mem",
            tokenizer: fake_tc(),
            config: &config,
            strategy: ReorderStrategy::EdgePlacement,
        });

        assert_eq!(selected.len(), 1);
        assert_eq!(source_ids.len(), 1);
        assert_eq!(source_ids[0].1, "[S1]");
    }

    #[test]
    fn test_compile_sources_truncation_reported() {
        let config = BudgetConfig::default();
        // Create hits where the last one will need truncation
        let hits = vec![
            make_hit(0, 100, 0.9),
            make_hit(1, 100, 0.8),
        ];

        let (selected, report, _) = compile_sources(CompileParams {
            hits,
            context_window: 1500,
            max_output_tokens: 500,
            // instruction "one two three" = 3 words/tokens, memory "four five" = 2
            instruction_text: "one two three",
            memory_text: "four five",
            tokenizer: fake_tc(),
            config: &config,
            // available = 1500 - 500 = 1000
            // sources = 1000 - 3 - 2 - 256 = 739
            // Both 100-token chunks fit easily
            strategy: ReorderStrategy::RankOrder,
        });

        assert_eq!(selected.len(), 2);
        assert_eq!(report.chunks_truncated, 0);
        assert_eq!(report.instruction_tokens, 3);
        assert_eq!(report.memory_tokens, 2);
    }

    #[test]
    fn test_compile_sources_high_utilization() {
        // Milestone target: 85%+ utilization with 50 chunks and 4096 context
        let config = BudgetConfig::default();
        let hits: Vec<RetrievalHit> = (0..50)
            .map(|i| make_hit(i, 100, 0.9 - i as f32 * 0.01))
            .collect();

        let (_, report, _) = compile_sources(CompileParams {
            hits,
            context_window: 4096,
            max_output_tokens: 1024,
            instruction_text: "short",
            memory_text: "",
            tokenizer: fake_tc(),
            config: &config,
            strategy: ReorderStrategy::EdgePlacement,
        });

        assert!(
            report.utilization >= 0.85,
            "utilization should be >= 85%, got {:.1}%",
            report.utilization * 100.0
        );
    }

    #[test]
    fn test_compile_sources_source_ids_match_reordered_order() {
        let config = BudgetConfig::default();
        let hits = vec![
            make_hit(0, 50, 0.9),
            make_hit(1, 50, 0.8),
            make_hit(2, 50, 0.7),
        ];

        let (selected, _, source_ids) = compile_sources(CompileParams {
            hits,
            context_window: 4096,
            max_output_tokens: 1024,
            instruction_text: "i",
            memory_text: "",
            tokenizer: fake_tc(),
            config: &config,
            strategy: ReorderStrategy::EdgePlacement,
        });

        // Source IDs should match reordered chunk positions
        for (i, (chunk_id, sid)) in source_ids.iter().enumerate() {
            assert_eq!(*chunk_id, selected[i].chunk.chunk_id);
            assert_eq!(*sid, format!("[S{}]", i + 1));
        }
    }
}
