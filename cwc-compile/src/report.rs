use serde::{Deserialize, Serialize};

use cwc_core::types::TokenBudget;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BudgetReport {
    pub context_window: u32,
    pub max_output: u32,
    pub instruction_tokens: u32,
    pub memory_tokens: u32,
    pub sources_budget: u32,
    pub sources_used: u32,
    pub chunks_selected: usize,
    pub chunks_dropped: usize,
    pub chunks_truncated: usize,
    pub utilization: f32,
}

impl BudgetReport {
    pub fn new(
        budget: &TokenBudget,
        sources_used: u32,
        total_input_chunks: usize,
        chunks_selected: usize,
        chunks_truncated: usize,
    ) -> Self {
        let utilization = if budget.sources > 0 {
            sources_used as f32 / budget.sources as f32
        } else {
            0.0
        };

        Self {
            context_window: budget.total,
            max_output: budget.output_reserved,
            instruction_tokens: budget.instruction,
            memory_tokens: budget.memory,
            sources_budget: budget.sources,
            sources_used,
            chunks_selected,
            chunks_dropped: total_input_chunks.saturating_sub(chunks_selected),
            chunks_truncated,
            utilization,
        }
    }

    pub fn log(&self) {
        tracing::info!(
            context_window = self.context_window,
            max_output = self.max_output,
            instruction_tokens = self.instruction_tokens,
            memory_tokens = self.memory_tokens,
            sources_budget = self.sources_budget,
            sources_used = self.sources_used,
            chunks_selected = self.chunks_selected,
            chunks_dropped = self.chunks_dropped,
            chunks_truncated = self.chunks_truncated,
            utilization = format!("{:.1}%", self.utilization * 100.0),
            "budget report"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_budget_report_utilization() {
        let budget = TokenBudget {
            total: 4096,
            instruction: 200,
            sources: 2000,
            memory: 100,
            output_reserved: 1024,
            remaining: 772,
        };
        let report = BudgetReport::new(&budget, 1700, 50, 10, 1);
        assert!((report.utilization - 0.85).abs() < 0.01);
        assert_eq!(report.sources_budget, 2000);
        assert_eq!(report.sources_used, 1700);
    }

    #[test]
    fn test_budget_report_chunks_dropped() {
        let budget = TokenBudget {
            total: 4096,
            instruction: 200,
            sources: 500,
            memory: 100,
            output_reserved: 1024,
            remaining: 0,
        };
        let report = BudgetReport::new(&budget, 450, 20, 5, 0);
        assert_eq!(report.chunks_dropped, 15);
        assert_eq!(report.chunks_selected, 5);
    }

    #[test]
    fn test_budget_report_zero_budget() {
        let budget = TokenBudget {
            total: 0,
            instruction: 0,
            sources: 0,
            memory: 0,
            output_reserved: 0,
            remaining: 0,
        };
        let report = BudgetReport::new(&budget, 0, 0, 0, 0);
        assert_eq!(report.utilization, 0.0);
        assert_eq!(report.chunks_dropped, 0);
    }
}
