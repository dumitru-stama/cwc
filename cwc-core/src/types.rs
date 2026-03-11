use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// A chunk of text with full provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    pub chunk_id: Uuid,
    pub doc_id: Uuid,
    pub doc_version: u32,
    pub source_path: String,
    pub section_path: Vec<String>,
    pub char_offset: usize,
    pub char_len: usize,
    pub token_count: u32,
    pub text: String,
    pub metadata: HashMap<String, String>,
}

/// A retrieval result with scores from multiple stages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalHit {
    pub chunk: Chunk,
    pub score_sparse: f32,
    pub score_dense: f32,
    pub score_fused: f32,
    pub score_rerank: f32,
}

/// The compiled context bundle, ready for prompt composition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompiledContext {
    pub instruction_block: String,
    pub sources: Vec<Chunk>,
    pub output_schema: Option<serde_json::Value>,
    pub budget: TokenBudget,
}

/// Token budget breakdown.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenBudget {
    pub total: u32,
    pub instruction: u32,
    pub sources: u32,
    pub memory: u32,
    pub output_reserved: u32,
    pub remaining: u32,
}

impl TokenBudget {
    /// Compute a budget from a total token count and fractional allocations.
    ///
    /// Fractions should sum to <= 1.0. `remaining` is whatever is left after
    /// all allocations.
    pub fn from_fractions(
        total: u32,
        instruction_frac: f32,
        sources_frac: f32,
        memory_frac: f32,
        output_frac: f32,
    ) -> Self {
        let instruction = (total as f32 * instruction_frac) as u32;
        let sources = (total as f32 * sources_frac) as u32;
        let memory = (total as f32 * memory_frac) as u32;
        let output_reserved = (total as f32 * output_frac) as u32;
        let allocated = instruction + sources + memory + output_reserved;
        let remaining = total.saturating_sub(allocated);
        Self {
            total,
            instruction,
            sources,
            memory,
            output_reserved,
            remaining,
        }
    }
}

/// Verification result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Verdict {
    Pass,
    Fail { issues: Vec<VerificationIssue> },
    Abstain { reason: String },
}

impl Verdict {
    pub fn is_pass(&self) -> bool {
        matches!(self, Self::Pass)
    }

    pub fn is_abstain(&self) -> bool {
        matches!(self, Self::Abstain { .. })
    }

    pub fn is_fail(&self) -> bool {
        matches!(self, Self::Fail { .. })
    }

    pub fn issues(&self) -> &[VerificationIssue] {
        match self {
            Self::Fail { issues } => issues,
            _ => &[],
        }
    }
}

/// A single verification issue found in the output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationIssue {
    pub kind: IssueKind,
    pub description: String,
    pub claim_text: Option<String>,
}

/// Categories of verification issues.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum IssueKind {
    MissingCitation,
    InvalidCitationId,
    SchemaViolation,
    UnsupportedClaim,
}

/// A chat message for multi-turn LLM interaction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

/// Chat message role.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}
