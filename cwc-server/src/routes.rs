use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::Json;
use axum::routing::{get, post};
use axum::Router;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};

use cwc_session::config::ModelProfileConfig;
use cwc_session::{SessionManager, SessionManagerConfig};

use crate::sse::{self, SseEvent};
use crate::state::AppState;

pub fn api_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/health", get(health))
        .route("/v1/query", post(query))
        .route("/v1/query/stream", post(query_stream))
        .route("/v1/search", post(search))
        .route("/v1/session/optimize", post(session_optimize))
        .route("/v1/session/status", post(session_status))
        .route("/v1/status", get(status))
}

// --- Request/Response types ---

#[derive(Deserialize)]
pub struct QueryRequest {
    pub query: String,
    pub task_type: Option<String>,
}

#[derive(Serialize)]
pub struct QueryResponse {
    pub text: String,
    pub citations: Vec<String>,
    pub is_abstention: bool,
    pub verdict: String,
    pub attempts: usize,
    pub timing: TimingResponse,
    pub budget: BudgetResponse,
}

#[derive(Serialize)]
pub struct TimingResponse {
    pub retrieval_ms: u64,
    pub compilation_ms: u64,
    pub generation_ms: u64,
    pub verification_ms: u64,
    pub total_ms: u64,
}

#[derive(Serialize)]
pub struct BudgetResponse {
    pub context_window: u32,
    pub sources_used: u32,
    pub chunks_selected: usize,
    pub utilization: f32,
}

#[derive(Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub top_k: Option<usize>,
}

#[derive(Serialize)]
pub struct SearchHit {
    pub chunk_id: String,
    pub source_path: String,
    pub score: f32,
    pub text_preview: String,
}

#[derive(Deserialize)]
pub struct SessionOptimizeRequest {
    pub messages: Vec<serde_json::Value>,
    #[serde(default = "default_format")]
    pub format: String,
    #[serde(default = "default_model")]
    pub model: String,
    /// Enable LLM-enhanced memory consolidation.
    #[serde(default)]
    pub llm_consolidate: bool,
}

fn default_format() -> String {
    "auto".into()
}

fn default_model() -> String {
    "local_large".into()
}

#[derive(Serialize)]
pub struct SessionOptimizeResponse {
    pub messages: Vec<serde_json::Value>,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub tokens_saved: u32,
    pub trim_action: String,
    pub memory_facts: usize,
}

#[derive(Deserialize)]
pub struct SessionStatusRequest {
    pub messages: Vec<serde_json::Value>,
    #[serde(default = "default_format")]
    pub format: String,
    #[serde(default = "default_model")]
    pub model: String,
}

#[derive(Serialize)]
pub struct SessionStatusResponse {
    pub total_tokens: u32,
    pub usable_budget: u32,
    pub utilization_percent: f32,
    pub action_needed: String,
    pub turns: usize,
    pub memory_facts: usize,
}

// --- Handlers ---

async fn health() -> &'static str {
    "ok"
}

async fn status(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let has_compiler = state.compiler.is_some();
    Json(serde_json::json!({
        "status": "running",
        "compiler_available": has_compiler,
    }))
}

async fn query(
    State(state): State<Arc<AppState>>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, (StatusCode, String)> {
    let compiler = state
        .compiler
        .as_ref()
        .ok_or((StatusCode::SERVICE_UNAVAILABLE, "Compiler not configured (no LLM/indexes)".into()))?;

    let output = compiler
        .query(&req.query, None)
        .await
        .map_err(|e| {
            tracing::error!("Query failed: {e}");
            (StatusCode::INTERNAL_SERVER_ERROR, "Internal server error".to_string())
        })?;

    Ok(Json(QueryResponse {
        text: output.response.raw_text,
        citations: output.response.citations,
        is_abstention: output.response.is_abstention,
        verdict: format!("{:?}", output.verdict),
        attempts: output.attempts,
        timing: TimingResponse {
            retrieval_ms: output.timing.retrieval_ms,
            compilation_ms: output.timing.compilation_ms,
            generation_ms: output.timing.generation_ms,
            verification_ms: output.timing.verification_ms,
            total_ms: output.timing.total_ms,
        },
        budget: BudgetResponse {
            context_window: output.budget_report.context_window,
            sources_used: output.budget_report.sources_used,
            chunks_selected: output.budget_report.chunks_selected,
            utilization: output.budget_report.utilization,
        },
    }))
}

async fn query_stream(
    State(state): State<Arc<AppState>>,
    Json(req): Json<QueryRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, String)> {
    let compiler = state
        .compiler
        .clone()
        .ok_or((StatusCode::SERVICE_UNAVAILABLE, "Compiler not configured".into()))?;

    let (tx, sse_response) = sse::token_stream();

    // Spawn the query in a background task so SSE starts immediately
    let query_text = req.query;
    tokio::spawn(async move {
        let tx_clone = tx.clone();
        let on_token: cwc_core::traits::StreamCallback = Box::new(move |chunk: &str| {
            if tx_clone.try_send(SseEvent::Token(chunk.to_string())).is_err() {
                tracing::warn!("SSE channel full, token dropped");
            }
        });

        match compiler.query_streaming(&query_text, None, on_token).await {
            Ok(output) => {
                let result = serde_json::json!({
                    "text": output.response.raw_text,
                    "citations": output.response.citations,
                    "is_abstention": output.response.is_abstention,
                    "verdict": format!("{:?}", output.verdict),
                    "attempts": output.attempts,
                    "timing": {
                        "retrieval_ms": output.timing.retrieval_ms,
                        "compilation_ms": output.timing.compilation_ms,
                        "generation_ms": output.timing.generation_ms,
                        "verification_ms": output.timing.verification_ms,
                        "total_ms": output.timing.total_ms,
                    },
                });
                let _ = tx.send(SseEvent::Done(result.to_string())).await;
            }
            Err(e) => {
                tracing::error!("Streaming query failed: {e}");
                let _ = tx.send(SseEvent::Error("Internal server error".to_string())).await;
            }
        }
    });

    Ok(sse_response)
}

async fn search(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<Vec<SearchHit>>, (StatusCode, String)> {
    let compiler = state
        .compiler
        .as_ref()
        .ok_or((StatusCode::SERVICE_UNAVAILABLE, "Compiler not configured".into()))?;

    let config = compiler.config();
    let hits = compiler
        .retriever()
        .retrieve(&req.query, req.top_k.unwrap_or(config.retrieval.final_top_k))
        .map_err(|e| {
            tracing::error!("Search failed: {e}");
            (StatusCode::INTERNAL_SERVER_ERROR, "Internal server error".to_string())
        })?;

    Ok(Json(
        hits.iter()
            .map(|h| SearchHit {
                chunk_id: h.chunk.chunk_id.to_string(),
                source_path: h.chunk.source_path.clone(),
                score: h.score_fused,
                text_preview: h.chunk.text.chars().take(200).collect(),
            })
            .collect(),
    ))
}

async fn session_optimize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SessionOptimizeRequest>,
) -> Result<Json<SessionOptimizeResponse>, (StatusCode, String)> {
    let parsed = parse_session_msgs(&req.format, &req.messages, &*state.tokenizer)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let config = SessionManagerConfig {
        session: cwc_session::SessionConfig {
            model: ModelProfileConfig::Preset(req.model),
            ..Default::default()
        },
        llm_consolidation: cwc_session::LlmConsolidationConfig {
            enabled: req.llm_consolidate,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut mgr = SessionManager::new(config, state.tokenizer.clone())
        .map_err(|e| {
            tracing::error!("SessionManager init failed: {e}");
            (StatusCode::INTERNAL_SERVER_ERROR, "Internal server error".to_string())
        })?;

    // Wire LLM client from compiler if available (needed for LLM consolidation)
    if req.llm_consolidate {
        if let Some(compiler) = &state.compiler {
            mgr.set_llm(compiler.llm().clone());
        }
    }

    let (mut optimized, report) = mgr
        .optimize_messages(parsed)
        .map_err(|e| {
            tracing::error!("Session optimize failed: {e}");
            (StatusCode::INTERNAL_SERVER_ERROR, "Internal server error".to_string())
        })?;

    // Run async LLM consolidation post-step if enabled
    if req.llm_consolidate {
        let _llm_report = mgr.llm_consolidate().await;
        // Refresh the memory message to reflect LLM consolidation results
        mgr.refresh_memory_in_messages(&mut optimized);
    }

    let output_json = cwc_session::format::openai::to_openai(&optimized);

    Ok(Json(SessionOptimizeResponse {
        messages: output_json,
        input_tokens: report.input_tokens,
        output_tokens: report.output_tokens,
        tokens_saved: report.tokens_saved,
        trim_action: format!("{:?}", report.trim),
        memory_facts: report.memory_facts,
    }))
}

async fn session_status(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SessionStatusRequest>,
) -> Result<Json<SessionStatusResponse>, (StatusCode, String)> {
    let parsed = parse_session_msgs(&req.format, &req.messages, &*state.tokenizer)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let config = SessionManagerConfig {
        session: cwc_session::SessionConfig {
            model: ModelProfileConfig::Preset(req.model),
            ..Default::default()
        },
        ..Default::default()
    };

    let mgr = SessionManager::new(config, state.tokenizer.clone())
        .map_err(|e| {
            tracing::error!("SessionManager init failed: {e}");
            (StatusCode::INTERNAL_SERVER_ERROR, "Internal server error".to_string())
        })?;

    let mut session = cwc_session::Session::new(state.tokenizer.clone());
    for msg in parsed {
        session.messages_mut().push(msg);
    }
    session.recalculate_total();

    let status = mgr.budget_status(&session);

    Ok(Json(SessionStatusResponse {
        total_tokens: status.total_tokens,
        usable_budget: status.usable_budget,
        utilization_percent: status.utilization_percent,
        action_needed: format!("{:?}", status.action_needed),
        turns: status.turns,
        memory_facts: status.memory_facts,
    }))
}

fn parse_session_msgs(
    format: &str,
    messages: &[serde_json::Value],
    tokenizer: &dyn cwc_core::traits::TokenCounter,
) -> cwc_session::Result<Vec<cwc_session::SessionMessage>> {
    match format {
        "openai" => cwc_session::format::openai::from_openai(messages, tokenizer),
        "anthropic" => cwc_session::format::anthropic::from_anthropic(None, messages, tokenizer),
        "auto" | "" => cwc_session::format::auto_detect_and_parse(messages, None, tokenizer),
        other => Err(cwc_session::SessionError::InvalidFormat(format!(
            "unknown format: {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_format() {
        assert_eq!(default_format(), "auto");
    }

    #[test]
    fn test_default_model() {
        assert_eq!(default_model(), "local_large");
    }
}
