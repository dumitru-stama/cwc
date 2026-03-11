# CWC — Context Window Compiler

A Rust toolkit for building accurate, private AI systems on local LLMs. Two capabilities in one package:

1. **Document grounding** — retrieves evidence from your documents, fits it into a token budget, generates answers with citations, and verifies the output.
2. **Agent session management** — keeps AI agents focused during long multi-tool tasks by compacting tool output, extracting memory, trimming context, and reinforcing goals.

Everything runs on your hardware. Nothing leaves the building.

## Quick start

```bash
# Build
cargo build --release --workspace

# Index your documents
cwc ingest --path ./docs --recursive

# Ask a question (full pipeline: retrieve -> budget -> generate -> verify)
cwc query "What CDC strategy was used for PCIe on the Kintex board?"

# Interactive REPL with streaming
cwc repl

# Start the HTTP API server
cwc-serve --bind 0.0.0.0:3000 --llm-endpoint http://localhost:8080

# Optimize an agent conversation (no LLM needed)
cwc session optimize conversation.json --format openai --model local_large
```

## What you need

| Requirement | Notes |
|-------------|-------|
| Rust 1.75+ | Edition 2021 |
| LLM server | llama.cpp, vLLM, or any OpenAI-compatible endpoint. CWC is a client, not a runtime. |
| Embedding model | ONNX format, ~100MB. Download separately (not included). |
| PostgreSQL + pgvector | Only for dense vector storage. In-memory HNSW mode available for no-DB setups. |

See `examples/` for sample configuration files.

## How it works

### Document pipeline

```
Your Documents ──> Ingest ──> Index (BM25 + Vector)
                                       │
User Query ──> Hybrid Retrieval ──> Rerank ──> RRF + MMR
                                                   │
                                    Token Budget + Reorder ──> Prompt Scaffold
                                                                     │
                                                     LLM + Constrained Decode
                                                                     │
                                                              Verify ──> Output
```

| Stage | What happens |
|-------|-------------|
| **Ingest** | Reads txt/md/json/code files. Chunks by token count and section boundaries. Indexes by keywords (BM25 via Tantivy) and meaning (dense vectors). |
| **Retrieve** | Searches both indexes. Combines with Reciprocal Rank Fusion. Removes near-duplicates with MMR. |
| **Rerank** | Optional cross-encoder reranking (ONNX) in complex mode. |
| **Budget** | Allocates tokens across instruction, sources, memory, and output. Selects the highest-scoring chunks that fit. |
| **Reorder** | Places strongest evidence at positions 1 and N (edge placement) to counteract "lost in the middle." |
| **Scaffold** | Wraps sources in clear delimiters with citation rules. Supports bracket, XML, and markdown formats. |
| **Generate** | Calls your LLM with optional GBNF grammar constraints for structured output. Supports streaming (SSE). |
| **Verify** | Checks citations against source text, validates JSON schema, detects abstention. Revision loop retries on failure. |

### Agent session management

When AI agents use tools over many turns, their context fills with raw output and the model degrades — repeating actions, forgetting goals, hallucinating. CWC's session manager fixes this:

| Layer | What it does | When |
|-------|-------------|------|
| **Tool compaction** | Shrinks 20KB tool outputs to ~800 byte summaries. Full output stored as retrievable artifacts. | Every tool result |
| **Memory extraction** | Pulls key findings deterministically from each tool result. No LLM call needed. | Every tool result |
| **Goal anchoring** | Keeps the original user request visible at all times. | Continuously |
| **Reinforcement** | Injects brief nudge messages between turns to keep the model on track. | Every N tool results |
| **Sliding window** | Drops oldest turns when context reaches 50% of budget. Memory and goals survive. | At threshold |
| **Hard reset** | Rebuilds entire context from extracted memory when sliding window isn't enough. | At 60% threshold |
| **Pre-flight** | Catches orphan tool results, infinite loops, and runaway tool calls before they reach the model. | Every optimize call |
| **Memory consolidation** | Merges groups of related facts into denser summaries (heuristic or LLM-enhanced). | During optimize |

Result: agents run 3-5x more tool calls before degrading, especially smaller local models.

## Three ways to use it

### 1. Command line

```bash
# Document pipeline
cwc init                              # Initialize project
cwc ingest --path ./docs --recursive  # Index documents
cwc query "your question"             # Full pipeline
cwc search hybrid "ownership"         # Search only (no LLM)
cwc repl                              # Interactive REPL with streaming
cwc repl --session conv.json          # REPL with session management

# Session management
cwc session optimize conv.json --format openai --model local_large --report
cwc session status conv.json --format openai
cwc session memory conv.json
cwc session compact conv.json --output compacted.json
cwc session eval --model local_large
cwc session recommend --context-window 32768 --max-output 4096
```

### 2. HTTP API

Run `cwc-serve` and call from any language:

```bash
cwc-serve --bind 0.0.0.0:3000 --config cwc.toml --llm-endpoint http://localhost:8080
```

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/v1/status` | GET | Server capabilities |
| `/v1/query` | POST | Full pipeline — answer with citations |
| `/v1/query/stream` | POST | Same, but tokens stream back as SSE |
| `/v1/search` | POST | Retrieval only — no LLM call |
| `/v1/session/optimize` | POST | Optimize a conversation for token efficiency |
| `/v1/session/status` | POST | Check context budget utilization |

Session endpoints work without a configured LLM or index — useful standalone for any AI application.

**Query:**
```bash
curl -X POST http://localhost:3000/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is ownership in Rust?"}'
```

**Response:**
```json
{
  "text": "Ownership is Rust's system for...",
  "citations": ["[Source 1: ownership.md]"],
  "is_abstention": false,
  "verdict": "Pass",
  "timing": { "retrieval_ms": 12, "generation_ms": 850, "total_ms": 866 },
  "budget": { "context_window": 8192, "chunks_selected": 5, "utilization": 0.73 }
}
```

**Streaming (SSE):**
```bash
curl -N -X POST http://localhost:3000/v1/query/stream \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is ownership in Rust?"}'
```

Events: `token` (generated text chunk), `done` (final JSON with citations/verdict), `error`.

**Session optimization (works without LLM):**
```bash
curl -X POST http://localhost:3000/v1/session/optimize \
  -H 'Content-Type: application/json' \
  -d '{"messages": [...], "format": "openai", "model": "local_large"}'
```

### 3. Rust library

Add only the crates you need:

```toml
[dependencies]
cwc-core    = { path = "cwc-core" }
cwc-session = { path = "cwc-session" }   # Agent session management
# cwc-index   = { path = "cwc-index" }   # Retrieval
# cwc-compile = { path = "cwc-compile" } # Token budget + scaffold
# cwc-cli     = { path = "cwc-cli" }     # Full pipeline orchestrator
```

**One-shot session optimization (simplest):**

```rust
use cwc_session::{optimize, SessionManagerConfig};

let messages: Vec<serde_json::Value> = load_conversation();
let config = SessionManagerConfig::default();
let (optimized, report) = optimize(&messages, config)?;

// optimized is Vec<serde_json::Value> in OpenAI format — pass to your LLM
println!("Saved {} tokens", report.tokens_saved);
```

**Incremental processing (for agent loops):**

```rust
use std::sync::Arc;
use cwc_session::{SessionManager, SessionManagerConfig, Session, SessionMessage, SessionRole};

let config = SessionManagerConfig::default();
let mut mgr = SessionManager::new(config, tokenizer.clone())?;
let mut session = Session::new(tokenizer);

loop {
    // Add user/tool messages
    let msg = SessionMessage::text(SessionRole::User, &user_input);
    let report = mgr.process_message(&mut session, msg)?;

    // Context is automatically compacted, trimmed, and reinforced
    let messages = session.messages();
    let response = call_your_llm(messages);

    let reply = SessionMessage::text(SessionRole::Assistant, &response);
    mgr.process_message(&mut session, reply)?;
}
```

**Format conversion (OpenAI / Anthropic / auto-detect):**

```rust
use cwc_session::format;

// Auto-detect and normalize
let session_msgs = format::auto_detect_and_parse(&msgs, None, &*tokenizer)?;

// Convert back to OpenAI format
let openai_output = format::openai::to_openai(&session_msgs);
```

**Budget monitoring:**

```rust
let status = mgr.budget_status(&session);
println!("Utilization: {:.1}%", status.utilization_percent);
println!("Action: {:?}", status.action_needed);
println!("Memory facts: {}", status.memory_facts);
```

## Configuration

CWC uses a TOML config file (`cwc.toml`). See `examples/` for ready-to-use templates:

| Example | Use case |
|---------|----------|
| `cwc.toml` | Minimal config — sensible defaults, get started fast |
| `cwc-local-7b.toml` | Small local model (7-8B) with aggressive context management |
| `cwc-local-70b.toml` | Large local model (70B) with full pipeline |
| `cwc-session-only.toml` | Session optimization only — no document pipeline, no LLM needed |
| `cwc-cloud.toml` | Cloud-class model (GPT-4 / Claude Opus) with minimal compaction |

### Key config sections

```toml
[mode]
mode = "simple"                  # "simple" (heuristic verify) or "complex" (cross-encoder + CoVe)

[model]
llm_endpoint = "http://localhost:8080"
embedding_model = "models/bge-small-en-v1.5.onnx"
context_window = 4096
max_output_tokens = 1024

[retrieval]
dense_top_k = 50                 # Candidates from vector search
sparse_top_k = 50                # Candidates from BM25
final_top_k = 10                 # Final chunks after fusion + MMR
mmr_lambda = 0.7                 # Diversity vs relevance (0=diverse, 1=relevant)

[budget]
instruction_fraction = 0.15      # Token budget split
sources_fraction = 0.65
memory_fraction = 0.10
output_fraction = 0.10

[verify]
check_citations = true
check_schema = true
check_abstention = true

[paths]
data_dir = "data"
index_dir = "index"
models_dir = "models"
```

### Model profiles for session management

Pre-tuned profiles match common model sizes. Pass to CLI with `--model` or use `recommend_config()` in code:

| Profile | Models | Context | Compaction | Nudge frequency |
|---------|--------|---------|------------|-----------------|
| `local_small` | 7-8B (Mistral, Llama-8B) | 8K | Aggressive | Every tool result |
| `local_medium` | 14-32B (Qwen-32B, Codestral) | 32K | Moderate | Every tool result |
| `local_large` | 70B (Llama-70B, Mixtral) | 128K | Moderate | Every 2 results |
| `cloud_weak` | Haiku, GPT-3.5 | 32K | Moderate | Every 3 results |
| `cloud_strong` | Opus, GPT-4, Sonnet | 200K | Minimal | Disabled |

## Crate map

Pick the pieces you need. Each crate is focused and independently useful.

```
cwc-core          Shared types, traits, config, tokenizer          (always needed)
  │
  ├── cwc-ingest      Document loading + chunking
  ├── cwc-index       BM25 (Tantivy) + vector (pgvector/HNSW) indexes
  ├── cwc-embed       ONNX embedding model inference + cache
  ├── cwc-retrieve    Hybrid retrieval, RRF fusion, MMR diversity
  ├── cwc-compile     Token budget allocation, reordering, prompt scaffold
  ├── cwc-llm         LLM client (llama.cpp, vLLM), GBNF grammars, streaming
  ├── cwc-verify      Citation checking, schema validation, abstention detection
  ├── cwc-memory      Conversation history + long-term persistent memory
  ├── cwc-eval        Evaluation harness, retrieval/generation metrics, benchmarks
  ├── cwc-session     Agent session management (standalone, only needs cwc-core)
  │     └── memory consolidation, format conversion, eval/tuning
  ├── cwc-cli         CLI binary + ContextWindowCompiler orchestrator library
  └── cwc-server      HTTP API server (Axum, REST + SSE streaming)
```

## Use cases

**Searching internal documents** — Index your docs with `cwc ingest`, query with `cwc query`. Every answer cites its source with file and line number. When evidence is missing, CWC says "insufficient evidence" instead of guessing.

**Keeping agents focused** — Wrap your agent loop with CWC's session manager. Tool outputs get compacted, key findings get extracted into memory, goals stay visible. A 14B model goes from degrading at turn 8 to staying coherent for 25+ tool calls.

**Cutting token costs** — Run `optimize()` on conversations before sending to any LLM. On real conversations, this typically saves 30-50% of tokens while improving quality on long threads (less noise for the model to wade through).

**Streaming answers in a web UI** — Point your frontend at `cwc-serve`'s SSE endpoint. Tokens stream back in real time. Standard SSE format, works with any language.

**Verifying AI output** — Use `cwc-verify` standalone to check any LLM's output for fabricated citations, schema violations, and unsupported claims.

**Format conversion** — CWC auto-detects OpenAI and Anthropic message formats, normalizes them, and converts back. Handles edge cases (empty content blocks, orphan tool results, system message differences).

**Just the search** — Take only `cwc-index` for hybrid BM25 + semantic search. Feed results to your own prompt builder.

**Just the budget math** — Take only `cwc-compile` to select and order chunks within a token budget.

## Why local LLMs?

| | Cloud AI | CWC + Local Model |
|-|----------|--------------------|
| **Data privacy** | Documents sent to external servers | Everything on your hardware |
| **Air-gap** | Requires internet | Fully offline |
| **Reproducibility** | Model updates change behavior silently | You control the model version |
| **Auditability** | Black box | Every decision logged and traceable |
| **Cost at scale** | Per-token API fees | Fixed hardware cost, unlimited queries |
| **Citation tracking** | "Trust me" | Every claim linked to a source |

CWC also works with cloud APIs if you prefer — the session management and verification features are useful regardless of where your model runs.

## Testing

```bash
cargo test --workspace                     # 1044 unit tests (all pass)
cargo test --workspace -- --ignored        # Integration tests (needs PostgreSQL + ONNX models)
cargo clippy --workspace --all-targets     # 0 warnings
```

## Project quality

- 1,044 tests passing, 35 ignored (require external services)
- Zero compiler warnings, zero clippy warnings
- 7 rounds of code audits with all issues fixed
- No TODOs, FIXMEs, or HACKs in the codebase
- No hardcoded paths, no embedded secrets

## Known limitations

- Embedding models are not included — download ONNX models separately (e.g., BGE-small-en-v1.5)
- PostgreSQL with pgvector required for dense vector storage (in-memory HNSW available as fallback)
- LLM server must be running externally — CWC is a client, not a model runtime
- Streaming does not support the verification revision loop (verifies once after full response)
- `task_type` field is accepted but not yet wired to automatic schema selection

## License

[MIT](LICENSE)
