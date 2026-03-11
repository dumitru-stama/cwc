use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use cwc_core::traits::TokenCounter;

#[derive(Parser)]
#[command(name = "cwc-serve", about = "CWC HTTP API Server")]
struct Args {
    /// Listen address
    #[arg(long, default_value = "0.0.0.0:3000")]
    bind: String,

    /// Path to configuration file
    #[arg(long, default_value = "cwc.toml")]
    config: PathBuf,

    /// LLM server endpoint (overrides config)
    #[arg(long)]
    llm_endpoint: Option<String>,

    /// LLM backend: llamacpp or vllm
    #[arg(long, default_value = "llamacpp")]
    backend: String,

    /// Model name (for vllm backend)
    #[arg(long, default_value = "default")]
    model: String,
}

struct WordTokenizer;
impl TokenCounter for WordTokenizer {
    fn count_tokens(&self, text: &str) -> u32 {
        text.split_whitespace().count() as u32
    }
    fn truncate_to_tokens(&self, text: &str, max: u32) -> String {
        text.split_whitespace()
            .take(max as usize)
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    // Try to load config
    let config = if args.config.exists() {
        cwc_core::config::CwcConfig::load(&args.config).unwrap_or_else(|e| {
            eprintln!("Warning: failed to load {}: {e}. Using defaults.", args.config.display());
            cwc_core::config::CwcConfig::default()
        })
    } else {
        cwc_core::config::CwcConfig::default()
    };

    let tokenizer: Arc<dyn TokenCounter> = match cwc_core::Tokenizer::default_tokenizer() {
        Ok(t) => Arc::new(t),
        Err(_) => Arc::new(WordTokenizer),
    };

    // Try to build full compiler (requires indexes + LLM)
    let compiler = try_build_compiler(&config, &args, tokenizer.clone());

    let state = Arc::new(cwc_server::state::AppState {
        compiler: compiler.map(Arc::new),
        tokenizer,
    });

    let router = cwc_server::build_router(state);

    let listener = tokio::net::TcpListener::bind(&args.bind)
        .await
        .unwrap_or_else(|e| {
            eprintln!("Failed to bind to {}: {e}", args.bind);
            std::process::exit(1);
        });

    eprintln!("CWC server listening on {}", args.bind);
    eprintln!("Endpoints:");
    eprintln!("  GET  /health              — Health check");
    eprintln!("  GET  /v1/status           — Server status");
    eprintln!("  POST /v1/query            — Full query pipeline");
    eprintln!("  POST /v1/query/stream     — Streaming query (SSE)");
    eprintln!("  POST /v1/search           — Retrieval-only search");
    eprintln!("  POST /v1/session/optimize — Optimize a conversation");
    eprintln!("  POST /v1/session/status   — Check session budget");

    axum::serve(listener, router)
        .await
        .unwrap_or_else(|e| {
            eprintln!("Server error: {e}");
            std::process::exit(1);
        });
}

fn try_build_compiler(
    config: &cwc_core::config::CwcConfig,
    args: &Args,
    tokenizer: Arc<dyn TokenCounter>,
) -> Option<cwc_cli::compiler::ContextWindowCompiler> {
    use cwc_cli::builder::CompilerBuilder;

    let index_dir = PathBuf::from(&config.paths.index_dir);
    let sparse_index = cwc_index::SparseIndex::open_or_create(&index_dir).ok()?;
    let retriever: Arc<dyn cwc_core::traits::Retriever> =
        Arc::new(cwc_index::SparseRetriever::new(sparse_index));

    // Embedder — dummy for now (real embedder requires ONNX model)
    struct DummyEmbedder;
    impl cwc_core::traits::Embedder for DummyEmbedder {
        fn embed(&self, texts: &[&str]) -> cwc_core::error::Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.0; 384]).collect())
        }
        fn dim(&self) -> usize { 384 }
    }
    let embedder: Arc<dyn cwc_core::traits::Embedder> = Arc::new(DummyEmbedder);

    let endpoint = args
        .llm_endpoint
        .as_deref()
        .unwrap_or(&config.model.llm_endpoint);

    let params = cwc_llm::GenerateParams::default();
    let llm: Arc<dyn cwc_core::traits::LlmClient> = match args.backend.as_str() {
        "vllm" => Arc::new(cwc_llm::vllm::VllmClient::new(endpoint, &args.model, params)),
        _ => Arc::new(cwc_llm::llamacpp::LlamaCppClient::new(endpoint, params)),
    };

    CompilerBuilder::new(config.clone())
        .with_retriever(retriever)
        .with_embedder(embedder)
        .with_llm(llm)
        .with_tokenizer(tokenizer)
        .build()
        .ok()
}
