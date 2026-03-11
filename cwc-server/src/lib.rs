pub mod routes;
pub mod state;
pub mod sse;

use std::sync::Arc;

use axum::extract::DefaultBodyLimit;
use axum::Router;
use tower_http::cors::{Any, CorsLayer};

use state::AppState;

/// Build the Axum router with all CWC endpoints.
///
/// CORS is configured to allow any origin by default. This is suitable for
/// local/trusted network use. For production deployments exposed to the
/// internet, configure a reverse proxy with restrictive CORS policies.
pub fn build_router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
        .allow_headers(Any);

    Router::new()
        .merge(routes::api_routes())
        .layer(cors)
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024)) // 10 MB request body limit
        .with_state(state)
}
