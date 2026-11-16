//! OpenAI chat completions frontend (CONFORMANCE.md sections 1–8).
//!
//! PUBLIC API IS FROZEN: `router`.

use std::sync::Arc;

use bridge_types::{ChatService, Config};

mod error;
mod parse;
mod response;
mod server;

/// Build the axum router: /v1/chat/completions, /v1/models, /v1/models/{id},
/// /admin/status, plus OpenAI-format 404/405 fallbacks.
pub fn router(service: Arc<dyn ChatService>, config: &Config) -> axum::Router {
    server::build(service, config)
}
