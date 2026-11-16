//! The interface between the HTTP frontend and the engine.

use async_trait::async_trait;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::ir::{ChatRequest, ChoiceEvent};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Id the caller uses, e.g. "bridge/auto".
    pub id: String,
    /// Unix seconds.
    pub created: u64,
    pub owned_by: String,
    /// Context window, for documentation / admin output (not part of the OpenAI schema).
    pub context_limit: Option<u64>,
}

#[async_trait]
pub trait ChatService: Send + Sync {
    /// Models to list in `/v1/models`.
    fn models(&self) -> Vec<ModelInfo>;

    /// Run a validated request.
    ///
    /// Stream contract:
    /// - For each choice index `0..n`, events follow the `TurnEvent` ordering contract.
    /// - If the request fails before any output for any choice, the FIRST item of the
    ///   stream is `ChoiceEvent { index: 0, event: TurnEvent::Error }` and the stream ends.
    ///   The frontend turns that into a plain HTTP error response.
    /// - Failover between backends happens inside the engine before the first item.
    /// - Cancelling `cancel` (client went away) must stop backend work promptly.
    async fn chat(
        &self,
        request: ChatRequest,
        cancel: CancellationToken,
    ) -> BoxStream<'static, ChoiceEvent>;

    /// Free-form status for `/admin/status`.
    fn status(&self) -> serde_json::Value;
}
