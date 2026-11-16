//! Error classes shared by every layer. The frontend maps them to HTTP/OpenAI errors.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorClass {
    /// Bad request from the caller (400). `param` should be set when known.
    InvalidRequest,
    /// Auth failure on the bridge's own API (401).
    Unauthorized,
    /// Unknown model id (404).
    ModelNotFound,
    /// Unknown route (404).
    NotFound,
    /// Request body too large (413).
    PayloadTooLarge,
    /// Conversation exceeds the backend's context window (400 context_length_exceeded).
    ContextTooLong,
    /// Every eligible backend is out of quota (429 insufficient_quota, Retry-After).
    QuotaExhausted,
    /// Too many concurrent requests (429 rate_limit_exceeded, Retry-After).
    RateLimited,
    /// Backend overloaded or a transient upstream failure (503, retryable).
    Unavailable,
    /// Backend not logged in / misconfigured (503).
    BackendAuth,
    /// No first token in time (504).
    Timeout,
    /// Anything else (500).
    Internal,
}

/// An error that may be shown to the caller. `message` must already be safe to show:
/// no backend names, paths, session ids or raw upstream text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, thiserror::Error)]
#[error("{class:?}: {message}")]
pub struct BridgeError {
    pub class: ErrorClass,
    pub message: String,
    /// Request field the error refers to (OpenAI `param`).
    pub param: Option<String>,
    /// Seconds until retrying makes sense (Retry-After), if known.
    pub retry_after: Option<u64>,
}

impl BridgeError {
    pub fn new(class: ErrorClass, message: impl Into<String>) -> Self {
        BridgeError { class, message: message.into(), param: None, retry_after: None }
    }

    pub fn invalid(message: impl Into<String>, param: Option<&str>) -> Self {
        BridgeError {
            class: ErrorClass::InvalidRequest,
            message: message.into(),
            param: param.map(str::to_string),
            retry_after: None,
        }
    }

    pub fn with_retry_after(mut self, secs: u64) -> Self {
        self.retry_after = Some(secs);
        self
    }

    /// Whether the engine may transparently retry on another backend
    /// (only valid before any output reached the caller).
    pub fn is_failover_candidate(&self) -> bool {
        matches!(
            self.class,
            ErrorClass::QuotaExhausted
                | ErrorClass::RateLimited
                | ErrorClass::Unavailable
                | ErrorClass::BackendAuth
                | ErrorClass::Timeout
        )
    }
}
