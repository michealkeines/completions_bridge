//! OpenAI-format errors (CONFORMANCE section 8).

use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use bridge_types::{BridgeError, ErrorClass};
use serde_json::{json, Value};

/// An error as it goes out on the wire.
#[derive(Debug, Clone, PartialEq)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
    pub kind: &'static str,
    pub param: Option<String>,
    pub code: Option<String>,
    pub retry_after: Option<u64>,
}

impl ApiError {
    /// 400 invalid_request_error with an explicit code.
    pub fn invalid(message: impl Into<String>, param: Option<&str>, code: Option<&str>) -> Self {
        ApiError {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            kind: "invalid_request_error",
            param: param.map(str::to_string),
            code: code.map(str::to_string),
            retry_after: None,
        }
    }

    pub fn not_found_route(method: &str, path: &str) -> Self {
        ApiError {
            status: StatusCode::NOT_FOUND,
            message: format!("Invalid URL ({method} {path})"),
            kind: "invalid_request_error",
            param: None,
            code: None,
            retry_after: None,
        }
    }

    pub fn method_not_allowed(method: &str, path: &str) -> Self {
        ApiError {
            status: StatusCode::METHOD_NOT_ALLOWED,
            message: format!("Method {method} is not allowed for {path}."),
            kind: "invalid_request_error",
            param: None,
            code: Some("method_not_allowed".into()),
            retry_after: None,
        }
    }

    pub fn unauthorized(message: impl Into<String>) -> Self {
        ApiError {
            status: StatusCode::UNAUTHORIZED,
            message: message.into(),
            kind: "invalid_request_error",
            param: None,
            code: Some("invalid_api_key".into()),
            retry_after: None,
        }
    }

    pub fn model_not_found(model: &str) -> Self {
        ApiError {
            status: StatusCode::NOT_FOUND,
            message: format!(
                "The model `{model}` does not exist or you do not have access to it."
            ),
            kind: "invalid_request_error",
            param: Some("model".into()),
            code: Some("model_not_found".into()),
            retry_after: None,
        }
    }

    pub fn payload_too_large(limit_mb: u64) -> Self {
        ApiError {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            message: format!("Request body too large. The maximum request size is {limit_mb} MB."),
            kind: "invalid_request_error",
            param: None,
            code: Some("request_too_large".into()),
            retry_after: None,
        }
    }

    pub fn internal() -> Self {
        ApiError::from(BridgeError::new(
            ErrorClass::Internal,
            "The server had an error while processing your request. Sorry about that!",
        ))
    }

    /// The `{"error": {...}}` body.
    pub fn body(&self) -> Value {
        json!({
            "error": {
                "message": self.message,
                "type": self.kind,
                "param": self.param,
                "code": self.code,
            }
        })
    }
}

impl From<BridgeError> for ApiError {
    fn from(e: BridgeError) -> Self {
        let (status, kind, code): (StatusCode, &'static str, Option<&str>) = match e.class {
            ErrorClass::InvalidRequest => (StatusCode::BAD_REQUEST, "invalid_request_error", None),
            ErrorClass::ContextTooLong => {
                (StatusCode::BAD_REQUEST, "invalid_request_error", Some("context_length_exceeded"))
            }
            ErrorClass::Unauthorized => {
                (StatusCode::UNAUTHORIZED, "invalid_request_error", Some("invalid_api_key"))
            }
            ErrorClass::ModelNotFound => {
                (StatusCode::NOT_FOUND, "invalid_request_error", Some("model_not_found"))
            }
            ErrorClass::NotFound => (StatusCode::NOT_FOUND, "invalid_request_error", None),
            ErrorClass::PayloadTooLarge => {
                (StatusCode::PAYLOAD_TOO_LARGE, "invalid_request_error", Some("request_too_large"))
            }
            ErrorClass::QuotaExhausted => {
                (StatusCode::TOO_MANY_REQUESTS, "insufficient_quota", Some("insufficient_quota"))
            }
            ErrorClass::RateLimited => {
                (StatusCode::TOO_MANY_REQUESTS, "requests", Some("rate_limit_exceeded"))
            }
            ErrorClass::Internal => (StatusCode::INTERNAL_SERVER_ERROR, "server_error", None),
            ErrorClass::Unavailable => {
                (StatusCode::SERVICE_UNAVAILABLE, "server_error", Some("overloaded"))
            }
            ErrorClass::BackendAuth => (StatusCode::SERVICE_UNAVAILABLE, "server_error", None),
            ErrorClass::Timeout => (StatusCode::GATEWAY_TIMEOUT, "server_error", Some("timeout")),
        };
        ApiError {
            status,
            message: e.message,
            kind,
            param: e.param,
            code: code.map(str::to_string),
            retry_after: e.retry_after,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = self.body().to_string();
        let mut resp = (self.status, body).into_response();
        let h = resp.headers_mut();
        h.insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(secs) = self.retry_after {
            if let Ok(v) = HeaderValue::from_str(&secs.to_string()) {
                h.insert(header::RETRY_AFTER, v);
            }
        }
        resp
    }
}
