//! Error classification (ARCHITECTURE 8.6). Raw backend text is only logged; the
//! caller-visible message comes from a fixed table.

use bridge_types::{BridgeError, ErrorClass};

// Caller-visible error messages. Not configurable: they are part of the API contract
// (fixed, backend-neutral text that never leaks raw backend output).
pub const MSG_CONTEXT: &str = "The conversation is too long for the model's context window.";
pub const MSG_QUOTA: &str = "The model's usage limit has been reached. Try again later.";
pub const MSG_UNAVAILABLE: &str = "The model is temporarily unavailable. Try again later.";
pub const MSG_AUTH: &str = "The model backend is not signed in or not authorised.";
pub const MSG_INTERNAL: &str = "The model backend failed to produce a response.";
pub const MSG_TIMEOUT: &str = "The model did not respond in time.";
pub const MSG_INVALID_PLAN: &str = "The conversation history could not be prepared for the model.";

pub fn error(class: ErrorClass) -> BridgeError {
    let msg = match class {
        ErrorClass::ContextTooLong => MSG_CONTEXT,
        ErrorClass::QuotaExhausted => MSG_QUOTA,
        ErrorClass::Unavailable => MSG_UNAVAILABLE,
        ErrorClass::BackendAuth => MSG_AUTH,
        ErrorClass::Timeout => MSG_TIMEOUT,
        _ => MSG_INTERNAL,
    };
    BridgeError::new(class, msg)
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Classifies raw error text from claude (result text, stderr) plus the API status if known.
/// `resets_at`: unix seconds from the last rate_limit_event, used for Retry-After.
pub fn classify(raw: &str, api_status: Option<u64>, resets_at: Option<u64>) -> BridgeError {
    let t = raw.to_ascii_lowercase();
    let has = |needles: &[&str]| needles.iter().any(|n| t.contains(n));
    let class = if has(&[
        "prompt is too long",
        "context length",
        "context window",
        "too many tokens",
        "maximum context",
        "input length and `max_tokens` exceed",
        "exceed context limit",
    ]) {
        ErrorClass::ContextTooLong
    } else if has(&[
        "not logged in",
        "please run /login",
        "invalid api key",
        "authentication_error",
        "authentication failed",
        "oauth token",
        "token has expired",
        "permission_error",
        "credit balance is too low",
    ]) || matches!(api_status, Some(401) | Some(403))
    {
        ErrorClass::BackendAuth
    } else if has(&[
        "usage limit",
        "limit reached",
        "rate limit",
        "rate_limit",
        "quota",
        "out of extra usage",
    ]) || api_status == Some(429)
    {
        ErrorClass::QuotaExhausted
    } else if has(&[
        "overloaded",
        "api error: 5",
        "internal server error",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "connection error",
        "econnreset",
        "econnrefused",
        "etimedout",
        "socket hang up",
        "network",
        "fetch failed",
        "timed out",
    ]) || matches!(api_status, Some(s) if s >= 500)
    {
        ErrorClass::Unavailable
    } else {
        ErrorClass::Internal
    };
    let mut e = error(class);
    if class == ErrorClass::QuotaExhausted {
        if let Some(r) = resets_at {
            e = e.with_retry_after(r.saturating_sub(now_secs()).max(1));
        }
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table() {
        let cases: &[(&str, Option<u64>, ErrorClass)] = &[
            ("API Error: 400 {\"type\":\"error\",\"error\":{\"message\":\"prompt is too long: 250000 tokens > 200000 maximum\"}}", Some(400), ErrorClass::ContextTooLong),
            ("Invalid API key · Please run /login", None, ErrorClass::BackendAuth),
            ("Not logged in · Please run /login", None, ErrorClass::BackendAuth),
            ("API Error: 401 authentication_error", Some(401), ErrorClass::BackendAuth),
            ("Claude AI usage limit reached|1794835200", None, ErrorClass::QuotaExhausted),
            ("You've hit your usage limit", None, ErrorClass::QuotaExhausted),
            ("API Error: 429", Some(429), ErrorClass::QuotaExhausted),
            ("API Error: 529 {\"type\":\"overloaded_error\"}", Some(529), ErrorClass::Unavailable),
            ("API Error: 500 Internal server error", Some(500), ErrorClass::Unavailable),
            ("Connection error.", None, ErrorClass::Unavailable),
            ("something odd", None, ErrorClass::Internal),
        ];
        for (raw, st, class) in cases {
            let e = classify(raw, *st, None);
            assert_eq!(e.class, *class, "{raw}");
            assert!(!e.message.to_lowercase().contains("claude"));
            assert!(!e.message.contains('/'));
        }
    }

    #[test]
    fn retry_after_from_reset() {
        let e = classify("usage limit reached", None, Some(now_secs() + 120));
        assert!(matches!(e.retry_after, Some(s) if (100..=120).contains(&s)));
        assert!(e.is_failover_candidate());
    }
}
