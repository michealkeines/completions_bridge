//! Routes, middleware (auth, request id), handlers, SSE writer.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, Path, Request, State};
use axum::http::{header, HeaderMap, HeaderValue, Method, Uri};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bridge_types::{ChatService, ChoiceEvent, Config, ModelInfo, TurnEvent};
use futures::StreamExt;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::error::ApiError;
use crate::parse::{parse_request, Limits};
use crate::response::{
    new_completion_id, now_unix, sse_ping, Collected, Collector, Meta, Step, StreamEncoder,
};

pub(crate) struct AppState {
    service: Arc<dyn ChatService>,
    token: Option<String>,
    max_body_mb: u64,
    max_body_bytes: usize,
    max_n: u32,
    reasoning_to_caller: bool,
    header_wait: Duration,
    ping_every: Duration,
    /// Oversize body drain bounds (`server.oversize_drain_mb` / `server.oversize_drain_s`).
    drain: Drain,
}

/// How much of a too-large body is read and discarded before answering 413, and for how long.
#[derive(Clone, Copy)]
struct Drain {
    max_bytes: usize,
    max_time: Duration,
}

type Shared = Arc<AppState>;

/// Cancels the token when dropped: client disconnect (body or handler future dropped)
/// and normal completion both release backend work.
struct CancelOnDrop(CancellationToken);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

pub fn build(service: Arc<dyn ChatService>, config: &Config) -> Router {
    // `keepalive_ping_s: 0` cannot mean "never ping" (an idle SSE stream would be cut by
    // proxies/clients), so it falls back to a 15 s interval.
    let ping_s = if config.server.keepalive_ping_s == 0 { 15 } else { config.server.keepalive_ping_s };
    let limit_bytes = usize::try_from(config.server.max_body_mb.saturating_mul(1024 * 1024)).unwrap_or(usize::MAX);
    let state: Shared = Arc::new(AppState {
        service,
        token: config.server.token.clone().filter(|t| !t.is_empty()),
        max_body_mb: config.server.max_body_mb,
        max_body_bytes: limit_bytes,
        max_n: config.output.max_n,
        reasoning_to_caller: config.output.reasoning_to_caller,
        header_wait: Duration::from_millis(config.server.header_wait_ms),
        ping_every: Duration::from_secs(ping_s),
        drain: Drain {
            max_bytes: usize::try_from(config.server.oversize_drain_mb.saturating_mul(1024 * 1024))
                .unwrap_or(usize::MAX),
            max_time: Duration::from_secs(config.server.oversize_drain_s),
        },
    });

    let mut router = Router::new();
    for prefix in ["/v1", ""] {
        router = router
            .route(&format!("{prefix}/chat/completions"), post(chat_completions))
            .route(&format!("{prefix}/chat/completions/"), post(chat_completions))
            .route(&format!("{prefix}/models"), get(list_models))
            .route(&format!("{prefix}/models/"), get(list_models))
            .route(&format!("{prefix}/models/{{*id}}"), get(get_model));
    }
    router
        .route("/admin/status", get(admin_status))
        .route("/admin/status/", get(admin_status))
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed)
        .layer(DefaultBodyLimit::max(limit_bytes))
        .layer(middleware::from_fn_with_state(state.clone(), auth_and_request_id))
        .with_state(state)
}

// ------------------------------------------------------------------ middleware

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn presented_key(headers: &HeaderMap) -> Option<String> {
    if let Some(v) = headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()) {
        let v = v.trim();
        let (scheme, rest) = v.split_once(' ').unwrap_or((v, ""));
        if scheme.eq_ignore_ascii_case("bearer") {
            return Some(rest.trim().to_string());
        }
    }
    headers.get("api-key").and_then(|v| v.to_str().ok()).map(|s| s.trim().to_string())
}

async fn auth_and_request_id(State(st): State<Shared>, req: Request, next: Next) -> Response {
    let request_id = format!("req_{}", uuid::Uuid::new_v4().simple());
    let mut resp = match &st.token {
        Some(token) => match presented_key(req.headers()) {
            None => ApiError::unauthorized(
                "You didn't provide an API key. You need to provide your API key in an Authorization \
                 header using Bearer auth (i.e. Authorization: Bearer YOUR_KEY).",
            )
            .into_response(),
            Some(k) if !ct_eq(k.as_bytes(), token.as_bytes()) => {
                ApiError::unauthorized("Incorrect API key provided.").into_response()
            }
            Some(_) => next.run(req).await,
        },
        None => next.run(req).await,
    };
    if let Ok(v) = HeaderValue::from_str(&request_id) {
        resp.headers_mut().insert("x-request-id", v);
    }
    resp
}

// ------------------------------------------------------------------ simple routes

async fn not_found(method: Method, uri: Uri) -> Response {
    ApiError::not_found_route(method.as_str(), uri.path()).into_response()
}

async fn method_not_allowed(method: Method, uri: Uri) -> Response {
    ApiError::method_not_allowed(method.as_str(), uri.path()).into_response()
}

fn model_json(m: &ModelInfo) -> Value {
    json!({"id": m.id, "object": "model", "created": m.created, "owned_by": m.owned_by})
}

async fn list_models(State(st): State<Shared>) -> Response {
    let data: Vec<Value> = st.service.models().iter().map(model_json).collect();
    Json(json!({"object": "list", "data": data})).into_response()
}

async fn get_model(State(st): State<Shared>, Path(id): Path<String>) -> Response {
    let id = id.trim_end_matches('/');
    let models = st.service.models();
    if id.is_empty() {
        let data: Vec<Value> = models.iter().map(model_json).collect();
        return Json(json!({"object": "list", "data": data})).into_response();
    }
    match models.iter().find(|m| m.id == id) {
        Some(m) => Json(model_json(m)).into_response(),
        None => {
            let mut e = ApiError::model_not_found(id);
            e.message = format!("The model '{id}' does not exist");
            e.into_response()
        }
    }
}

async fn admin_status(State(st): State<Shared>) -> Response {
    Json(st.service.status()).into_response()
}

// ------------------------------------------------------------------ chat

/// Read the body up to `limit` bytes. On overflow, keep reading and discarding (bounded in
/// size and time) before answering 413: replying while the client is still uploading makes
/// the socket close with unread data, and clients then see a connection reset instead of
/// the 413. The drain is bounded by `drain` (`server.oversize_drain_mb` / `_s`).
async fn read_body(body: Body, limit: usize, limit_mb: u64, drain: Drain) -> Result<Bytes, ApiError> {
    let mut stream = body.into_data_stream();
    let mut buf = bytes::BytesMut::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_| ApiError::invalid("Failed to read the request body.", None, None))?;
        if buf.len() + chunk.len() > limit {
            let mut drained = buf.len() + chunk.len();
            drop(buf);
            let _ = tokio::time::timeout(drain.max_time, async {
                while drained < drain.max_bytes {
                    match stream.next().await {
                        Some(Ok(c)) => drained += c.len(),
                        _ => break,
                    }
                }
            })
            .await;
            return Err(ApiError::payload_too_large(limit_mb));
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(buf.freeze())
}

async fn chat_completions(State(st): State<Shared>, req: Request) -> Response {
    let body = match read_body(req.into_body(), st.max_body_bytes, st.max_body_mb, st.drain).await {
        Ok(b) => b,
        Err(e) => return e.into_response(),
    };

    let models = st.service.models();
    let req = match parse_request(&body, &models, Limits { max_n: st.max_n }) {
        Ok(r) => r,
        Err(e) => return e.into_response(),
    };
    drop(body);

    let meta = Meta {
        id: new_completion_id(),
        created: now_unix(),
        model: req.model.clone(),
        reasoning_to_caller: st.reasoning_to_caller,
        include_usage: req.include_usage,
    };
    let n = req.n;
    let streaming = req.stream;

    let cancel = CancellationToken::new();
    let guard = CancelOnDrop(cancel.clone());
    let events = st.service.chat(req, cancel).await;

    if streaming {
        stream_response(&st, events, meta, n, guard).await
    } else {
        collect_response(events, meta, n, guard).await
    }
}

type EventStream = futures::stream::BoxStream<'static, ChoiceEvent>;

async fn collect_response(mut events: EventStream, meta: Meta, n: u32, guard: CancelOnDrop) -> Response {
    let mut col = Collector::new(meta, n);
    while let Some(ev) = events.next().await {
        match col.push(ev) {
            Collected::Continue => {}
            Collected::AllDone => break,
            Collected::Failed(e) => return e.into_response(),
        }
    }
    drop(events);
    drop(guard);
    if !col.is_complete() {
        tracing::error!("chat service stream ended before every choice finished");
        return ApiError::internal().into_response();
    }
    Json(col.into_json()).into_response()
}

async fn stream_response(
    st: &AppState,
    mut events: EventStream,
    meta: Meta,
    n: u32,
    guard: CancelOnDrop,
) -> Response {
    // STR-12: hold headers back briefly so an early failure is a plain HTTP error.
    let deadline = tokio::time::Instant::now() + st.header_wait;
    let mut first: Option<ChoiceEvent> = None;
    loop {
        match tokio::time::timeout_at(deadline, events.next()).await {
            Ok(Some(ev)) => match ev.event {
                TurnEvent::Limit { .. } => continue,
                TurnEvent::Error { error } => return ApiError::from(error).into_response(),
                _ => {
                    first = Some(ev);
                    break;
                }
            },
            Ok(None) => {
                tracing::error!("chat service stream ended without any event");
                return ApiError::internal().into_response();
            }
            Err(_) => break,
        }
    }

    let ping_every = st.ping_every;
    let body = async_stream::stream! {
        let _guard = guard;
        let mut events = events;
        let mut enc = StreamEncoder::new(meta, n);
        match first {
            None => yield Ok::<Bytes, Infallible>(sse_ping()),
            Some(ev) => match enc.push(ev) {
                Step::Chunks(cs) => for c in cs { yield Ok(c) },
                Step::Error(b) => { yield Ok(b); return; }
            },
        }
        loop {
            if enc.all_done() {
                for c in enc.finish() { yield Ok(c) }
                return;
            }
            match tokio::time::timeout(ping_every, events.next()).await {
                Err(_) => yield Ok(sse_ping()),
                Ok(None) => {
                    tracing::error!("chat service stream ended before every choice finished");
                    yield Ok(StreamEncoder::error_event(&ApiError::internal()));
                    return;
                }
                Ok(Some(ev)) => match enc.push(ev) {
                    Step::Chunks(cs) => for c in cs { yield Ok(c) },
                    Step::Error(b) => { yield Ok(b); return; }
                },
            }
        }
    };

    let mut resp = Response::new(Body::from_stream(body));
    let h = resp.headers_mut();
    h.insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream; charset=utf-8"));
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    h.insert("x-accel-buffering", HeaderValue::from_static("no"));
    resp
}
