//! Integration tests: the router bound on 127.0.0.1:0 with a scripted mock ChatService.
#![recursion_limit = "256"]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bridge_types::{
    BridgeError, CancellationToken, ChatRequest, ChatService, ChoiceEvent, Config, ErrorClass,
    FinishReason, ImageSource, ModelInfo, Part, Role, ToolCall, TurnEvent, Usage,
};
use futures::stream::BoxStream;
use futures::StreamExt;
use serde_json::{json, Value};

// ------------------------------------------------------------------ mock

#[derive(Clone)]
struct Step {
    delay_ms: u64,
    ev: ChoiceEvent,
}

fn ev(index: u32, event: TurnEvent) -> Step {
    Step { delay_ms: 0, ev: ChoiceEvent { index, event } }
}

fn delayed(ms: u64, s: Step) -> Step {
    Step { delay_ms: ms, ..s }
}

fn text(i: u32, t: &str) -> Step {
    ev(i, TurnEvent::TextDelta { text: t.into() })
}

fn done(i: u32, f: FinishReason) -> Step {
    ev(i, TurnEvent::Done { finish: f })
}

fn usage(i: u32, p: u64, c: u64) -> Step {
    ev(i, TurnEvent::Usage { usage: Usage { prompt_tokens: p, completion_tokens: c, cached_tokens: 3, reasoning_tokens: 1 } })
}

fn err(class: ErrorClass, msg: &str) -> Step {
    ev(0, TurnEvent::Error { error: BridgeError::new(class, msg) })
}

#[derive(Default)]
struct Mock {
    scripts: Mutex<VecDeque<Vec<Step>>>,
    requests: Mutex<Vec<ChatRequest>>,
    tokens: Mutex<Vec<CancellationToken>>,
}

impl Mock {
    fn script(&self, steps: Vec<Step>) {
        self.scripts.lock().unwrap().push_back(steps);
    }
    fn last_request(&self) -> ChatRequest {
        self.requests.lock().unwrap().last().cloned().expect("no request")
    }
    fn last_token(&self) -> CancellationToken {
        self.tokens.lock().unwrap().last().cloned().expect("no token")
    }
}

#[async_trait]
impl ChatService for Mock {
    fn models(&self) -> Vec<ModelInfo> {
        vec![
            ModelInfo { id: "bridge/auto".into(), created: 1700000000, owned_by: "bridge".into(), context_limit: None },
            ModelInfo { id: "bridge/claude-haiku".into(), created: 1700000000, owned_by: "bridge".into(), context_limit: None },
        ]
    }

    async fn chat(&self, request: ChatRequest, cancel: CancellationToken) -> BoxStream<'static, ChoiceEvent> {
        self.requests.lock().unwrap().push(request);
        self.tokens.lock().unwrap().push(cancel.clone());
        let steps = self
            .scripts
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| vec![text(0, "Hello"), usage(0, 10, 2), done(0, FinishReason::Stop)]);
        let s = async_stream::stream! {
            for st in steps {
                if st.delay_ms > 0 {
                    tokio::select! {
                        _ = tokio::time::sleep(Duration::from_millis(st.delay_ms)) => {}
                        _ = cancel.cancelled() => return,
                    }
                }
                yield st.ev;
            }
        };
        s.boxed()
    }

    fn status(&self) -> Value {
        json!({"ok": true})
    }
}

// ------------------------------------------------------------------ harness

fn test_config() -> Config {
    let mut c = Config::default();
    c.server.header_wait_ms = 300;
    c.server.keepalive_ping_s = 1;
    c.server.max_body_mb = 1;
    c
}

async fn start_with(config: Config) -> (String, Arc<Mock>) {
    let mock = Arc::new(Mock::default());
    let app = bridge_openai::router(mock.clone(), &config);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{addr}"), mock)
}

async fn start() -> (String, Arc<Mock>) {
    start_with(test_config()).await
}

fn client() -> reqwest::Client {
    reqwest::Client::new()
}

fn body(extra: Value) -> Value {
    let mut b = json!({"model": "bridge/auto", "messages": [{"role": "user", "content": "hi"}]});
    if let (Some(o), Value::Object(e)) = (b.as_object_mut(), extra) {
        for (k, v) in e {
            o.insert(k, v);
        }
    }
    b
}

async fn post(base: &str, b: &Value) -> reqwest::Response {
    client().post(format!("{base}/v1/chat/completions")).json(b).send().await.unwrap()
}

async fn post_json(base: &str, b: &Value) -> (u16, Value) {
    let r = post(base, b).await;
    let status = r.status().as_u16();
    (status, r.json().await.unwrap())
}

/// SSE events: `Some(json)` for data events, `None` for `[DONE]`; pings counted separately.
struct Sse {
    events: Vec<Value>,
    done: bool,
    pings: usize,
    raw: String,
}

fn parse_sse(raw: &str) -> Sse {
    let mut events = Vec::new();
    let mut done = false;
    let mut pings = 0;
    for block in raw.split("\n\n").filter(|b| !b.is_empty()) {
        if block == ": ping" {
            pings += 1;
            continue;
        }
        let data = block.strip_prefix("data: ").unwrap_or_else(|| panic!("bad block {block:?}"));
        assert!(!done, "event after [DONE]");
        if data == "[DONE]" {
            done = true;
        } else {
            events.push(serde_json::from_str(data).unwrap());
        }
    }
    assert!(raw.ends_with("\n\n"), "stream must end with a blank line");
    Sse { events, done, pings, raw: raw.to_string() }
}

/// Streaming request; returns (status, parsed SSE). Callers needing headers use `post`.
async fn stream(base: &str, b: &Value) -> (u16, Sse) {
    let r = post(base, b).await;
    let status = r.status().as_u16();
    (status, parse_sse(&r.text().await.unwrap()))
}

fn assert_error_shape(v: &Value) {
    let e = &v["error"];
    assert!(e["message"].is_string(), "{v}");
    assert!(e["type"].is_string(), "{v}");
    assert!(e.get("param").is_some(), "{v}");
    assert!(e.get("code").is_some(), "{v}");
}

// ------------------------------------------------------------------ non-streaming

#[tokio::test]
async fn nonstream_text_shape() {
    let (base, mock) = start().await;
    let r = post(&base, &body(json!({}))).await;
    assert_eq!(r.status(), 200);
    assert!(r.headers()["content-type"].to_str().unwrap().starts_with("application/json"));
    assert!(r.headers().contains_key("x-request-id"));
    let v: Value = r.json().await.unwrap();
    assert!(v["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert!(v["id"].as_str().unwrap().len() >= "chatcmpl-".len() + 24);
    assert_eq!(v["object"], "chat.completion");
    assert!(v["created"].as_u64().unwrap() > 1_600_000_000);
    assert_eq!(v["model"], "bridge/auto");
    assert_eq!(v["system_fingerprint"], "fp_bridge");
    let c = &v["choices"][0];
    assert_eq!(c["index"], 0);
    assert_eq!(c["message"]["role"], "assistant");
    assert_eq!(c["message"]["content"], "Hello");
    assert_eq!(c["message"]["refusal"], Value::Null);
    assert!(c["message"].get("tool_calls").is_none());
    assert_eq!(c["logprobs"], Value::Null);
    assert_eq!(c["finish_reason"], "stop");
    assert_eq!(v["usage"]["prompt_tokens"], 10);
    assert_eq!(v["usage"]["completion_tokens"], 2);
    assert_eq!(v["usage"]["total_tokens"], 12);
    assert_eq!(v["usage"]["prompt_tokens_details"]["cached_tokens"], 3);
    assert_eq!(v["usage"]["completion_tokens_details"]["reasoning_tokens"], 1);
    // Token released after a normal finish.
    assert!(mock.last_token().is_cancelled());
}

#[tokio::test]
async fn nonstream_tool_calls_and_reasoning() {
    let (base, mock) = start().await;
    mock.script(vec![
        ev(0, TurnEvent::ReasoningDelta { text: "thinking".into() }),
        ev(0, TurnEvent::ToolCalls { calls: vec![
            ToolCall { id: "call_1".into(), name: "read.file".into(), arguments: "{\"p\":\"a\\nb\"}".into() },
            ToolCall { id: "call_2".into(), name: "ls".into(), arguments: "{}".into() },
        ] }),
        done(0, FinishReason::ToolCalls),
    ]);
    let (s, v) = post_json(&base, &body(json!({}))).await;
    assert_eq!(s, 200);
    let m = &v["choices"][0]["message"];
    assert_eq!(m["content"], Value::Null);
    assert_eq!(m["reasoning_content"], "thinking");
    assert_eq!(m["tool_calls"][0]["id"], "call_1");
    assert_eq!(m["tool_calls"][0]["type"], "function");
    assert_eq!(m["tool_calls"][0]["function"]["name"], "read.file");
    assert_eq!(m["tool_calls"][0]["function"]["arguments"], "{\"p\":\"a\\nb\"}");
    assert_eq!(m["tool_calls"][1]["id"], "call_2");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    // Usage always present even without Usage events.
    assert_eq!(v["usage"]["total_tokens"], 0);
}

#[tokio::test]
async fn reasoning_hidden_when_disabled() {
    let mut c = test_config();
    c.output.reasoning_to_caller = false;
    let (base, mock) = start_with(c).await;
    mock.script(vec![ev(0, TurnEvent::ReasoningDelta { text: "x".into() }), text(0, "y"), done(0, FinishReason::Stop)]);
    let (_, v) = post_json(&base, &body(json!({}))).await;
    assert!(v["choices"][0]["message"].get("reasoning_content").is_none());
    mock.script(vec![ev(0, TurnEvent::ReasoningDelta { text: "x".into() }), text(0, "y"), done(0, FinishReason::Stop)]);
    let (_, sse) = stream(&base, &body(json!({"stream": true}))).await;
    assert!(!sse.raw.contains("reasoning_content"));
}

#[tokio::test]
async fn nonstream_n2() {
    let (base, mock) = start().await;
    mock.script(vec![
        text(1, "B"), text(0, "A"), usage(0, 10, 2), usage(1, 10, 5),
        done(1, FinishReason::Stop), done(0, FinishReason::Length),
    ]);
    let (s, v) = post_json(&base, &body(json!({"n": 2}))).await;
    assert_eq!(s, 200);
    assert_eq!(mock.last_request().n, 2);
    assert_eq!(v["choices"][0]["index"], 0);
    assert_eq!(v["choices"][0]["message"]["content"], "A");
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert_eq!(v["choices"][1]["message"]["content"], "B");
    assert_eq!(v["usage"]["prompt_tokens"], 10);
    assert_eq!(v["usage"]["completion_tokens"], 7);
    assert_eq!(v["usage"]["total_tokens"], 17);
}

#[tokio::test]
async fn nonstream_error_and_retry_after() {
    let (base, mock) = start().await;
    mock.script(vec![ev(0, TurnEvent::Error {
        error: BridgeError::new(ErrorClass::QuotaExhausted, "You exceeded your current quota.").with_retry_after(42),
    })]);
    let r = post(&base, &body(json!({}))).await;
    assert_eq!(r.status(), 429);
    assert_eq!(r.headers()["retry-after"], "42");
    let v: Value = r.json().await.unwrap();
    assert_error_shape(&v);
    assert_eq!(v["error"]["type"], "insufficient_quota");
    assert_eq!(v["error"]["code"], "insufficient_quota");

    // Error after partial output, non-streaming: still an HTTP error.
    mock.script(vec![text(0, "par"), err(ErrorClass::Unavailable, "Overloaded.")]);
    let (s, v) = post_json(&base, &body(json!({}))).await;
    assert_eq!(s, 503);
    assert_eq!(v["error"]["code"], "overloaded");
    assert_eq!(v["error"]["type"], "server_error");
}

#[tokio::test]
async fn error_class_mapping() {
    let (base, mock) = start().await;
    for (class, status, code) in [
        (ErrorClass::ContextTooLong, 400, Some("context_length_exceeded")),
        (ErrorClass::RateLimited, 429, Some("rate_limit_exceeded")),
        (ErrorClass::Internal, 500, None),
        (ErrorClass::BackendAuth, 503, None),
        (ErrorClass::Timeout, 504, Some("timeout")),
    ] {
        mock.script(vec![err(class, "boom")]);
        let (s, v) = post_json(&base, &body(json!({}))).await;
        assert_eq!(s, status, "{class:?}");
        assert_error_shape(&v);
        assert_eq!(v["error"]["code"].as_str(), code, "{class:?}");
        assert_eq!(v["error"]["message"], "boom");
    }
}

// ------------------------------------------------------------------ streaming

#[tokio::test]
async fn stream_sequence() {
    let (base, mock) = start().await;
    mock.script(vec![text(0, "Hel"), text(0, "lo"), usage(0, 10, 2), done(0, FinishReason::Stop)]);
    let r = post(&base, &body(json!({"stream": true}))).await;
    assert_eq!(r.status(), 200);
    assert!(r.headers()["content-type"].to_str().unwrap().starts_with("text/event-stream"));
    assert_eq!(r.headers()["cache-control"], "no-cache");
    assert!(r.headers().contains_key("x-request-id"));
    let sse = parse_sse(&r.text().await.unwrap());
    assert!(sse.done);
    assert_eq!(sse.pings, 0);
    let ev = &sse.events;
    assert_eq!(ev.len(), 4);
    let id = ev[0]["id"].clone();
    for e in ev {
        assert_eq!(e["id"], id);
        assert_eq!(e["object"], "chat.completion.chunk");
        assert_eq!(e["model"], "bridge/auto");
        assert_eq!(e["created"], ev[0]["created"]);
        assert_eq!(e["system_fingerprint"], "fp_bridge");
        assert!(e.get("usage").is_none());
    }
    assert_eq!(ev[0]["choices"][0]["delta"], json!({"role": "assistant", "content": ""}));
    assert_eq!(ev[0]["choices"][0]["finish_reason"], Value::Null);
    assert_eq!(ev[1]["choices"][0]["delta"]["content"], "Hel");
    assert_eq!(ev[2]["choices"][0]["delta"]["content"], "lo");
    assert_eq!(ev[3]["choices"][0]["delta"], json!({}));
    assert_eq!(ev[3]["choices"][0]["finish_reason"], "stop");
    assert!(mock.last_token().is_cancelled());
}

#[tokio::test]
async fn stream_include_usage() {
    let (base, mock) = start().await;
    mock.script(vec![text(0, "x"), usage(0, 10, 2), done(0, FinishReason::Stop)]);
    let (_, sse) = stream(&base, &body(json!({"stream": true, "stream_options": {"include_usage": true}}))).await;
    assert!(sse.done);
    let ev = &sse.events;
    let last = ev.last().unwrap();
    assert_eq!(last["choices"], json!([]));
    assert_eq!(last["usage"]["prompt_tokens"], 10);
    assert_eq!(last["usage"]["completion_tokens"], 2);
    assert_eq!(last["usage"]["total_tokens"], 12);
    for e in &ev[..ev.len() - 1] {
        assert_eq!(e["usage"], Value::Null);
        assert!(e.get("usage").is_some());
    }
    assert_eq!(ev[ev.len() - 2]["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn stream_tool_call_chunks() {
    let (base, mock) = start().await;
    mock.script(vec![
        text(0, "Let me look."),
        ev(0, TurnEvent::ToolCalls { calls: vec![
            ToolCall { id: "call_a".into(), name: "read".into(), arguments: "{\"x\":1}".into() },
            ToolCall { id: "call_b".into(), name: "glob".into(), arguments: "{}".into() },
        ] }),
        done(0, FinishReason::ToolCalls),
    ]);
    let (_, sse) = stream(&base, &body(json!({"stream": true}))).await;
    let ev = &sse.events;
    assert_eq!(ev.len(), 5);
    let tc0 = &ev[2]["choices"][0]["delta"]["tool_calls"][0];
    assert_eq!(tc0["index"], 0);
    assert_eq!(tc0["id"], "call_a");
    assert_eq!(tc0["type"], "function");
    assert_eq!(tc0["function"]["name"], "read");
    assert_eq!(tc0["function"]["arguments"], "{\"x\":1}");
    let tc1 = &ev[3]["choices"][0]["delta"]["tool_calls"][0];
    assert_eq!(tc1["index"], 1);
    assert_eq!(tc1["id"], "call_b");
    assert_eq!(ev[4]["choices"][0]["finish_reason"], "tool_calls");
}

#[tokio::test]
async fn stream_n2_interleaved() {
    let (base, mock) = start().await;
    mock.script(vec![
        text(0, "a1"), text(1, "b1"), text(0, "a2"), done(1, FinishReason::Stop), text(0, "a3"),
        done(0, FinishReason::Stop),
    ]);
    let (_, sse) = stream(&base, &body(json!({"stream": true, "n": 2}))).await;
    assert!(sse.done);
    let seq: Vec<(u64, Value, Value)> = sse
        .events
        .iter()
        .map(|e| {
            let c = &e["choices"][0];
            (c["index"].as_u64().unwrap(), c["delta"].clone(), c["finish_reason"].clone())
        })
        .collect();
    let expected = vec![
        (0, json!({"role":"assistant","content":""}), Value::Null),
        (0, json!({"content":"a1"}), Value::Null),
        (1, json!({"role":"assistant","content":""}), Value::Null),
        (1, json!({"content":"b1"}), Value::Null),
        (0, json!({"content":"a2"}), Value::Null),
        (1, json!({}), json!("stop")),
        (0, json!({"content":"a3"}), Value::Null),
        (0, json!({}), json!("stop")),
    ];
    assert_eq!(seq, expected);
}

#[tokio::test]
async fn stream_early_error_is_http_error() {
    let (base, mock) = start().await;
    mock.script(vec![
        ev(0, TurnEvent::Limit { info: bridge_types::LimitInfo { status: bridge_types::LimitStatus::Exhausted, utilization: None, resets_at: None } }),
        delayed(50, ev(0, TurnEvent::Error { error: BridgeError::new(ErrorClass::RateLimited, "Too many requests.").with_retry_after(7) })),
    ]);
    let r = post(&base, &body(json!({"stream": true}))).await;
    assert_eq!(r.status(), 429);
    assert_eq!(r.headers()["retry-after"], "7");
    assert!(r.headers()["content-type"].to_str().unwrap().starts_with("application/json"));
    let v: Value = r.json().await.unwrap();
    assert_error_shape(&v);
    assert_eq!(v["error"]["code"], "rate_limit_exceeded");
}

#[tokio::test]
async fn stream_late_error_is_sse_event() {
    let (base, mock) = start().await;
    mock.script(vec![text(0, "partial"), err(ErrorClass::Unavailable, "The server is overloaded.")]);
    let r = post(&base, &body(json!({"stream": true}))).await;
    assert_eq!(r.status(), 200);
    let sse = parse_sse(&r.text().await.unwrap());
    assert!(!sse.done, "no [DONE] after an error");
    let last = sse.events.last().unwrap();
    assert_error_shape(last);
    assert_eq!(last["error"]["code"], "overloaded");
    assert_eq!(sse.events[1]["choices"][0]["delta"]["content"], "partial");
    assert!(mock.last_token().is_cancelled());
}

#[tokio::test]
async fn stream_error_after_header_wait() {
    // Header wait (300 ms) expires, then the error arrives: it must be an SSE error event.
    let (base, mock) = start().await;
    mock.script(vec![delayed(600, err(ErrorClass::Timeout, "Request timed out."))]);
    let r = post(&base, &body(json!({"stream": true}))).await;
    assert_eq!(r.status(), 200);
    let sse = parse_sse(&r.text().await.unwrap());
    assert!(sse.pings >= 1);
    assert!(!sse.done);
    assert_eq!(sse.events.len(), 1);
    assert_eq!(sse.events[0]["error"]["code"], "timeout");
}

#[tokio::test]
async fn stream_keepalive_pings_on_slow_start() {
    let (base, mock) = start().await;
    mock.script(vec![delayed(1700, text(0, "late")), done(0, FinishReason::Stop)]);
    let started = std::time::Instant::now();
    let r = post(&base, &body(json!({"stream": true}))).await;
    // Headers come after header_wait_ms, well before the first event.
    assert!(started.elapsed() < Duration::from_millis(1200), "{:?}", started.elapsed());
    assert_eq!(r.status(), 200);
    let raw = r.text().await.unwrap();
    assert!(raw.starts_with(": ping\n\n"), "{raw}");
    let sse = parse_sse(&raw);
    assert!(sse.pings >= 2, "pings: {}", sse.pings);
    assert!(sse.done);
    assert_eq!(sse.events[1]["choices"][0]["delta"]["content"], "late");
}

#[tokio::test]
async fn stream_client_disconnect_cancels() {
    let (base, mock) = start().await;
    let mut steps = vec![text(0, "start")];
    for _ in 0..200 {
        steps.push(delayed(50, text(0, "x")));
    }
    steps.push(done(0, FinishReason::Stop));
    mock.script(steps);
    let r = post(&base, &body(json!({"stream": true}))).await;
    let mut s = r.bytes_stream();
    let first = s.next().await.unwrap().unwrap();
    assert!(!first.is_empty());
    let token = mock.last_token();
    assert!(!token.is_cancelled());
    drop(s);
    tokio::time::timeout(Duration::from_secs(3), token.cancelled()).await.expect("token not cancelled");
}

#[tokio::test]
async fn nonstream_client_disconnect_cancels() {
    let (base, mock) = start().await;
    mock.script(vec![delayed(10_000, text(0, "late")), done(0, FinishReason::Stop)]);
    let b = body(json!({}));
    let req = tokio::spawn(async move {
        let _ = client().post(format!("{base}/v1/chat/completions")).json(&b).send().await;
    });
    // Wait until the service saw the request.
    for _ in 0..100 {
        if !mock.tokens.lock().unwrap().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let token = mock.last_token();
    req.abort();
    tokio::time::timeout(Duration::from_secs(3), token.cancelled()).await.expect("token not cancelled");
}

// ------------------------------------------------------------------ http surface

#[tokio::test]
async fn auth() {
    let mut c = test_config();
    c.server.token = Some("secret".into());
    let (base, _mock) = start_with(c).await;
    let url = format!("{base}/v1/chat/completions");
    let b = body(json!({}));

    let r = client().post(&url).json(&b).send().await.unwrap();
    assert_eq!(r.status(), 401);
    assert!(r.headers().contains_key("x-request-id"));
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert_eq!(v["error"]["code"], "invalid_api_key");

    let r = client().post(&url).bearer_auth("wrong").json(&b).send().await.unwrap();
    assert_eq!(r.status(), 401);

    let r = client()
        .post(&url)
        .bearer_auth("secret")
        .header("OpenAI-Organization", "org-x")
        .header("OpenAI-Project", "proj-y")
        .json(&b)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);

    let r = client().post(&url).header("api-key", "secret").json(&b).send().await.unwrap();
    assert_eq!(r.status(), 200);

    let r = client().get(format!("{base}/v1/models")).send().await.unwrap();
    assert_eq!(r.status(), 401);
}

#[tokio::test]
async fn not_found_and_method_not_allowed() {
    let (base, _mock) = start().await;
    let r = client().post(format!("{base}/v1/embeddings")).json(&json!({})).send().await.unwrap();
    assert_eq!(r.status(), 404);
    assert!(r.headers().contains_key("x-request-id"));
    assert_error_shape(&r.json().await.unwrap());

    let r = client().post(format!("{base}/v1/completions")).json(&json!({})).send().await.unwrap();
    assert_eq!(r.status(), 404);

    let r = client().get(format!("{base}/v1/chat/completions")).send().await.unwrap();
    assert_eq!(r.status(), 405);
    assert_error_shape(&r.json().await.unwrap());
}

#[tokio::test]
async fn payload_too_large() {
    let (base, _mock) = start().await;
    let big = "x".repeat(2 * 1024 * 1024);
    let r = post(&base, &body(json!({"messages": [{"role": "user", "content": big}]}))).await;
    assert_eq!(r.status(), 413);
    assert!(r.headers().contains_key("x-request-id"));
    let v: Value = r.json().await.unwrap();
    assert_error_shape(&v);
    assert_eq!(v["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn malformed_json() {
    let (base, _mock) = start().await;
    for raw in ["{not json", "[1,2]", ""] {
        let r = client()
            .post(format!("{base}/v1/chat/completions"))
            .header("content-type", "application/json")
            .body(raw)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 400, "{raw}");
        assert_error_shape(&r.json().await.unwrap());
    }
    // Missing content-type is tolerated.
    let r = client().post(format!("{base}/v1/chat/completions")).body(body(json!({})).to_string()).send().await.unwrap();
    assert_eq!(r.status(), 200);
}

#[tokio::test]
async fn models_endpoints_and_paths() {
    let (base, _mock) = start().await;
    let v: Value = client().get(format!("{base}/v1/models")).send().await.unwrap().json().await.unwrap();
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"][0], json!({"id":"bridge/auto","object":"model","created":1700000000,"owned_by":"bridge"}));

    let r = client().get(format!("{base}/v1/models/bridge/auto")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["id"], "bridge/auto");
    assert_eq!(v["object"], "model");

    let r = client().get(format!("{base}/v1/models/nope")).send().await.unwrap();
    assert_eq!(r.status(), 404);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["error"]["code"], "model_not_found");

    for path in ["/v1/models/", "/models"] {
        let r = client().get(format!("{base}{path}")).send().await.unwrap();
        assert_eq!(r.status(), 200, "{path}");
    }
    for path in ["/v1/chat/completions/", "/chat/completions", "/chat/completions/"] {
        let r = client().post(format!("{base}{path}")).json(&body(json!({}))).send().await.unwrap();
        assert_eq!(r.status(), 200, "{path}");
    }

    let v: Value = client().get(format!("{base}/admin/status")).send().await.unwrap().json().await.unwrap();
    assert_eq!(v, json!({"ok": true}));
}

// ------------------------------------------------------------------ validation

async fn expect_400(base: &str, b: Value, param: &str) -> Value {
    let (s, v) = post_json(base, &b).await;
    assert_eq!(s, 400, "{v}");
    assert_error_shape(&v);
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert_eq!(v["error"]["param"], param, "{v}");
    v
}

#[tokio::test]
async fn validation_ranges_and_types() {
    let (base, _mock) = start().await;
    expect_400(&base, body(json!({"temperature": 3})), "temperature").await;
    expect_400(&base, body(json!({"top_p": 1.5})), "top_p").await;
    expect_400(&base, body(json!({"presence_penalty": -2.5})), "presence_penalty").await;
    expect_400(&base, body(json!({"frequency_penalty": 9})), "frequency_penalty").await;
    expect_400(&base, body(json!({"n": 0})), "n").await;
    expect_400(&base, body(json!({"n": 5})), "n").await;
    expect_400(&base, body(json!({"max_tokens": 0})), "max_tokens").await;
    expect_400(&base, body(json!({"max_completion_tokens": -1})), "max_completion_tokens").await;
    expect_400(&base, body(json!({"stop": ["a","b","c","d","e"]})), "stop").await;
    expect_400(&base, body(json!({"temperature": "warm"})), "temperature").await;
    expect_400(&base, body(json!({"stream": "yes"})), "stream").await;
    expect_400(&base, body(json!({"messages": []})), "messages").await;
    expect_400(&base, json!({"model": "bridge/auto"}), "messages").await;
    expect_400(&base, body(json!({"messages": [{"role": "robot", "content": "x"}]})), "messages[0].role").await;
    expect_400(&base, body(json!({"messages": [{"role": "user", "content": 5}]})), "messages[0].content").await;
    expect_400(&base, body(json!({"modalities": ["text", "audio"]})), "modalities").await;
    expect_400(&base, body(json!({"audio": {"voice": "alloy", "format": "mp3"}})), "audio").await;
    expect_400(&base, body(json!({"messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}]}]})), "messages[0].content[0].type").await;
    expect_400(&base, body(json!({"reasoning_effort": "extreme"})), "reasoning_effort").await;
    expect_400(&base, body(json!({"tool_choice": "required"})), "tool_choice").await;
    expect_400(&base, body(json!({"response_format": {"type": "yaml"}})), "response_format.type").await;
}

#[tokio::test]
async fn unknown_model_404() {
    let (base, _mock) = start().await;
    let (s, v) = post_json(&base, &body(json!({"model": "gpt-9"}))).await;
    assert_eq!(s, 404);
    assert_eq!(v["error"]["code"], "model_not_found");
    assert_eq!(v["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn conv11_tool_without_calls() {
    let (base, _mock) = start().await;
    let v = expect_400(&base, body(json!({"messages": [
        {"role": "user", "content": "hi"},
        {"role": "tool", "tool_call_id": "call_x", "content": "r"}
    ]})), "messages.[1].role").await;
    assert_eq!(
        v["error"]["message"],
        "Invalid parameter: messages with role 'tool' must be a response to a preceeding message with 'tool_calls'."
    );
    // Wrong id also fails.
    expect_400(&base, body(json!({"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": null, "tool_calls": [{"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "b", "content": "r"}
    ]})), "messages.[2].role").await;
}

#[tokio::test]
async fn conv12_unanswered_calls() {
    let (base, _mock) = start().await;
    let v = expect_400(&base, body(json!({"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": null, "tool_calls": [
            {"id": "call_a", "type": "function", "function": {"name": "f", "arguments": "{}"}},
            {"id": "call_b", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_a", "content": "r"},
        {"role": "user", "content": "next"}
    ]})), "messages.[1].role").await;
    assert_eq!(
        v["error"]["message"],
        "An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'. The following tool_call_ids did not have response messages: call_b"
    );
}

#[tokio::test]
async fn accepted_shapes_to_ir() {
    let (base, mock) = start().await;
    let b = json!({
        "model": "bridge/claude-haiku",
        "some_vendor_field": {"x": 1},
        "temperature": 0.2, "top_p": 1, "seed": 7, "logprobs": false, "top_logprobs": 0,
        "logit_bias": {"50256": -100}, "user": "u", "metadata": {"a": "b"}, "store": false,
        "service_tier": "auto", "prediction": {"type": "content", "content": "x"},
        "web_search_options": {}, "prompt_cache_key": "k", "safety_identifier": "s",
        "max_tokens": 50, "max_completion_tokens": 100,
        "stop": "END",
        "parallel_tool_calls": false,
        "reasoning_effort": "low",
        "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}, "strict": true}},
        "tools": [{"type": "function", "function": {"name": "todo.write", "description": "d", "parameters": {"type": "object"}}},
                  {"type": "function", "function": {"name": "noargs"}}],
        "tool_choice": {"type": "function", "function": {"name": "todo.write"}},
        "messages": [
            {"role": "developer", "content": "dev rules", "name": "x"},
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": [
                {"type": "text", "text": "look "},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo=", "detail": "high"}},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}},
                {"type": "text", "text": "here"}
            ]},
            {"role": "assistant", "content": "", "refusal": null, "tool_calls": [
                {"id": "call:odd/id", "type": "function", "function": {"name": "todo.write", "arguments": "{\"a\":1}"}},
                {"id": "c2", "type": "function", "function": {"name": "noargs", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c2", "content": [{"type": "text", "text": "part1 "}, {"type": "text", "text": "part2"}]},
            {"role": "tool", "tool_call_id": "call:odd/id", "content": "done"},
            {"role": "user", "content": "ok"},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "prefill"}
        ]
    });
    let (s, v) = post_json(&base, &b).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["model"], "bridge/claude-haiku");
    let r = mock.last_request();
    assert_eq!(r.model, "bridge/claude-haiku");
    assert_eq!(r.max_tokens, Some(100));
    assert_eq!(r.stop, vec!["END".to_string()]);
    assert!(!r.parallel_tool_calls);
    assert_eq!(r.reasoning_effort, Some(bridge_types::ReasoningEffort::Low));
    assert!(matches!(r.response_format, bridge_types::ResponseFormat::JsonSchema { strict: true, .. }));
    assert_eq!(r.tool_choice, bridge_types::ToolChoice::Function { name: "todo.write".into() });
    assert_eq!(r.tools[1].parameters, json!({"type": "object", "properties": {}}));
    assert_eq!(r.messages[0], bridge_types::Message::text(Role::System, "dev rules"));
    assert_eq!(r.messages[1].role, Role::System);
    let user = &r.messages[2];
    assert_eq!(user.parts.len(), 4);
    match &user.parts[1] {
        Part::Image(img) => {
            assert_eq!(img.media_type, "image/png");
            assert_eq!(img.source, ImageSource::Base64 { data: "iVBORw0KGgo=".into() });
        }
        p => panic!("{p:?}"),
    }
    match &user.parts[2] {
        Part::Image(img) => assert_eq!(img.source, ImageSource::Url { url: "https://example.com/a.jpg".into() }),
        p => panic!("{p:?}"),
    }
    assert_eq!(user.text_content(), "look here");
    assert_eq!(r.messages[3].tool_calls().len(), 2);
    assert_eq!(r.messages[3].tool_calls()[0].id, "call:odd/id");
    assert_eq!(r.messages[4].tool_results()[0].content, "part1 part2");
    assert_eq!(r.messages[4].role, Role::Tool);
    assert_eq!(r.messages[8].text_content(), "prefill");
    assert_eq!(r.tool_choice, bridge_types::ToolChoice::Function { name: "todo.write".into() });
}

#[tokio::test]
async fn bad_image_url() {
    let (base, _mock) = start().await;
    expect_400(&base, body(json!({"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:text/plain;base64,aGVsbG8="}}]}]})),
        "messages[0].content[0].image_url.url").await;
    expect_400(&base, body(json!({"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "ftp://x/y.png"}}]}]})),
        "messages[0].content[0].image_url.url").await;
}

#[tokio::test]
async fn defaults_and_stream_flags() {
    let (base, mock) = start().await;
    post_json(&base, &body(json!({"tools": [{"type": "function", "function": {"name": "f"}}]}))).await;
    let r = mock.last_request();
    assert_eq!(r.tool_choice, bridge_types::ToolChoice::Auto);
    assert!(r.parallel_tool_calls);
    assert!(!r.stream);
    assert!(!r.include_usage);
    let _ = stream(&base, &body(json!({"stream": true, "stream_options": {"include_usage": true}}))).await;
    let r = mock.last_request();
    assert!(r.stream && r.include_usage);
}

// ------------------------------------------------------------------ streamed tool calls (17.3)

fn call_start(i: u32, index: u32, id: &str, name: &str) -> Step {
    ev(i, TurnEvent::ToolCallStart { index, id: id.into(), name: name.into() })
}

fn call_delta(i: u32, index: u32, a: &str) -> Step {
    ev(i, TurnEvent::ToolCallDelta { index, arguments: a.into() })
}

const BIG_ARGS_FRAGMENTS: [&str; 5] = ["{\"path\": \"a.txt\",", " \"content\": \"line 1\\n", "line \\\"2\\\" é", "\\n\"", "}"];

fn streamed_script() -> Vec<Step> {
    let args: String = BIG_ARGS_FRAGMENTS.concat();
    let mut s = vec![text(0, "Writing."), call_start(0, 0, "call_w", "write")];
    s.extend(BIG_ARGS_FRAGMENTS.iter().map(|f| call_delta(0, 0, f)));
    s.push(call_start(0, 1, "call_r", "read"));
    s.push(call_delta(0, 1, "{\"path\":\"b\"}"));
    s.push(ev(0, TurnEvent::ToolCalls { calls: vec![
        ToolCall { id: "call_w".into(), name: "write".into(), arguments: args },
        ToolCall { id: "call_r".into(), name: "read".into(), arguments: "{\"path\":\"b\"}".into() },
        ToolCall { id: "call_g".into(), name: "glob".into(), arguments: "{}".into() },
    ] }));
    s.push(usage(0, 10, 50));
    s.push(done(0, FinishReason::ToolCalls));
    s
}

#[tokio::test]
async fn stream_streamed_tool_call_chunks() {
    let (base, mock) = start().await;
    mock.script(streamed_script());
    let (_, sse) = stream(&base, &body(json!({"stream": true}))).await;
    assert!(sse.done);
    let deltas: Vec<&Value> = sse.events.iter().map(|e| &e["choices"][0]["delta"]).collect();
    // role, text, start, 5 deltas, start, delta, committed-only call, finish
    assert_eq!(deltas.len(), 12, "{}", sse.raw);
    assert_eq!(deltas[2]["tool_calls"][0], json!({"index": 0, "id": "call_w", "type": "function", "function": {"name": "write", "arguments": ""}}));
    for (k, f) in BIG_ARGS_FRAGMENTS.iter().enumerate() {
        assert_eq!(deltas[3 + k]["tool_calls"][0], json!({"index": 0, "function": {"arguments": f}}));
    }
    assert_eq!(deltas[8]["tool_calls"][0]["index"], 1);
    assert_eq!(deltas[8]["tool_calls"][0]["id"], "call_r");
    assert_eq!(deltas[10]["tool_calls"][0], json!({"index": 2, "id": "call_g", "type": "function", "function": {"name": "glob", "arguments": "{}"}}));
    assert_eq!(sse.events[11]["choices"][0]["finish_reason"], "tool_calls");
}

#[tokio::test]
async fn stream_error_after_partial_streamed_tool_call() {
    let (base, mock) = start().await;
    mock.script(vec![
        call_start(0, 0, "call_w", "write"),
        call_delta(0, 0, "{\"content\": \"par"),
        delayed(20, err(ErrorClass::Unavailable, "The server is overloaded.")),
    ]);
    let r = post(&base, &body(json!({"stream": true}))).await;
    assert_eq!(r.status(), 200);
    let sse = parse_sse(&r.text().await.unwrap());
    assert!(!sse.done);
    assert_eq!(sse.events.len(), 4);
    assert_eq!(sse.events[1]["choices"][0]["delta"]["tool_calls"][0]["id"], "call_w");
    let last = sse.events.last().unwrap();
    assert_error_shape(last);
    assert_eq!(last["error"]["code"], "overloaded");
}

#[tokio::test]
async fn nonstream_ignores_tool_call_previews() {
    let (base, mock) = start().await;
    mock.script(streamed_script());
    let (status, v) = post_json(&base, &body(json!({}))).await;
    assert_eq!(status, 200);
    let tcs = v["choices"][0]["message"]["tool_calls"].as_array().unwrap();
    assert_eq!(tcs.len(), 3);
    assert_eq!(tcs[0]["function"]["arguments"], BIG_ARGS_FRAGMENTS.concat());
    assert_eq!(v["choices"][0]["message"]["content"], "Writing.");
}

/// Optional: the official Python SDK's stream accumulator (`chat.completions.stream`)
/// must rebuild exactly the committed calls. Skipped when `target/sdk/py` is absent.
#[tokio::test]
async fn sdk_python_accumulates_streamed_tool_calls() {
    let py = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/sdk/py/bin/python");
    if !py.exists() {
        eprintln!("skipped: {} not found", py.display());
        return;
    }
    let (base, mock) = start().await;
    mock.script(streamed_script());
    let script = r#"
import json, sys, openai
c = openai.OpenAI(base_url=sys.argv[1] + "/v1", api_key="x")
with c.chat.completions.stream(model="bridge/auto", messages=[{"role": "user", "content": "hi"}]) as s:
    for _ in s:
        pass
    m = s.get_final_completion().choices[0].message
print(json.dumps([[t.index if hasattr(t, "index") else i, t.id, t.function.name, t.function.arguments] for i, t in enumerate(m.tool_calls)]))
"#;
    let out = tokio::process::Command::new(&py).arg("-I").arg("-c").arg(script).arg(&base).output().await.unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let got: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(got, json!([
        [0, "call_w", "write", BIG_ARGS_FRAGMENTS.concat()],
        [1, "call_r", "read", "{\"path\":\"b\"}"],
        [2, "call_g", "glob", "{}"],
    ]));
}

/// Optional: same check with the official Node SDK (`target/sdk/node`).
#[tokio::test]
async fn sdk_node_accumulates_streamed_tool_calls() {
    let module = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/sdk/node/node_modules/openai");
    let node_ok = std::process::Command::new("node").arg("--version").output().map(|o| o.status.success()).unwrap_or(false);
    if !module.exists() || !node_ok {
        eprintln!("skipped: node or {} not found", module.display());
        return;
    }
    let (base, mock) = start().await;
    mock.script(streamed_script());
    let script = r#"
const OpenAI = require(process.argv[2]).default ?? require(process.argv[2]);
(async () => {
  const c = new OpenAI({ baseURL: process.argv[1] + "/v1", apiKey: "x" });
  const s = c.chat.completions.stream({ model: "bridge/auto", messages: [{ role: "user", content: "hi" }] });
  const done = await s.finalChatCompletion();
  const m = done.choices[0].message;
  console.log(JSON.stringify(m.tool_calls.map((t, i) => [i, t.id, t.function.name, t.function.arguments])));
})().catch((e) => { console.error(e); process.exit(1); });
"#;
    let out = tokio::process::Command::new("node")
        .arg("-e")
        .arg(script)
        .arg(&base)
        .arg(module.canonicalize().unwrap())
        .output()
        .await
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let got: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(got, json!([
        [0, "call_w", "write", BIG_ARGS_FRAGMENTS.concat()],
        [1, "call_r", "read", "{\"path\":\"b\"}"],
        [2, "call_g", "glob", "{}"],
    ]));
}
