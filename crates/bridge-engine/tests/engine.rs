//! Engine-level tests with the scripted FakeBackend.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use bridge_engine::fake::FakeBackend;
use bridge_engine::Engine;
use bridge_types::{
    Backend, BackendConfig, CancellationToken, ChatRequest, ChatService, CheckpointKind, ChoiceEvent, Config,
    ErrorClass, FinishReason, Message, ModelRoute, Part, Role, ToolCall, ToolChoice, ToolResult, ToolSpec, TurnEvent,
};
use futures::StreamExt;
use serde_json::json;

struct H {
    engine: Arc<Engine>,
    fakes: Vec<Arc<FakeBackend>>,
    dir: tempfile::TempDir,
    config: Config,
}

impl H {
    fn fake(&self, i: usize) -> &FakeBackend {
        &self.fakes[i]
    }
}

struct B {
    name: &'static str,
    priority: u32,
    options: serde_json::Value,
    max_concurrency: u32,
    queue_wait_ms: u64,
}

fn b(name: &'static str, priority: u32, options: serde_json::Value) -> B {
    B { name, priority, options, max_concurrency: 4, queue_wait_ms: 2_000 }
}

fn config_for(backends: &[B]) -> Config {
    let mut config = Config::default();
    config.models.clear();
    config.backends.clear();
    config.models.insert("m/auto".into(), ModelRoute::default());
    for be in backends {
        config.models.insert(
            format!("m/{}", be.name),
            ModelRoute { backend: Some(be.name.into()), model: Some("big".into()), context_limit: None },
        );
        config.backends.insert(
            be.name.into(),
            BackendConfig {
                kind: "fake".into(),
                priority: be.priority,
                max_concurrency: be.max_concurrency,
                queue_wait_ms: be.queue_wait_ms,
                default_model: "default".into(),
                side_request_model: Some("small".into()),
                options: be.options.clone(),
                ..Default::default()
            },
        );
    }
    config.server.first_token_timeout_s = 5;
    config
}

async fn build(config: &Config, dir: &Path) -> (Arc<Engine>, Vec<Arc<FakeBackend>>) {
    let mut fakes = Vec::new();
    let mut dyns: Vec<Arc<dyn Backend>> = Vec::new();
    for (name, cfg) in &config.backends {
        let f = Arc::new(FakeBackend::new(name, cfg, &dir.join("snapshots").join(name)).unwrap());
        dyns.push(f.clone());
        fakes.push(f);
    }
    let engine = Engine::new(config, dir, dyns).await.unwrap();
    (engine, fakes)
}

async fn setup(backends: Vec<B>) -> H {
    setup_with(backends, |_| {}).await
}

async fn setup_with(backends: Vec<B>, tweak: impl FnOnce(&mut Config)) -> H {
    let dir = tempfile::tempdir().unwrap();
    let mut config = config_for(&backends);
    tweak(&mut config);
    // Fakes are created in config.backends (BTreeMap) order; tests name backends so
    // that order matches their declaration ("a", "b", ...).
    let (engine, fakes) = build(&config, dir.path()).await;
    H { engine, fakes, dir, config }
}

fn sys(t: &str) -> Message {
    Message::text(Role::System, t)
}
fn user(t: &str) -> Message {
    Message::text(Role::User, t)
}
fn asst(t: &str) -> Message {
    Message::text(Role::Assistant, t)
}
fn tool_result(id: &str, content: &str) -> Message {
    Message { role: Role::Tool, parts: vec![Part::ToolResult(ToolResult { call_id: id.into(), content: content.into() })] }
}
fn read_tool() -> ToolSpec {
    ToolSpec {
        name: "read".into(),
        description: "read a file".into(),
        parameters: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
        strict: false,
    }
}

/// A request that is not a side request (it offers a tool).
fn req(model: &str, messages: Vec<Message>) -> ChatRequest {
    let mut r = ChatRequest::new(model, messages);
    r.tools = vec![read_tool()];
    r
}

async fn run(engine: &Engine, r: ChatRequest) -> Vec<ChoiceEvent> {
    let stream = engine.chat(r, CancellationToken::new()).await;
    tokio::time::timeout(Duration::from_secs(20), stream.collect::<Vec<_>>()).await.expect("stream finished")
}

fn text(events: &[ChoiceEvent], index: u32) -> String {
    events
        .iter()
        .filter(|e| e.index == index)
        .filter_map(|e| if let TurnEvent::TextDelta { text } = &e.event { Some(text.as_str()) } else { None })
        .collect()
}

fn calls(events: &[ChoiceEvent], index: u32) -> Vec<ToolCall> {
    events
        .iter()
        .filter(|e| e.index == index)
        .filter_map(|e| if let TurnEvent::ToolCalls { calls } = &e.event { Some(calls.clone()) } else { None })
        .flatten()
        .collect()
}

fn finish(events: &[ChoiceEvent], index: u32) -> Option<FinishReason> {
    events.iter().filter(|e| e.index == index).find_map(|e| match e.event {
        TurnEvent::Done { finish } => Some(finish),
        _ => None,
    })
}

fn first_error(events: &[ChoiceEvent]) -> Option<bridge_types::BridgeError> {
    match events.first().map(|e| &e.event) {
        Some(TurnEvent::Error { error }) => Some(error.clone()),
        _ => None,
    }
}

/// The assistant message the caller would resend for this answer.
fn as_message(events: &[ChoiceEvent], index: u32) -> Message {
    let mut parts = Vec::new();
    let t = text(events, index);
    if !t.is_empty() {
        parts.push(Part::Text { text: t });
    }
    parts.extend(calls(events, index).into_iter().map(Part::ToolCall));
    Message { role: Role::Assistant, parts }
}

fn assert_safe(err: &bridge_types::BridgeError) {
    let m = err.message.to_lowercase();
    assert!(!m.contains("fake") && !m.contains("/secret") && !m.contains("backend"), "leaky error: {}", err.message);
}

// ---------------------------------------------------------------- conversations

#[tokio::test]
async fn new_conversation_creates_idle_checkpoint() {
    let h = setup(vec![b("a", 1, json!({}))]).await;
    let ev = run(&h.engine, req("m/auto", vec![sys("be nice"), user("hello")])).await;
    assert_eq!(text(&ev, 0), "echo: hello");
    assert_eq!(finish(&ev, 0), Some(FinishReason::Stop));
    assert!(ev.iter().any(|e| matches!(e.event, TurnEvent::Usage { .. })));
    assert_eq!(h.engine.checkpoint_stats().0, 1);
    let rec = &h.fake(0).records()[0];
    assert!(rec.base_path.is_none());
    assert_eq!(rec.trigger, "user");
    assert_eq!(rec.appended, 0);
    assert_eq!(rec.system, "be nice");
    assert_eq!(rec.model, "default");
    assert!(rec.frozen);
    let log = h.engine.recent_requests(1);
    assert_eq!(log[0].backend.as_deref(), Some("a"));
    assert_eq!(log[0].finish_reason.as_deref(), Some("stop"));
    assert_eq!(log[0].checkpoint_hit_depth, Some(0));
}

#[tokio::test]
async fn follow_up_uses_idle_checkpoint() {
    let h = setup(vec![b("a", 1, json!({"reply": "history"}))]).await;
    let m1 = vec![sys("s"), user("u1")];
    let ev = run(&h.engine, req("m/auto", m1.clone())).await;
    assert_eq!(text(&ev, 0), "history: 1 messages; users: u1; last user: u1");
    let mut m2 = m1.clone();
    m2.push(as_message(&ev, 0));
    m2.push(user("u2"));
    let ev2 = run(&h.engine, req("m/auto", m2)).await;
    assert_eq!(text(&ev2, 0), "history: 3 messages; users: u1 | u2; last user: u2");
    let recs = h.fake(0).records();
    assert!(recs[1].base_path.is_some());
    assert_eq!(recs[1].base_kind, Some(CheckpointKind::Idle));
    assert_eq!(recs[1].appended, 0);
    assert_eq!(recs[1].base_messages, 2);
    assert_eq!(h.engine.recent_requests(1)[0].checkpoint_hit_depth, Some(3));
}

#[tokio::test]
async fn trailing_whitespace_in_resent_history_still_hits() {
    let h = setup(vec![b("a", 1, json!({}))]).await;
    let m1 = vec![sys("s"), user("u1")];
    let ev = run(&h.engine, req("m/auto", m1.clone())).await;
    let mut m2 = m1.clone();
    m2.push(asst(&format!("{}\r\n  ", text(&ev, 0))));
    m2.push(user("u2"));
    run(&h.engine, req("m/auto", m2)).await;
    assert!(h.fake(0).records()[1].base_path.is_some());
}

#[tokio::test]
async fn tool_round_hits_awaiting_checkpoint() {
    let h = setup(vec![b(
        "a",
        1,
        json!({"rules": [
            {"match": "use the tool", "text": "Reading.", "tool_calls": [{"name": "read", "arguments": {"path": "x"}}]},
            {"match": "tool_results", "text": "the file says hi"}
        ]}),
    )])
    .await;
    let m1 = vec![sys("s"), user("please use the tool")];
    let ev = run(&h.engine, req("m/auto", m1.clone())).await;
    let c = calls(&ev, 0);
    assert_eq!(c.len(), 1);
    assert_eq!(c[0].name, "read");
    assert_eq!(finish(&ev, 0), Some(FinishReason::ToolCalls));
    assert_eq!(text(&ev, 0), "Reading.");

    let mut m2 = m1.clone();
    m2.push(as_message(&ev, 0));
    m2.push(tool_result(&c[0].id, "hi"));
    let ev2 = run(&h.engine, req("m/auto", m2.clone())).await;
    assert_eq!(text(&ev2, 0), "the file says hi");
    let recs = h.fake(0).records();
    assert_eq!(recs[1].base_kind, Some(CheckpointKind::Awaiting));
    assert!(recs[1].patched);
    assert_eq!(recs[1].appended, 1);
    assert_eq!(recs[1].trigger, "continue");

    // And the next user turn hits the Idle checkpoint after the tool round.
    let mut m3 = m2.clone();
    m3.push(as_message(&ev2, 0));
    m3.push(user("thanks"));
    run(&h.engine, req("m/auto", m3)).await;
    let recs = h.fake(0).records();
    assert_eq!(recs[2].base_kind, Some(CheckpointKind::Idle));
    assert_eq!(recs[2].base_messages, 4);
    assert_eq!(recs[2].appended, 0);
}

#[tokio::test]
async fn tool_args_key_order_change_still_hits() {
    let h = setup(vec![b(
        "a",
        1,
        json!({"rules": [{"match": "go", "tool_calls": [{"name": "read", "arguments": {"a": 1, "b": 2}}]}]}),
    )])
    .await;
    let m1 = vec![user("go")];
    let ev = run(&h.engine, req("m/auto", m1.clone())).await;
    let c = &calls(&ev, 0)[0];
    let reordered = ToolCall { id: c.id.clone(), name: c.name.clone(), arguments: "{ \"b\": 2, \"a\": 1 }".into() };
    let m2 = vec![
        user("go"),
        Message { role: Role::Assistant, parts: vec![Part::ToolCall(reordered)] },
        tool_result(&c.id, "r"),
    ];
    run(&h.engine, req("m/auto", m2)).await;
    assert_eq!(h.fake(0).records()[1].base_kind, Some(CheckpointKind::Awaiting));
}

#[tokio::test]
async fn regenerate_same_request_twice_uses_same_base() {
    let h = setup(vec![b("a", 1, json!({}))]).await;
    let m1 = vec![sys("s"), user("u1")];
    let ev = run(&h.engine, req("m/auto", m1.clone())).await;
    let mut m2 = m1.clone();
    m2.push(as_message(&ev, 0));
    m2.push(user("u2"));
    // Sequential and concurrent (CONV-3, CONC-2).
    let a = run(&h.engine, req("m/auto", m2.clone())).await;
    let (b1, b2) = tokio::join!(run(&h.engine, req("m/auto", m2.clone())), run(&h.engine, req("m/auto", m2.clone())));
    for ev in [&a, &b1, &b2] {
        assert_eq!(text(ev, 0), "echo: u2");
        assert_eq!(finish(ev, 0), Some(FinishReason::Stop));
    }
    let recs = h.fake(0).records();
    let base = recs[1].base_path.clone();
    assert!(base.is_some());
    assert_eq!(recs[2].base_path, base);
    assert_eq!(recs[3].base_path, base);
    // The base is untouched and still used; 1 + 3 checkpoints exist.
    assert_eq!(h.engine.checkpoint_stats().0, 4);
}

#[tokio::test]
async fn undo_edit_and_fork_answer_from_request_history() {
    let h = setup(vec![b("a", 1, json!({"reply": "history"}))]).await;
    // Build u1 → a1 → u2 → a2 → u3 → a3.
    let mut conv = vec![sys("s")];
    for u in ["u1", "u2", "u3"] {
        conv.push(user(u));
        let ev = run(&h.engine, req("m/auto", conv.clone())).await;
        conv.push(as_message(&ev, 0));
    }
    // Undo (CONV-5): drop the last two turns, ask something else.
    let mut undo = conv[..3].to_vec(); // sys u1 a1
    undo.push(user("u2-alt"));
    let ev = run(&h.engine, req("m/auto", undo)).await;
    assert_eq!(text(&ev, 0), "history: 3 messages; users: u1 | u2-alt; last user: u2-alt");
    let last = h.fake(0).records().last().cloned().unwrap();
    assert_eq!(last.base_messages, 2); // forked from the checkpoint after a1

    // Edit in the middle (CONV-4): u1 changed, rest kept.
    let mut edit = conv.clone();
    edit[1] = user("u1-edited");
    edit.push(user("u4"));
    let ev = run(&h.engine, req("m/auto", edit)).await;
    assert_eq!(text(&ev, 0), "history: 7 messages; users: u1-edited | u2 | u3 | u4; last user: u4");
    let last = h.fake(0).records().last().cloned().unwrap();
    assert!(last.base_path.is_none());
    assert_eq!(last.appended, 6);

    // Fork (CONV-6): another conversation sharing the first 4 messages.
    let mut fork = conv[..5].to_vec(); // sys u1 a1 u2 a2
    fork.push(user("other branch"));
    let ev = run(&h.engine, req("m/auto", fork)).await;
    assert_eq!(text(&ev, 0), "history: 5 messages; users: u1 | u2 | other branch; last user: other branch");
    assert_eq!(h.fake(0).records().last().unwrap().base_messages, 4);
}

#[tokio::test]
async fn foreign_history_without_checkpoint() {
    let h = setup(vec![b("a", 1, json!({"reply": "history"}))]).await;
    let msgs = vec![sys("s"), user("hi"), asst("I am another model"), sys("mid-system"), user("next")];
    let ev = run(&h.engine, req("m/auto", msgs)).await;
    assert_eq!(text(&ev, 0), "history: 4 messages; users: hi | next; last user: next");
    let rec = &h.fake(0).records()[0];
    assert!(rec.base_path.is_none());
    assert_eq!(rec.appended, 3); // hi, assistant, mid-system
    assert_eq!(rec.system, "s");
}

#[tokio::test]
async fn ends_with_tool_results_without_checkpoint_uses_continue() {
    let h = setup(vec![b("a", 1, json!({"rules": [{"match": "tool_results", "text": "ok"}]}))]).await;
    let call = ToolCall { id: "call:weird/1".into(), name: "read".into(), arguments: "{}".into() };
    let msgs = vec![
        user("u"),
        Message { role: Role::Assistant, parts: vec![Part::ToolCall(call)] },
        tool_result("call:weird/1", "data"),
    ];
    let ev = run(&h.engine, req("m/auto", msgs)).await;
    assert_eq!(text(&ev, 0), "ok");
    let rec = &h.fake(0).records()[0];
    assert_eq!(rec.trigger, "continue");
    assert_eq!(rec.appended, 3);
}

#[tokio::test]
async fn restart_keeps_checkpoints() {
    let dir = tempfile::tempdir().unwrap();
    let config = config_for(&[b("a", 1, json!({"reply": "history"}))]);
    let m1 = vec![sys("s"), user("u1")];
    let first = {
        let (engine, _fakes) = build(&config, dir.path()).await;
        run(&engine, req("m/auto", m1.clone())).await
    };
    let (engine, fakes) = build(&config, dir.path()).await;
    let mut m2 = m1.clone();
    m2.push(as_message(&first, 0));
    m2.push(user("u2"));
    let ev = run(&engine, req("m/auto", m2)).await;
    assert_eq!(text(&ev, 0), "history: 3 messages; users: u1 | u2; last user: u2");
    assert!(fakes[0].records()[0].base_path.is_some());
}

#[tokio::test]
async fn version_change_ignores_old_checkpoints_and_reruns_self_test() {
    let h = setup(vec![b("a", 1, json!({}))]).await;
    assert_eq!(h.fake(0).self_test_count(), 1);
    let m1 = vec![user("u1"), asst("x"), user("u2")];
    let ev = run(&h.engine, req("m/auto", m1.clone())).await;
    h.fake(0).set_version("fake-2");
    h.engine.expire_version_check();
    let mut m2 = m1.clone();
    m2.push(as_message(&ev, 0));
    m2.push(user("u3"));
    run(&h.engine, req("m/auto", m2)).await;
    assert!(h.fake(0).records()[1].base_path.is_none());
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(h.fake(0).self_test_count(), 2);
}

#[tokio::test]
async fn missing_snapshot_file_falls_back_to_full_history() {
    let h = setup(vec![b("a", 1, json!({"reply": "history"}))]).await;
    let m1 = vec![user("u1"), asst("a0"), user("u2")];
    let ev = run(&h.engine, req("m/auto", m1.clone())).await;
    let path = h.fake(0).records()[0].clone();
    assert!(path.frozen);
    // Delete every snapshot file behind the engine's back.
    for entry in std::fs::read_dir(h.dir.path().join("snapshots").join("a")).unwrap() {
        std::fs::remove_file(entry.unwrap().path()).unwrap();
    }
    let mut m2 = m1.clone();
    m2.push(as_message(&ev, 0));
    m2.push(user("u3"));
    let ev2 = run(&h.engine, req("m/auto", m2)).await;
    assert_eq!(text(&ev2, 0), "history: 5 messages; users: u1 | u2 | u3; last user: u3");
    let recs = h.fake(0).records();
    // The failed attempt with a base is not recorded by the fake (it fails while loading);
    // the successful one has no base.
    assert!(recs.last().unwrap().base_path.is_none());
}

// ---------------------------------------------------------------- side requests

#[tokio::test]
async fn side_requests_are_ephemeral_and_use_side_model() {
    let h = setup(vec![b("a", 1, json!({}))]).await;
    let mut r = ChatRequest::new("m/auto", vec![sys("Generate a title"), user("hello there")]);
    r.tools.clear();
    let ev = run(&h.engine, r).await;
    assert_eq!(text(&ev, 0), "echo: hello there");
    let rec = &h.fake(0).records()[0];
    assert!(rec.ephemeral);
    assert_eq!(rec.model, "small");
    assert!(!rec.frozen);
    assert!(rec.discarded);
    assert_eq!(h.engine.checkpoint_stats().0, 0);
    // A pinned model keeps its model even for side requests.
    let r = ChatRequest::new("m/a", vec![user("title")]);
    run(&h.engine, r).await;
    assert_eq!(h.fake(0).records()[1].model, "big");
    assert!(h.fake(0).records()[1].ephemeral);
}

// ---------------------------------------------------------------- failover and quota

#[tokio::test]
async fn failover_on_start_error_before_output() {
    let h = setup(vec![
        b("a", 1, json!({"rules": [{"match": "*", "error": "quota", "error_in_start": true, "retry_after": 100}]})),
        b("b", 2, json!({})),
    ])
    .await;
    let ev = run(&h.engine, req("m/auto", vec![user("hi"), asst("x"), user("again")])).await;
    assert_eq!(text(&ev, 0), "echo: again");
    assert_eq!(h.fake(1).records().len(), 1);
    let status = h.engine.status();
    let a = status["backends"].as_array().unwrap().iter().find(|b| b["name"] == "a").unwrap().clone();
    assert_eq!(a["quota"]["status"], "exhausted");
    // Next request skips the exhausted backend entirely.
    run(&h.engine, req("m/auto", vec![user("one more")])).await;
    assert_eq!(h.fake(0).records().len(), 1);
    assert_eq!(h.fake(1).records().len(), 2);
}

#[tokio::test]
async fn failover_on_first_event_error_and_on_timeout() {
    let h = setup_with(
        vec![
            b("a", 1, json!({"rules": [
                {"match": "unavail", "error": "unavailable"},
                {"match": "slow", "delay_ms": 3000}
            ]})),
            b("b", 2, json!({})),
        ],
        |c| c.server.first_token_timeout_s = 1,
    )
    .await;
    let ev = run(&h.engine, req("m/auto", vec![user("unavail")])).await;
    assert_eq!(text(&ev, 0), "echo: unavail");
    let ev = run(&h.engine, req("m/auto", vec![user("slow")])).await;
    assert_eq!(text(&ev, 0), "echo: slow");
    let recs = h.fake(0).records();
    assert!(recs[0].discarded);
    assert!(recs[1].aborted && recs[1].discarded);
    assert_eq!(h.fake(1).records().len(), 2);
}

#[tokio::test]
async fn failover_moves_full_history_to_new_backend() {
    let h = setup(vec![b("a", 1, json!({"reply": "history"})), b("b", 2, json!({"reply": "history"}))]).await;
    let m1 = vec![sys("s"), user("u1"), asst("a0"), user("u2")];
    let ev = run(&h.engine, req("m/auto", m1.clone())).await;
    assert_eq!(h.fake(0).records().len(), 1);
    // Switching backend mid-conversation (pinned here): b has no checkpoint, so the whole
    // history goes into `append` and the answer still reflects it (LEAK-7).
    let mut m2 = m1.clone();
    m2.push(as_message(&ev, 0));
    m2.push(user("u3"));
    let ev2 = run(&h.engine, req("m/b", m2)).await;
    assert_eq!(text(&ev2, 0), "history: 5 messages; users: u1 | u2 | u3; last user: u3");
    let rec = &h.fake(1).records()[0];
    assert!(rec.base_path.is_none());
    assert_eq!(rec.appended, 4);
}

#[tokio::test]
async fn sticky_routing_prefers_checkpoint_owner() {
    let h = setup(vec![b("a", 1, json!({})), b("b", 2, json!({}))]).await;
    // Create a checkpoint on b by pinning.
    let m1 = vec![user("u1"), asst("x"), user("u2")];
    let ev = run(&h.engine, req("m/b", m1.clone())).await;
    let mut m2 = m1.clone();
    m2.push(as_message(&ev, 0));
    m2.push(user("u3"));
    // Auto route: b owns the deepest checkpoint, so it goes first despite priority.
    run(&h.engine, req("m/auto", m2)).await;
    assert_eq!(h.fake(0).records().len(), 0);
    assert_eq!(h.fake(1).records().len(), 2);
    assert!(h.fake(1).records()[1].base_path.is_some());
}

#[tokio::test]
async fn single_exhausted_backend_returns_quota_error() {
    let h = setup(vec![b(
        "a",
        1,
        json!({"rules": [{"match": "*", "error": "quota", "error_in_start": true, "retry_after": 120}]}),
    )])
    .await;
    let ev = run(&h.engine, req("m/auto", vec![user("hi")])).await;
    assert_eq!(ev.len(), 1);
    let err = first_error(&ev).unwrap();
    assert_eq!(ev[0].index, 0);
    assert_eq!(err.class, ErrorClass::QuotaExhausted);
    let ra = err.retry_after.unwrap();
    assert!((110..=120).contains(&ra), "retry_after {ra}");
    assert_safe(&err);
    // Second request: rejected without touching the backend.
    let ev = run(&h.engine, req("m/auto", vec![user("hi")])).await;
    let err = first_error(&ev).unwrap();
    assert_eq!(err.class, ErrorClass::QuotaExhausted);
    assert!(err.retry_after.is_some());
    assert_eq!(h.fake(0).records().len(), 1);
    let log = h.engine.recent_requests(2);
    assert_eq!(log[0].error_class.as_deref(), Some("quota_exhausted"));
}

#[tokio::test]
async fn limit_event_marks_backend_exhausted() {
    let h = setup(vec![
        b("a", 1, json!({"limit": {"status": "exhausted", "resets_at": 4_000_000_000u64}})),
        b("b", 2, json!({})),
    ])
    .await;
    // The turn itself still completes on a.
    let ev = run(&h.engine, req("m/auto", vec![user("one")])).await;
    assert_eq!(text(&ev, 0), "echo: one");
    // Limit events never reach the caller.
    assert!(!ev.iter().any(|e| matches!(e.event, TurnEvent::Limit { .. })));
    run(&h.engine, req("m/auto", vec![user("two")])).await;
    assert_eq!(h.fake(0).records().len(), 1);
    assert_eq!(h.fake(1).records().len(), 1);
}

#[tokio::test]
async fn context_error_does_not_fail_over() {
    let h = setup(vec![
        b("a", 1, json!({"rules": [{"match": "*", "error": "context"}]})),
        b("b", 2, json!({})),
    ])
    .await;
    let ev = run(&h.engine, req("m/auto", vec![user("big")])).await;
    let err = first_error(&ev).unwrap();
    assert_eq!(err.class, ErrorClass::ContextTooLong);
    assert_safe(&err);
    assert_eq!(h.fake(1).records().len(), 0);
}

#[tokio::test]
async fn error_after_output_is_a_mid_stream_error() {
    let h = setup(vec![
        b("a", 1, json!({"rules": [{"match": "*", "text": "partial answer", "deltas": 2, "error": "unavailable", "error_after_events": 1}]})),
        b("b", 2, json!({})),
    ])
    .await;
    let ev = run(&h.engine, req("m/auto", vec![user("x")])).await;
    assert!(matches!(ev[0].event, TurnEvent::TextDelta { .. }));
    let last = ev.last().unwrap();
    match &last.event {
        TurnEvent::Error { error } => {
            assert_eq!(error.class, ErrorClass::Unavailable);
            assert_safe(error);
        }
        other => panic!("expected error, got {other:?}"),
    }
    assert_eq!(h.fake(1).records().len(), 0);
    assert_eq!(h.engine.checkpoint_stats().0, 0);
}

#[tokio::test]
async fn pinned_model_uses_only_its_backend_and_unknown_model_is_404() {
    let h = setup(vec![
        b("a", 1, json!({"rules": [{"match": "*", "error": "unavailable", "error_in_start": true}]})),
        b("b", 2, json!({})),
    ])
    .await;
    let ev = run(&h.engine, req("m/a", vec![user("x")])).await;
    let err = first_error(&ev).unwrap();
    assert_eq!(err.class, ErrorClass::Unavailable);
    assert_eq!(h.fake(1).records().len(), 0);
    assert_eq!(h.fake(0).records()[0].model, "big");

    let ev = run(&h.engine, req("nope", vec![user("x")])).await;
    assert_eq!(first_error(&ev).unwrap().class, ErrorClass::ModelNotFound);
}

#[tokio::test]
async fn disabled_and_transcript_modes() {
    let h = setup(vec![b("a", 1, json!({"mode": "disabled"})), b("b", 2, json!({"mode": "transcript"}))]).await;
    let m1 = vec![user("u1"), asst("x"), user("u2")];
    let ev = run(&h.engine, req("m/auto", m1.clone())).await;
    assert_eq!(text(&ev, 0), "echo: u2");
    assert_eq!(h.fake(0).records().len(), 0);
    // Transcript mode: no checkpoints are created or used.
    assert_eq!(h.engine.checkpoint_stats().0, 0);
    let mut m2 = m1.clone();
    m2.push(as_message(&ev, 0));
    m2.push(user("u3"));
    run(&h.engine, req("m/auto", m2)).await;
    assert!(h.fake(1).records()[1].base_path.is_none());
    let st = h.engine.status();
    let modes: Vec<String> =
        st["backends"].as_array().unwrap().iter().map(|b| b["mode"].as_str().unwrap().to_string()).collect();
    assert_eq!(modes, vec!["disabled", "transcript"]);
}

// ---------------------------------------------------------------- concurrency, cancellation

#[tokio::test]
async fn concurrency_limit_returns_rate_limited() {
    let mut be = b("a", 1, json!({"rules": [{"match": "slow", "delay_ms": 1500}]}));
    be.max_concurrency = 1;
    be.queue_wait_ms = 100;
    let h = setup(vec![be]).await;
    let engine = h.engine.clone();
    let slow = tokio::spawn(async move { run(&engine, req("m/auto", vec![user("slow")])).await });
    tokio::time::sleep(Duration::from_millis(200)).await;
    let ev = run(&h.engine, req("m/auto", vec![user("fast")])).await;
    let err = first_error(&ev).unwrap();
    assert_eq!(err.class, ErrorClass::RateLimited);
    assert!(err.retry_after.is_some());
    assert_safe(&err);
    let slow = slow.await.unwrap();
    assert_eq!(text(&slow, 0), "echo: slow");
    // Slot freed: works again.
    let ev = run(&h.engine, req("m/auto", vec![user("fast")])).await;
    assert_eq!(text(&ev, 0), "echo: fast");
}

#[tokio::test]
async fn queued_request_waits_for_slot() {
    let mut be = b("a", 1, json!({"rules": [{"match": "slow", "delay_ms": 300}]}));
    be.max_concurrency = 1;
    be.queue_wait_ms = 3000;
    let h = setup(vec![be]).await;
    let (x, y) = tokio::join!(
        run(&h.engine, req("m/auto", vec![user("slow")])),
        run(&h.engine, req("m/auto", vec![user("slow 2")]))
    );
    assert_eq!(text(&x, 0), "echo: slow");
    assert_eq!(text(&y, 0), "echo: slow 2");
}

#[tokio::test]
async fn cancellation_aborts_and_discards() {
    let h = setup(vec![b("a", 1, json!({"rules": [{"match": "*", "text": "a b c d e f", "deltas": 6, "delta_delay_ms": 300}]}))])
        .await;
    let cancel = CancellationToken::new();
    let mut stream = h.engine.chat(req("m/auto", vec![user("x")]), cancel.clone()).await;
    let first = stream.next().await.unwrap();
    assert!(matches!(first.event, TurnEvent::TextDelta { .. }));
    cancel.cancel();
    let rest = tokio::time::timeout(Duration::from_secs(2), stream.collect::<Vec<_>>()).await.unwrap();
    assert!(rest.iter().all(|e| !matches!(e.event, TurnEvent::Done { .. })));
    tokio::time::sleep(Duration::from_millis(50)).await;
    let rec = &h.fake(0).records()[0];
    assert!(rec.aborted && rec.discarded && !rec.frozen);
    assert_eq!(h.engine.checkpoint_stats().0, 0);
    let st = h.engine.status();
    assert_eq!(st["backends"][0]["in_flight"], 0);
}

#[tokio::test]
async fn dropping_the_stream_aborts() {
    let h = setup(vec![b("a", 1, json!({"rules": [{"match": "*", "text": "a b c d", "deltas": 4, "delta_delay_ms": 300}]}))])
        .await;
    let mut stream = h.engine.chat(req("m/auto", vec![user("x")]), CancellationToken::new()).await;
    let _ = stream.next().await;
    drop(stream);
    tokio::time::sleep(Duration::from_millis(200)).await;
    let rec = &h.fake(0).records()[0];
    assert!(rec.aborted && rec.discarded);
}

// ---------------------------------------------------------------- n, policy

#[tokio::test]
async fn n_two_runs_two_forks() {
    let h = setup(vec![b("a", 1, json!({}))]).await;
    let m1 = vec![user("u1"), asst("x"), user("u2")];
    let ev = run(&h.engine, req("m/auto", m1.clone())).await;
    let mut m2 = m1.clone();
    m2.push(as_message(&ev, 0));
    m2.push(user("u3"));
    let mut r = req("m/auto", m2);
    r.n = 2;
    let ev = run(&h.engine, r).await;
    assert_eq!(text(&ev, 0), "echo: u3");
    assert_eq!(text(&ev, 1), "echo: u3");
    assert_eq!(finish(&ev, 0), Some(FinishReason::Stop));
    assert_eq!(finish(&ev, 1), Some(FinishReason::Stop));
    let recs = h.fake(0).records();
    assert_eq!(recs.len(), 3);
    assert!(recs[1].base_path.is_some());
    assert_eq!(recs[1].base_path, recs[2].base_path);
}

#[tokio::test]
async fn n_two_with_one_early_failure_returns_only_the_error() {
    let h = setup(vec![b(
        "a",
        1,
        json!({"rules": [{"match": "*", "error": "context", "error_in_start": true, "times": 1}]}),
    )])
    .await;
    let mut r = req("m/auto", vec![user("x")]);
    r.n = 2;
    let ev = run(&h.engine, r).await;
    assert_eq!(ev.len(), 1, "{ev:?}");
    assert_eq!(ev[0].index, 0);
    assert_eq!(first_error(&ev).unwrap().class, ErrorClass::ContextTooLong);
}

#[tokio::test]
async fn policy_retry_tool_choice_required() {
    let h = setup(vec![b(
        "a",
        1,
        json!({"rules": [
            {"match": "do it", "text": "I'd rather talk.", "times": 1},
            {"match": "do it", "tool_calls": [{"name": "read", "arguments": {"path": "p"}}]}
        ]}),
    )])
    .await;
    let mut r = req("m/auto", vec![user("do it")]);
    r.tool_choice = ToolChoice::Required;
    let ev = run(&h.engine, r).await;
    assert_eq!(text(&ev, 0), "");
    assert_eq!(calls(&ev, 0).len(), 1);
    assert_eq!(finish(&ev, 0), Some(FinishReason::ToolCalls));
    let recs = h.fake(0).records();
    assert_eq!(recs.len(), 2);
    assert!(recs[0].discarded && !recs[0].frozen);
    assert!(recs[1].frozen);
    assert!(recs[1].system.contains("must call at least one tool"));
}

#[tokio::test]
async fn altered_output_creates_no_checkpoint() {
    let h = setup(vec![b("a", 1, json!({"rules": [{"match": "*", "text": "one two STOP three"}]}))]).await;
    let mut r = req("m/auto", vec![user("x")]);
    r.stop = vec!["STOP".into()];
    let ev = run(&h.engine, r).await;
    assert_eq!(text(&ev, 0), "one two ");
    assert_eq!(finish(&ev, 0), Some(FinishReason::Stop));
    assert_eq!(h.engine.checkpoint_stats().0, 0);
    let rec = &h.fake(0).records()[0];
    assert!(rec.aborted && rec.discarded);
}

#[tokio::test]
async fn reasoning_is_forwarded_and_effort_schema_passed() {
    let h = setup(vec![b("a", 1, json!({"json_schema": true, "rules": [{"match": "*", "text": "{\"a\":1}", "reasoning": "hmm"}]}))])
        .await;
    let mut r = req("m/auto", vec![user("x")]);
    r.reasoning_effort = Some(bridge_types::ReasoningEffort::High);
    r.response_format = bridge_types::ResponseFormat::JsonSchema {
        name: "s".into(),
        schema: json!({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}),
        strict: true,
    };
    let ev = run(&h.engine, r).await;
    assert!(ev.iter().any(|e| matches!(&e.event, TurnEvent::ReasoningDelta { text } if text == "hmm")));
    assert_eq!(text(&ev, 0), "{\"a\":1}");
    let rec = &h.fake(0).records()[0];
    assert!(rec.json_schema);
    assert_eq!(rec.effort, Some(bridge_types::ReasoningEffort::High));
}

// ---------------------------------------------------------------- gc, models, status

#[tokio::test]
async fn gc_deletes_snapshots_through_backend() {
    let h = setup_with(vec![b("a", 1, json!({}))], |c| c.checkpoints.max_total_mb = 0).await;
    run(&h.engine, req("m/auto", vec![user("u1"), asst("x"), user("u2")])).await;
    run(&h.engine, req("m/auto", vec![user("v1"), asst("x"), user("v2")])).await;
    // A cap of 0 is exceeded by every checkpoint, so it is enforced right after each
    // request (not only on the periodic sweep). The caller's stream ends before the
    // checkpoint is frozen and the post-request GC runs, so wait for that background work.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while h.fake(0).deleted_snapshots().len() < 2 && std::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(h.engine.checkpoint_stats().0, 0);
    let deleted = h.fake(0).deleted_snapshots();
    assert_eq!(deleted.len(), 2);
    assert!(deleted.iter().all(|p| !Path::new(p).exists()));
}

#[tokio::test]
async fn models_and_status() {
    let h = setup(vec![b("a", 1, json!({})), b("b", 2, json!({}))]).await;
    let models = h.engine.models();
    let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(ids, vec!["m/a", "m/auto", "m/b"]);
    assert!(models.iter().all(|m| m.owned_by == "completions-bridge" && m.created > 0));
    assert_eq!(models[0].context_limit, Some(h.config.backends["a"].context_limit));
    let st = h.engine.status();
    assert_eq!(st["backends"].as_array().unwrap().len(), 2);
    assert_eq!(st["backends"][0]["version"], "fake-1");
    assert_eq!(st["backends"][0]["quota"]["status"], "available");
    assert!(st["uptime_s"].is_u64());
    assert_eq!(st["checkpoints"]["count"], 0);
}

// ---------------------------------------------------------------- pending checkpoints (G2, 17.1)

/// Read the stream until every choice reported `Done` (or an error), then drop it, like
/// a client that hangs up right after `[DONE]`. Returns the events and the time to the
/// last `Done`.
async fn run_until_done(engine: &Engine, r: ChatRequest, cancel: CancellationToken) -> (Vec<ChoiceEvent>, Duration) {
    let n = r.n.max(1) as usize;
    let started = std::time::Instant::now();
    let mut stream = engine.chat(r, cancel).await;
    let mut out = Vec::new();
    let mut done = 0;
    let fut = async {
        while let Some(ev) = stream.next().await {
            let end = matches!(ev.event, TurnEvent::Done { .. } | TurnEvent::Error { .. });
            out.push(ev);
            if end {
                done += 1;
                if done == n {
                    break;
                }
            }
        }
    };
    tokio::time::timeout(Duration::from_secs(20), fut).await.expect("Done in time");
    (out, started.elapsed())
}

/// Rules for a tool loop: "go" calls `read`; after results, `rounds` more calls, then text.
fn tool_loop_rules(rounds: u32, extra: serde_json::Value) -> serde_json::Value {
    let mut o = json!({"rules": [
        {"match": "go", "tool_calls": [{"name": "read", "arguments": {"path": "a"}}]},
        {"match": "tool_results", "tool_calls": [{"name": "read", "arguments": {"path": "b"}}], "times": rounds},
        {"match": "tool_results", "text": "all done"}
    ]});
    if let (Some(o), Some(extra)) = (o.as_object_mut(), extra.as_object()) {
        for (k, v) in extra {
            o.insert(k.clone(), v.clone());
        }
    }
    o
}

/// The next request of a tool loop: previous messages + the answer + its tool results.
fn next_round(prev: &[Message], ev: &[ChoiceEvent]) -> Vec<Message> {
    let mut m = prev.to_vec();
    m.push(as_message(ev, 0));
    for c in calls(ev, 0) {
        m.push(tool_result(&c.id, "file contents"));
    }
    m
}

#[tokio::test]
async fn conc7_next_request_right_after_done_hits_newest_checkpoint() {
    // Small freeze delay so the race against the save is real on every round.
    let h = setup(vec![b("a", 1, tool_loop_rules(4, json!({"freeze_delay_ms": 60})))]).await;
    let mut msgs = vec![sys("s"), user("go")];
    for round in 0..6 {
        let (ev, _) = run_until_done(&h.engine, req("m/auto", msgs.clone()), CancellationToken::new()).await;
        let rec = h.fake(0).records()[round].clone();
        if round == 0 {
            assert!(rec.base_path.is_none());
        } else {
            // Base = the checkpoint of the turn that just finished (2 messages per round).
            assert_eq!(rec.base_kind, Some(CheckpointKind::Awaiting), "round {round}");
            assert_eq!(rec.base_messages, 2 * round, "round {round}");
            assert!(rec.patched);
            assert_eq!(rec.appended, 1);
        }
        if round == 5 {
            assert_eq!(text(&ev, 0), "all done");
            break;
        }
        assert_eq!(finish(&ev, 0), Some(FinishReason::ToolCalls));
        msgs = next_round(&msgs, &ev);
    }
    // Request log: depth = request length - 1 on every tool round.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let log = h.engine.recent_requests(6);
    let mut depths: Vec<u64> = log.iter().filter_map(|r| r.checkpoint_hit_depth).collect();
    depths.sort_unstable();
    assert_eq!(depths, vec![0, 3, 5, 7, 9, 11]);
}

#[tokio::test]
async fn s19_waits_for_pending_checkpoint_and_continues_warm() {
    let h = setup(vec![b("a", 1, tool_loop_rules(0, json!({"freeze_delay_ms": 400})))]).await;
    let m1 = vec![sys("s"), user("go")];
    let (ev, _) = run_until_done(&h.engine, req("m/auto", m1.clone()), CancellationToken::new()).await;
    let m2 = next_round(&m1, &ev);
    let (ev2, elapsed) = run_until_done(&h.engine, req("m/auto", m2), CancellationToken::new()).await;
    assert_eq!(text(&ev2, 0), "all done");
    assert!(elapsed >= Duration::from_millis(250), "it waited for the freeze: {elapsed:?}");
    let rec = h.fake(0).records()[1].clone();
    assert_eq!(rec.base_kind, Some(CheckpointKind::Awaiting));
    assert_eq!(rec.base_messages, 2);
    assert!(rec.patched);
    assert_eq!(rec.appended, 1);
}

#[tokio::test]
async fn s20_freeze_failure_while_waiting_falls_back() {
    let h = setup(vec![b("a", 1, tool_loop_rules(0, json!({"freeze_delay_ms": 300, "freeze_fail": true})))]).await;
    let m1 = vec![sys("s"), user("go")];
    let (ev, _) = run_until_done(&h.engine, req("m/auto", m1.clone()), CancellationToken::new()).await;
    let m2 = next_round(&m1, &ev);
    let (ev2, elapsed) = run_until_done(&h.engine, req("m/auto", m2), CancellationToken::new()).await;
    assert_eq!(text(&ev2, 0), "all done");
    assert!(first_error(&ev2).is_none());
    assert!(elapsed >= Duration::from_millis(200), "waited for the failing freeze: {elapsed:?}");
    assert!(elapsed < Duration::from_millis(3000), "did not wait for the full budget: {elapsed:?}");
    let rec = h.fake(0).records()[1].clone();
    assert!(rec.base_path.is_none());
    assert_eq!(rec.appended, 3); // u, a1, t1 rewritten from the request
    assert_eq!(rec.trigger, "continue");
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(h.engine.status()["checkpoints"]["pending"], 0);
    assert_eq!(h.engine.checkpoint_stats().0, 0);
}

#[tokio::test]
async fn s21_wait_timeout_falls_back_and_late_checkpoint_is_still_indexed() {
    let h = setup_with(vec![b("a", 1, tool_loop_rules(0, json!({"freeze_delay_ms": 1200})))], |c| {
        c.engine.pending_wait_ms = 200
    })
    .await;
    let m1 = vec![sys("s"), user("go")];
    let (ev, _) = run_until_done(&h.engine, req("m/auto", m1.clone()), CancellationToken::new()).await;
    let m2 = next_round(&m1, &ev);
    let (ev2, elapsed) = run_until_done(&h.engine, req("m/auto", m2.clone()), CancellationToken::new()).await;
    assert_eq!(text(&ev2, 0), "all done");
    assert!(elapsed >= Duration::from_millis(150) && elapsed < Duration::from_millis(1000), "{elapsed:?}");
    assert!(h.fake(0).records()[1].base_path.is_none());

    // The late checkpoint still lands and serves a later request (regeneration of m2).
    tokio::time::sleep(Duration::from_millis(1500)).await;
    run(&h.engine, req("m/auto", m2)).await;
    let rec = h.fake(0).records()[2].clone();
    assert_eq!(rec.base_kind, Some(CheckpointKind::Awaiting));
    assert_eq!(rec.base_messages, 2);
}

#[tokio::test]
async fn s22_regenerate_during_pending_freeze_does_not_wait() {
    let h = setup(vec![b("a", 1, json!({"freeze_delay_ms": 1500}))]).await;
    let m1 = vec![sys("s"), user("u1"), asst("x"), user("u2")];
    let (ev, _) = run_until_done(&h.engine, req("m/auto", m1.clone()), CancellationToken::new()).await;
    assert_eq!(text(&ev, 0), "echo: u2");
    // Same request again: the promise (depth 5) is not a prefix of it.
    let (ev2, elapsed) = run_until_done(&h.engine, req("m/auto", m1), CancellationToken::new()).await;
    assert_eq!(text(&ev2, 0), "echo: u2");
    assert!(elapsed < Duration::from_millis(1000), "no wait: {elapsed:?}");
    assert!(h.fake(0).records()[1].base_path.is_none());
}

#[tokio::test]
async fn s23_cancel_while_waiting_stops_immediately() {
    let h = setup(vec![b("a", 1, tool_loop_rules(0, json!({"freeze_delay_ms": 3000})))]).await;
    let m1 = vec![sys("s"), user("go")];
    let (ev, _) = run_until_done(&h.engine, req("m/auto", m1.clone()), CancellationToken::new()).await;
    let m2 = next_round(&m1, &ev);
    let cancel = CancellationToken::new();
    let started = std::time::Instant::now();
    let stream = h.engine.chat(req("m/auto", m2), cancel.clone()).await;
    let c = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        c.cancel();
    });
    let rest = tokio::time::timeout(Duration::from_secs(2), stream.collect::<Vec<_>>()).await.expect("stream ended");
    assert!(started.elapsed() < Duration::from_millis(1000), "{:?}", started.elapsed());
    assert!(rest.iter().all(|e| !matches!(e.event, TurnEvent::Done { .. })));
    // Nothing was started on the backend for the cancelled request.
    assert_eq!(h.fake(0).records().len(), 1);
    let log = h.engine.recent_requests(1);
    assert_eq!(log[0].error_class.as_deref(), Some("cancelled"));
}

#[tokio::test]
async fn waiting_does_not_hold_a_permit() {
    // One slot and no queueing: if the waiter held (or raced for) the slot while the
    // previous turn still owned it, this would deadlock or return 429.
    let mut be = b("a", 1, tool_loop_rules(0, json!({"freeze_delay_ms": 300})));
    be.max_concurrency = 1;
    be.queue_wait_ms = 0;
    let h = setup(vec![be]).await;
    let m1 = vec![sys("s"), user("go")];
    let (ev, _) = run_until_done(&h.engine, req("m/auto", m1.clone()), CancellationToken::new()).await;
    let m2 = next_round(&m1, &ev);
    let (ev2, _) = run_until_done(&h.engine, req("m/auto", m2), CancellationToken::new()).await;
    assert!(first_error(&ev2).is_none(), "{ev2:?}");
    assert_eq!(text(&ev2, 0), "all done");
    assert_eq!(h.fake(0).records()[1].base_messages, 2);
}

#[tokio::test]
async fn n_two_registers_one_promise_per_choice() {
    let h = setup(vec![b("a", 1, json!({"freeze_delay_ms": 300}))]).await;
    let m1 = vec![sys("s"), user("u1"), asst("x"), user("u2")];
    let mut r = req("m/auto", m1.clone());
    r.n = 2;
    let (ev, _) = run_until_done(&h.engine, r, CancellationToken::new()).await;
    assert_eq!(finish(&ev, 0), Some(FinishReason::Stop));
    assert_eq!(finish(&ev, 1), Some(FinishReason::Stop));
    // Both choices produced the same text: duplicate keys, either promise serves.
    let mut m2 = m1.clone();
    m2.push(as_message(&ev, 1));
    m2.push(user("u3"));
    let (ev2, _) = run_until_done(&h.engine, req("m/auto", m2), CancellationToken::new()).await;
    assert_eq!(text(&ev2, 0), "echo: u3");
    let rec = h.fake(0).records()[2].clone();
    assert_eq!(rec.base_kind, Some(CheckpointKind::Idle));
    assert_eq!(rec.base_messages, 4);
    tokio::time::sleep(Duration::from_millis(800)).await;
    assert_eq!(h.engine.checkpoint_stats().0, 3);
}

#[tokio::test]
async fn altered_and_side_turns_promise_nothing() {
    // An altered (stop sequence) turn never registers a promise, so the follow-up
    // does not wait for the slow freeze that never comes.
    let h = setup(vec![b("a", 1, json!({"freeze_delay_ms": 2000, "rules": [{"match": "*", "text": "one STOP two"}]}))])
        .await;
    let m1 = vec![sys("s"), user("u1")];
    let mut r = req("m/auto", m1.clone());
    r.stop = vec!["STOP".into()];
    let (ev, _) = run_until_done(&h.engine, r, CancellationToken::new()).await;
    assert_eq!(h.engine.status()["checkpoints"]["pending"], 0);
    let mut m2 = m1.clone();
    m2.push(as_message(&ev, 0));
    m2.push(user("u2"));
    let (_, elapsed) = run_until_done(&h.engine, req("m/auto", m2), CancellationToken::new()).await;
    assert!(elapsed < Duration::from_millis(1000), "{elapsed:?}");
}

// ---------------------------------------------------------------- streamed tool calls (G3, 17.3)

fn streamed_args(events: &[ChoiceEvent], index: u32, call: u32) -> (Option<(String, String)>, String) {
    let mut start = None;
    let mut args = String::new();
    for e in events.iter().filter(|e| e.index == index) {
        match &e.event {
            TurnEvent::ToolCallStart { index: i, id, name } if *i == call => start = Some((id.clone(), name.clone())),
            TurnEvent::ToolCallDelta { index: i, arguments } if *i == call => args.push_str(arguments),
            _ => {}
        }
    }
    (start, args)
}

#[tokio::test]
async fn streamed_tool_events_pass_through() {
    let h = setup(vec![b(
        "a",
        1,
        json!({"stream_tool_args": true, "rules": [
            {"match": "write", "deltas": 5, "tool_calls": [
                {"name": "read", "arguments": {"path": "a/very/long/path.txt", "x": [1, 2, 3]}},
                {"name": "read", "arguments": {"path": "b"}}
            ]},
            {"match": "tool_results", "text": "ok"}
        ]}),
    )])
    .await;
    let m1 = vec![sys("s"), user("write it")];
    let ev = run(&h.engine, req("m/auto", m1.clone())).await;
    let committed = calls(&ev, 0);
    assert_eq!(committed.len(), 2);
    // Previews come first and reconstruct the commit byte for byte.
    assert!(matches!(ev[0].event, TurnEvent::ToolCallStart { index: 0, .. }), "{:?}", ev[0]);
    for (i, c) in committed.iter().enumerate() {
        let (start, args) = streamed_args(&ev, 0, i as u32);
        assert_eq!(start, Some((c.id.clone(), c.name.clone())));
        assert_eq!(args, c.arguments);
    }
    let deltas = ev.iter().filter(|e| matches!(e.event, TurnEvent::ToolCallDelta { .. })).count();
    assert!(deltas >= 4, "several deltas: {deltas}");
    assert_eq!(finish(&ev, 0), Some(FinishReason::ToolCalls));
    // Unaltered: checkpointed, and the next round hits it.
    let m2 = next_round(&m1, &ev);
    let ev2 = run(&h.engine, req("m/auto", m2)).await;
    assert_eq!(text(&ev2, 0), "ok");
    assert_eq!(h.fake(0).records()[1].base_kind, Some(CheckpointKind::Awaiting));
}

#[tokio::test]
async fn streamed_tool_start_locks_out_failover() {
    // Error after Start + one Delta: the caller already has output, so this is a
    // mid-stream error and backend b is never tried.
    let h = setup(vec![
        b(
            "a",
            1,
            json!({"stream_tool_args": true, "rules": [{"match": "*", "deltas": 4,
                "tool_calls": [{"name": "read", "arguments": {"path": "abcdefgh"}}],
                "error": "unavailable", "error_after_events": 2}]}),
        ),
        b("b", 2, json!({})),
    ])
    .await;
    let ev = run(&h.engine, req("m/auto", vec![user("x")])).await;
    assert!(matches!(ev[0].event, TurnEvent::ToolCallStart { index: 0, .. }), "{ev:?}");
    match &ev.last().unwrap().event {
        TurnEvent::Error { error } => {
            assert_eq!(error.class, ErrorClass::Unavailable);
            assert_safe(error);
        }
        other => panic!("expected a mid-stream error, got {other:?}"),
    }
    assert_eq!(h.fake(1).records().len(), 0);
    assert_eq!(h.engine.checkpoint_stats().0, 0);
}

#[tokio::test]
async fn streamed_tool_start_satisfies_first_token_timeout() {
    // STR-17 shape: the call takes ~2.4 s (> first_token_timeout 1 s) but Start arrives at
    // once, so it is neither a timeout nor failed over. (The engine's timeout covers only
    // the first backend event of any kind; this guards against that regressing.)
    let h = setup_with(
        vec![
            b(
                "a",
                1,
                json!({"stream_tool_args": true, "rules": [{"match": "*", "deltas": 8, "delta_delay_ms": 250,
                    "tool_calls": [{"name": "read", "arguments": {"path": "0123456789abcdef"}}]}]}),
            ),
            b("b", 2, json!({})),
        ],
        |c| c.server.first_token_timeout_s = 1,
    )
    .await;
    let ev = run(&h.engine, req("m/auto", vec![user("x")])).await;
    assert_eq!(finish(&ev, 0), Some(FinishReason::ToolCalls));
    assert!(first_error(&ev).is_none());
    assert_eq!(h.fake(1).records().len(), 0);
}

#[tokio::test]
async fn buffered_streamed_call_is_live_without_emitting() {
    // tool_choice required: the policy swallows Start/Delta, so the caller sees nothing
    // until the commit (~2.4 s). Backend previews still count as liveness, so the 1 s
    // first-token timeout must not fire (STR-17) and b is never tried.
    let h = setup_with(
        vec![
            b(
                "a",
                1,
                json!({"stream_tool_args": true, "rules": [{"match": "*", "deltas": 8, "delta_delay_ms": 300,
                    "tool_calls": [{"name": "read", "arguments": {"path": "0123456789abcdef"}}]}]}),
            ),
            b("b", 2, json!({})),
        ],
        |c| c.server.first_token_timeout_s = 1,
    )
    .await;
    let mut r = req("m/auto", vec![user("x")]);
    r.tool_choice = ToolChoice::Required;
    let ev = run(&h.engine, r).await;
    assert!(first_error(&ev).is_none(), "{ev:?}");
    assert_eq!(finish(&ev, 0), Some(FinishReason::ToolCalls));
    assert_eq!(calls(&ev, 0).len(), 1);
    assert!(!ev.iter().any(|e| matches!(e.event, TurnEvent::ToolCallStart { .. } | TurnEvent::ToolCallDelta { .. })));
    assert_eq!(h.fake(1).records().len(), 0);
}
