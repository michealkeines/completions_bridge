//! Scripted fake backend for tests and for running the conformance suite without quota.
//! Configured via `BackendConfig { kind: "fake", options: {...} }`.
//!
//! Options (all optional):
//! ```json
//! {
//!   "version": "fake-1",
//!   "mode": "native" | "transcript" | "disabled",
//!   "reply": "echo" | "history",
//!   "json_schema": false,            // capability flag
//!   "limit": {"status": "warning", "utilization": 0.9, "resets_at": 1700000000},
//!   "stream_tool_args": false,       // ToolCallStart + ToolCallDelta* before the ToolCalls commit
//!   "freeze_delay_ms": 0,            // freeze takes this long (pending checkpoint tests)
//!   "freeze_fail": false,            // freeze returns an error (after the delay)
//!   "rules": [{
//!     "match": "<substring of the last user text>" | "tool_results" | "*",
//!     "text": "...", "tool_calls": [{"name": "read", "arguments": {"path": "a"}}],
//!     "reasoning": "...",
//!     "error": "quota|unavailable|context|auth|timeout|internal",
//!     "error_after_events": 0, "error_in_start": false, "retry_after": 30,
//!     "delay_ms": 0, "delta_delay_ms": 0, "deltas": 3, "times": 1
//!   }]
//! }
//! ```
//! Default reply ("echo"): `echo: <last user text>`. "history" reply:
//! `history: N messages; users: a | b; last user: b` where N counts every message the
//! model saw (base snapshot + appended + trigger; leading system messages are passed as
//! the system prompt and not counted).
//!
//! Snapshots are real: `freeze` writes the full history (including the produced
//! assistant message) as JSON into `snapshot_dir`, and a turn with a base loads it.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bridge_types::{
    Backend, BackendConfig, BackendMode, BridgeError, Capabilities, CheckpointKind, ErrorClass, FinishReason,
    LimitInfo, Message, Part, ReasoningEffort, Role, SnapshotInfo, ToolCall, Trigger, Turn, TurnControl, TurnEvent,
    TurnSpec, Usage,
};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct FakeOptions {
    version: Option<String>,
    mode: Option<String>,
    reply: Option<String>,
    json_schema: bool,
    limit: Option<LimitInfo>,
    rules: Vec<FakeRule>,
    stream_tool_args: bool,
    freeze_delay_ms: u64,
    freeze_fail: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct FakeRule {
    #[serde(rename = "match")]
    pattern: String,
    text: Option<String>,
    tool_calls: Vec<FakeToolCall>,
    reasoning: Option<String>,
    error: Option<String>,
    error_after_events: usize,
    error_in_start: bool,
    retry_after: Option<u64>,
    delay_ms: u64,
    delta_delay_ms: u64,
    deltas: Option<usize>,
    times: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct FakeToolCall {
    name: String,
    arguments: serde_json::Value,
}

/// Diagnostics for one `start_turn` call.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct TurnRecord {
    /// Snapshot path of the base checkpoint, if any.
    pub base_path: Option<String>,
    pub base_kind: Option<CheckpointKind>,
    /// Messages loaded from the base snapshot.
    pub base_messages: usize,
    /// Messages in `spec.append`.
    pub appended: usize,
    /// True if the base was Awaiting and `append` started with its results.
    pub patched: bool,
    /// "user" or "continue".
    pub trigger: String,
    /// Messages the model saw (base + append + trigger).
    pub history_len: usize,
    pub model: String,
    pub system: String,
    pub tools: Vec<String>,
    pub effort: Option<ReasoningEffort>,
    pub json_schema: bool,
    pub ephemeral: bool,
    /// The text reply this turn produces (before any policy).
    pub reply: String,
    pub aborted: bool,
    pub frozen: bool,
    pub discarded: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct SnapshotFile {
    backend: String,
    version: String,
    history: Vec<Message>,
    pending_call_ids: Vec<String>,
}

pub struct FakeBackend {
    name: String,
    version: Mutex<String>,
    configured_mode: BackendMode,
    mode: Mutex<BackendMode>,
    options: FakeOptions,
    rule_uses: Vec<AtomicU32>,
    snapshot_dir: PathBuf,
    records: Arc<Mutex<Vec<TurnRecord>>>,
    deleted: Arc<Mutex<Vec<String>>>,
    self_tests: AtomicU32,
    context_limit: u64,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn parse_mode(s: &str) -> BackendMode {
    match s {
        "transcript" => BackendMode::Transcript,
        "disabled" => BackendMode::Disabled,
        _ => BackendMode::Native,
    }
}

impl FakeBackend {
    pub fn new(name: &str, config: &BackendConfig, snapshot_dir: &std::path::Path) -> anyhow::Result<Self> {
        let options: FakeOptions = if config.options.is_null() {
            FakeOptions::default()
        } else {
            serde_json::from_value(config.options.clone())
                .map_err(|e| anyhow::anyhow!("invalid options for fake backend {name}: {e}"))?
        };
        std::fs::create_dir_all(snapshot_dir)?;
        let configured_mode = parse_mode(options.mode.as_deref().unwrap_or("native"));
        Ok(FakeBackend {
            name: name.to_string(),
            version: Mutex::new(options.version.clone().unwrap_or_else(|| "fake-1".into())),
            configured_mode,
            mode: Mutex::new(configured_mode),
            rule_uses: options.rules.iter().map(|_| AtomicU32::new(0)).collect(),
            options,
            snapshot_dir: snapshot_dir.to_path_buf(),
            records: Arc::new(Mutex::new(Vec::new())),
            deleted: Arc::new(Mutex::new(Vec::new())),
            self_tests: AtomicU32::new(0),
            context_limit: config.context_limit,
        })
    }

    /// Per-turn diagnostics, in `start_turn` order.
    pub fn records(&self) -> Vec<TurnRecord> {
        lock(&self.records).clone()
    }

    /// Snapshot paths removed through `delete_snapshot`.
    pub fn deleted_snapshots(&self) -> Vec<String> {
        lock(&self.deleted).clone()
    }

    /// Number of `self_test` calls so far.
    pub fn self_test_count(&self) -> u32 {
        self.self_tests.load(Ordering::SeqCst)
    }

    /// Simulate an upgrade of the underlying tool.
    pub fn set_version(&self, version: &str) {
        *lock(&self.version) = version.to_string();
    }

    fn pick_rule(&self, user_text: Option<&str>) -> Option<FakeRule> {
        for (i, rule) in self.options.rules.iter().enumerate() {
            let matches = match (rule.pattern.as_str(), user_text) {
                ("*", _) | ("", _) => true,
                ("tool_results", None) => true,
                (p, Some(t)) => t.contains(p),
                _ => false,
            };
            if !matches {
                continue;
            }
            if let Some(times) = rule.times {
                // Reserve one use atomically; skip the rule once used up.
                let used = self.rule_uses[i].fetch_add(1, Ordering::SeqCst);
                if used >= times {
                    continue;
                }
            }
            return Some(rule.clone());
        }
        None
    }
}

fn fake_error(kind: &str, backend: &str, retry_after: Option<u64>) -> BridgeError {
    let class = match kind {
        "quota" => ErrorClass::QuotaExhausted,
        "unavailable" => ErrorClass::Unavailable,
        "context" => ErrorClass::ContextTooLong,
        "auth" => ErrorClass::BackendAuth,
        "timeout" => ErrorClass::Timeout,
        "rate" => ErrorClass::RateLimited,
        _ => ErrorClass::Internal,
    };
    // Deliberately "leaky" text: the engine must never pass it to the caller.
    let mut e = BridgeError::new(class, format!("fake backend {backend} failed ({kind}) at /secret/fake/path"));
    e.retry_after = retry_after;
    e
}

fn message_chars(m: &Message) -> usize {
    m.parts
        .iter()
        .map(|p| match p {
            Part::Text { text } | Part::Reasoning { text } => text.chars().count(),
            Part::ToolCall(c) => c.name.len() + c.arguments.chars().count(),
            Part::ToolResult(r) => r.content.chars().count(),
            Part::Image(_) => 1000,
        })
        .sum()
}

fn split_deltas(text: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return Vec::new();
    }
    let n = n.clamp(1, chars.len());
    let size = chars.len().div_ceil(n);
    chars.chunks(size).map(|c| c.iter().collect()).collect()
}

fn render_history(history: &[Message]) -> String {
    let users: Vec<String> = history.iter().filter(|m| m.role == Role::User).map(Message::text_content).collect();
    format!(
        "history: {} messages; users: {}; last user: {}",
        history.len(),
        users.join(" | "),
        users.last().cloned().unwrap_or_default()
    )
}

struct TurnState {
    produced: Message,
    done: bool,
}

#[async_trait]
impl Backend for FakeBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities { images: true, json_schema: self.options.json_schema, effort: true, max_context: self.context_limit }
    }

    fn version(&self) -> String {
        lock(&self.version).clone()
    }

    fn mode(&self) -> BackendMode {
        *lock(&self.mode)
    }

    async fn self_test(&self) -> BackendMode {
        self.self_tests.fetch_add(1, Ordering::SeqCst);
        *lock(&self.mode) = self.configured_mode;
        self.configured_mode
    }

    async fn start_turn(&self, spec: TurnSpec) -> Result<Turn, BridgeError> {
        if self.mode() == BackendMode::Disabled {
            return Err(fake_error("auth", &self.name, None));
        }
        let version = self.version();

        // Load the base snapshot.
        let mut history: Vec<Message> = Vec::new();
        let mut record = TurnRecord {
            appended: spec.append.len(),
            model: spec.model.clone(),
            system: spec.system.clone(),
            tools: spec.tools.iter().map(|t| t.name.clone()).collect(),
            effort: spec.effort,
            json_schema: spec.json_schema.is_some(),
            ephemeral: spec.ephemeral,
            ..Default::default()
        };
        if let Some(base) = &spec.base {
            let raw = tokio::fs::read(&base.snapshot.path)
                .await
                .map_err(|_| fake_error("internal", &self.name, None))?;
            let snap: SnapshotFile =
                serde_json::from_slice(&raw).map_err(|_| fake_error("internal", &self.name, None))?;
            record.base_path = Some(base.snapshot.path.clone());
            record.base_kind = Some(base.snapshot.kind);
            record.base_messages = snap.history.len();
            if base.snapshot.kind == CheckpointKind::Awaiting {
                if !crate::planner::compatible(CheckpointKind::Awaiting, &snap.pending_call_ids, &spec.append) {
                    // The engine must never attach anything but the pending results.
                    return Err(fake_error("internal", &self.name, None));
                }
                record.patched = true;
            }
            history = snap.history;
        }
        history.extend(spec.append.iter().cloned());
        let user_text = match &spec.trigger {
            Trigger::User { message } => {
                history.push(message.clone());
                record.trigger = "user".into();
                Some(message.text_content())
            }
            Trigger::ContinueAfterResults => {
                record.trigger = "continue".into();
                None
            }
        };
        record.history_len = history.len();

        let rule = self.pick_rule(user_text.as_deref()).unwrap_or_default();
        let last_user = history
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(Message::text_content)
            .unwrap_or_default();
        let default_text = if self.options.reply.as_deref() == Some("history") {
            render_history(&history)
        } else {
            format!("echo: {last_user}")
        };
        let text = match (&rule.text, rule.tool_calls.is_empty(), &rule.error) {
            (Some(t), _, _) => t.clone(),
            (None, false, _) => String::new(),
            (None, true, _) => default_text,
        };
        record.reply = text.clone();

        let index = {
            let mut recs = lock(&self.records);
            recs.push(record);
            recs.len() - 1
        };

        if rule.error_in_start {
            if let Some(kind) = &rule.error {
                return Err(fake_error(kind, &self.name, rule.retry_after));
            }
        }

        // Build the scripted event list.
        let mut events: Vec<TurnEvent> = Vec::new();
        if let Some(info) = &self.options.limit {
            events.push(TurnEvent::Limit { info: info.clone() });
        }
        if let Some(r) = &rule.reasoning {
            events.push(TurnEvent::ReasoningDelta { text: r.clone() });
        }
        for d in split_deltas(&text, rule.deltas.unwrap_or(3)) {
            events.push(TurnEvent::TextDelta { text: d });
        }
        let calls: Vec<ToolCall> = rule
            .tool_calls
            .iter()
            .map(|c| ToolCall {
                id: format!("call_{}", &uuid::Uuid::new_v4().simple().to_string()[..12]),
                name: c.name.clone(),
                arguments: if c.arguments.is_null() { "{}".into() } else { c.arguments.to_string() },
            })
            .collect();
        if !calls.is_empty() {
            if self.options.stream_tool_args {
                // Previews (17.3): the commit's arguments are byte-identical to the
                // concatenation of each call's deltas.
                for (i, c) in calls.iter().enumerate() {
                    let index = u32::try_from(i).unwrap_or(u32::MAX);
                    events.push(TurnEvent::ToolCallStart { index, id: c.id.clone(), name: c.name.clone() });
                    for arguments in split_deltas(&c.arguments, rule.deltas.unwrap_or(3).max(2)) {
                        events.push(TurnEvent::ToolCallDelta { index, arguments });
                    }
                }
            }
            events.push(TurnEvent::ToolCalls { calls: calls.clone() });
        }
        let prompt_chars: usize = spec.system.chars().count() + history.iter().map(message_chars).sum::<usize>();
        let completion_chars = text.chars().count()
            + calls.iter().map(|c| c.name.len() + c.arguments.len()).sum::<usize>()
            + rule.reasoning.as_ref().map(|r| r.chars().count()).unwrap_or(0);
        // Fixed 4 chars/token: the fake only sees its BackendConfig (frozen `new` signature),
        // not `engine.chars_per_token`; its usage numbers are test fixtures.
        events.push(TurnEvent::Usage {
            usage: Usage {
                prompt_tokens: (prompt_chars / 4) as u64,
                completion_tokens: (completion_chars / 4) as u64,
                cached_tokens: 0,
                reasoning_tokens: 0,
            },
        });
        events.push(TurnEvent::Done { finish: if calls.is_empty() { FinishReason::Stop } else { FinishReason::ToolCalls } });
        if let Some(kind) = &rule.error {
            events.truncate(rule.error_after_events.min(events.len()));
            events.push(TurnEvent::Error { error: fake_error(kind, &self.name, rule.retry_after) });
        }

        let token = CancellationToken::new();
        let state = Arc::new(Mutex::new(TurnState {
            produced: Message { role: Role::Assistant, parts: Vec::new() },
            done: false,
        }));
        let stream_token = token.clone();
        let stream_state = state.clone();
        let delay = Duration::from_millis(rule.delay_ms);
        let delta_delay = Duration::from_millis(rule.delta_delay_ms);
        let stream = async_stream::stream! {
            if !delay.is_zero() {
                tokio::select! {
                    _ = stream_token.cancelled() => return,
                    _ = tokio::time::sleep(delay) => {}
                }
            }
            for (i, ev) in events.into_iter().enumerate() {
                if stream_token.is_cancelled() {
                    return;
                }
                if i > 0 && !delta_delay.is_zero() {
                    tokio::select! {
                        _ = stream_token.cancelled() => return,
                        _ = tokio::time::sleep(delta_delay) => {}
                    }
                }
                {
                    let mut st = lock(&stream_state);
                    match &ev {
                        TurnEvent::TextDelta { text } => match st.produced.parts.first_mut() {
                            Some(Part::Text { text: t }) => t.push_str(text),
                            _ => st.produced.parts.insert(0, Part::Text { text: text.clone() }),
                        },
                        TurnEvent::ToolCalls { calls } => {
                            st.produced.parts.extend(calls.iter().cloned().map(Part::ToolCall));
                        }
                        TurnEvent::Done { .. } => st.done = true,
                        _ => {}
                    }
                }
                yield ev;
            }
        };

        Ok(Turn {
            events: Box::pin(stream),
            control: Box::new(FakeControl {
                name: self.name.clone(),
                version,
                history,
                state,
                token,
                snapshot_dir: self.snapshot_dir.clone(),
                records: self.records.clone(),
                index,
                freeze_delay: Duration::from_millis(self.options.freeze_delay_ms),
                freeze_fail: self.options.freeze_fail,
            }),
        })
    }

    async fn delete_snapshot(&self, snapshot: &SnapshotInfo) {
        let _ = tokio::fs::remove_file(&snapshot.path).await;
        lock(&self.deleted).push(snapshot.path.clone());
    }
}

struct FakeControl {
    name: String,
    version: String,
    history: Vec<Message>,
    state: Arc<Mutex<TurnState>>,
    token: CancellationToken,
    snapshot_dir: PathBuf,
    records: Arc<Mutex<Vec<TurnRecord>>>,
    index: usize,
    freeze_delay: Duration,
    freeze_fail: bool,
}

impl FakeControl {
    fn update(&self, f: impl FnOnce(&mut TurnRecord)) {
        if let Some(r) = lock(&self.records).get_mut(self.index) {
            f(r);
        }
    }
}

#[async_trait]
impl TurnControl for FakeControl {
    async fn abort(&mut self) {
        self.token.cancel();
        self.update(|r| r.aborted = true);
    }

    async fn freeze(self: Box<Self>) -> Result<SnapshotInfo, BridgeError> {
        let (produced, done) = {
            let st = lock(&self.state);
            (st.produced.clone(), st.done)
        };
        if !done {
            return Err(BridgeError::new(ErrorClass::Internal, "freeze before done"));
        }
        if !self.freeze_delay.is_zero() {
            tokio::time::sleep(self.freeze_delay).await;
        }
        if self.freeze_fail {
            return Err(BridgeError::new(ErrorClass::Internal, "scripted freeze failure"));
        }
        let pending: Vec<String> = produced.tool_calls().iter().map(|c| c.id.clone()).collect();
        let kind = if pending.is_empty() { CheckpointKind::Idle } else { CheckpointKind::Awaiting };
        let mut history = self.history.clone();
        history.push(produced);
        let file = SnapshotFile {
            backend: self.name.clone(),
            version: self.version.clone(),
            history,
            pending_call_ids: pending.clone(),
        };
        let bytes = serde_json::to_vec(&file).map_err(|_| BridgeError::new(ErrorClass::Internal, "encode"))?;
        let path: PathBuf = self.snapshot_dir.join(format!("{}.json", uuid::Uuid::new_v4()));
        write_file(&path, &bytes).await?;
        self.update(|r| r.frozen = true);
        Ok(SnapshotInfo {
            path: path.to_string_lossy().into_owned(),
            kind,
            pending_call_ids: pending,
            size_bytes: bytes.len() as u64,
            backend_version: self.version.clone(),
        })
    }

    async fn discard(self: Box<Self>) {
        self.token.cancel();
        self.update(|r| r.discarded = true);
    }
}

async fn write_file(path: &Path, bytes: &[u8]) -> Result<(), BridgeError> {
    tokio::fs::write(path, bytes).await.map_err(|_| BridgeError::new(ErrorClass::Internal, "write snapshot"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    fn backend(options: serde_json::Value, dir: &Path) -> FakeBackend {
        let cfg = BackendConfig { kind: "fake".into(), options, ..Default::default() };
        FakeBackend::new("fake", &cfg, dir).unwrap()
    }

    fn spec(trigger: Trigger) -> TurnSpec {
        TurnSpec {
            model: "m".into(),
            system: "sys".into(),
            tools: vec![],
            base: None,
            append: vec![],
            trigger,
            effort: None,
            json_schema: None,
            ephemeral: false,
        }
    }

    async fn collect(turn: &mut Turn) -> Vec<TurnEvent> {
        let mut out = Vec::new();
        while let Some(ev) = turn.events.next().await {
            out.push(ev);
        }
        out
    }

    #[tokio::test]
    async fn echo_and_freeze_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let b = backend(serde_json::json!({"reply": "history"}), dir.path());
        let mut turn = b
            .start_turn(spec(Trigger::User { message: Message::text(Role::User, "hi") }))
            .await
            .unwrap();
        let evs = collect(&mut turn).await;
        let text: String = evs
            .iter()
            .filter_map(|e| if let TurnEvent::TextDelta { text } = e { Some(text.clone()) } else { None })
            .collect();
        assert_eq!(text, "history: 1 messages; users: hi; last user: hi");
        assert!(matches!(evs.last(), Some(TurnEvent::Done { finish: FinishReason::Stop })));
        let snap = turn.control.freeze().await.unwrap();
        assert_eq!(snap.kind, CheckpointKind::Idle);

        // Resume from the snapshot.
        let mut s = spec(Trigger::User { message: Message::text(Role::User, "again") });
        s.base = Some(bridge_types::CheckpointRef { backend: "fake".into(), snapshot: snap });
        let mut turn = b.start_turn(s).await.unwrap();
        let evs = collect(&mut turn).await;
        let text: String = evs
            .iter()
            .filter_map(|e| if let TurnEvent::TextDelta { text } = e { Some(text.clone()) } else { None })
            .collect();
        assert_eq!(text, "history: 3 messages; users: hi | again; last user: again");
        let recs = b.records();
        assert_eq!(recs[1].base_messages, 2);
        assert!(recs[0].frozen);
    }

    #[tokio::test]
    async fn rules_tool_calls_errors_and_times() {
        let dir = tempfile::tempdir().unwrap();
        let b = backend(
            serde_json::json!({"rules": [
                {"match": "boom", "error": "quota", "error_in_start": true, "retry_after": 5},
                {"match": "late", "error": "unavailable", "error_after_events": 1},
                {"match": "once", "text": "first", "times": 1},
                {"match": "once", "tool_calls": [{"name": "read", "arguments": {"p": 1}}]}
            ]}),
            dir.path(),
        );
        let user = |t: &str| Trigger::User { message: Message::text(Role::User, t) };
        let err = b.start_turn(spec(user("boom"))).await.err().unwrap();
        assert_eq!(err.class, ErrorClass::QuotaExhausted);
        assert_eq!(err.retry_after, Some(5));

        let mut t = b.start_turn(spec(user("late"))).await.unwrap();
        let evs = collect(&mut t).await;
        assert_eq!(evs.len(), 2);
        assert!(matches!(evs[1], TurnEvent::Error { .. }));

        let mut t = b.start_turn(spec(user("once"))).await.unwrap();
        assert!(collect(&mut t).await.iter().any(|e| matches!(e, TurnEvent::TextDelta { .. })));
        let mut t = b.start_turn(spec(user("once"))).await.unwrap();
        let evs = collect(&mut t).await;
        assert!(evs.iter().any(|e| matches!(e, TurnEvent::ToolCalls { calls } if calls[0].id.starts_with("call_"))));
        let snap = t.control.freeze().await.unwrap();
        assert_eq!(snap.kind, CheckpointKind::Awaiting);
        assert_eq!(snap.pending_call_ids.len(), 1);
    }

    #[tokio::test]
    async fn streamed_tool_args_are_byte_identical_to_commit() {
        let dir = tempfile::tempdir().unwrap();
        let b = backend(
            serde_json::json!({"stream_tool_args": true, "rules": [{"match": "*", "deltas": 4, "tool_calls": [
                {"name": "write", "arguments": {"path": "a.txt", "content": "hello world, ünïcode"}},
                {"name": "read", "arguments": {"path": "b"}}
            ]}]}),
            dir.path(),
        );
        let mut t = b.start_turn(spec(Trigger::User { message: Message::text(Role::User, "x") })).await.unwrap();
        let evs = collect(&mut t).await;
        let commit = evs
            .iter()
            .find_map(|e| if let TurnEvent::ToolCalls { calls } = e { Some(calls.clone()) } else { None })
            .unwrap();
        assert_eq!(commit.len(), 2);
        let commit_pos = evs.iter().position(|e| matches!(e, TurnEvent::ToolCalls { .. })).unwrap();
        for (i, call) in commit.iter().enumerate() {
            let starts: Vec<_> = evs[..commit_pos]
                .iter()
                .filter_map(|e| match e {
                    TurnEvent::ToolCallStart { index, id, name } if *index as usize == i => Some((id, name)),
                    _ => None,
                })
                .collect();
            assert_eq!(starts, vec![(&call.id, &call.name)]);
            let deltas: Vec<&str> = evs[..commit_pos]
                .iter()
                .filter_map(|e| match e {
                    TurnEvent::ToolCallDelta { index, arguments } if *index as usize == i => Some(arguments.as_str()),
                    _ => None,
                })
                .collect();
            assert!(deltas.len() >= 2, "several deltas: {deltas:?}");
            assert_eq!(deltas.concat(), call.arguments);
        }
        // Start precedes its deltas; previews of call 0 come before call 1's.
        assert!(matches!(evs[0], TurnEvent::ToolCallStart { index: 0, .. }));
    }

    #[tokio::test]
    async fn freeze_delay_and_failure() {
        let dir = tempfile::tempdir().unwrap();
        let b = backend(serde_json::json!({"freeze_delay_ms": 150}), dir.path());
        let mut t = b.start_turn(spec(Trigger::User { message: Message::text(Role::User, "x") })).await.unwrap();
        collect(&mut t).await;
        let started = std::time::Instant::now();
        t.control.freeze().await.unwrap();
        assert!(started.elapsed() >= Duration::from_millis(150));

        let b = backend(serde_json::json!({"freeze_fail": true}), dir.path());
        let mut t = b.start_turn(spec(Trigger::User { message: Message::text(Role::User, "x") })).await.unwrap();
        collect(&mut t).await;
        assert!(t.control.freeze().await.is_err());
        assert!(!b.records()[0].frozen);
    }

    #[tokio::test]
    async fn abort_stops_stream() {
        let dir = tempfile::tempdir().unwrap();
        let b = backend(serde_json::json!({"rules": [{"match": "*", "delay_ms": 10_000}]}), dir.path());
        let mut t = b.start_turn(spec(Trigger::User { message: Message::text(Role::User, "x") })).await.unwrap();
        t.control.abort().await;
        let evs = tokio::time::timeout(Duration::from_secs(2), collect(&mut t)).await.unwrap();
        assert!(evs.is_empty());
        assert!(b.records()[0].aborted);
    }
}
