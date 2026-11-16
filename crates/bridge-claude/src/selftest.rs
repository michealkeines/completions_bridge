//! Self-test (ARCHITECTURE 8.5): a three-turn canary on the cheapest model that verifies
//! the session-file tricks for the installed claude version. Cached per version.

use std::sync::Arc;

use bridge_types::{
    BackendMode, CheckpointRef, ErrorClass, FinishReason, Message, Part, Role, SnapshotInfo, ToolCall, ToolResult,
    ToolSpec, Trigger, TurnEvent, TurnSpec,
};
use futures::stream::{BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::session::SessionFile;
use crate::turn::{self, RunMode};
use crate::Shared;

/// Built-in self-test model when neither `selftest_model` nor `side_request_model` is set.
pub const CANARY_MODEL: &str = "claude-haiku-5-5";

/// Self-test model: `selftest_model`, else `side_request_model`, else [`CANARY_MODEL`].
fn canary_model(shared: &Shared) -> String {
    let c = &shared.config;
    c.selftest_model
        .clone()
        .or_else(|| c.side_request_model.clone())
        .filter(|m| !m.trim().is_empty())
        .unwrap_or_else(|| CANARY_MODEL.to_string())
}
// Canary values the self-test plants and checks for. Not configurable: they are test
// fixtures (any distinctive string works), not behaviour an operator tunes.
const CODE: &str = "PURPLE-HIPPO-42";
const WORD: &str = "MARMALADE";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SelfTestRecord {
    pub version: String,
    pub mode: BackendMode,
    pub detail: String,
    pub at: String,
}

/// Collected output of one turn.
#[derive(Debug, Default, Clone)]
pub struct Collected {
    pub text: String,
    pub reasoning: String,
    pub calls: Vec<ToolCall>,
    pub finish: Option<FinishReason>,
    pub error: Option<bridge_types::BridgeError>,
    pub events: Vec<TurnEvent>,
}

pub async fn collect(mut events: BoxStream<'static, TurnEvent>) -> Collected {
    let mut c = Collected::default();
    while let Some(ev) = events.next().await {
        match &ev {
            TurnEvent::TextDelta { text } => c.text.push_str(text),
            TurnEvent::ReasoningDelta { text } => c.reasoning.push_str(text),
            TurnEvent::ToolCalls { calls } => c.calls.extend(calls.iter().cloned()),
            TurnEvent::Done { finish } => c.finish = Some(*finish),
            TurnEvent::Error { error } => c.error = Some(error.clone()),
            _ => {}
        }
        let terminal = matches!(ev, TurnEvent::Done { .. } | TurnEvent::Error { .. });
        c.events.push(ev);
        if terminal {
            break;
        }
    }
    c
}

pub fn load_cache(shared: &Shared) -> Option<SelfTestRecord> {
    let text = std::fs::read_to_string(shared.selftest_path()).ok()?;
    serde_json::from_str(&text).ok()
}

fn save_cache(shared: &Shared, rec: &SelfTestRecord) {
    if let Ok(s) = serde_json::to_string_pretty(rec) {
        let _ = std::fs::write(shared.selftest_path(), s);
    }
}

enum Verdict {
    /// Conclusive result; cached.
    Mode(BackendMode, String),
    /// Not conclusive (quota, network): keep the current mode, don't cache.
    Inconclusive(String),
}

fn canary_tool() -> ToolSpec {
    ToolSpec {
        name: "lookup_code".into(),
        description: "Look up the secret code stored under a key.".into(),
        parameters: json!({"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}),
        strict: false,
    }
}

fn spec(model: &str, system: &str, tools: Vec<ToolSpec>, base: Option<CheckpointRef>, append: Vec<Message>, trigger: Trigger) -> TurnSpec {
    TurnSpec {
        model: model.into(),
        system: system.into(),
        tools,
        base,
        append,
        trigger,
        effort: None,
        json_schema: None,
        ephemeral: false,
    }
}

fn on_error(e: &bridge_types::BridgeError, step: &str) -> Verdict {
    if e.class == ErrorClass::BackendAuth {
        Verdict::Mode(BackendMode::Disabled, format!("{step}: not signed in"))
    } else if e.is_failover_candidate() {
        Verdict::Inconclusive(format!("{step}: {:?}", e.class))
    } else {
        Verdict::Mode(BackendMode::Transcript, format!("{step}: error {:?}", e.class))
    }
}

fn normalized(s: &str) -> String {
    s.to_uppercase().chars().filter(|c| c.is_ascii_alphanumeric()).collect()
}

async fn canary(shared: &Arc<Shared>) -> Verdict {
    let mut snapshots: Vec<SnapshotInfo> = Vec::new();
    let v = canary_steps(shared, &mut snapshots).await;
    for s in snapshots {
        let _ = std::fs::remove_file(&s.path);
    }
    v
}

async fn canary_steps(shared: &Arc<Shared>, snapshots: &mut Vec<SnapshotInfo>) -> Verdict {
    // (a) new session, tool call, stop, placeholder present.
    let a = spec(
        &canary_model(shared),
        "You are a test harness. Follow the user's instruction by calling the tool. Do not write any text before the tool call.",
        vec![canary_tool()],
        None,
        vec![],
        Trigger::User { message: Message::text(Role::User, "Call lookup_code with key \"alpha\" now.") },
    );
    let (events, mut ctl) = match turn::start(shared.clone(), a, RunMode::Native).await {
        Ok(x) => x,
        Err(e) => return on_error(&e, "step 1"),
    };
    let out = collect(events).await;
    if let Some(e) = &out.error {
        ctl.discard_inner().await;
        return on_error(e, "step 1");
    }
    if out.finish != Some(FinishReason::ToolCalls) || out.calls.is_empty() {
        ctl.discard_inner().await;
        return Verdict::Inconclusive("step 1: model did not call the tool".into());
    }
    let snap = match ctl.freeze_inner().await {
        Ok(s) => s,
        Err(_) => return Verdict::Mode(BackendMode::Transcript, "step 1: tool-call stop could not be frozen".into()),
    };
    snapshots.push(snap.clone());
    let ok = SessionFile::load(std::path::Path::new(&snap.path))
        .map(|f| out.calls.iter().all(|c| f.tool_result_for(&c.id).is_some()))
        .unwrap_or(false);
    if !ok {
        return Verdict::Mode(BackendMode::Transcript, "step 1: placeholder missing".into());
    }

    // (b) fork + patch + resume: the answer uses the patched result, no "interrupted".
    let results: Vec<Message> = out
        .calls
        .iter()
        .map(|c| Message {
            role: Role::Tool,
            parts: vec![Part::ToolResult(ToolResult { call_id: c.id.clone(), content: format!("The secret code is {CODE}.") })],
        })
        .collect();
    let b = spec(
        &canary_model(shared),
        "You are a test harness. Reply with the secret code from the tool result, exactly as written, and nothing else.",
        vec![canary_tool()],
        Some(CheckpointRef { backend: shared.name.clone(), snapshot: snap.clone() }),
        results,
        Trigger::ContinueAfterResults,
    );
    let (events, mut ctl) = match turn::start(shared.clone(), b, RunMode::Native).await {
        Ok(x) => x,
        Err(e) => return on_error(&e, "step 2"),
    };
    let out_b = collect(events).await;
    if let Some(e) = &out_b.error {
        ctl.discard_inner().await;
        return on_error(e, "step 2");
    }
    let outcome = ctl.wait().await.unwrap_or_default();
    let interrupted = ctl
        .session_path()
        .and_then(|p| SessionFile::load(p).ok())
        .map(|f| f.has_interrupted_marker())
        .unwrap_or(true);
    if !normalized(&out_b.text).contains(&normalized(CODE)) || interrupted || !outcome.verified {
        ctl.discard_inner().await;
        return Verdict::Mode(
            BackendMode::Transcript,
            format!("step 2: patched result not used (interrupted={interrupted}, verified={})", outcome.verified),
        );
    }
    match ctl.freeze_inner().await {
        Ok(s) => {
            snapshots.push(s.clone());
            let clean = SessionFile::load(std::path::Path::new(&s.path)).map(|f| f.count_synthetic() == 0).unwrap_or(false);
            if !clean {
                return Verdict::Mode(BackendMode::Transcript, "step 2: filler not stripped".into());
            }
        }
        Err(_) => return Verdict::Mode(BackendMode::Transcript, "step 2: freeze failed".into()),
    }

    // (c) history written from scratch + user trigger.
    let c = spec(
        &canary_model(shared),
        "You are a test harness. Answer briefly.",
        vec![],
        None,
        vec![
            Message::text(Role::User, format!("Remember this: the secret word is {WORD}.")),
            Message::text(Role::Assistant, format!("Got it. The secret word is {WORD}.")),
        ],
        Trigger::User { message: Message::text(Role::User, "What is the secret word? Reply with just the word.") },
    );
    let (events, mut ctl) = match turn::start(shared.clone(), c, RunMode::Native).await {
        Ok(x) => x,
        Err(e) => return on_error(&e, "step 3"),
    };
    let out_c = collect(events).await;
    ctl.discard_inner().await;
    if let Some(e) = &out_c.error {
        return on_error(e, "step 3");
    }
    if !normalized(&out_c.text).contains(WORD) {
        return Verdict::Mode(BackendMode::Transcript, "step 3: history written from scratch not used".into());
    }
    Verdict::Mode(BackendMode::Native, "all steps passed".into())
}

/// Runs (or reuses) the self-test and returns the resulting mode.
pub async fn run(shared: Arc<Shared>) -> BackendMode {
    let version = match crate::probe_version(&shared.binary) {
        Some(v) => v,
        None => {
            shared.set_mode(BackendMode::Disabled);
            return BackendMode::Disabled;
        }
    };
    if !shared.is_stale() {
        if let Some(rec) = load_cache(&shared) {
            if rec.version == version && rec.mode != BackendMode::Disabled {
                shared.set_mode(rec.mode);
                return rec.mode;
            }
        }
    }
    match canary(&shared).await {
        Verdict::Mode(mode, detail) => {
            tracing::info!(target: "bridge_claude", "self-test: {mode:?} ({detail})");
            if mode != BackendMode::Disabled {
                save_cache(
                    &shared,
                    &SelfTestRecord { version, mode, detail, at: crate::session::now_timestamp() },
                );
            }
            shared.clear_stale();
            shared.set_mode(mode);
            mode
        }
        Verdict::Inconclusive(detail) => {
            tracing::warn!(target: "bridge_claude", "self-test inconclusive: {detail}");
            let m = shared.mode();
            if m == BackendMode::Disabled {
                // Reachable but not testable right now: allow the safe mode.
                shared.set_mode(BackendMode::Transcript);
                return BackendMode::Transcript;
            }
            m
        }
    }
}
