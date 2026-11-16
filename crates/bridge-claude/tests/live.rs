//! Live tests against the real `claude` CLI (haiku). Skipped unless BRIDGE_LIVE_CLAUDE=1.
//!
//! The parking MCP server is the test binary `bridge-claude-park` from this crate.
//! Each test uses its own temp workdir and removes its `~/.claude/projects/<slug>` dir.

use std::path::{Path, PathBuf};
use std::time::Duration;

use bridge_claude::selftest::{collect, Collected};
use bridge_claude::session::SessionFile;
use bridge_claude::ClaudeBackend;
use bridge_types::*;
use futures::StreamExt;
use serde_json::json;

fn live() -> bool {
    std::env::var("BRIDGE_LIVE_CLAUDE").map(|v| v == "1").unwrap_or(false)
}

macro_rules! require_live {
    () => {
        if !live() {
            eprintln!("skipped: set BRIDGE_LIVE_CLAUDE=1");
            return;
        }
    };
}

struct Live {
    _dir: tempfile::TempDir,
    backend: ClaudeBackend,
}

impl Live {
    fn new() -> Live {
        let dir = tempfile::tempdir().unwrap();
        let park = vec![PathBuf::from(env!("CARGO_BIN_EXE_bridge-claude-park"))];
        let backend = ClaudeBackend::new("claude", &BackendConfig::default(), dir.path(), park).unwrap();
        Live { _dir: dir, backend }
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        // Session files of this test's cwd only.
        let _ = std::fs::remove_dir_all(&self.backend.shared().projects_dir);
    }
}

const SYSTEM: &str = "You are a coding assistant. The user's project is reachable only through your tools. Be brief.";

fn read_tool(name: &str) -> ToolSpec {
    ToolSpec {
        name: name.into(),
        description: "Read a file in the user's project. This is the only way to see project files.".into(),
        parameters: json!({"type": "object", "properties": {"filePath": {"type": "string"}}, "required": ["filePath"]}),
        strict: false,
    }
}

fn spec(system: &str, tools: Vec<ToolSpec>, base: Option<SnapshotInfo>, append: Vec<Message>, trigger: Trigger) -> TurnSpec {
    TurnSpec {
        model: "haiku".into(),
        system: system.into(),
        tools,
        base: base.map(|s| CheckpointRef { backend: "claude".into(), snapshot: s }),
        append,
        trigger,
        effort: None,
        json_schema: None,
        ephemeral: false,
    }
}

fn user(t: &str) -> Trigger {
    Trigger::User { message: Message::text(Role::User, t) }
}

fn results(calls: &[ToolCall], f: impl Fn(&ToolCall) -> String) -> Vec<Message> {
    calls
        .iter()
        .map(|c| Message { role: Role::Tool, parts: vec![Part::ToolResult(ToolResult { call_id: c.id.clone(), content: f(c) })] })
        .collect()
}

async fn run(b: &ClaudeBackend, s: TurnSpec) -> (Collected, Box<dyn TurnControl>) {
    let turn = b.start_turn(s).await.expect("start_turn");
    let out = collect(turn.events).await;
    eprintln!("text={:?} calls={:?} finish={:?} error={:?}", out.text, out.calls, out.finish, out.error);
    (out, turn.control)
}

fn file_bytes(p: &str) -> Vec<u8> {
    std::fs::read(p).unwrap()
}

#[tokio::test]
async fn live_tool_loop_forks_and_follow_up() {
    require_live!();
    let l = Live::new();
    let b = &l.backend;
    let tools = vec![read_tool("read")];
    // 1. tool call, stop, freeze AWAITING.
    let (out, ctl) = run(
        b,
        spec(SYSTEM, tools.clone(), None, vec![], user("Read src/a.ts and src/b.ts (both at once, in one message), then tell me what they export.")),
    )
    .await;
    assert_eq!(out.finish, Some(FinishReason::ToolCalls));
    assert!(!out.calls.is_empty());
    assert!(out.calls.iter().all(|c| c.name == "read"));
    let snap = ctl.freeze().await.unwrap();
    assert_eq!(snap.kind, CheckpointKind::Awaiting);
    let f = SessionFile::load(Path::new(&snap.path)).unwrap();
    for c in &out.calls {
        assert!(f.tool_result_for(&c.id).is_some(), "placeholder for {}", c.id);
    }
    let before = file_bytes(&snap.path);

    // 2. two forks of the same checkpoint with different results.
    let mk = |word: &'static str| {
        spec(
            SYSTEM,
            tools.clone(),
            Some(snap.clone()),
            results(&out.calls, move |c| format!("// {}\nexport const {word}_{} = 1;\n", c.arguments, c.id.len() % 7)),
            Trigger::ContinueAfterResults,
        )
    };
    let (a, ctl_a) = run(b, mk("ALPHA")).await;
    let (o, ctl_o) = run(b, mk("OMEGA")).await;
    assert!(a.text.contains("ALPHA") && !a.text.contains("OMEGA"), "{}", a.text);
    assert!(o.text.contains("OMEGA") && !o.text.contains("ALPHA"), "{}", o.text);
    ctl_o.discard().await;
    let snap_a = ctl_a.freeze().await.unwrap();
    assert_eq!(snap_a.kind, CheckpointKind::Idle);
    assert_eq!(file_bytes(&snap.path), before, "base checkpoint untouched");
    let fa = SessionFile::load(Path::new(&snap_a.path)).unwrap();
    assert_eq!(fa.count_synthetic(), 0);
    assert!(!fa.has_interrupted_marker());
    let conv = fa.conversation(&Default::default(), &Default::default(), &Default::default());
    assert!(!conv.iter().any(|m| m.role == Role::User && m.text_content().trim() == bridge_claude::session::TRIGGER_TEXT.trim()));

    // 3. follow-up user message on the IDLE checkpoint, with effort.
    let mut s = spec(
        SYSTEM,
        tools.clone(),
        Some(snap_a.clone()),
        vec![],
        user("Without reading anything again: which exported constant names did you see? Reply with just the names."),
    );
    s.effort = Some(ReasoningEffort::Low);
    let (f3, ctl3) = run(b, s).await;
    assert_eq!(f3.finish, Some(FinishReason::Stop));
    assert!(f3.text.contains("ALPHA"), "{}", f3.text);
    let snap3 = ctl3.freeze().await.unwrap();
    let f3file = SessionFile::load(Path::new(&snap3.path)).unwrap();
    assert_eq!(f3file.count_synthetic(), 0);
    for s in [&snap, &snap_a, &snap3] {
        b.delete_snapshot(s).await;
    }
}

#[tokio::test]
async fn live_history_from_scratch_with_codecs() {
    require_live!();
    let l = Live::new();
    let history = vec![
        Message::text(Role::System, "You are a helpful assistant. Be brief."),
        Message::text(Role::User, "My codename is PELICAN. Read notes.txt please."),
        Message {
            role: Role::Assistant,
            parts: vec![
                Part::Text { text: "Reading it.".into() },
                Part::ToolCall(ToolCall { id: "call:abc/1".into(), name: "files.read".into(), arguments: r#"{"filePath":"notes.txt"}"#.into() }),
            ],
        },
        Message { role: Role::Tool, parts: vec![Part::ToolResult(ToolResult { call_id: "call:abc/1".into(), content: "Meeting moved to Tuesday 3pm in room B7.".into() })] },
        Message::text(Role::Assistant, "Noted: the meeting is on Tuesday at 3pm in room B7."),
    ];
    let (out, ctl) = run(
        &l.backend,
        spec("You are a helpful assistant. Be brief.", vec![read_tool("files.read")], None, history, user("What is my codename and which room is the meeting in? One line.")),
    )
    .await;
    assert_eq!(out.finish, Some(FinishReason::Stop));
    assert!(out.text.contains("PELICAN") && out.text.contains("B7"), "{}", out.text);
    let snap = ctl.freeze().await.unwrap();
    let f = SessionFile::load(Path::new(&snap.path)).unwrap();
    assert_eq!(f.count_synthetic(), 0, "file ended with an assistant entry: no filler");
    l.backend.delete_snapshot(&snap).await;
}

#[tokio::test]
async fn live_transcript_mode() {
    require_live!();
    let l = Live::new();
    l.backend.set_mode(BackendMode::Transcript);
    let history = vec![
        Message::text(Role::User, "Hi! Remember that my favourite fruit is the DURIAN."),
        Message::text(Role::Assistant, "Got it, your favourite fruit is the durian."),
    ];
    let (out, ctl) = run(&l.backend, spec("Be brief.", vec![read_tool("read")], None, history, user("What is my favourite fruit? One word."))).await;
    assert!(out.text.to_uppercase().contains("DURIAN"), "{}", out.text);
    assert!(ctl.freeze().await.is_err());
}

#[tokio::test]
async fn live_self_test_and_cache() {
    require_live!();
    let l = Live::new();
    let mode = l.backend.self_test().await;
    assert_eq!(mode, BackendMode::Native);
    let cache = l.backend.shared().selftest_path();
    assert!(cache.exists());
    let t = std::time::Instant::now();
    assert_eq!(l.backend.self_test().await, BackendMode::Native);
    assert!(t.elapsed() < Duration::from_secs(3), "second call served from cache");
    assert_eq!(std::fs::read_dir(&l.backend.shared().snapshots_dir).unwrap().count(), 0, "canary snapshots removed");
}

#[tokio::test]
async fn live_json_schema_sanitizer_and_image() {
    require_live!();
    let l = Live::new();
    let b = &l.backend;
    // JSON schema.
    let mut s = spec("Be brief.", vec![], None, vec![], user("Describe the color red."));
    s.json_schema = Some(json!({"type": "object", "properties": {"name": {"type": "string"}, "hex": {"type": "string"}}, "required": ["name", "hex"], "additionalProperties": false}));
    s.ephemeral = true;
    let (out, ctl) = run(b, s).await;
    let v: serde_json::Value = serde_json::from_str(&out.text).expect("json text");
    assert!(v.get("hex").is_some());
    ctl.discard().await;

    // Sanitizer: tool names in prose never carry the internal prefix.
    let tools = vec![read_tool("todo.write"), read_tool("read")];
    let (out, ctl) = run(
        b,
        spec("Answer questions about yourself precisely.", tools, None, vec![], user("List the exact, full internal names of every tool you can call, one per line, verbatim as they appear in your tool list. Do not call any tool.")),
    )
    .await;
    assert!(!out.text.contains("mcp__"), "{}", out.text);
    ctl.discard().await;

    // Image in written history (16x16 red PNG).
    let png = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAF0lEQVR4nGP4z8BAEiJN9aiGUQ1DSgMAkPn/Afnh+ngAAAAASUVORK5CYII=";
    let history = vec![
        Message {
            role: Role::User,
            parts: vec![
                Part::Text { text: "Here is an image.".into() },
                Part::Image(Image { media_type: "image/png".into(), source: ImageSource::Base64 { data: png.into() } }),
            ],
        },
        Message::text(Role::Assistant, "I see the image."),
    ];
    let (out, ctl) = run(b, spec("Be brief.", vec![], None, history, user("What single color fills the image I sent? One word."))).await;
    assert!(out.text.to_lowercase().contains("red"), "{}", out.text);
    ctl.discard().await;
}

#[tokio::test]
async fn live_abort_stops_claude() {
    require_live!();
    let l = Live::new();
    let (mut events, mut ctl) = l
        .backend
        .start_turn_concrete(spec("Be verbose.", vec![read_tool("read")], None, vec![], user("Count from 1 to 300, one number per line.")))
        .await
        .unwrap();
    let sid = ctl.session_id().to_string();
    loop {
        match events.next().await {
            Some(TurnEvent::TextDelta { .. }) => break,
            Some(TurnEvent::Done { .. }) | Some(TurnEvent::Error { .. }) | None => panic!("ended before text"),
            _ => {}
        }
    }
    use bridge_types::TurnControl as _;
    ctl.abort().await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let ps = std::process::Command::new("pgrep").args(["-f", &sid]).output().unwrap();
    assert!(String::from_utf8_lossy(&ps.stdout).trim().is_empty(), "claude still running");
    let park = std::process::Command::new("pgrep").args(["-f", &format!("turns/{sid}/tools.json")]).output().unwrap();
    assert!(String::from_utf8_lossy(&park.stdout).trim().is_empty(), "park server still running");
    Box::new(ctl).discard().await;
    assert!(!l.backend.shared().session_path(&sid).exists());
}
