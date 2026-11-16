//! Process tests against a fake claude (`tests/fixtures/fake_claude.py`): spawning,
//! stop-at-tool-call, placeholder fix-up, freeze/fork, abort, drop, transcript mode.

use std::path::{Path, PathBuf};
use std::time::Duration;

use bridge_claude::selftest::{collect, SelfTestRecord};
use bridge_claude::session::SessionFile;
use bridge_claude::ClaudeBackend;
use bridge_types::*;
use futures::StreamExt;
use serde_json::{json, Value};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

struct Env {
    _dir: tempfile::TempDir,
    work: PathBuf,
    projects: PathBuf,
    pidfile: PathBuf,
}

impl Env {
    fn new() -> Env {
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path().join("work");
        let projects = dir.path().join("projects");
        std::fs::create_dir_all(&projects).unwrap();
        let pidfile = dir.path().join("pids");
        Env { _dir: dir, work, projects, pidfile }
    }

    fn backend(&self, fixture: &str, mode: &str) -> ClaudeBackend {
        self.backend_with(fixture, mode, Value::Null)
    }

    fn backend_with(&self, fixture: &str, mode: &str, options: Value) -> ClaudeBackend {
        self.backend_cfg(fixture, mode, BackendConfig { options, ..BackendConfig::default() })
    }

    /// Backend over the fake claude; `base` supplies everything but binary/extra_args.
    fn backend_cfg(&self, fixture: &str, mode: &str, base: BackendConfig) -> ClaudeBackend {
        let cfg = BackendConfig {
            binary: Some(fixtures().join("fake_claude.py").to_string_lossy().into_owned()),
            extra_args: vec![
            "--fixture".into(),
            fixtures().join(fixture).to_string_lossy().into_owned(),
            "--fake-projects".into(),
            self.projects.to_string_lossy().into_owned(),
            "--fake-mode".into(),
            mode.into(),
            "--pidfile".into(),
            self.pidfile.to_string_lossy().into_owned(),
            ],
            ..base
        };
        let park = vec![PathBuf::from(env!("CARGO_BIN_EXE_bridge-claude-park"))];
        ClaudeBackend::with_projects_root("claude", &cfg, &self.work, park, self.projects.clone()).unwrap()
    }

    fn pids(&self) -> Vec<i32> {
        std::fs::read_to_string(&self.pidfile)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| l.trim().parse().ok())
            .collect()
    }

    fn stdin_lines(&self) -> Vec<Value> {
        std::fs::read_to_string(self.pidfile.with_extension("stdin"))
            .unwrap_or_default()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    fn argv(&self) -> Vec<String> {
        serde_json::from_str(&std::fs::read_to_string(self.pidfile.with_extension("argv")).unwrap()).unwrap()
    }

    fn session_files(&self) -> Vec<PathBuf> {
        let mut out = Vec::new();
        for d in std::fs::read_dir(&self.projects).unwrap().flatten() {
            for f in std::fs::read_dir(d.path()).unwrap().flatten() {
                out.push(f.path());
            }
        }
        out
    }
}

fn alive(pid: i32) -> bool {
    unsafe { libc::kill(pid, 0) == 0 }
}

async fn assert_all_dead(pids: &[i32]) {
    assert!(!pids.is_empty());
    for _ in 0..100 {
        if pids.iter().all(|&p| !alive(p)) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("processes still alive: {:?}", pids.iter().filter(|&&p| alive(p)).collect::<Vec<_>>());
}

fn read_tool() -> ToolSpec {
    ToolSpec {
        name: "read".into(),
        description: "Read a file.".into(),
        parameters: json!({"type": "object", "properties": {"filePath": {"type": "string"}}}),
        strict: false,
    }
}

fn spec(trigger: Trigger) -> TurnSpec {
    TurnSpec {
        model: "haiku".into(),
        system: "sys".into(),
        tools: vec![read_tool()],
        base: None,
        append: vec![],
        trigger,
        effort: None,
        json_schema: None,
        ephemeral: false,
    }
}

fn user(text: &str) -> Trigger {
    Trigger::User { message: Message::text(Role::User, text) }
}

#[tokio::test]
async fn text_turn_streams_and_freezes_idle() {
    let env = Env::new();
    let b = env.backend("stream_text.jsonl", "replay");
    assert_eq!(b.version(), "2.1.325");
    assert_eq!(b.mode(), BackendMode::Native);
    let mut s = spec(user("hi"));
    s.effort = Some(ReasoningEffort::Minimal);
    let turn = b.start_turn(s).await.unwrap();
    let out = collect(turn.events).await;
    assert_eq!(out.text, "Hello there, friend, welcome!");
    assert_eq!(out.finish, Some(FinishReason::Stop));
    let usage = out.events.iter().find_map(|e| if let TurnEvent::Usage { usage } = e { Some(*usage) } else { None }).unwrap();
    assert_eq!(usage, Usage { prompt_tokens: 1042, completion_tokens: 14, cached_tokens: 0, reasoning_tokens: 0 });
    assert!(out.events.iter().any(|e| matches!(e, TurnEvent::Limit { .. })));
    // Ordering: Usage right before Done.
    let n = out.events.len();
    assert!(matches!(out.events[n - 2], TurnEvent::Usage { .. }));

    let argv = env.argv();
    let pos = |f: &str| argv.iter().position(|a| a == f).unwrap_or_else(|| panic!("missing {f}"));
    assert_eq!(argv[pos("--model") + 1], "haiku");
    assert_eq!(argv[pos("--tools") + 1], "");
    assert_eq!(argv[pos("--allowedTools") + 1], "mcp__bridge__read");
    assert_eq!(argv[pos("--effort") + 1], "low");
    assert_eq!(argv[pos("--autocompact") + 1], "1000000");
    assert_eq!(argv[pos("--setting-sources") + 1], "");
    assert!(argv.contains(&"--session-id".to_string()), "new conversation uses --session-id");
    assert!(!argv.contains(&"--no-session-persistence".to_string()));
    let tools: Value = serde_json::from_str(&std::fs::read_to_string(env.pidfile.with_extension("tools")).unwrap()).unwrap();
    assert_eq!(tools[0]["name"], "read");
    assert_eq!(env.stdin_lines()[0], json!({"type": "user", "message": {"role": "user", "content": "hi"}}));

    let snap = turn.control.freeze().await.unwrap();
    assert_eq!(snap.kind, CheckpointKind::Idle);
    assert!(snap.pending_call_ids.is_empty());
    assert_eq!(snap.backend_version, "2.1.325");
    assert!(snap.size_bytes > 0);
    let f = SessionFile::load(Path::new(&snap.path)).unwrap();
    let conv = f.conversation(&Default::default(), &Default::default(), &Default::default());
    assert_eq!(conv.len(), 2);
    assert_eq!(conv[1].text_content(), "Hello there, friend, welcome!");
    assert!(env.session_files().is_empty(), "working file moved away");
    assert_all_dead(&env.pids()).await;
    assert_eq!(std::fs::read_dir(env.work.join("claude/turns")).unwrap().count(), 0);
    b.delete_snapshot(&snap).await;
    assert!(!Path::new(&snap.path).exists());
}

/// Streamed call events from a turn: (index, id, name, concatenated deltas).
fn streamed_calls(events: &[TurnEvent]) -> Vec<(u32, String, String, String)> {
    let mut out: Vec<(u32, String, String, String)> = Vec::new();
    for e in events {
        match e {
            TurnEvent::ToolCallStart { index, id, name } => out.push((*index, id.clone(), name.clone(), String::new())),
            TurnEvent::ToolCallDelta { index, arguments } => {
                out.iter_mut().find(|c| c.0 == *index).expect("delta after start").3.push_str(arguments)
            }
            _ => {}
        }
    }
    out
}

#[tokio::test]
async fn streamed_tool_calls_match_commit() {
    let env = Env::new();
    let b = env.backend("stream_tool_calls_streamed.jsonl", "replay");
    let mut s = spec(user("three calls"));
    s.tools = ["read", "list_todos", "write"]
        .iter()
        .map(|n| ToolSpec { name: (*n).into(), description: String::new(), parameters: json!({"type": "object"}), strict: false })
        .collect();
    let turn = b.start_turn(s).await.unwrap();
    let out = collect(turn.events).await;
    assert_eq!(out.finish, Some(FinishReason::ToolCalls));
    let streamed = streamed_calls(&out.events);
    assert_eq!(streamed.len(), 3);
    assert_eq!(out.calls.len(), 3);
    for (k, (index, id, name, args)) in streamed.iter().enumerate() {
        assert_eq!(*index as usize, k);
        assert_eq!(&out.calls[k].id, id);
        assert_eq!(&out.calls[k].name, name, "caller name, no mcp prefix");
        assert_eq!(&out.calls[k].arguments, args, "byte-identical commit");
    }
    assert_eq!(out.calls[0].name, "list_todos");
    assert_eq!(out.calls[0].arguments, "{}");
    assert_eq!(out.calls[1].arguments, r#"{"filePath": "src/main.rs"}"#);
    // Order: every Start/Delta before the commit.
    let commit = out.events.iter().position(|e| matches!(e, TurnEvent::ToolCalls { .. })).unwrap();
    let last_preview = out.events.iter().rposition(|e| matches!(e, TurnEvent::ToolCallStart { .. } | TurnEvent::ToolCallDelta { .. })).unwrap();
    assert!(last_preview < commit);
    turn.control.discard().await;
    assert_all_dead(&env.pids()).await;
}

#[tokio::test]
async fn tool_call_stop_fixup_patch_and_continue() {
    let env = Env::new();
    let b = env.backend("stream_tool_call.jsonl", "replay");
    let turn = b.start_turn(spec(user("read both"))).await.unwrap();
    let started = std::time::Instant::now();
    let out = collect(turn.events).await;
    assert!(started.elapsed() < Duration::from_secs(5));
    assert_eq!(out.finish, Some(FinishReason::ToolCalls));
    assert_eq!(out.calls.len(), 2);
    assert_eq!(out.calls[0].name, "read");
    assert_eq!(out.calls[0].id, "toolu_01PkKJCGD5vZ1xGpPhdE7HRY");
    assert_eq!(serde_json::from_str::<Value>(&out.calls[1].arguments).unwrap(), json!({"filePath": "src/b.ts"}));
    let tc = out.events.iter().position(|e| matches!(e, TurnEvent::ToolCalls { .. })).unwrap();
    assert!(matches!(out.events[tc + 1], TurnEvent::Usage { .. }));
    let snap = turn.control.freeze().await.unwrap();
    assert_all_dead(&env.pids()).await;
    assert_eq!(snap.kind, CheckpointKind::Awaiting);
    assert_eq!(snap.pending_call_ids, vec!["toolu_01PkKJCGD5vZ1xGpPhdE7HRY", "toolu_014RKrfC8uJtgYANq32i7Tjj"]);
    let f = SessionFile::load(Path::new(&snap.path)).unwrap();
    let counts = f.tool_result_counts();
    assert_eq!(counts.len(), 2, "fake wrote one placeholder; the fix-up added the other");
    assert!(counts.values().all(|&c| c == 1));

    // Streamed previews were emitted and agree with the commit (byte-identical arguments).
    let streamed = streamed_calls(&out.events);
    assert_eq!(streamed.iter().map(|c| c.3.clone()).collect::<Vec<_>>(), out.calls.iter().map(|c| c.arguments.clone()).collect::<Vec<_>>());
    assert_eq!(out.calls[0].arguments, r#"{"filePath": "src/a.ts"}"#);

    // Next turn: results patch the placeholders; glue filler + trigger; glue kept (17.2).
    let b2 = env.backend("stream_text.jsonl", "replay");
    let mut s = spec(Trigger::ContinueAfterResults);
    s.base = Some(CheckpointRef { backend: "claude".into(), snapshot: snap.clone() });
    s.append = out
        .calls
        .iter()
        .map(|c| Message { role: Role::Tool, parts: vec![Part::ToolResult(ToolResult { call_id: c.id.clone(), content: format!("content of {}", c.id) })] })
        .collect();
    let turn = b2.start_turn(s).await.unwrap();
    let out2 = collect(turn.events).await;
    assert!(env.argv().contains(&"--resume".to_string()));
    assert_eq!(out2.finish, Some(FinishReason::Stop));
    assert_eq!(env.stdin_lines().last().unwrap()["message"]["content"], bridge_claude::session::TRIGGER_TEXT);
    let snap2 = turn.control.freeze().await.unwrap();
    assert_eq!(snap2.kind, CheckpointKind::Idle);
    let f2 = SessionFile::load(Path::new(&snap2.path)).unwrap();
    assert_eq!(f2.count_synthetic(), 0);
    assert_eq!(f2.count_glue(), 2, "our filler and the trigger entry are kept");
    let trig = f2.entries().find(|e| e["type"] == "user" && e["message"]["content"] == bridge_claude::session::TRIGGER_TEXT).unwrap();
    assert_eq!(trig[bridge_claude::session::GLUE_KEY], true);
    let conv = f2.conversation(&Default::default(), &Default::default(), &Default::default());
    assert!(!conv.iter().any(|m| m.role == Role::User && m.text_content() == bridge_claude::session::TRIGGER_TEXT), "{conv:?}");
    for c in &out.calls {
        assert_eq!(f2.tool_result_for(&c.id), Some((format!("content of {}", c.id), false)));
    }
    // The base snapshot is untouched (immutable checkpoint).
    let base_again = SessionFile::load(Path::new(&snap.path)).unwrap();
    assert_eq!(base_again, f);
    // Every entry of the fork carries the fork's session id.
    let sids: std::collections::HashSet<_> = f2.entries().filter_map(|e| e.get("sessionId")).cloned().collect();
    assert_eq!(sids.len(), 1);
}

/// Custom prompts and tuning of one backend reach claude: system prompt file, stdin
/// trigger, glue filler in the session file, `--autocompact`; the glue is marked and kept.
#[tokio::test]
async fn configured_prompts_and_tuning_are_used() {
    let env = Env::new();
    let base = BackendConfig {
        prompts: BackendPrompts {
            default_system: Some("You are Parsley.".into()),
            runtime_note: Some("RUNTIME-NOTE-42".into()),
            continue_trigger: Some("[runtime] continue after the tool results".into()),
            glue_filler: Some("Reading the results now.".into()),
            ..BackendPrompts::default()
        },
        tuning: BackendTuning { autocompact_tokens: 640_000, ..BackendTuning::default() },
        ..BackendConfig::default()
    };
    let b = env.backend_cfg("stream_tool_call.jsonl", "replay", base.clone());
    let mut s = spec(user("read both"));
    s.system = String::new();
    let turn = b.start_turn(s).await.unwrap();
    let out = collect(turn.events).await;
    let snap = turn.control.freeze().await.unwrap();
    let argv = env.argv();
    let pos = argv.iter().position(|a| a == "--autocompact").unwrap();
    assert_eq!(argv[pos + 1], "640000");
    assert_eq!(std::fs::read_to_string(env.pidfile.with_extension("system")).unwrap(), "You are Parsley.\n\nRUNTIME-NOTE-42");

    let b2 = env.backend_cfg("stream_text.jsonl", "replay", base);
    let mut s = spec(Trigger::ContinueAfterResults);
    s.system = "caller system".into();
    s.base = Some(CheckpointRef { backend: "claude".into(), snapshot: snap });
    s.append = out
        .calls
        .iter()
        .map(|c| Message { role: Role::Tool, parts: vec![Part::ToolResult(ToolResult { call_id: c.id.clone(), content: "x".into() })] })
        .collect();
    let turn = b2.start_turn(s).await.unwrap();
    let _ = collect(turn.events).await;
    assert_eq!(env.stdin_lines().last().unwrap()["message"]["content"], "[runtime] continue after the tool results");
    assert_eq!(std::fs::read_to_string(env.pidfile.with_extension("system")).unwrap(), "caller system\n\nRUNTIME-NOTE-42");
    let snap2 = turn.control.freeze().await.unwrap();
    let f2 = SessionFile::load(Path::new(&snap2.path)).unwrap();
    assert_eq!(f2.count_synthetic(), 0);
    assert_eq!(f2.count_glue(), 2, "custom filler and custom trigger kept as glue");
    let glue: Vec<&Value> = f2.entries().filter(|e| e[bridge_claude::session::GLUE_KEY] == true).collect();
    assert_eq!(glue[0]["message"]["content"][0]["text"], "Reading the results now.");
    assert_eq!(glue[1]["message"]["content"], "[runtime] continue after the tool results");
    assert!(!f2.entries().any(|e| e.to_string().contains(bridge_claude::session::BRIDGE_FILLER_TEXT)));
}

#[tokio::test]
async fn keep_glue_false_strips_glue_at_freeze() {
    let env = Env::new();
    // tuning.keep_glue for the first round, the deprecated options.keep_glue for the second.
    let cfg = BackendConfig { tuning: BackendTuning { keep_glue: false, ..BackendTuning::default() }, ..BackendConfig::default() };
    let b = env.backend_cfg("stream_tool_call.jsonl", "replay", cfg);
    let turn = b.start_turn(spec(user("read both"))).await.unwrap();
    let out = collect(turn.events).await;
    let snap = turn.control.freeze().await.unwrap();
    let b2 = env.backend_with("stream_text.jsonl", "replay", json!({"keep_glue": false}));
    let mut s = spec(Trigger::ContinueAfterResults);
    s.base = Some(CheckpointRef { backend: "claude".into(), snapshot: snap });
    s.append = out
        .calls
        .iter()
        .map(|c| Message { role: Role::Tool, parts: vec![Part::ToolResult(ToolResult { call_id: c.id.clone(), content: "x".into() })] })
        .collect();
    let turn = b2.start_turn(s).await.unwrap();
    let _ = collect(turn.events).await;
    let snap2 = turn.control.freeze().await.unwrap();
    let f2 = SessionFile::load(Path::new(&snap2.path)).unwrap();
    assert_eq!(f2.count_glue(), 0);
    assert_eq!(f2.count_synthetic(), 0);
    assert!(!f2.entries().any(|e| e["message"]["content"] == bridge_claude::session::TRIGGER_TEXT));
    // One chain: the answer hangs off the tool results.
    let uuids = f2.uuids();
    assert!(f2.entries().filter_map(|e| e["parentUuid"].as_str()).all(|p| uuids.contains(p)));
}

#[tokio::test]
async fn awaiting_base_requires_matching_results() {
    let env = Env::new();
    let b = env.backend("stream_tool_call.jsonl", "replay");
    let turn = b.start_turn(spec(user("read both"))).await.unwrap();
    let _ = collect(turn.events).await;
    let snap = turn.control.freeze().await.unwrap();
    let mut s = spec(user("ignore the tools"));
    s.base = Some(CheckpointRef { backend: "claude".into(), snapshot: snap });
    let err = b.start_turn(s).await.err().unwrap();
    assert_eq!(err.class, ErrorClass::Internal);
    assert!(!err.message.contains('/'));
}

#[tokio::test]
async fn abort_kills_process_group_and_discard_cleans_up() {
    let env = Env::new();
    let b = env.backend("stream_text.jsonl", "hang");
    let mut turn = b.start_turn(spec(user("hi"))).await.unwrap();
    let first = turn.events.next().await.unwrap();
    assert_eq!(first, TurnEvent::TextDelta { text: "partial".into() });
    let pids = env.pids();
    assert!(pids.len() >= 3, "claude, mcp, sleep: {pids:?}");
    let t = std::time::Instant::now();
    turn.control.abort().await;
    turn.control.abort().await; // idempotent
    assert!(t.elapsed() < Duration::from_secs(8));
    assert_all_dead(&pids).await;
    turn.control.discard().await;
    assert!(env.session_files().is_empty());
}

#[tokio::test]
async fn drop_kills_process_group() {
    let env = Env::new();
    let b = env.backend("stream_text.jsonl", "hang");
    let mut turn = b.start_turn(spec(user("hi"))).await.unwrap();
    let _ = turn.events.next().await;
    let pids = env.pids();
    drop(turn);
    assert_all_dead(&pids).await;
    assert!(env.session_files().is_empty());
}

#[tokio::test]
async fn crash_is_classified_safely() {
    let env = Env::new();
    let b = env.backend("stream_text.jsonl", "crash");
    let turn = b.start_turn(spec(user("hi"))).await.unwrap();
    let out = collect(turn.events).await;
    let e = out.error.expect("error event");
    assert_eq!(e.class, ErrorClass::BackendAuth);
    assert!(!e.message.to_lowercase().contains("claude") && !e.message.contains("/login"));
    assert!(turn.control.freeze().await.is_err());
}

#[tokio::test]
async fn structured_output_becomes_text() {
    let env = Env::new();
    let b = env.backend("stream_schema.jsonl", "replay");
    let mut s = spec(user("red"));
    s.json_schema = Some(json!({"type": "object"}));
    let turn = b.start_turn(s).await.unwrap();
    let out = collect(turn.events).await;
    assert_eq!(serde_json::from_str::<Value>(&out.text).unwrap(), json!({"name": "Red", "hex": "#FF0000"}));
    assert_eq!(out.finish, Some(FinishReason::Stop));
    assert!(env.argv().contains(&"--json-schema".to_string()));
    turn.control.discard().await;
}

#[tokio::test]
async fn ephemeral_turn_keeps_nothing() {
    let env = Env::new();
    let b = env.backend("stream_text.jsonl", "replay");
    let mut s = spec(user("title please"));
    s.ephemeral = true;
    s.append = vec![Message::text(Role::User, "earlier"), Message::text(Role::Assistant, "ok")];
    let turn = b.start_turn(s).await.unwrap();
    let out = collect(turn.events).await;
    assert_eq!(out.finish, Some(FinishReason::Stop));
    let argv = env.argv();
    assert!(argv.contains(&"--no-session-persistence".to_string()));
    assert!(argv.contains(&"--resume".to_string()), "history written, then resumed");
    assert!(turn.control.freeze().await.is_err());
    assert!(env.session_files().is_empty());
}

#[tokio::test]
async fn transcript_mode_renders_history() {
    let env = Env::new();
    let b = env.backend("stream_tool_call.jsonl", "replay");
    b.set_mode(BackendMode::Transcript);
    let mut s = spec(user("and now?"));
    s.append = vec![
        Message::text(Role::System, "sys"),
        Message::text(Role::User, "my name is Ann"),
        Message::text(Role::Assistant, "hello Ann"),
    ];
    let turn = b.start_turn(s).await.unwrap();
    let out = collect(turn.events).await;
    assert_eq!(out.finish, Some(FinishReason::ToolCalls), "tool calls still intercepted");
    let content = env.stdin_lines()[0]["message"]["content"].as_str().unwrap().to_string();
    assert!(content.contains("<transcript>") && content.contains("[User]\nmy name is Ann") && content.contains("[Assistant]\nhello Ann"));
    assert!(content.ends_with("and now?"));
    assert!(!content.contains("[System]\nsys"), "leading system prompt goes via --system-prompt");
    let argv = env.argv();
    assert!(argv.contains(&"--no-session-persistence".to_string()) && argv.contains(&"--session-id".to_string()));
    assert!(turn.control.freeze().await.is_err());
    assert_all_dead(&env.pids()).await;
}

#[tokio::test]
async fn selftest_cache_is_used_per_version() {
    let env = Env::new();
    let root = env.work.join("claude");
    std::fs::create_dir_all(&root).unwrap();
    let rec = SelfTestRecord { version: "2.1.325".into(), mode: BackendMode::Transcript, detail: "test".into(), at: "x".into() };
    std::fs::write(root.join("selftest.json"), serde_json::to_string(&rec).unwrap()).unwrap();
    let b = env.backend("stream_text.jsonl", "replay");
    assert_eq!(b.mode(), BackendMode::Transcript);
    assert_eq!(b.self_test().await, BackendMode::Transcript);
    assert!(env.pids().is_empty(), "no canary run");
    // Another version's cache is ignored at construction.
    let rec = SelfTestRecord { version: "0.0.1".into(), ..rec };
    std::fs::write(root.join("selftest.json"), serde_json::to_string(&rec).unwrap()).unwrap();
    let b = env.backend("stream_text.jsonl", "replay");
    assert_eq!(b.mode(), BackendMode::Native);
}

#[tokio::test]
async fn missing_binary_disables_backend() {
    let env = Env::new();
    let cfg = BackendConfig { binary: Some("/nonexistent/claude-binary".into()), ..BackendConfig::default() };
    let b = ClaudeBackend::with_projects_root("claude", &cfg, &env.work, vec![], env.projects.clone()).unwrap();
    assert_eq!(b.mode(), BackendMode::Disabled);
    assert_eq!(b.self_test().await, BackendMode::Disabled);
    let err = b.start_turn(spec(user("hi"))).await.err().unwrap();
    assert_eq!(err.class, ErrorClass::BackendAuth);
    let caps = b.capabilities();
    assert!(caps.images && caps.json_schema && caps.effort);
    assert_eq!(caps.max_context, 200_000);
}
