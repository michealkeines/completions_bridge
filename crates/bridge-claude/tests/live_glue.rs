//! Live verification of persistent glue (ARCHITECTURE 17.2, gates V1-V3 in section 21)
//! against the real `claude` CLI. Skipped unless BRIDGE_LIVE_CLAUDE=1.
//!
//! - `live_v1_v3_glued_tool_loop`: a 10-round tool loop through start_turn/freeze, each
//!   round forked from the previous snapshot. Checks the `bridgeGlue` marker survives every
//!   resume (V1), prints cache_read/prompt per round and checks the cache extends (V3), and
//!   asks meta questions on the real glued chain plus a direct-answer check (V2).
//! - `live_v2_meta_questions_after_25_glued_rounds`: a session with 25 glued tool rounds
//!   and 3 real user messages written directly, then meta questions and a direct-answer
//!   check after a tool round (V2).
//!
//! Numbers are printed with `--nocapture`.

use std::path::{Path, PathBuf};

use bridge_claude::codec::{CallIdCodec, ToolNameCodec};
use bridge_claude::selftest::{collect, Collected};
use bridge_claude::session::{self, EntryContext, SessionFile};
use bridge_claude::ClaudeBackend;
use bridge_types::*;
use serde_json::json;

const MODEL: &str = "claude-haiku-5-5";

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
        let _ = std::fs::remove_dir_all(&self.backend.shared().projects_dir);
    }
}

/// A realistic-size system prompt (~5k tokens) so the cache numbers are not dominated by
/// the per-round tail, as in real agent traffic (opencode's prompt is larger still).
fn system() -> String {
    let mut s = String::from(
        "You are a coding assistant working in the user's project. The project is reachable only \
         through your tools. Be brief. Call at most one tool per message unless told otherwise.\n\n# Guidelines\n",
    );
    let topics = [
        "naming", "error handling", "testing", "logging", "formatting", "dependencies", "documentation",
        "performance", "security", "concurrency", "configuration", "release process",
    ];
    for (i, t) in topics.iter().enumerate() {
        for j in 0..6 {
            s.push_str(&format!(
                "- Guideline {i}.{j} on {t}: prefer small, reviewable changes; keep {t} consistent with the \
                 surrounding code; explain any non-obvious {t} decision in one short sentence; never change \
                 unrelated code while working on {t}; when unsure about {t}, read the relevant files first.\n"
            ));
        }
    }
    s
}

fn read_tool() -> ToolSpec {
    ToolSpec {
        name: "read".into(),
        description: "Read a file in the user's project. This is the only way to see project files.".into(),
        parameters: json!({"type": "object", "properties": {"filePath": {"type": "string"}}, "required": ["filePath"]}),
        strict: false,
    }
}

fn spec(base: Option<SnapshotInfo>, append: Vec<Message>, trigger: Trigger) -> TurnSpec {
    TurnSpec {
        model: MODEL.into(),
        system: system(),
        tools: vec![read_tool()],
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

fn tool_msg(id: &str, content: &str) -> Message {
    Message { role: Role::Tool, parts: vec![Part::ToolResult(ToolResult { call_id: id.into(), content: content.into() })] }
}

fn usage_of(out: &Collected) -> Usage {
    out.events.iter().find_map(|e| if let TurnEvent::Usage { usage } = e { Some(*usage) } else { None }).unwrap_or_default()
}

/// Status talk the direct-answer check rejects (E3).
fn status_talk(text: &str) -> bool {
    let t = text.to_lowercase();
    ["previous task", "nothing is pending", "nothing pending", "what would you like", "what's next", "anything else",
        "no response", "continue", "system-reminder", "runtime", "let me look at the tool results"]
        .iter()
        .any(|p| t.contains(p))
}

struct Run {
    out: Collected,
    snap: Option<SnapshotInfo>,
    /// claude's own `<synthetic>` fillers in the working file before freeze.
    claude_fillers: usize,
}

async fn run(b: &ClaudeBackend, s: TurnSpec, freeze: bool) -> Run {
    let (events, mut ctl) = b.start_turn_concrete(s).await.expect("start_turn");
    let out = collect(events).await;
    let _ = ctl.wait().await;
    let claude_fillers = ctl.session_path().and_then(|p| SessionFile::load(p).ok()).map(|f| f.count_synthetic()).unwrap_or(0);
    let snap = if freeze && out.error.is_none() { ctl.freeze_inner().await.ok() } else {
        ctl.discard_inner().await;
        None
    };
    Run { out, snap, claude_fillers }
}

fn short(s: &str) -> String {
    s.chars().take(300).collect::<String>().replace('\n', " ")
}

const WORDS: [&str; 10] = ["AMBER", "BISON", "CEDAR", "DELTA", "EMBER", "FJORD", "GECKO", "HAZEL", "IGLOO", "JASPER"];

fn chain_file(k: usize) -> String {
    // A few hundred bytes per result, like a small file read.
    let body: String = (0..8).map(|i| format!("// line {i}: helper code for step {k}\n")).collect();
    if k + 1 < WORDS.len() {
        format!("{body}Code word: {}. Next file: chain/{}.txt", WORDS[k], k + 2)
    } else {
        format!("{body}Code word: {}. END of chain.", WORDS[k])
    }
}

#[tokio::test]
async fn live_v1_v3_glued_tool_loop() {
    require_live!();
    let l = Live::new();
    let b = &l.backend;
    assert!(b.shared().keep_glue());
    let mut runs = 0;
    let mut rows: Vec<(usize, u64, u64, u64)> = Vec::new(); // round, prompt, cache_read, completion
    let mut snaps: Vec<SnapshotInfo> = Vec::new();

    let r = run(
        b,
        spec(None, vec![], user("Follow the chain: read chain/1.txt. Each file names the next file. Read them one at a time, one tool call per message, until a file says END, then list all code words you saw, in order, comma-separated.")),
        true,
    )
    .await;
    runs += 1;
    assert_eq!(r.out.finish, Some(FinishReason::ToolCalls), "{:?}", r.out.text);
    let u = usage_of(&r.out);
    rows.push((1, u.prompt_tokens, u.cached_tokens, u.completion_tokens));
    let mut snap = r.snap.expect("freeze round 1");
    let mut calls = r.out.calls.clone();
    snaps.push(snap.clone());
    let mut final_text = String::new();
    let mut glue_uuids: Vec<String> = Vec::new();
    let mut fed = 0usize;

    for round in 2..=13 {
        // Answer every call (the model should make one per message).
        let results: Vec<Message> = calls
            .iter()
            .map(|c| {
                let k = fed.min(WORDS.len() - 1);
                fed += 1;
                tool_msg(&c.id, &chain_file(k))
            })
            .collect();
        let r = run(b, spec(Some(snap.clone()), results, Trigger::ContinueAfterResults), true).await;
        runs += 1;
        assert!(r.out.error.is_none(), "round {round}: {:?}", r.out.error);
        assert_eq!(r.claude_fillers, 0, "round {round}: claude inserted its own filler");
        let u = usage_of(&r.out);
        rows.push((round, u.prompt_tokens, u.cached_tokens, u.completion_tokens));
        let s = r.snap.expect("freeze");
        snaps.push(s.clone());
        // V1: every earlier glue entry is still marked after claude resumed the file.
        let f = SessionFile::load(Path::new(&s.path)).unwrap();
        let now: Vec<String> = f
            .entries()
            .filter(|e| session::is_glue(e))
            .filter_map(|e| e["uuid"].as_str().map(str::to_string))
            .collect();
        for g in &glue_uuids {
            assert!(now.contains(g), "round {round}: glue entry {g} lost its marker");
        }
        assert_eq!(now.len(), glue_uuids.len() + 2, "round {round}: filler + trigger added");
        assert_eq!(f.count_synthetic(), 0);
        glue_uuids = now;
        if r.out.finish != Some(FinishReason::ToolCalls) {
            final_text = r.out.text.clone();
            snap = s;
            break;
        }
        calls = r.out.calls.clone();
        snap = s;
    }

    eprintln!("V3 round | prompt | cache_read | ratio | completion");
    let mut prev_prompt = 0;
    for (round, prompt, cached, completion) in &rows {
        let ratio = if *prompt > 0 { *cached as f64 / *prompt as f64 } else { 0.0 };
        eprintln!(
            "V3 {round:>5} | {prompt:>6} | {cached:>10} | {:>5.1}% | {completion:>4} | previous prompt not read from cache: {}",
            ratio * 100.0,
            prev_prompt as i64 - *cached as i64
        );
        prev_prompt = *prompt;
    }
    eprintln!("V2 final answer after the loop: {}", short(&final_text));
    for w in WORDS {
        assert!(final_text.to_uppercase().contains(w), "missing {w}: {final_text}");
    }
    assert!(!status_talk(&final_text), "status talk: {final_text}");

    // V2 on the real glued chain.
    let q = run(b, spec(Some(snap.clone()), vec![], user("Quick question about our chat, not the code: how many messages have I, the user, typed to you in this conversation before this one? Reply with just the number.")), false).await;
    let q2 = run(b, spec(Some(snap.clone()), vec![], user("Did I ever type the word \"Continue\" to you in this conversation? Answer yes or no first, then one short sentence.")), false).await;
    runs += 2;
    eprintln!("V2 real-chain count: {}", short(&q.out.text));
    eprintln!("V2 real-chain continue: {}", short(&q2.out.text));
    let mentions = [&q.out.text, &q2.out.text]
        .iter()
        .filter(|t| {
            let t = t.to_lowercase();
            t.contains("system note") || t.contains("automatic") || t.contains("let me look at the tool results")
        })
        .count();
    eprintln!("V2 real-chain answers mentioning glue: {mentions}/2 (reported, not asserted)");
    eprintln!("live runs used: {runs}");
    for s in &snaps {
        b.delete_snapshot(s).await;
    }
    assert!(q.out.text.contains('1') && !q.out.text.contains("11"), "count: {}", q.out.text);
    assert!(q2.out.text.trim_start().to_lowercase().starts_with("no"), "{}", q2.out.text);
    // V3 gate: from round 2 on the cache covers >= 90% of the prompt.
    for (round, prompt, cached, _) in rows.iter().skip(1) {
        assert!(*cached as f64 >= 0.9 * *prompt as f64, "round {round}: cache {cached}/{prompt}");
    }
}

/// Writes a session with 25 glued tool rounds and 3 real user messages, shaped like the
/// files the real flow produces (results, our filler, the marked trigger entry, the call).
fn glued_history(ctx: &EntryContext, awaiting_tail: bool) -> (SessionFile, Vec<String>) {
    let texts = session::Texts::default();
    let mut f = SessionFile::default();
    let mut names = ToolNameCodec::new(["read"]);
    let mut ids = CallIdCodec::default();
    let mut n = 0usize;
    let mut add = |f: &mut SessionFile, m: Vec<Message>| f.append_messages(ctx, &m, &mut names, &mut ids, false, &texts);
    let call = |id: &str, path: &str| Message {
        role: Role::Assistant,
        parts: vec![Part::ToolCall(ToolCall { id: id.into(), name: "read".into(), arguments: json!({"filePath": path}).to_string() })],
    };
    let glue = |f: &mut SessionFile| {
        f.append_bridge_filler(ctx, &texts.glue_filler);
        let pre = f.uuids();
        let mut nm = ToolNameCodec::default();
        let mut idc = CallIdCodec::default();
        f.append_messages(ctx, &[Message::text(Role::User, texts.continue_trigger.as_str())], &mut nm, &mut idc, false, &texts);
        assert!(f.mark_trigger_glue(&pre, &texts));
    };
    let users = [
        "Hi! I'm reviewing a small Rust project called parsley. Please read the files under src/ one per step and give me a short summary at the end.",
        "Thanks. Now go through the tests/ directory the same way and count the TODO comments as you go.",
        "Great. Last thing: look at the benches/ directory too, then tell me if any file looks slow.",
    ];
    let dirs = ["src", "tests", "benches"];
    let rounds = [10usize, 10, 5];
    for (u, (dir, k)) in users.iter().zip(dirs.iter().zip(rounds)) {
        add(&mut f, vec![Message::text(Role::User, *u)]);
        for i in 0..k {
            let id = format!("toolu_v2_{n:03}");
            n += 1;
            let path = format!("{dir}/file_{i}.rs");
            add(&mut f, vec![call(&id, &path)]);
            let todo = if i % 3 == 0 { "// TODO: handle errors\n" } else { "" };
            add(&mut f, vec![tool_msg(&id, &format!("// {path}\n{todo}pub fn f{i}() -> u32 {{ {i} }}\n"))]);
            glue(&mut f);
        }
        let summary = match *dir {
            "src" => "Summary of src/: ten small modules, each exporting one helper function returning a constant.",
            "tests" => "The tests/ directory has 4 TODO comments (file_0, file_3, file_6, file_9).",
            _ => "All five bench files are trivial one-line functions returning a constant, so none of them looks meaningfully slower than the others.",
        };
        add(&mut f, vec![Message::text(Role::Assistant, summary)]);
    }
    let mut pending = Vec::new();
    if awaiting_tail {
        add(&mut f, vec![Message::text(Role::User, "What edition does Cargo.toml declare?")]);
        add(&mut f, vec![call("toolu_v2_final", "Cargo.toml")]);
        // The placeholder claude writes when stopped at a call.
        let calls = vec![session::ClaudeCall { id: "toolu_v2_final".into(), name: "mcp__bridge__read".into(), input: json!({"filePath": "Cargo.toml"}) }];
        f.ensure_tool_round(ctx, None, &calls);
        pending.push("toolu_v2_final".to_string());
    }
    (f, pending)
}

fn write_snapshot(b: &ClaudeBackend, f: &SessionFile, pending: Vec<String>) -> SnapshotInfo {
    let path = b.shared().snapshots_dir.join(format!("{}.jsonl", uuid_like()));
    f.save(&path).unwrap();
    SnapshotInfo {
        path: path.to_string_lossy().into_owned(),
        kind: if pending.is_empty() { CheckpointKind::Idle } else { CheckpointKind::Awaiting },
        pending_call_ids: pending,
        size_bytes: std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0),
        backend_version: b.version(),
    }
}

fn uuid_like() -> String {
    format!("{:x}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos())
}

#[tokio::test]
async fn live_v2_meta_questions_after_25_glued_rounds() {
    require_live!();
    let l = Live::new();
    let b = &l.backend;
    let ctx = EntryContext {
        session_id: "written".into(),
        cwd: b.shared().cwd.to_string_lossy().into_owned(),
        version: b.version(),
        model_id: session::DEFAULT_HISTORY_MODEL.into(),
    };
    let (idle, _) = glued_history(&ctx, false);
    assert_eq!(idle.count_glue(), 50);
    let idle_snap = write_snapshot(b, &idle, vec![]);
    // Control: the same history with all glue stripped (v2 shape), used only to tell
    // glue-caused failures from questions the model gets wrong anyway.
    let mut control = idle.clone();
    control.strip_fillers(&Default::default(), false, false, &Default::default());
    assert_eq!(control.count_glue(), 0);
    let control_snap = write_snapshot(b, &control, vec![]);
    let mut runs = 0;
    let mut fails = Vec::new();

    type Check = fn(&str) -> bool;
    let questions: [(&str, Check); 5] = [
        (
            "Quick question about our chat, not the code: how many messages have I, the user, typed to you in this conversation before this one? Reply with just the number.",
            |t| t.contains('3') && !t.contains("13") && !t.contains("28"),
        ),
        (
            "Did I ever type the word \"Continue\" to you in this conversation? Answer yes or no first, then one short sentence.",
            |t| t.trim_start().to_lowercase().starts_with("no"),
        ),
        (
            "Did you ever give me an empty, wrong or 'no response' reply in this conversation, or decline one of my requests? Answer yes or no first, then one short sentence.",
            |t| t.trim_start().to_lowercase().starts_with("no"),
        ),
        (
            "Did you ever answer me with something like \"No response requested\", or ignore one of my requests? Answer yes or no first, then one short sentence.",
            |t| t.trim_start().to_lowercase().starts_with("no"),
        ),
        (
            "Quote my second message to you verbatim, nothing else.",
            |t| t.contains("count the TODO comments") && !t.contains("system-reminder"),
        ),
    ];
    for (q, ok) in questions {
        let r = run(b, spec(Some(idle_snap.clone()), vec![], user(q)), false).await;
        runs += 1;
        let pass = r.out.error.is_none() && ok(&r.out.text) && !r.out.text.to_lowercase().contains("let me look at the tool results");
        eprintln!("V2 meta [{}] {q} -> {}", if pass { "PASS" } else { "FAIL" }, short(&r.out.text));
        if !pass {
            let c = run(b, spec(Some(control_snap.clone()), vec![], user(q)), false).await;
            runs += 1;
            let control_pass = c.out.error.is_none() && ok(&c.out.text);
            eprintln!("V2 control (no glue) [{}] -> {}", if control_pass { "PASS" } else { "FAIL" }, short(&c.out.text));
            if control_pass {
                fails.push(q.to_string());
            }
        }
    }

    // Direct answer after a tool round (E3/S7), 3 forks of the same awaiting checkpoint.
    let (awaiting, pending) = glued_history(&ctx, true);
    let aw_snap = write_snapshot(b, &awaiting, pending);
    let mut direct = 0;
    for _ in 0..3 {
        let r = run(
            b,
            spec(
                Some(aw_snap.clone()),
                vec![tool_msg("toolu_v2_final", "[package]\nname = \"parsley\"\nversion = \"0.3.1\"\nedition = \"2021\"\n")],
                Trigger::ContinueAfterResults,
            ),
            false,
        )
        .await;
        runs += 1;
        let pass = r.out.error.is_none() && r.out.text.contains("2021") && !status_talk(&r.out.text) && r.claude_fillers == 0;
        eprintln!("V2 direct [{}] -> {}", if pass { "PASS" } else { "FAIL" }, short(&r.out.text));
        if pass {
            direct += 1;
        }
    }
    eprintln!("V2 direct answers: {direct}/3; live runs used: {runs}");
    b.delete_snapshot(&idle_snap).await;
    b.delete_snapshot(&control_snap).await;
    b.delete_snapshot(&aw_snap).await;
    assert!(fails.is_empty(), "meta questions failed because of glue: {fails:?}");
    assert_eq!(direct, 3);
}

/// Repeat probe for the one glue-sensitive meta question (run explicitly with --ignored).
#[tokio::test]
#[ignore]
async fn live_v2_repeat_no_response_question() {
    require_live!();
    let l = Live::new();
    let b = &l.backend;
    let ctx = EntryContext {
        session_id: "written".into(),
        cwd: b.shared().cwd.to_string_lossy().into_owned(),
        version: b.version(),
        model_id: session::DEFAULT_HISTORY_MODEL.into(),
    };
    let (idle, _) = glued_history(&ctx, false);
    let snap = write_snapshot(b, &idle, vec![]);
    let q = "Did you ever answer me with something like \"No response requested\", or ignore one of my requests? Answer yes or no first, then one short sentence.";
    let n: usize = std::env::var("BRIDGE_V2_REPEAT").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    let mut mentions = 0;
    for _ in 0..n {
        let r = run(b, spec(Some(snap.clone()), vec![], user(q)), false).await;
        let t = r.out.text.to_lowercase();
        let m = t.contains("let me look at the tool results") || t.contains("system-reminder") || t.contains("automatic");
        if m {
            mentions += 1;
        }
        eprintln!("V2 repeat [{}] -> {}", if m { "MENTIONS GLUE" } else { "clean" }, short(&r.out.text));
    }
    eprintln!("V2 repeat: {mentions}/{n} answers mention glue");
    b.delete_snapshot(&snap).await;
}
