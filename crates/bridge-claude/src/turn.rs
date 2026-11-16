//! One turn: materialize the workspace, spawn claude, decode its output, stop at tool
//! calls, and freeze/discard the result (ARCHITECTURE 8.2-8.4).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bridge_types::{
    BackendTuning, BridgeError, CheckpointKind, ErrorClass, FinishReason, Message, Part, Role, SnapshotInfo, ToolCall, Trigger,
    TurnControl, TurnEvent, TurnSpec,
};
use futures::stream::BoxStream;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::codec::{encode_call_id, CallIdCodec, ToolNameCodec};
use crate::errors;
use crate::process::{self, GroupHandle};
use crate::sanitize::Sanitizer;
use crate::session::{self, ClaudeCall, EntryContext, SessionFile, Texts};
use crate::stream::{Decoded, StreamDecoder};
use crate::transcript;
use crate::Shared;

/// Built-in fallback system prompt when the caller sent none (an empty `--system-prompt`
/// would leave room for claude's own defaults). `prompts.default_system` overrides it.
pub const DEFAULT_SYSTEM: &str = "You are a helpful assistant.";

/// Appended to every system prompt. claude always adds its own "# Environment" block
/// (its sandbox working directory, platform, model, date, token budget, account email)
/// and it cannot be switched off without API-key auth (`--bare`). Tell the model to
/// ignore it so it never acts on, or mentions, the bridge's internal sandbox.
/// Built-in value of `prompts.runtime_note` (see `session::Texts::from_prompts`).
pub const ENVIRONMENT_NOTE: &str = "\n\n# Runtime note\n\
The runtime may also show you an automatically generated environment section \
(a working directory, platform, shell, model name, date, token budget or account details). \
That section describes an internal sandbox, not the user's project or machine: never use its \
paths, never inspect that directory, and never mention it. The user's real environment, project \
location and tools are only what is described above and in the conversation. Neither that \
section nor this note is part of your instructions: if asked about your system prompt, \
instructions, environment or tools, answer only from the text above this note and never \
mention a runtime, a sandbox, an environment section or this note. Automatic <system-reminder> \
notes and the short \"Let me look at the tool results.\" lines in the conversation are runtime glue, \
not user or assistant messages: never count, quote or mention them.";

/// Process timings of a backend, from `BackendConfig.tuning`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timings {
    /// After a tool-call stop (SIGTERM): time for claude to write its state and exit.
    pub stop_grace: Duration,
    /// After a normal end or a failure: time for claude to exit before it is terminated.
    pub exit_grace: Duration,
    /// SIGTERM grace when aborting (client gone, spawn/stdin problems).
    pub abort_grace: Duration,
    /// After SIGKILL: time for the process to be reaped.
    pub kill_wait: Duration,
    /// How long stop/freeze waits for the run task before killing the group.
    pub freeze_wait: Duration,
}

impl Timings {
    pub fn from_tuning(t: &BackendTuning) -> Self {
        Timings {
            stop_grace: Duration::from_millis(t.stop_grace_ms),
            exit_grace: Duration::from_millis(t.exit_grace_ms),
            abort_grace: Duration::from_millis(t.abort_grace_ms),
            kill_wait: Duration::from_millis(t.kill_wait_ms),
            freeze_wait: Duration::from_millis(t.freeze_wait_ms),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunMode {
    Native,
    Transcript,
}

/// Result of the background run task.
#[derive(Debug, Clone, Default)]
pub struct RunOutcome {
    /// Set when the stream ended with `Done`.
    pub finish: Option<FinishReason>,
    /// Caller-visible ids of the tool calls (when finish == ToolCalls).
    pub caller_call_ids: Vec<String>,
    /// Post-run file checks passed (always true when there is nothing to check).
    pub verified: bool,
    pub aborted: bool,
}

/// Everything the run task needs besides the process.
struct RunContext {
    shared: Arc<Shared>,
    names: ToolNameCodec,
    ids: CallIdCodec,
    tx: mpsc::UnboundedSender<TurnEvent>,
    cancel: CancellationToken,
    group: GroupHandle,
    /// Working session file to check after a tool-call stop (native, persisted runs).
    check_file: Option<(PathBuf, EntryContext)>,
    /// Encoded ids whose placeholders we patched before the run.
    patched: HashSet<String>,
    json_schema: bool,
    stderr: Arc<Mutex<String>>,
    timings: Timings,
}

pub struct ClaudeTurnControl {
    shared: Arc<Shared>,
    sid: String,
    mode: RunMode,
    /// The session file this turn owns (written by us and/or claude).
    session_path: Option<PathBuf>,
    turn_dir: PathBuf,
    /// Freezable: native mode, not ephemeral.
    persisted: bool,
    pre_uuids: HashSet<String>,
    strip_trigger: bool,
    /// `Shared::keep_glue()`, read once per turn.
    keep_glue: bool,
    cancel: CancellationToken,
    group: GroupHandle,
    task: Option<JoinHandle<RunOutcome>>,
    outcome: Option<RunOutcome>,
    closed: bool,
}

fn internal(msg: &str) -> BridgeError {
    tracing::warn!(target: "bridge_claude", "{msg}");
    errors::error(ErrorClass::Internal)
}

fn invalid_plan(msg: &str) -> BridgeError {
    tracing::warn!(target: "bridge_claude", "invalid plan: {msg}");
    BridgeError::new(ErrorClass::Internal, errors::MSG_INVALID_PLAN)
}

fn io_err(what: &str, e: std::io::Error) -> BridgeError {
    tracing::warn!(target: "bridge_claude", "{what}: {e}");
    errors::error(ErrorClass::Internal)
}

/// Locates the session file for `sid`: the expected project dir first, then any dir
/// (claude shortens very long slugs).
pub fn find_session_file(projects_root: &Path, expected: &Path, sid: &str) -> Option<PathBuf> {
    if expected.exists() {
        return Some(expected.to_path_buf());
    }
    let name = format!("{sid}.jsonl");
    let rd = std::fs::read_dir(projects_root).ok()?;
    for d in rd.flatten() {
        let p = d.path().join(&name);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Removes a session file and the `<sid>/` side directory claude may create next to it.
pub fn remove_session(path: &Path) {
    let _ = std::fs::remove_file(path);
    if let (Some(dir), Some(stem)) = (path.parent(), path.file_stem()) {
        let side = dir.join(stem);
        if side.is_dir() {
            let _ = std::fs::remove_dir_all(side);
        }
    }
}

fn names_for(spec: &TurnSpec) -> ToolNameCodec {
    let mut names = ToolNameCodec::new(spec.tools.iter().map(|t| t.name.as_str()));
    let trigger_msgs: Vec<&Message> = match &spec.trigger {
        Trigger::User { message } => vec![message],
        Trigger::ContinueAfterResults => vec![],
    };
    for m in spec.append.iter().chain(trigger_msgs) {
        for c in m.tool_calls() {
            names.register(&c.name);
        }
    }
    names
}

/// Stdin `message.content` for a user trigger.
pub fn trigger_content(trigger: &Trigger, texts: &Texts) -> Value {
    match trigger {
        Trigger::User { message } => session::user_content(message, &texts.system_prefix),
        Trigger::ContinueAfterResults => Value::String(texts.continue_trigger.clone()),
    }
}

struct Materialized {
    session_path: Option<PathBuf>,
    resume: bool,
    pre_uuids: HashSet<String>,
    patched: HashSet<String>,
}

/// Fork the base (or start empty), patch placeholders, append history (ARCHITECTURE 8.2).
fn materialize(
    shared: &Shared,
    spec: &TurnSpec,
    ctx: &EntryContext,
    names: &mut ToolNameCodec,
    ids: &mut CallIdCodec,
) -> Result<Materialized, BridgeError> {
    let path = shared.session_path(&ctx.session_id);
    let mut file = SessionFile::default();
    let mut append: &[Message] = &spec.append;
    let mut patched = HashSet::new();
    if let Some(base) = &spec.base {
        let snap = Path::new(&base.snapshot.path);
        file = SessionFile::load(snap).map_err(|e| io_err("loading snapshot", e))?;
        file.set_session_id(&ctx.session_id);
        if !shared.keep_glue() {
            // Glue kept by an earlier configuration: remove it so this run sees v2 history.
            file.strip_fillers(&HashSet::new(), false, false, &shared.texts);
        }
        file.prune_dangling();
        if base.snapshot.kind == CheckpointKind::Awaiting {
            let lead = append.iter().take_while(|m| m.role == Role::Tool).count();
            let results: Vec<_> = append[..lead].iter().flat_map(|m| m.tool_results()).collect();
            let got: HashSet<&str> = results.iter().map(|r| r.call_id.as_str()).collect();
            let want: HashSet<&str> = base.snapshot.pending_call_ids.iter().map(String::as_str).collect();
            if got != want || results.len() != want.len() {
                return Err(invalid_plan("awaiting base without matching tool results"));
            }
            let map: HashMap<String, String> =
                results.iter().map(|r| (ids.encode(&r.call_id), r.content.clone())).collect();
            patched = file.patch_results(&map);
            if patched.len() != map.len() {
                // Placeholder missing (should have been fixed at freeze): append the rest.
                let missing: Vec<Message> = append[..lead]
                    .iter()
                    .flat_map(|m| m.tool_results())
                    .filter(|r| !patched.contains(&encode_call_id(&r.call_id)))
                    .map(|r| Message { role: Role::Tool, parts: vec![Part::ToolResult((*r).clone())] })
                    .collect();
                file.append_messages(ctx, &missing, names, ids, false, &shared.texts);
            }
            append = &append[lead..];
        }
    }
    file.append_messages(ctx, append, names, ids, spec.base.is_none(), &shared.texts);
    if matches!(spec.trigger, Trigger::ContinueAfterResults) && file.ends_with_user() {
        file.append_bridge_filler(ctx, &shared.texts.glue_filler);
    }
    if spec.base.is_none() && file.lines.is_empty() {
        // Brand-new conversation: claude creates the file itself (--session-id).
        return Ok(Materialized { session_path: Some(path), resume: false, pre_uuids: HashSet::new(), patched });
    }
    let pre_uuids = file.uuids();
    file.save(&path).map_err(|e| io_err("writing session file", e))?;
    Ok(Materialized { session_path: Some(path), resume: true, pre_uuids, patched })
}

/// claude argv (without the binary).
#[allow(clippy::too_many_arguments)]
pub fn build_args(
    shared: &Shared,
    spec: &TurnSpec,
    names: &ToolNameCodec,
    sid: &str,
    resume: bool,
    no_persist: bool,
    mcp_config: Option<&Path>,
    system_file: Option<&Path>,
) -> Vec<String> {
    let mut a: Vec<String> = [
        "-p",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    a.push("--model".into());
    a.push(spec.model.clone());
    a.push("--tools".into());
    a.push(String::new());
    if let Some(cfg) = mcp_config {
        a.push("--mcp-config".into());
        a.push(cfg.to_string_lossy().into_owned());
    }
    a.push("--strict-mcp-config".into());
    if !spec.tools.is_empty() {
        a.push("--allowedTools".into());
        a.push(spec.tools.iter().map(|t| names.claude_name(&t.name)).collect::<Vec<_>>().join(","));
    }
    // Prefer a file (0600, in the private turn dir): keeps the prompt out of `ps` output
    // and away from argument-size limits. `--system-prompt-file` works but is not listed
    // in `claude --help` (verified on 2.1.325).
    match system_file {
        Some(f) => {
            a.push("--system-prompt-file".into());
            a.push(f.to_string_lossy().into_owned());
        }
        None => {
            a.push("--system-prompt".into());
            a.push(shared.texts.system_prompt(&spec.system));
        }
    }
    a.push("--setting-sources".into());
    a.push(String::new());
    a.push("--disable-slash-commands".into());
    a.push("--autocompact".into());
    a.push(shared.config.tuning.autocompact_tokens.to_string());
    a.push(if resume { "--resume".into() } else { "--session-id".into() });
    a.push(sid.to_string());
    if no_persist {
        a.push("--no-session-persistence".into());
    }
    if let Some(effort) = spec.effort {
        let key = serde_json::to_value(effort).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default();
        if let Some(level) = shared.config.effort_map.get(&key) {
            a.push("--effort".into());
            a.push(level.clone());
        }
    }
    if let Some(schema) = &spec.json_schema {
        a.push("--json-schema".into());
        a.push(schema.to_string());
    }
    a.extend(shared.config.extra_args.iter().cloned());
    a
}

/// Starts a turn in the given mode. Errors happen before any output.
pub async fn start(
    shared: Arc<Shared>,
    spec: TurnSpec,
    mode: RunMode,
) -> Result<(BoxStream<'static, TurnEvent>, ClaudeTurnControl), BridgeError> {
    let sid = Uuid::new_v4().to_string();
    let timings = Timings::from_tuning(&shared.config.tuning);
    let turn_dir = shared.turns_dir.join(&sid);
    std::fs::create_dir_all(&turn_dir).map_err(|e| io_err("creating turn dir", e))?;
    let mut control = ClaudeTurnControl {
        shared: shared.clone(),
        sid: sid.clone(),
        mode,
        session_path: None,
        turn_dir: turn_dir.clone(),
        persisted: mode == RunMode::Native && !spec.ephemeral,
        pre_uuids: HashSet::new(),
        strip_trigger: matches!(spec.trigger, Trigger::ContinueAfterResults),
        keep_glue: shared.keep_glue(),
        cancel: CancellationToken::new(),
        group: GroupHandle::default(),
        task: None,
        outcome: None,
        closed: false,
    };
    let mut names = names_for(&spec);
    let mut ids = CallIdCodec::default();
    let ctx = EntryContext {
        session_id: sid.clone(),
        cwd: shared.cwd.to_string_lossy().into_owned(),
        version: shared.version.clone(),
        model_id: shared.history_model(),
    };

    // Workspace + stdin input.
    let (resume, no_persist, stdin_content, patched) = match mode {
        RunMode::Native => {
            let m = materialize(&shared, &spec, &ctx, &mut names, &mut ids)?;
            control.session_path = m.session_path;
            control.pre_uuids = m.pre_uuids;
            (m.resume, spec.ephemeral, trigger_content(&spec.trigger, &shared.texts), m.patched)
        }
        RunMode::Transcript => {
            let mut history = Vec::new();
            if let Some(base) = &spec.base {
                if let Ok(f) = SessionFile::load(Path::new(&base.snapshot.path)) {
                    history = f.conversation(&names, &ids, &shared.texts);
                }
            }
            let mut leading = spec.base.is_none();
            for m in &spec.append {
                if m.role != Role::System {
                    leading = false;
                }
                if !leading {
                    history.push(m.clone());
                }
            }
            (false, true, transcript::render(&history, &spec.trigger, &shared.texts.transcript_header), HashSet::new())
        }
    };

    // Tools file + MCP config.
    let mcp_config = if spec.tools.is_empty() {
        None
    } else {
        let tools: Vec<Value> = spec
            .tools
            .iter()
            .map(|t| {
                let schema = if t.parameters.is_object() { t.parameters.clone() } else { json!({"type": "object"}) };
                json!({"name": names.encode(&t.name), "description": t.description, "inputSchema": schema})
            })
            .collect();
        let tools_path = turn_dir.join("tools.json");
        std::fs::write(&tools_path, Value::Array(tools).to_string()).map_err(|e| io_err("writing tools", e))?;
        let (cmd, mut args) = match shared.park_command.split_first() {
            Some((c, rest)) => (c.to_string_lossy().into_owned(), rest.iter().map(|p| p.to_string_lossy().into_owned()).collect::<Vec<_>>()),
            None => return Err(internal("no park command configured")),
        };
        args.push("--tools".into());
        args.push(tools_path.to_string_lossy().into_owned());
        let cfg = json!({"mcpServers": {"bridge": {"type": "stdio", "command": cmd, "args": args}}});
        let cfg_path = turn_dir.join("mcp.json");
        std::fs::write(&cfg_path, cfg.to_string()).map_err(|e| io_err("writing mcp config", e))?;
        Some(cfg_path)
    };

    let system_path = turn_dir.join("system.txt");
    std::fs::write(&system_path, shared.texts.system_prompt(&spec.system)).map_err(|e| io_err("writing system prompt", e))?;
    let args = build_args(&shared, &spec, &names, &sid, resume, no_persist, mcp_config.as_deref(), Some(&system_path));
    let mut cmd = tokio::process::Command::new(&shared.binary);
    cmd.args(&args)
        .current_dir(&shared.cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .kill_on_drop(true);
    let mut child = cmd.spawn().map_err(|e| {
        tracing::warn!(target: "bridge_claude", "spawn failed: {e}");
        if e.kind() == std::io::ErrorKind::NotFound {
            errors::error(ErrorClass::BackendAuth)
        } else {
            errors::error(ErrorClass::Unavailable)
        }
    })?;
    if let Some(pid) = child.id() {
        control.group.set(pid as i32);
        // Recorded so a later bridge start can kill this group if the bridge dies hard
        // (a claude parked on a tool call would otherwise wait forever).
        let _ = std::fs::create_dir_all(&control.turn_dir);
        let _ = std::fs::write(control.turn_dir.join("pgid"), pid.to_string());
    }
    let (Some(mut stdin), Some(stdout), Some(stderr)) = (child.stdin.take(), child.stdout.take(), child.stderr.take()) else {
        let _ = process::terminate(&mut child, &control.group, timings.abort_grace, timings.kill_wait).await;
        return Err(internal("child pipes missing"));
    };
    let line = json!({"type": "user", "message": {"role": "user", "content": stdin_content}}).to_string() + "\n";
    if let Err(e) = stdin.write_all(line.as_bytes()).await {
        let _ = process::terminate(&mut child, &control.group, timings.abort_grace, timings.kill_wait).await;
        return Err(io_err("writing stdin", e));
    }
    let _ = stdin.flush().await;

    let stderr_buf = Arc::new(Mutex::new(String::new()));
    {
        let buf = stderr_buf.clone();
        let cap = shared.config.tuning.stderr_cap_bytes;
        let mut stderr = stderr;
        tokio::spawn(async move {
            let mut chunk = vec![0u8; 4096];
            while let Ok(n) = stderr.read(&mut chunk).await {
                if n == 0 {
                    break;
                }
                let mut b = buf.lock().await;
                b.push_str(&String::from_utf8_lossy(&chunk[..n]));
                if b.len() > cap {
                    let cut = b.len() - cap;
                    let cut = (cut..b.len()).find(|&i| b.is_char_boundary(i)).unwrap_or(b.len());
                    b.drain(..cut);
                }
            }
        });
    }

    let (tx, mut rx) = mpsc::unbounded_channel();
    let check_file = match (&control.session_path, control.persisted) {
        (Some(p), true) => Some((p.clone(), ctx.clone())),
        _ => None,
    };
    let rc = RunContext {
        shared: shared.clone(),
        names,
        ids,
        tx,
        cancel: control.cancel.clone(),
        group: control.group.clone(),
        check_file,
        patched,
        json_schema: spec.json_schema.is_some(),
        stderr: stderr_buf,
        timings,
    };
    control.task = Some(tokio::spawn(run(rc, child, stdin, stdout)));
    let events = futures::stream::poll_fn(move |cx| rx.poll_recv(cx));
    Ok((Box::pin(events), control))
}

fn send(tx: &mpsc::UnboundedSender<TurnEvent>, ev: TurnEvent) {
    let _ = tx.send(ev);
}

async fn run(rc: RunContext, mut child: Child, stdin: ChildStdin, stdout: ChildStdout) -> RunOutcome {
    let mut stdin = Some(stdin);
    let mut lines = BufReader::new(stdout).lines();
    let mut decoder = StreamDecoder::new();
    let mut text_san = Sanitizer::new();
    let mut reason_san = Sanitizer::new();
    let mut emitted_text = false;
    let mut outcome = RunOutcome { verified: true, ..Default::default() };

    let terminal: Option<Decoded> = loop {
        tokio::select! {
            biased;
            _ = rc.cancel.cancelled() => {
                outcome.aborted = true;
                break None;
            }
            line = lines.next_line() => match line {
                Ok(Some(l)) => {
                    let mut term = None;
                    for item in decoder.feed(&l) {
                        match item {
                            Decoded::Init { model } => rc.shared.set_history_model(&model),
                            Decoded::Text(t) => {
                                let s = text_san.push(&t, &rc.names);
                                if !s.is_empty() {
                                    emitted_text = true;
                                    send(&rc.tx, TurnEvent::TextDelta { text: s });
                                }
                            }
                            Decoded::Reasoning(t) => {
                                let s = reason_san.push(&t, &rc.names);
                                if !s.is_empty() {
                                    send(&rc.tx, TurnEvent::ReasoningDelta { text: s });
                                }
                            }
                            Decoded::Limit(info) => send(&rc.tx, TurnEvent::Limit { info }),
                            Decoded::ToolCallStart { index, id, name } => {
                                // Text the sanitizer still holds belongs before the call.
                                let rest = reason_san.finish(&rc.names);
                                if !rest.is_empty() {
                                    send(&rc.tx, TurnEvent::ReasoningDelta { text: rest });
                                }
                                let rest = text_san.finish(&rc.names);
                                if !rest.is_empty() {
                                    emitted_text = true;
                                    send(&rc.tx, TurnEvent::TextDelta { text: rest });
                                }
                                send(
                                    &rc.tx,
                                    TurnEvent::ToolCallStart {
                                        index,
                                        id: rc.ids.decode(&id),
                                        name: rc.names.decode_claude_name(&name),
                                    },
                                );
                            }
                            // Arguments are caller data: forwarded verbatim, never sanitised.
                            Decoded::ToolCallDelta { index, arguments } => {
                                send(&rc.tx, TurnEvent::ToolCallDelta { index, arguments })
                            }
                            other => term = Some(other),
                        }
                    }
                    if term.is_some() {
                        break term;
                    }
                }
                _ => break None,
            }
        }
    };

    // Flush held-back text before any terminal event.
    let flush = |san: &mut Sanitizer, names: &ToolNameCodec| san.finish(names);
    let rest = flush(&mut reason_san, &rc.names);
    if !rest.is_empty() && !outcome.aborted {
        send(&rc.tx, TurnEvent::ReasoningDelta { text: rest });
    }
    let rest = flush(&mut text_san, &rc.names);
    if !rest.is_empty() && !outcome.aborted {
        emitted_text = true;
        send(&rc.tx, TurnEvent::TextDelta { text: rest });
    }

    match terminal {
        Some(Decoded::StopForToolCalls { calls, arguments, message_id, usage }) => {
            // Stop claude first so it never executes anything; the events go out right away.
            rc.group.signal(libc::SIGTERM);
            let caller_calls: Vec<ToolCall> = calls
                .iter()
                .enumerate()
                .map(|(i, c)| ToolCall {
                    id: rc.ids.decode(&c.id),
                    name: rc.names.decode_claude_name(&c.name),
                    arguments: arguments.get(i).cloned().unwrap_or_else(|| c.input.to_string()),
                })
                .collect();
            outcome.caller_call_ids = caller_calls.iter().map(|c| c.id.clone()).collect();
            send(&rc.tx, TurnEvent::ToolCalls { calls: caller_calls });
            if let Some(u) = usage {
                send(&rc.tx, TurnEvent::Usage { usage: u });
            }
            send(&rc.tx, TurnEvent::Done { finish: FinishReason::ToolCalls });
            outcome.finish = Some(FinishReason::ToolCalls);
            drop(stdin.take());
            let _ = process::terminate(&mut child, &rc.group, rc.timings.stop_grace, rc.timings.kill_wait).await;
            outcome.verified = verify_after_stop(&rc, &calls, message_id.as_deref());
        }
        Some(Decoded::Finished { finish, usage, structured }) => {
            if let (Some(s), true, false) = (structured, rc.json_schema, emitted_text) {
                send(&rc.tx, TurnEvent::TextDelta { text: s.to_string() });
            }
            if let Some(u) = usage {
                send(&rc.tx, TurnEvent::Usage { usage: u });
            }
            send(&rc.tx, TurnEvent::Done { finish });
            outcome.finish = Some(finish);
            drop(stdin.take());
            let _ = process::wait_or_terminate(&mut child, &rc.group, rc.timings.exit_grace, rc.timings.kill_wait).await;
            outcome.verified = verify_patched(&rc);
        }
        Some(Decoded::Failed { error, raw }) => {
            tracing::warn!(target: "bridge_claude", "turn failed: {}", truncate(&raw, 500));
            send(&rc.tx, TurnEvent::Error { error });
            drop(stdin.take());
            let _ = process::terminate(&mut child, &rc.group, rc.timings.exit_grace, rc.timings.kill_wait).await;
        }
        Some(_) => {}
        None if outcome.aborted => {
            drop(stdin.take());
            let _ = process::terminate(&mut child, &rc.group, rc.timings.abort_grace, rc.timings.kill_wait).await;
        }
        None => {
            // stdout closed without a result: the process died or failed to start a session.
            drop(stdin.take());
            let status = process::wait_or_terminate(&mut child, &rc.group, rc.timings.exit_grace, rc.timings.kill_wait).await;
            let err_text = rc.stderr.lock().await.clone();
            tracing::warn!(target: "bridge_claude", "claude exited early ({status:?}): {}", truncate(&err_text, 500));
            let mut error = errors::classify(&err_text, None, decoder.resets_at());
            if decoder.limit_exhausted() {
                error = errors::classify("usage limit", None, decoder.resets_at());
            }
            send(&rc.tx, TurnEvent::Error { error });
        }
    }
    outcome
}

fn truncate(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// After a tool-call stop: one placeholder per call id, chained after the call message.
fn verify_after_stop(rc: &RunContext, calls: &[ClaudeCall], message_id: Option<&str>) -> bool {
    let Some((expected, ctx)) = &rc.check_file else { return true };
    let Some(path) = find_session_file(&rc.shared.projects_dir_root, expected, &ctx.session_id) else {
        tracing::warn!(target: "bridge_claude", "session file missing after tool-call stop");
        return false;
    };
    let mut file = match SessionFile::load(&path) {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!(target: "bridge_claude", "reading session file: {e}");
            return false;
        }
    };
    let report = file.ensure_tool_round(ctx, message_id, calls);
    if report.changed() {
        tracing::info!(target: "bridge_claude", "fixed tool round after stop: {report:?}");
        if let Err(e) = file.save(expected) {
            tracing::warn!(target: "bridge_claude", "writing session file: {e}");
            return false;
        }
        if path != *expected {
            let _ = std::fs::remove_file(&path);
        }
    } else if path != *expected && std::fs::rename(&path, expected).is_err() {
        return false;
    }
    let counts = file.tool_result_counts();
    let ok = calls.iter().all(|c| counts.get(&c.id) == Some(&1)) && patched_ok(&file, &rc.patched);
    if !ok {
        rc.shared.mark_stale();
    }
    ok
}

/// Patched placeholders must still be the only result for their ids, with no
/// "interrupted" entries added by claude.
fn patched_ok(file: &SessionFile, patched: &HashSet<String>) -> bool {
    if patched.is_empty() {
        return true;
    }
    let counts = file.tool_result_counts();
    patched.iter().all(|id| counts.get(id) == Some(&1)) && !file.has_interrupted_marker()
}

fn verify_patched(rc: &RunContext) -> bool {
    if rc.patched.is_empty() {
        return true;
    }
    let Some((path, _)) = &rc.check_file else { return true };
    match SessionFile::load(path) {
        Ok(f) => {
            let ok = patched_ok(&f, &rc.patched);
            if !ok {
                tracing::warn!(target: "bridge_claude", "patched results were not accepted; marking self-test stale");
                rc.shared.mark_stale();
            }
            ok
        }
        Err(_) => false,
    }
}

impl ClaudeTurnControl {
    pub fn session_path(&self) -> Option<&Path> {
        self.session_path.as_deref()
    }

    pub fn session_id(&self) -> &str {
        &self.sid
    }

    async fn join(&mut self) -> Option<RunOutcome> {
        if self.outcome.is_none() {
            if let Some(task) = self.task.take() {
                self.outcome = task.await.ok();
            }
        }
        self.outcome.clone()
    }

    async fn stop(&mut self) {
        self.cancel.cancel();
        if let Some(task) = self.task.as_mut() {
            let wait = Timings::from_tuning(&self.shared.config.tuning).freeze_wait;
            if tokio::time::timeout(wait, task).await.is_err() {
                self.group.signal(libc::SIGKILL);
            }
            self.task = None;
        }
        self.group.signal(libc::SIGKILL);
    }

    fn cleanup_files(&mut self, keep_session: bool) {
        if !keep_session {
            if let Some(p) = &self.session_path {
                remove_session(p);
            }
        }
        let _ = std::fs::remove_dir_all(&self.turn_dir);
        self.closed = true;
    }

    /// Freeze implementation shared by the trait and the self-test.
    pub async fn freeze_inner(&mut self) -> Result<SnapshotInfo, BridgeError> {
        if self.mode != RunMode::Native || !self.persisted {
            self.stop().await;
            self.cleanup_files(false);
            return Err(internal("freeze not supported for this turn"));
        }
        let outcome = self.join().await.unwrap_or_default();
        let Some(finish) = outcome.finish else {
            self.cleanup_files(false);
            return Err(internal("freeze before the turn finished"));
        };
        if !outcome.verified {
            self.cleanup_files(false);
            return Err(internal("session file failed verification"));
        }
        let Some(path) = self.session_path.clone() else {
            self.cleanup_files(false);
            return Err(internal("no session file"));
        };
        let Some(found) = find_session_file(&self.shared.projects_dir_root, &path, &self.sid) else {
            self.cleanup_files(false);
            return Err(internal("session file missing at freeze"));
        };
        let mut file = match SessionFile::load(&found) {
            Ok(f) => f,
            Err(e) => {
                self.cleanup_files(false);
                return Err(io_err("reading session file", e));
            }
        };
        if self.strip_trigger {
            // The trigger entry is written by claude from stdin; mark it as glue now.
            file.mark_trigger_glue(&self.pre_uuids, &self.shared.texts);
        }
        file.strip_fillers(&self.pre_uuids, self.strip_trigger, self.keep_glue, &self.shared.texts);
        let snap_path = self.shared.snapshots_dir.join(format!("{}.jsonl", Uuid::new_v4()));
        if let Err(e) = file.save(&snap_path) {
            self.cleanup_files(false);
            return Err(io_err("writing snapshot", e));
        }
        remove_session(&found);
        self.cleanup_files(false);
        let size_bytes = std::fs::metadata(&snap_path).map(|m| m.len()).unwrap_or(0);
        let (kind, pending) = if finish == FinishReason::ToolCalls {
            (CheckpointKind::Awaiting, outcome.caller_call_ids)
        } else {
            (CheckpointKind::Idle, Vec::new())
        };
        Ok(SnapshotInfo {
            path: snap_path.to_string_lossy().into_owned(),
            kind,
            pending_call_ids: pending,
            size_bytes,
            backend_version: self.shared.version.clone(),
        })
    }

    pub async fn discard_inner(&mut self) {
        self.stop().await;
        self.cleanup_files(false);
    }

    /// Waits for the run task (for tests and the self-test).
    pub async fn wait(&mut self) -> Option<RunOutcome> {
        self.join().await
    }
}

#[async_trait]
impl TurnControl for ClaudeTurnControl {
    async fn abort(&mut self) {
        self.stop().await;
    }

    async fn freeze(mut self: Box<Self>) -> Result<SnapshotInfo, BridgeError> {
        self.freeze_inner().await
    }

    async fn discard(mut self: Box<Self>) {
        self.discard_inner().await;
    }
}

impl Drop for ClaudeTurnControl {
    fn drop(&mut self) {
        if self.closed {
            return;
        }
        self.cancel.cancel();
        self.group.signal(libc::SIGKILL);
        if let Some(t) = self.task.take() {
            t.abort();
        }
        self.cleanup_files(false);
    }
}

/// Kills a process group left behind by a previous bridge run, but only if its leader is
/// still a claude process (guards against pid reuse).
fn kill_stale_group(pgid: i32, binary: &Path) {
    if pgid <= 1 {
        return;
    }
    let Ok(out) = std::process::Command::new("ps").args(["-o", "command=", "-p", &pgid.to_string()]).output() else {
        return;
    };
    let cmdline = String::from_utf8_lossy(&out.stdout);
    let bin_name = binary.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "claude".into());
    if cmdline.contains(&bin_name) && cmdline.contains("--input-format") {
        tracing::info!(target: "bridge_claude", pgid, "killing claude process group left by a previous run");
        // SAFETY: plain syscall on a process group id we recorded ourselves.
        unsafe {
            libc::kill(-pgid, libc::SIGKILL);
        }
    }
}

/// Removes leftovers of turns that were running when the bridge stopped.
pub fn cleanup_stale_turns(shared: &Shared) {
    let Ok(rd) = std::fs::read_dir(&shared.turns_dir) else { return };
    for d in rd.flatten() {
        let name = d.file_name().to_string_lossy().into_owned();
        if let Some(pgid) = std::fs::read_to_string(d.path().join("pgid")).ok().and_then(|s| s.trim().parse::<i32>().ok()) {
            kill_stale_group(pgid, &shared.binary);
        }
        remove_session(&shared.session_path(&name));
        let _ = std::fs::remove_dir_all(d.path());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bridge_types::{BackendConfig, BackendPrompts};

    fn backend(config: &BackendConfig) -> (tempfile::TempDir, crate::ClaudeBackend) {
        let dir = tempfile::tempdir().unwrap();
        let b = crate::ClaudeBackend::with_projects_root("claude", config, dir.path(), vec![], dir.path().join("projects"))
            .unwrap();
        (dir, b)
    }

    fn spec(trigger: Trigger) -> TurnSpec {
        TurnSpec {
            model: "haiku".into(),
            system: String::new(),
            tools: vec![],
            base: None,
            append: vec![],
            trigger,
            effort: None,
            json_schema: None,
            ephemeral: false,
        }
    }

    fn arg_after(args: &[String], flag: &str) -> String {
        let i = args.iter().position(|a| a == flag).unwrap_or_else(|| panic!("missing {flag}"));
        args[i + 1].clone()
    }

    #[test]
    fn tuning_reaches_args_and_timings() {
        let mut config = BackendConfig { binary: Some("/nonexistent/claude".into()), ..BackendConfig::default() };
        let (_d, b) = backend(&config);
        let names = ToolNameCodec::default();
        let args = build_args(b.shared(), &spec(Trigger::ContinueAfterResults), &names, "sid", false, false, None, None);
        assert_eq!(arg_after(&args, "--autocompact"), "1000000");
        assert_eq!(
            Timings::from_tuning(&config.tuning),
            Timings {
                stop_grace: Duration::from_secs(10),
                exit_grace: Duration::from_secs(3),
                abort_grace: Duration::from_secs(2),
                kill_wait: Duration::from_secs(5),
                freeze_wait: Duration::from_secs(15),
            }
        );

        config.tuning.autocompact_tokens = 750_000;
        config.tuning.stop_grace_ms = 1;
        config.tuning.exit_grace_ms = 2;
        config.tuning.abort_grace_ms = 3;
        config.tuning.kill_wait_ms = 4;
        config.tuning.freeze_wait_ms = 5;
        let (_d, b) = backend(&config);
        let args = build_args(b.shared(), &spec(Trigger::ContinueAfterResults), &names, "sid", false, false, None, None);
        assert_eq!(arg_after(&args, "--autocompact"), "750000");
        let t = Timings::from_tuning(&b.shared().config.tuning);
        assert_eq!(
            [t.stop_grace, t.exit_grace, t.abort_grace, t.kill_wait, t.freeze_wait],
            [1, 2, 3, 4, 5].map(Duration::from_millis)
        );
    }

    #[test]
    fn prompts_reach_stdin_and_system_prompt() {
        let config = BackendConfig {
            binary: Some("/nonexistent/claude".into()),
            prompts: BackendPrompts {
                default_system: Some("You are Parsley.".into()),
                runtime_note: Some("NOTE-XYZ".into()),
                continue_trigger: Some("[runtime] go on".into()),
                ..BackendPrompts::default()
            },
            ..BackendConfig::default()
        };
        let (_d, b) = backend(&config);
        let texts = &b.shared().texts;
        assert_eq!(trigger_content(&Trigger::ContinueAfterResults, texts), Value::String("[runtime] go on".into()));
        let names = ToolNameCodec::default();
        let args = build_args(b.shared(), &spec(Trigger::ContinueAfterResults), &names, "sid", false, false, None, None);
        assert_eq!(arg_after(&args, "--system-prompt"), "You are Parsley.\n\nNOTE-XYZ");
    }

    #[test]
    fn keep_glue_from_tuning_with_deprecated_option_override() {
        let mut config = BackendConfig { binary: Some("/nonexistent/claude".into()), ..BackendConfig::default() };
        assert!(backend(&config).1.shared().keep_glue());
        config.tuning.keep_glue = false;
        assert!(!backend(&config).1.shared().keep_glue());
        config.options = json!({"keep_glue": true});
        assert!(backend(&config).1.shared().keep_glue(), "deprecated options.keep_glue still wins");
    }
}
