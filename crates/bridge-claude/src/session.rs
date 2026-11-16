//! claude session files (`~/.claude/projects/<slug>/<session-id>.jsonl`).
//!
//! Port of `probes/session_lab.py`: writing history entries ourselves, appending after
//! the last conversation entry, patching the placeholder tool results claude writes when
//! it is stopped, stripping filler/trigger entries, fork-copying, and reading the
//! conversation back.

use std::collections::{HashMap, HashSet};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use bridge_types::{BackendPrompts, Image, ImageSource, Message, Part, Role, ToolCall, ToolResult};
use serde_json::{json, Map, Value};
use uuid::Uuid;

use crate::codec::{CallIdCodec, ToolNameCodec, TOOL_PREFIX};

/// Built-in stdin input for a ContinueAfterResults turn (`prompts.continue_trigger`
/// overrides it per backend). Tuned with real opencode traffic: a bare "Continue." made the
/// model answer with status talk ("the previous task is done", "nothing is pending")
/// instead of simply answering after the tool results.
pub const TRIGGER_TEXT: &str = "<system-reminder>The tool results for your last tool calls are above. \
Continue your reply to the user's request from where you left off. This note is automatic, not a new \
user message: do not acknowledge it, and do not summarise status or ask what to do next unless the \
user's request is fully done.</system-reminder>";
/// The trigger used by early bridge versions. Still recognised when stripping unmarked
/// trigger entries written during a run, so older checkpoints and runs clean up the same way.
pub const LEGACY_TRIGGER_TEXT: &str = "Continue.";
/// Filler claude writes itself when a run starts after a user entry. Recognised through
/// its `<synthetic>` model (never by text), so it is stripped whatever the configured texts.
pub const CLAUDE_FILLER_TEXT: &str = "No response requested.";

/// The texts this backend injects into the model's context, resolved once from
/// `BackendConfig.prompts` (`None` = the built-in constant). Passed explicitly to every
/// function that writes or recognises them: they are per backend, never process-global.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Texts {
    /// System prompt when the caller sent none (`prompts.default_system`).
    pub default_system: String,
    /// Suffix appended to every system prompt, including its leading separator; empty =
    /// no note (`prompts.runtime_note`).
    pub runtime_note: String,
    /// Stdin input of a ContinueAfterResults turn (`prompts.continue_trigger`).
    pub continue_trigger: String,
    /// Our `<synthetic>` assistant filler before that trigger (`prompts.glue_filler`).
    pub glue_filler: String,
    /// Prefix of mid-conversation system messages written as user text
    /// (`prompts.system_message_prefix`).
    pub system_prefix: String,
    /// Header of the transcript-mode rendering (`prompts.transcript_header`).
    pub transcript_header: String,
}

impl Default for Texts {
    fn default() -> Self {
        Texts::from_prompts(&BackendPrompts::default())
    }
}

impl Texts {
    /// Resolves configured texts. A blank `default_system` falls back to the built-in one
    /// (an empty system prompt would leave room for claude's own defaults). A configured
    /// `runtime_note` is appended after a blank line; an empty one disables the note. When
    /// only `glue_filler` is customised, the built-in runtime note names the configured
    /// filler instead of the built-in one.
    pub fn from_prompts(p: &BackendPrompts) -> Self {
        let glue_filler = p.glue_filler.clone().unwrap_or_else(|| BRIDGE_FILLER_TEXT.to_string());
        let runtime_note = match &p.runtime_note {
            Some(n) if n.trim().is_empty() => String::new(),
            Some(n) => format!("\n\n{n}"),
            None => crate::turn::ENVIRONMENT_NOTE.replace(BRIDGE_FILLER_TEXT, &glue_filler),
        };
        Texts {
            default_system: p
                .default_system
                .clone()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| crate::turn::DEFAULT_SYSTEM.to_string()),
            runtime_note,
            continue_trigger: p.continue_trigger.clone().unwrap_or_else(|| TRIGGER_TEXT.to_string()),
            glue_filler,
            system_prefix: p.system_message_prefix.clone().unwrap_or_else(|| SYSTEM_PREFIX.to_string()),
            transcript_header: p.transcript_header.clone().unwrap_or_else(|| crate::transcript::HEADER.to_string()),
        }
    }

    /// The system prompt for a turn: the caller's (or the default) plus the runtime note.
    pub fn system_prompt(&self, system: &str) -> String {
        let base = if system.trim().is_empty() { self.default_system.as_str() } else { system };
        format!("{base}{}", self.runtime_note)
    }

    /// The trigger this backend sends (marking at freeze only accepts this one: it is the
    /// text claude wrote from our stdin during the run).
    fn is_current_trigger(&self, text: &str) -> bool {
        let want = self.continue_trigger.trim();
        !want.is_empty() && text.trim() == want
    }

    /// Any trigger text the bridge has sent: the configured one, the built-in one and the
    /// legacy "Continue." (stripping unmarked trigger entries of a run).
    fn is_known_trigger(&self, text: &str) -> bool {
        let t = text.trim();
        self.is_current_trigger(t) || t == TRIGGER_TEXT.trim() || t == LEGACY_TRIGGER_TEXT
    }
}

/// Private key marking entries the bridge adds as runtime glue (our `<synthetic>` filler
/// and the ContinueAfterResults trigger entry). claude ignores unknown keys (V1). Marked
/// entries are kept at freeze when `keep_glue` is on, so request N+1 is an exact extension
/// of request N and the prompt cache keeps working (ARCHITECTURE 17.2).
pub const GLUE_KEY: &str = "bridgeGlue";
/// `message.model` of entries claude invents itself ("No response requested." etc.).
/// Not configurable: a fact of claude's session format.
pub const SYNTHETIC_MODEL: &str = "<synthetic>";
/// Built-in text of the bridge's own filler before a ContinueAfterResults trigger (probe
/// variant B); `prompts.glue_filler` overrides it per backend.
pub const BRIDGE_FILLER_TEXT: &str = "Let me look at the tool results.";
/// Content claude writes into the tool_result it records when it is stopped mid-call.
/// Not configurable: written by claude, the bridge only recognises it.
pub const PLACEHOLDER_TEXT: &str = "Connection closed";
/// Built-in prefix for system messages delivered as user-visible text
/// (`prompts.system_message_prefix` overrides it).
pub const SYSTEM_PREFIX: &str = "[System] ";
/// Text used for an assistant message without any content.
pub const EMPTY_ASSISTANT_TEXT: &str = "(no content)";
/// Model id written into assistant entries we create. claude only treats `<synthetic>`
/// specially; any real model id works (verified with this one in the probes).
pub const DEFAULT_HISTORY_MODEL: &str = "claude-haiku-5-5";

/// Static fields of entries we write.
#[derive(Debug, Clone)]
pub struct EntryContext {
    pub session_id: String,
    pub cwd: String,
    pub version: String,
    pub model_id: String,
}

/// One line of the file. Lines we cannot parse are kept verbatim.
#[derive(Debug, Clone, PartialEq)]
pub enum Line {
    Json(Value),
    Raw(String),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionFile {
    pub lines: Vec<Line>,
}

/// One tool call as claude sees it (encoded id and full claude name).
#[derive(Debug, Clone, PartialEq)]
pub struct ClaudeCall {
    pub id: String,
    pub name: String,
    pub input: Value,
}

/// What `ensure_tool_round` had to repair.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FixupReport {
    pub added_assistant_entries: usize,
    pub added_placeholders: usize,
    pub merged_result_entries: usize,
    pub relinked: bool,
}

impl FixupReport {
    pub fn changed(&self) -> bool {
        self.added_assistant_entries > 0 || self.added_placeholders > 0 || self.merged_result_entries > 0 || self.relinked
    }
}

pub fn now_timestamp() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()
}

/// `~/.claude/projects` directory name for a cwd: every non-alphanumeric char becomes '-'.
pub fn project_slug(cwd: &Path) -> String {
    cwd.to_string_lossy().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect()
}

/// Root of claude's project dirs: `$CLAUDE_CONFIG_DIR/projects` or `~/.claude/projects`.
pub fn default_projects_root() -> PathBuf {
    if let Some(dir) = std::env::var_os("CLAUDE_CONFIG_DIR") {
        return PathBuf::from(dir).join("projects");
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join(".claude").join("projects")
}

fn str_field<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Value::as_str)
}

fn entry_type(v: &Value) -> Option<&str> {
    str_field(v, "type")
}

fn is_conversation(v: &Value) -> bool {
    matches!(entry_type(v), Some("user") | Some("assistant")) && v.get("isSidechain").and_then(Value::as_bool) != Some(true)
}

fn is_synthetic_assistant(v: &Value) -> bool {
    entry_type(v) == Some("assistant") && v.pointer("/message/model").and_then(Value::as_str) == Some(SYNTHETIC_MODEL)
}

/// True for entries the bridge marked as runtime glue.
pub fn is_glue(v: &Value) -> bool {
    v.get(GLUE_KEY).and_then(Value::as_bool) == Some(true)
}

/// A user entry whose whole content (string, or text blocks only) satisfies `is_trigger`.
fn is_trigger_user(v: &Value, is_trigger: impl Fn(&str) -> bool) -> bool {
    if entry_type(v) != Some("user") {
        return false;
    }
    match v.pointer("/message/content") {
        Some(Value::String(s)) => is_trigger(s),
        Some(Value::Array(blocks)) => {
            !blocks.is_empty()
                && blocks.iter().all(|b| str_field(b, "type") == Some("text"))
                && is_trigger(&blocks.iter().filter_map(|b| str_field(b, "text")).collect::<String>())
        }
        _ => false,
    }
}

fn content_blocks(v: &Value) -> Vec<Value> {
    match v.pointer("/message/content") {
        Some(Value::String(s)) => vec![json!({"type": "text", "text": s})],
        Some(Value::Array(a)) => a.clone(),
        _ => Vec::new(),
    }
}

/// Text of a tool_result `content` (string or array of text blocks).
pub fn tool_result_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(a)) => a.iter().filter_map(|b| str_field(b, "text")).collect::<Vec<_>>().join(""),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

pub fn image_block(img: &Image) -> Value {
    match &img.source {
        ImageSource::Base64 { data } => {
            let mt = if img.media_type.is_empty() { "image/png" } else { img.media_type.as_str() };
            json!({"type": "image", "source": {"type": "base64", "media_type": mt, "data": data}})
        }
        ImageSource::Url { url } => json!({"type": "image", "source": {"type": "url", "url": url}}),
    }
}

/// claude `message.content` for a user-side message (text and images). System messages
/// get `system_prefix`.
pub fn user_content(msg: &Message, system_prefix: &str) -> Value {
    let system = msg.role == Role::System;
    let has_image = msg.parts.iter().any(|p| matches!(p, Part::Image(_)));
    if !has_image {
        let text = msg.text_content();
        return Value::String(if system { format!("{system_prefix}{text}") } else { text });
    }
    let mut blocks = Vec::new();
    let mut first_text = true;
    for p in &msg.parts {
        match p {
            Part::Text { text } => {
                let t = if system && first_text { format!("{system_prefix}{text}") } else { text.clone() };
                first_text = false;
                if !t.is_empty() {
                    blocks.push(json!({"type": "text", "text": t}));
                }
            }
            Part::Image(img) => blocks.push(image_block(img)),
            _ => {}
        }
    }
    Value::Array(blocks)
}

/// Tool call arguments (JSON string) → tool_use `input` object.
pub fn arguments_to_input(arguments: &str) -> Value {
    if arguments.trim().is_empty() {
        return json!({});
    }
    match serde_json::from_str::<Value>(arguments) {
        Ok(Value::Object(m)) => Value::Object(m),
        Ok(other) => json!({ "input": other }),
        Err(_) => json!({ "input": arguments }),
    }
}

fn random_message_id() -> String {
    let mut s = Uuid::new_v4().simple().to_string();
    s.truncate(24);
    format!("msg_{s}")
}

impl SessionFile {
    pub fn parse(text: &str) -> Self {
        let lines = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| match serde_json::from_str::<Value>(l) {
                Ok(v) => Line::Json(v),
                Err(_) => Line::Raw(l.to_string()),
            })
            .collect();
        SessionFile { lines }
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        Ok(Self::parse(&std::fs::read_to_string(path)?))
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        for l in &self.lines {
            match l {
                Line::Json(v) => out.push_str(&v.to_string()),
                Line::Raw(s) => out.push_str(s),
            }
            out.push('\n');
        }
        out
    }

    /// Atomic write (temp file + rename).
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let tmp = path.with_extension(format!("tmp-{}", Uuid::new_v4().simple()));
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(self.render().as_bytes())?;
            f.sync_all().ok();
        }
        std::fs::rename(&tmp, path).inspect_err(|_| {
            let _ = std::fs::remove_file(&tmp);
        })
    }

    pub fn entries(&self) -> impl Iterator<Item = &Value> {
        self.lines.iter().filter_map(|l| match l {
            Line::Json(v) => Some(v),
            Line::Raw(_) => None,
        })
    }

    fn entries_mut(&mut self) -> impl Iterator<Item = &mut Value> {
        self.lines.iter_mut().filter_map(|l| match l {
            Line::Json(v) => Some(v),
            Line::Raw(_) => None,
        })
    }

    fn json_at(&self, i: usize) -> Option<&Value> {
        match self.lines.get(i) {
            Some(Line::Json(v)) => Some(v),
            _ => None,
        }
    }

    pub fn push(&mut self, v: Value) {
        self.lines.push(Line::Json(v));
    }

    /// All entry uuids (used to tell pre-existing entries from new ones).
    pub fn uuids(&self) -> HashSet<String> {
        self.entries().filter_map(|e| str_field(e, "uuid")).map(str::to_string).collect()
    }

    /// Rewrites `sessionId` on every entry that has one (fork by copy, P2).
    pub fn set_session_id(&mut self, sid: &str) {
        for e in self.entries_mut() {
            if let Some(obj) = e.as_object_mut() {
                if obj.contains_key("sessionId") {
                    obj.insert("sessionId".into(), Value::String(sid.to_string()));
                }
            }
        }
    }

    /// uuid of the last user/assistant entry in file order.
    pub fn last_conversation_uuid(&self) -> Option<String> {
        self.entries().filter(|e| is_conversation(e)).filter_map(|e| str_field(e, "uuid")).last().map(str::to_string)
    }

    fn last_assistant_uuid(&self) -> Option<String> {
        self.entries()
            .filter(|e| is_conversation(e) && entry_type(e) == Some("assistant") && !is_synthetic_assistant(e))
            .filter_map(|e| str_field(e, "uuid"))
            .last()
            .map(str::to_string)
    }

    /// Appends a `<synthetic>` assistant entry so the file no longer ends with a user entry.
    /// Used before a ContinueAfterResults trigger: otherwise claude inserts its own filler
    /// ("No response requested."), which reads to the model as if it declined to answer.
    /// The entry is marked `bridgeGlue`: kept at freeze with `keep_glue`, stripped without.
    pub fn append_bridge_filler(&mut self, ctx: &EntryContext, text: &str) {
        let parent = self.last_conversation_uuid();
        let mut e =
            Self::assistant_entry(ctx, parent.as_deref(), &random_message_id(), vec![json!({"type": "text", "text": text})]);
        if let Some(m) = e.get_mut("message").and_then(Value::as_object_mut) {
            m.insert("model".into(), Value::String(SYNTHETIC_MODEL.to_string()));
        }
        if let Some(obj) = e.as_object_mut() {
            obj.insert(GLUE_KEY.into(), Value::Bool(true));
        }
        self.push(e);
    }

    /// Marks the ContinueAfterResults trigger entry claude wrote from stdin during this run
    /// as `bridgeGlue`. Candidates are user entries that did not exist before the run
    /// (`pre_run`) and whose whole content is the trigger text; exactly one stdin message is
    /// sent per run, so only the first candidate in file order is marked (a candidate
    /// parented to our own glue filler is preferred). Returns true if an entry was marked.
    /// Only the backend's configured trigger (`texts.continue_trigger`) is recognised here.
    pub fn mark_trigger_glue(&mut self, pre_run: &HashSet<String>, texts: &Texts) -> bool {
        let glue_uuids: HashSet<String> =
            self.entries().filter(|e| is_glue(e)).filter_map(|e| str_field(e, "uuid")).map(str::to_string).collect();
        let candidates: Vec<usize> = (0..self.lines.len())
            .filter(|&i| {
                self.json_at(i)
                    .map(|v| {
                        is_trigger_user(v, |t| texts.is_current_trigger(t))
                            && !is_glue(v)
                            && str_field(v, "uuid").map(|u| !pre_run.contains(u)).unwrap_or(false)
                    })
                    .unwrap_or(false)
            })
            .collect();
        let chosen = candidates
            .iter()
            .copied()
            .find(|&i| {
                self.json_at(i).and_then(|v| str_field(v, "parentUuid")).map(|p| glue_uuids.contains(p)).unwrap_or(false)
            })
            .or_else(|| candidates.first().copied());
        match chosen.and_then(|i| self.lines.get_mut(i)) {
            Some(Line::Json(Value::Object(obj))) => {
                obj.insert(GLUE_KEY.into(), Value::Bool(true));
                true
            }
            _ => false,
        }
    }

    /// Removes non-conversation entries (attachments etc.) that nothing continues from,
    /// repeatedly, so whole dangling attachment branches go. Returns how many were removed.
    ///
    /// claude hangs `attachment` entries (token reminders, prompt snapshots) off the last
    /// message of a run. Entries the bridge appends later are parented to the last
    /// conversation entry, which leaves those attachments as a second leaf, and on
    /// `--resume` claude may continue from that stale leaf: the appended tool round is
    /// then invisible to the model (observed with claude 2.1.325: the model repeated the
    /// previous tool call). Pruning leaves exactly one leaf.
    pub fn prune_dangling(&mut self) -> usize {
        let mut total = 0;
        loop {
            let referenced: HashSet<String> = self
                .entries()
                .flat_map(|e| ["parentUuid", "logicalParentUuid"].into_iter().filter_map(move |k| str_field(e, k)))
                .map(str::to_string)
                .collect();
            let before = self.lines.len();
            self.lines.retain(|l| match l {
                Line::Json(v) => match str_field(v, "uuid") {
                    Some(u) => matches!(entry_type(v), Some("user") | Some("assistant")) || referenced.contains(u),
                    None => true,
                },
                Line::Raw(_) => true,
            });
            let removed = before - self.lines.len();
            if removed == 0 {
                return total;
            }
            total += removed;
        }
    }

    /// True if the last conversation entry is a user entry (claude would add a filler).
    pub fn ends_with_user(&self) -> bool {
        self.entries().filter(|e| is_conversation(e)).last().map(|e| entry_type(e) == Some("user")).unwrap_or(false)
    }

    fn base_entry(ctx: &EntryContext, parent: Option<&str>, kind: &str) -> Map<String, Value> {
        let mut m = Map::new();
        m.insert("parentUuid".into(), parent.map(|p| Value::String(p.to_string())).unwrap_or(Value::Null));
        m.insert("isSidechain".into(), Value::Bool(false));
        m.insert("type".into(), Value::String(kind.to_string()));
        m.insert("uuid".into(), Value::String(Uuid::new_v4().to_string()));
        m.insert("timestamp".into(), Value::String(now_timestamp()));
        m.insert("userType".into(), Value::String("external".into()));
        m.insert("entrypoint".into(), Value::String("sdk-cli".into()));
        m.insert("cwd".into(), Value::String(ctx.cwd.clone()));
        m.insert("sessionId".into(), Value::String(ctx.session_id.clone()));
        m.insert("version".into(), Value::String(ctx.version.clone()));
        m
    }

    fn assistant_entry(ctx: &EntryContext, parent: Option<&str>, message_id: &str, content: Vec<Value>) -> Value {
        let has_calls = content.iter().any(|b| str_field(b, "type") == Some("tool_use"));
        let mut e = Self::base_entry(ctx, parent, "assistant");
        e.insert(
            "message".into(),
            json!({
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": ctx.model_id,
                "content": content,
                "stop_reason": if has_calls { "tool_use" } else { "end_turn" },
                "stop_sequence": null,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }),
        );
        Value::Object(e)
    }

    /// Appends IR messages after the last conversation entry.
    ///
    /// `skip_leading_system`: drop system messages that come before any other message
    /// (they are the conversation's system prompt, passed with `--system-prompt`). Other
    /// system messages become user text `[System] ...`. Consecutive tool messages are
    /// grouped into one user entry.
    pub fn append_messages(
        &mut self,
        ctx: &EntryContext,
        messages: &[Message],
        names: &mut ToolNameCodec,
        ids: &mut CallIdCodec,
        skip_leading_system: bool,
        texts: &Texts,
    ) {
        let mut parent = self.last_conversation_uuid();
        let mut last_assistant = self.last_assistant_uuid();
        let mut leading = skip_leading_system;
        let mut i = 0;
        while i < messages.len() {
            let m = &messages[i];
            if m.role != Role::System {
                leading = false;
            }
            let entry = match m.role {
                Role::System if leading => {
                    i += 1;
                    continue;
                }
                Role::System | Role::User => {
                    let mut e = Self::base_entry(ctx, parent.as_deref(), "user");
                    e.insert("message".into(), json!({"role": "user", "content": user_content(m, &texts.system_prefix)}));
                    Value::Object(e)
                }
                Role::Assistant => {
                    let mut content = Vec::new();
                    for p in &m.parts {
                        match p {
                            Part::Text { text } if !text.is_empty() => content.push(json!({"type": "text", "text": text})),
                            Part::ToolCall(c) => content.push(json!({
                                "type": "tool_use",
                                "id": ids.encode(&c.id),
                                "name": format!("{TOOL_PREFIX}{}", names.register(&c.name)),
                                "input": arguments_to_input(&c.arguments),
                            })),
                            _ => {}
                        }
                    }
                    if content.is_empty() {
                        content.push(json!({"type": "text", "text": EMPTY_ASSISTANT_TEXT}));
                    }
                    let e = Self::assistant_entry(ctx, parent.as_deref(), &random_message_id(), content);
                    last_assistant = str_field(&e, "uuid").map(str::to_string);
                    e
                }
                Role::Tool => {
                    let mut blocks = Vec::new();
                    let mut j = i;
                    while j < messages.len() && messages[j].role == Role::Tool {
                        for r in messages[j].tool_results() {
                            blocks.push(json!({
                                "type": "tool_result",
                                "content": r.content,
                                "tool_use_id": ids.encode(&r.call_id),
                            }));
                        }
                        j += 1;
                    }
                    i = j - 1;
                    let first = blocks.first().map(|b| tool_result_text(b.get("content"))).unwrap_or_default();
                    let mut e = Self::base_entry(ctx, parent.as_deref(), "user");
                    e.insert("message".into(), json!({"role": "user", "content": blocks}));
                    e.insert("toolUseResult".into(), Value::String(first));
                    if let Some(a) = &last_assistant {
                        e.insert("sourceToolAssistantUUID".into(), Value::String(a.clone()));
                    }
                    Value::Object(e)
                }
            };
            parent = str_field(&entry, "uuid").map(str::to_string);
            self.push(entry);
            i += 1;
        }
    }

    /// Overwrites placeholder tool_results with real results (by encoded call id).
    /// Drops `is_error` and `toolDenialKind`, sets `toolUseResult`. Returns the set of
    /// ids patched.
    pub fn patch_results(&mut self, results: &HashMap<String, String>) -> HashSet<String> {
        let mut patched = HashSet::new();
        for e in self.entries_mut() {
            if entry_type(e) != Some("user") {
                continue;
            }
            let mut first_text: Option<String> = None;
            if let Some(Value::Array(blocks)) = e.pointer_mut("/message/content") {
                for b in blocks.iter_mut() {
                    if str_field(b, "type") != Some("tool_result") {
                        continue;
                    }
                    let Some(id) = str_field(b, "tool_use_id").map(str::to_string) else { continue };
                    if let Some(text) = results.get(&id) {
                        if let Some(obj) = b.as_object_mut() {
                            obj.insert("content".into(), Value::String(text.clone()));
                            obj.remove("is_error");
                        }
                        if first_text.is_none() {
                            first_text = Some(text.clone());
                        }
                        patched.insert(id);
                    }
                }
            }
            if let (Some(t), Some(obj)) = (first_text, e.as_object_mut()) {
                obj.insert("toolUseResult".into(), Value::String(t));
                obj.remove("toolDenialKind");
            }
        }
        patched
    }

    /// Removes claude's synthetic filler replies and unmarked trigger entries and rewires
    /// `parentUuid` (and other uuid references) past them (P8, P9).
    ///
    /// With `keep_glue` (ARCHITECTURE 17.2) entries marked `bridgeGlue` are kept; only
    /// claude's own unmarked `<synthetic>` fillers and unmarked trigger entries go. Without
    /// it every synthetic entry and every glue-marked entry is removed (v2 behaviour).
    /// Unmarked trigger user entries are removed only when `strip_trigger` is set and their
    /// uuid is not in `protected` (entries that existed before the run, e.g. a real user
    /// message with the same text). Trigger entries are recognised by the configured
    /// trigger, the built-in one and the legacy "Continue."; claude's own filler ("No
    /// response requested.") by its `<synthetic>` model.
    pub fn strip_fillers(
        &mut self,
        protected: &HashSet<String>,
        strip_trigger: bool,
        keep_glue: bool,
        texts: &Texts,
    ) -> usize {
        let is_trigger = |e: &Value| is_trigger_user(e, |t| texts.is_known_trigger(t));
        let mut removed: HashMap<String, Option<String>> = HashMap::new();
        for e in self.entries() {
            let Some(uuid) = str_field(e, "uuid") else { continue };
            let glue = is_glue(e);
            let drop = if keep_glue {
                !glue
                    && (is_synthetic_assistant(e) || (strip_trigger && is_trigger(e) && !protected.contains(uuid)))
            } else {
                glue
                    || is_synthetic_assistant(e)
                    || (strip_trigger && is_trigger(e) && !protected.contains(uuid))
            };
            if drop {
                removed.insert(uuid.to_string(), str_field(e, "parentUuid").map(str::to_string));
            }
        }
        if removed.is_empty() {
            return 0;
        }
        let resolve = |start: &str| -> Option<String> {
            let mut cur = Some(start.to_string());
            let mut guard = 0;
            while let Some(u) = cur.clone() {
                match removed.get(&u) {
                    Some(p) if guard < 10_000 => {
                        cur = p.clone();
                        guard += 1;
                    }
                    _ => break,
                }
            }
            cur
        };
        self.lines.retain(|l| match l {
            Line::Json(v) => !str_field(v, "uuid").map(|u| removed.contains_key(u)).unwrap_or(false),
            Line::Raw(_) => true,
        });
        for e in self.entries_mut() {
            let Some(obj) = e.as_object_mut() else { continue };
            for key in ["parentUuid", "logicalParentUuid", "sourceToolAssistantUUID", "leafUuid"] {
                let target = obj.get(key).and_then(Value::as_str).map(str::to_string);
                if let Some(t) = target {
                    if removed.contains_key(&t) {
                        obj.insert(key.into(), resolve(&t).map(Value::String).unwrap_or(Value::Null));
                    }
                }
            }
        }
        removed.len()
    }

    fn find_tool_use(&self, id: &str) -> Option<usize> {
        self.lines.iter().position(|l| match l {
            Line::Json(v) => {
                entry_type(v) == Some("assistant")
                    && content_blocks(v)
                        .iter()
                        .any(|b| str_field(b, "type") == Some("tool_use") && str_field(b, "id") == Some(id))
            }
            Line::Raw(_) => false,
        })
    }

    /// Indices of user entries holding a tool_result for any of `ids`.
    fn result_entries(&self, ids: &HashSet<&str>) -> Vec<usize> {
        (0..self.lines.len())
            .filter(|&i| {
                self.json_at(i)
                    .map(|v| {
                        entry_type(v) == Some("user")
                            && content_blocks(v).iter().any(|b| {
                                str_field(b, "type") == Some("tool_result")
                                    && str_field(b, "tool_use_id").map(|t| ids.contains(t)).unwrap_or(false)
                            })
                    })
                    .unwrap_or(false)
            })
            .collect()
    }

    /// After stopping claude at a tool call: make sure the file holds every tool_use of the
    /// call message, chained, followed by exactly one user entry with one placeholder
    /// tool_result per call id (in call order), parented to the last tool_use entry.
    ///
    /// Observed (2.1.325): with parallel calls, claude may write a placeholder for the first
    /// call only, parented to the FIRST tool_use entry, which leaves the second call on a
    /// dead branch. This repairs that layout.
    pub fn ensure_tool_round(&mut self, ctx: &EntryContext, message_id: Option<&str>, calls: &[ClaudeCall]) -> FixupReport {
        let mut report = FixupReport::default();
        if calls.is_empty() {
            return report;
        }
        let msg_id = message_id.map(str::to_string).unwrap_or_else(random_message_id);
        // 1. tool_use entries.
        let mut last_assistant_idx: Option<usize> = None;
        for c in calls {
            match self.find_tool_use(&c.id) {
                Some(i) => last_assistant_idx = Some(last_assistant_idx.map_or(i, |m| m.max(i))),
                None => {
                    let parent = match last_assistant_idx.and_then(|i| self.json_at(i)) {
                        Some(v) => str_field(v, "uuid").map(str::to_string),
                        None => self.last_conversation_uuid(),
                    };
                    let block = json!({"type": "tool_use", "id": c.id, "name": c.name, "input": c.input});
                    let e = Self::assistant_entry(ctx, parent.as_deref(), &msg_id, vec![block]);
                    self.push(e);
                    last_assistant_idx = Some(self.lines.len() - 1);
                    report.added_assistant_entries += 1;
                }
            }
        }
        let Some(la_idx) = last_assistant_idx else { return report };
        let Some(la_uuid) = self.json_at(la_idx).and_then(|v| str_field(v, "uuid")).map(str::to_string) else {
            return report;
        };

        // 2. existing result entries.
        let id_set: HashSet<&str> = calls.iter().map(|c| c.id.as_str()).collect();
        let entries = self.result_entries(&id_set);
        let mut existing: HashMap<String, Value> = HashMap::new();
        let mut counts: HashMap<String, usize> = HashMap::new();
        for &i in &entries {
            if let Some(v) = self.json_at(i) {
                for b in content_blocks(v) {
                    if let Some(id) = str_field(&b, "tool_use_id") {
                        if id_set.contains(id) {
                            *counts.entry(id.to_string()).or_default() += 1;
                            existing.entry(id.to_string()).or_insert(b.clone());
                        }
                    }
                }
            }
        }
        let clean = entries.len() == 1
            && entries[0] > la_idx
            && self.json_at(entries[0]).and_then(|v| str_field(v, "parentUuid")) == Some(la_uuid.as_str())
            && calls.iter().all(|c| counts.get(&c.id) == Some(&1))
            && self.json_at(entries[0]).map(|v| content_blocks(v).len()) == Some(calls.len());
        if clean {
            return report;
        }

        // 3. build one merged entry.
        let mut blocks = Vec::new();
        for c in calls {
            match existing.get(&c.id) {
                Some(b) => blocks.push(b.clone()),
                None => {
                    blocks.push(json!({
                        "type": "tool_result",
                        "content": PLACEHOLDER_TEXT,
                        "is_error": true,
                        "tool_use_id": c.id,
                    }));
                    report.added_placeholders += 1;
                }
            }
        }
        let mut kept: Map<String, Value> = match entries.first().and_then(|&i| self.json_at(i)) {
            Some(Value::Object(m)) => m.clone(),
            _ => {
                let mut m = Self::base_entry(ctx, Some(&la_uuid), "user");
                m.insert("toolUseResult".into(), Value::String(format!("Error: {PLACEHOLDER_TEXT}")));
                m
            }
        };
        if entries.len() > 1 {
            report.merged_result_entries = entries.len() - 1;
        }
        if kept.get("parentUuid").and_then(Value::as_str) != Some(la_uuid.as_str()) && !entries.is_empty() {
            report.relinked = true;
        }
        kept.insert("parentUuid".into(), Value::String(la_uuid.clone()));
        kept.insert("sourceToolAssistantUUID".into(), Value::String(la_uuid.clone()));
        kept.insert("message".into(), json!({"role": "user", "content": blocks}));
        let kept_uuid = kept.get("uuid").and_then(Value::as_str).map(str::to_string).unwrap_or_default();

        // Remove old result entries; re-point their children to the merged entry.
        let removed_uuids: HashSet<String> = entries
            .iter()
            .filter_map(|&i| self.json_at(i).and_then(|v| str_field(v, "uuid")).map(str::to_string))
            .filter(|u| *u != kept_uuid)
            .collect();
        let remove_idx: HashSet<usize> = entries.iter().copied().collect();
        let mut new_lines = Vec::with_capacity(self.lines.len() + 1);
        for (i, l) in std::mem::take(&mut self.lines).into_iter().enumerate() {
            if !remove_idx.contains(&i) {
                new_lines.push(l);
            }
        }
        self.lines = new_lines;
        for e in self.entries_mut() {
            if let Some(obj) = e.as_object_mut() {
                let p = obj.get("parentUuid").and_then(Value::as_str).map(str::to_string);
                if let Some(p) = p {
                    if removed_uuids.contains(&p) {
                        obj.insert("parentUuid".into(), Value::String(kept_uuid.clone()));
                    }
                }
            }
        }
        // The merged entry goes to the end, after every tool_use entry.
        self.push(Value::Object(kept));
        report
    }

    /// Number of tool_result blocks per call id (verification: exactly one each).
    pub fn tool_result_counts(&self) -> HashMap<String, usize> {
        let mut out: HashMap<String, usize> = HashMap::new();
        for e in self.entries().filter(|e| entry_type(e) == Some("user")) {
            for b in content_blocks(e) {
                if str_field(&b, "type") == Some("tool_result") {
                    if let Some(id) = str_field(&b, "tool_use_id") {
                        *out.entry(id.to_string()).or_default() += 1;
                    }
                }
            }
        }
        out
    }

    /// Content text of the (first) tool_result for `id`.
    pub fn tool_result_for(&self, id: &str) -> Option<(String, bool)> {
        for e in self.entries().filter(|e| entry_type(e) == Some("user")) {
            for b in content_blocks(e) {
                if str_field(&b, "type") == Some("tool_result") && str_field(&b, "tool_use_id") == Some(id) {
                    let is_err = b.get("is_error").and_then(Value::as_bool).unwrap_or(false);
                    return Some((tool_result_text(b.get("content")), is_err));
                }
            }
        }
        None
    }

    /// True if any conversation entry mentions claude's "interrupted" marker.
    pub fn has_interrupted_marker(&self) -> bool {
        self.entries().filter(|e| is_conversation(e)).any(|e| {
            e.get("message").map(|m| m.to_string().contains("[Request interrupted")).unwrap_or(false)
        })
    }

    /// claude's own `<synthetic>` entries (not our glue filler).
    pub fn count_synthetic(&self) -> usize {
        self.entries().filter(|e| is_synthetic_assistant(e) && !is_glue(e)).count()
    }

    /// Entries marked `bridgeGlue`.
    pub fn count_glue(&self) -> usize {
        self.entries().filter(|e| is_glue(e)).count()
    }

    /// The conversation as IR messages, in caller terms (file order; synthetic and glue
    /// entries skipped; consecutive assistant entries of one API message merged; thinking
    /// skipped).
    /// Mid-conversation system messages are recognised by `texts.system_prefix`.
    pub fn conversation(&self, names: &ToolNameCodec, ids: &CallIdCodec, texts: &Texts) -> Vec<Message> {
        let mut out: Vec<Message> = Vec::new();
        let mut last_msg_id: Option<String> = None;
        for e in self.entries().filter(|e| is_conversation(e)) {
            if is_glue(e) {
                last_msg_id = None;
                continue;
            }
            match entry_type(e) {
                Some("assistant") => {
                    if is_synthetic_assistant(e) {
                        last_msg_id = None;
                        continue;
                    }
                    let mid = e.pointer("/message/id").and_then(Value::as_str).map(str::to_string);
                    let mut parts = Vec::new();
                    for b in content_blocks(e) {
                        match str_field(&b, "type") {
                            Some("text") => {
                                let t = str_field(&b, "text").unwrap_or_default();
                                if t != EMPTY_ASSISTANT_TEXT {
                                    parts.push(Part::Text { text: t.to_string() });
                                }
                            }
                            Some("tool_use") => parts.push(Part::ToolCall(ToolCall {
                                id: ids.decode(str_field(&b, "id").unwrap_or_default()),
                                name: names.decode_claude_name(str_field(&b, "name").unwrap_or_default()),
                                arguments: b.get("input").map(Value::to_string).unwrap_or_else(|| "{}".into()),
                            })),
                            _ => {}
                        }
                    }
                    let merge = mid.is_some()
                        && mid == last_msg_id
                        && out.last().map(|m| m.role == Role::Assistant).unwrap_or(false);
                    if merge {
                        if let Some(last) = out.last_mut() {
                            last.parts.extend(parts);
                        }
                    } else {
                        out.push(Message { role: Role::Assistant, parts });
                    }
                    last_msg_id = mid;
                }
                _ => {
                    last_msg_id = None;
                    let mut user_parts = Vec::new();
                    for b in content_blocks(e) {
                        match str_field(&b, "type") {
                            Some("tool_result") => out.push(Message {
                                role: Role::Tool,
                                parts: vec![Part::ToolResult(ToolResult {
                                    call_id: ids.decode(str_field(&b, "tool_use_id").unwrap_or_default()),
                                    content: tool_result_text(b.get("content")),
                                })],
                            }),
                            Some("text") => user_parts.push(Part::Text { text: str_field(&b, "text").unwrap_or_default().to_string() }),
                            Some("image") => {
                                let src = b.get("source").cloned().unwrap_or(Value::Null);
                                let img = match str_field(&src, "type") {
                                    Some("url") => Image {
                                        media_type: String::new(),
                                        source: ImageSource::Url { url: str_field(&src, "url").unwrap_or_default().into() },
                                    },
                                    _ => Image {
                                        media_type: str_field(&src, "media_type").unwrap_or_default().into(),
                                        source: ImageSource::Base64 { data: str_field(&src, "data").unwrap_or_default().into() },
                                    },
                                };
                                user_parts.push(Part::Image(img));
                            }
                            _ => {}
                        }
                    }
                    if !user_parts.is_empty() {
                        let mut role = Role::User;
                        if let Some(Part::Text { text }) = user_parts.first_mut() {
                            let prefix = texts.system_prefix.as_str();
                            if let Some(rest) = text.strip_prefix(prefix).filter(|_| !prefix.is_empty()) {
                                *text = rest.to_string();
                                role = Role::System;
                            }
                        }
                        out.push(Message { role, parts: user_parts });
                    }
                }
            }
        }
        out
    }
}

/// Copies `src` to `dst` with every `sessionId` rewritten to `new_sid` (fork by copy).
pub fn fork_copy(src: &Path, dst: &Path, new_sid: &str) -> std::io::Result<SessionFile> {
    let mut f = SessionFile::load(src)?;
    f.set_session_id(new_sid);
    f.save(dst)?;
    Ok(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> EntryContext {
        EntryContext {
            session_id: "sid-1".into(),
            cwd: "/w/claude/empty".into(),
            version: "2.1.325".into(),
            model_id: DEFAULT_HISTORY_MODEL.into(),
        }
    }

    fn call(id: &str, name: &str, args: &str) -> Part {
        Part::ToolCall(ToolCall { id: id.into(), name: name.into(), arguments: args.into() })
    }

    fn result(id: &str, text: &str) -> Message {
        Message { role: Role::Tool, parts: vec![Part::ToolResult(ToolResult { call_id: id.into(), content: text.into() })] }
    }

    fn conv(f: &SessionFile) -> Vec<&Value> {
        f.entries().filter(|e| is_conversation(e)).collect()
    }

    fn history() -> Vec<Message> {
        vec![
            Message::text(Role::System, "sys prompt"),
            Message::text(Role::User, "My codename is PELICAN. Read notes.txt please."),
            Message {
                role: Role::Assistant,
                parts: vec![
                    Part::Text { text: "Reading it.".into() },
                    call("call:1", "read", r#"{"filePath":"notes.txt"}"#),
                    call("toolu_2", "todo.write", r#"{"x":1}"#),
                ],
            },
            result("call:1", "Meeting moved to Tuesday."),
            result("toolu_2", "ok"),
            Message::text(Role::System, "Be terse."),
            Message::text(Role::Assistant, "Noted."),
        ]
    }

    #[test]
    fn write_entries_match_expected_json() {
        let mut f = SessionFile::default();
        let mut names = ToolNameCodec::default();
        let mut ids = CallIdCodec::default();
        f.append_messages(&ctx(), &history(), &mut names, &mut ids, true, &Texts::default());
        let c = conv(&f);
        assert_eq!(c.len(), 5, "leading system skipped; results grouped");
        // user
        assert_eq!(c[0]["parentUuid"], Value::Null);
        assert_eq!(c[0]["type"], "user");
        assert_eq!(c[0]["message"], json!({"role": "user", "content": "My codename is PELICAN. Read notes.txt please."}));
        for k in ["isSidechain", "uuid", "timestamp", "userType", "entrypoint", "cwd", "sessionId", "version"] {
            assert!(c[0].get(k).is_some(), "missing {k}");
        }
        assert_eq!(c[0]["sessionId"], "sid-1");
        // assistant
        assert_eq!(c[1]["parentUuid"], c[0]["uuid"]);
        let m = &c[1]["message"];
        assert!(m["id"].as_str().unwrap().starts_with("msg_"));
        assert_eq!(m["type"], "message");
        assert_eq!(m["role"], "assistant");
        assert_eq!(m["model"], DEFAULT_HISTORY_MODEL);
        assert_eq!(m["stop_reason"], "tool_use");
        assert_eq!(m["usage"], json!({"input_tokens": 0, "output_tokens": 0}));
        let id1 = crate::codec::encode_call_id("call:1");
        let todo = crate::codec::encode_tool_name("todo.write");
        assert_eq!(
            m["content"],
            json!([
                {"type": "text", "text": "Reading it."},
                {"type": "tool_use", "id": id1, "name": "mcp__bridge__read", "input": {"filePath": "notes.txt"}},
                {"type": "tool_use", "id": "toolu_2", "name": format!("mcp__bridge__{todo}"), "input": {"x": 1}},
            ])
        );
        // grouped results
        assert_eq!(c[2]["parentUuid"], c[1]["uuid"]);
        assert_eq!(c[2]["sourceToolAssistantUUID"], c[1]["uuid"]);
        assert_eq!(
            c[2]["message"]["content"],
            json!([
                {"type": "tool_result", "content": "Meeting moved to Tuesday.", "tool_use_id": id1},
                {"type": "tool_result", "content": "ok", "tool_use_id": "toolu_2"},
            ])
        );
        assert_eq!(c[2]["toolUseResult"], "Meeting moved to Tuesday.");
        // later system message as user text
        assert_eq!(c[3]["message"]["content"], "[System] Be terse.");
        assert_eq!(c[4]["message"]["stop_reason"], "end_turn");
        assert_eq!(c[4]["parentUuid"], c[3]["uuid"]);

        // Read back in caller terms.
        let back = f.conversation(&names, &ids, &Texts::default());
        assert_eq!(back, history()[1..].to_vec());
    }

    #[test]
    fn images_and_empty_assistant() {
        let msgs = vec![
            Message {
                role: Role::User,
                parts: vec![
                    Part::Text { text: "look".into() },
                    Part::Image(Image { media_type: "image/jpeg".into(), source: ImageSource::Base64 { data: "AAAA".into() } }),
                    Part::Image(Image { media_type: String::new(), source: ImageSource::Url { url: "https://x/y.png".into() } }),
                ],
            },
            Message { role: Role::Assistant, parts: vec![] },
        ];
        let mut f = SessionFile::default();
        f.append_messages(&ctx(), &msgs, &mut ToolNameCodec::default(), &mut CallIdCodec::default(), true, &Texts::default());
        let c = conv(&f);
        assert_eq!(
            c[0]["message"]["content"],
            json!([
                {"type": "text", "text": "look"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}},
                {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}},
            ])
        );
        assert_eq!(c[1]["message"]["content"], json!([{"type": "text", "text": EMPTY_ASSISTANT_TEXT}]));
        let back = f.conversation(&ToolNameCodec::default(), &CallIdCodec::default(), &Texts::default());
        assert_eq!(back[0], msgs[0]);
        assert!(back[1].parts.is_empty());
    }

    fn real_tool_call_file() -> SessionFile {
        // Shape recorded from claude 2.1.325 after SIGTERM during two parallel calls
        // (tests/fixtures/session_parallel_stop.jsonl, ids shortened).
        SessionFile::parse(include_str!("../tests/fixtures/session_parallel_stop.jsonl"))
    }

    #[test]
    fn prune_dangling_leaves_one_leaf() {
        let mut f = SessionFile::default();
        f.append_messages(&ctx(), &[Message::text(Role::User, "hi"), Message::text(Role::Assistant, "yo")],
            &mut ToolNameCodec::default(), &mut CallIdCodec::default(), false, &Texts::default());
        let last = f.last_conversation_uuid().unwrap();
        // Two chained attachments hanging off the last message, plus one in the middle of the chain.
        f.push(json!({"type": "attachment", "uuid": "att-1", "parentUuid": last}));
        f.push(json!({"type": "attachment", "uuid": "att-2", "parentUuid": "att-1"}));
        f.push(json!({"type": "queue-operation"}));
        assert_eq!(f.prune_dangling(), 2);
        f.append_messages(&ctx(), &[Message::text(Role::User, "next")], &mut ToolNameCodec::default(), &mut CallIdCodec::default(), false, &Texts::default());
        let uuids: Vec<String> = f.entries().filter_map(|e| str_field(e, "uuid")).map(str::to_string).collect();
        let parents: HashSet<String> =
            f.entries().filter_map(|e| str_field(e, "parentUuid")).map(str::to_string).collect();
        let leaves: Vec<&String> = uuids.iter().filter(|u| !parents.contains(*u)).collect();
        assert_eq!(leaves.len(), 1);
        assert_eq!(f.lines.len(), 4); // 3 messages + the uuid-less queue-operation line
        // An attachment that something continues from stays.
        let mut g = SessionFile::default();
        g.append_messages(&ctx(), &[Message::text(Role::User, "hi")], &mut ToolNameCodec::default(), &mut CallIdCodec::default(), false, &Texts::default());
        let u = g.last_conversation_uuid().unwrap();
        g.push(json!({"type": "attachment", "uuid": "att-x", "parentUuid": u}));
        g.push(json!({"type": "assistant", "uuid": "a-1", "parentUuid": "att-x", "message": {"role": "assistant", "content": []}}));
        assert_eq!(g.prune_dangling(), 0);
    }

    #[test]
    fn append_continues_after_last_conversation_entry() {
        let mut f = real_tool_call_file();
        let last = f.last_conversation_uuid().unwrap();
        f.append_messages(&ctx(), &[Message::text(Role::User, "next")], &mut ToolNameCodec::default(), &mut CallIdCodec::default(), false, &Texts::default());
        let c = conv(&f);
        assert_eq!(c.last().unwrap()["parentUuid"], Value::String(last));
        assert!(f.ends_with_user());
        // Raw (unparsable) lines survive round trips.
        let f2 = SessionFile::parse(&format!("{}not json\n", f.render()));
        assert_eq!(f2.lines.len(), f.lines.len() + 1);
        assert!(f2.render().ends_with("not json\n"));
    }

    #[test]
    fn fixup_repairs_parallel_stop_layout() {
        let mut f = real_tool_call_file();
        let calls = vec![
            ClaudeCall { id: "toolu_01PkKJCGD5vZ1xGpPhdE7HRY".into(), name: "mcp__bridge__read".into(), input: json!({"filePath": "src/a.ts"}) },
            ClaudeCall { id: "toolu_014RKrfC8uJtgYANq32i7Tjj".into(), name: "mcp__bridge__read".into(), input: json!({"filePath": "src/b.ts"}) },
        ];
        let before_counts = f.tool_result_counts();
        assert_eq!(before_counts.len(), 1, "claude wrote one placeholder only");
        let r = f.ensure_tool_round(&ctx(), Some("msg_x"), &calls);
        assert_eq!(r.added_placeholders, 1);
        assert_eq!(r.added_assistant_entries, 0);
        assert!(r.relinked);
        let counts = f.tool_result_counts();
        assert_eq!(counts.len(), 2);
        assert!(counts.values().all(|&n| n == 1));
        let c = conv(&f);
        let last = *c.last().unwrap();
        let second_call = c.iter().find(|e| e.to_string().contains("toolu_014RKrfC8uJtgYANq32i7Tjj") && e["type"] == "assistant").unwrap();
        assert_eq!(last["parentUuid"], second_call["uuid"]);
        assert_eq!(last["sourceToolAssistantUUID"], second_call["uuid"]);
        let ids: Vec<_> = last["message"]["content"].as_array().unwrap().iter().map(|b| b["tool_use_id"].clone()).collect();
        assert_eq!(ids, vec![json!("toolu_01PkKJCGD5vZ1xGpPhdE7HRY"), json!("toolu_014RKrfC8uJtgYANq32i7Tjj")]);
        // Idempotent.
        let r2 = f.ensure_tool_round(&ctx(), Some("msg_x"), &calls);
        assert!(!r2.changed(), "{r2:?}");
        // Patch the placeholders.
        let mut res = HashMap::new();
        res.insert("toolu_01PkKJCGD5vZ1xGpPhdE7HRY".to_string(), "export const A = 1;".to_string());
        res.insert("toolu_014RKrfC8uJtgYANq32i7Tjj".to_string(), "export const B = 2;".to_string());
        assert_eq!(f.patch_results(&res).len(), 2);
        let last = conv(&f).last().cloned().unwrap().clone();
        assert!(last["message"]["content"].as_array().unwrap().iter().all(|b| b.get("is_error").is_none()));
        assert_eq!(last["toolUseResult"], "export const A = 1;");
        assert!(last.get("toolDenialKind").is_none());
        assert_eq!(f.tool_result_for("toolu_014RKrfC8uJtgYANq32i7Tjj"), Some(("export const B = 2;".into(), false)));
    }

    #[test]
    fn fixup_writes_missing_tool_use_and_placeholder() {
        let mut f = SessionFile::default();
        f.append_messages(&ctx(), &[Message::text(Role::User, "hi")], &mut ToolNameCodec::default(), &mut CallIdCodec::default(), false, &Texts::default());
        let user_uuid = f.last_conversation_uuid().unwrap();
        let calls = vec![ClaudeCall { id: "toolu_X".into(), name: "mcp__bridge__read".into(), input: json!({}) }];
        let r = f.ensure_tool_round(&ctx(), Some("msg_1"), &calls);
        assert_eq!((r.added_assistant_entries, r.added_placeholders), (1, 1));
        let c = conv(&f);
        assert_eq!(c.len(), 3);
        assert_eq!(c[1]["parentUuid"], Value::String(user_uuid));
        assert_eq!(c[1]["message"]["id"], "msg_1");
        assert_eq!(c[2]["parentUuid"], c[1]["uuid"]);
        assert_eq!(c[2]["message"]["content"][0]["content"], PLACEHOLDER_TEXT);
        assert_eq!(c[2]["message"]["content"][0]["is_error"], true);
    }

    #[test]
    fn strip_fillers_relinks() {
        let raw = r#"{"type":"user","uuid":"u1","parentUuid":null,"message":{"role":"user","content":"q"}}
{"type":"assistant","uuid":"a1","parentUuid":"u1","message":{"model":"m","content":[{"type":"tool_use","id":"t1","name":"mcp__bridge__read","input":{}}]}}
{"type":"user","uuid":"r1","parentUuid":"a1","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"x"}]}}
{"type":"attachment","uuid":"at1","parentUuid":"r1"}
{"type":"assistant","uuid":"s1","parentUuid":"at1","message":{"model":"<synthetic>","content":[{"type":"text","text":"No response requested."}]}}
{"type":"user","uuid":"c1","parentUuid":"s1","message":{"role":"user","content":"Continue."}}
{"type":"attachment","uuid":"at2","parentUuid":"c1"}
{"type":"assistant","uuid":"a2","parentUuid":"at2","message":{"model":"m","content":[{"type":"text","text":"done"}]}}
{"type":"user","uuid":"c0","parentUuid":"a2","message":{"role":"user","content":"Continue."}}
{"type":"last-prompt","leafUuid":"c1"}
"#;
        let trigger = serde_json::to_string(TRIGGER_TEXT).unwrap();
        let text = &raw.replace("\"Continue.\"", &trigger);
        let mut f = SessionFile::parse(text);
        let protected: HashSet<String> = ["c0".to_string()].into_iter().collect();
        assert_eq!(f.strip_fillers(&protected, true, false, &Texts::default()), 2);
        let get = |u: &str| f.entries().find(|e| e["uuid"] == u).cloned();
        assert!(get("s1").is_none() && get("c1").is_none());
        assert_eq!(get("at2").unwrap()["parentUuid"], "at1");
        assert!(get("c0").is_some(), "protected user 'Continue.' kept");
        let lp = f.entries().find(|e| e["type"] == "last-prompt").unwrap();
        assert_eq!(lp["leafUuid"], "at1");
        // Without strip_trigger only the synthetic entry goes.
        let mut g = SessionFile::parse(text);
        assert_eq!(g.strip_fillers(&HashSet::new(), false, false, &Texts::default()), 1);
        assert!(g.entries().any(|e| e["uuid"] == "c1"));
        assert_eq!(g.entries().find(|e| e["uuid"] == "c1").unwrap()["parentUuid"], "at1");
        // keep_glue: unmarked (legacy) entries are stripped exactly the same way.
        let mut h = SessionFile::parse(text);
        assert_eq!(h.strip_fillers(&protected, true, true, &Texts::default()), 2);
        assert_eq!(h.entries().find(|e| e["uuid"] == "at2").unwrap()["parentUuid"], "at1");
    }

    /// A file after a ContinueAfterResults run: history with a tool round, our glue filler,
    /// then what claude writes: the stdin trigger entry, an attachment, its own synthetic
    /// filler (should not happen, must still go) and the answer.
    fn glued_run() -> (SessionFile, HashSet<String>) {
        let mut f = SessionFile::default();
        let msgs = vec![
            Message::text(Role::User, "Read a.txt"),
            Message { role: Role::Assistant, parts: vec![call("toolu_1", "read", r#"{"filePath":"a.txt"}"#)] },
            result("toolu_1", "A"),
        ];
        f.append_messages(&ctx(), &msgs, &mut ToolNameCodec::default(), &mut CallIdCodec::default(), false, &Texts::default());
        f.append_bridge_filler(&ctx(), BRIDGE_FILLER_TEXT);
        let pre = f.uuids();
        let glue_a = f.last_conversation_uuid().unwrap();
        let t = serde_json::to_string(TRIGGER_TEXT).unwrap();
        let tail = format!(
            r#"{{"type":"user","uuid":"trig","parentUuid":"{glue_a}","message":{{"role":"user","content":{t}}}}}
{{"type":"attachment","uuid":"att","parentUuid":"trig"}}
{{"type":"assistant","uuid":"syn","parentUuid":"att","message":{{"model":"<synthetic>","content":[{{"type":"text","text":"No response requested."}}]}}}}
{{"type":"assistant","uuid":"ans","parentUuid":"syn","message":{{"id":"msg_a","model":"m","content":[{{"type":"text","text":"It says A."}}]}}}}
"#
        );
        for l in SessionFile::parse(&tail).lines {
            f.lines.push(l);
        }
        (f, pre)
    }

    #[test]
    fn glue_filler_is_marked() {
        let (f, _) = glued_run();
        let filler = f.entries().find(|e| is_synthetic_assistant(e) && is_glue(e)).unwrap();
        assert_eq!(filler[GLUE_KEY], true);
        assert_eq!(filler["message"]["content"][0]["text"], BRIDGE_FILLER_TEXT);
        assert_eq!(f.count_glue(), 1);
        assert_eq!(f.count_synthetic(), 1, "claude's own filler, not ours");
    }

    #[test]
    fn keep_glue_marks_trigger_keeps_glue_strips_claude_filler() {
        let (mut f, pre) = glued_run();
        assert!(f.mark_trigger_glue(&pre, &Texts::default()));
        assert!(!f.mark_trigger_glue(&pre, &Texts::default()), "only one trigger entry per run");
        assert_eq!(f.entries().find(|e| e["uuid"] == "trig").unwrap()[GLUE_KEY], true);
        assert_eq!(f.strip_fillers(&pre, true, true, &Texts::default()), 1);
        assert_eq!(f.count_glue(), 2);
        assert_eq!(f.count_synthetic(), 0);
        let get = |u: &str| f.entries().find(|e| e["uuid"] == u).cloned().unwrap();
        assert_eq!(get("ans")["parentUuid"], "att", "relinked past claude's filler");
        // Glue never shows up in the caller-side conversation.
        let back = f.conversation(&ToolNameCodec::default(), &CallIdCodec::default(), &Texts::default());
        assert_eq!(back.len(), 4, "{back:?}");
        assert_eq!(back[3].text_content(), "It says A.");
        assert!(!back.iter().any(|m| m.text_content().contains("system-reminder") || m.text_content() == BRIDGE_FILLER_TEXT));
        // Single chain: every conversation entry's parent exists.
        let uuids = f.uuids();
        for e in f.entries() {
            if let Some(p) = str_field(e, "parentUuid") {
                assert!(uuids.contains(p), "dangling parent {p}");
            }
        }
    }

    #[test]
    fn keep_glue_false_strips_everything_like_v2() {
        let (mut f, pre) = glued_run();
        f.mark_trigger_glue(&pre, &Texts::default());
        // The filler uuid is in `pre` (protected) but glue goes regardless.
        assert_eq!(f.strip_fillers(&pre, true, false, &Texts::default()), 3);
        assert_eq!(f.count_glue(), 0);
        assert_eq!(f.count_synthetic(), 0);
        let ans = f.entries().find(|e| e["uuid"] == "ans").unwrap();
        assert_eq!(ans["parentUuid"], "att");
        let att = f.entries().find(|e| e["uuid"] == "att").unwrap();
        let results = f.entries().find(|e| e.get("toolUseResult").is_some()).unwrap();
        assert_eq!(att["parentUuid"], results["uuid"], "relinked past filler and trigger");
        // Unmarked trigger (marking skipped) is stripped too.
        let (mut g, pre) = glued_run();
        assert_eq!(g.strip_fillers(&pre, true, false, &Texts::default()), 3);
    }

    #[test]
    fn mark_trigger_ignores_pre_existing_and_prefers_glue_child() {
        let t = serde_json::to_string(TRIGGER_TEXT).unwrap();
        let raw = format!(
            r#"{{"type":"user","uuid":"old","parentUuid":null,"message":{{"role":"user","content":{t}}}}}
{{"type":"assistant","uuid":"g","parentUuid":"old","bridgeGlue":true,"message":{{"model":"<synthetic>","content":[{{"type":"text","text":"x"}}]}}}}
{{"type":"user","uuid":"stray","parentUuid":null,"message":{{"role":"user","content":[{{"type":"text","text":{t}}}]}}}}
{{"type":"user","uuid":"new","parentUuid":"g","message":{{"role":"user","content":[{{"type":"text","text":{t}}}]}}}}
"#
        );
        let mut f = SessionFile::parse(&raw);
        let pre: HashSet<String> = ["old".to_string(), "g".to_string()].into_iter().collect();
        assert!(f.mark_trigger_glue(&pre, &Texts::default()));
        let marked: Vec<String> = f.entries().filter(|e| is_glue(e) && e["type"] == "user").map(|e| e["uuid"].as_str().unwrap().to_string()).collect();
        assert_eq!(marked, vec!["new".to_string()]);
        // A user message with other text is never marked.
        let mut g = SessionFile::default();
        g.append_messages(&ctx(), &[Message::text(Role::User, "Continue.")], &mut ToolNameCodec::default(), &mut CallIdCodec::default(), false, &Texts::default());
        assert!(!g.mark_trigger_glue(&HashSet::new(), &Texts::default()));
    }

    /// Request N+1 = request N + new tail: freezing a glued round and materializing the
    /// next round (prune, patch, filler) leaves every entry up to the call message as is.
    #[test]
    fn next_round_extends_previous_bytes() {
        let (mut f, pre) = glued_run();
        // Round N ended in a tool call instead of an answer: drop "ans", add a call + placeholder.
        f.lines.retain(|l| !matches!(l, Line::Json(v) if v["uuid"] == "ans"));
        f.push(json!({"type":"assistant","uuid":"call2","parentUuid":"syn","message":{"id":"msg_2","model":"m","content":[{"type":"tool_use","id":"toolu_2","name":"mcp__bridge__read","input":{}}]}}));
        f.push(json!({"type":"attachment","uuid":"att-end","parentUuid":"call2"}));
        f.push(json!({"type":"user","uuid":"ph","parentUuid":"call2","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_2","content":PLACEHOLDER_TEXT,"is_error":true}]}}));
        let calls = vec![ClaudeCall { id: "toolu_2".into(), name: "mcp__bridge__read".into(), input: json!({}) }];
        assert!(!f.ensure_tool_round(&ctx(), Some("msg_2"), &calls).changed());
        f.mark_trigger_glue(&pre, &Texts::default());
        f.strip_fillers(&pre, true, true, &Texts::default());
        let frozen = f.clone();
        let call_idx = frozen.lines.iter().position(|l| matches!(l, Line::Json(v) if v["uuid"] == "call2")).unwrap();

        // Materialize round N+1.
        let mut next = frozen.clone();
        next.set_session_id("sid-1");
        next.prune_dangling();
        let mut res = HashMap::new();
        res.insert("toolu_2".to_string(), "B".to_string());
        assert_eq!(next.patch_results(&res).len(), 1);
        next.append_bridge_filler(&ctx(), BRIDGE_FILLER_TEXT);
        let conv_before: Vec<&Value> = frozen.entries().filter(|e| is_conversation(e)).take_while(|e| e["uuid"] != "ph").collect();
        let conv_after: Vec<&Value> = next.entries().filter(|e| is_conversation(e)).take(conv_before.len()).collect();
        assert_eq!(conv_before, conv_after);
        for (a, b) in frozen.lines[..=call_idx].iter().zip(next.lines.iter()) {
            if let (Line::Json(a), Line::Json(b)) = (a, b) {
                if a["type"] == "attachment" && a["uuid"] == "att-end" {
                    continue;
                }
                assert_eq!(a, b);
            }
        }
        // The new tail: patched result, then the next glue filler parented to it.
        let tail: Vec<&Value> = next.entries().filter(|e| is_conversation(e)).skip(conv_before.len()).collect();
        assert_eq!(tail.len(), 2);
        assert_eq!(tail[0]["uuid"], "ph");
        assert_eq!(tail[1]["parentUuid"], "ph");
        assert!(is_glue(tail[1]));
    }

    fn custom_texts() -> Texts {
        Texts::from_prompts(&BackendPrompts {
            continue_trigger: Some("[runtime] go on".into()),
            glue_filler: Some("Checking the results.".into()),
            system_message_prefix: Some("(sys) ".into()),
            ..BackendPrompts::default()
        })
    }

    #[test]
    fn texts_resolve_against_builtins() {
        let d = Texts::default();
        assert_eq!(d.continue_trigger, TRIGGER_TEXT);
        assert_eq!(d.glue_filler, BRIDGE_FILLER_TEXT);
        assert_eq!(d.system_prefix, SYSTEM_PREFIX);
        assert_eq!(d.transcript_header, crate::transcript::HEADER);
        assert_eq!(d.system_prompt(""), format!("{}{}", crate::turn::DEFAULT_SYSTEM, crate::turn::ENVIRONMENT_NOTE));
        assert_eq!(d.system_prompt("be brief"), format!("be brief{}", crate::turn::ENVIRONMENT_NOTE));
        // Custom default system + runtime note.
        let t = Texts::from_prompts(&BackendPrompts {
            default_system: Some("You are Parsley.".into()),
            runtime_note: Some("Ignore the sandbox.".into()),
            ..BackendPrompts::default()
        });
        assert_eq!(t.system_prompt("  "), "You are Parsley.\n\nIgnore the sandbox.");
        assert_eq!(t.system_prompt("caller"), "caller\n\nIgnore the sandbox.");
        // Empty note disables it; blank default system falls back to the built-in one.
        let t = Texts::from_prompts(&BackendPrompts {
            default_system: Some(" ".into()),
            runtime_note: Some(String::new()),
            ..BackendPrompts::default()
        });
        assert_eq!(t.system_prompt(""), crate::turn::DEFAULT_SYSTEM);
        // A custom filler is named by the built-in runtime note.
        let c = custom_texts();
        assert!(c.runtime_note.contains("Checking the results.") && !c.runtime_note.contains(BRIDGE_FILLER_TEXT));
    }

    #[test]
    fn custom_system_prefix_round_trips() {
        let texts = custom_texts();
        let msgs = vec![Message::text(Role::User, "hi"), Message::text(Role::System, "Be terse."), Message::text(Role::Assistant, "ok")];
        let mut f = SessionFile::default();
        f.append_messages(&ctx(), &msgs, &mut ToolNameCodec::default(), &mut CallIdCodec::default(), false, &texts);
        assert_eq!(conv(&f)[1]["message"]["content"], "(sys) Be terse.");
        assert_eq!(f.conversation(&ToolNameCodec::default(), &CallIdCodec::default(), &texts), msgs);
        // An empty prefix never turns user text into system messages.
        let none = Texts { system_prefix: String::new(), ..Texts::default() };
        let back = f.conversation(&ToolNameCodec::default(), &CallIdCodec::default(), &none);
        assert_eq!(back[0].role, Role::User);
    }

    /// `glued_run` with the backend's custom trigger and filler.
    fn custom_glued_run(texts: &Texts) -> (SessionFile, HashSet<String>) {
        let mut f = SessionFile::default();
        let msgs = vec![
            Message::text(Role::User, "Read a.txt"),
            Message { role: Role::Assistant, parts: vec![call("toolu_1", "read", r#"{"filePath":"a.txt"}"#)] },
            result("toolu_1", "A"),
        ];
        f.append_messages(&ctx(), &msgs, &mut ToolNameCodec::default(), &mut CallIdCodec::default(), false, texts);
        f.append_bridge_filler(&ctx(), &texts.glue_filler);
        let pre = f.uuids();
        let glue_a = f.last_conversation_uuid().unwrap();
        let t = serde_json::to_string(&texts.continue_trigger).unwrap();
        let tail = format!(
            r#"{{"type":"user","uuid":"trig","parentUuid":"{glue_a}","message":{{"role":"user","content":{t}}}}}
{{"type":"assistant","uuid":"syn","parentUuid":"trig","message":{{"model":"<synthetic>","content":[{{"type":"text","text":"{CLAUDE_FILLER_TEXT}"}}]}}}}
{{"type":"assistant","uuid":"ans","parentUuid":"syn","message":{{"id":"msg_a","model":"m","content":[{{"type":"text","text":"It says A."}}]}}}}
"#
        );
        f.lines.extend(SessionFile::parse(&tail).lines);
        (f, pre)
    }

    #[test]
    fn custom_trigger_and_filler_are_marked_kept_and_stripped() {
        let texts = custom_texts();
        let (mut f, pre) = custom_glued_run(&texts);
        let filler = f.entries().find(|e| is_glue(e)).unwrap();
        assert_eq!(filler["message"]["content"][0]["text"], "Checking the results.");
        // The built-in trigger text is not what this backend sent: not marked.
        assert!(!f.clone().mark_trigger_glue(&pre, &Texts::default()));
        assert!(f.mark_trigger_glue(&pre, &texts));
        assert_eq!(f.entries().find(|e| e["uuid"] == "trig").unwrap()[GLUE_KEY], true);
        // keep_glue: our filler and the trigger stay; claude's own filler goes.
        let mut kept = f.clone();
        assert_eq!(kept.strip_fillers(&pre, true, true, &texts), 1);
        assert_eq!(kept.count_glue(), 2);
        assert_eq!(kept.count_synthetic(), 0);
        let back = kept.conversation(&ToolNameCodec::default(), &CallIdCodec::default(), &texts);
        assert_eq!(back.len(), 4);
        assert!(!back.iter().any(|m| m.text_content().contains("go on") || m.text_content().contains("Checking")));
        // keep_glue off: everything goes, chain relinked to the tool results.
        assert_eq!(f.strip_fillers(&pre, true, false, &texts), 3);
        assert_eq!(f.count_glue(), 0);
        let results = f.entries().find(|e| e.get("toolUseResult").is_some()).unwrap()["uuid"].clone();
        assert_eq!(f.entries().find(|e| e["uuid"] == "ans").unwrap()["parentUuid"], results);
        // Unmarked custom trigger written during the run is stripped too.
        let (mut g, pre) = custom_glued_run(&texts);
        assert_eq!(g.strip_fillers(&pre, true, true, &texts), 2);
        assert!(!g.entries().any(|e| e["uuid"] == "trig"));
    }

    #[test]
    fn legacy_triggers_and_claude_filler_still_stripped_with_custom_texts() {
        let texts = custom_texts();
        for legacy in [LEGACY_TRIGGER_TEXT, TRIGGER_TEXT] {
            let raw = format!(
                r#"{{"type":"user","uuid":"u1","parentUuid":null,"message":{{"role":"user","content":"q"}}}}
{{"type":"assistant","uuid":"s1","parentUuid":"u1","message":{{"model":"<synthetic>","content":[{{"type":"text","text":"{CLAUDE_FILLER_TEXT}"}}]}}}}
{{"type":"user","uuid":"c1","parentUuid":"s1","message":{{"role":"user","content":{}}}}}
{{"type":"assistant","uuid":"a2","parentUuid":"c1","message":{{"model":"m","content":[{{"type":"text","text":"done"}}]}}}}
"#,
                serde_json::to_string(legacy).unwrap()
            );
            let mut f = SessionFile::parse(&raw);
            assert_eq!(f.strip_fillers(&HashSet::new(), true, true, &texts), 2, "{legacy}");
            assert_eq!(f.entries().find(|e| e["uuid"] == "a2").unwrap()["parentUuid"], "u1");
            // Marking only accepts the configured trigger.
            let mut g = SessionFile::parse(&raw);
            assert!(!g.mark_trigger_glue(&HashSet::new(), &texts));
        }
    }

    #[test]
    fn fork_rewrites_session_id() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("a.jsonl");
        real_tool_call_file().save(&src).unwrap();
        let before = std::fs::read(&src).unwrap();
        let dst = dir.path().join("b.jsonl");
        let f = fork_copy(&src, &dst, "new-sid").unwrap();
        for e in f.entries() {
            if let Some(s) = e.get("sessionId") {
                assert_eq!(s, "new-sid");
            }
        }
        assert!(f.entries().filter(|e| e.get("sessionId").is_some()).count() > 3);
        assert_eq!(std::fs::read(&src).unwrap(), before, "source untouched");
        assert_eq!(SessionFile::load(&dst).unwrap(), f);
    }

    #[test]
    fn slug_matches_claude() {
        assert_eq!(project_slug(Path::new("/private/tmp/cb_probe/lab/empty")), "-private-tmp-cb-probe-lab-empty");
        assert_eq!(project_slug(Path::new("/Users/a.b/x y")), "-Users-a-b-x-y");
    }

    #[test]
    fn arguments_mapping() {
        assert_eq!(arguments_to_input(""), json!({}));
        assert_eq!(arguments_to_input(r#"{"a":1}"#), json!({"a": 1}));
        assert_eq!(arguments_to_input("[1]"), json!({"input": [1]}));
        assert_eq!(arguments_to_input("nope"), json!({"input": "nope"}));
    }
}
