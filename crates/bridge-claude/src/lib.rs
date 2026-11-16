//! Claude Code CLI backend (ARCHITECTURE.md section 8, findings in section 0.1).
//!
//! PUBLIC API IS FROZEN: `ClaudeBackend::new`, `impl Backend for ClaudeBackend`,
//! `run_park_server`.
//!
//! Layout owned by the backend under `<workdir>/claude/`:
//! - `empty/`        cwd of every claude process (kept empty)
//! - `snapshots/`    frozen checkpoints (`<uuid>.jsonl`)
//! - `turns/<sid>/`  per-turn files (tools.json, mcp.json); removed when the turn ends
//! - `selftest.json` cached self-test result per claude version
//!
//! Working session files live where claude expects them:
//! `<projects>/<slug of realpath(empty)>/<sid>.jsonl`.

pub mod codec;
pub mod errors;
pub mod park;
pub mod process;
pub mod sanitize;
pub mod selftest;
pub mod session;
pub mod stream;
pub mod transcript;
pub mod turn;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bridge_types::{
    Backend, BackendConfig, BackendMode, BridgeError, Capabilities, ErrorClass, SnapshotInfo, Turn, TurnSpec,
};

/// State shared by the backend and its running turns.
pub struct Shared {
    pub name: String,
    pub config: BackendConfig,
    pub binary: PathBuf,
    pub version: String,
    /// `<workdir>/claude`
    pub root: PathBuf,
    /// Canonical cwd for claude (`<root>/empty`).
    pub cwd: PathBuf,
    /// claude's projects root (`~/.claude/projects`).
    pub projects_dir_root: PathBuf,
    /// `<projects root>/<slug>`.
    pub projects_dir: PathBuf,
    pub snapshots_dir: PathBuf,
    pub turns_dir: PathBuf,
    pub park_command: Vec<PathBuf>,
    /// Texts injected into the model's context (`config.prompts` resolved against the
    /// built-in defaults). Per backend: two backends may use different texts.
    pub texts: session::Texts,
    mode: AtomicU8,
    stale: AtomicBool,
    history_model: Mutex<String>,
}

fn mode_to_u8(m: BackendMode) -> u8 {
    match m {
        BackendMode::Native => 0,
        BackendMode::Transcript => 1,
        BackendMode::Disabled => 2,
    }
}

fn mode_from_u8(v: u8) -> BackendMode {
    match v {
        0 => BackendMode::Native,
        1 => BackendMode::Transcript,
        _ => BackendMode::Disabled,
    }
}

impl Shared {
    pub fn session_path(&self, sid: &str) -> PathBuf {
        self.projects_dir.join(format!("{sid}.jsonl"))
    }

    pub fn selftest_path(&self) -> PathBuf {
        self.root.join("selftest.json")
    }

    pub fn mode(&self) -> BackendMode {
        mode_from_u8(self.mode.load(Ordering::SeqCst))
    }

    pub fn set_mode(&self, m: BackendMode) {
        self.mode.store(mode_to_u8(m), Ordering::SeqCst);
    }

    /// A post-run check failed: the next `self_test` ignores the cache.
    pub fn mark_stale(&self) {
        self.stale.store(true, Ordering::SeqCst);
    }

    pub fn clear_stale(&self) {
        self.stale.store(false, Ordering::SeqCst);
    }

    pub fn is_stale(&self) -> bool {
        self.stale.load(Ordering::SeqCst)
    }

    /// `tuning.keep_glue` (default true, ARCHITECTURE 17.2): keep the bridge's glue
    /// entries in frozen checkpoints so consecutive requests extend each other exactly.
    ///
    /// Deprecated: a boolean `options.keep_glue` (the setting's old home) still overrides
    /// `tuning.keep_glue` when present; a warning is logged at construction.
    pub fn keep_glue(&self) -> bool {
        legacy_keep_glue(&self.config).unwrap_or(self.config.tuning.keep_glue)
    }

    pub fn history_model(&self) -> String {
        self.history_model
            .lock()
            .map(|m| m.clone())
            .unwrap_or_else(|_| session::DEFAULT_HISTORY_MODEL.to_string())
    }

    pub fn set_history_model(&self, model: &str) {
        if model.is_empty() || model == session::SYNTHETIC_MODEL {
            return;
        }
        if let Ok(mut m) = self.history_model.lock() {
            if *m != model {
                *m = model.to_string();
            }
        }
    }
}

/// The deprecated `options.keep_glue` override, when set to a boolean.
fn legacy_keep_glue(config: &BackendConfig) -> Option<bool> {
    config.options.get("keep_glue").and_then(serde_json::Value::as_bool)
}

/// `claude --version` → first token (e.g. "2.1.325"), or `None` if it can't run.
pub fn probe_version(binary: &Path) -> Option<String> {
    let out = std::process::Command::new(binary)
        .arg("--version")
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout).split_whitespace().next().map(str::to_string)
}

pub struct ClaudeBackend {
    shared: Arc<Shared>,
}

impl ClaudeBackend {
    /// `name`: configured backend name. `workdir`: bridge state dir; the backend owns
    /// `<workdir>/claude/` (empty cwd, snapshots, per-turn files).
    /// `park_command`: argv prefix that starts the parking MCP server; the backend appends
    /// `--tools <file>` (e.g. [current_exe, "mcp-park"]).
    pub fn new(
        name: &str,
        config: &BackendConfig,
        workdir: &Path,
        park_command: Vec<PathBuf>,
    ) -> anyhow::Result<Self> {
        Self::with_projects_root(name, config, workdir, park_command, session::default_projects_root())
    }

    /// Like `new`, with an explicit claude projects root (tests use a temp dir together
    /// with a fake claude).
    pub fn with_projects_root(
        name: &str,
        config: &BackendConfig,
        workdir: &Path,
        park_command: Vec<PathBuf>,
        projects_root: PathBuf,
    ) -> anyhow::Result<Self> {
        if let Some(v) = legacy_keep_glue(config) {
            tracing::warn!(
                target: "bridge_claude",
                backend = name,
                "options.keep_glue is deprecated; use tuning.keep_glue (using {v})"
            );
        }
        let root = workdir.join("claude");
        let empty = root.join("empty");
        let snapshots_dir = root.join("snapshots");
        let turns_dir = root.join("turns");
        for d in [&empty, &snapshots_dir, &turns_dir] {
            std::fs::create_dir_all(d)?;
        }
        let cwd = std::fs::canonicalize(&empty)?;
        let projects_dir = projects_root.join(session::project_slug(&cwd));
        std::fs::create_dir_all(&projects_dir)?;
        let binary = PathBuf::from(config.binary.clone().unwrap_or_else(|| "claude".to_string()));
        let version = probe_version(&binary);
        let mut initial = if version.is_some() { BackendMode::Native } else { BackendMode::Disabled };
        let shared = Shared {
            name: name.to_string(),
            config: config.clone(),
            binary,
            version: version.clone().unwrap_or_else(|| "unknown".to_string()),
            root,
            cwd,
            projects_dir_root: projects_root,
            projects_dir,
            snapshots_dir,
            turns_dir,
            park_command,
            texts: session::Texts::from_prompts(&config.prompts),
            mode: AtomicU8::new(mode_to_u8(initial)),
            stale: AtomicBool::new(false),
            // Model id written into assistant entries of history the bridge writes itself;
            // replaced by the id claude actually reports as soon as a turn has run.
            history_model: Mutex::new(if config.default_model.trim().is_empty() {
                session::DEFAULT_HISTORY_MODEL.to_string()
            } else {
                config.default_model.clone()
            }),
        };
        if let (Some(v), Some(rec)) = (&version, selftest::load_cache(&shared)) {
            if &rec.version == v {
                initial = rec.mode;
                shared.set_mode(initial);
            }
        }
        turn::cleanup_stale_turns(&shared);
        Ok(ClaudeBackend { shared: Arc::new(shared) })
    }

    /// Forces a mode (tests; operators via config in the future).
    pub fn set_mode(&self, mode: BackendMode) {
        self.shared.set_mode(mode);
    }

    pub fn shared(&self) -> &Arc<Shared> {
        &self.shared
    }

    /// Starts a turn and returns the concrete control handle (tests, self-test).
    pub async fn start_turn_concrete(
        &self,
        spec: TurnSpec,
    ) -> Result<(futures::stream::BoxStream<'static, bridge_types::TurnEvent>, turn::ClaudeTurnControl), BridgeError> {
        let mode = match self.shared.mode() {
            BackendMode::Native => turn::RunMode::Native,
            BackendMode::Transcript => turn::RunMode::Transcript,
            BackendMode::Disabled => return Err(errors::error(ErrorClass::BackendAuth)),
        };
        turn::start(self.shared.clone(), spec, mode).await
    }
}

#[async_trait]
impl Backend for ClaudeBackend {
    fn name(&self) -> &str {
        &self.shared.name
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities { images: true, json_schema: true, effort: true, max_context: self.shared.config.context_limit }
    }

    fn version(&self) -> String {
        self.shared.version.clone()
    }

    fn mode(&self) -> BackendMode {
        self.shared.mode()
    }

    async fn self_test(&self) -> BackendMode {
        selftest::run(self.shared.clone()).await
    }

    async fn start_turn(&self, spec: TurnSpec) -> Result<Turn, BridgeError> {
        let (events, control) = self.start_turn_concrete(spec).await?;
        Ok(Turn { events, control: Box::new(control) })
    }

    async fn delete_snapshot(&self, snapshot: &SnapshotInfo) {
        let p = Path::new(&snapshot.path);
        let inside = match (std::fs::canonicalize(p), std::fs::canonicalize(&self.shared.snapshots_dir)) {
            (Ok(a), Ok(b)) => a.starts_with(b),
            _ => false,
        };
        if inside {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// Stdio MCP server that advertises the tools in `tools_path` (JSON array of
/// `{name, description, inputSchema}` objects with already-encoded names; `parameters` is
/// accepted instead of `inputSchema`) and never answers tool calls. Runs until stdin closes.
pub async fn run_park_server(tools_path: &Path) -> anyhow::Result<()> {
    let tools = park::load_tools(tools_path);
    park::serve(tools, tokio::io::stdin(), tokio::io::stdout()).await
}
