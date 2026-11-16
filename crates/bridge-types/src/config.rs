//! Configuration file schema (YAML). Parsed by the binary; shared by all crates.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub server: ServerConfig,
    /// Caller-visible model ids → routing.
    pub models: BTreeMap<String, ModelRoute>,
    /// Backend name → settings.
    pub backends: BTreeMap<String, BackendConfig>,
    pub checkpoints: CheckpointConfig,
    pub output: OutputConfig,
    pub engine: EngineConfig,
}

/// Engine tuning. Every value is today's built-in default.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EngineConfig {
    /// How long a request may wait for a deeper checkpoint that is still being saved
    /// (pending checkpoint) before planning from an older one (ARCHITECTURE 17.1).
    pub pending_wait_ms: u64,
    /// Minimum time between checkpoint TTL sweeps. The size cap is enforced immediately.
    pub gc_interval_s: u64,
    /// How often backend versions are re-checked (a change re-runs the self-test).
    pub version_check_interval_s: u64,
    /// How long a backend that failed authentication is skipped.
    pub auth_cooldown_s: u64,
    /// Quota backoff when a limit error carries no reset time.
    pub quota_default_backoff_s: u64,
    /// Upper bound for the doubling quota backoff.
    pub quota_max_backoff_s: u64,
    /// After the caller received Done: how long to keep draining the backend so the turn
    /// can still be frozen as a checkpoint.
    pub drain_after_done_ms: u64,
    /// SQLite busy timeout for the state database.
    pub store_busy_timeout_ms: u64,
    /// Characters per token for usage estimates (when a backend reports none) and for the
    /// max_tokens budget.
    pub chars_per_token: u32,
    /// `created` timestamp reported for every model in /v1/models (unix seconds).
    pub model_created_unix: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            pending_wait_ms: 5000,
            gc_interval_s: 60,
            version_check_interval_s: 300,
            auth_cooldown_s: 60,
            quota_default_backoff_s: 15 * 60,
            quota_max_backoff_s: 6 * 60 * 60,
            drain_after_done_ms: 30_000,
            store_busy_timeout_ms: 5000,
            chars_per_token: 4,
            model_created_unix: 1_735_689_600,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        let mut models = BTreeMap::new();
        models.insert("bridge/auto".to_string(), ModelRoute::default());
        // Explicit model ids, so an alias moving to a newer model never changes behaviour silently.
        for (id, m) in [
            ("bridge/claude-haiku", "claude-haiku-5-5"),
            ("bridge/claude-sonnet", "claude-sonnet-5-5"),
            ("bridge/claude-sonnet-5-5", "claude-sonnet-5-5"),
            ("bridge/claude-opus", "claude-opus-5-5"),
        ] {
            models.insert(
                id.to_string(),
                ModelRoute { backend: Some("claude".into()), model: Some(m.into()), context_limit: None },
            );
        }
        let mut backends = BTreeMap::new();
        backends.insert("claude".to_string(), BackendConfig::default());
        Config {
            server: ServerConfig::default(),
            models,
            backends,
            checkpoints: CheckpointConfig::default(),
            output: OutputConfig::default(),
            engine: EngineConfig::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    pub listen: String,
    /// Bearer token callers must send. `None` = no auth (still localhost only by default).
    pub token: Option<String>,
    /// Bridge state directory (database, snapshots, work dirs). `~` is expanded by the binary.
    pub workdir: PathBuf,
    pub max_body_mb: u64,
    /// Max wait for the first event of a turn before it counts as a timeout.
    pub first_token_timeout_s: u64,
    /// SSE comment ping interval once headers are sent.
    pub keepalive_ping_s: u64,
    /// For streaming: how long to hold back response headers waiting for the first event,
    /// so early failures can still be returned as plain HTTP errors (STR-12).
    pub header_wait_ms: u64,
    /// On shutdown: how long to wait for in-flight turns (e.g. checkpoints being saved).
    pub shutdown_grace_s: u64,
    /// For a too-large body: how much more to read and discard before answering 413, so the
    /// client sees the 413 instead of a connection reset.
    pub oversize_drain_mb: u64,
    /// Time limit for that drain.
    pub oversize_drain_s: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            listen: "127.0.0.1:4141".into(),
            token: None,
            workdir: PathBuf::from("~/.completions-bridge"),
            max_body_mb: 20,
            first_token_timeout_s: 120,
            keepalive_ping_s: 10,
            header_wait_ms: 8000,
            shutdown_grace_s: 30,
            oversize_drain_mb: 64,
            oversize_drain_s: 10,
        }
    }
}

/// Routing for one caller-visible model id.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelRoute {
    /// Pin to one backend; `None` = auto (router picks).
    pub backend: Option<String>,
    /// Backend model name; `None` = the backend's `default_model`.
    pub model: Option<String>,
    pub context_limit: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BackendConfig {
    /// "claude-cli" or "fake".
    pub kind: String,
    pub enabled: bool,
    /// Lower = preferred.
    pub priority: u32,
    pub max_concurrency: u32,
    /// How long a request may wait for a concurrency slot before 429.
    pub queue_wait_ms: u64,
    pub default_model: String,
    /// Model for side requests (titles etc.); `None` = default_model.
    pub side_request_model: Option<String>,
    /// Model for the startup self-test; `None` = side_request_model, then the backend's
    /// cheapest built-in choice.
    pub selftest_model: Option<String>,
    /// Path to the CLI binary; `None` = look up on PATH.
    pub binary: Option<String>,
    pub extra_args: Vec<String>,
    /// reasoning_effort → backend effort level.
    pub effort_map: BTreeMap<String, String>,
    pub context_limit: u64,
    /// Process and session tuning (CLI backends).
    pub tuning: BackendTuning,
    /// Text the backend injects (unset = built-in text).
    pub prompts: BackendPrompts,
    /// Backend-specific options (used by the fake backend for scripting).
    pub options: serde_json::Value,
}

/// Process and session tuning for CLI backends. Every value is today's built-in default.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BackendTuning {
    /// After a tool-call stop (SIGTERM): how long the CLI may take to write its state and exit.
    pub stop_grace_ms: u64,
    /// After a normal end: how long to wait for the CLI to exit before killing it.
    pub exit_grace_ms: u64,
    /// Grace period when aborting a turn (client gone, spawn problems) before SIGKILL.
    pub abort_grace_ms: u64,
    /// After SIGKILL: how long to wait for the process to be reaped.
    pub kill_wait_ms: u64,
    /// How long freeze waits for the turn's run task to finish.
    pub freeze_wait_ms: u64,
    /// How much of the CLI's stderr to keep for diagnostics.
    pub stderr_cap_bytes: usize,
    /// The CLI's own auto-compaction window; set far above the context so it never
    /// summarises on its own (callers compact; ARCHITECTURE 0.2 / Q15).
    pub autocompact_tokens: u64,
    /// Keep the bridge's glue entries in checkpoints (ARCHITECTURE 17.2). false = strip them.
    pub keep_glue: bool,
}

impl Default for BackendTuning {
    fn default() -> Self {
        BackendTuning {
            stop_grace_ms: 10_000,
            exit_grace_ms: 3_000,
            abort_grace_ms: 2_000,
            kill_wait_ms: 5_000,
            freeze_wait_ms: 15_000,
            stderr_cap_bytes: 16 * 1024,
            autocompact_tokens: 1_000_000,
            keep_glue: true,
        }
    }
}

/// Texts a backend injects into the model's context. `None` = the built-in text
/// (see the backend crate for the defaults and why they are worded as they are).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BackendPrompts {
    /// System prompt used when the caller sends none.
    pub default_system: Option<String>,
    /// Note appended to every system prompt (neutralises the CLI's own environment block).
    pub runtime_note: Option<String>,
    /// Live input after tool results (ContinueAfterResults trigger).
    pub continue_trigger: Option<String>,
    /// The bridge's own assistant filler before that trigger.
    pub glue_filler: Option<String>,
    /// Prefix for system messages that appear mid-conversation (written as user text).
    pub system_message_prefix: Option<String>,
    /// Header of the transcript rendering used in transcript (fallback) mode.
    pub transcript_header: Option<String>,
}

impl Default for BackendConfig {
    fn default() -> Self {
        let mut effort_map = BTreeMap::new();
        for (k, v) in [("minimal", "low"), ("low", "low"), ("medium", "medium"), ("high", "high")] {
            effort_map.insert(k.to_string(), v.to_string());
        }
        BackendConfig {
            kind: "claude-cli".into(),
            enabled: true,
            priority: 1,
            max_concurrency: 12,
            queue_wait_ms: 30_000,
            default_model: "claude-sonnet-5-5".into(),
            side_request_model: Some("claude-haiku-5-5".into()),
            selftest_model: None,
            binary: None,
            extra_args: Vec::new(),
            effort_map,
            context_limit: 200_000,
            tuning: BackendTuning::default(),
            prompts: BackendPrompts::default(),
            options: serde_json::Value::Null,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CheckpointConfig {
    pub max_total_mb: u64,
    pub ttl_hours: u64,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        CheckpointConfig { max_total_mb: 2048, ttl_hours: 168 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutputConfig {
    /// Send model reasoning as `delta.reasoning_content` / `message.reasoning_content`.
    pub reasoning_to_caller: bool,
    pub max_n: u32,
    /// Re-runs allowed when the output violates tool_choice / response_format.
    pub enforcement_retries: u32,
    /// How many `{...}` candidates JSON extraction tries in a non-JSON reply.
    pub json_extract_max_candidates: usize,
    /// Instruction texts appended to the system prompt (unset = built-in text).
    pub instructions: PolicyInstructions,
}

impl Default for OutputConfig {
    fn default() -> Self {
        OutputConfig {
            reasoning_to_caller: true,
            max_n: 4,
            enforcement_retries: 1,
            json_extract_max_candidates: 32,
            instructions: PolicyInstructions::default(),
        }
    }
}

/// Output-policy instruction texts. `None` = built-in text. `{name}` and `{schema}` are
/// replaced with the tool name / pretty-printed JSON schema.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PolicyInstructions {
    pub tool_choice_none: Option<String>,
    pub tool_choice_required: Option<String>,
    pub tool_choice_function: Option<String>,
    pub parallel_off: Option<String>,
    pub json_object: Option<String>,
    pub json_schema: Option<String>,
}
