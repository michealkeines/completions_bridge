//! The backend interface. A backend turns one planned turn into a stream of
//! `TurnEvent`s and can freeze the resulting state as an immutable checkpoint.
//!
//! Contract highlights (see ARCHITECTURE.md sections 3, 7, 8):
//! - Everything is in caller terms: caller tool names, caller-visible call ids.
//!   Backends translate to and from their own naming internally.
//! - A turn never mutates the base checkpoint. It works on a private copy.
//! - `freeze` is only called after the event stream ended with `Done` and the
//!   engine decided the caller received the output unaltered.

use std::fmt;

use async_trait::async_trait;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};

use crate::ir::{Message, ReasoningEffort, ToolSpec, TurnEvent};
use crate::BridgeError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointKind {
    /// Ends with an assistant final answer; next turn adds a user message.
    Idle,
    /// Ends with an assistant message holding tool calls (plus placeholders);
    /// next turn must start with the results for `pending_call_ids`.
    Awaiting,
}

/// What the backend reports when a workspace is frozen.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnapshotInfo {
    /// Absolute path of the frozen snapshot (owned by the backend's snapshot dir).
    pub path: String,
    pub kind: CheckpointKind,
    /// Caller-visible ids of the tool calls the snapshot is waiting for (Awaiting only).
    pub pending_call_ids: Vec<String>,
    pub size_bytes: u64,
    /// Backend version that wrote the snapshot; snapshots from other versions are ignored.
    pub backend_version: String,
}

/// A checkpoint handed back to a backend as the base of a new turn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointRef {
    pub backend: String,
    pub snapshot: SnapshotInfo,
}

/// How the turn is started once the base and appended history are in place.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Trigger {
    /// The newest user message. It is NOT part of `append`; the backend sends it as the
    /// live input (ARCHITECTURE 0.1, P7: never write it into history and send "Continue.").
    User { message: Message },
    /// History ends with tool results (or an assistant prefill); the backend continues
    /// with its fixed internal trigger.
    ContinueAfterResults,
}

/// Everything a backend needs to run one turn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TurnSpec {
    /// Backend model name (e.g. "haiku"), already resolved from the caller's model id.
    pub model: String,
    /// System prompt to use: caller system message(s) plus policy instructions.
    pub system: String,
    /// Tools offered this turn, caller names and schemas.
    pub tools: Vec<ToolSpec>,
    /// Checkpoint to fork from; `None` = empty history.
    pub base: Option<CheckpointRef>,
    /// Messages to write into history after the base, in order, before triggering.
    /// May contain user, assistant (text and/or tool calls) and tool-result messages.
    /// If `base` is Awaiting and `append` starts with Tool messages answering exactly its
    /// `pending_call_ids`, the backend fills its placeholders with them instead of appending.
    /// System messages after the first are delivered as user-visible instruction text.
    pub append: Vec<Message>,
    pub trigger: Trigger,
    pub effort: Option<ReasoningEffort>,
    /// JSON Schema the final answer must satisfy, if the backend supports it natively.
    pub json_schema: Option<serde_json::Value>,
    /// True for throwaway requests (titles etc.): nothing will be frozen, the backend
    /// may skip persistence entirely.
    pub ephemeral: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendMode {
    /// Checkpoints, forks and history writing all verified.
    Native,
    /// Self-test failed: every turn starts fresh with the history rendered as text.
    Transcript,
    /// Backend not usable at all (e.g. not logged in).
    Disabled,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Capabilities {
    pub images: bool,
    pub json_schema: bool,
    pub effort: bool,
    /// Context window in tokens.
    pub max_context: u64,
}

/// Control half of a running turn.
#[async_trait]
pub trait TurnControl: Send {
    /// Stop generation now (client abort, max_tokens, stop sequence). Idempotent.
    async fn abort(&mut self);
    /// Turn the private workspace into an immutable snapshot. Only valid after the
    /// event stream ended with `Done`. Removes any internal filler/trigger entries.
    async fn freeze(self: Box<Self>) -> Result<SnapshotInfo, BridgeError>;
    /// Throw the workspace away.
    async fn discard(self: Box<Self>);
}

/// A running turn: an event stream plus its control handle.
pub struct Turn {
    pub events: BoxStream<'static, TurnEvent>,
    pub control: Box<dyn TurnControl>,
}

impl fmt::Debug for Turn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Turn { .. }")
    }
}

#[async_trait]
pub trait Backend: Send + Sync {
    /// Configured name, e.g. "claude".
    fn name(&self) -> &str;
    fn capabilities(&self) -> Capabilities;
    /// Version string of the underlying tool; part of snapshot compatibility.
    fn version(&self) -> String;
    /// Current mode (set by `self_test`).
    fn mode(&self) -> BackendMode;
    /// Verify the backend's session tricks; called at startup and when the version changes.
    async fn self_test(&self) -> BackendMode;
    /// Start a turn. Errors returned here happen before any output (failover allowed).
    async fn start_turn(&self, spec: TurnSpec) -> Result<Turn, BridgeError>;
    /// Delete a snapshot written by this backend (checkpoint GC).
    async fn delete_snapshot(&self, snapshot: &SnapshotInfo);
}
