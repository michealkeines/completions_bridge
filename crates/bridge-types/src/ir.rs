//! Canonical intermediate representation (IR) of a chat request and of model output.
//!
//! Frontends convert their wire format into these types; backends consume them.
//! Nothing here knows about OpenAI JSON or claude session files.

use serde::{Deserialize, Serialize};

/// A fully validated chat request, as the engine sees it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Model id exactly as the caller sent it (e.g. "bridge/auto"); echoed back in responses.
    pub model: String,
    /// Full conversation, in order. System/developer messages stay in place (role `System`).
    pub messages: Vec<Message>,
    /// Tools the caller offers for this request (may be empty).
    pub tools: Vec<ToolSpec>,
    pub tool_choice: ToolChoice,
    /// `parallel_tool_calls` from the request; default true.
    pub parallel_tool_calls: bool,
    pub stream: bool,
    /// `stream_options.include_usage`.
    pub include_usage: bool,
    /// `max_completion_tokens` or `max_tokens`.
    pub max_tokens: Option<u32>,
    /// Stop sequences (0..=4).
    pub stop: Vec<String>,
    /// Number of choices (>= 1).
    pub n: u32,
    pub response_format: ResponseFormat,
    pub reasoning_effort: Option<ReasoningEffort>,
}

impl ChatRequest {
    /// Convenience constructor with OpenAI defaults.
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        ChatRequest {
            model: model.into(),
            messages,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: true,
            stream: false,
            include_usage: false,
            max_tokens: None,
            stop: Vec::new(),
            n: 1,
            response_format: ResponseFormat::Text,
            reasoning_effort: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// `system` and `developer` both map here.
    System,
    User,
    Assistant,
    /// A tool result message; its parts are `Part::ToolResult`.
    Tool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub parts: Vec<Part>,
}

impl Message {
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Message { role, parts: vec![Part::Text { text: text.into() }] }
    }

    /// Concatenated text of all `Text` parts.
    pub fn text_content(&self) -> String {
        self.parts
            .iter()
            .filter_map(|p| match p {
                Part::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn tool_calls(&self) -> Vec<&ToolCall> {
        self.parts
            .iter()
            .filter_map(|p| match p {
                Part::ToolCall(c) => Some(c),
                _ => None,
            })
            .collect()
    }

    pub fn tool_results(&self) -> Vec<&ToolResult> {
        self.parts
            .iter()
            .filter_map(|p| match p {
                Part::ToolResult(r) => Some(r),
                _ => None,
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Part {
    Text { text: String },
    /// Model reasoning ("thinking"). Never sent back to a backend as text.
    Reasoning { text: String },
    Image(Image),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Image {
    /// e.g. "image/png". Empty when only a URL is known.
    pub media_type: String,
    pub source: ImageSource,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageSource {
    /// Base64 data (no `data:` prefix).
    Base64 { data: String },
    Url { url: String },
}

/// A tool call as the caller sees it: caller's tool name, caller-visible id,
/// arguments as a JSON string (exactly what goes into `function.arguments`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: String,
    /// Text content of the result (array parts already joined).
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Caller's tool name, verbatim.
    pub name: String,
    pub description: String,
    /// JSON Schema of the arguments (defaults to `{"type":"object"}`).
    pub parameters: serde_json::Value,
    pub strict: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Function { name: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        schema: serde_json::Value,
        strict: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
}

// ---------------------------------------------------------------- output

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Total input tokens the model processed (uncached + cache read + cache write).
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// Part of `prompt_tokens` served from the prompt cache.
    pub cached_tokens: u64,
    pub reasoning_tokens: u64,
}

impl Usage {
    pub fn total(&self) -> u64 {
        self.prompt_tokens + self.completion_tokens
    }
}

/// Quota information reported by a backend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LimitInfo {
    pub status: LimitStatus,
    /// 0.0..=1.0 if known.
    pub utilization: Option<f64>,
    /// Unix seconds when the limit resets, if known.
    pub resets_at: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LimitStatus {
    Ok,
    Warning,
    Exhausted,
}

/// One event of a model turn, in caller terms (caller tool names and ids).
///
/// Ordering contract for a successful turn:
///   (TextDelta | ReasoningDelta | ToolCallStart | ToolCallDelta)* , ToolCalls? , Usage? , Done
/// `Limit` may appear anywhere. A failed turn ends with `Error` instead of `Done`.
///
/// Streaming tool calls (ARCHITECTURE 17.3): `ToolCallStart`/`ToolCallDelta` are optional
/// previews of a call while the model writes it. `ToolCalls` stays the authoritative commit:
/// every started call appears in it with the same id and name, and its `arguments` is
/// byte-identical to the concatenation of that call's deltas. `index` is the call's position
/// among this assistant message's tool calls (0..k).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum TurnEvent {
    TextDelta { text: String },
    ReasoningDelta { text: String },
    /// A tool call began: caller-visible id and name are known, arguments follow as deltas.
    ToolCallStart { index: u32, id: String, name: String },
    /// A raw fragment of the JSON arguments of call `index`.
    ToolCallDelta { index: u32, arguments: String },
    /// All tool calls of the assistant message, emitted once, complete (the commit).
    ToolCalls { calls: Vec<ToolCall> },
    Usage { usage: Usage },
    Limit { info: LimitInfo },
    Done { finish: FinishReason },
    Error { error: crate::BridgeError },
}

/// Events of one choice, tagged with the choice index (for n > 1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceEvent {
    pub index: u32,
    pub event: TurnEvent,
}
