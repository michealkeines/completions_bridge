//! Output policy (ARCHITECTURE.md section 7.2).
//!
//! PUBLIC API IS FROZEN (signatures below). The engine drives one `OutputPolicy` per
//! choice attempt: every backend `TurnEvent` goes through `on_event`, and the policy
//! decides what the caller sees.

mod json;

use bridge_types::{
    ChatRequest, EngineConfig, FinishReason, Message, OutputConfig, Part, PolicyInstructions,
    ResponseFormat, Role, ToolCall, ToolChoice, TurnEvent, Usage,
};
use json::JsonVerdict;

/// The subset of a request the policy enforces.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicySpec {
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: bool,
    pub max_tokens: Option<u32>,
    pub stop: Vec<String>,
    pub response_format: ResponseFormat,
    /// Caller tool names offered in this request.
    pub tool_names: Vec<String>,
    /// Characters per token for the max_tokens character budget
    /// (`engine.chars_per_token`; 0 is treated as 1).
    pub chars_per_token: u32,
    /// How many `{...}` candidates JSON extraction tries in a non-JSON reply
    /// (`output.json_extract_max_candidates`).
    pub json_max_candidates: usize,
    /// Instruction texts for [`system_addendum`] (`output.instructions`; `None` = built-in).
    pub instructions: PolicyInstructions,
}

impl PolicySpec {
    /// Request-derived spec with the built-in defaults for the configurable fields
    /// (`chars_per_token`, `json_max_candidates`, `instructions`); the engine overrides
    /// them from the configuration.
    pub fn from_request(req: &ChatRequest) -> Self {
        PolicySpec {
            tool_choice: req.tool_choice.clone(),
            parallel_tool_calls: req.parallel_tool_calls,
            max_tokens: req.max_tokens,
            stop: req.stop.clone(),
            response_format: req.response_format.clone(),
            tool_names: req.tools.iter().map(|t| t.name.clone()).collect(),
            chars_per_token: EngineConfig::default().chars_per_token,
            json_max_candidates: OutputConfig::default().json_extract_max_candidates,
            instructions: PolicyInstructions::default(),
        }
    }
}

/// Extra system-prompt text that steers the model toward the requested behaviour
/// (tool_choice, parallel_tool_calls=false, JSON formats). `None` if nothing is needed.
///
/// Texts come from `spec.instructions` (built-in defaults below when unset). In every
/// text `{name}` is replaced with the tool name (tool_choice function) or the schema
/// name (json_schema), and `{schema}` with the pretty-printed JSON schema.
pub fn system_addendum(spec: &PolicySpec) -> Option<String> {
    let ins = &spec.instructions;
    let mut lines: Vec<String> = Vec::new();
    match &spec.tool_choice {
        ToolChoice::Auto => {}
        // With no tools offered nothing can be called, so there is nothing to say.
        ToolChoice::None if spec.tool_names.is_empty() => {}
        ToolChoice::None => lines.push(fill(
            text_or(&ins.tool_choice_none, DEFAULT_TOOL_CHOICE_NONE),
            "",
            "",
        )),
        ToolChoice::Required => lines.push(fill(
            text_or(&ins.tool_choice_required, DEFAULT_TOOL_CHOICE_REQUIRED),
            "",
            "",
        )),
        ToolChoice::Function { name } => lines.push(fill(
            text_or(&ins.tool_choice_function, DEFAULT_TOOL_CHOICE_FUNCTION),
            name,
            "",
        )),
    }
    // "At most one call" is moot when no tools may be called at all.
    if !spec.parallel_tool_calls
        && !spec.tool_names.is_empty()
        && spec.tool_choice != ToolChoice::None
    {
        lines.push(fill(
            text_or(&ins.parallel_off, DEFAULT_PARALLEL_OFF),
            "",
            "",
        ));
    }
    match &spec.response_format {
        ResponseFormat::Text => {}
        ResponseFormat::JsonObject => {
            lines.push(fill(text_or(&ins.json_object, DEFAULT_JSON_OBJECT), "", ""))
        }
        ResponseFormat::JsonSchema { name, schema, .. } => {
            let pretty =
                serde_json::to_string_pretty(schema).unwrap_or_else(|_| schema.to_string());
            lines.push(fill(
                text_or(&ins.json_schema, DEFAULT_JSON_SCHEMA),
                name,
                &pretty,
            ));
        }
    }
    // An instruction configured as "" is dropped rather than leaving an empty line.
    lines.retain(|l| !l.is_empty());
    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

// Built-in instruction texts (used when `output.instructions.*` is unset).
const DEFAULT_TOOL_CHOICE_NONE: &str = "Do not call any tools in your next reply; answer in text.";
const DEFAULT_TOOL_CHOICE_REQUIRED: &str = "You must call at least one tool in your next reply.";
const DEFAULT_TOOL_CHOICE_FUNCTION: &str = "Your next reply must call the tool `{name}`.";
const DEFAULT_PARALLEL_OFF: &str = "Call at most one tool per reply.";
const DEFAULT_JSON_OBJECT: &str =
    "Respond with a single valid JSON object only, no prose, no code fences.";
const DEFAULT_JSON_SCHEMA: &str =
    "Respond with a single valid JSON object only, no prose, no code fences. \
     The JSON object must conform to this JSON Schema:\n{schema}";

fn text_or<'a>(configured: &'a Option<String>, default: &'a str) -> &'a str {
    configured.as_deref().unwrap_or(default)
}

/// Replace `{name}` and `{schema}` in one pass (substituted values are not re-scanned).
fn fill(template: &str, name: &str, schema: &str) -> String {
    let mut out = String::with_capacity(template.len() + name.len() + schema.len());
    let mut rest = template;
    while let Some(pos) = rest.find('{') {
        out.push_str(&rest[..pos]);
        let tail = &rest[pos..];
        if let Some(after) = tail.strip_prefix("{name}") {
            out.push_str(name);
            rest = after;
        } else if let Some(after) = tail.strip_prefix("{schema}") {
            out.push_str(schema);
            rest = after;
        } else {
            out.push('{');
            rest = &tail[1..];
        }
    }
    out.push_str(rest);
    out
}

/// What the engine must do after feeding an event.
#[derive(Debug, Clone, PartialEq)]
pub enum PolicyOutput {
    /// Forward this event to the caller.
    Emit(TurnEvent),
    /// Stop the backend now (limit or stop sequence reached). The policy has already
    /// emitted the final `Done` for the caller.
    StopBackend,
    /// The attempt violates the request (e.g. tool_choice) and nothing has reached the
    /// caller yet: discard this attempt and re-run it. Never returned when
    /// `final_attempt` is true or after anything was emitted.
    Retry { reason: String },
}

/// Per-attempt state machine. One instance per choice per attempt.
///
/// Pipeline for content text: stop-sequence scan (with hold-back) -> max_tokens
/// budget -> either emitted immediately (streaming) or buffered until `Done`
/// (buffered modes), where tool_choice / parallel / JSON rules are applied.
#[derive(Debug)]
pub struct OutputPolicy {
    spec: PolicySpec,
    final_attempt: bool,
    buffering: bool,
    /// Non-empty stop strings.
    stops: Vec<String>,
    /// Bytes held back for the stop scan: longest stop string length - 1.
    hold_back: usize,
    /// Raw content text not yet released by the stop scan.
    pending: String,
    /// Character budget derived from max_tokens (tokens * chars_per_token).
    char_budget: Option<u64>,
    /// Content characters released so far.
    used_chars: u64,
    /// Buffered-mode content and tool calls awaiting the decision.
    buf_text: String,
    buf_calls: Vec<ToolCall>,
    held_usage: Option<Usage>,
    /// What the caller actually received.
    emitted_text: String,
    emitted_calls: Vec<ToolCall>,
    emitted_content: bool,
    /// Tool calls whose Start/Delta previews were forwarded to the caller (pass-through
    /// modes only), with the arguments forwarded so far.
    streamed: Vec<StreamedCall>,
    altered: bool,
    /// Terminal state reached (Done/Error emitted, backend stopped or retry requested):
    /// everything except `Limit` is swallowed from now on.
    closed: bool,
}

impl OutputPolicy {
    /// `final_attempt`: no retries left; the policy must make the best of this output.
    pub fn new(mut spec: PolicySpec, final_attempt: bool) -> Self {
        // `none` is the default when a request has no tools. With no tools offered there
        // is nothing to enforce, and buffering would turn a plain chat stream into one
        // chunk at the end (and mid-stream errors into plain HTTP errors).
        if spec.tool_choice == ToolChoice::None && spec.tool_names.is_empty() {
            spec.tool_choice = ToolChoice::Auto;
        }
        let buffering =
            spec.tool_choice != ToolChoice::Auto || spec.response_format != ResponseFormat::Text;
        let stops: Vec<String> = spec
            .stop
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect();
        let hold_back = stops
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(0)
            .saturating_sub(1);
        let per_token = u64::from(spec.chars_per_token.max(1));
        let char_budget = spec
            .max_tokens
            .map(|t| u64::from(t).saturating_mul(per_token));
        OutputPolicy {
            spec,
            final_attempt,
            buffering,
            stops,
            hold_back,
            pending: String::new(),
            char_budget,
            used_chars: 0,
            buf_text: String::new(),
            buf_calls: Vec::new(),
            held_usage: None,
            emitted_text: String::new(),
            emitted_calls: Vec::new(),
            emitted_content: false,
            streamed: Vec::new(),
            altered: false,
            closed: false,
        }
    }

    /// True if the policy holds back content until the turn is complete
    /// (tool_choice none/required/function, JSON formats).
    pub fn needs_buffering(&self) -> bool {
        self.buffering
    }

    /// Feed one backend event; returns what to do, in order.
    pub fn on_event(&mut self, event: TurnEvent) -> Vec<PolicyOutput> {
        let mut out = Vec::new();
        if self.closed {
            // Quota information is always useful to the engine; the rest is swallowed.
            if let TurnEvent::Limit { .. } = event {
                out.push(PolicyOutput::Emit(event));
            }
            return out;
        }
        match event {
            TurnEvent::ToolCallStart { index, id, name } => {
                self.on_call_start(index, id, name, &mut out)
            }
            TurnEvent::ToolCallDelta { index, arguments } => {
                self.on_call_delta(index, arguments, &mut out)
            }
            TurnEvent::TextDelta { text } => self.on_text(text, &mut out),
            TurnEvent::ReasoningDelta { .. } | TurnEvent::Limit { .. } => {
                out.push(PolicyOutput::Emit(event))
            }
            TurnEvent::ToolCalls { calls } => {
                self.end_text(&mut out);
                if self.closed {
                    return out; // limit hit while flushing: calls are dropped
                }
                if self.buffering {
                    self.buf_calls.extend(calls);
                } else {
                    let calls = self.limit_parallel(calls);
                    if !calls.is_empty() {
                        self.emitted_content = true;
                        self.emitted_calls.extend(calls.iter().cloned());
                    }
                    out.push(PolicyOutput::Emit(TurnEvent::ToolCalls { calls }));
                }
            }
            TurnEvent::Usage { usage } => {
                self.end_text(&mut out);
                if self.closed {
                    return out;
                }
                if self.buffering {
                    self.held_usage = Some(usage);
                } else {
                    out.push(PolicyOutput::Emit(TurnEvent::Usage { usage }));
                }
            }
            TurnEvent::Done { finish } => {
                self.end_text(&mut out);
                if self.closed {
                    return out;
                }
                if self.buffering {
                    self.finalize(finish, false, &mut out);
                } else {
                    self.check_streamed();
                    let finish = normalize_finish(finish, !self.emitted_calls.is_empty());
                    out.push(PolicyOutput::Emit(TurnEvent::Done { finish }));
                    self.closed = true;
                }
            }
            TurnEvent::Error { .. } => {
                self.pending.clear();
                out.push(PolicyOutput::Emit(event));
                self.closed = true;
            }
        }
        out
    }

    /// The backend stream ended. Flush anything held back. If no `Done` was seen,
    /// the policy decides (usually nothing; the engine handles missing Done as error).
    pub fn finish(&mut self) -> Vec<PolicyOutput> {
        let mut out = Vec::new();
        if self.closed {
            return out;
        }
        // No Done seen. Streaming mode: release the stop hold-back (the text was
        // produced and passed the scan). Buffered mode: nothing was validated, emit
        // nothing; the engine treats the missing Done as an error.
        if !self.buffering {
            self.end_text(&mut out);
        }
        self.closed = true;
        out
    }

    /// True if what the caller received differs from what the backend produced
    /// (truncated, calls dropped, JSON extracted...). Then no checkpoint may be frozen.
    pub fn altered(&self) -> bool {
        self.altered
    }

    /// The assistant message exactly as the caller received it (text, tool calls;
    /// reasoning excluded). Used as the last message of the new checkpoint's key.
    pub fn assistant_message(&self) -> Message {
        let mut parts = Vec::with_capacity(1 + self.emitted_calls.len());
        if !self.emitted_text.is_empty() {
            parts.push(Part::Text {
                text: self.emitted_text.clone(),
            });
        }
        parts.extend(self.emitted_calls.iter().cloned().map(Part::ToolCall));
        Message {
            role: Role::Assistant,
            parts,
        }
    }

    // ------------------------------------------------------------ internals

    fn can_retry(&self) -> bool {
        !self.final_attempt && !self.emitted_content
    }

    /// Stop-sequence scan with hold-back.
    fn on_text(&mut self, text: String, out: &mut Vec<PolicyOutput>) {
        if self.stops.is_empty() {
            self.release(&text, out);
            return;
        }
        self.pending.push_str(&text);
        let hit = self
            .stops
            .iter()
            .filter_map(|s| self.pending.find(s.as_str()))
            .min();
        if let Some(pos) = hit {
            let before = self.pending[..pos].to_string();
            self.pending.clear();
            self.release(&before, out);
            if !self.closed {
                self.terminate(FinishReason::Stop, out);
            }
            return;
        }
        let mut cut = self.pending.len().saturating_sub(self.hold_back);
        while cut > 0 && !self.pending.is_char_boundary(cut) {
            cut -= 1;
        }
        if cut > 0 {
            let ready: String = self.pending.drain(..cut).collect();
            self.release(&ready, out);
        }
    }

    /// End of content text: release whatever the stop scan still holds.
    fn end_text(&mut self, out: &mut Vec<PolicyOutput>) {
        if !self.pending.is_empty() {
            let rest = std::mem::take(&mut self.pending);
            self.release(&rest, out);
        }
    }

    /// max_tokens budget, then accept.
    fn release(&mut self, text: &str, out: &mut Vec<PolicyOutput>) {
        if text.is_empty() || self.closed {
            return;
        }
        if let Some(budget) = self.char_budget {
            let remaining = budget.saturating_sub(self.used_chars);
            let n = text.chars().count() as u64;
            if n > remaining {
                let byte_end = text
                    .char_indices()
                    .nth(usize::try_from(remaining).unwrap_or(usize::MAX))
                    .map(|(i, _)| i)
                    .unwrap_or(text.len());
                self.used_chars = budget;
                self.accept(&text[..byte_end], out);
                self.pending.clear();
                self.terminate(FinishReason::Length, out);
                return;
            }
            self.used_chars += n;
        }
        self.accept(text, out);
    }

    fn accept(&mut self, text: &str, out: &mut Vec<PolicyOutput>) {
        if text.is_empty() {
            return;
        }
        if self.buffering {
            self.buf_text.push_str(text);
        } else {
            self.emitted_text.push_str(text);
            self.emitted_content = true;
            out.push(PolicyOutput::Emit(TurnEvent::TextDelta {
                text: text.to_string(),
            }));
        }
    }

    /// Stop sequence or token limit reached: end the turn for the caller and stop
    /// the backend.
    fn terminate(&mut self, finish: FinishReason, out: &mut Vec<PolicyOutput>) {
        self.altered = true;
        if self.buffering {
            self.finalize(finish, true, out);
        } else {
            out.push(PolicyOutput::Emit(TurnEvent::Done { finish }));
            out.push(PolicyOutput::StopBackend);
            self.closed = true;
        }
    }

    fn limit_parallel(&mut self, mut calls: Vec<ToolCall>) -> Vec<ToolCall> {
        if !self.spec.parallel_tool_calls && calls.len() > 1 {
            // Keep the call the caller already saw streaming, if any; else the first.
            let keep = self
                .streamed
                .first()
                .and_then(|s| calls.iter().position(|c| c.id == s.id))
                .unwrap_or(0);
            let kept = calls.swap_remove(keep);
            calls = vec![kept];
            self.altered = true;
        }
        calls
    }

    /// A tool call preview began (ARCHITECTURE 17.3).
    ///
    /// Buffered modes swallow previews: the decision is made at the commit, and nothing
    /// reaching the caller keeps Retry possible. Pass-through forwards them immediately
    /// (they count as emitted content); with `parallel_tool_calls: false` only the first
    /// started call is forwarded, renumbered to index 0.
    fn on_call_start(&mut self, index: u32, id: String, name: String, out: &mut Vec<PolicyOutput>) {
        if self.buffering {
            return;
        }
        // A tool call ends the current text segment: release the stop-scan hold-back
        // first so the caller sees text and call in order.
        self.end_text(out);
        if self.closed {
            return;
        }
        if self.streamed.iter().any(|c| c.backend_index == index) {
            return; // duplicate Start for the same call: ignore
        }
        let caller_index = if self.spec.parallel_tool_calls {
            index
        } else if self.streamed.is_empty() {
            0
        } else {
            // A second call under parallel_tool_calls=false: never shown, dropped at commit.
            self.altered = true;
            return;
        };
        self.emitted_content = true;
        self.streamed.push(StreamedCall {
            backend_index: index,
            caller_index,
            id: id.clone(),
            name: name.clone(),
            arguments: String::new(),
        });
        out.push(PolicyOutput::Emit(TurnEvent::ToolCallStart {
            index: caller_index,
            id,
            name,
        }));
    }

    /// A raw arguments fragment: forwarded verbatim for forwarded calls only.
    fn on_call_delta(&mut self, index: u32, arguments: String, out: &mut Vec<PolicyOutput>) {
        if self.buffering {
            return;
        }
        let Some(call) = self.streamed.iter_mut().find(|c| c.backend_index == index) else {
            return; // swallowed call (parallel=false) or Delta without Start
        };
        call.arguments.push_str(&arguments);
        out.push(PolicyOutput::Emit(TurnEvent::ToolCallDelta {
            index: call.caller_index,
            arguments,
        }));
    }

    /// At `Done`: every forwarded preview must be committed unchanged, otherwise what the
    /// caller saw differs from `assistant_message()` and no checkpoint may be frozen.
    fn check_streamed(&mut self) {
        let consistent = self.streamed.iter().all(|s| {
            self.emitted_calls
                .iter()
                .any(|c| c.id == s.id && c.name == s.name && c.arguments == s.arguments)
        });
        if !consistent {
            self.altered = true;
        }
    }

    fn retry(&mut self, reason: String, out: &mut Vec<PolicyOutput>) {
        out.push(PolicyOutput::Retry { reason });
        self.closed = true;
    }

    /// Buffered mode decision point: validate, then release (or ask for a retry).
    fn finalize(&mut self, finish: FinishReason, stopping: bool, out: &mut Vec<PolicyOutput>) {
        let mut calls = std::mem::take(&mut self.buf_calls);
        let mut text = std::mem::take(&mut self.buf_text);
        let can_retry = self.can_retry();

        match self.spec.tool_choice.clone() {
            ToolChoice::Auto => {}
            ToolChoice::None => {
                if !calls.is_empty() {
                    if can_retry {
                        return self.retry("tool_choice none: model called a tool".into(), out);
                    }
                    calls.clear();
                    self.altered = true;
                }
            }
            ToolChoice::Required => {
                if calls.is_empty() && can_retry {
                    return self.retry("tool_choice required: no tool call".into(), out);
                }
            }
            ToolChoice::Function { name } => {
                let matching = calls.iter().filter(|c| c.name == name).count();
                if matching == 0 {
                    if can_retry {
                        return self
                            .retry(format!("tool_choice function: `{name}` not called"), out);
                    }
                } else if matching < calls.len() {
                    if can_retry {
                        return self.retry(
                            format!("tool_choice function: tools other than `{name}` called"),
                            out,
                        );
                    }
                    calls.retain(|c| c.name == name);
                    self.altered = true;
                }
            }
        }

        calls = self.limit_parallel(calls);

        // JSON formats apply to text replies. A length-truncated reply is expected to
        // be incomplete, so it is returned as is rather than retried.
        if self.spec.response_format != ResponseFormat::Text
            && calls.is_empty()
            && finish != FinishReason::Length
        {
            match json::check(
                &text,
                &self.spec.response_format,
                self.spec.json_max_candidates,
            ) {
                JsonVerdict::Valid => {}
                JsonVerdict::Extracted(extracted) => {
                    text = extracted;
                    self.altered = true;
                }
                JsonVerdict::Invalid => {
                    if can_retry {
                        return self
                            .retry("response_format: content is not valid JSON".into(), out);
                    }
                }
            }
        }

        if !text.is_empty() {
            self.emitted_text.push_str(&text);
            self.emitted_content = true;
            out.push(PolicyOutput::Emit(TurnEvent::TextDelta { text }));
        }
        if !calls.is_empty() {
            self.emitted_calls.extend(calls.iter().cloned());
            self.emitted_content = true;
            out.push(PolicyOutput::Emit(TurnEvent::ToolCalls { calls }));
        }
        if let Some(usage) = self.held_usage.take() {
            out.push(PolicyOutput::Emit(TurnEvent::Usage { usage }));
        }
        let finish = normalize_finish(finish, !self.emitted_calls.is_empty());
        out.push(PolicyOutput::Emit(TurnEvent::Done { finish }));
        if stopping {
            out.push(PolicyOutput::StopBackend);
        }
        self.closed = true;
    }
}

/// A tool call whose previews reached the caller.
#[derive(Debug)]
struct StreamedCall {
    /// `index` as produced by the backend.
    backend_index: u32,
    /// `index` as forwarded to the caller.
    caller_index: u32,
    id: String,
    name: String,
    /// Concatenation of the forwarded deltas.
    arguments: String,
}

/// Keep `finish_reason` consistent with what the caller actually got.
fn normalize_finish(finish: FinishReason, has_calls: bool) -> FinishReason {
    match (finish, has_calls) {
        (FinishReason::ToolCalls, false) => FinishReason::Stop,
        (FinishReason::Stop, true) => FinishReason::ToolCalls,
        (f, _) => f,
    }
}

#[cfg(test)]
mod tests;
