//! Decoder for `claude -p --output-format stream-json --include-partial-messages`.
//!
//! Pure state machine: feed stdout lines, get `Decoded` items. Malformed lines are ignored.

use std::collections::HashMap;

use bridge_types::{BridgeError, FinishReason, LimitInfo, LimitStatus, Usage};
use serde_json::Value;

use crate::codec::TOOL_PREFIX;
use crate::errors;
use crate::session::ClaudeCall;

/// Name of the internal tool claude uses to implement `--json-schema`.
/// Not configurable: chosen by claude, the bridge only recognises it.
pub const STRUCTURED_OUTPUT_TOOL: &str = "StructuredOutput";

#[derive(Debug, Clone, PartialEq)]
pub enum Decoded {
    /// First line of a run; carries the resolved model id.
    Init { model: String },
    Text(String),
    Reasoning(String),
    Limit(LimitInfo),
    /// A bridge tool call began (`content_block_start` of type `tool_use`). `index` is the
    /// call's ordinal in this message; `id` and `name` are claude-side (encoded) values.
    ToolCallStart { index: u32, id: String, name: String },
    /// A raw `input_json_delta` fragment of call `index` (verbatim, never sanitised).
    ToolCallDelta { index: u32, arguments: String },
    /// The assistant message holding our tool calls is complete: stop claude now.
    /// `arguments[i]` is the arguments string of `calls[i]` for the commit: for streamed
    /// calls the exact concatenation of their deltas, otherwise the serialised input.
    StopForToolCalls { calls: Vec<ClaudeCall>, arguments: Vec<String>, message_id: Option<String>, usage: Option<Usage> },
    /// Normal end of the turn.
    Finished { finish: FinishReason, usage: Option<Usage>, structured: Option<Value> },
    /// claude reported an error (`result` with `is_error`).
    Failed { error: BridgeError, raw: String },
}

/// A bridge tool call seen as `content_block_start` (in ordinal order).
#[derive(Debug, Default, Clone)]
struct StreamedCall {
    id: String,
    name: String,
    /// Concatenation of all `partial_json` fragments.
    arguments: String,
}

#[derive(Debug, Default)]
pub struct StreamDecoder {
    /// Complete calls from `assistant` events (authoritative input).
    calls: Vec<ClaudeCall>,
    /// Started calls, by ordinal.
    streamed: Vec<StreamedCall>,
    /// Content-block index → ordinal, for the current message.
    block_calls: HashMap<u64, usize>,
    message_id: Option<String>,
    message_usage: Option<Value>,
    stop_reason: Option<String>,
    structured: Option<Value>,
    resets_at: Option<u64>,
    exhausted: bool,
    done: bool,
}

/// claude usage object → IR usage.
pub fn usage_from(v: &Value) -> Usage {
    let n = |k: &str| v.get(k).and_then(Value::as_u64).unwrap_or(0);
    let input = n("input_tokens");
    let cache_write = n("cache_creation_input_tokens");
    let cache_read = n("cache_read_input_tokens");
    Usage {
        prompt_tokens: input + cache_write + cache_read,
        completion_tokens: n("output_tokens"),
        cached_tokens: cache_read,
        reasoning_tokens: v.pointer("/output_tokens_details/thinking_tokens").and_then(Value::as_u64).unwrap_or(0),
    }
}

pub fn finish_from(stop_reason: Option<&str>) -> FinishReason {
    match stop_reason {
        Some("max_tokens") | Some("model_context_window_exceeded") => FinishReason::Length,
        Some("refusal") => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    }
}

fn status_from(s: Option<&str>) -> Option<LimitStatus> {
    match s? {
        "allowed" => Some(LimitStatus::Ok),
        "allowed_warning" => Some(LimitStatus::Warning),
        "rejected" | "blocked" | "exhausted" => Some(LimitStatus::Exhausted),
        _ => None,
    }
}

/// `rate_limit_info` → LimitInfo.
///
/// Observed (2.1.325): the top-level status can describe the *overage* window
/// (`rateLimitType: "overage"`, `isUsingOverage: false`, utilization 0.99, status
/// "allowed_warning") while the subscription windows in `unifiedWindows` are at a few
/// percent. In that case the worst unified window is reported instead.
pub fn limit_from(info: &Value) -> LimitInfo {
    let top_status = status_from(info.get("status").and_then(Value::as_str)).unwrap_or(LimitStatus::Ok);
    let overage_only = info.get("rateLimitType").and_then(Value::as_str) == Some("overage")
        && info.get("isUsingOverage").and_then(Value::as_bool) == Some(false);
    if overage_only && top_status != LimitStatus::Exhausted {
        if let Some(Value::Object(windows)) = info.get("unifiedWindows") {
            let mut best: Option<LimitInfo> = None;
            for w in windows.values() {
                let status = status_from(w.get("status").and_then(Value::as_str)).unwrap_or(LimitStatus::Ok);
                let li = LimitInfo {
                    status,
                    utilization: w.get("utilization").and_then(Value::as_f64),
                    resets_at: w.get("resetsAt").and_then(Value::as_u64),
                };
                let rank = |l: &LimitInfo| (l.status as u8, (l.utilization.unwrap_or(0.0) * 1e6) as u64);
                if best.as_ref().map(|b| rank(&li) > rank(b)).unwrap_or(true) {
                    best = Some(li);
                }
            }
            if let Some(b) = best {
                return b;
            }
        }
    }
    LimitInfo {
        status: top_status,
        utilization: info.get("utilization").and_then(Value::as_f64),
        resets_at: info.get("resetsAt").and_then(Value::as_u64),
    }
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// True once a terminal item (stop, finished, failed) was produced.
    pub fn is_done(&self) -> bool {
        self.done
    }

    pub fn calls(&self) -> &[ClaudeCall] {
        &self.calls
    }

    pub fn message_id(&self) -> Option<&str> {
        self.message_id.as_deref()
    }

    /// Reset time of the last rate-limit event, if any (unix seconds).
    pub fn resets_at(&self) -> Option<u64> {
        self.resets_at
    }

    /// True if a rate-limit event said the limit is exhausted.
    pub fn limit_exhausted(&self) -> bool {
        self.exhausted
    }

    pub fn feed(&mut self, line: &str) -> Vec<Decoded> {
        let mut out = Vec::new();
        if self.done {
            return out;
        }
        let Ok(v) = serde_json::from_str::<Value>(line.trim()) else { return out };
        // Events of sub-agents (never enabled here) are ignored.
        if v.get("parent_tool_use_id").map(|p| !p.is_null()).unwrap_or(false) {
            return out;
        }
        match v.get("type").and_then(Value::as_str) {
            Some("system")
                if v.get("subtype").and_then(Value::as_str) == Some("init") => {
                    let model = v.get("model").and_then(Value::as_str).unwrap_or_default().to_string();
                    out.push(Decoded::Init { model });
                }
            Some("stream_event") => self.stream_event(v.get("event").unwrap_or(&Value::Null), &mut out),
            Some("assistant") => self.assistant(&v),
            Some("rate_limit_event") => {
                if let Some(info) = v.get("rate_limit_info") {
                    let li = limit_from(info);
                    if li.resets_at.is_some() {
                        self.resets_at = li.resets_at;
                    }
                    self.exhausted = li.status == LimitStatus::Exhausted;
                    out.push(Decoded::Limit(li));
                }
            }
            Some("result") => self.result(&v, &mut out),
            _ => {}
        }
        out
    }

    fn stream_event(&mut self, ev: &Value, out: &mut Vec<Decoded>) {
        match ev.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                let m = ev.get("message").unwrap_or(&Value::Null);
                if let Some(id) = m.get("id").and_then(Value::as_str) {
                    self.message_id = Some(id.to_string());
                }
                self.message_usage = m.get("usage").cloned();
                self.stop_reason = None;
                self.block_calls.clear();
            }
            Some("content_block_start") => {
                // A thinking/text block may already carry text.
                let b = ev.get("content_block").unwrap_or(&Value::Null);
                match b.get("type").and_then(Value::as_str) {
                    Some("text") => push_nonempty(out, b.get("text"), Decoded::Text),
                    Some("thinking") => push_nonempty(out, b.get("thinking"), Decoded::Reasoning),
                    Some("tool_use") => self.tool_use_start(ev, b, out),
                    _ => {}
                }
            }
            Some("content_block_delta") => {
                let d = ev.get("delta").unwrap_or(&Value::Null);
                match d.get("type").and_then(Value::as_str) {
                    Some("text_delta") => push_nonempty(out, d.get("text"), Decoded::Text),
                    Some("thinking_delta") => push_nonempty(out, d.get("thinking"), Decoded::Reasoning),
                    Some("input_json_delta") => {
                        let ordinal = ev.get("index").and_then(Value::as_u64).and_then(|i| self.block_calls.get(&i)).copied();
                        let fragment = d.get("partial_json").and_then(Value::as_str).unwrap_or_default();
                        if let (Some(o), false) = (ordinal, fragment.is_empty()) {
                            if let Some(c) = self.streamed.get_mut(o) {
                                c.arguments.push_str(fragment);
                                out.push(Decoded::ToolCallDelta { index: o as u32, arguments: fragment.to_string() });
                            }
                        }
                    }
                    _ => {}
                }
            }
            Some("message_delta") => {
                if let Some(sr) = ev.pointer("/delta/stop_reason").and_then(Value::as_str) {
                    self.stop_reason = Some(sr.to_string());
                }
                if let Some(u) = ev.get("usage") {
                    let mut merged = self.message_usage.take().unwrap_or_else(|| Value::Object(Default::default()));
                    if let (Some(dst), Some(src)) = (merged.as_object_mut(), u.as_object()) {
                        for (k, val) in src {
                            if !val.is_null() {
                                dst.insert(k.clone(), val.clone());
                            }
                        }
                    }
                    self.message_usage = Some(merged);
                }
            }
            Some("message_stop") if self.has_calls() => {
                let usage = self.message_usage.as_ref().map(usage_from);
                self.commit(usage, out);
            }
            _ => {}
        }
    }

    fn has_calls(&self) -> bool {
        !self.calls.is_empty() || !self.streamed.is_empty()
    }

    /// `content_block_start` of a `tool_use` block: bridge tools only (the internal
    /// StructuredOutput tool and anything else claude might call are not caller calls).
    fn tool_use_start(&mut self, ev: &Value, b: &Value, out: &mut Vec<Decoded>) {
        let name = b.get("name").and_then(Value::as_str).unwrap_or_default();
        let id = b.get("id").and_then(Value::as_str).unwrap_or_default();
        let Some(block) = ev.get("index").and_then(Value::as_u64) else { return };
        if !name.starts_with(TOOL_PREFIX) || id.is_empty() || self.streamed.iter().any(|c| c.id == id) {
            return;
        }
        let ordinal = self.streamed.len();
        self.block_calls.insert(block, ordinal);
        self.streamed.push(StreamedCall { id: id.to_string(), name: name.to_string(), arguments: String::new() });
        out.push(Decoded::ToolCallStart { index: ordinal as u32, id: id.to_string(), name: name.to_string() });
    }

    /// Emits the commit: streamed calls in ordinal order (arguments = concatenated
    /// deltas; a call that got no deltas first gets one Delta with its serialised input),
    /// then calls that were never started (arguments = serialised input).
    fn commit(&mut self, usage: Option<Usage>, out: &mut Vec<Decoded>) {
        self.done = true;
        let mut calls = Vec::new();
        let mut arguments = Vec::new();
        for (i, s) in self.streamed.iter_mut().enumerate() {
            let complete = self.calls.iter().find(|c| c.id == s.id);
            let input = match complete {
                Some(c) => c.input.clone(),
                None => crate::session::arguments_to_input(&s.arguments),
            };
            if s.arguments.is_empty() {
                s.arguments = input.to_string();
                out.push(Decoded::ToolCallDelta { index: i as u32, arguments: s.arguments.clone() });
            }
            let name = complete.map(|c| c.name.clone()).unwrap_or_else(|| s.name.clone());
            calls.push(ClaudeCall { id: s.id.clone(), name, input });
            arguments.push(s.arguments.clone());
        }
        for c in &self.calls {
            if !self.streamed.iter().any(|s| s.id == c.id) {
                calls.push(c.clone());
                arguments.push(c.input.to_string());
            }
        }
        out.push(Decoded::StopForToolCalls { calls, arguments, message_id: self.message_id.clone(), usage });
    }

    fn assistant(&mut self, v: &Value) {
        let m = v.get("message").unwrap_or(&Value::Null);
        if m.get("model").and_then(Value::as_str) == Some(crate::session::SYNTHETIC_MODEL) {
            return;
        }
        let Some(blocks) = m.get("content").and_then(Value::as_array) else { return };
        for b in blocks {
            if b.get("type").and_then(Value::as_str) != Some("tool_use") {
                continue;
            }
            let name = b.get("name").and_then(Value::as_str).unwrap_or_default();
            let input = b.get("input").cloned().unwrap_or_else(|| Value::Object(Default::default()));
            if name == STRUCTURED_OUTPUT_TOOL {
                self.structured = Some(input);
            } else if name.starts_with(TOOL_PREFIX) {
                let id = b.get("id").and_then(Value::as_str).unwrap_or_default().to_string();
                if !self.calls.iter().any(|c| c.id == id) {
                    self.calls.push(ClaudeCall { id, name: name.to_string(), input });
                }
            }
        }
    }

    fn result(&mut self, v: &Value, out: &mut Vec<Decoded>) {
        if self.has_calls() {
            // Normally we stop at message_stop; if the result came first, still stop.
            let usage = v.get("usage").map(usage_from).or_else(|| self.message_usage.as_ref().map(usage_from));
            self.commit(usage, out);
            return;
        }
        self.done = true;
        let is_error = v.get("is_error").and_then(Value::as_bool).unwrap_or(false)
            || v.get("subtype").and_then(Value::as_str).map(|s| s.starts_with("error")).unwrap_or(false);
        if is_error {
            let raw = match v.get("result") {
                Some(Value::String(s)) => s.clone(),
                Some(other) => other.to_string(),
                None => v.get("errors").map(Value::to_string).unwrap_or_default(),
            };
            let status = v.get("api_error_status").and_then(Value::as_u64);
            let mut error = errors::classify(&raw, status, self.resets_at);
            if self.exhausted && error.class == bridge_types::ErrorClass::Internal {
                error = errors::classify("usage limit", None, self.resets_at);
            }
            out.push(Decoded::Failed { error, raw });
            return;
        }
        let stop = v.get("stop_reason").and_then(Value::as_str).map(str::to_string).or_else(|| self.stop_reason.clone());
        let structured = v.get("structured_output").filter(|s| !s.is_null()).cloned().or_else(|| self.structured.clone());
        out.push(Decoded::Finished {
            finish: finish_from(stop.as_deref()),
            usage: v.get("usage").map(usage_from).or_else(|| self.message_usage.as_ref().map(usage_from)),
            structured,
        });
    }
}

fn push_nonempty(out: &mut Vec<Decoded>, v: Option<&Value>, f: fn(String) -> Decoded) {
    if let Some(s) = v.and_then(Value::as_str) {
        if !s.is_empty() {
            out.push(f(s.to_string()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_all(text: &str) -> Vec<Decoded> {
        let mut d = StreamDecoder::new();
        text.lines().flat_map(|l| d.feed(l)).collect()
    }

    #[test]
    fn text_turn() {
        let items = decode_all(include_str!("../tests/fixtures/stream_text.jsonl"));
        let text: String = items.iter().filter_map(|i| if let Decoded::Text(t) = i { Some(t.as_str()) } else { None }).collect();
        assert_eq!(text, "Hello there, friend, welcome!");
        assert!(matches!(items[0], Decoded::Init { ref model } if model == "claude-haiku-5-5"));
        let limit = items.iter().find_map(|i| if let Decoded::Limit(l) = i { Some(l.clone()) } else { None }).unwrap();
        // Overage-only warning is replaced by the worst subscription window.
        assert_eq!(limit.status, LimitStatus::Ok);
        assert_eq!(limit.utilization, Some(0.06));
        match items.last().unwrap() {
            Decoded::Finished { finish, usage, structured } => {
                assert_eq!(*finish, FinishReason::Stop);
                assert_eq!(*usage, Some(Usage { prompt_tokens: 1042, completion_tokens: 14, cached_tokens: 0, reasoning_tokens: 0 }));
                assert!(structured.is_none());
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn tool_call_turn_stops_at_message_stop() {
        let text = include_str!("../tests/fixtures/stream_tool_call.jsonl");
        let mut d = StreamDecoder::new();
        let mut items = Vec::new();
        for l in text.lines() {
            items.extend(d.feed(l));
        }
        match items.last().unwrap() {
            Decoded::StopForToolCalls { calls, message_id, usage, .. } => {
                assert_eq!(calls.len(), 2);
                assert_eq!(calls[0].id, "toolu_01PkKJCGD5vZ1xGpPhdE7HRY");
                assert_eq!(calls[0].name, "mcp__bridge__read");
                assert_eq!(calls[1].input, serde_json::json!({"filePath": "src/b.ts"}));
                assert_eq!(message_id.as_deref(), Some("msg_011Ch6UyXbHqrVvQXFFTBxrx"));
                assert_eq!(usage.unwrap().completion_tokens, 117);
                assert_eq!(usage.unwrap().prompt_tokens, 1062);
            }
            other => panic!("{other:?}"),
        }
        assert!(d.is_done());
        assert!(d.feed(r#"{"type":"result","is_error":false}"#).is_empty());
    }

    /// Per-call accumulation of the streamed events: (id, name, concatenated deltas).
    fn streamed(items: &[Decoded]) -> Vec<(String, String, String)> {
        let mut out: Vec<(String, String, String)> = Vec::new();
        for i in items {
            match i {
                Decoded::ToolCallStart { index, id, name } => {
                    assert_eq!(*index as usize, out.len(), "ordinals are dense and in order");
                    out.push((id.clone(), name.clone(), String::new()));
                }
                Decoded::ToolCallDelta { index, arguments } => {
                    assert!(!arguments.is_empty(), "no empty deltas");
                    out[*index as usize].2.push_str(arguments);
                }
                _ => {}
            }
        }
        out
    }

    fn commit(items: &[Decoded]) -> (Vec<ClaudeCall>, Vec<String>) {
        match items.last().unwrap() {
            Decoded::StopForToolCalls { calls, arguments, .. } => (calls.clone(), arguments.clone()),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn streamed_calls_from_recorded_fixture() {
        // Recorded live (claude 2.1.325, haiku): three parallel calls, the first with no
        // arguments (only an empty input_json_delta), the last with escapes and unicode.
        let items = decode_all(include_str!("../tests/fixtures/stream_tool_calls_streamed.jsonl"));
        let s = streamed(&items);
        assert_eq!(s.len(), 3);
        assert_eq!(s[0].1, "mcp__bridge__list_todos");
        assert_eq!(s[1].1, "mcp__bridge__read");
        assert_eq!(s[2].1, "mcp__bridge__write");
        // Empty-input call: one Delta with the serialised input, emitted before the commit.
        assert_eq!(s[0].2, "{}");
        let fill = items.iter().position(|i| matches!(i, Decoded::ToolCallDelta { index: 0, .. })).unwrap();
        let stop = items.iter().position(|i| matches!(i, Decoded::StopForToolCalls { .. })).unwrap();
        assert_eq!(fill, stop - 1);
        // Verbatim fragments: claude's spacing is kept.
        assert_eq!(s[1].2, r#"{"filePath": "src/main.rs"}"#);
        let (calls, args) = commit(&items);
        assert_eq!(calls.len(), 3);
        for (k, (id, name, concat)) in s.iter().enumerate() {
            assert_eq!(&calls[k].id, id);
            assert_eq!(&calls[k].name, name);
            assert_eq!(&args[k], concat, "commit arguments byte-identical to the deltas");
            let parsed: Value = serde_json::from_str(concat).unwrap();
            assert_eq!(parsed, calls[k].input, "same JSON as the complete input");
        }
        assert_ne!(args[1], calls[1].input.to_string(), "re-serialising would change bytes");
        assert_eq!(calls[2].input["content"], "Hello, \"quoted\" world!\nLine two with unicode: caf\u{e9} \u{2713}.\n");
        // Starts come before their deltas and before any text-free commit.
        let first_start = items.iter().position(|i| matches!(i, Decoded::ToolCallStart { .. })).unwrap();
        assert!(first_start < fill);
    }

    #[test]
    fn streamed_calls_two_call_fixture() {
        let items = decode_all(include_str!("../tests/fixtures/stream_tool_call.jsonl"));
        let s = streamed(&items);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].2, r#"{"filePath": "src/a.ts"}"#);
        let (calls, args) = commit(&items);
        assert_eq!(args, vec![s[0].2.clone(), s[1].2.clone()]);
        assert_eq!(calls[1].input, serde_json::json!({"filePath": "src/b.ts"}));
    }

    #[test]
    fn structured_output_and_foreign_tools_not_streamed() {
        let items = decode_all(include_str!("../tests/fixtures/stream_schema.jsonl"));
        assert!(!items.iter().any(|i| matches!(i, Decoded::ToolCallStart { .. } | Decoded::ToolCallDelta { .. })));
        let mut d = StreamDecoder::new();
        let ev = |e: &str| format!(r#"{{"type":"stream_event","event":{e}}}"#);
        assert!(d.feed(&ev(r#"{"type":"message_start","message":{"id":"m1"}}"#)).is_empty());
        let it = d.feed(&ev(r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t0","name":"StructuredOutput","input":{}}}"#));
        assert!(it.is_empty());
        assert!(d.feed(&ev(r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"a\":1}"}}"#)).is_empty());
        let it = d.feed(&ev(r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t1","name":"Bash","input":{}}}"#));
        assert!(it.is_empty());
        assert!(d.feed(&ev(r#"{"type":"message_stop"}"#)).is_empty(), "no bridge calls: no stop");
    }

    #[test]
    fn streamed_call_without_assistant_event_still_commits() {
        let mut d = StreamDecoder::new();
        let ev = |e: &str| format!(r#"{{"type":"stream_event","event":{e}}}"#);
        d.feed(&ev(r#"{"type":"message_start","message":{"id":"m1"}}"#));
        d.feed(&ev(r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#));
        let it = d.feed(&ev(r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t1","name":"mcp__bridge__grep","input":{}}}"#));
        assert_eq!(it, vec![Decoded::ToolCallStart { index: 0, id: "t1".into(), name: "mcp__bridge__grep".into() }]);
        let it = d.feed(&ev(r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"q\" : [1, 2"}}"#));
        assert_eq!(it, vec![Decoded::ToolCallDelta { index: 0, arguments: r#"{"q" : [1, 2"#.into() }]);
        d.feed(&ev(r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"]}"}}"#));
        let it = d.feed(&ev(r#"{"type":"message_stop"}"#));
        let (calls, args) = commit(&it);
        assert_eq!(args, vec![r#"{"q" : [1, 2]}"#.to_string()]);
        assert_eq!(calls[0].input, serde_json::json!({"q": [1, 2]}));
        assert_eq!(calls[0].name, "mcp__bridge__grep");
    }

    #[test]
    fn structured_output_turn() {
        let items = decode_all(include_str!("../tests/fixtures/stream_schema.jsonl"));
        assert!(!items.iter().any(|i| matches!(i, Decoded::StopForToolCalls { .. })));
        match items.last().unwrap() {
            Decoded::Finished { finish, structured, usage } => {
                assert_eq!(*finish, FinishReason::Stop);
                assert_eq!(structured.as_ref().unwrap()["hex"], "#FF0000");
                assert_eq!(usage.unwrap().reasoning_tokens, 151);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn errors_and_garbage() {
        let mut d = StreamDecoder::new();
        assert!(d.feed("not json").is_empty());
        assert!(d.feed("{\"type\":\"stream_event\"}").is_empty());
        assert!(d.feed("{\"type\":\"assistant\",\"message\":{\"content\":\"weird\"}}").is_empty());
        let items = d.feed(r#"{"type":"result","subtype":"success","is_error":true,"result":"API Error: 400 prompt is too long: 1 > 0","api_error_status":400}"#);
        match &items[0] {
            Decoded::Failed { error, .. } => assert_eq!(error.class, bridge_types::ErrorClass::ContextTooLong),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn stop_reasons_and_limits() {
        assert_eq!(finish_from(Some("end_turn")), FinishReason::Stop);
        assert_eq!(finish_from(Some("max_tokens")), FinishReason::Length);
        assert_eq!(finish_from(Some("refusal")), FinishReason::ContentFilter);
        assert_eq!(finish_from(None), FinishReason::Stop);
        let l = limit_from(&serde_json::json!({"status": "rejected", "resetsAt": 5, "utilization": 1.0, "rateLimitType": "five_hour"}));
        assert_eq!(l, LimitInfo { status: LimitStatus::Exhausted, utilization: Some(1.0), resets_at: Some(5) });
        let l = limit_from(&serde_json::json!({"status": "allowed_warning", "utilization": 0.9, "rateLimitType": "seven_day"}));
        assert_eq!(l.status, LimitStatus::Warning);
        let mut d = StreamDecoder::new();
        d.feed(r#"{"type":"rate_limit_event","rate_limit_info":{"status":"rejected","resetsAt":4102444800}}"#);
        let items = d.feed(r#"{"type":"result","is_error":true,"result":"weird failure"}"#);
        match &items[0] {
            Decoded::Failed { error, .. } => {
                assert_eq!(error.class, bridge_types::ErrorClass::QuotaExhausted);
                assert!(error.retry_after.is_some());
            }
            other => panic!("{other:?}"),
        }
        // thinking deltas
        let mut d = StreamDecoder::new();
        let it = d.feed(r#"{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}}"#);
        assert_eq!(it, vec![Decoded::Reasoning("hmm".into())]);
        let it = d.feed(r#"{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":""}}}"#);
        assert!(it.is_empty());
    }
}
