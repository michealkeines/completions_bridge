//! IR events → OpenAI response JSON (CONFORMANCE section 6) and SSE chunks (section 7).

use bridge_types::{ChoiceEvent, FinishReason, ToolCall, TurnEvent, Usage};
use bytes::Bytes;
use serde_json::{json, Value};

use crate::error::ApiError;

/// Fixed identifier reported as `system_fingerprint` (stable on purpose: callers may compare it).
pub const SYSTEM_FINGERPRINT: &str = "fp_bridge";

/// Values shared by every object of one response.
#[derive(Debug, Clone)]
pub struct Meta {
    pub id: String,
    pub created: u64,
    pub model: String,
    pub reasoning_to_caller: bool,
    pub include_usage: bool,
}

pub fn new_completion_id() -> String {
    format!("chatcmpl-{}", uuid::Uuid::new_v4().simple())
}

pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn finish_str(f: FinishReason) -> &'static str {
    match f {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolCalls => "tool_calls",
        FinishReason::ContentFilter => "content_filter",
    }
}

/// Usage across choices: prompt = max, completion = sum.
#[derive(Debug, Default, Clone, Copy)]
pub struct UsageAcc {
    prompt: u64,
    completion: u64,
    cached: u64,
    reasoning: u64,
}

impl UsageAcc {
    pub fn add(&mut self, u: &Usage) {
        self.prompt = self.prompt.max(u.prompt_tokens);
        self.cached = self.cached.max(u.cached_tokens);
        self.completion += u.completion_tokens;
        self.reasoning += u.reasoning_tokens;
    }

    pub fn to_json(self) -> Value {
        json!({
            "prompt_tokens": self.prompt,
            "completion_tokens": self.completion,
            "total_tokens": self.prompt + self.completion,
            "prompt_tokens_details": {"cached_tokens": self.cached, "audio_tokens": 0},
            "completion_tokens_details": {
                "reasoning_tokens": self.reasoning,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0
            }
        })
    }
}

fn tool_call_json(c: &ToolCall) -> Value {
    json!({"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}})
}

// ------------------------------------------------------------------ non-streaming

#[derive(Debug, Default, Clone)]
struct ChoiceAcc {
    text: String,
    reasoning: String,
    calls: Vec<ToolCall>,
    finish: Option<FinishReason>,
}

/// Collects events for a non-streaming response.
#[derive(Debug)]
pub struct Collector {
    meta: Meta,
    choices: Vec<ChoiceAcc>,
    usage: UsageAcc,
}

pub enum Collected {
    Continue,
    AllDone,
    Failed(ApiError),
}

impl Collector {
    pub fn new(meta: Meta, n: u32) -> Self {
        Collector { meta, choices: vec![ChoiceAcc::default(); n as usize], usage: UsageAcc::default() }
    }

    pub fn push(&mut self, ev: ChoiceEvent) -> Collected {
        let Some(c) = self.choices.get_mut(ev.index as usize) else {
            tracing::warn!(index = ev.index, "event for unknown choice index ignored");
            return Collected::Continue;
        };
        match ev.event {
            TurnEvent::TextDelta { text } => c.text.push_str(&text),
            TurnEvent::ReasoningDelta { text } => c.reasoning.push_str(&text),
            TurnEvent::ToolCalls { calls } => c.calls.extend(calls),
            // Non-streaming: previews are ignored; the commit builds message.tool_calls.
            TurnEvent::ToolCallStart { .. } | TurnEvent::ToolCallDelta { .. } => {}
            TurnEvent::Usage { usage } => self.usage.add(&usage),
            TurnEvent::Limit { .. } => {}
            TurnEvent::Done { finish } => c.finish = Some(finish),
            TurnEvent::Error { error } => return Collected::Failed(error.into()),
        }
        if self.choices.iter().all(|c| c.finish.is_some()) {
            Collected::AllDone
        } else {
            Collected::Continue
        }
    }

    pub fn is_complete(&self) -> bool {
        self.choices.iter().all(|c| c.finish.is_some())
    }

    pub fn into_json(self) -> Value {
        let choices: Vec<Value> = self
            .choices
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let content = if c.text.is_empty() && !c.calls.is_empty() {
                    Value::Null
                } else {
                    Value::String(c.text.clone())
                };
                let mut message = json!({"role": "assistant", "content": content, "refusal": null});
                if !c.calls.is_empty() {
                    message["tool_calls"] = Value::Array(c.calls.iter().map(tool_call_json).collect());
                }
                if self.meta.reasoning_to_caller && !c.reasoning.is_empty() {
                    message["reasoning_content"] = Value::String(c.reasoning.clone());
                }
                json!({
                    "index": i,
                    "message": message,
                    "logprobs": null,
                    "finish_reason": finish_str(c.finish.unwrap_or(FinishReason::Stop)),
                })
            })
            .collect();
        json!({
            "id": self.meta.id,
            "object": "chat.completion",
            "created": self.meta.created,
            "model": self.meta.model,
            "choices": choices,
            "usage": self.usage.to_json(),
            "service_tier": "default",
            "system_fingerprint": SYSTEM_FINGERPRINT,
        })
    }
}

// ------------------------------------------------------------------ streaming

#[derive(Debug, Default, Clone)]
struct ChoiceState {
    started: bool,
    /// Next free OpenAI `tool_calls[].index` of this choice (shared by streamed and
    /// committed-only calls).
    tool_index: usize,
    /// Calls whose Start was sent as a chunk (ARCHITECTURE 17.3).
    streamed: Vec<StreamedCall>,
    done: bool,
}

#[derive(Debug, Clone)]
struct StreamedCall {
    /// `index` of the IR events.
    event_index: u32,
    /// `tool_calls[].index` on the wire.
    wire_index: usize,
    id: String,
    /// Arguments sent so far, to check the commit against.
    arguments: String,
}

pub fn sse_data(v: &Value) -> Bytes {
    Bytes::from(format!("data: {v}\n\n"))
}

pub fn sse_ping() -> Bytes {
    Bytes::from_static(b": ping\n\n")
}

pub fn sse_done() -> Bytes {
    Bytes::from_static(b"data: [DONE]\n\n")
}

/// Result of feeding one event into the stream encoder.
pub enum Step {
    Chunks(Vec<Bytes>),
    /// Error after headers: emit this and close (no `[DONE]`).
    Error(Bytes),
}

/// Turns `ChoiceEvent`s into `chat.completion.chunk` SSE events.
#[derive(Debug)]
pub struct StreamEncoder {
    meta: Meta,
    choices: Vec<ChoiceState>,
    usage: UsageAcc,
}

impl StreamEncoder {
    pub fn new(meta: Meta, n: u32) -> Self {
        StreamEncoder { meta, choices: vec![ChoiceState::default(); n as usize], usage: UsageAcc::default() }
    }

    fn chunk(&self, index: u32, delta: Value, finish: Option<FinishReason>) -> Bytes {
        let mut v = json!({
            "id": self.meta.id,
            "object": "chat.completion.chunk",
            "created": self.meta.created,
            "model": self.meta.model,
            "service_tier": "default",
            "system_fingerprint": SYSTEM_FINGERPRINT,
            "choices": [{
                "index": index,
                "delta": delta,
                "logprobs": null,
                "finish_reason": finish.map(finish_str),
            }],
        });
        if self.meta.include_usage {
            v["usage"] = Value::Null;
        }
        sse_data(&v)
    }

    pub fn error_event(err: &ApiError) -> Bytes {
        sse_data(&err.body())
    }

    pub fn all_done(&self) -> bool {
        self.choices.iter().all(|c| c.done)
    }

    pub fn push(&mut self, ev: ChoiceEvent) -> Step {
        let idx = ev.index;
        let Some(done) = self.choices.get(idx as usize).map(|s| s.done) else {
            tracing::warn!(index = idx, "event for unknown choice index ignored");
            return Step::Chunks(Vec::new());
        };
        if let TurnEvent::Error { error } = ev.event {
            return Step::Error(Self::error_event(&error.into()));
        }
        if done {
            return Step::Chunks(Vec::new());
        }
        let mut out = Vec::new();
        match ev.event {
            TurnEvent::TextDelta { text } => {
                if !text.is_empty() {
                    self.ensure_started(idx, &mut out);
                    out.push(self.chunk(idx, json!({"content": text}), None));
                }
            }
            TurnEvent::ReasoningDelta { text } => {
                if self.meta.reasoning_to_caller && !text.is_empty() {
                    self.ensure_started(idx, &mut out);
                    out.push(self.chunk(idx, json!({"reasoning_content": text}), None));
                }
            }
            TurnEvent::ToolCallStart { index, id, name } => {
                self.on_call_start(idx, index, id, name, &mut out)
            }
            TurnEvent::ToolCallDelta { index, arguments } => {
                self.on_call_delta(idx, index, arguments, &mut out)
            }
            TurnEvent::ToolCalls { calls } => {
                self.ensure_started(idx, &mut out);
                for c in calls {
                    if self.is_streamed_commit(idx, &c) {
                        continue;
                    }
                    let ti = self.next_tool_index(idx);
                    let delta = json!({"tool_calls": [{
                        "index": ti,
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments},
                    }]});
                    out.push(self.chunk(idx, delta, None));
                }
            }
            TurnEvent::Usage { usage } => self.usage.add(&usage),
            TurnEvent::Limit { .. } => {}
            TurnEvent::Done { finish } => {
                self.ensure_started(idx, &mut out);
                out.push(self.chunk(idx, json!({}), Some(finish)));
                if let Some(s) = self.choices.get_mut(idx as usize) {
                    s.done = true;
                }
            }
            TurnEvent::Error { .. } => {}
        }
        Step::Chunks(out)
    }

    /// First chunk of a choice carries the role.
    fn ensure_started(&mut self, idx: u32, out: &mut Vec<Bytes>) {
        if let Some(s) = self.choices.get_mut(idx as usize) {
            if !s.started {
                s.started = true;
                out.push(self.chunk(idx, json!({"role": "assistant", "content": ""}), None));
            }
        }
    }

    /// Allocate the next `tool_calls[].index` of a choice.
    fn next_tool_index(&mut self, idx: u32) -> usize {
        match self.choices.get_mut(idx as usize) {
            Some(s) => {
                s.tool_index += 1;
                s.tool_index - 1
            }
            None => 0,
        }
    }

    /// Start → first `tool_calls` delta of the call: index, id, type, name, empty arguments.
    fn on_call_start(&mut self, idx: u32, index: u32, id: String, name: String, out: &mut Vec<Bytes>) {
        let already = self
            .choices
            .get(idx as usize)
            .is_some_and(|s| s.streamed.iter().any(|c| c.event_index == index || c.id == id));
        if already {
            tracing::warn!(choice = idx, index, "duplicate ToolCallStart ignored");
            return;
        }
        self.ensure_started(idx, out);
        let wire_index = self.next_tool_index(idx);
        if let Some(s) = self.choices.get_mut(idx as usize) {
            s.streamed.push(StreamedCall { event_index: index, wire_index, id: id.clone(), arguments: String::new() });
        }
        let delta = json!({"tool_calls": [{
            "index": wire_index,
            "id": id,
            "type": "function",
            "function": {"name": name, "arguments": ""},
        }]});
        out.push(self.chunk(idx, delta, None));
    }

    /// Delta → `tool_calls` delta carrying only an arguments fragment.
    fn on_call_delta(&mut self, idx: u32, index: u32, arguments: String, out: &mut Vec<Bytes>) {
        if arguments.is_empty() {
            return;
        }
        let Some(call) = self
            .choices
            .get_mut(idx as usize)
            .and_then(|s| s.streamed.iter_mut().find(|c| c.event_index == index))
        else {
            tracing::warn!(choice = idx, index, "ToolCallDelta without ToolCallStart ignored");
            return;
        };
        call.arguments.push_str(&arguments);
        let wire_index = call.wire_index;
        let delta = json!({"tool_calls": [{"index": wire_index, "function": {"arguments": arguments}}]});
        out.push(self.chunk(idx, delta, None));
    }

    /// True if the committed call was already streamed (then nothing more is sent). A
    /// commit whose arguments differ from the streamed ones breaks the IR contract: the
    /// caller keeps what was streamed, a second copy would corrupt its accumulation.
    fn is_streamed_commit(&self, idx: u32, c: &ToolCall) -> bool {
        let Some(sc) = self
            .choices
            .get(idx as usize)
            .and_then(|s| s.streamed.iter().find(|sc| sc.id == c.id))
        else {
            return false;
        };
        if sc.arguments != c.arguments {
            tracing::warn!(
                choice = idx,
                call_id = %c.id,
                streamed_len = sc.arguments.len(),
                committed_len = c.arguments.len(),
                "committed tool call arguments differ from the streamed ones; keeping the streamed version"
            );
        }
        true
    }

    /// Final usage chunk (if requested) and `[DONE]`.
    pub fn finish(&self) -> Vec<Bytes> {
        let mut out = Vec::new();
        if self.meta.include_usage {
            let v = json!({
                "id": self.meta.id,
                "object": "chat.completion.chunk",
                "created": self.meta.created,
                "model": self.meta.model,
                "service_tier": "default",
                "system_fingerprint": SYSTEM_FINGERPRINT,
                "choices": [],
                "usage": self.usage.to_json(),
            });
            out.push(sse_data(&v));
        }
        out.push(sse_done());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bridge_types::{BridgeError, ErrorClass};
    use std::collections::BTreeMap;

    fn meta() -> Meta {
        Meta { id: "chatcmpl-x".into(), created: 1, model: "m".into(), reasoning_to_caller: false, include_usage: false }
    }

    fn ce(index: u32, event: TurnEvent) -> ChoiceEvent {
        ChoiceEvent { index, event }
    }
    fn start(c: u32, index: u32, id: &str, name: &str) -> ChoiceEvent {
        ce(c, TurnEvent::ToolCallStart { index, id: id.into(), name: name.into() })
    }
    fn delta(c: u32, index: u32, a: &str) -> ChoiceEvent {
        ce(c, TurnEvent::ToolCallDelta { index, arguments: a.into() })
    }
    fn tc(id: &str, name: &str, args: &str) -> ToolCall {
        ToolCall { id: id.into(), name: name.into(), arguments: args.into() }
    }
    fn commit(c: u32, calls: Vec<ToolCall>) -> ChoiceEvent {
        ce(c, TurnEvent::ToolCalls { calls })
    }
    fn done(c: u32) -> ChoiceEvent {
        ce(c, TurnEvent::Done { finish: FinishReason::ToolCalls })
    }

    fn parse(b: &Bytes) -> Value {
        let s = std::str::from_utf8(b).unwrap();
        let d = s.strip_prefix("data: ").unwrap().strip_suffix("\n\n").unwrap();
        serde_json::from_str(d).unwrap()
    }

    /// Feed events; return every chunk as JSON (an error event ends the run).
    fn run(n: u32, events: Vec<ChoiceEvent>) -> (Vec<Value>, Option<Value>) {
        let mut enc = StreamEncoder::new(meta(), n);
        let mut out = Vec::new();
        for e in events {
            match enc.push(e) {
                Step::Chunks(cs) => out.extend(cs.iter().map(parse)),
                Step::Error(b) => return (out, Some(parse(&b))),
            }
        }
        assert!(enc.all_done());
        (out, None)
    }

    /// (choice, delta, finish_reason) per chunk.
    fn seq(chunks: &[Value]) -> Vec<(u64, Value, Value)> {
        chunks
            .iter()
            .map(|v| {
                let c = &v["choices"][0];
                (c["index"].as_u64().unwrap(), c["delta"].clone(), c["finish_reason"].clone())
            })
            .collect()
    }

    /// Accumulate tool calls the way the official SDKs do: per choice, per
    /// `tool_calls[].index`, first fragment sets id/name, `arguments` are concatenated.
    fn sdk_accumulate(chunks: &[Value]) -> BTreeMap<(u64, u64), ToolCall> {
        let mut acc: BTreeMap<(u64, u64), ToolCall> = BTreeMap::new();
        for v in chunks {
            let c = &v["choices"][0];
            let ci = c["index"].as_u64().unwrap();
            let Some(tcs) = c["delta"]["tool_calls"].as_array() else { continue };
            for t in tcs {
                let ti = t["index"].as_u64().unwrap();
                let e = acc.entry((ci, ti)).or_insert_with(|| tc("", "", ""));
                if let Some(id) = t["id"].as_str() {
                    e.id.push_str(id);
                }
                if let Some(n) = t["function"]["name"].as_str() {
                    e.name.push_str(n);
                }
                if let Some(a) = t["function"]["arguments"].as_str() {
                    e.arguments.push_str(a);
                }
            }
        }
        acc
    }

    const ARGS: &str = "{\"path\": \"a\\\"b.txt\",\n \"content\":\"héllo wörld\"}";

    #[test]
    fn streamed_call_chunk_sequence() {
        let (chunks, err) = run(1, vec![
            start(0, 0, "call_a", "write"),
            delta(0, 0, &ARGS[..9]),
            delta(0, 0, ""),
            delta(0, 0, &ARGS[9..]),
            commit(0, vec![tc("call_a", "write", ARGS)]),
            done(0),
        ]);
        assert!(err.is_none());
        let expected = vec![
            (0, json!({"role": "assistant", "content": ""}), Value::Null),
            (0, json!({"tool_calls": [{"index": 0, "id": "call_a", "type": "function", "function": {"name": "write", "arguments": ""}}]}), Value::Null),
            (0, json!({"tool_calls": [{"index": 0, "function": {"arguments": &ARGS[..9]}}]}), Value::Null),
            (0, json!({"tool_calls": [{"index": 0, "function": {"arguments": &ARGS[9..]}}]}), Value::Null),
            (0, json!({}), json!("tool_calls")),
        ];
        assert_eq!(seq(&chunks), expected);
    }

    #[test]
    fn text_then_streamed_call_role_chunk_once() {
        let (chunks, _) = run(1, vec![
            ce(0, TurnEvent::TextDelta { text: "Writing.".into() }),
            start(0, 0, "call_a", "write"),
            delta(0, 0, "{}"),
            commit(0, vec![tc("call_a", "write", "{}")]),
            done(0),
        ]);
        let s = seq(&chunks);
        assert_eq!(s.len(), 5);
        assert_eq!(s.iter().filter(|(_, d, _)| d.get("role").is_some()).count(), 1);
        assert_eq!(s[1].1, json!({"content": "Writing."}));
    }

    #[test]
    fn two_streamed_parallel_calls() {
        let (chunks, _) = run(1, vec![
            start(0, 0, "call_a", "read"),
            delta(0, 0, "{\"x\":"),
            delta(0, 0, "1}"),
            start(0, 1, "call_b", "glob"),
            delta(0, 1, "{\"p\":\"*\"}"),
            commit(0, vec![tc("call_a", "read", "{\"x\":1}"), tc("call_b", "glob", "{\"p\":\"*\"}")]),
            done(0),
        ]);
        // role, 2 starts, 3 deltas, finish: the commit adds nothing.
        assert_eq!(chunks.len(), 7);
        let acc = sdk_accumulate(&chunks);
        assert_eq!(acc.len(), 2);
        assert_eq!(acc[&(0, 0)], tc("call_a", "read", "{\"x\":1}"));
        assert_eq!(acc[&(0, 1)], tc("call_b", "glob", "{\"p\":\"*\"}"));
        // Starts carry type function; deltas carry only arguments.
        assert_eq!(chunks[4]["choices"][0]["delta"]["tool_calls"][0]["index"], 1);
        assert_eq!(chunks[4]["choices"][0]["delta"]["tool_calls"][0]["type"], "function");
        assert!(chunks[5]["choices"][0]["delta"]["tool_calls"][0].get("id").is_none());
    }

    #[test]
    fn committed_only_calls_unchanged() {
        let calls = vec![tc("call_a", "read", "{\"x\":1}"), tc("call_b", "glob", "{}")];
        let (chunks, _) = run(1, vec![commit(0, calls.clone()), done(0)]);
        assert_eq!(chunks.len(), 4);
        let t0 = &chunks[1]["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(t0, &json!({"index": 0, "id": "call_a", "type": "function", "function": {"name": "read", "arguments": "{\"x\":1}"}}));
        let acc = sdk_accumulate(&chunks);
        assert_eq!(acc.values().cloned().collect::<Vec<_>>(), calls);
    }

    #[test]
    fn mix_of_streamed_and_committed_only_uses_next_free_index() {
        let (chunks, _) = run(1, vec![
            start(0, 0, "call_a", "read"),
            delta(0, 0, "{}"),
            commit(0, vec![tc("call_a", "read", "{}"), tc("call_b", "glob", "{\"p\":1}")]),
            done(0),
        ]);
        let acc = sdk_accumulate(&chunks);
        assert_eq!(acc.len(), 2);
        assert_eq!(acc[&(0, 0)], tc("call_a", "read", "{}"));
        assert_eq!(acc[&(0, 1)], tc("call_b", "glob", "{\"p\":1}"));
    }

    #[test]
    fn mismatching_commit_does_not_duplicate() {
        let (chunks, _) = run(1, vec![
            start(0, 0, "call_a", "read"),
            delta(0, 0, "{\"x\": 1}"),
            commit(0, vec![tc("call_a", "read", "{\"x\":1}")]),
            done(0),
        ]);
        let acc = sdk_accumulate(&chunks);
        assert_eq!(acc.len(), 1);
        assert_eq!(acc[&(0, 0)].arguments, "{\"x\": 1}");
        assert_eq!(chunks.len(), 4);
    }

    #[test]
    fn delta_without_start_and_duplicate_start_ignored() {
        let (chunks, _) = run(1, vec![
            delta(0, 3, "{}"),
            start(0, 0, "call_a", "read"),
            start(0, 0, "call_a", "read"),
            delta(0, 0, "{}"),
            commit(0, vec![tc("call_a", "read", "{}")]),
            done(0),
        ]);
        let acc = sdk_accumulate(&chunks);
        assert_eq!(acc.len(), 1);
        assert_eq!(acc[&(0, 0)], tc("call_a", "read", "{}"));
    }

    #[test]
    fn n2_interleaved_streamed_calls_have_per_choice_indices() {
        let (chunks, _) = run(2, vec![
            start(0, 0, "call_a", "read"),
            start(1, 0, "call_x", "glob"),
            delta(1, 0, "{\"p\":"),
            delta(0, 0, "{\"x\":"),
            delta(0, 0, "1}"),
            start(1, 1, "call_y", "read"),
            delta(1, 0, "2}"),
            delta(1, 1, "{}"),
            commit(1, vec![tc("call_x", "glob", "{\"p\":2}"), tc("call_y", "read", "{}"), tc("call_z", "ls", "{}")]),
            done(1),
            commit(0, vec![tc("call_a", "read", "{\"x\":1}")]),
            done(0),
        ]);
        let acc = sdk_accumulate(&chunks);
        assert_eq!(acc.len(), 4);
        assert_eq!(acc[&(0, 0)], tc("call_a", "read", "{\"x\":1}"));
        assert_eq!(acc[&(1, 0)], tc("call_x", "glob", "{\"p\":2}"));
        assert_eq!(acc[&(1, 1)], tc("call_y", "read", "{}"));
        assert_eq!(acc[&(1, 2)], tc("call_z", "ls", "{}"));
        // Each choice starts with exactly one role chunk, before its first tool chunk.
        for ci in [0u64, 1] {
            let mine: Vec<_> = seq(&chunks).into_iter().filter(|(c, _, _)| *c == ci).collect();
            assert!(mine[0].1.get("role").is_some());
            assert_eq!(mine.iter().filter(|(_, d, _)| d.get("role").is_some()).count(), 1);
            assert_eq!(mine.last().unwrap().2, json!("tool_calls"));
        }
    }

    #[test]
    fn error_after_partial_streamed_call_is_error_event() {
        let (chunks, err) = run(1, vec![
            start(0, 0, "call_a", "write"),
            delta(0, 0, "{\"content\":\"par"),
            ce(0, TurnEvent::Error { error: BridgeError::new(ErrorClass::Unavailable, "The server is overloaded.") }),
        ]);
        assert_eq!(chunks.len(), 3);
        let err = err.unwrap();
        assert!(err["error"]["message"].is_string());
        assert_eq!(err["error"]["code"], "overloaded");
    }

    #[test]
    fn sdk_accumulation_equals_commit_for_large_fragmented_call() {
        let content: String = (0..2000).map(|i| format!("line {i} \"q\" \\ é\n")).collect();
        let args = serde_json::to_string(&json!({"path": "big.txt", "content": content})).unwrap();
        let mut events = vec![start(0, 0, "call_w", "write")];
        let mut rest = args.as_str();
        while !rest.is_empty() {
            let mut cut = rest.len().min(37);
            while !rest.is_char_boundary(cut) {
                cut += 1;
            }
            events.push(delta(0, 0, &rest[..cut]));
            rest = &rest[cut..];
        }
        events.push(commit(0, vec![tc("call_w", "write", &args)]));
        events.push(done(0));
        let (chunks, _) = run(1, events);
        assert!(chunks.len() > 100);
        let acc = sdk_accumulate(&chunks);
        assert_eq!(acc.len(), 1);
        assert_eq!(acc[&(0, 0)], tc("call_w", "write", &args));
    }

    #[test]
    fn collector_ignores_previews() {
        let mut col = Collector::new(meta(), 1);
        for e in [
            start(0, 0, "call_a", "read"),
            delta(0, 0, "{\"x\":"),
            delta(0, 0, "1}"),
            commit(0, vec![tc("call_a", "read", "{\"x\":1}")]),
            done(0),
        ] {
            let _ = col.push(e);
        }
        assert!(col.is_complete());
        let v = col.into_json();
        assert_eq!(v["choices"][0]["message"]["tool_calls"], json!([
            {"id": "call_a", "type": "function", "function": {"name": "read", "arguments": "{\"x\":1}"}}
        ]));
        assert_eq!(v["choices"][0]["message"]["content"], Value::Null);
    }
}
