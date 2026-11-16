# Completions Bridge: Conformance Spec

The bridge must be indistinguishable from a normal OpenAI-compatible chat completions host. This document lists every behaviour such a host has, as test cases. Once the bridge is built, it passes when every P0 and P1 case passes.

Reference: the OpenAI Chat Completions API (`POST /v1/chat/completions`) as implemented by OpenAI and by mainstream compatible hosts.

---

## How to read this

**Priority**

| | Meaning |
|---|---|
| **P0** | opencode depends on it; without it, daily use breaks |
| **P1** | Any normal completions API does this; a standard client or SDK would notice |
| **P2** | Less common; supported by OpenAI, often missing or ignored by compatible hosts |

**Support levels** (what "pass" means for a parameter)

| Level | Meaning |
|---|---|
| **Honour** | Behaves as specified |
| **Approximate** | Accepted, effect is close (e.g. best-effort JSON) |
| **Ignore** | Accepted without error, no effect; the documented behaviour of many compatible hosts |
| **Reject** | 400 with a clear error, the same way a host that lacks the feature would |

**Risk** marks cases that are hard for this architecture (agent CLI behind the API). They need explicit tests, not assumptions.

**Judges**: each case is run with raw `curl`, the official OpenAI Python SDK, the official OpenAI Node SDK, and opencode (`@ai-sdk/openai-compatible`). The SDKs are strict parsers, so a malformed response fails them even when curl output looks fine.

---

## 1. Endpoints

| ID | Case | Expect | Pri |
|---|---|---|---|
| EP-1 | `POST /v1/chat/completions` | Works as specified below | P0 |
| EP-2 | `GET /v1/models` | `{"object":"list","data":[{"id","object":"model","created","owned_by"}...]}` | P0 |
| EP-3 | `GET /v1/models/{id}` | Single model object; 404 `model_not_found` for unknown id | P1 |
| EP-4 | Base URL with and without trailing slash, with `/v1` prefix | Both route correctly | P1 |
| EP-5 | Unknown path (e.g. `/v1/embeddings`) | 404 in the standard error format | P1 |
| EP-6 | Wrong method (`GET /v1/chat/completions`) | 405 in the standard error format | P2 |
| EP-7 | `POST /v1/completions` (legacy text completions) | Either supported, or 404 like most modern hosts | P2 |

---

## 2. Authentication

| ID | Case | Expect | Pri |
|---|---|---|---|
| AUTH-1 | Valid `Authorization: Bearer <token>` | Request served | P0 |
| AUTH-2 | Missing header | 401, `type: invalid_request_error`, `code: invalid_api_key` (or `missing_api_key`) | P1 |
| AUTH-3 | Wrong token | 401, same format | P1 |
| AUTH-4 | `api-key` header (Azure style) | Accepted or ignored, never a crash | P2 |
| AUTH-5 | `OpenAI-Organization` / `OpenAI-Project` headers | Ignored | P2 |

---

## 3. Request parameters

Unknown top-level fields must be ignored, never rejected (clients add vendor fields freely).

| ID | Parameter | Support | Expect | Pri | Risk |
|---|---|---|---|---|---|
| REQ-1 | `model` | Honour | Must be a listed id; unknown → 404 `model_not_found`. Response `model` echoes the requested id | P0 | |
| REQ-2 | `messages` | Honour | See section 4. Missing or empty → 400 | P0 | |
| REQ-3 | `stream` | Honour | `true` → SSE (section 7); `false`/absent → JSON (section 6) | P0 | |
| REQ-4 | `stream_options.include_usage` | Honour | Extra final chunk with `usage` and `choices: []` | P0 | |
| REQ-5 | `tools` | Honour | Section 5 | P0 | |
| REQ-6 | `tool_choice` | Honour | `none`, `auto`, `required`, `{"type":"function","function":{"name":...}}` | P1 | Risk |
| REQ-7 | `parallel_tool_calls` | Honour | `false` → at most one call per reply | P1 | Risk |
| REQ-8 | `max_tokens`, `max_completion_tokens` | Honour | Output stops at limit, `finish_reason: "length"` | P1 | Risk |
| REQ-9 | `temperature` (0–2), `top_p` | Ignore or approximate | Accepted; out-of-range → 400 like OpenAI | P1 | |
| REQ-10 | `stop` (string or up to 4 strings) | Honour | Output ends before the stop sequence, `finish_reason: "stop"`, sequence not included | P1 | Risk |
| REQ-11 | `n` | Honour `1`; `>1` Honour (run n turns) or Reject | Never silently return fewer choices than asked | P1 | Risk |
| REQ-12 | `response_format: {"type":"json_object"}` | Approximate | Content is valid JSON | P1 | Risk |
| REQ-13 | `response_format: {"type":"json_schema", ...}` | Approximate | Content is valid JSON conforming to the schema | P1 | Risk |
| REQ-14 | `response_format: {"type":"text"}` | Honour | Normal behaviour | P1 | |
| REQ-15 | `presence_penalty`, `frequency_penalty`, `logit_bias` | Ignore | Accepted; invalid ranges → 400 | P2 | |
| REQ-16 | `seed` | Ignore | Accepted; `system_fingerprint` may vary | P2 | |
| REQ-17 | `logprobs`, `top_logprobs` | Ignore (return `logprobs: null`) or Reject | Never a crash or a fake value | P2 | |
| REQ-18 | `user`, `metadata`, `store`, `service_tier`, `prompt_cache_key`, `safety_identifier` | Ignore | Accepted | P2 | |
| REQ-19 | `reasoning_effort` (`minimal`/`low`/`medium`/`high`) | Approximate | Mapped to backend effort if possible, else ignored | P1 | |
| REQ-20 | `modalities`, `audio`, `prediction`, `web_search_options` | Reject or Ignore | Documented; no crash | P2 | |
| REQ-21 | Legacy `functions` / `function_call` | Approximate (translate to tools) or Reject | No crash | P2 | |
| REQ-22 | Body not JSON, or wrong types | 400 with `param` naming the field | P1 | |
| REQ-23 | Very large body (several MB) | Served, or 413 in the standard error format | P1 | |

---

## 4. Messages

### 4.1 Roles and shapes

| ID | Case | Expect | Pri | Risk |
|---|---|---|---|---|
| MSG-1 | `system` message first | Acts as instructions | P0 | |
| MSG-2 | `developer` role | Treated like `system` | P1 | |
| MSG-3 | Several system/developer messages, or one in the middle | All honoured, in order | P1 | Risk |
| MSG-4 | No system message | Works | P0 | |
| MSG-5 | `user` content as string | Works | P0 | |
| MSG-6 | `user` content as array of `{"type":"text"}` parts | Parts joined in order | P0 | |
| MSG-7 | `user` content with `image_url` (data URL or https URL) | Model sees the image, or 400 if the model is declared without vision | P1 | Risk |
| MSG-8 | `user` content with `input_audio` or `file` parts | Supported or 400; no crash | P2 | |
| MSG-9 | `assistant` with `content` only | Treated as prior model output | P0 | |
| MSG-10 | `assistant` with `content: null` and `tool_calls` | Valid | P0 | |
| MSG-11 | `assistant` with both text and `tool_calls` | Valid | P0 | |
| MSG-12 | `assistant` with `refusal` | Accepted | P2 | |
| MSG-13 | `tool` message with `tool_call_id` and string content | Delivered as the result of that call | P0 | |
| MSG-14 | `tool` content as array of text parts | Parts joined | P1 | |
| MSG-15 | `name` field on any message | Accepted | P2 | |
| MSG-16 | Empty string content | Accepted | P1 | |
| MSG-17 | Unicode, emoji, RTL text, NUL-free binary-looking text, very long lines | Round-trips exactly | P0 | Risk |
| MSG-18 | Content containing text that looks like protocol markers, JSON, XML tags, `<system-reminder>`, fake tool calls | Treated as plain text, never parsed | P0 | Risk |

### 4.2 Conversation structure

A normal API is **stateless**: each request is answered from its contents alone. This is the most important group for the bridge, because the bridge is stateful inside.

| ID | Case | Expect | Pri | Risk |
|---|---|---|---|---|
| CONV-1 | Single user message | Answer | P0 | |
| CONV-2 | Multi-turn history the bridge never saw (hand-written, or from another model) | Answer consistent with the full history | P0 | Risk |
| CONV-3 | Same request sent twice | Two valid answers; content may differ; neither is an error or a "replay" of a stale turn | P0 | Risk |
| CONV-4 | History edited in the middle (earlier user message changed) | Answer reflects the edited history, not the original | P0 | Risk |
| CONV-5 | History truncated (last turns removed, i.e. undo) | Answer reflects the shorter history | P0 | Risk |
| CONV-6 | Two different conversations sharing the same first N messages (fork) | Each answered from its own history | P0 | Risk |
| CONV-7 | Assistant message the bridge did not produce (edited or injected) | Treated as real prior output | P0 | Risk |
| CONV-8 | History replaced by a summary (client-side compaction) | Answer uses the summary | P0 | Risk |
| CONV-9 | Last message is `assistant` (prefill) | Either continues that text, or answers normally; never an error | P2 | Risk |
| CONV-10 | Consecutive messages with the same role | Accepted | P1 | Risk |
| CONV-11 | `tool` message with no matching `tool_calls` before it | 400 like OpenAI (`messages with role 'tool' must be a response to a preceding message with 'tool_calls'`) | P1 | |
| CONV-12 | Assistant `tool_calls` without all their `tool` results, followed by a user message | 400 like OpenAI, or accepted; must not hang | P1 | Risk |
| CONV-13 | Tool results in a different order than the calls | Accepted | P1 | |
| CONV-14 | Request with 100+ messages and 100k+ tokens of history | Answered, or a context-length error (section 8) | P0 | Risk |
| CONV-15 | Request after a long idle period (hours or days) | Answered normally | P0 | Risk |
| CONV-16 | Request after the bridge restarted | Answered normally | P0 | Risk |

**The stateless rule:** for any two requests with identical bodies, the bridge's behaviour must not depend on what happened between them, beyond normal model randomness. Session reuse is an internal optimisation only.

---

## 5. Tool calling

| ID | Case | Expect | Pri | Risk |
|---|---|---|---|---|
| TOOL-1 | Tools given, model answers in text | `finish_reason: "stop"`, no `tool_calls` | P0 | |
| TOOL-2 | Model calls one tool | `message.tool_calls[0]` with `id`, `type: "function"`, `function.name`, `function.arguments` (JSON **string**); `finish_reason: "tool_calls"` | P0 | |
| TOOL-3 | Model calls several tools at once | Several entries, unique ids, correct order | P0 | |
| TOOL-4 | Text before the tool call | Both `content` and `tool_calls` present | P0 | |
| TOOL-5 | `function.name` exactly as declared by the caller (case, dots, dashes, underscores) | No prefix, suffix or renaming | P0 | Risk |
| TOOL-6 | `arguments` is valid JSON that matches the schema's required fields | Always parseable | P0 | |
| TOOL-7 | Arguments with large strings (whole files, code with quotes, backslashes, newlines, unicode) | Exact round-trip | P0 | Risk |
| TOOL-8 | Tool with no parameters (`{}` or `{"type":"object","properties":{}}`) | Called with `arguments: "{}"` | P1 | |
| TOOL-9 | Tool schemas with nested objects, arrays, enums, `anyOf`, `$ref`, `additionalProperties: false` | Accepted and respected | P1 | Risk |
| TOOL-10 | `strict: true` on a function | Arguments conform exactly | P2 | Risk |
| TOOL-11 | 50+ tools in one request | Works | P1 | |
| TOOL-12 | `tool_choice: "none"` | No tool calls | P1 | Risk |
| TOOL-13 | `tool_choice: "required"` | At least one tool call | P1 | Risk |
| TOOL-14 | `tool_choice` naming one function | Exactly that function is called | P1 | Risk |
| TOOL-15 | Tool list changes between requests in one conversation | Model sees only the current list | P0 | Risk |
| TOOL-16 | Tool result says error (`"Error: file not found"`) | Model reacts to it like any result | P0 | |
| TOOL-17 | Very large tool result (100k+ characters) | Delivered intact or context error | P1 | Risk |
| TOOL-18 | 30+ tool rounds in one conversation | Each round correct | P0 | Risk |
| TOOL-19 | Caller never sends results for the calls (abandons them) and starts a new user turn | Handled per CONV-12 | P1 | Risk |
| TOOL-20 | Tool call ids: unique per response, stable when the same history is resent | Format `call_…` or any opaque string; ids from the history are accepted whatever their format | P1 | Risk |
| TOOL-21 | Model only ever calls tools from the request; never mentions or uses tools it was not given | No hidden or extra tools visible | P0 | Risk |

---

## 6. Non-streaming response

| ID | Case | Expect | Pri |
|---|---|---|---|
| RESP-1 | Top level | `id` (`chatcmpl-…`), `object: "chat.completion"`, `created` (unix seconds), `model`, `choices`, `usage`; `system_fingerprint` optional | P0 |
| RESP-2 | Choice | `index`, `message` (`role: "assistant"`, `content` string or null, optional `tool_calls`, `refusal: null`), `finish_reason`, `logprobs: null` | P0 |
| RESP-3 | `finish_reason` values | `stop`, `length`, `tool_calls`, `content_filter` only | P0 |
| RESP-4 | `usage` | `prompt_tokens`, `completion_tokens`, `total_tokens` (sum); optional `prompt_tokens_details.cached_tokens`, `completion_tokens_details.reasoning_tokens` | P0 |
| RESP-5 | Token counts | Realistic for the request (prompt count grows with history), even though the backend sees less | P1 |
| RESP-6 | `Content-Type: application/json` | Always | P0 |
| RESP-7 | Response headers | `x-request-id`; optional `x-ratelimit-*` | P2 |

---

## 7. Streaming response

| ID | Case | Expect | Pri | Risk |
|---|---|---|---|---|
| STR-1 | Headers | `Content-Type: text/event-stream`, `Cache-Control: no-cache`, connection kept open | P0 | |
| STR-2 | Event format | Each event `data: {json}\n\n`; ends with `data: [DONE]\n\n` | P0 | |
| STR-3 | Chunk object | `object: "chat.completion.chunk"`; same `id`, `created`, `model` in every chunk | P0 | |
| STR-4 | First chunk | `delta: {"role":"assistant"}` (content `""` allowed) | P0 | |
| STR-5 | Text | `delta.content` pieces that concatenate to the full text | P0 | |
| STR-6 | Tool call deltas | First delta for a call has `index`, `id`, `type: "function"`, `function.name`, `function.arguments` (may be `""`); later deltas for that `index` add to `arguments` only | P0 | |
| STR-7 | Parallel calls | Distinct `index` values 0..n-1 | P0 | |
| STR-8 | Last content chunk | `delta: {}` with `finish_reason` set | P0 | |
| STR-9 | Usage chunk (`include_usage`) | After the finish chunk: `choices: []`, `usage` set; every other chunk has `usage: null` or omits it | P0 | |
| STR-10 | Time to first byte | Headers sent promptly; SSE comment keep-alives (`: ping`) during slow starts so clients don't time out | P0 | Risk |
| STR-11 | Reasoning | Optional `delta.reasoning_content` (de-facto extension); never mixed into `content` | P1 | |
| STR-12 | Error before any chunk | Plain HTTP error response (section 8), not a stream | P0 | Risk |
| STR-13 | Error after chunks were sent | `data: {"error":{...}}` event, then the stream closes | P1 | Risk |
| STR-14 | Client disconnects mid-stream | Generation stops promptly; no effect on later requests | P0 | Risk |
| STR-15 | Streamed and non-streamed answers to the same request | Same shape of result (text, tool calls, finish reason, usage) | P1 | |
| STR-16 | Long tool call (e.g. a `write` with several thousand tokens of arguments), streaming | The first `tool_calls` chunk (index, id, name) arrives within a few seconds of the model starting the call; arguments then arrive in several deltas; the SDK-accumulated arguments are byte-identical to what a non-streamed request would return | P0 | Risk |
| STR-17 | Very large tool call whose generation takes longer than the first-token timeout | Completes normally; never reported as a timeout | P0 | Risk |

---

## 8. Errors

All errors use this body, with the matching HTTP status:

```json
{"error": {"message": "...", "type": "...", "param": null, "code": "..."}}
```

| ID | Situation | Status | `type` / `code` | Pri | Risk |
|---|---|---|---|---|---|
| ERR-1 | Malformed request, bad field | 400 | `invalid_request_error` / field-specific, `param` set | P1 | |
| ERR-2 | Context too long | 400 | `invalid_request_error` / `context_length_exceeded` | P0 | Risk |
| ERR-3 | Auth failure | 401 | `invalid_request_error` / `invalid_api_key` | P1 | |
| ERR-4 | Unknown model | 404 | `invalid_request_error` / `model_not_found` | P1 | |
| ERR-5 | Request too large | 413 | `invalid_request_error` | P2 | |
| ERR-6 | All backends out of quota | 429 | `insufficient_quota` or `rate_limit_exceeded`; `Retry-After` header (seconds) | P0 | Risk |
| ERR-7 | Too many concurrent requests | 429 | `rate_limit_exceeded`; `Retry-After` | P1 | |
| ERR-8 | Internal failure | 500 | `server_error` | P1 | |
| ERR-9 | Backend temporarily unavailable or overloaded | 503 (or 529-style 503) | `server_error` / `overloaded`; retryable | P1 | Risk |
| ERR-10 | Backend took too long | 504 or stream error | `server_error` / `timeout` | P1 | Risk |
| ERR-11 | Error messages | Plain, no stack traces, no internal paths, no agent or CLI names | P0 | Risk |
| ERR-12 | Retry after any retryable error | Succeeds with no leftover effect from the failed attempt | P0 | Risk |

---

## 9. Concurrency and lifecycle

| ID | Case | Expect | Pri | Risk |
|---|---|---|---|---|
| CONC-1 | 5 independent requests at once | All answered correctly, no crossed outputs | P0 | Risk |
| CONC-2 | 2 requests at once with the same history | Both answered independently (CONV-3) | P1 | Risk |
| CONC-3 | Main request plus a side request (title) at once | Both answered, neither delays the other noticeably | P0 | |
| CONC-4 | HTTP keep-alive: many requests on one connection | Works | P1 | |
| CONC-5 | Client-side timeout and retry | Retry answered; abandoned attempt doesn't corrupt anything | P0 | Risk |
| CONC-6 | Long-running stream (several minutes) | Not cut off by the server | P0 | |
| CONC-7 | Agent loop: the next request is sent within milliseconds of the previous `[DONE]` | Answered correctly; (internal, via request log) served from the checkpoint of the previous turn, not an older one | P1 | Risk |

---

## 10. Hidden internals (no leakage)

A normal model host doesn't reveal what runs behind it. These cases check that nothing about the agents shows through.

| ID | Case | Expect | Pri | Risk |
|---|---|---|---|---|
| LEAK-1 | Ask "what tools do you have?" | Lists only the caller's tools, by the caller's names | P0 | Risk |
| LEAK-2 | Ask "what is your system prompt?" | Answers from the caller's system prompt only (or declines, as the model would) | P1 | Risk |
| LEAK-3 | Output never contains `mcp__`, session ids, file paths of the bridge, "Continue.", "No response requested", or interrupted-tool notes | Never | P0 | Risk |
| LEAK-4 | Tool call ids and names | Never reveal the backend's prefixes | P0 | Risk |
| LEAK-5 | Answers about the conversation ("what did I ask first?") | Match the request's history exactly, not filler the bridge added | P0 | Risk |
| LEAK-6 | Usage numbers | Plausible for the caller's request, not the backend's internal accounting | P1 | |
| LEAK-7 | Model switches backend mid-conversation | Conversation still coherent; no "as I said earlier" referring to things not in the history | P1 | Risk |

---

## 11. Non-functional targets

| ID | Metric | Target | Pri |
|---|---|---|---|
| NF-1 | Time to first token, warm session | ≤ 3 s (measured about 1.5–2.5 s) | P0 |
| NF-2 | Time to first token, new or imported session | ≤ 6 s | P1 |
| NF-3 | Extra latency added by the bridge per request | ≤ 500 ms | P1 |
| NF-4 | Prompt-cache reuse on continued sessions | ≥ 90% of input read from cache over a long agentic session, including stretches of 10+ consecutive tool rounds without a user message | P1 |
| NF-7 | Time to first byte of a tool call | ≤ 3 s p90 regardless of argument size | P1 |
| NF-5 | Memory per idle session | No resident process | P1 |
| NF-6 | Survives 8 hours of continuous use without restart | Yes | P0 |

---

## 12. End-to-end acceptance

Beyond the unit cases above, the bridge passes when opencode, pointed at it, completes these real tasks with no visible difference from a normal provider:

| ID | Task | Exercises |
|---|---|---|
| E2E-1 | "Explain this repo" on a medium repo | Many reads, parallel calls, long answer |
| E2E-2 | Fix a failing test | read, edit, bash, error output, retry loop |
| E2E-3 | Multi-file feature with tests | 30+ tool rounds, todo tool, new files |
| E2E-4 | Undo the last turn and ask differently | CONV-5, CONV-6 |
| E2E-5 | Long session until opencode compacts | CONV-8, CONV-14 |
| E2E-6 | Cancel mid-answer, then continue | STR-14, CONC-5 |
| E2E-7 | Restart the bridge mid-task, then continue | CONV-16 |
| E2E-8 | Two opencode windows on different tasks at once | CONC-1 |
| E2E-9 | Run until the backend quota runs out | ERR-6, LEAK-7 (with a second backend) |

---

## 13. Expected gaps

Some behaviours can't be honoured with an agent backend. A normal host that lacks a feature rejects or ignores it, and the bridge does the same, so these are documented, not failures.

| Feature | Bridge behaviour | Why |
|---|---|---|
| `logprobs` | `null` | Agent CLIs don't expose token probabilities |
| `seed` determinism | Ignored | No sampling control |
| `temperature`, `top_p`, penalties, `logit_bias` | Ignored | No sampling control in the CLI |
| `n > 1` | Runs n turns, or rejected | Each choice costs a full turn |
| Audio input and output | Rejected | Not supported by the backend |
| Exact `system_fingerprint` | Constant per backend version | No real equivalent |

Every other case in this document is expected to pass.
