# Completions Bridge: Architecture (v2)

Status: design for review, no code yet. First backend: Claude Code CLI (`claude`).
Requirements: `CONFORMANCE.md`. Every design decision below traces back to case IDs there.

---

## 0. What changed from v1, and why

v1 treated the bridge as stateful: a "bridge session" with a ledger, a lock, a DIRTY state and a REPLAY action. `CONFORMANCE.md` requires the opposite: **the API is stateless; the request body is the only truth** (CONV-2..8, CONV-3, CONC-2, CONC-5, ERR-12).

v2 keeps the backend sessions (for speed and caching) but makes them an **immutable cache**:

| v1 | v2 |
|---|---|
| One mutable backend session per conversation | **Immutable checkpoints**; every turn runs on a fresh **fork** |
| Session matched by identity, then diffed | Request matched to the **longest cached prefix**, nothing else |
| Locks, DIRTY state, truncation on abort | Not needed: a fork is thrown away on abort or error |
| REPLAY of identical requests | Removed: identical requests get independent answers |
| Ledger of messages kept in the database | Removed: the request always carries the full history |
| NEW / IMPORT / CONTINUE / REBUILD paths | **One path**: fork nearest checkpoint, append what's missing, run |
| Side requests special-cased | Same path; only storage policy differs |

New in v2: an output policy layer (tool_choice, stop, max_tokens, JSON formats, n), name and id codecs, a sanitizer for leaks, and a per-version self-test of the claude session tricks.

---

## 0.1 Verified by probes (claude 2.1.325, Haiku; `probes/session_lab.py`)

| # | Finding | Result |
|---|---|---|
| P1 | Session file written entirely by us (user, assistant text, tool_use, tool_result) is resumed as normal history | ✅ (T1, T7) |
| P2 | A copied file with `sessionId` rewritten resumes; two forks of one checkpoint give independent answers; the original is untouched | ✅ (T4) |
| P3 | A changed system prompt and tool list take effect on resume | ✅ (T5) |
| P4 | History may use tools that aren't offered now, or no tools at all | ✅ (T6) |
| P5 | Call ids must match `^[a-zA-Z0-9_-]+$`; `call_abc` is fine, `call:abc/1` is a 400 | ✅ codec needed (T7) |
| P6 | `--autocompact 1000000` accepted; `--system-prompt-file` doesn't exist | ✅ pass prompt as argument (T0) |
| P7 | **When the file ends with a user message, claude inserts a synthetic `"No response requested."` reply before the stdin prompt.** A user question written into the file and triggered with `Continue.` was ignored 3/3 times | ❌ static trigger for user messages rejected (T2) |
| P8 | Filler and `Continue.` left in history make the model believe it gave a wrong reply and that the user typed "Continue" | ❌ must be stripped (T8) |
| P9 | Stripping filler and trigger entries (relinking `parentUuid`) leaves a clean history; the model answers meta questions about the conversation correctly | ✅ (T8) |
| P10 | Asked to list its tools, the model may print the `mcp__bridge__` prefix | ⚠️ sanitizer must rewrite it in text (T5) |

**Decision: hybrid trigger.** Everything that already happened is written into the session file. The input passed to claude is:
- the newest **user message** itself, when the request ends with one (no filler is inserted, because the file ends with an assistant message);
- the static **`Continue.`** only when the request ends with tool results (the only case where the file must end with a user entry).

Filler and trigger entries are stripped from every checkpoint when it is frozen, so the model never sees more than the current turn's pair.

### 0.2 Found during the real opencode end-to-end run (`e2e/opencode_e2e.py`)

| # | Finding | Resolution |
|---|---|---|
| E1 | claude always adds its own `# Environment` block (sandbox cwd, platform, model name, date, token budget, account email) to the system prompt. `CLAUDE_CODE_DISABLE_ATTACHMENTS`, `CLAUDE_CODE_HIDE_CWD` don't remove it; only `--bare` does, which needs API-key auth. The model then saw two environments and inspected the sandbox dir | Every system prompt gets a fixed runtime note telling the model that block is an internal sandbox to ignore and never mention (`bridge-claude/src/turn.rs`, `ENVIRONMENT_NOTE`) |
| E2 | The frontend drops the stream right after the last `Done`, which fired the cancel token while the engine was still draining the backend, so no turn was ever frozen (0 checkpoint hits) | After `Done` is emitted the engine ignores cancellation and drains (max 30 s) so the turn can be frozen |
| E3 | With claude's own filler (`"No response requested."`) plus a bare `Continue.`, answers after tool results came wrapped in status talk ("the previous task is done", "nothing is pending, what next?") | Before a ContinueAfterResults trigger the bridge writes its own `<synthetic>` assistant entry ("Let me look at the tool results."), so claude inserts no filler, and the trigger is an explicit `<system-reminder>` saying the results are above and to continue the reply without acknowledging the note. Measured 4/4 direct answers vs 3/4 before. Both are stripped at freeze. Configurable via `backends.<name>.prompts.glue_filler` / `prompts.continue_trigger` |
| E4 | Stopping the bridge right after a turn killed checkpoints still being frozen | `serve` waits up to 30 s for in-flight turns after the listener stops |
| E5 | With parallel tool calls, claude writes a placeholder for only the first call | The claude backend repairs the tool round after each stop (`ensure_tool_round`) |
| E6 | Session snapshots contain claude's `attachment` entries, including the account email | Stays local in the bridge workdir; deleted with the checkpoint (TTL/LRU) |

---

## 1. Principles

1. **The request is the truth.** The answer depends only on the request body (plus model randomness). Cached state may only make things faster, never different.
2. **Cache misses cost time, never correctness.** If anything doesn't match (opencode reformats a message, a checkpoint is gone, the backend changed), the bridge rebuilds from the request and still answers correctly.
3. **Checkpoints are immutable.** A turn reads a checkpoint, writes to its own copy, and may publish a new checkpoint. Nothing is ever modified in place, so concurrency, retries, aborts and forks need no locking.
4. **A checkpoint equals exactly what the caller saw.** If the bridge changed the output on the way out (truncated, filtered, enforced), it doesn't publish a checkpoint for it.
5. **The caller runs tools.** The backend only decides; its own tools are off.
6. **Nothing internal leaks.** Names, prefixes, filler text, paths and error details from the backend never reach the caller.

---

## 2. Big picture

```
                        CALLER'S VIEW: a normal LLM host
 ┌──────────────────────────────────────────────────────────────────────┐
 │ opencode / OpenAI SDK / curl                                         │
 │   POST /v1/chat/completions  {model, messages, tools, stream, ...}   │
 │   ◄── JSON or SSE: content / tool_calls / finish_reason / usage      │
 └──────────────────────────────────┬───────────────────────────────────┘
═════════════════════════════════════╪═════════════════════════════════════
 ┌──────────────────────────────────▼───────────────────────────────────┐
 │ BRIDGE (127.0.0.1)                                                   │
 │                                                                      │
 │  Frontend ─► Normalizer ─► Router ─► Planner ─► TurnRunner ─► Output │
 │  (OpenAI)    (IR, codecs)  (quota)   (prefix     (backend     Policy │
 │                                       match)      adapter)    ─► SSE │
 │                                         │            │               │
 │                                         ▼            ▼               │
 │                               ┌─────────────────────────────────┐    │
 │                               │ CheckpointStore (immutable)     │    │
 │                               │  prefix-hash → backend snapshot │    │
 │                               └─────────────────────────────────┘    │
 └──────────────────────────────────────────┬───────────────────────────┘
═════════════════════════════════════════════╪══════ hidden from caller ═══
                                             ▼
               ┌───────────────────────────────────────────────────┐
               │ claude -p --resume <fork-id>  (subscription login) │
               │   built-in tools OFF, system prompt = caller's    │
               │   caller's tools via parking MCP (never executes) │
               └───────────────────────────────────────────────────┘
```

---

## 3. Checkpoints and forks (the core idea)

### 3.1 What a checkpoint is

A checkpoint is a frozen backend session that represents **exactly** a prefix of some conversation, ending at a point where the next request can attach:

| Kind | Ends with | Next request must add |
|---|---|---|
| **IDLE** | An assistant final answer | A user message |
| **AWAITING** | An assistant message with tool calls, plus placeholder results | The tool results (and optionally a user message) |

Key: `prefixHash` = rolling hash over the normalised messages of that prefix (section 6.2). System prompt and tools are **not** in the key; they're passed fresh on every run (TOOL-15).

### 3.2 Conversations form a tree; checkpoints are nodes

```
 request history                               checkpoint tree (one backend)

 [u1]                                          ∅ (empty)
 [u1 a1(call#1)]                               └─ C1  [u1 a1(call#1)]            AWAITING
 [u1 a1(call#1) t1 a2]                            └─ C2  [... t1 a2]             IDLE
 [u1 a1(call#1) t1 a2 u2 a3]                         ├─ C3 [... u2 a3]           IDLE
                                                     │
 regenerate a3 (same request as C3's):               ├─ C3' [... u2 a3']         IDLE   (CONV-3)
 undo u2, ask u2' instead:                           └─ C4  [... u2' a4]         IDLE   (CONV-5, CONV-6)
```

Every request starts from the deepest node whose prefix matches, forks it, and adds one child. Regenerate, undo, edit and fork all fall out naturally, with no special cases.

### 3.3 Matching a request

```
 incoming messages:   m1   m2   m3   m4   m5   m6   (m6 = newest)
 prefix hashes:       h1   h2   h3   h4   h5   h6
 lookup, longest first: h5? h4? h3? ...  → first hit with a compatible kind
```

A hit at `h_k` is usable only if the suffix `m_{k+1..n}` can be attached to that checkpoint's kind:

| Checkpoint kind | Suffix starts with | Usable? |
|---|---|---|
| IDLE | user / system message | Yes |
| IDLE | tool message | No (invalid history; rejected earlier by validation) |
| AWAITING | tool results for exactly its pending call ids | Yes, placeholders get patched |
| AWAITING | anything else | No; try a shorter prefix |
| none found | n/a | Start from empty |

The suffix may contain anything, including assistant messages the bridge never produced (CONV-7) or a whole foreign history (CONV-2). Everything except the trigger is **written into the fork's session file**; only the final turn runs the model.

### 3.4 Lifecycle of one checkpoint

```
            materialise (copy to new id)       run turn          freeze
 checkpoint ───────────────────────────► fork ─────────► fork' ─────────► new checkpoint
 (immutable)                             (private to     (output    (only if output was
                                          this request)   complete)  returned unaltered)
                                            │
                                            └── abort / error / altered output ──► deleted
```

Retention: LRU with a size cap and a TTL (defaults 2 GB, 7 days). Deleting a checkpoint only makes some future request slower (Principle 2).

---

## 4. Components

```
┌────────────────────────────────────────────── Bridge process ──────────────────────────────────────────────┐
│                                                                                                            │
│ ┌───────────────────────────┐   ┌────────────────────────────┐   ┌──────────────────────────────────────┐  │
│ │ HttpServer                │   │ OpenAIFrontend             │   │ Normalizer                           │  │
│ │  routes /v1/*, auth,      │──►│  validate (CONFORMANCE §3, │──►│  → IR ChatRequest                    │  │
│ │  body limits, keep-alive, │   │  §4 rules, OpenAI errors)  │   │  ToolNameCodec, CallIdCodec          │  │
│ │  SSE writer with pings    │   │  IR → JSON / SSE chunks    │   │  prefix hashes                       │  │
│ └───────────────────────────┘   └────────────────────────────┘   └─────────────────┬────────────────────┘  │
│                                                                                    ▼                       │
│ ┌───────────────────────────┐   ┌────────────────────────────┐   ┌──────────────────────────────────────┐  │
│ │ QuotaManager              │◄──│ Router                     │◄──│ RequestOrchestrator                  │  │
│ │  per-backend state        │   │  model id → candidates,    │   │  plan → run → enforce → publish,     │  │
│ │  (AVAILABLE/WARNING/      │   │  stickiness by checkpoint, │   │  failover before first byte,         │  │
│ │   EXHAUSTED/PROBING)      │   │  ConcurrencyLimiter        │   │  n>1 fan-out                         │  │
│ └───────────────────────────┘   └────────────────────────────┘   └───────┬───────────────┬──────────────┘  │
│                                                                          │               │                 │
│ ┌───────────────────────────┐   ┌────────────────────────────┐           │               │                 │
│ │ CheckpointStore           │◄──│ Planner                    │◄──────────┘               │                 │
│ │  index (SQLite):          │   │  longest usable prefix,    │                           │                 │
│ │   hash → snapshot, kind,  │   │  suffix, trigger, policy   │                           ▼                 │
│ │   pending ids, backend    │   └────────────────────────────┘   ┌──────────────────────────────────────┐  │
│ │  blobs: frozen session    │                                    │ OutputPolicy                         │  │
│ │  files                    │                                    │  tool_choice, parallel_tool_calls,   │  │
│ │  LRU + TTL                │                                    │  stop, max_tokens, response_format,  │  │
│ └───────────────────────────┘                                    │  Sanitizer (no leaks), Usage         │  │
│                                                                  └──────────────────┬───────────────────┘  │
│                                                                                     │ events               │
│                                  ┌──────────────────────────────────────────────────┴──────────────────┐   │
│                                  │ «interface» Backend                                                │   │
│                                  └──────────────────────────────────────────┬─────────────────────────┘   │
│                                                                             │ implements                  │
│                                  ┌──────────────────────────────────────────▼─────────────────────────┐   │
│                                  │ ClaudeCliBackend                                                   │   │
│                                  │  Workspace        (fork: copy snapshot, new id, rewrite sessionId) │   │
│                                  │  SessionFileCodec (IR ↔ claude JSONL entries; patch placeholders)  │   │
│                                  │  ClaudeProcess    (spawn, stdin, SIGTERM, timeouts)                │   │
│                                  │  StreamDecoder    (stream-json → TurnEvents)                       │   │
│                                  │  ErrorClassifier  (limit / context / auth / transient / fatal)     │   │
│                                  │  SelfTest         (verifies the session tricks per claude version) │   │
│                                  │  parking MCP server (stdio; advertises tools, never answers)       │   │
│                                  └────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Class diagram

```
┌──────────────────────────────────────┐      ┌──────────────────────────────────────┐
│ ChatRequest (IR)                     │      │ Message (IR)                         │
│──────────────────────────────────────│ 1  * │──────────────────────────────────────│
│ model: string                        │◆────►│ role: system|user|assistant|tool     │
│ messages: Message[]                  │      │ parts: Part[]                        │
│ tools: ToolSpec[]                    │      │ norm(): bytes   // for hashing       │
│ toolChoice: none|auto|required|{name}│      └──────────────────┬───────────────────┘
│ parallelToolCalls: bool              │                         │ 1..*
│ stream, includeUsage: bool           │                         ▼
│ maxTokens?, stop[]?, n: int          │      ┌──────────────────────────────────────┐
│ responseFormat: text|json|schema     │      │ «union» Part                         │
│ reasoningEffort?                     │      │  Text{text} Image{mime,data|url}     │
│ prefixHashes: string[]  // derived   │      │  ToolCall{id,name,argsJson}          │
└──────────────────────────────────────┘      │  ToolResult{callId,text}             │
                                              │  Reasoning{text}                     │
┌──────────────────────────────────────┐      └──────────────────────────────────────┘
│ TurnPlan                             │
│──────────────────────────────────────│      ┌──────────────────────────────────────┐
│ backend: string, model: string       │      │ Checkpoint                           │
│ base?: Checkpoint                    │─────►│──────────────────────────────────────│
│ append: Message[]  // written to file│      │ prefixHash: string  (PK with backend)│
│ trigger: Trigger                     │      │ backend, backendVersion              │
│ system: string, tools: ToolSpec[]    │      │ kind: IDLE | AWAITING                │
│ policy: OutputPolicySpec             │      │ pendingCallIds: string[]             │
└──────────────────────────────────────┘      │ snapshotRef: path                    │
                                              │ sizeBytes, createdAt, lastUsedAt     │
┌──────────────────────────────────────┐      └──────────────────────────────────────┘
│ «union» Trigger                      │
│  UserInput{message}                  │      ┌──────────────────────────────────────┐
│  ContinueAfterResults{}              │      │ «union» TurnEvent                    │
└──────────────────────────────────────┘      │  TextDelta  ReasoningDelta           │
                                              │  ToolCalls{calls[]}  Usage  Limit    │
┌──────────────────────────────────────┐      │  Done{finish}  Error{class,retryAt}  │
│ OutputPolicySpec                     │      └──────────────────────────────────────┘
│──────────────────────────────────────│
│ toolChoice, parallelToolCalls        │      ┌──────────────────────────────────────┐
│ maxTokens?, stop[]                   │      │ ErrorClass                           │
│ responseFormat                       │      │  LIMIT(resetAt)  CONTEXT_TOO_LONG    │
│ needsBuffering(): bool               │      │  TRANSIENT  BACKEND_AUTH  TIMEOUT    │
└──────────────────────────────────────┘      │  INVALID_HISTORY  INTERNAL           │
                                              └──────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────────────┐
│ «interface» Backend                                                                   │
│───────────────────────────────────────────────────────────────────────────────────────│
│ name; capabilities { forkableSessions, images, jsonSchema, maxContext, effort }       │
│ mode: NATIVE | TRANSCRIPT            // set by selfTest()                             │
│───────────────────────────────────────────────────────────────────────────────────────│
│ selfTest(): Mode                     // run at startup and when the CLI version moves │
│ materialize(base?, append[], trigger): Workspace   // fork + write suffix             │
│ run(ws, plan): Turn                  // spawn, stream, stop at tool calls             │
│ freeze(ws, returnedAssistant): Checkpoint?          // publish if unaltered           │
│ discard(ws)                                                                           │
│───────────────────────────────────────────────────────────────────────────────────────│
│ Turn = { events: AsyncStream<TurnEvent>, abort(), done: Promise<Outcome> }            │
└───────────────────────────────────────────────────────────────────────────────────────┘
         ▲ implements                                 ▲ implements (later)
┌────────┴─────────────────────┐   ┌──────────────────┴─────────────────────────────────┐
│ ClaudeCliBackend             │   │ HttpChatBackend: stateless pass-through to a free  │
│  NATIVE: session-file forks  │   │ OpenAI-compatible API; materialize = keep messages;│
│  TRANSCRIPT: full history as │   │ freeze = no-op. OtherCliBackend: own tricks,       │
│  one user message (fallback) │   │ same interface                                     │
└──────────────────────────────┘   └────────────────────────────────────────────────────┘
```

---

## 6. Normalisation, hashing and codecs

### 6.1 Validation (before anything else)

Rules copied from OpenAI's behaviour, returning the same 400s:
- `tool` message without a preceding assistant `tool_calls` with that id (CONV-11).
- Assistant `tool_calls` not fully answered before the next non-tool message (CONV-12, TOOL-19).
- Unknown model (REQ-1, 404), bad types and ranges (REQ-9, REQ-15, REQ-22).
- Unknown top-level fields are ignored, never rejected.

### 6.2 Message normalisation for hashing

`norm(m)` = role + text parts (trailing whitespace trimmed, `\r\n` → `\n`) + tool calls (id, name, arguments re-serialised as canonical JSON) + tool results (call id, text) + image digests. `system` and `developer` are equal. `name`, `refusal` and unknown fields are excluded.

`prefixHash_k = SHA256(prefixHash_{k-1} ‖ norm(m_k))`, with system/developer messages included in position.

If opencode re-sends messages slightly differently (open question Q2), the only effect is a cache miss.

### 6.3 Tool name codec (TOOL-5, LEAK-4)

claude tool names must match `^[a-zA-Z0-9_-]{1,64}$`, and MCP adds the `mcp__bridge__` prefix (13 characters). The codec maps each caller name to a safe short name per request (`read` → `read`; `todo.write` → `todo_write_3f2a`; long names are hashed), and maps calls back to the exact caller name. The mapping is deterministic, so a name maps the same way across requests and checkpoints stay valid.

### 6.4 Call id codec (TOOL-20)

Ids produced by claude (`toolu_…`) go to the caller unchanged. Ids from the caller's history that aren't valid Anthropic ids are mapped deterministically (`call:abc` → `toolu_x_<hash>`) when written to a session file, and mapped back on output.

---

## 7. The single request path

```
 ┌───────────────────────────────────────────────────────────────────────────────────┐
 │ 1  HttpServer: auth, size limit; send headers + ": ping" early when streaming     │
 │ 2  Frontend: validate → 400/401/404 in OpenAI format                              │
 │ 3  Normalizer: IR, codecs, prefix hashes                                          │
 │ 4  Router: candidate backends for the model id, in order:                         │
 │      a. backend owning the deepest usable checkpoint (if AVAILABLE/WARNING)       │
 │      b. other AVAILABLE backends by priority, then WARNING ones                   │
 │      none → 429 insufficient_quota, Retry-After = earliest reset                  │
 │ 5  ConcurrencyLimiter: slot on that backend (bounded wait) → else 429             │
 │ 6  Planner: base checkpoint, append[], trigger, policy                            │
 │ 7  Backend.materialize → private workspace                                        │
 │ 8  Backend.run → TurnEvents                                                       │
 │ 9  OutputPolicy: enforce, sanitise, map to SSE / JSON                             │
 │      error before first content byte and retryable → back to step 4 with the      │
 │      next backend (transparent failover)                                          │
 │      error after first byte → SSE error event (STR-13)                            │
 │ 10 Backend.freeze if the output went out complete and unaltered → new checkpoint  │
 │    else discard workspace                                                         │
 │ 11 request_log entry                                                              │
 └───────────────────────────────────────────────────────────────────────────────────┘
```

### 7.1 Planner: building append[] and trigger

```
 suffix after base:   [ ... messages ... ][ final part ]
                        └──── append ────┘ └ trigger ─┘

 final part = last user message           → NOT written to file; Trigger UserInput(that message)
            = tool results (base AWAITING
              and ids == pending)          → patch placeholders; Trigger ContinueAfterResults
            = tool results (otherwise)     → write as normal results; Trigger ContinueAfterResults
            = tool results + user message  → results written; Trigger UserInput(user message)
            = assistant message (prefill)  → written to file; Trigger ContinueAfterResults (CONV-9)
```

Never write the final user message into the file and trigger with `Continue.` (P7: the model ignores the question). The file must end with an assistant entry whenever the trigger is a user message.

### 7.2 Output policy

| Request field | Enforcement | Buffering needed |
|---|---|---|
| `tool_choice: none` (TOOL-12) | Tools still listed (history may reference them), instruction appended to system prompt; if the model calls one anyway, rerun from the same base once, then return text-only | Yes (until first token decides) |
| `tool_choice: required` / named (TOOL-13/14) | Instruction appended; validate; rerun once on miss | Yes |
| `parallel_tool_calls: false` (REQ-7) | Instruction appended; if several calls arrive, return the first only and **don't publish a checkpoint** | No |
| `max_tokens` (REQ-8) | Count output tokens; stop the process at the limit, `finish_reason: "length"`, no checkpoint | No |
| `stop` (REQ-10) | Scan the stream with a hold-back buffer of the longest stop string; cut, `finish_reason: "stop"`, no checkpoint | Small hold-back |
| `response_format: json_object` / `json_schema` (REQ-12/13) | `claude --json-schema` when given a schema, otherwise instruction; validate; rerun once | Yes |
| `n > 1` (REQ-11) | n workspaces from the same base, run in parallel, choices indexed 0..n-1 | No |
| `reasoning_effort` (REQ-19) | `--effort` | No |
| Sampling fields (REQ-9, 15, 16, 17) | Accepted, ignored; `logprobs: null` | No |

"Buffering needed" means content isn't streamed until the decision is safe; keep-alive pings continue meanwhile (STR-10).

### 7.3 Sanitizer (section 10 of CONFORMANCE)

- Tool names and ids mapped back through the codecs; `mcp__bridge__` never appears, including inside streamed text, where the prefix is rewritten to the caller's tool name with a short hold-back buffer (P10).
- Error messages rewritten from a fixed table; raw backend text is only logged.
- `model` in responses is always the requested id.
- Usage: `prompt_tokens` = backend input + cache read + cache write; `cached_tokens` = cache read; `completion_tokens` = output; reasoning tokens if reported (RESP-4, LEAK-6).

---

## 8. Claude backend internals

### 8.1 Per-run command

```
claude -p --resume <fork-id>
  --input-format stream-json --output-format stream-json --verbose --include-partial-messages
  --model <model> [--effort <level>] [--json-schema <schema>]
  --tools "" --mcp-config <ws>/mcp.json --strict-mcp-config
  --allowedTools mcp__bridge__<each encoded tool>
  --system-prompt <caller system + policy instructions>       (no -file variant, P6)
  --setting-sources "" --disable-slash-commands
  --autocompact 1000000      # never compacts on its own; context errors surface (ERR-2)
cwd = <bridge>/work/empty    (nothing to read)
```

A new conversation without a base uses `--session-id <fork-id>` instead of `--resume`.

### 8.2 materialize

```
 base checkpoint? ──no──► empty file, new uuid
      │yes
      ▼
 copy snapshot → ~/.claude/projects/<work-slug>/<fork-id>.jsonl
 rewrite "sessionId" on every entry to <fork-id>                      (Q11)
      │
      ▼
 base AWAITING and suffix starts with its results?
      │yes → patch each placeholder tool_result: content = real text,
      │       drop is_error / toolDenialKind                          (verified F6)
      ▼
 append[] → write entries with SessionFileCodec:                     (Q1)
      user text / images, assistant text, assistant tool_use,
      tool_result; parentUuid chain continued from the last entry
      │
      ▼
 workspace ready; trigger kept for stdin
```

### 8.3 run

```
 spawn ─► write trigger to stdin
             UserInput → the user message (text and images); file ends with an assistant entry, no filler
             ContinueAfterResults → "Continue."   (claude adds its synthetic filler first; both stripped at freeze)
   │
   ▼ decode stream
   text_delta → TextDelta        thinking_delta → ReasoningDelta
   tool_use mcp__bridge__X → collect (decode name and id)
   message_stop with calls → SIGTERM → ToolCalls, Done(tool_calls)
   result → Usage, Done(stop | length | content_filter)
   rate_limit_event → Limit (QuotaManager)
   exit code / error text → ErrorClassifier
   │
   ▼ after exit, quick check
   tool calls: one placeholder per call id present
   no new "interrupted" entries for ids we patched
   failure → mark self-test stale, discard workspace, no checkpoint
```

### 8.4 freeze

Called only when the caller received the complete, unaltered output. First the file is cleaned: claude's synthetic entries (`model: "<synthetic>"`, e.g. `"No response requested."`) and our `Continue.` trigger entries are removed, and `parentUuid` links are rewired past them (P8, P9). Then the workspace file is moved into the checkpoint blob directory and indexed under `prefixHash(request messages + returned assistant message)`, with kind AWAITING (and the pending call ids) or IDLE. A workspace is never reused after freezing; the next request forks it again.

### 8.5 Self-test and modes

At startup, and whenever `claude --version` changes, the backend runs a tiny canary on the cheapest model (about 3 short turns):
1. New session, tool call, stop, placeholder present.
2. Fork by copy and patch, resume, the model uses the patched result and no "interrupted" entry appears.
3. Session file built from scratch (append path), resume, answer consistent.

All pass → **NATIVE** mode. Any fail → **TRANSCRIPT** mode: every request starts a new session whose first user message is the full history rendered as a labelled transcript, with tool calls still intercepted the same way. Slower and no cache reuse, but correct. The mode is shown in `/admin/status`.

### 8.6 Error classification

| claude signal | Class | HTTP to caller (if nothing streamed yet) |
|---|---|---|
| Usage limit / `rate_limit_event` exhausted | LIMIT(resetAt) | Failover; else 429 `insufficient_quota` + Retry-After |
| "prompt is too long" | CONTEXT_TOO_LONG | 400 `context_length_exceeded` (no failover unless a bigger-context backend exists) |
| Overloaded / 5xx from upstream | TRANSIENT | Failover or retry once; else 503 |
| Not logged in / auth error | BACKEND_AUTH | Backend disabled; failover; else 503 |
| No first token within timeout | TIMEOUT | Failover; else 504 |
| Anything else | INTERNAL | 500 `server_error` |

---

## 9. Data model

```
checkpoints                                   quota
──────────────────────────────────────        ───────────────────────────────
backend          PK part                      backend      PK
prefix_hash      PK part                      status       AVAILABLE|WARNING|EXHAUSTED|PROBING
backend_version                               utilization
kind             IDLE | AWAITING              resets_at
pending_call_ids JSON                         updated_at
snapshot_path
size_bytes                                    backend_state
created_at, last_used_at                      ───────────────────────────────
                                              backend      PK
request_log                                   mode         NATIVE|TRANSCRIPT
──────────────────────────────────────        cli_version
id, created_at, model, backend                self_test_at
checkpoint_hit_depth, appended_messages       self_test_result
finish_reason, error_class
tokens_in, tokens_out, cache_read
ttfb_ms, total_ms
```

No sessions, no ledger, no locks: everything the bridge needs to answer comes from the request.

---

## 10. State machines

### 10.1 Backend quota

```
 AVAILABLE ──warning event──► WARNING ──utilization drops──► AVAILABLE
     │                           │
     └──── limit error / 429 / exhausted event ────┐
                                                   ▼
                                  EXHAUSTED (until resetsAt or backoff)
                                                   │ time passes
                                                   ▼
                                  PROBING ──success──► AVAILABLE
                                     └──fail──► EXHAUSTED (longer backoff)
```

### 10.2 Workspace (per request; replaces v1's session state machine)

```
 MATERIALIZED ──spawn──► RUNNING ──complete, unaltered──► FROZEN (= new checkpoint)
                            │
                            ├──complete, altered (stop/max_tokens/filtered)──► DISCARDED
                            ├──error / timeout──────────────────────────────► DISCARDED
                            └──client abort─────────────────────────────────► DISCARDED
```

---

## 11. Scenario flows

Notation: `oc` caller, `BR` bridge (frontend, planner, policy), `CS` CheckpointStore, `CL` claude backend, `cli` claude process.

### S1. New conversation, text answer (CONV-1, STR-*)

```
oc ── [sys,u1] ──► BR: no checkpoint → base ∅, trigger UserInput(u1)
BR ──► CL.materialize(∅) ─► run: --session-id F1, stdin u1
cli ── text deltas ──► BR ── SSE content ──► oc
cli ── result ──► BR ── finish stop, usage, [DONE] ──► oc
BR ──► CL.freeze(F1) ──► CS: key h([sys,u1,a1]) IDLE
```

### S2. Tool call (TOOL-2, TOOL-4)

```
oc ── [sys,u1] ──► BR ──► CL: run F1
cli ── "I'll read it." ──► SSE content
cli ── tool_use read, message_stop ──► CL: SIGTERM; placeholder written
BR ── tool_calls[0] {id: toolu_…, name: "read"}, finish tool_calls ──► oc
BR ──► CS: key h([sys,u1,a1(call)]) AWAITING pending=[toolu_…]
```

### S3. Tool results, continue (TOOL-18)

```
oc ── [sys,u1,a1(call),t1] ──► BR: deepest hit = AWAITING checkpoint, suffix = [t1] = pending
CL.materialize: copy → F2, patch placeholder with t1 ─► run --resume F2, stdin "Continue."
cli ── answer / next calls ──► oc
CS: new checkpoint for [..., t1, a2]; the AWAITING one stays (another request may still use it)
```

### S4. Parallel tool calls (TOOL-3, STR-7)

As S2; stop only at `message_stop`, so all calls are collected; checkpoint pending = all ids. The next request must answer all of them (validation guarantees it, CONV-12).

### S5. Follow-up user message (CONV-1)

Deepest hit is IDLE; suffix [u2]; fork, run with stdin u2. No patching, no filler.

### S6. Regenerate: the same request twice (CONV-3, CONC-2)

```
req A ── [..., u2] ──► hit C2 ─► fork F5 ─► answer a3  ─► freeze C3
req B ── [..., u2] ──► hit C2 ─► fork F6 ─► answer a3' ─► freeze C3'
```

Both run from the same immutable C2, concurrently or not. Two checkpoints differ by the assistant message, so both stay usable.

### S7. Undo, edit, fork (CONV-4, CONV-5, CONV-6)

The request's history diverges from a cached path at some message. The deepest matching prefix is above the divergence; everything after it goes into `append[]`. Correct by construction.

### S8. History the bridge never saw (CONV-2, CONV-7, CONV-16)

No hit (or a short one). The whole history is written into a fresh file by the codec (append path) and the trigger runs. One slower turn; the result is frozen, so the next turn is warm.

### S9. Caller compaction (CONV-8)

The history now starts with a summary, so no prefix matches; same as S8, with a much shorter history.

### S10. Side request: title, summary (CONC-3)

Same path. No tools and history of one or two messages, so the policy marks it ephemeral: run with `--no-session-persistence`, no checkpoint. Router may prefer the cheapest model configured for side requests.

### S11. Client abort (STR-14, CONC-5)

```
oc ── disconnect ──► BR: turn.abort() ──► CL: SIGTERM, discard workspace
nothing else changes; the retry is just another request (hits the same base as before)
```

### S12. Quota exhausted

```
before first byte:  CL error LIMIT(resetAt) ─► QuotaManager EXHAUSTED
                    Router: next backend? yes → plan on it (no checkpoint → S8 path), stream
                                           no  → 429 insufficient_quota, Retry-After
after first byte:   SSE error event, stream closed (STR-13); the caller's retry fails over
```

With only claude registered, the caller sees a 429 with the reset time.

### S13. Context too long (ERR-2, CONV-14)

claude refuses with "prompt is too long" (autocompact is disabled, so it never summarises silently). The bridge returns 400 `context_length_exceeded`, opencode compacts, and the next request goes through S9. `/v1/models` doesn't carry context limits, so the opencode provider config declares the limit to make opencode compact before that happens.

### S14. Bridge restart (CONV-16)

The checkpoint index and blobs are on disk, so the next request hits warm checkpoints. In-flight workspaces are deleted at startup.

### S15. Tools or system prompt change mid-conversation (TOOL-15)

Not part of the key; each run passes the current system prompt and tool list. If a history refers to a tool no longer offered, the codec still writes the old call (the name stays the same) and the model sees only the current list (Q4, Q12).

### S16. Self-test fails after a claude update

```
startup / version change ─► SelfTest step 2 fails ─► mode TRANSCRIPT
all requests: new session with history rendered as transcript; tool calls still intercepted
/admin/status shows the reason; checkpoints of the old version are ignored (backend_version)
```

### S17. n = 2 (REQ-11)

Plan once; materialize two forks from the same base; run concurrently; stream choices with index 0 and 1; freeze both.

### S18. `stop` sequence hit (REQ-10)

The OutputPolicy holds back up to the longest stop string, cuts at the match, stops the process, returns `finish_reason: "stop"`. The output was altered, so no checkpoint; the next request writes the shortened assistant message via the append path.

---

## 12. Conformance traceability

| CONFORMANCE group | Mechanism |
|---|---|
| EP, AUTH | HttpServer, Frontend |
| REQ (parameters) | Frontend validation, OutputPolicy (7.2), capability flags |
| MSG (roles, shapes) | Normalizer, SessionFileCodec (images, multiple system messages written as user-visible instruction entries) |
| CONV (stateless) | Immutable checkpoints, longest-prefix match, append path (sections 3, 7.1) |
| TOOL | Parking MCP, ToolNameCodec, CallIdCodec, OutputPolicy |
| RESP, STR | Frontend IR → JSON / SSE mapping, keep-alive pings, error events |
| ERR | ErrorClassifier → fixed error table, failover before first byte |
| CONC | No shared mutable state; ConcurrencyLimiter per backend |
| LEAK | Sanitizer, codecs, fixed error texts; filler and "Continue." stay inside the backend (residual risk, Q13) |
| NF | Warm checkpoints, prompt cache, early headers |

---

## 13. Configuration

```yaml
server:
  listen: 127.0.0.1:4141
  token: <random>
  workdir: ~/.completions-bridge
  max_body_mb: 20
  first_token_timeout_s: 60
  keepalive_ping_s: 10

models:                      # ids listed in /v1/models
  bridge/auto:          { route: auto }
  bridge/claude-haiku:  { backend: claude, model: haiku }
  bridge/claude-sonnet: { backend: claude, model: sonnet }

backends:
  claude:
    type: claude-cli
    priority: 1
    max_concurrency: 3
    side_request_model: haiku
    effort_map: { minimal: low, low: low, medium: medium, high: high }

checkpoints:
  max_total_gb: 2
  ttl_days: 7

output:
  reasoning_to_caller: true
  max_n: 4
  enforcement_retries: 1
```

---

## 14. Open questions (must be answered before or during build)

Answered by probes (section 0.1): **Q1 yes, Q4 yes, Q5 codec needed, Q6 no file flag, Q11 yes, Q12 yes, Q13 solved by hybrid trigger plus stripping.** Remaining: Q2, Q7, Q8, Q14, Q15 (flag accepted, effect unproven), Q16.

| # | Question | Why it matters | Fallback |
|---|---|---|---|
| Q1 | Can the codec write entries (user, assistant text, tool_use, tool_result, images) that claude resumes as normal history? | Append path: CONV-2, 4–8 | TRANSCRIPT mode for appended parts |
| Q2 | Does opencode resend earlier messages identically? Any session header? | Cache hit rate only | Looser normalisation |
| Q4 | Do changed system prompt and tools apply on `--resume`? | TOOL-15 | Rebuild without base when they change |
| Q6 | Does `--system-prompt-file` exist; any size limit on `--system-prompt`? | Large opencode prompts | Pass as argument |
| Q7 | Exact signals for quota exhaustion | ERR-6 | Treat unknown errors as TRANSIENT with backoff |
| Q8 | Can stopping claude lose the tool_use entry itself? | Checkpoint correctness | Wait for the entry before SIGTERM |
| Q11 | Does a copied file with rewritten `sessionId` resume correctly? (`--fork-session` exists but forks before our patch) | Every fork | Use claude's own `--fork-session`, if the new file can be patched before the model call (ordering to verify); else TRANSCRIPT mode |
| Q12 | Does the API accept history tool_use for tools not in the current list? | TOOL-15, `tool_choice: none` | Keep old tool definitions listed but instruct not to use |
| Q13 | Does the filler ("No response requested." + "Continue.") ever surface in answers? | LEAK-3, LEAK-5 | Better trigger wording; strip internal entries when copying |
| Q14 | Can non-conversation entries (`api-request-blob` etc.) be dropped from snapshots to save disk? | Checkpoint size | Keep them |
| Q15 | Does `--autocompact 1000000` stop auto-compaction for our model's window? | ERR-2 honesty, stateless rule | Check for `compact_boundary` events and fail the turn |
| Q16 | Does `--json-schema` work together with tools and our stop-at-tool-call flow? | REQ-13 | Instruction + validation only |

---

## 15. Build order

1. Logging proxy to capture real opencode traffic (Q2, plus confirm CONFORMANCE assumptions about opencode).
2. Probes for Q1, Q4, Q8, Q11, Q12, Q13, Q15; these define NATIVE vs TRANSCRIPT.
3. Frontend, Normalizer, codecs, validation; conformance tests for RESP/STR/ERR running against a fake backend.
4. ClaudeCliBackend in TRANSCRIPT mode first (simplest, always correct), passing CONV and TOOL cases.
5. NATIVE mode: checkpoints, forks, append path, self-test.
6. OutputPolicy features (tool_choice, stop, max_tokens, JSON, n).
7. Router, QuotaManager, failover with a second backend.
8. Full CONFORMANCE run with curl, both SDKs and opencode; then E2E tasks.

---

# Part II — v3 structural fixes (implemented; see 23 for results)

Three gaps measured in the 24-minute Sonnet run (`e2e/bigtask.py --long`, 72 requests, 5.0M input tokens):

| Gap | Evidence | Cost |
|---|---|---|
| G1 Prompt cache collapses in long tool loops | requests #58–60: history 146–153k tokens, cache read fell from 119k to 8.4k (system part only) | 76% cache overall instead of ~95%; ~1.1M extra uncached tokens in that run |
| G2 Next request races the checkpoint save | request #62 missed #61's checkpoint (depth 0 instead of 5); conformance agent saw the same in tool loops | a cold, rewritten tail; the model loses its own original turn bytes (thinking, exact formatting) |
| G3 Tool calls are not streamed | time to first token p90 37 s, max 88 s, all big `write`/`edit` calls | client shows nothing for up to 1.5 min; worse: a write longer than `first_token_timeout_s` (120 s) would be **killed as a timeout** |

Recommended implementation order: **G2 → G1 → G3**. G2 is a prerequisite for G1's full gain (see 16.2), and G3 is independent but touches the most crates.

## 16. Root causes in the current architecture

### 16.1 Why G1 happens

How the prompt cache works (Anthropic, as documented): cache entries are keyed by the exact request prefix up to a cache breakpoint. claude marks breakpoints on the system/tools block and on the last message. A new request reuses an entry only if that exact prefix reappears; the lookup walks back a limited number of content blocks (about 20) from each breakpoint.

What our design does on a tool round (sections 0.2 E3, 8.4):

```
request N   sent to API:  [... , toolResult(N-1), glueA(N), glueU(N)]          ← cached up to glueU(N)
freeze N:                  strip glueA(N), glueU(N)                              ← E3
request N+1 sent to API:  [... , toolResult(N-1), assistant(N), toolResult(N), glueA(N+1), glueU(N+1)]
                                               ▲
                           diverges from request N right here, so request N's entry is unusable
```

(glueA = our `<synthetic>` "Let me look at the tool results."; glueU = our `<system-reminder>` trigger.)

So request N+1 can only reuse an *older* entry whose prefix still matches: the last turn that was triggered by a real user message (those leave nothing stripped). Every tool round adds about 3 content blocks, so after roughly 6–7 consecutive tool rounds that older entry falls out of the look-back window, and only the system/tools entry (8.4k) still hits. That is exactly #58–60: long agentic stretches with no user message in between. Short tool loops hide the problem, which is why the earlier runs showed 79–90%.

Root cause: **E3 traded cache-stability for a clean model-visible history.** Stripping was introduced because claude's own glue ("No response requested." + "Continue.") confused the model (P8). Our current glue no longer has that wording problem (E3 measured 4/4 direct answers), so the reason for stripping is mostly gone, but the stripping stayed.

### 16.2 Why G2 happens

Timeline of one tool round today (engine `run_attempt`, claude `turn.rs`):

```
t0      claude emits tool_use … message_stop
t0      backend → ToolCalls, Done        engine → policy → sink → frontend → client gets finish_reason tool_calls, [DONE]
t0+ε    client runs the tool locally (read/grep: milliseconds)
t0+ε    client sends request N+1  ───────────────►  engine.prepare(): store.lookup() sees only INDEXED checkpoints
t0      meanwhile: backend SIGTERMs claude, claude writes placeholders, exits (0.5–1 s, up to 10 s grace)
        → ensure_tool_round fix-ups → strip → write snapshot → engine store.insert_checkpoint()   ← indexed at t0+~1 s
```

The caller learns the answer is complete (t0) before the checkpoint exists (t0+~1 s). Fast local tools win that race. The planner then picks the previous checkpoint and **re-writes the missing assistant message and tool results through the codec** (append path). That is correct (Principle 2), but:

- the tail is written by us, not by claude: the assistant entry loses its thinking block and signature and differs byte-wise from what claude produced, so
- the cache misses from that point (this also defeats G1's fix), and
- the model sees a slightly different version of its own previous turn than it would on a warm continuation.

Root cause: **checkpoint publication is asynchronous with respect to the response the caller already has**, and the planner has no notion of "a deeper checkpoint is about to exist".

### 16.3 Why G3 happens

- IR (`bridge-types`): only `TurnEvent::ToolCalls { calls }`, defined as "all tool calls of the assistant message, emitted once, complete".
- claude decoder (`stream.rs`): ignores `content_block_start` of type `tool_use` and `input_json_delta`; collects calls from the completed `assistant` event and emits them at `message_stop`.
- Policy: treats calls atomically (needed for tool_choice / parallel enforcement).
- Frontend: renders a call as one chunk with full arguments.

A 9k-token `write` call therefore produces zero bytes for the whole generation time. The engine's first-token timeout (`first_token_timeout_s`, default 120 s) runs until the first backend event, so a model that starts its reply directly with a very large tool call (≥ ~120 s of generation) would be aborted as a Timeout even though it is working normally. This is a latent correctness bug, not just latency.

Root cause: **the IR made "tool call" an atomic event**, so no layer below the frontend could stream it.

## 17. Fix designs

### 17.1 G2 — Pending checkpoints (engine)

**Idea:** publish a *promise* of the checkpoint at the moment the caller is told the turn is complete, and let the planner wait briefly for a deeper promised checkpoint instead of planning around it.

New engine component `PendingRegistry` (in memory, per process):

```
PendingRegistry
  register(backend, key, kind, pending_call_ids) -> PendingGuard      // before Done reaches the caller
  resolve(guard, Ok(CheckpointRecord) | Failed)                       // after freeze+index, or on discard
  find_deepest(backend, prefix_hashes) -> Option<(depth, kind, ids, watch::Receiver)>
PendingGuard: Drop ⇒ resolve(Failed)   (panics, cancellation, early returns can never leave a waiter hanging)
```

Changes in `run_attempt` (orchestrator):

```
on PolicyOutput::Emit(Done{finish}):
    if mode == Native && !ephemeral && backend_done && !policy.altered():
        key  = next_hash(request prefix, policy.assistant_message())
        kind = Awaiting (ids = caller-visible call ids) if finish == ToolCalls else Idle
        guard = pending.register(backend, key, kind, ids)          ← BEFORE sink.emit(Done)
    sink.emit(Done)
... later, existing code:
    freeze ok  → store.insert_checkpoint(...) → guard.resolve(Ok(record))
    freeze err / discard / altered later → guard dropped → Failed
```

Changes in `prepare` (planner input):

```
stored  = store.lookup(backend, version, hashes)                 (unchanged)
best    = choose_base(messages, stored)                          (unchanged)
pending = registry.find_deepest(backend, hashes) with depth > best.depth and kind compatible with the suffix
if pending:
    wait for it, bounded by min(config.engine.pending_wait_ms (default 5000), request cancellation)
    Ok(record)  → add to `stored`, re-run choose_base           → warm, exact continuation
    Failed/timeout → keep `best`                                 → today's behaviour
CandidateInfo.depth includes the pending depth, so routing prefers the backend that is about to own it.
```

Rules and invariants:
- The wait happens **before** acquiring a concurrency permit, so a waiter can never starve the turn it is waiting for.
- Waiting only ever replaces "use an older base" with "use the exact base"; on any failure the old path runs. Worst case added latency = `pending_wait_ms`, and only when a deeper checkpoint is genuinely in flight.
- Duplicate keys (two identical regenerations finishing together) are allowed; `find_deepest` returns any live one.
- The registry is not persisted. On graceful shutdown in-flight freezes complete first (E4); after a crash, waiters don't exist either.
- Kind compatibility is checked before waiting (an Idle promise is useless for a suffix that starts with tool results), so no pointless waits.

### 17.2 G1 — Persistent glue (claude backend)

**Idea:** make every backend request an exact extension of the previous one by keeping our glue in the session instead of stripping it at freeze.

Changes:
1. `append_bridge_filler` and the ContinueAfterResults trigger mark their entries with a private field `"bridgeGlue": true` (extra JSONL keys are ignored by claude; verified behaviour required, see V1).
2. `freeze_inner`: `strip_fillers` keeps entries with `bridgeGlue`, still removes claude's own `<synthetic>` fillers ("No response requested.", which should no longer occur) and any trigger entry without the marker (legacy snapshots).
3. `ENVIRONMENT_NOTE` gains one sentence: automatic `<system-reminder>` notes and the short "Let me look at the tool results." lines are runtime glue, not user or assistant messages; never count, quote or mention them. This protects meta questions ("how many messages have I sent?", CONV/LEAK-5).
4. The append path (history written from scratch: import, rebuild, foreign history) does **not** insert glue between tool results and the next assistant message. That sequence is the natural API shape; such chains are cold anyway; mixing glued and unglued stretches is harmless.
5. Option `backends.claude.tuning.keep_glue` (default `true`). `false` restores today's stripping if the quality checks (V2) fail.

Resulting invariant (with G2 in place):

```
request N+1 prompt = request N prompt
                     + assistant(N) exactly as claude wrote it (incl. thinking)
                     + toolResult(N) (placeholder patched in place)
                     + glueA(N+1) + glueU(N+1)
⇒ the whole of request N is a cached prefix of request N+1
```

Principle 4 is refined, not broken: *the conversation a checkpoint represents equals what the caller saw; the backend session may also contain backend-private glue that is deterministic, never returned to the caller, and not part of the key.* Keys (`prefixHash`) are unchanged, so no index migration; old stripped snapshots keep working (they are merely cache-colder).

Cost: about 60 extra context tokens per tool round (100 rounds ≈ 6k tokens), reported in usage like everything else, so opencode's compaction trigger accounts for it.

### 17.3 G3 — Streaming tool calls (types, claude, policy, engine, frontend)

**IR change (`bridge-types`, coordinator-owned):**

```rust
TurnEvent::ToolCallStart { index: u32, id: String, name: String }   // caller-visible id and name
TurnEvent::ToolCallDelta { index: u32, arguments: String }           // raw JSON text fragment
TurnEvent::ToolCalls { calls }                                       // unchanged: the authoritative commit
```

Contract added to the ordering rules: Start/Delta are optional previews. If a call was started, the `ToolCalls` commit contains that call with the same id and name, and its `arguments` is **byte-identical** to the concatenation of its deltas. `index` is the call's position among this message's tool calls (0..k), not claude's content-block index.

**claude backend:**
- `content_block_start` with `tool_use` → map block index → next call ordinal; decode name (strip `mcp__bridge__`, codec reverse) and id (codec reverse); emit Start. Skip the internal `StructuredOutput` tool (json_schema) and any non-bridge tool.
- `input_json_delta` → Delta with `partial_json` verbatim. Never sanitised (arguments are caller data).
- Commit: the arguments string is the concatenation of the deltas, not a re-serialisation of the parsed `input` (re-serialising could change spacing or key order and break byte identity). If a started call received no deltas, emit one Delta with the serialised input (normally `{}`) before the commit.
- `content_block_start(tool_use)` arrives within ~1–2 s of the model deciding to call a tool, so it also satisfies the first-event timeout.

**Policy** (per mode):

| Mode | Start/Delta |
|---|---|
| pass-through (auto, parallel true, no JSON format) | forwarded immediately; counts as emitted content (no Retry afterwards) |
| `parallel_tool_calls: false` | first call forwarded; later calls' Start/Delta swallowed; commit keeps the first call; `altered = true` if anything was dropped |
| buffered (`tool_choice` none/required/function, JSON formats) | swallowed; decision at commit as today; the frontend renders committed calls in full |
| `max_tokens` / `stop` | unchanged (they apply to content text only) |

**Engine:** Start/Delta pass through `Sink` like text: the first one locks out failover and satisfies the first-token timeout. No change to freeze or keys: the checkpoint key uses `policy.assistant_message()`, built from the commit, and the hash canonicalises arguments.

**Frontend (streaming):**
- Start → `delta.tool_calls[{index, id, type:"function", function:{name, arguments:""}}]`
- Delta → `delta.tool_calls[{index, function:{arguments:"…"}}]`
- Commit → nothing for calls already started (tracked by id); full single chunk for calls never started (buffered modes), using the next free index.
- Non-streaming: Start/Delta ignored; the commit builds `message.tool_calls` as today.

**Fake backend:** option `stream_tool_args` emits Start + several Deltas for scripted calls, so every layer can be tested without quota.

## 18. Impact and regression analysis

| Area | G2 pending | G1 glue | G3 streaming |
|---|---|---|---|
| Stateless rule (CONV-*) | Waits only for a checkpoint that represents a prefix of *this* request; can't change answers | Glue is not in the key and never reaches the caller; meta questions protected by the note (risk, V2) | No effect |
| Checkpoint immutability | Unchanged (promise resolves to an immutable record) | Snapshots now contain glue; still immutable | Unchanged |
| Failover | Pending depth influences routing; waiting happens before failover decisions | n/a | First Start/Delta locks out failover, same as first text |
| Policy enforcement (REQ-6/7, TOOL-12–14) | n/a | n/a | Buffered modes never stream calls, so Retry still works; parallel=false can never leak a second call |
| Errors after output (STR-13) | n/a | n/a | A stream can now end with a partial tool call plus an error event, which is valid OpenAI behaviour |
| Client abort (STR-14) | Guard drop resolves the promise as Failed | n/a | Abort mid-arguments: claude killed, no commit, nothing frozen |
| Concurrency (CONC-*) | No locks added; waits bounded; permits not held while waiting | n/a | n/a |
| n > 1 | Each choice registers its own key | n/a | Indices are per choice |
| Usage / LEAK-6 | n/a | +~60 tokens per tool round, real and reported | n/a |
| Disk | n/a | +~0.5 KB per tool round per snapshot | n/a |
| Restart (CONV-16) | Registry lost; graceful shutdown completes freezes first | Old stripped snapshots still valid | n/a |
| Frozen-API crates | engine internal only | claude internal only | **every crate** matches on `TurnEvent` exhaustively and must handle 2 new variants |

## 19. New scenarios

### S19. Long tool loop, cache stays warm (G1 + G2)

```
req N    [.. tr(N-1) gA gU]          → assistant(N) tool_use, ToolCalls, Done
         engine registers pending P(N) (Awaiting) before Done reaches the client
client   runs tool, sends req N+1 = [.., assistant(N), tool(N)]
engine   stored deepest = checkpoint(N-1); pending P(N) is deeper and Awaiting-compatible → wait
claude   exits, placeholder written, freeze keeps gA/gU, indexed, P(N) resolves (≈1 s)
engine   base = checkpoint(N): fork, patch placeholder with tool(N), add gA(N+1) gU(N+1)
API      prompt(N+1) = prompt(N) + assistant(N) + tool(N) + gA + gU   → cache read ≈ whole of prompt(N)
```

Expected: cache read ≥ 90% for every turn of a long loop; depth = (request length − 1) on every tool round.

### S20. Freeze fails while a request waits (G2)

Freeze errors (disk full, verification failed) → guard drops → `Failed` → the waiting request plans from the best stored checkpoint and rewrites the tail (today's path). One cold turn; no error to the caller.

### S21. Freeze slower than the wait budget (G2)

claude takes the full 10 s grace to exit; the waiter gives up after `pending_wait_ms` (5 s) and plans from the older base. The late checkpoint still gets indexed and serves later requests (forks of this conversation, regenerations).

### S22. Regenerate while the previous turn is still freezing (G2)

The client resends request N (not N+1). Its prefix ends before assistant(N), so P(N) is not a prefix of it and is not waited for; it plans from checkpoint(N-1) and produces an independent answer (CONV-3 unchanged).

### S23. Client cancels while waiting on a pending checkpoint (G2)

Waiting is selected together with the request's cancellation token, so it stops immediately; nothing was started on the backend.

### S24. Big file write is streamed (G3)

```
t=0.0  request
t=1.6  claude content_block_start(tool_use write) → Start{0, toolu_…, "write"} → client sees the call begin
t=1.7… input_json_delta × ~400                    → Delta{0, "…"}             → client renders progress
t=70   message_stop → commit ToolCalls (arguments == concat) → Done(tool_calls)
```

Time to first byte ≈ 1.6 s instead of 70 s; no first-token timeout possible.

### S25. parallel_tool_calls=false with two calls (G3)

Call 0 streamed; call 1's Start/Delta swallowed; commit keeps call 0; `altered` → no checkpoint; next request writes the one-call assistant message via the append path (S18-style).

### S26. tool_choice required, model calls a tool (G3)

Buffered mode: Start/Delta swallowed, commit validated, then emitted as one full chunk. If the model answered in text instead, Retry is still possible because nothing was emitted.

### S27. Upstream error in the middle of streamed arguments (G3)

Client has a started, partial call; stream ends with the error event (STR-13). Engine discards the turn; no checkpoint, no pending promise survives. Client retry → normal request.

### S28. Client disconnects mid-arguments (G3 + G2)

Cancellation → abort → claude killed → discard; the pending promise was never registered (Done not reached). Nothing changes for later requests.

### S29. Meta question after 40 glued tool rounds (G1)

"How many messages have I sent you?" / "Did I ever say Continue?" The history contains 40 glue pairs. With the strengthened runtime note the model must count only real user messages. Gate V2: if it doesn't, set `keep_glue: false` (back to today's behaviour) and accept the cache cost.

### S30. Import of a foreign history, then a tool loop (G1)

The imported part has no glue (cold turn); from the first tool round on, glue is added and S19 applies.

### S31. Compaction during a long run (G1 + G2, already observed live)

Compaction request resumes the long chain warm; the first post-compaction request is a fresh short chain (cold); S19 applies again from its first tool round. Pending promises from the long chain are simply never matched.

## 20. Theoretical re-run of the 24-minute Sonnet run with v3

Using the measured request log (71 turns, 5.04M input tokens, 3.85M cache read):

| Metric | Measured (v2) | Expected (v3) | Why |
|---|---|---|---|
| Turns resuming the immediately preceding checkpoint | ~60/71 | ~69/71 (all except the first turn and the first post-compaction turn) | G2 removes the race misses (#62 etc.) |
| Cache read | 76% | ~93–96% | G1: every tool round reuses all of the previous prompt; only per-turn new tail (tool output + new glue) is uncached |
| Uncached input tokens | ~1.19M | ~0.2–0.35M | follows from the line above |
| Time to first byte p90 | 37 s | ~2–3 s | G3: first Start within ~1–2 s of the tool decision |
| Max time to first byte | 88 s | ~3 s | G3 |
| Risk of first-token timeout on huge writes | present (≥120 s writes) | none | G3 |
| Added latency from waiting | 0 | typically 0–1 s on ~10–20% of turns, bounded by 5 s | G2, only when racing |
| Context per long session | baseline | +~60 tokens per tool round | G1 glue |

## 21. Verification plan (before and after implementing)

| # | Check | How | Pass criterion |
|---|---|---|---|
| V1 | claude tolerates the extra `bridgeGlue` key and keeps it on resume | probe: write marker, resume twice, inspect file | key preserved, no API error |
| V2 | Model quality with persistent glue | session_lab-style probe with 30–40 glued rounds + meta questions; conformance CONV/LEAK reruns | correct message counts; never mentions glue; S7 direct-answer test 4/4 |
| V3 | Cache extension really happens | live tool loop of 15 rounds without user messages | cache read per turn ≥ 90% of prompt from round 2 on |
| V4 | Pending wait works and is bounded | fake backend with configurable freeze delay; engine tests for S19–S23 | hits after wait; fallback on failure/timeout; cancel stops wait |
| V5 | Streaming shape equals OpenAI | conformance STR-6/7 + new STR-16 with raw HTTP and both SDKs (SDK accumulates arguments) | SDK-reconstructed arguments byte-equal to the commit |
| V6 | Enforcement modes never leak calls | policy unit tests for every mode in 17.3 with Start/Delta input | no Start/Delta forwarded in buffered modes; parallel=false never forwards call 1 |
| V7 | End to end | `e2e/opencode_e2e.py`, `e2e/bigtask.py --long` on Sonnet and Haiku | all checks pass; cache ≥ 90%; TTFB p90 ≤ 3 s; 0 errors |

## 22. Change list by crate

| Crate | Change | Size |
|---|---|---|
| bridge-types | 2 `TurnEvent` variants + ordering contract; `EngineConfig.pending_wait_ms` (new `engine` config section, default 5000) | small |
| bridge-engine | `PendingRegistry`; register before Done; wait in `prepare`; pending depth in routing; pass-through of new events; fake backend `stream_tool_args` + configurable freeze delay | medium |
| bridge-claude | glue marker + keep at freeze + `keep_glue` option + runtime-note sentence (G1); decoder Start/Delta, byte-exact commit (G3) | medium |
| bridge-policy | handling of Start/Delta per mode (G3) | small–medium |
| bridge-openai | stream Start/Delta chunks; skip already-started calls at commit (G3) | small |
| tests | engine S19–S23 with fake; policy V6; claude decoder fixtures with recorded `input_json_delta`; conformance STR-16 / NF-4 / NF-7 / CONC-7 | medium |

## 23. Implementation status

All three fixes are implemented. Workspace: 275 tests pass, clippy clean.

### Deviations from sections 17.1–17.3

| Where | Deviation | Why |
|---|---|---|
| engine (G2) | The concurrency slot is released after freeze+index but **before** the promise resolves | Otherwise a woken waiter competes with a still-held slot and can get 429 when `queue_wait_ms` is small |
| engine (G2) | Routing depth is computed after the wait (only a promise that actually arrived counts); several backends' promises are waited on concurrently with one budget | A failed/timed-out promise can't be a base, so it must not influence routing |
| engine (G2) | Pending lookup happens before the store lookup | A checkpoint that finishes between the two is then always seen |
| engine (G3) | The first-token timeout counts backend events before the policy | In buffered policy modes nothing reaches the caller until the commit; long buffered tool calls must not time out (STR-17) |
| policy (G3) | With `parallel_tool_calls: false` the forwarded call is renumbered to index 0 and the commit keeps the call the caller saw (by id) | Wire indices must be 0..k-1 |
| policy (G3) | At Done, every forwarded call is checked against the commit (same id, name, byte-identical arguments); a mismatch sets `altered` | Never freeze a checkpoint that differs from what the caller saw |
| claude (G1) | The trigger entry is written by claude, so it is marked `bridgeGlue` at freeze (only a new, trigger-only user entry from this run, preferring one parented on our filler) | The marker can't be set before claude writes it |
| claude (G1) | `keep_glue: false` also strips glue already present in a base snapshot when the next turn starts | So the switch takes effect on existing chains |
| claude (G3) | An empty concatenation counts as "no deltas" (one Delta with the serialised input, normally `{}`, is emitted) | claude always sends one empty `""` delta first, even for calls without arguments |

### Gate results

| Gate | Result |
|---|---|
| V1 marker survives resumes | Pass: 11 resumes, marker kept, glue count +2 per round, claude never inserted its own filler |
| V3 cache extension | Pass: from round 2 every request reads the entire previous prompt from cache: 96.2% → 97.2% per round over 11 rounds (was down to ~6% in long loops) |
| V2 quality with glue | Mixed: factual answers all correct (message counts, "did I say Continue" → No, verbatim quotes, direct answers 7/7), but 2 of 8 *meta* answers mentioned the glue ("several of my turns contained only 'Let me look at the tool results.'"; "the 'Continue' messages came from automatic system notes"). Decision: keep `keep_glue: true` (quota saving), watch the conformance LEAK probes; next mitigation if needed: remove the word "Continue" from the trigger text, or `keep_glue: false` |
| V4 pending wait | Pass (engine tests S19–S23, CONC-7: every round of a 6-round loop based on the newest checkpoint) |
| V5/V6 streaming shape and enforcement | Pass (frontend + policy tests; official Python and Node SDK accumulators reproduce arguments byte for byte) |
| V7 end to end | Pass (Sonnet 5.5): 13-phase long run 18.2 min, 71 requests, 0 errors, cache 96% (was 76%), uncached input 0.21M (was 1.19M), checkpoint reuse 97%, first token p50 3.3 s / p90 12.6 s (was 5.1 s / 37 s), built service 117 tests + 21/21 HTTP checks; opencode e2e 18/18; conformance 217/217 with raw HTTP + Python + Node SDKs, including all LEAK probes with glue kept. Streamed tool call measured directly: first `tool_calls` chunk at 1.8 s, 982 argument deltas, final JSON valid |

Remaining first-token outliers (20–63 s) are Sonnet thinking before acting: its thinking arrives with empty text, so nothing can be streamed until the tool call or text starts. This matches reasoning models on normal APIs; keep-alive pings hold the connection.

Not yet automated in the conformance suite: STR-16, STR-17, CONC-7, NF-4, NF-7. They are covered by the engine/frontend tests, the direct streaming measurement and the long-run metrics above.
