> **Historical note.** These are the first exploratory notes (context strategies, token estimates). The design they led to is in [ARCHITECTURE.md](ARCHITECTURE.md); the measurements here are still referenced there.

# Completions bridge: lifecycle walk-through and context strategy

One CLI (opencode) talks to a local OpenAI-compatible server. The server answers each request by one-shotting a backend agent CLI (`claude`, `codex`, others) that still has free quota today.

This document runs one realistic coding task through the system on paper, lists what goes wrong, and compares two ways of handling context.

**Status of the facts used here**

- Verified: `claude -p` with its tools disabled and a replaced system prompt behaves as a plain model. It followed the text tool-call protocol, streamed, took about 3 seconds, and emitted a `rate_limit_event` with quota utilisation and reset times.
- Not verified: `codex exec` (the local config points it at an Ollama provider with a missing key), the exact shape and size of opencode's requests, and how each provider counts cached tokens against free limits.
- All token numbers below are estimates for comparing options, not measurements.

## 1. The two options

| | Option A: full transcript | Option B: layer-held condensed context |
|---|---|---|
| What the backend gets | Everything opencode sent: system prompt, tools, the whole history | A brief written by the layer, the last few turns, selected file contents |
| Who owns context | opencode (it resends everything each request) | The layer (it decides what the backend sees) |
| Layer state | Small: sticky backend, quota, retry cache | Large: brief, pinned files, session matching |
| Backend switch | Next backend gets the same full transcript | Next backend gets the brief |

In both options opencode runs the tools and owns the loop. The layer only produces one assistant turn per request.

## 2. The task

> "Add rate limiting to the `/login` endpoint, with tests."
> (Express + TypeScript repo)

It exercises: todo list, search, several file reads, parallel tool calls, a new file, an edit, a test run that fails, a fix, a backend running out of quota mid-task, and one malformed tool call.

Assumed sizes: opencode system prompt plus tool definitions plus our protocol instructions come to about 10,300 tokens, sent on every request. The user message is about 100.

## 3. Lifecycle of one request (both options)

1. opencode POSTs `/v1/chat/completions` with `stream: true`, the system prompt, about a dozen tool schemas, and the full message history (including earlier `tool_calls` and `tool` results).
2. The layer picks a backend: the session's current one if it has quota, otherwise the next in the list.
3. The layer renders a one-shot prompt:
   - fixed part: protocol rules, opencode's system prompt, the tool catalogue;
   - the conversation as a labelled transcript;
   - "write the next assistant turn".
4. The backend replies in text. If it wants tools, it ends with a block fenced by a per-request random marker:
   ```
   I'll look at the route first.
   <<<CALLS_7f3a
   [{"name":"read","arguments":{"filePath":"src/routes/auth.ts"}}]
   CALLS_7f3a>>>
   ```
5. The layer streams the prose to opencode as content deltas, holds back anything that could be the start of the marker, parses the block, validates each call against the tool schema, and emits real `tool_calls` with `finish_reason: "tool_calls"`.
6. opencode runs the tools and sends the next request with the results appended.

## 4. The task, request by request (Option A)

| # | What opencode sends that is new | What the model returns | Input tokens | Output |
|---|---|---|---|---|
| 0 | Separate title-generation request, no tools | A title | ~300 | 20 |
| 1 | User prompt | `todowrite` + `grep "login"` (two calls) | 10,400 | 150 |
| 2 | Todo ack, grep hits (500) | `read src/routes/auth.ts` | 11,050 | 80 |
| 3 | File (2,500) | `read app.ts` + `read package.json` in parallel | 13,630 | 100 |
| 4 | Two files (2,400) | `read tests/auth.test.ts` | 16,130 | 80 |
| 5 | File (3,000) | `write src/middleware/rateLimit.ts` | 19,210 | 650 |
| 6 | Write ack | `edit src/routes/auth.ts` | 19,910 | 300 |
| 7 | Edit ack | `write tests/rateLimit.test.ts` | 20,310 | 950 |
| 8 | Write ack | `bash npm test` | 21,310 | 60 |
| 9 | Failing test output (2,000) | `edit rateLimit.ts` (the fix) | 23,370 | 350 |
| 10 | Edit ack | `bash npm test` | 23,820 | 60 |
| 11 | Passing output (800) | `bash npx tsc --noEmit` | 24,680 | 60 |
| 12 | Clean output (300) | Final summary, no tool call | 25,040 | 300 |

Totals for requests 1–12: about **229,000 input** and **3,100 output** tokens across **12 requests**.

Where the input goes:

- Fixed overhead (system prompt and tools, 10,300 × 12): about 124,000, or **54%**.
- History: about 104,000, or 45%.

On a task this size the repeated system prompt and tool list cost more than the history does.

### Events during the run

- **Request 0 (title).** A small side request with no tools. It should go to the cheapest backend and must not create or disturb the main session.
- **Request 3 (parallel calls).** The block holds two calls; the layer emits two `tool_calls` with distinct ids, and opencode returns two `tool` messages.
- **Request 6 (malformed call).** The backend returns broken JSON in the block. The layer re-asks the same backend once with the parse error, and opencode never sees the bad output. That costs one extra backend request.
- **Request 7 (quota runs out).** `claude` returns a limit error before any output. The layer marks it exhausted until the reported reset time and replays the same request on `codex`. opencode sees a slower response, nothing else.
- **Request 9 (failed edit).** If the `edit` old-text doesn't match, opencode returns an error as the tool result and the model retries. That is the normal loop; the layer does nothing special.

## 5. The same task under Option B

Two variants, because "condensed" can mean two things.

**B1: brief plus pinned file contents.** The layer keeps the latest content of every file the task touches, since edits need exact text. In this task the history is mostly those files, so there is little to drop: grep hits, the stale failing test output, old chatter.

- Input for the 12 requests: about 216,000 (6% less than A).
- Two brief updates, each a backend one-shot that reads about 15,000 tokens: +30,000.
- Net: about **246,000**, which is worse than A, with 14 backend requests instead of 12.

**B2: brief plus recent turns only, no pinned files.**

- Input: about 13,000 per request, 156,000 total.
- Brief updates: +30,000.
- The model no longer has the exact text of `auth.ts` when it edits, so it re-reads or produces a failing edit. Assume three extra turns: +42,000.
- Net: about **228,000**, the same as A, with about 17 requests and a real chance of a wrong edit.

## 6. A long task (60 requests, history growing to 120,000 tokens)

| Strategy | Approx. input tokens |
|---|---|
| A, no compaction | 4.2 million (history is 85% of it) |
| A, with opencode compacting at about 60,000 | 2.6 million |
| B, condensing at checkpoints | About the same as the row above |

History only dominates on long tasks, and opencode already has auto-compaction for that. The threshold follows the context limit declared for the model in opencode's provider config, so declaring a smaller limit makes it compact earlier with no context logic in the layer. This config behaviour needs confirming.

## 7. Which is more token-friendly

It depends on what the free limit counts.

- **Limits counted in requests** (typical of free API tiers): A wins. B adds brief-writing requests and re-read turns.
- **Limits weighted by tokens** (Claude and Codex subscriptions): roughly a tie on short tasks; on long tasks compaction matters, and A gets it from opencode.
- **Prompt caching.** A's prompt only ever grows at the end, so the repeated prefix can be served from cache on the same backend. B rewrites the middle of the prompt whenever the brief changes, which breaks the cache. Two caveats:
  - The `claude` probe showed a cache write; a cache read across separate one-shot calls is not yet confirmed.
  - A transcript flattened into one text block may only cache the system-prompt part on Claude. Feeding the history as separate messages (`--input-format stream-json`) may fix that and is worth testing.

## 8. Downsides and failure modes

### Shared by both options

| Issue | Handling |
|---|---|
| Model writes prose after the call block, or invents a tool result | Cut everything after the closing marker |
| Marker text appears in the user's code | Marker carries a random per-request suffix |
| Large file content inside JSON arguments gets mis-escaped | Tolerant parser, then one repair re-ask |
| Backend tries to use its own tools | Disabled for `claude`; for `codex`, read-only sandbox in an empty directory and ignore its tool events (untested) |
| Quota error after output has started streaming | Cannot be hidden; end the stream with an error. Rare, since limit errors arrive up front |
| Backends differ in ability and style mid-task | Keep a session on one backend until it runs out |
| Backends differ in context window | Declare the smallest one to opencode |
| Process start-up adds seconds per turn | Accept; send keep-alives so opencode doesn't time out |
| Parallel requests (title, subagents) | Per-backend concurrency cap |
| User aborts | Kill the child process on client disconnect |
| Images in messages | Dropped with a note in the first version |
| File content that tells the model to do things | Same exposure as any agent; opencode's permission prompts remain the guard |

### Specific to Option A

- Pays the full history on every request until opencode compacts.
- A backend switch starts with a cold cache.

### Specific to Option B

- The layer decides what the model sees. A stale or wrong brief causes wrong edits that are hard to trace.
- Session matching is fragile: the API has no session id, and opencode can compact, undo or fork the history under us.
- Two summarisers (ours and opencode's compaction) can stack, losing detail twice.
- Extra backend requests to maintain the brief.
- Much more code, and more state that can break.

## 9. Recommendation

**Option A, with a thin stateful layer.** It is simpler, loses nothing, is cache-friendly, and on the numbers above B does not save tokens on short tasks and only matches opencode's own compaction on long ones.

Keep these cheap savings, none of which rewrite history:

1. Put the fixed part (protocol, system prompt, tool catalogue) where the backend can cache it.
2. Shorten the tool catalogue text, since fixed overhead is over half the cost on short tasks.
3. Cap very large tool outputs.
4. Declare a modest context limit to opencode so it compacts early.
5. Send title and other no-tool requests to the cheapest backend.

Layer state is limited to: which backend a session is stuck to, quota and reset times per backend, a retry cache, and logs.

## 9a. Measured update: resumed sessions (supersedes the caching caveats in section 7)

Test on `claude` (Haiku, one run each, a transcript of about 21,000 tokens):

| Call | Wall time | Tokens written to cache | Tokens read from cache |
|---|---|---|---|
| One-shot, full transcript | 2.8s | 20,788 | 0 |
| One-shot, full transcript plus one more turn | 3.0s | 21,104 | 0 |
| Resume the session, send only the new turn | 3.3s | 693 | 20,440 |

- **Time:** a full transcript is not slower. All calls take about 3 seconds, the same as a tiny prompt.
- **Tokens:** a flattened transcript gets no cache reuse as it grows; the whole thing is processed fresh each turn. Resuming a backend session and sending only the new input makes about 97% of it a cache read.

Revised design: **resume the backend's session by id and send only the delta; fall back to a full transcript when that isn't possible.**

- No long-lived process or stdin injection is needed. Each turn is still a separate CLI call (`claude --resume <id>`, `codex exec resume <id>`), so it works for any CLI that can resume a session.
- The layer keeps a map from "hash of the messages already delivered" to a backend session id. If an incoming request is that history plus new messages, it sends only the new ones.
- Fallback to a new session with the full transcript when: the history doesn't match (opencode compacted, undid or forked), the system prompt or tool list changed, the backend switched, or the backend can't resume.
- New downsides: the backend may compact its own session without us knowing; a repair re-ask stays in the backend's history; persisted session files accumulate; a retried request must not be injected twice.
- Unverified: `codex` resume, and how each provider counts cached tokens against its free limit.

## 10. To verify before building further

1. Capture a real opencode request: actual system prompt and tool sizes, and whether any header identifies the session.
2. `codex exec` as a one-shot backend with user config ignored.
3. Whether cache reads happen across separate `claude -p` calls, and whether multi-message input improves it.
4. That opencode's compaction threshold follows the declared context limit.
5. The exact limit-error output of each backend, for exhaustion detection.
