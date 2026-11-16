# completions-bridge

**An OpenAI-compatible chat completions API that answers with the agent CLI you are already logged into.**

Many coding agents come with a subscription or a free daily allowance but no API key: you can use them through their own CLI, not from your favourite editor or agent harness. completions-bridge runs locally, exposes `/v1/chat/completions` and `/v1/models`, and behind that drives the CLI for you. To any client (opencode, the official OpenAI SDKs, curl, …) it looks like a normal model provider.

The first backend is **Claude Code** (`claude`). The design is backend-agnostic; see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

> Not affiliated with Anthropic or OpenAI. Using a subscription CLI this way may fall outside the terms of some plans; check yours before relying on it.

## How it works, in one paragraph

Your client sends normal requests with its tools; the bridge never runs tools itself. Each turn, the bridge resumes a claude session it wrote itself, with claude's own tools switched off and your client's tools offered through a small MCP server that never executes anything. When the model calls a tool, the bridge stops claude and returns a standard `tool_calls` response; your client runs the tool and sends the result in the next request. Sessions are saved as immutable checkpoints and forked per request, so the API stays stateless (edits, undo, regenerate, parallel sessions all just work) while consecutive turns reuse claude's prompt cache (~96% of input tokens in long agentic sessions).

## Status

Built and tested on macOS (arm64) with claude 2.1.325. Verified against a written conformance spec with raw HTTP plus the official OpenAI Python and Node SDKs, with real opencode, and with 12 parallel opencode sessions. Details, numbers and known gaps: [docs/PRODUCTION.md](docs/PRODUCTION.md).

Scope: one user, localhost. Not designed for multi-user or network-exposed hosting.

## Requirements

- Rust (stable) to build.
- `claude` on your `PATH`, logged in (run `claude` once interactively).
- Unix-like OS (uses process groups; Linux should work, Windows is not supported).

## Quick start

```sh
cargo build --release
mkdir -p ~/.completions-bridge
cp config.example.yaml ~/.completions-bridge/config.yaml   # then set server.token
./target/release/completions-bridge serve
```

It listens on `http://127.0.0.1:4141/v1`. The first start runs a short self-test against claude (a few Haiku calls); the result is cached per claude version. `completions-bridge selftest` runs it on demand.

```sh
curl http://127.0.0.1:4141/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"model": "bridge/claude-haiku", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Use with opencode

Add a provider to `~/.config/opencode/opencode.json`:

```json
{
  "provider": {
    "bridge": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Completions Bridge",
      "options": { "baseURL": "http://127.0.0.1:4141/v1", "apiKey": "<server.token>" },
      "models": {
        "bridge/claude-sonnet": { "name": "Claude Sonnet (bridge)", "tool_call": true, "limit": { "context": 180000, "output": 32000 } },
        "bridge/claude-haiku":  { "name": "Claude Haiku (bridge)",  "tool_call": true, "limit": { "context": 150000, "output": 32000 } }
      }
    }
  }
}
```

Then select `bridge/bridge/claude-sonnet` (or `…-haiku`) in opencode. Keep `limit.context` comfortably below the model's real window: the client is responsible for compacting long sessions, exactly as with any API; the bridge never summarises behind your back.

## Other clients

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:4141/v1", api_key="<server.token>")
r = client.chat.completions.create(model="bridge/claude-haiku",
                                   messages=[{"role": "user", "content": "Hi"}])
```

Streaming, tool calls (streamed while the model writes them), parallel tool calls, `tool_choice`, `response_format` (JSON object / schema), `stop`, `max_tokens`, `n` and `reasoning_effort` are supported. Sampling parameters (`temperature`, `top_p`, `seed`, penalties) and `logprobs` are accepted and ignored; audio is rejected.

## Configuration

Everything is in one YAML file: `config.example.yaml` lists every setting with its default (`completions-bridge print-config` prints the same). Highlights:

| Setting | Default | Meaning |
|---|---|---|
| `server.listen` / `server.token` | `127.0.0.1:4141` / none | where to listen, bearer token callers must send |
| `models` | `bridge/auto`, `bridge/claude-{haiku,sonnet,opus}` | caller-visible model ids → backend + backend model |
| `backends.claude.default_model` | `claude-sonnet-5-5` | model for `bridge/auto` |
| `backends.claude.max_concurrency` | `12` | parallel claude processes; extra requests queue, then get 429 with Retry-After |
| `checkpoints.max_total_mb` / `ttl_hours` | `2048` / `168` | disk cap and lifetime of saved sessions |
| `backends.claude.tuning.*` | — | process grace periods, glue handling, auto-compaction window |
| `backends.claude.prompts.*` | built-in | every text the bridge injects |
| `engine.*`, `output.*` | — | pending-checkpoint wait, GC, quota backoff, policy texts |

Models and limits are configuration: change the file and restart, no rebuild.

## Files and privacy

State lives in `server.workdir` (default `~/.completions-bridge`), created owner-only (0700, files 0600):

- `state/bridge.db`: checkpoint index, quota state, request log (no conversation content).
- `claude/snapshots/`: saved sessions. They contain your conversations and code, and claude's own context; pruned by size and age.

Logs go to stderr (`BRIDGE_LOG=debug` for detail); conversation content is not logged at the default level.

## Repository layout

| Path | Contents |
|---|---|
| `crates/bridge-types` | shared types: request/response IR, backend trait, config schema |
| `crates/bridge-openai` | HTTP frontend: OpenAI wire format, validation, SSE, errors |
| `crates/bridge-engine` | stateless request handling: prefix hashing, checkpoints, planner, routing, quota, pending checkpoints, scripted fake backend |
| `crates/bridge-policy` | output policy: tool_choice, parallel calls, stop, max_tokens, JSON formats |
| `crates/bridge-claude` | Claude Code backend: session files, forks, stream decoding, parking MCP server, self-test |
| `crates/completions-bridge` | the binary |
| `docs/` | architecture, conformance spec, production-readiness report, early design notes |
| `e2e/` | end-to-end harnesses (real opencode, conformance with SDKs, readiness, long and parallel runs) |
| `probes/` | research scripts that established how claude behaves |

## Tests

```sh
cargo test --release                                       # unit + integration (fake claude, no quota)
BRIDGE_LIVE_CLAUDE=1 cargo test --release -p bridge-claude --test live   # live claude, ~15 Haiku calls
python3 e2e/readiness.py                                   # operational checks (load, crash, shutdown, permissions, ...)
python3 e2e/opencode_e2e.py --model claude-haiku           # real opencode → bridge → claude
python3 e2e/conformance/run.py                             # spec conformance, raw HTTP + Python + Node SDKs
python3 e2e/parallel12.py                                  # 12 parallel opencode sessions
```

The e2e harnesses need `opencode`, `node` and `uv`/`npm` (the conformance runner installs the SDKs under `target/sdk/`). They spend real claude quota; each script's header says roughly how much.

## License

MIT, see [LICENSE](LICENSE).
