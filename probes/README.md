# Probes

Research scripts used while designing the bridge. They talk to the real `claude` CLI and
established the facts the architecture relies on (docs/ARCHITECTURE.md, sections 0.1 and 0.2):
which flags disable claude's own tools, how its session files look, how stopping at a tool
call and resuming behaves, and which wording keeps the model answering normally.

They are kept for reference and for re-checking those facts after a claude update. They are
not needed to build or run the service.

| Script | What it checks |
|---|---|
| `claude_intercept.py` | Live turn-by-turn tool interception: stop at a tool call, patch the placeholder result, resume |
| `session_lab.py` | Session files written from scratch, forks, system prompt / tool changes on resume, filler behaviour (T0–T8) |
| `fake_mcp.py` | Minimal stdio MCP server that advertises tools and logs (or parks) calls |
| `fake_openai.py` | Scripted OpenAI-compatible server, used to see exactly what a client sends |
| `summarize_stream.py` | Summarises a claude `stream-json` or opencode `--format json` event file |

Most of them cost a few real claude calls (Haiku). Run them in a scratch directory.
