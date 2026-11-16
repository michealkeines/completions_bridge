---
name: completions-bridge
description: Start the completions-bridge OpenAI-compatible API (backed by the local, logged-in Claude Code CLI) from this repository and report its base URL, API key and model names. Use when someone wants to run, start, restart or stop the bridge, get its URL or models, or connect opencode or an OpenAI SDK to it.
---

# Start completions-bridge and report the URL and models

completions-bridge is a local server that exposes an OpenAI-compatible chat completions API (`/v1/chat/completions`, `/v1/models`) and answers with the Claude Code CLI the user is already logged into. This skill builds it, starts it in the background and reports how to call it.

## Steps

1. **Run the start script from the repository root:**

   ```sh
   ./scripts/start.sh
   ```

   It is idempotent:
   - checks that `cargo`, `claude` and `curl` exist;
   - on first run creates `~/.completions-bridge/config.yaml` from `config.example.yaml`, with a random API key and a free port (4141 or the next free one), owner-only;
   - builds the release binary if it is missing or out of date (first build takes a few minutes);
   - starts the server in the background and waits until it answers (the first start runs a short self-test against claude, up to about a minute);
   - if the server is already running, just reports it.

   Optional environment: `BRIDGE_HOME` (state/config directory, default `~/.completions-bridge`), `BRIDGE_PORT` (port used when the config is first created).

2. **Report the script's final block to the user, verbatim values:** base URL, API key, backend mode and every model id listed. Example of what it prints:

   ```
   completions-bridge is running
     base URL : http://127.0.0.1:4141/v1
     api key  : cb-…
     backend  : claude (mode: native)
     models   :
       - bridge/auto
       - bridge/claude-haiku
       - bridge/claude-opus
       - bridge/claude-sonnet
       - bridge/claude-sonnet-5-5
   ```

3. **If the user wants to connect opencode**, give them this provider block with the reported URL and key filled in (add one entry per model they want), and tell them to pick `bridge/<model id>` in opencode:

   ```json
   {
     "provider": {
       "bridge": {
         "npm": "@ai-sdk/openai-compatible",
         "name": "Completions Bridge",
         "options": { "baseURL": "<base URL>", "apiKey": "<api key>" },
         "models": {
           "bridge/claude-haiku":  { "name": "Claude Haiku (bridge)",  "tool_call": true, "limit": { "context": 150000, "output": 32000 } },
           "bridge/claude-sonnet": { "name": "Claude Sonnet (bridge)", "tool_call": true, "limit": { "context": 180000, "output": 32000 } }
         }
       }
     }
   }
   ```

   For an OpenAI SDK: `base_url=<base URL>`, `api_key=<api key>`, `model=<model id>`.

4. **To stop it:** `./scripts/stop.sh` (graceful; in-flight turns finish saving).

## If something goes wrong

| Symptom | What to do |
|---|---|
| `cargo not found` | Install Rust from https://rustup.rs, then rerun. |
| `claude not found`, or `mode: disabled` | Install Claude Code and log in by running `claude` once interactively; then `./scripts/stop.sh && ./scripts/start.sh`. |
| `server exited` / not ready | Show the last lines of `~/.completions-bridge/bridge.log` to the user. |
| Port already in use | Edit `server.listen` in `~/.completions-bridge/config.yaml`, then restart. |
| Wrong or missing models | Models are configuration (`models:` and `backends.claude.default_model` in the config file); edit and restart, no rebuild needed. |

## Notes

- It listens on 127.0.0.1 only and requires the API key on every route.
- Requests spend the user's own claude subscription quota. Using a subscription CLI this way may fall outside some plans' terms; mention this if the user seems unaware.
- More detail: `README.md`, `docs/ARCHITECTURE.md`, `docs/PRODUCTION.md`.
