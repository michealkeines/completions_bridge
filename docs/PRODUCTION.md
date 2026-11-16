# Production readiness

Scope: a single-user service on the user's own machine, exposing an OpenAI-compatible API on localhost, backed by the Claude Code CLI the user is logged into. "Production" here means: safe to run all day as the only model provider for opencode (and other OpenAI clients), unattended, with predictable failure behaviour.

Verdict: **ready for that scope**, with the open items in section 4 accepted. Not ready (and not designed) for multi-user, network-exposed or hosted use.

All evidence below was produced on the final build (2026-11-16, claude 2.1.325) unless stated otherwise. Re-run commands are in section 5.

## 1. What was checked and confirmed

### 1.1 Functional correctness

| Criterion | Evidence | Result |
|---|---|---|
| Unit and integration tests, all crates | `cargo test --release` | 291 pass, 0 fail |
| Static analysis | `cargo clippy --release --all-targets` | 0 findings |
| Behaves like a normal OpenAI-compatible API for any client | `e2e/conformance/run.py`: every client-observable P0/P1 case in CONFORMANCE.md, judged by raw HTTP, official OpenAI Python SDK and official Node SDK, real claude (Sonnet 5.5) | 217/217, 0 P0/P1 failures |
| Works with the real target client | `e2e/opencode_e2e.py` on Sonnet 5.5: fix a bug, add a feature with tests, memory across turns, fork, two parallel sessions, direct answers after tool results, bridge restart mid-session | 18/18 |
| Long real work | `e2e/bigtask.py --long` (v3 build): 13 tickets in one opencode session, 18 min, 94 tool calls, 71 API requests; built service independently verified over HTTP | 0 errors; built service 117 tests + 21/21 checks |
| Streamed tool calls | direct measurement: 14.8k-char `write` call | first tool-call chunk at 1.8 s, 982 argument deltas, final JSON valid |
| 12 parallel opencode sessions (Haiku) | `e2e/parallel12.py`: 12 different projects, build + extend each, one bridge at its 12-slot limit | 12/12 successful, 820 project tests pass, 170 requests, 0 errors, peak 12 in flight / 12 claude processes, bridge RSS ≤ 24 MB, cache 89%, checkpoint reuse 92%, 0 stray processes |
| Compaction lifecycle | observed live twice: opencode compacts, the next request starts a fresh short backend session, later turns warm again | as designed |

### 1.2 Reliability and failure behaviour (`e2e/readiness.py`, 16/16)

| ID | Check | Result |
|---|---|---|
| R1 | 3000 mixed streaming/non-streaming requests, concurrency 24 (fake backend) | 0 errors, ~3,900–4,500 req/s, bridge overhead p50 6 ms / p99 13 ms (target ≤ 500 ms) |
| R1 | Leaks after 3000 requests | RSS 11 → 16 MB, file descriptors 17 → 17 |
| R1 | Every request logged | 3050/3050 rows |
| R2 | Concurrency limit 12 | 12 served, 18 got 429 with Retry-After |
| R3 | SIGTERM with a real claude turn in flight | clean exit (code 0), claude stopped |
| R4 | `kill -9` mid-turn, then restart | orphaned claude killed on restart, turn dirs cleaned, next request 200 |
| R5 | claude binary missing | server starts, backend disabled, requests get OpenAI-format 503, no internals in the message |
| R6 | Every checkpoint file corrupted | next request answered correctly from the request history |
| R7 | Disk cap (1 MB) under heavy writes | held at the cap |
| R9 | Invalid config (bad YAML, unknown field, route to missing backend) | refuses to start, clear message |
| R11 | Self-test caching | first start 6.9 s, second 0.2 s |
| — | Stray processes after all checks | none |

### 1.3 Security

| Criterion | Evidence | Result |
|---|---|---|
| Network exposure | binds 127.0.0.1 by default; warns when bound elsewhere or without a token | ok |
| Auth on every route | R10: `/v1/models`, `/v1/models/{id}`, `/v1/chat/completions`, `/admin/status` without token → 401 | ok |
| Data at rest is private | R8: workdir 0700, every file owner-only (umask 077 for the bridge and the claude processes it starts) | ok (was world-readable before this check) |
| Prompts not exposed to other local users | R8: system prompt passed via a 0600 file (`--system-prompt-file`), not on the command line | ok (was visible in `ps` before this check) |
| No hidden capabilities | claude runs with all built-in tools disabled, in an empty working directory, without user/project settings; the parking MCP server never executes anything | ok |
| No leaks of internals to callers | conformance LEAK-* probes, e2e leak scans (no `mcp__`, sandbox paths, glue text, account email in any response or error) | ok |
| Dependencies | `cargo audit` | clean (fixed: SQLite CVE via rusqlite 0.43; replaced unmaintained `serde_yaml` with `yaml_serde`) |
| Injection | all child-process arguments are passed as argv vectors (no shell); caller text reaches claude only via stdin JSON and files | ok by construction |

### 1.4 Operability

| Criterion | State |
|---|---|
| Configuration | every behavioural constant is in the config file (`config.example.yaml`, `completions-bridge print-config`); only protocol facts are fixed, each with a comment |
| Status | `/admin/status`: backend mode, version, quota state, in-flight turns, checkpoint count/size/pending, uptime |
| Request log | SQLite `request_log`: model, backend, checkpoint depth, appended messages, finish reason, error class, tokens, cache reads, time to first token, duration |
| Logs | stderr, `BRIDGE_LOG` filter; no conversation content at info level (checked) |
| Shutdown | SIGINT and SIGTERM both graceful; waits up to `server.shutdown_grace_s` for checkpoints being saved |
| Upgrades of claude | version change re-runs the self-test; snapshots of other versions are ignored; falls back to transcript mode if session tricks stop working |
| Disk | snapshots capped (`checkpoints.max_total_mb`, enforced immediately) and expired (`ttl_hours`); measured ~0.8 MB per checkpoint, ~58 MB per 18-minute heavy session |

## 2. Performance (Sonnet 5.5, 13-ticket run, v3 build)

| Metric | Value |
|---|---|
| Prompt-cache read | 96% of input tokens |
| Checkpoint reuse | 97% of turns (misses: first turn, first turn after compaction) |
| Time to first token | p50 3.3 s, p90 12.6 s |
| Bridge overhead | p99 13 ms |

## 3. What cannot be confirmed yet (and why)

| Item | Why not confirmed | Risk | Mitigation in place |
|---|---|---|---|
| Real quota exhaustion | never hit the real subscription limit; the exact claude signals are unknown (Q7) | medium: the first real exhaustion might be classified as a generic error instead of 429 with reset time | 429/Retry-After path verified with the fake backend; unknown errors back off; `rate_limit_event` utilisation is tracked |
| claude auto-compaction really disabled | flag accepted, effect unproven (Q15) | low: callers compact far earlier | `--autocompact` set to 1M |
| 8-hour soak, idle-for-hours requests | not run (NF-6, CONV-15) | low–medium | no long-lived processes; R1 shows no growth over 3000 requests; checkpoints are files |
| Failover between real backends | only claude exists | n/a until a second backend is added | failover verified between fake backends |
| Glue mentioned in meta answers | seen 2/8 in an isolated probe; never in conformance or e2e | low | `tuning.keep_glue: false` turns it off (costs cache) |
| Model thinking delays | first token can take 20–60 s when Sonnet thinks before acting (thinking text arrives empty) | UX only | keep-alive pings; same as reasoning models on normal APIs |
| Platforms | built and tested on macOS (arm64) only; process groups use Unix APIs | Linux likely fine, Windows unsupported | — |
| Terms of service | automating a subscription CLI as an API may fall outside some plans' terms | policy, not technical | documented in README |

## 4. Accepted limitations for this scope

- Single user, localhost; no TLS, no per-user isolation, no rate limiting per caller (one bearer token).
- Sampling parameters (`temperature`, `top_p`, `seed`, penalties), `logprobs` and audio are not supported (accepted/ignored or rejected like a normal host without the feature).
- The service depends on claude's undocumented session-file format and on `--system-prompt-file` (not listed in `claude --help`). The self-test detects breakage after updates and degrades to transcript mode; a claude update can still reduce cache efficiency until the bridge is adapted.

## 5. How to re-confirm

```sh
cargo build --release
cargo test --release && cargo clippy --release --all-targets && cargo audit
python3 e2e/readiness.py                                   # 16 checks, ~8 real Haiku calls
BRIDGE_LIVE_CLAUDE=1 cargo test --release -p bridge-claude --test live
python3 e2e/opencode_e2e.py --model claude-sonnet          # real opencode
CONF_MODEL=bridge/claude-sonnet python3 e2e/conformance/run.py   # ~115 real calls
python3 e2e/bigtask.py --bin target/release/completions-bridge --model claude-sonnet --long   # ~20 min
```
