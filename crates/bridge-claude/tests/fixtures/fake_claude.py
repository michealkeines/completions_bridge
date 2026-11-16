#!/usr/bin/env python3
"""Fake `claude` for process tests (no quota).

Replays recorded stream-json lines and mimics the session-file behaviour observed on
claude 2.1.325:
  - writes the stdin user message (after a synthetic "No response requested." filler
    when the file ends with a user entry) and the assistant entries of the fixture;
  - when stopped (SIGTERM) after tool calls, writes ONE "Connection closed" placeholder
    for the first call, parented to the first tool_use entry (the parallel-call quirk);
  - starts the MCP server from --mcp-config (so tests can check it gets killed).

Extra args (passed via BackendConfig.extra_args):
  --fixture FILE     stream-json lines to replay
  --fake-projects D  projects root to write session files into
  --fake-mode M      replay (default) | hang | crash
  --pidfile F        append own pid and child pids here
"""
import json, os, re, signal, subprocess, sys, time, uuid

argv = sys.argv[1:]
if "--version" in argv:
    print("2.1.325 (Claude Code)")
    sys.exit(0)


def opt(name, default=None):
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


fixture = opt("--fixture")
projects = opt("--fake-projects")
mode = opt("--fake-mode", "replay")
pidfile = opt("--pidfile")
sid = opt("--session-id") or opt("--resume")
persist = "--no-session-persistence" not in argv
cwd = os.path.realpath(os.getcwd())
path = None
if projects and sid:
    d = os.path.join(projects, re.sub(r"[^a-zA-Z0-9]", "-", cwd))
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, sid + ".jsonl")


def note_pid(pid):
    if pidfile:
        with open(pidfile, "a") as f:
            f.write(f"{pid}\n")


note_pid(os.getpid())

children = []
mcp = opt("--mcp-config")
if mcp:
    cfg = json.load(open(mcp))["mcpServers"]["bridge"]
    p = subprocess.Popen([cfg["command"]] + cfg.get("args", []), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}}) + "\n")
    p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n")
    p.stdin.flush()
    p.stdout.readline()
    tools = json.loads(p.stdout.readline())["result"]["tools"]
    if pidfile:
        with open(pidfile + ".tools", "w") as f:
            json.dump(tools, f)
    children.append(p)
    note_pid(p.pid)

if mode == "crash":
    sys.stderr.write("Error: Not logged in · Please run /login\n")
    sys.exit(1)

entries = []
if path and os.path.exists(path):
    entries = [json.loads(l) for l in open(path) if l.strip()]


def conv():
    return [e for e in entries if e.get("type") in ("user", "assistant")]


def parent():
    c = conv()
    return c[-1]["uuid"] if c else None


def add(kind, message, par=None, **extra):
    e = {"parentUuid": par if par is not None else parent(), "isSidechain": False, "type": kind,
         "message": message, "uuid": str(uuid.uuid4()), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
         "sessionId": sid, "cwd": cwd, "version": "2.1.325"}
    e.update(extra)
    entries.append(e)
    return e


def save():
    if path and persist:
        with open(path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")


line = sys.stdin.readline()
if pidfile:
    with open(pidfile + ".stdin", "a") as f:
        f.write(line)
    with open(pidfile + ".argv", "w") as f:
        json.dump(argv, f)
    sysfile = opt("--system-prompt-file")
    if sysfile:
        with open(pidfile + ".system", "w") as f:
            f.write(open(sysfile).read())
msg = json.loads(line)["message"] if line.strip() else {"role": "user", "content": ""}
if conv() and conv()[-1]["type"] == "user":
    add("assistant", {"id": "msg_synth", "model": "<synthetic>", "role": "assistant", "type": "message",
                      "content": [{"type": "text", "text": "No response requested."}]})
add("user", {"role": "user", "content": msg.get("content", "")})
save()

tool_entries = []


def on_term(signum, frame):
    if tool_entries:
        first = tool_entries[0]
        tid = first["message"]["content"][0]["id"]
        add("user", {"role": "user", "content": [{"type": "tool_result", "content": "Connection closed",
                                                   "is_error": True, "tool_use_id": tid}]},
            par=first["uuid"], toolUseResult="Error: Connection closed", sourceToolAssistantUUID=first["uuid"])
        save()
    for c in children:
        try:
            c.terminate()
        except Exception:
            pass
    os._exit(143)


signal.signal(signal.SIGTERM, on_term)

if mode == "hang":
    s = subprocess.Popen(["sleep", "1000"])
    note_pid(s.pid)
    print(json.dumps({"type": "stream_event", "event": {"type": "content_block_delta", "index": 0,
                                                           "delta": {"type": "text_delta", "text": "partial"}}}), flush=True)
    while True:
        time.sleep(1)

has_result = False
for raw in open(fixture):
    raw = raw.strip()
    if not raw:
        continue
    e = json.loads(raw)
    if e.get("type") == "assistant":
        m = e["message"]
        if any(b.get("type") == "tool_use" and b.get("name", "").startswith("mcp__") for b in m["content"]):
            tool_entries.append(add("assistant", m))
        else:
            add("assistant", m)
        save()
    if e.get("type") == "result":
        has_result = True
    sys.stdout.write(raw + "\n")
    sys.stdout.flush()
    time.sleep(0.005)

if has_result:
    # Like claude in stream-json input mode: wait for more input until stdin closes.
    sys.stdin.read()
    for c in children:
        c.terminate()
    sys.exit(0)
while True:  # waiting for the (parked) tool call
    time.sleep(1)
