#!/usr/bin/env python3
"""Live probe: drive one claude session turn by turn, intercepting MCP tool calls.

Each turn starts `claude -p` (new session or --resume), feeds one stream-json
message on stdin, prints events live, and stops claude once the assistant
message that requested our MCP tools is complete. Before the next resume the
session file is cut back to that assistant message, removing the synthetic
"Connection closed" tool_result claude writes when it is stopped, so the real
tool results we send follow the tool calls directly.

usage: claude_intercept.py WORKDIR
"""
import json
import os
import signal
import subprocess
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.abspath(sys.argv[1])
CWD = os.path.join(WORK, "empty")
os.makedirs(CWD, exist_ok=True)
SID = str(uuid.uuid4())
T0 = time.time()
PREFIX = "mcp__bridge__"

# Caller's tools, as opencode would send them (OpenAI format).
CALLER_TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "Read a file in the user's project.",
     "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}}, "required": ["filePath"]}}},
    {"type": "function", "function": {"name": "edit", "description": "Replace exact text in a project file.",
     "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}, "oldString": {"type": "string"},
                    "newString": {"type": "string"}}, "required": ["filePath", "oldString", "newString"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command in the user's project.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "description": {"type": "string"}},
                    "required": ["command"]}}},
]
CALLER_SYSTEM = ("You are a coding assistant. The user's project is reachable only through your tools. "
                 "When you need several independent reads, request them in parallel in one message. Be brief.")

tools_path = os.path.join(WORK, "tools.json")
with open(tools_path, "w") as f:
    json.dump(CALLER_TOOLS, f)
mcp_path = os.path.join(WORK, "mcp.json")
with open(mcp_path, "w") as f:
    json.dump({"mcpServers": {"bridge": {"type": "stdio", "command": "python3",
                                         "args": [os.path.join(HERE, "fake_mcp.py")],
                                         "env": {"FAKE_MCP_LOG": os.path.join(WORK, "mcp.log"),
                                                 "FAKE_MCP_TOOLS": tools_path,
                                                 "FAKE_MCP_DELAY": "3600"}}}}, f)


def say(tag, text):
    print(f"[{time.time()-T0:6.1f}s] {tag:<10} {text}", flush=True)


def session_file():
    root = os.path.expanduser("~/.claude/projects")
    for d in os.listdir(root):
        p = os.path.join(root, d, SID + ".jsonl")
        if os.path.exists(p):
            return p
    return None


def describe(e):
    content = e["message"]["content"]
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    out = []
    for b in content:
        if b["type"] == "text":
            out.append("text: " + b["text"][:80].replace("\n", " "))
        elif b["type"] == "tool_use":
            out.append(f"tool_use {b['name'][len(PREFIX):]} id=..{b['id'][-6:]} {json.dumps(b['input'])[:50]}")
        elif b["type"] == "tool_result":
            out.append(f"tool_result id=..{b['tool_use_id'][-6:]} {str(b.get('content'))[:50]!r}")
        else:
            out.append(b["type"])
    return "; ".join(out)


def dump_conversation():
    """Print only the user/assistant entries, i.e. what the model will see."""
    for line in open(session_file()):
        e = json.loads(line)
        if e.get("type") in ("user", "assistant"):
            print(f"{'':20}{e['type']:<9} {describe(e)}")


def cut_session_after(tool_ids):
    """Drop everything after the last assistant entry holding one of tool_ids."""
    p = session_file()
    lines = open(p).readlines()
    keep = None
    for i, line in enumerate(lines):
        e = json.loads(line)
        if e.get("type") == "assistant" and isinstance(e["message"]["content"], list):
            if any(b.get("type") == "tool_use" and b.get("id") in tool_ids for b in e["message"]["content"]):
                keep = i
    dropped = [json.loads(x) for x in lines[keep + 1:]]
    with open(p, "w") as f:
        f.writelines(lines[:keep + 1])
    say("CUT", f"removed {len(dropped)} trailing entries: " +
        ", ".join(f"{d['type']}({describe(d)})" if d.get("type") in ("user", "assistant") else d.get("type")
                  for d in dropped))


def turn(name, message, resume):
    say("TURN", f"=== {name} ({'resume' if resume else 'new session'})")
    cmd = ["claude", "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
           "--include-partial-messages",
           "--model", "haiku", "--tools", "", "--mcp-config", mcp_path, "--strict-mcp-config",
           "--allowedTools", ",".join(PREFIX + t["function"]["name"] for t in CALLER_TOOLS),
           "--system-prompt", CALLER_SYSTEM, "--setting-sources", "", "--disable-slash-commands"]
    cmd += ["--resume", SID] if resume else ["--session-id", SID]
    proc = subprocess.Popen(cmd, cwd=CWD, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=open(os.path.join(WORK, "stderr.txt"), "a"), text=True)
    assert proc.stdin and proc.stdout
    proc.stdin.write(json.dumps({"type": "user", "message": message}) + "\n")
    proc.stdin.flush()
    say("SEND", json.dumps(message)[:160])
    calls, text = [], ""
    for line in proc.stdout:
        e = json.loads(line)
        t = e.get("type")
        if t == "stream_event":
            ev = e["event"]
            if ev.get("type") == "content_block_delta" and ev["delta"].get("type") == "text_delta":
                text += ev["delta"]["text"]
            if ev.get("type") == "message_stop" and calls:
                break  # assistant message complete, all its tool calls known
        elif t == "assistant":
            for b in e["message"]["content"]:
                if b["type"] == "tool_use" and b["name"].startswith(PREFIX):
                    calls.append({"id": b["id"], "name": b["name"][len(PREFIX):], "arguments": b["input"]})
                    say("TOOL_USE", f"{calls[-1]['name']} id=..{b['id'][-6:]} {json.dumps(b['input'])}")
        elif t == "result":
            say("RESULT", f"{e.get('subtype')} api_ms={e.get('duration_api_ms')} "
                f"cache_read={e.get('usage', {}).get('cache_read_input_tokens')} "
                f"cache_write={e.get('usage', {}).get('cache_creation_input_tokens')}")
            break
    if text:
        say("TEXT", text[:300].replace("\n", " "))
    if calls:
        say("INTERCEPT", "-> caller gets tool_calls: " + json.dumps(calls))
    proc.stdin.close()
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    say("SESSION", "conversation now stored:")
    dump_conversation()
    return calls


def patch_results(calls, outputs):
    """Overwrite the placeholder tool_result entries claude wrote on exit with the real outputs."""
    p = session_file()
    assert p
    by_id = {c["id"]: outputs[c["name"]] for c in calls}
    out, patched = [], 0
    for line in open(p):
        e = json.loads(line)
        if e.get("type") == "user" and isinstance(e["message"]["content"], list):
            for b in e["message"]["content"]:
                if b.get("type") == "tool_result" and b.get("tool_use_id") in by_id:
                    b["content"] = by_id[b["tool_use_id"]]
                    b.pop("is_error", None)
                    e["toolUseResult"] = b["content"]
                    e.pop("toolDenialKind", None)
                    patched += 1
            line = json.dumps(e) + "\n"
        out.append(line)
    with open(p, "w") as f:
        f.writelines(out)
    say("PATCH", f"wrote {patched}/{len(calls)} real tool results into the session file")


def results(calls, outputs):
    patch_results(calls, outputs)
    return {"role": "user", "content": "Continue."}


FILES = {"read": "export function add(a: number, b: number) {\n  return a + b;\n}\n",
         "edit": "Edit applied successfully.",
         "bash": " Test Files  1 passed (1)\n      Tests  2 passed (2)\n"}

c = turn("1: user request", {"role": "user", "content": "Read src/calc.ts and tell me the function name."}, False)
if c:
    c = turn("2: tool result", results(c, FILES), True)
c = turn("3: follow-up", {"role": "user", "content": "Change it to subtract instead and rename it to sub. "
                          "Then run npm test."}, True)
for n in range(4, 8):
    if not c:
        break
    c = turn(f"{n}: tool result", results(c, FILES), True)
