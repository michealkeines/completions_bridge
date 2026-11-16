#!/usr/bin/env python3
"""Session-file lab: build claude session files ourselves, resume with a static prompt.

Tests the "write everything into the session file, resume with 'Continue.'" design:
  T0  flags: --autocompact and --system-prompt-file accepted
  T1  history built from scratch (incl. a tool round) is understood
  T2  leak: does the model treat "Continue." as the user's latest message?
  T3  full tool loop: call -> stop -> patch placeholder -> continue -> new user msg -> call
  T4  fork by copy: two copies of one checkpoint, different results, independent
  T5  system prompt and tool list changed on resume
  T6  history uses a tool not offered now (and no tools at all)
  T7  non-"toolu_" call ids in history

usage: session_lab.py WORKDIR [TEST ...]
"""
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.realpath(sys.argv[1])
ONLY = set(sys.argv[2:])
CWD = os.path.join(WORK, "empty")
os.makedirs(CWD, exist_ok=True)
PROJ = os.path.expanduser("~/.claude/projects/" + re.sub(r"[^a-zA-Z0-9]", "-", CWD))
os.makedirs(PROJ, exist_ok=True)
VERSION = subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.split()[0]
MODEL_ID = "claude-haiku-5-5"
PREFIX = "mcp__bridge__"
TRIGGER = "Continue."

TOOLS = {
    "read": {"description": "Read a file in the user's project.",
             "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}}, "required": ["filePath"]}},
    "edit": {"description": "Replace exact text in a project file.",
             "parameters": {"type": "object", "properties": {"filePath": {"type": "string"},
                            "oldString": {"type": "string"}, "newString": {"type": "string"}},
                            "required": ["filePath", "oldString", "newString"]}},
    "grep_files": {"description": "Search project files for a pattern.",
                   "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
}
SYSTEM = "You are a coding assistant. The user's project is reachable only through your tools. Be brief."


# ---------------------------------------------------------------- session file writing

def path(sid):
    return os.path.join(PROJ, sid + ".jsonl")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())


def base_entry(sid, parent, kind):
    return {"parentUuid": parent, "isSidechain": False, "type": kind, "uuid": str(uuid.uuid4()),
            "timestamp": now(), "userType": "external", "entrypoint": "sdk-cli", "cwd": CWD,
            "sessionId": sid, "version": VERSION}


def write_session(sid, messages):
    """messages: list of ("user", text) | ("assistant", text, [calls]) | ("results", [(id, text)])
    calls: (id, name, args)."""
    lines, parent, last_assistant_uuid = [], None, None
    for m in messages:
        if m[0] == "user":
            e = base_entry(sid, parent, "user")
            e["message"] = {"role": "user", "content": m[1]}
        elif m[0] == "assistant":
            e = base_entry(sid, parent, "assistant")
            content = ([{"type": "text", "text": m[1]}] if m[1] else []) + [
                {"type": "tool_use", "id": cid, "name": PREFIX + name, "input": args} for cid, name, args in m[2]]
            e["message"] = {"id": "msg_" + uuid.uuid4().hex[:24], "type": "message", "role": "assistant",
                            "model": MODEL_ID, "content": content,
                            "stop_reason": "tool_use" if m[2] else "end_turn", "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0}}
            last_assistant_uuid = e["uuid"]
        else:
            e = base_entry(sid, parent, "user")
            e["message"] = {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": cid, "content": text} for cid, text in m[1]]}
            e["toolUseResult"] = m[1][0][1]
            e["sourceToolAssistantUUID"] = last_assistant_uuid
        parent = e["uuid"]
        lines.append(json.dumps(e))
    with open(path(sid), "w") as f:
        f.write("\n".join(lines) + "\n")


def append_messages(sid, messages):
    """Append our messages after the last conversation entry of an existing file."""
    entries = [json.loads(l) for l in open(path(sid))]
    parent = next((e["uuid"] for e in reversed(entries) if e.get("type") in ("user", "assistant")), None)
    tmp = str(uuid.uuid4())
    write_session(tmp, messages)
    new = [json.loads(l) for l in open(path(tmp))]
    os.remove(path(tmp))
    new[0]["parentUuid"] = parent
    with open(path(sid), "a") as f:
        for e in new:
            e["sessionId"] = sid
            f.write(json.dumps(e) + "\n")


def patch_results(sid, results):
    """Overwrite placeholder tool_results claude wrote when stopped."""
    out, patched = [], 0
    for line in open(path(sid)):
        e = json.loads(line)
        if e.get("type") == "user" and isinstance(e["message"]["content"], list):
            for b in e["message"]["content"]:
                if b.get("type") == "tool_result" and b.get("tool_use_id") in results:
                    b["content"] = results[b["tool_use_id"]]
                    b.pop("is_error", None)
                    e["toolUseResult"] = b["content"]
                    e.pop("toolDenialKind", None)
                    patched += 1
        out.append(json.dumps(e) + "\n")
    open(path(sid), "w").writelines(out)
    return patched


def fork(src):
    dst = str(uuid.uuid4())
    with open(path(dst), "w") as f:
        for line in open(path(src)):
            e = json.loads(line)
            if "sessionId" in e:
                e["sessionId"] = dst
            f.write(json.dumps(e) + "\n")
    return dst


def strip_fillers(sid):
    """Remove claude's synthetic filler replies and our trigger messages, relinking the parent chain."""
    entries = [json.loads(l) for l in open(path(sid))]
    removed = {}
    for e in entries:
        if e.get("type") == "assistant" and e["message"].get("model") == "<synthetic>":
            removed[e["uuid"]] = e.get("parentUuid")
        elif e.get("type") == "user" and e["message"].get("content") in (TRIGGER, [{"type": "text", "text": TRIGGER}]):
            removed[e["uuid"]] = e.get("parentUuid")

    def resolve(u):
        while u in removed:
            u = removed[u]
        return u
    with open(path(sid), "w") as f:
        for e in entries:
            if e.get("uuid") in removed:
                continue
            if e.get("parentUuid") in removed:
                e["parentUuid"] = resolve(e["parentUuid"])
            f.write(json.dumps(e) + "\n")
    return len(removed)


def file_hash(sid):
    return hashlib.sha256(open(path(sid), "rb").read()).hexdigest()[:12]


def conversation(sid):
    rows = []
    for line in open(path(sid)):
        e = json.loads(line)
        if e.get("type") not in ("user", "assistant"):
            continue
        c = e["message"]["content"]
        c = [{"type": "text", "text": c}] if isinstance(c, str) else c
        for b in c:
            t = b["type"]
            if t == "text":
                d = "text: " + b["text"][:90].replace("\n", " ")
            elif t == "tool_use":
                d = f"tool_use {b['name'][len(PREFIX):]} {json.dumps(b['input'])[:50]}"
            elif t == "tool_result":
                d = f"tool_result {str(b.get('content'))[:50]!r}"
            else:
                d = t
            rows.append(f"{e['type']:<9} {d}")
    return rows


def interrupted(sid):
    return "interrupted" in open(path(sid)).read()


# ---------------------------------------------------------------- running claude

def run(sid, tools=("read", "edit"), system=SYSTEM, prompt=TRIGGER, new=False, extra=(), use_mcp=True):
    tdefs = [{"type": "function", "function": {"name": n, **TOOLS[n]}} for n in tools]
    tpath = os.path.join(WORK, f"tools-{sid}.json")
    json.dump(tdefs, open(tpath, "w"))
    mpath = os.path.join(WORK, f"mcp-{sid}.json")
    json.dump({"mcpServers": {"bridge": {"type": "stdio", "command": "python3",
                                         "args": [os.path.join(HERE, "fake_mcp.py")],
                                         "env": {"FAKE_MCP_LOG": os.path.join(WORK, "mcp.log"),
                                                 "FAKE_MCP_TOOLS": tpath, "FAKE_MCP_DELAY": "3600"}}}},
              open(mpath, "w"))
    cmd = ["claude", "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
           "--include-partial-messages", "--model", "haiku", "--tools", "",
           "--setting-sources", "", "--disable-slash-commands", "--system-prompt", system]
    if use_mcp and tools:
        cmd += ["--mcp-config", mpath, "--strict-mcp-config", "--allowedTools", ",".join(PREFIX + t for t in tools)]
    else:
        cmd += ["--strict-mcp-config"]
    cmd += ["--session-id", sid] if new else ["--resume", sid]
    cmd += list(extra)
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=CWD, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    assert proc.stdin and proc.stdout and proc.stderr
    proc.stdin.write(json.dumps({"type": "user", "message": {"role": "user", "content": prompt}}) + "\n")
    proc.stdin.flush()
    out = {"text": "", "calls": [], "tools_seen": None, "error": None, "result": None}
    for line in proc.stdout:
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = e.get("type")
        if t == "system" and e.get("subtype") == "init":
            out["tools_seen"] = [x[len(PREFIX):] if x.startswith(PREFIX) else x for x in e.get("tools", [])]
        elif t == "stream_event":
            ev = e["event"]
            if ev.get("type") == "content_block_delta" and ev["delta"].get("type") == "text_delta":
                out["text"] += ev["delta"]["text"]
            if ev.get("type") == "message_stop" and out["calls"]:
                break
        elif t == "assistant":
            for b in e["message"]["content"]:
                if b["type"] == "tool_use":
                    out["calls"].append({"id": b["id"], "name": b["name"].replace(PREFIX, ""), "args": b["input"]})
        elif t == "result":
            out["result"] = e.get("subtype")
            if e.get("is_error"):
                out["error"] = e.get("result")
            break
    proc.stdin.close()
    proc.send_signal(signal.SIGTERM)
    try:
        _, err = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        _, err = proc.communicate()
    if not out["result"] and not out["calls"]:
        out["error"] = (out["error"] or "") + " | exit=%s stderr=%s" % (proc.returncode, err.strip()[-300:])
    out["secs"] = round(time.time() - t0, 1)
    return out


def show(label, r):
    print(f"  [{label}] {r['secs']}s tools={r['tools_seen']}")
    if r["text"]:
        print(f"      text : {r['text'][:400]!r}")
    for c in r["calls"]:
        print(f"      call : {c['name']} {json.dumps(c['args'])[:120]}")
    if r["error"]:
        print(f"      ERROR: {r['error'][:400]}")


def test(name):
    return not ONLY or name in ONLY


# ---------------------------------------------------------------- tests

if test("T0"):
    print("\n=== T0 flags")
    sid = str(uuid.uuid4())
    sp = os.path.join(WORK, "sys.txt")
    open(sp, "w").write("Answer with exactly one word.")
    r = run(sid, tools=(), new=True, prompt="Say hello.", system=SYSTEM,
            extra=["--autocompact", "1000000"], use_mcp=False)
    show("--autocompact 1000000", r)
    sid = str(uuid.uuid4())
    cmd_ok = subprocess.run(["claude", "--help"], capture_output=True, text=True).stdout
    print("  --system-prompt-file in help:", "--system-prompt-file" in cmd_ok)

HISTORY = [
    ("user", "My codename is PELICAN. Read notes.txt please."),
    ("assistant", "Reading it.", [("toolu_01Bld000000000000000001", "read", {"filePath": "notes.txt"})]),
    ("results", [("toolu_01Bld000000000000000001", "Meeting moved to Tuesday 3pm in room B.")]),
    ("assistant", "Noted: the meeting is on Tuesday at 3pm in room B.", []),
]

if test("T1"):
    print("\n=== T1 history built from scratch, static 'Continue.'")
    sid = str(uuid.uuid4())
    write_session(sid, HISTORY + [("user", "What is my codename and where is the meeting? One line.")])
    r = run(sid)
    show("static", r)
    print("      interrupted entries:", interrupted(sid))

if test("T2"):
    print("\n=== T2 leak: is 'Continue.' seen as the user's latest message?")
    q = "Repeat my most recent message to you word for word, and nothing else."
    hist = [("user", "Hi, remember the number 42."), ("assistant", "Got it, 42.", [])]
    for variant in ("static", "prompt"):
        for i in range(2):
            sid = str(uuid.uuid4())
            if variant == "static":
                write_session(sid, hist + [("user", q)])
                r = run(sid, tools=(), use_mcp=False)
            else:
                write_session(sid, hist)
                r = run(sid, tools=(), use_mcp=False, prompt=q)
            show(f"{variant} #{i+1}", r)
    sid = str(uuid.uuid4())
    write_session(sid, hist + [("user", "Reply with only the word OK.")])
    show("static, 'Reply with only OK'", run(sid, tools=(), use_mcp=False))
    sid = str(uuid.uuid4())
    write_session(sid, hist + [("user", "How many messages have I sent you so far? Just the number.")])
    show("static, 'how many messages' (truth: 2)", run(sid, tools=(), use_mcp=False))

if test("T3") or test("T4"):
    print("\n=== T3 full tool loop, everything written to file, always 'Continue.'")
    sid = str(uuid.uuid4())
    write_session(sid, [("user", "Read src/calc.ts and tell me the function name.")])
    r1 = run(sid)
    show("turn 1", r1)
    checkpoint = fork(sid)  # frozen AWAITING checkpoint (with placeholder)
    ok = r1["calls"] and patch_results(sid, {c["id"]: "export function add(a: number, b: number) {\n  return a + b;\n}\n"
                                             for c in r1["calls"]})
    print("      patched:", ok)
    r2 = run(sid)
    show("turn 2 (after patch)", r2)
    append_messages(sid, [("user", "Rename it to sub and make it subtract.")])
    r3 = run(sid)
    show("turn 3 (user msg appended)", r3)
    if r3["calls"]:
        patch_results(sid, {c["id"]: "Edit applied successfully." for c in r3["calls"]})
        show("turn 4 (after patch)", run(sid))
    print("      interrupted entries:", interrupted(sid))
    print("      final conversation:")
    for row in conversation(sid):
        print("        " + row)

    if test("T4") and r1["calls"]:
        print("\n=== T4 fork by copy: two forks of the AWAITING checkpoint")
        before = file_hash(checkpoint)
        for label, body in (("add", "export function add(a: number, b: number) { return a + b; }"),
                            ("multiply", "export function multiply(a: number, b: number) { return a * b; }")):
            f = fork(checkpoint)
            patch_results(f, {c["id"]: body for c in r1["calls"]})
            show(f"fork expecting '{label}'", run(f))
            print("      interrupted entries:", interrupted(f))
        print("      checkpoint unchanged:", before == file_hash(checkpoint))

if test("T5"):
    print("\n=== T5 system prompt and tools changed on resume")
    sid = str(uuid.uuid4())
    write_session(sid, HISTORY + [("user", "List the names of the tools you can use right now, then say hello.")])
    show("orig system, tools read+edit", run(fork(sid)))
    show("French system, tools read+grep_files",
         run(fork(sid), tools=("read", "grep_files"),
             system="You are a coding assistant. Always answer in French, and end every reply with '-- BOT'."))

if test("T6"):
    print("\n=== T6 history uses a tool that is not offered now")
    sid = str(uuid.uuid4())
    write_session(sid, [
        ("user", "Look up the deploy status."),
        ("assistant", "", [("toolu_01Leg000000000000000001", "legacy_tool", {"q": "deploy"})]),
        ("results", [("toolu_01Leg000000000000000001", "Deploy #812 succeeded at 10:42.")]),
        ("assistant", "Deploy #812 succeeded at 10:42.", []),
        ("user", "Which deploy number was it? Just the number."),
    ])
    show("tools read only", run(fork(sid), tools=("read",)))
    show("no tools at all", run(fork(sid), tools=(), use_mcp=False))

if test("T7"):
    print("\n=== T7 call ids not in toolu_ format")
    for cid in ("call_abc123XYZ", "call:abc/123"):
        sid = str(uuid.uuid4())
        h = [("user", "My codename is PELICAN. Read notes.txt please."),
             ("assistant", "Reading it.", [(cid, "read", {"filePath": "notes.txt"})]),
             ("results", [(cid, "Meeting moved to Tuesday 3pm in room B.")]),
             ("assistant", "Noted.", []),
             ("user", "Where is the meeting? One line.")]
        write_session(sid, h)
        show(f"id={cid}", run(sid))

if test("T8"):
    print("\n=== T8 hybrid: user message as prompt, tool results via file + 'Continue.', fillers stripped")
    hist = [("user", "Hi, remember the number 42."), ("assistant", "Got it, 42.", [])]
    for q in ("Reply with only the word OK.", "How many messages have I sent you so far, including this one? Just the number."):
        sid = str(uuid.uuid4())
        write_session(sid, hist)
        show(f"prompt={q[:30]!r}", run(sid, tools=(), use_mcp=False, prompt=q))

    for strip in (False, True):
        print(f"  --- tool loop, strip fillers = {strip}")
        sid = str(uuid.uuid4())
        r = run(sid, new=True, prompt="Read src/calc.ts and tell me the function name.")
        show("turn 1 (prompt)", r)
        for n, body in ((2, "export function add(a: number, b: number) {\n  return a + b;\n}\n"),):
            if not r["calls"]:
                break
            if strip:
                sid2 = fork(sid); strip_fillers(sid2); sid = sid2
            patch_results(sid, {c["id"]: body for c in r["calls"]})
            r = run(sid)
            show(f"turn {n} (results, Continue.)", r)
        if strip:
            sid2 = fork(sid); n_removed = strip_fillers(sid2); sid = sid2
            print("      fillers removed before last turn:", n_removed)
        q = ("Answer in one line each: (1) How many messages have I typed to you? "
             "(2) Did you ever give an empty, declined or wrong reply in this conversation? "
             "(3) Did I ever say the word 'Continue'?")
        show("turn 3 (prompt, meta questions)", run(sid, prompt=q))
        print("      visible conversation:")
        for row in conversation(sid):
            print("        " + row)
