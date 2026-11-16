#!/usr/bin/env python3
"""Minimal stdio MCP server for probing agents.

Exposes opencode-style client tools (read, bash, edit). Every call is logged
as one JSON line to $FAKE_MCP_LOG and answered with a canned result after
$FAKE_MCP_DELAY seconds (to test how long an agent will wait on a call).
"""
import json
import os
import sys
import time

LOG = os.environ.get("FAKE_MCP_LOG", "/tmp/fake_mcp.log")
DELAY = float(os.environ.get("FAKE_MCP_DELAY", "0"))

TOOLS = [
    {
        "name": "read",
        "description": "Read a file in the user's project. This is the only way to see project files.",
        "inputSchema": {
            "type": "object",
            "properties": {"filePath": {"type": "string"}},
            "required": ["filePath"],
        },
    },
    {
        "name": "bash",
        "description": "Run a shell command in the user's project. This is the only way to run commands.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "edit",
        "description": "Replace exact text in a project file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filePath": {"type": "string"},
                "oldString": {"type": "string"},
                "newString": {"type": "string"},
            },
            "required": ["filePath", "oldString", "newString"],
        },
    },
]

if os.environ.get("FAKE_MCP_TOOLS"):
    # OpenAI-style tool list from the caller: [{"type":"function","function":{name,description,parameters}}]
    with open(os.environ["FAKE_MCP_TOOLS"]) as f:
        TOOLS = [
            {
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "inputSchema": t["function"].get("parameters", {"type": "object"}),
            }
            for t in json.load(f)
        ]

CANNED = {
    "read": "export function add(a: number, b: number) {\n  return a + b;\n}\n",
    "bash": "README.md\npackage.json\nsrc\ntests\n",
    "edit": "Edit applied successfully.",
}


def log(event):
    event["t"] = round(time.time(), 3)
    with open(LOG, "a") as f:
        f.write(json.dumps(event) + "\n")


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def main():
    log({"event": "start", "pid": os.getpid()})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = req.get("method")
        rid = req.get("id")
        if rid is None:  # notification
            continue
        if method == "initialize":
            version = req.get("params", {}).get("protocolVersion", "2025-06-18")
            send({"jsonrpc": "2.0", "id": rid, "result": {
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "bridge", "version": "0.1"},
            }})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            params = req.get("params", {})
            name = params.get("name")
            log({"event": "call", "name": name, "arguments": params.get("arguments")})
            if DELAY:
                time.sleep(DELAY)
            send({"jsonrpc": "2.0", "id": rid, "result": {
                "content": [{"type": "text", "text": CANNED.get(name, "ok")}],
            }})
            log({"event": "answered", "name": name})
        elif method == "ping":
            send({"jsonrpc": "2.0", "id": rid, "result": {}})
        else:
            send({"jsonrpc": "2.0", "id": rid,
                  "error": {"code": -32601, "message": f"unknown method {method}"}})


if __name__ == "__main__":
    main()
