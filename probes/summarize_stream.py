#!/usr/bin/env python3
"""Summarise a claude stream-json or opencode json event file: tools offered, tool calls, final text."""
import json
import sys

for raw in open(sys.argv[1]):
    try:
        e = json.loads(raw)
    except json.JSONDecodeError:
        print("RAW", raw.strip()[:200])
        continue
    t = e.get("type")
    if t == "system" and e.get("subtype") == "init":
        print("INIT tools =", e.get("tools"))
        print("INIT mcp   =", e.get("mcp_servers"))
    elif t == "assistant":
        for block in e["message"]["content"]:
            if block["type"] == "tool_use":
                print("TOOL_USE", block["name"], json.dumps(block["input"]))
            elif block["type"] == "text":
                print("TEXT", block["text"][:600].replace("\n", " | "))
    elif t == "user":
        for block in e["message"].get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                c = block.get("content")
                print("TOOL_RESULT", str(c)[:150].replace("\n", " | "))
    elif t == "result":
        print("RESULT", e.get("subtype"), "turns=", e.get("num_turns"),
              "ms=", e.get("duration_ms"), "denials=", e.get("permission_denials"))
    elif t in ("rate_limit_event", "stream_event"):
        continue
    else:
        print(t, json.dumps(e)[:400])
