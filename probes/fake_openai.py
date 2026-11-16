#!/usr/bin/env python3
"""Scripted fake OpenAI-compatible chat completions server.

Logs every request (tool names offered, message roles) to $FAKE_OAI_LOG and
replies from a script, chosen by how many tool results the request already holds:
  0 tool results -> call the client tool named $FAKE_OAI_FIRST  (default "bash")
  1 tool result  -> call the client tool named $FAKE_OAI_SECOND (default "bridge_read")
  2+             -> plain text answer
Supports both stream and non-stream responses.
"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = os.environ.get("FAKE_OAI_LOG", "/tmp/fake_oai.log")
FIRST = os.environ.get("FAKE_OAI_FIRST", "bash")
SECOND = os.environ.get("FAKE_OAI_SECOND", "bridge_read")

ARGS = {
    "bash": {"command": "ls", "description": "List files"},
    "read": {"filePath": "src/calc.ts"},
}


def args_for(name):
    for key, value in ARGS.items():
        if name.endswith(key):
            return value
    return {}


def log(entry):
    entry["t"] = round(time.time(), 3)
    with open(LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"object": "list", "data": [{"id": "m", "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        msgs = req.get("messages", [])
        tools = [t.get("function", {}).get("name") for t in req.get("tools", []) or []]
        n_results = sum(1 for m in msgs if m.get("role") == "tool")
        log({
            "path": self.path,
            "stream": req.get("stream"),
            "tools": tools,
            "roles": [m.get("role") for m in msgs],
            "system_head": next((str(m.get("content"))[:200] for m in msgs if m.get("role") == "system"), None),
            "last": json.dumps(msgs[-1])[:400] if msgs else None,
        })
        if n_results == 0:
            reply = ("tool", FIRST)
        elif n_results == 1:
            reply = ("tool", SECOND)
        else:
            reply = ("text", "Done. Summary written by the fake model.")
        if req.get("stream"):
            self.stream(reply)
        else:
            self.plain(reply)

    def plain(self, reply):
        kind, value = reply
        if kind == "tool":
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call_{int(time.time()*1000)}", "type": "function",
                "function": {"name": value, "arguments": json.dumps(args_for(value))}}]}
            finish = "tool_calls"
        else:
            msg = {"role": "assistant", "content": value}
            finish = "stop"
        body = json.dumps({"id": "x", "object": "chat.completion", "created": int(time.time()), "model": "m",
                           "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                           "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def stream(self, reply):
        kind, value = reply
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def chunk(delta, finish=None, usage=None):
            c = {"id": "x", "object": "chat.completion.chunk", "created": int(time.time()), "model": "m",
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if usage:
                c["usage"] = usage
            self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
            self.wfile.flush()

        chunk({"role": "assistant", "content": ""})
        if kind == "tool":
            chunk({"tool_calls": [{"index": 0, "id": f"call_{int(time.time()*1000)}", "type": "function",
                                   "function": {"name": value, "arguments": json.dumps(args_for(value))}}]})
            chunk({}, "tool_calls", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        else:
            chunk({"content": value})
            chunk({}, "stop", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18080
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
