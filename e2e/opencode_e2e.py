#!/usr/bin/env python3
"""Real-world end-to-end test: real opencode → completions-bridge → real claude.

Starts the bridge binary with a fresh workdir, configures an isolated opencode with a
custom OpenAI-compatible provider pointing at the bridge, copies a fixture project with
a failing test, and runs a sequence of real coding tasks through `opencode run`.

Checks after each step: opencode succeeded, the project state is right (tests pass etc.),
the answer makes sense, no internals leaked, and (from the bridge's request log) how often
checkpoints were reused.

usage: opencode_e2e.py [--bin PATH] [--model claude-haiku] [--keep] [--only S1,S2,...]
"""
import argparse
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "e2e", "fixture")
import re
EVASIVE = re.compile(r"(is above|answered above|already (answered|told)|nothing (else )?is pending|no response requested|what would you like me to do next)", re.I)
LEAK_MARKERS = ["mcp__", "No response requested", "Connection closed", "[Tool call interrupted", "toolu_b_",
                "/claude/snapshots", "fake backend",
                "Let me look at the tool results", "bridgeGlue", "runtime glue", "automatic system note"]

results = []


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Bridge:
    def __init__(self, binary, root, port, token):
        self.binary, self.root, self.port, self.token = binary, root, port, token
        self.workdir = os.path.join(root, "bridge")
        self.config = os.path.join(root, "bridge-config.yaml")
        self.logfile = os.path.join(root, "bridge.log")
        self.proc = None
        with open(self.config, "w") as f:
            f.write(f"""server:
  listen: 127.0.0.1:{port}
  token: {token}
  workdir: {self.workdir}
  header_wait_ms: 3000
backends:
  claude:
    kind: claude-cli
    default_model: haiku
    side_request_model: haiku
    max_concurrency: 4
""")

    def start(self):
        env = dict(os.environ, BRIDGE_LOG="info,bridge_engine=debug,bridge_claude=debug")
        self.proc = subprocess.Popen([self.binary, "serve", "--config", self.config], cwd=self.root,
                                     stdout=open(self.logfile, "a"), stderr=subprocess.STDOUT, env=env)
        deadline = time.time() + 120  # self-test may run on first start
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"bridge exited early; see {self.logfile}")
            try:
                self.get("/v1/models")
                return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("bridge did not come up")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(15)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def get(self, path):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}",
                                     headers={"Authorization": f"Bearer {self.token}"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())

    def request_log(self, since_id=0):
        db = os.path.join(self.workdir, "state", "bridge.db")
        if not os.path.exists(db):
            return []
        con = sqlite3.connect(db)
        con.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in con.execute("SELECT * FROM request_log WHERE rowid > ? ORDER BY rowid",
                                                  (since_id,))]
        finally:
            con.close()

    def max_rowid(self):
        db = os.path.join(self.workdir, "state", "bridge.db")
        if not os.path.exists(db):
            return 0
        con = sqlite3.connect(db)
        try:
            return con.execute("SELECT COALESCE(MAX(rowid), 0) FROM request_log").fetchone()[0]
        finally:
            con.close()


class OpenCode:
    def __init__(self, root, port, token, model_id):
        self.root = os.path.join(root, "opencode")
        self.model = f"bridge/{model_id}"
        for d in ("data", "config", "state", "cache"):
            os.makedirs(os.path.join(self.root, d), exist_ok=True)
        self.config = {
            "$schema": "https://opencode.ai/config.json",
            "autoupdate": False,
            "share": "disabled",
            "provider": {
                "bridge": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Completions Bridge",
                    "options": {"baseURL": f"http://127.0.0.1:{port}/v1", "apiKey": token},
                    "models": {model_id: {"name": model_id, "tool_call": True,
                                          "limit": {"context": 180000, "output": 32000}}},
                }
            },
            "model": self.model,
        }

    def env(self):
        return dict(os.environ,
                    XDG_DATA_HOME=os.path.join(self.root, "data"),
                    XDG_CONFIG_HOME=os.path.join(self.root, "config"),
                    XDG_STATE_HOME=os.path.join(self.root, "state"),
                    XDG_CACHE_HOME=os.path.join(self.root, "cache"),
                    OPENCODE_CONFIG_CONTENT=json.dumps(self.config),
                    OPENCODE_DISABLE_PROJECT_CONFIG="1",
                    OPENCODE_DISABLE_CLAUDE_CODE="1",
                    OPENCODE_DISABLE_AUTOUPDATE="1")

    def run(self, cwd, prompt, session=None, fork=False, timeout=900):
        cmd = ["opencode", "run", "--standalone", "--format", "json", "--model", self.model, "--auto"]
        if session:
            cmd += ["--session", session]
        if fork:
            cmd += ["--fork"]
        cmd.append(prompt)
        t0 = time.time()
        env = self.env()
        env["PWD"] = cwd  # opencode trusts $PWD over the real cwd
        p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
        out = {"exit": p.returncode, "secs": round(time.time() - t0, 1), "session": None, "text": "",
               "tools": [], "errors": [], "raw": p.stdout, "stderr": p.stderr[-2000:]}
        for line in p.stdout.splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            out["session"] = out["session"] or e.get("sessionID")
            part = e.get("part") or {}
            if part.get("type") == "tool":
                st = part.get("state", {})
                out["tools"].append((part.get("tool"), st.get("status"), json.dumps(st.get("input"))[:120]))
            elif part.get("type") == "text":
                out["text"] += part.get("text") or ""
            elif e.get("type") == "error":
                out["errors"].append(json.dumps(e.get("error"))[:400])
        return out


def record(name, ok, detail):
    results.append((name, ok, detail))
    log(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")


def tests_pass(project):
    p = subprocess.run(["node", "--test"], cwd=project, capture_output=True, text=True, timeout=120)
    return p.returncode == 0


def leaks(out):
    blob = out["raw"]
    return [m for m in LEAK_MARKERS if m in blob]


def summarize_log(bridge, since):
    rows = bridge.request_log()
    rows = rows[since:] if since < len(rows) else []
    keys = rows[0].keys() if rows else []
    hit_key = next((k for k in keys if "hit" in k), None)
    hits = sum(1 for r in rows if hit_key and (r.get(hit_key) or 0) > 0)
    errors = [r for r in rows if any(k for k in keys if "error" in k and r.get(k))]
    return {"requests": len(rows), "checkpoint_hits": hits, "errors": len(errors), "hit_key": hit_key}


def show(out):
    log(f"  opencode exit={out['exit']} {out['secs']}s session={out['session']}")
    for t in out["tools"]:
        log(f"    tool {t[0]:<10} {t[1]:<10} {t[2]}")
    if out["text"]:
        log("    text: " + out["text"][:500].replace("\n", " | "))
    for e in out["errors"]:
        log("    ERROR: " + e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "target", "release", "completions-bridge"))
    ap.add_argument("--model", default="claude-haiku")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    args.bin = os.path.abspath(args.bin)
    only = set(filter(None, args.only.split(",")))
    want = lambda s: not only or s in only  # noqa: E731

    root = tempfile.mkdtemp(prefix="cb_e2e_")
    project = os.path.join(root, "project")
    shutil.copytree(FIXTURE, project)
    subprocess.run(["git", "init", "-q"], cwd=project)
    port, token = free_port(), "e2e-" + os.urandom(8).hex()
    bridge = Bridge(args.bin, root, port, token)
    oc = OpenCode(root, port, token, f"bridge/{args.model}")
    log(f"workspace {root}, bridge on :{port}")

    try:
        bridge.start()
        models = [m["id"] for m in bridge.get("/v1/models")["data"]]
        record("models", f"bridge/{args.model}" in models, f"listed: {models}")
        status = bridge.get("/admin/status")
        log("  status: " + json.dumps(status)[:600])

        session = None
        mark = 0

        if want("S1"):
            log("=== S1 fix a failing test (bash, read, edit, rerun)")
            out = oc.run(project, "The test suite is failing. Run `npm test`, find the bug, fix it in the source "
                                  "(not in the tests), then run `npm test` again to confirm everything passes.")
            show(out)
            session = out["session"]
            used = {t[0] for t in out["tools"]}
            record("S1 opencode ok", out["exit"] == 0 and not out["errors"], f"exit={out['exit']} errors={out['errors'][:1]}")
            record("S1 tests pass after", tests_pass(project), "node --test in project")
            record("S1 used shell+edit", ({"bash", "shell"} & used) and ({"edit", "write", "patch", "apply_patch"} & used),
                   f"tools used: {sorted(used)}")
            record("S1 no leaks", not leaks(out), f"markers found: {leaks(out)}")
            s = summarize_log(bridge, mark)
            record("S1 checkpoint reuse", s["requests"] >= 2 and s["checkpoint_hits"] >= 1, json.dumps(s))
            mark += s["requests"]

        if want("S2") and session:
            log("=== S2 follow-up in the same session (new feature + test)")
            out = oc.run(project, "Add a function `multiply(a, b)` to src/calc.js, add a test for it in "
                                  "test/calc.test.js, and run the tests.", session=session)
            show(out)
            src = open(os.path.join(project, "src", "calc.js")).read()
            record("S2 opencode ok", out["exit"] == 0 and not out["errors"], f"exit={out['exit']}")
            record("S2 multiply added + tests pass", "multiply" in src and tests_pass(project), "src + node --test")
            record("S2 no leaks", not leaks(out), f"markers: {leaks(out)}")
            s = summarize_log(bridge, mark)
            record("S2 checkpoint reuse", s["checkpoint_hits"] >= max(1, s["requests"] // 2), json.dumps(s))
            mark += s["requests"]

        if want("S3") and session:
            log("=== S3 memory across turns (stateless correctness)")
            out = oc.run(project, "Without using any tools: what was the bug you fixed at the start of this "
                                  "conversation, and what function did you add after that? Answer in two short lines.",
                         session=session)
            show(out)
            t = out["text"].lower()
            record("S3 remembers bug and feature", ("add" in t or "subtract" in t or "minus" in t or "-" in t)
                   and "multiply" in t, out["text"][:200])
            record("S3 no tools used", not out["tools"], f"tools: {out['tools']}")
            mark += summarize_log(bridge, mark)["requests"]

        if want("S4") and session:
            log("=== S4 fork the session (opencode --fork)")
            out = oc.run(project, "Without using tools, reply with only the name of the function you added most "
                                  "recently.", session=session, fork=True)
            show(out)
            record("S4 fork answer", "multiply" in out["text"].lower() and out["exit"] == 0, out["text"][:120])
            mark += summarize_log(bridge, mark)["requests"]

        if want("S5"):
            log("=== S5 two new sessions at the same time")
            prompts = ["Read README.md and tell me in one sentence what this project is.",
                       "Read package.json and tell me the exact test command. One line."]
            with ThreadPoolExecutor(2) as ex:
                outs = list(ex.map(lambda p: oc.run(project, p), prompts))
            for o in outs:
                show(o)
            record("S5 both ok", all(o["exit"] == 0 and not o["errors"] for o in outs),
                   f"exits={[o['exit'] for o in outs]}")
            record("S5 answers distinct and right", "calculator" in outs[0]["text"].lower()
                   and "node --test" in outs[1]["text"] and not any(EVASIVE.search(o["text"]) for o in outs), f"{outs[0]['text'][:80]!r} / {outs[1]['text'][:80]!r}")
            mark += summarize_log(bridge, mark)["requests"]

        if want("S7"):
            log("=== S7 answers right after a tool result read like a normal model (no status talk)")
            qs = [("Read README.md and tell me in one sentence what this project is.", ["calculator"]),
                  ("List the exported functions in src/calc.js.", ["add", "formattotal"]),
                  ("What does the test file check? Read it and answer in one or two sentences.", ["add"]),
                  ("Read package.json and tell me the exact test command. One line.", ["node --test"])]
            ok = 0
            for q, must in qs:
                o = oc.run(project, q)
                t = o["text"]
                good = o["exit"] == 0 and all(m in t.lower() for m in must) and not EVASIVE.search(t) \
                    and not re.search(r"(previous|earlier) (task|request|reply|answer)|is done", t, re.I)
                ok += good
                log(f"    {'ok ' if good else 'BAD'} {t[:160]!r}")
            record("S7 direct answers after tools", ok == len(qs), f"{ok}/{len(qs)} direct")

        if want("S6") and session:
            log("=== S6 restart the bridge, continue the original session")
            bridge.stop()
            bridge.start()
            out = oc.run(project, "List the exported functions in src/calc.js. Use the read tool.", session=session)
            show(out)
            t = out["text"]
            record("S6 continues after restart", out["exit"] == 0 and "multiply" in t and "add" in t
                   and "formatTotal" in t and not EVASIVE.search(t), t[:200])
            s = summarize_log(bridge, mark)
            record("S6 checkpoint reuse after restart", s["checkpoint_hits"] >= 1, json.dumps(s))
            mark += s["requests"]

    except Exception as e:  # report and fall through to cleanup
        record("harness", False, repr(e))
    finally:
        bridge.stop()
        log("bridge log tail:")
        try:
            print("".join(open(bridge.logfile).readlines()[-25:]))
        except OSError:
            pass
        passed = sum(1 for r in results if r[1])
        log(f"RESULT {passed}/{len(results)} checks passed")
        for name, ok, detail in results:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if args.keep:
            log(f"kept workspace {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)
    sys.exit(0 if results and all(r[1] for r in results) else 1)


if __name__ == "__main__":
    main()
