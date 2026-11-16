#!/usr/bin/env python3
"""Production-readiness checks against the release binary.

Operational properties that unit tests, conformance and the e2e runs do not cover:
load/overhead/leaks, concurrency limits, shutdown, crash recovery, backend outages,
corrupt state, disk caps, file permissions, prompt exposure, config errors, auth,
self-test caching. Uses the scripted fake backend wherever real claude isn't needed;
real claude (Haiku) only for shutdown, crash and corruption behaviour (~8 calls).

usage: readiness.py [--bin target/release/completions-bridge] [--only R1,R2,...]
"""
import argparse
import json
import os
import shutil
import signal
import socket
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results = []


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def record(rid, name, ok, detail):
    results.append((rid, name, ok, detail))
    log(f"{'PASS' if ok else 'FAIL'} {rid} {name}: {detail}")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, binary, cfg_extra, token="tok", workdir=None):
        self.binary = binary
        self.root = tempfile.mkdtemp(prefix="cb_ready_")
        self.port = free_port()
        self.token = token
        self.workdir = workdir or os.path.join(self.root, "w")
        self.cfg = os.path.join(self.root, "c.yaml")
        self.log = os.path.join(self.root, "bridge.log")
        with open(self.cfg, "w") as f:
            f.write(f"server:\n  listen: 127.0.0.1:{self.port}\n  token: {token}\n  workdir: {self.workdir}\n"
                    f"  header_wait_ms: 1000\n{cfg_extra}")
        self.proc = None

    def start(self, wait=180):
        self.proc = subprocess.Popen([self.binary, "serve", "--config", self.cfg], stdout=open(self.log, "a"),
                                     stderr=subprocess.STDOUT, env=dict(os.environ, BRIDGE_LOG="info,bridge_engine=debug"))
        t0 = time.time()
        while time.time() - t0 < wait:
            if self.proc.poll() is not None:
                return False
            try:
                self.req("GET", "/v1/models")
                return True
            except Exception:
                time.sleep(0.2)
        return False

    def stop(self, sig=signal.SIGTERM, timeout=40):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(sig)
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        return self.proc.returncode if self.proc else None

    def req(self, method, path, body=None, token=True, timeout=120):
        h = {"content-type": "application/json"}
        if token:
            h["authorization"] = f"Bearer {self.token}"
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", method=method, headers=h,
                                   data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                return resp.status, dict(resp.headers), resp.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read().decode()

    def chat(self, model, text, stream=False, history=None, **kw):
        msgs = (history or []) + [{"role": "user", "content": text}]
        return self.req("POST", "/v1/chat/completions", {"model": model, "messages": msgs, "stream": stream, **kw})

    def cleanup(self):
        self.stop()
        shutil.rmtree(self.root, ignore_errors=True)


def rss_kb(pid):
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) if out else 0


def fds(pid):
    out = subprocess.run(["lsof", "-p", str(pid)], capture_output=True, text=True).stdout
    return max(len(out.splitlines()) - 1, 0)


def claude_procs():
    out = subprocess.run(["pgrep", "-f", "claude -p --input-format"], capture_output=True, text=True).stdout.split()
    return set(out)


FAKE = """models:
  bridge/fake: {{ backend: fake, model: m }}
backends:
  fake:
    kind: fake
    default_model: m
    max_concurrency: {conc}
    queue_wait_ms: {queue}
    options: {opts}
checkpoints: {{ max_total_mb: {cap}, ttl_hours: 168 }}
"""

REAL = """models:
  bridge/claude-haiku: { backend: claude, model: claude-haiku-5-5 }
backends:
  claude:
    kind: claude-cli
    default_model: claude-haiku-5-5
    side_request_model: claude-haiku-5-5
    max_concurrency: 4
"""


def fake_cfg(conc=24, queue=30000, opts="{}", cap=2048):
    return FAKE.format(conc=conc, queue=queue, opts=opts, cap=cap)


# ------------------------------------------------------------------ checks

def r1_load(binary):
    s = Server(binary, fake_cfg())
    try:
        assert s.start(), "start"
        pid = s.proc.pid
        for _ in range(50):  # warm up
            s.chat("bridge/fake", "warm")
        rss0, fd0 = rss_kb(pid), fds(pid)
        n, lat, errs = 3000, [], []
        lock = threading.Lock()

        def one(i):
            t0 = time.time()
            hist = [{"role": "user", "content": f"conv {i % 50}"}, {"role": "assistant", "content": f"echo: conv {i % 50}"}]
            st, _, body = s.chat("bridge/fake", f"msg {i}", stream=(i % 2 == 0), history=hist)
            dt = (time.time() - t0) * 1000
            if i % 2 == 0 and st == 200:
                body = "".join(json.loads(l[6:])["choices"][0]["delta"].get("content") or ""
                               for l in body.splitlines() if l.startswith("data: {") and json.loads(l[6:]).get("choices"))
            with lock:
                lat.append(dt)
                if st != 200 or ("echo: msg" not in body):
                    errs.append((st, body[:100]))

        t0 = time.time()
        with ThreadPoolExecutor(24) as ex:
            list(ex.map(one, range(n)))
        wall = time.time() - t0
        time.sleep(2)
        rss1, fd1 = rss_kb(pid), fds(pid)
        lat.sort()
        p50, p99 = lat[len(lat) // 2], lat[int(len(lat) * 0.99)]
        record("R1", "3000 mixed requests at concurrency 24 (fake backend)", not errs,
               f"errors {len(errs)}, {n / wall:.0f} req/s, bridge latency p50 {p50:.0f} ms p99 {p99:.0f} ms")
        record("R1", "bridge overhead per request ≤ 500 ms p99 (NF-3)", p99 <= 500, f"p99 {p99:.0f} ms")
        record("R1", "no memory/fd leak after 3000 requests", rss1 < rss0 * 2 + 20000 and fd1 <= fd0 + 10,
               f"RSS {rss0 // 1024} → {rss1 // 1024} MB, fds {fd0} → {fd1}")
        db = os.path.join(s.workdir, "state", "bridge.db")
        rows = sqlite3.connect(db).execute("select count(*) from request_log").fetchone()[0]
        record("R1", "every request logged", rows >= n, f"{rows} rows")
    finally:
        s.cleanup()


def r2_concurrency(binary):
    opts = '{rules: [{match: "*", text: "slow answer", delay_ms: 2000}]}'
    s = Server(binary, fake_cfg(conc=12, queue=500, opts=opts))
    try:
        assert s.start(), "start"
        out = []
        with ThreadPoolExecutor(30) as ex:
            out = list(ex.map(lambda i: s.chat("bridge/fake", f"q{i}"), range(30)))
        ok = [o for o in out if o[0] == 200]
        limited = [o for o in out if o[0] == 429]
        ra = all("retry-after" in {k.lower() for k in o[1]} for o in limited)
        record("R2", "12-slot limit: 12 served, rest 429 + Retry-After when the queue wait expires",
               len(ok) == 12 and len(limited) == 18 and ra, f"200: {len(ok)}, 429: {len(limited)}, Retry-After on all: {ra}")
    finally:
        s.cleanup()


def r3_graceful(binary):
    s = Server(binary, REAL)
    try:
        assert s.start(), "start"
        before = claude_procs()
        res = {}

        def long_req():
            res["r"] = s.chat("bridge/claude-haiku", "Write a 600-word essay about the history of bridges.", stream=True)
        th = threading.Thread(target=long_req)
        th.start()
        time.sleep(6)
        during = claude_procs() - before
        t0 = time.time()
        code = s.stop(signal.SIGTERM)
        took = time.time() - t0
        th.join(30)
        time.sleep(1)
        after = claude_procs() - before
        record("R3", "SIGTERM with a turn in flight: clean exit, claude stopped", code == 0 and not after and bool(during),
               f"exit {code} after {took:.1f}s; claude during: {len(during)}, after: {len(after)}")
    finally:
        s.cleanup()


def r4_crash(binary):
    s = Server(binary, REAL)
    try:
        assert s.start(), "start"
        before = claude_procs()
        th = threading.Thread(target=lambda: s.chat("bridge/claude-haiku", "Write a 600-word essay about rivers.", stream=True))
        th.start()
        time.sleep(6)
        s.stop(signal.SIGKILL)
        th.join(30)
        time.sleep(1)
        orphans = claude_procs() - before
        turns = os.path.join(s.workdir, "claude", "turns")
        left = os.listdir(turns) if os.path.isdir(turns) else []
        assert s.start(), "restart"
        time.sleep(2)
        still = claude_procs() - before
        left_after = os.listdir(turns) if os.path.isdir(turns) else []
        st, _, body = s.chat("bridge/claude-haiku", "Reply with just OK.")
        record("R4", "kill -9 mid-turn, restart: orphans killed, turn dirs cleaned, service works",
               not still and not left_after and st == 200,
               f"orphans after crash {len(orphans)}, after restart {len(still)}; turn dirs {len(left)} → {len(left_after)}; next request {st}")
    finally:
        for p in claude_procs():
            pass
        s.cleanup()


def r5_backend_down(binary):
    cfg = REAL.replace("    max_concurrency: 4\n", "    max_concurrency: 4\n    binary: /nonexistent/claude\n")
    s = Server(binary, cfg)
    try:
        started = s.start(wait=30)
        st, _, body = s.chat("bridge/claude-haiku", "hi") if started else (None, None, "")
        status = json.loads(s.req("GET", "/admin/status")[2]) if started else {}
        mode = (status.get("backends") or [{}])[0].get("mode")
        try:
            err = json.loads(body).get("error", {})
        except Exception:
            err = {}
        clean = "nonexistent" not in body and "claude" not in err.get("message", "").lower()
        record("R5", "backend missing: server still starts, requests get an OpenAI-format 5xx, no internals leaked",
               started and st in (503, 500) and bool(err) and clean, f"started {started}, mode {mode}, status {st}, error {err}")
    finally:
        s.cleanup()


def r6_corrupt(binary):
    s = Server(binary, REAL)
    try:
        assert s.start(), "start"
        hist = [{"role": "user", "content": "My favourite fruit is mango. Reply OK."}]
        tools = [{"type": "function", "function": {"name": "noop", "description": "Unused.", "parameters": {"type": "object", "properties": {}}}}]
        st, _, body = s.chat("bridge/claude-haiku", hist[0]["content"], tools=tools)
        a1 = json.loads(body)["choices"][0]["message"]["content"]
        time.sleep(2)
        snaps = os.path.join(s.workdir, "claude", "snapshots")
        files = os.listdir(snaps)
        for f in files:
            with open(os.path.join(snaps, f), "w") as fh:
                fh.write("{not json\n\x00garbage")
        st2, _, body2 = s.chat("bridge/claude-haiku", "What is my favourite fruit? One word.",
                              history=hist + [{"role": "assistant", "content": a1}], tools=tools)
        ans = json.loads(body2)["choices"][0]["message"]["content"] if st2 == 200 else body2
        record("R6", "corrupted checkpoint files: request still answered correctly from the history",
               st2 == 200 and "mango" in ans.lower() and len(files) > 0, f"snapshots corrupted {len(files)}, status {st2}, answer {ans[:60]!r}")
    finally:
        s.cleanup()


def r7_disk_cap(binary):
    s = Server(binary, fake_cfg(cap=1))
    try:
        assert s.start(), "start"
        big = "x" * 20000
        for i in range(150):
            s.chat("bridge/fake", f"{big} {i}", history=[{"role": "user", "content": big}, {"role": "assistant", "content": "ok"}])
        time.sleep(2)
        d = os.path.join(s.workdir, "fake")
        size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(d) for f in fs)
        record("R7", "checkpoint disk cap enforced (max_total_mb: 1)", size <= 1.5 * 1024 * 1024, f"snapshot bytes on disk {size:,}")
    finally:
        s.cleanup()


def r8_perms(binary):
    s = Server(binary, REAL)
    try:
        assert s.start(), "start"
        marker = "SECRET-SYSTEM-PROMPT-" + os.urandom(4).hex()
        seen = {}

        def run():
            s.req("POST", "/v1/chat/completions", {"model": "bridge/claude-haiku", "messages": [
                {"role": "system", "content": f"{marker}. Be brief."},
                {"role": "user", "content": "Write 300 words about clouds."}]})
        th = threading.Thread(target=run)
        th.start()
        for _ in range(40):
            ps = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True).stdout
            if "--input-format" in ps:
                seen["ps"] = ps
                break
            time.sleep(0.25)
        th.join(120)
        time.sleep(2)
        bad = []
        for dp, dirs, fs in os.walk(s.workdir):
            for n in dirs + fs:
                m = os.stat(os.path.join(dp, n)).st_mode
                if m & (stat.S_IRWXG | stat.S_IRWXO):
                    bad.append(os.path.join(dp, n).replace(s.workdir, "~"))
        wd = oct(os.stat(s.workdir).st_mode & 0o777)
        in_ps = marker in seen.get("ps", "")
        record("R8", "workdir 0700 and every file owner-only", wd == "0o700" and not bad, f"workdir {wd}, group/other-readable: {bad[:3]}")
        record("R8", "system prompt not visible in process list", bool(seen) and not in_ps,
               f"claude seen in ps: {bool(seen)}, prompt visible: {in_ps}")
    finally:
        s.cleanup()


def r9_config(binary):
    d = tempfile.mkdtemp()
    cases = {"bad yaml": "server: [unclosed", "unknown field": "server:\n  listne: 1\n",
             "route to missing backend": "models:\n  x: { backend: nope }\n"}
    out = []
    for name, text in cases.items():
        p = os.path.join(d, "c.yaml")
        open(p, "w").write(text)
        r = subprocess.run([binary, "serve", "--config", p], capture_output=True, text=True, timeout=20)
        out.append((name, r.returncode, (r.stderr or r.stdout).strip().splitlines()[-1][:90] if (r.stderr or r.stdout) else ""))
    shutil.rmtree(d, ignore_errors=True)
    record("R9", "invalid config: refuses to start with a clear message", all(c != 0 for _, c, _ in out),
           "; ".join(f"{n} → exit {c}: {m}" for n, c, m in out))


def r10_auth(binary):
    s = Server(binary, fake_cfg())
    try:
        assert s.start(), "start"
        paths = [("GET", "/v1/models"), ("GET", "/admin/status"), ("POST", "/v1/chat/completions"), ("GET", "/v1/models/bridge/fake")]
        codes = [s.req(m, p, {"model": "bridge/fake", "messages": [{"role": "user", "content": "x"}]} if m == "POST" else None,
                       token=False)[0] for m, p in paths]
        record("R10", "every route requires the token", all(c == 401 for c in codes), f"{dict(zip([p for _, p in paths], codes))}")
    finally:
        s.cleanup()


def r11_selftest_cache(binary):
    root = tempfile.mkdtemp(prefix="cb_ready_st_")
    wd = os.path.join(root, "w")
    try:
        times = []
        for _ in range(2):
            s = Server(binary, REAL, workdir=wd)
            t0 = time.time()
            ok = s.start()
            times.append(time.time() - t0)
            s.stop()
            shutil.rmtree(s.root, ignore_errors=True)
        record("R11", "self-test result cached: second start is fast", ok and times[1] < 5,
               f"first start {times[0]:.1f}s, second {times[1]:.1f}s")
    finally:
        shutil.rmtree(root, ignore_errors=True)


CHECKS = {"R1": r1_load, "R2": r2_concurrency, "R3": r3_graceful, "R4": r4_crash, "R5": r5_backend_down,
          "R6": r6_corrupt, "R7": r7_disk_cap, "R8": r8_perms, "R9": r9_config, "R10": r10_auth, "R11": r11_selftest_cache}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "target", "release", "completions-bridge"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    for rid, fn in CHECKS.items():
        if only and rid not in only:
            continue
        log(f"=== {rid} {fn.__name__}")
        try:
            fn(a.bin)
        except Exception as e:  # noqa: BLE001
            record(rid, fn.__name__, False, f"harness error: {e!r}")
    stray = claude_procs()
    record("END", "no stray claude processes after all checks", not stray, f"{len(stray)} left")
    print("\nSUMMARY")
    for rid, name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {rid:<4} {name}")
    print(f"\n{sum(r[2] for r in results)}/{len(results)} passed")
    sys.exit(0 if all(r[2] for r in results) else 1)


if __name__ == "__main__":
    main()
