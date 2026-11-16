#!/usr/bin/env python3
"""12 real opencode sessions in parallel through one bridge (Haiku), each building and then
extending a different project. Measures how the bridge behaves at full concurrency: in-flight
turns, claude processes, queueing/429s, quota events, errors, cache, latency, memory.

usage: parallel12.py [--bin PATH] [--model claude-haiku] [--sessions 12] [--keep]
"""
import argparse
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("oe2e", os.path.join(HERE, "opencode_e2e.py"))
assert _spec and _spec.loader
oe2e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oe2e)

COMMON = ("Use Node.js with no npm dependencies (node:test for tests, CommonJS). Put the logic in src/, tests in test/, "
          "add an `npm test` script to package.json and a short README.md. Run the tests and make sure they all pass.")

PROJECTS = [
    ("markdown", "Build a Markdown-to-HTML converter library and CLI (`node md.js input.md`) supporting headings, "
                 "bold/italic, inline code, fenced code blocks, links, ordered and unordered lists, and paragraphs. "
                 "Escape HTML in text.", "Add support for tables (pipe syntax with header separator) and blockquotes, with tests."),
    ("shortener", "Build a URL shortener HTTP API with node:http: POST /links {url} returns a 6-char code, GET /:code "
                  "redirects (302), GET /links/:code/stats returns hit count. Validate URLs, store in a JSON file "
                  "(path from env DATA_PATH). Tests start the server on a random port.",
     "Add optional custom codes and expiry (expiresAt); expired links return 410. Add tests."),
    ("todo", "Build a todo CLI (`node todo.js add|list|done|rm`) with priorities (low/med/high), due dates (YYYY-MM-DD) "
             "and sorting by priority then due date; store in todos.json in the current directory.",
     "Add tags (`--tag`), filtering by tag and an `overdue` command. Add tests."),
    ("lru", "Build an LRU cache library with get/set/delete/has/size, a max size, optional TTL per entry, and eviction "
            "callbacks. Include thorough tests including TTL expiry with an injectable clock.",
     "Add a `stats()` method (hits, misses, evictions) and a `peek()` that doesn't update recency. Add tests."),
    ("csvstats", "Build a CSV statistics CLI (`node stats.js file.csv`) that parses CSV with quoted fields and prints, "
                 "per numeric column, count, mean, median, min, max and standard deviation.",
     "Add `--group-by <column>` to print the stats per group, and a `--json` output mode. Add tests."),
    ("ratelimit", "Build a rate-limiter library with two strategies: token bucket and sliding-window log, a common "
                  "interface `tryAcquire(key)`, and an injectable clock. Thorough tests for both.",
     "Add a `RateLimiterGroup` that applies several limits at once (e.g. 10/sec and 100/min) and reports which "
     "limit blocked. Add tests."),
    ("pubsub", "Build an event bus library with subscribe/unsubscribe/publish, wildcard topics (`orders.*`, "
               "`orders.#`), once-subscriptions and error isolation between subscribers. Thorough tests.",
     "Add async subscribers with `publishAsync` that awaits all handlers and returns per-handler results. Add tests."),
    ("validator", "Build a small JSON-schema-like validator supporting type, required, properties, items, enum, "
                  "minimum/maximum, minLength/maxLength and pattern, returning a list of errors with JSON paths.",
     "Add `oneOf`, `anyOf` and `$ref` to local definitions. Add tests."),
    ("adventure", "Build a text-adventure engine: rooms loaded from a JSON file, exits, items, inventory, "
                  "commands (go, look, take, drop, inventory) and a scripted test harness that runs a command list "
                  "and checks the output. Include a sample world.",
     "Add locked doors that need a key item and a save/load game command. Add tests."),
    ("matrix", "Build a matrix library: construction, add, subtract, multiply, transpose, determinant, inverse "
               "(with a clear error for singular matrices) and equality with tolerance. Thorough tests.",
     "Add LU decomposition and solving linear systems Ax=b. Add tests."),
    ("cron", "Build a cron expression parser (5 fields: minute hour day-of-month month day-of-week, with *, lists, "
             "ranges and steps) and a `nextRun(expr, fromDate)` function. Thorough tests.",
     "Add named months/days (JAN, MON) and `nextRuns(expr, from, n)`. Add tests."),
    ("ledger", "Build a bank ledger library: accounts, deposits, withdrawals (no overdraft), transfers that are "
               "atomic, an append-only audit log of every operation, and balance-at-date queries. Thorough tests.",
     "Add multi-currency accounts with a configurable exchange-rate table for transfers. Add tests."),
]


def ts():
    return time.strftime("%H:%M:%S")


def status(bridge):
    try:
        return bridge.get("/admin/status")
    except Exception:  # noqa: BLE001
        return {}


def sampler(bridge, stop, samples):
    pid = bridge.proc.pid
    while not stop.is_set():
        st = status(bridge)
        b = (st.get("backends") or [{}])[0]
        procs = len(subprocess.run(["pgrep", "-f", "claude -p --input-format"], capture_output=True, text=True).stdout.split())
        rss = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        samples.append({"t": time.time(), "in_flight": b.get("in_flight", 0), "procs": procs,
                        "rss_mb": int(rss or 0) // 1024, "quota": (b.get("quota") or {}).get("status"),
                        "util": (b.get("quota") or {}).get("utilization"), "pending": (st.get("checkpoints") or {}).get("pending")})
        time.sleep(1)


def run_session(oc, root, name, build, extend, results):
    project = os.path.join(root, "projects", name)
    os.makedirs(project)
    subprocess.run(["git", "init", "-q"], cwd=project)
    json.dump({"name": name, "version": "0.1.0", "private": True}, open(os.path.join(project, "package.json"), "w"))
    r = {"name": name, "phases": []}
    session = None
    for phase, prompt in (("build", f"{build}\n{COMMON}"), ("extend", f"{extend}\nRun the whole test suite again.")):
        t0 = time.time()
        out = oc.run(project, prompt, session=session, timeout=2400)
        session = session or out["session"]
        r["phases"].append({"phase": phase, "secs": round(time.time() - t0, 1), "exit": out["exit"],
                            "tools": len(out["tools"]), "failed_tools": sum(1 for t in out["tools"] if t[1] == "error"),
                            "errors": out["errors"], "leaks": [m for m in oe2e.LEAK_MARKERS if m in out["raw"]]})
        print(f"[{ts()}] {name:<10} {phase:<6} exit={out['exit']} {round(time.time() - t0)}s tools={len(out['tools'])} "
              f"errors={len(out['errors'])}{' ' + out['errors'][0][:120] if out['errors'] else ''}", flush=True)
    p = subprocess.run("npm test 2>&1 | tail -15", cwd=project, shell=True, capture_output=True, text=True, timeout=300)
    passed = [l for l in p.stdout.splitlines() if l.strip().startswith(("# pass", "ℹ pass"))]
    failed = [l for l in p.stdout.splitlines() if l.strip().startswith(("# fail", "ℹ fail"))]
    r["npm_test_ok"] = p.returncode == 0
    r["tests"] = (passed[-1].split()[-1] if passed else "?") + " passed / " + (failed[-1].split()[-1] if failed else "?") + " failed"
    r["files"] = int(subprocess.run("find src test -type f 2>/dev/null | wc -l", cwd=project, shell=True,
                                    capture_output=True, text=True).stdout.strip() or 0)
    results.append(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(os.path.dirname(HERE), "target", "release", "completions-bridge"))
    ap.add_argument("--model", default="claude-haiku")
    ap.add_argument("--sessions", type=int, default=12)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    a.bin = os.path.abspath(a.bin)

    root = tempfile.mkdtemp(prefix="cb_par12_")
    port, token = oe2e.free_port(), "par-" + os.urandom(8).hex()
    bridge = oe2e.Bridge(a.bin, root, port, token)
    # 12-way concurrency, generous queue wait (title requests also need slots).
    with open(bridge.config) as f:
        cfg = f.read()
    cfg = cfg.replace("    max_concurrency: 4\n", "    max_concurrency: 12\n    queue_wait_ms: 120000\n")
    with open(bridge.config, "w") as f:
        f.write(cfg)
    oc = oe2e.OpenCode(root, port, token, f"bridge/{a.model}")
    print(f"[{ts()}] workspace {root}; {a.sessions} parallel opencode sessions on bridge/{a.model}", flush=True)

    samples, results = [], []
    stop = threading.Event()
    t_start = time.time()
    try:
        bridge.start()
        th_s = threading.Thread(target=sampler, args=(bridge, stop, samples), daemon=True)
        th_s.start()
        threads = [threading.Thread(target=run_session, args=(oc, root, n, b, e, results)) for n, b, e in PROJECTS[:a.sessions]]
        for t in threads:
            t.start()
            time.sleep(0.5)
        last = 0
        while any(t.is_alive() for t in threads):
            time.sleep(1)
            if samples and time.time() - last >= 30:
                s = samples[-1]
                print(f"[{ts()}] -- live: in_flight={s['in_flight']} claude_procs={s['procs']} bridge_rss={s['rss_mb']}MB "
                      f"quota={s['quota']} util={s['util']} pending={s['pending']} sessions_done={len(results)}/{a.sessions}", flush=True)
                last = time.time()
        for t in threads:
            t.join()
    finally:
        stop.set()
        time.sleep(1)
        bridge.stop()

    wall = time.time() - t_start
    db = os.path.join(bridge.workdir, "state", "bridge.db")
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute("SELECT * FROM request_log ORDER BY rowid")]
    turns = [r for r in rows if (r["tokens_in"] or 0) >= 3000]
    tin = sum(r["tokens_in"] or 0 for r in rows)
    cread = sum(r["cache_read"] or 0 for r in rows)
    errs = {}
    for r in rows:
        if r["error_class"]:
            errs[r["error_class"]] = errs.get(r["error_class"], 0) + 1
    ttfb = sorted(r["ttfb_ms"] for r in turns if r["ttfb_ms"])
    dur = sorted(r["total_ms"] for r in turns if r["total_ms"])
    hits = sum(1 for r in turns if (r["checkpoint_hit_depth"] or 0) > 0)

    print("\n" + "=" * 100 + "\nPER SESSION\n" + "=" * 100)
    print(f"{'project':<11} {'build':>14} {'extend':>14} {'npm test':>22} {'files':>6} {'errors':>7} leaks")
    for r in sorted(results, key=lambda x: x["name"]):
        ph = {p["phase"]: p for p in r["phases"]}
        fmt = lambda p: f"{p['secs']:>5}s/{p['tools']:>2}t/e{p['exit']}" if p else "-"  # noqa: E731
        nerr = sum(len(p["errors"]) for p in r["phases"])
        leaks = sorted({m for p in r["phases"] for m in p["leaks"]})
        print(f"{r['name']:<11} {fmt(ph.get('build')):>14} {fmt(ph.get('extend')):>14} "
              f"{('OK ' if r['npm_test_ok'] else 'FAIL ') + r['tests']:>22} {r['files']:>6} {nerr:>7} {leaks or ''}")

    print("\n" + "=" * 100 + "\nBRIDGE UNDER LOAD\n" + "=" * 100)
    print(f"wall time: {wall / 60:.1f} min for {a.sessions} parallel sessions")
    print(f"peak turns in flight: {max((s['in_flight'] for s in samples), default=0)} (limit 12); "
          f"peak claude processes: {max((s['procs'] for s in samples), default=0)}; "
          f"seconds at the limit: {sum(1 for s in samples if s['in_flight'] >= 12)}")
    print(f"bridge memory: start {samples[0]['rss_mb'] if samples else '?'} MB, peak {max((s['rss_mb'] for s in samples), default=0)} MB, "
          f"end {samples[-1]['rss_mb'] if samples else '?'} MB")
    qs = sorted({s['quota'] for s in samples if s['quota']})
    print(f"quota states seen: {qs}; max utilization reported: {max((s['util'] or 0 for s in samples), default=0)}")
    print(f"API requests: {len(rows)} ({len(turns)} model turns); errors by class: {errs or 'none'}")
    print(f"checkpoint reuse: {hits}/{len(turns)} turns ({100 * hits / max(len(turns), 1):.0f}%); "
          f"cache read {100 * cread / max(tin, 1):.0f}% of {tin:,} input tokens")
    if ttfb:
        print(f"time to first token: p50 {ttfb[len(ttfb) // 2]} ms, p90 {ttfb[int(len(ttfb) * .9)]} ms, max {ttfb[-1]} ms")
        print(f"request duration:    p50 {dur[len(dur) // 2]} ms, p90 {dur[int(len(dur) * .9)]} ms, max {dur[-1]} ms")
    ok = sum(1 for r in results if r["npm_test_ok"] and all(p["exit"] == 0 and not p["errors"] for p in r["phases"]))
    print(f"\nsessions fully successful (both phases exit 0, no errors, tests pass): {ok}/{a.sessions}")
    warn = subprocess.run(f"sed 's/\\x1b\\[[0-9;]*m//g' {bridge.logfile} | grep -E ' WARN | ERROR ' | tail -10",
                          shell=True, capture_output=True, text=True).stdout
    print("bridge warnings/errors (last 10):\n" + (warn or "  none"))
    stray = subprocess.run(["pgrep", "-f", "claude -p --input-format"], capture_output=True, text=True).stdout.split()
    print(f"stray claude processes: {len(stray)}")
    print(f"kept {root}" if a.keep else "")
    if not a.keep:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
