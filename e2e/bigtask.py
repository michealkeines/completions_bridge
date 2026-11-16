#!/usr/bin/env python3
"""Long real-work run: real opencode → bridge → claude builds and evolves a bookstore REST API
over several phases in ONE opencode session, like a day of tickets.

Live output: every tool call opencode runs and every API request the bridge serves
(checkpoint reuse, tokens, cache, latency). At the end: lifecycle totals and an independent
verification that starts the server the model built and exercises it over HTTP.

usage: bigtask.py --bin PATH [--model claude-sonnet] [--keep]
"""
import argparse
import importlib.util
import json
import os
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("oe2e", os.path.join(HERE, "opencode_e2e.py"))
assert _spec and _spec.loader
oe2e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oe2e)

PHASES = [
    ("1-build", """Build a bookstore inventory REST API in Node.js using only the standard library (node:http, node:fs, node:crypto, node:test). No npm dependencies.

Resources:
- books: id, title, author, isbn (ISBN-13, validate the checksum), price (> 0, two decimals), stock (integer >= 0), tags (array of strings).
- orders: id, items [{bookId, qty}], status (pending | paid | shipped | cancelled), total (computed from book prices), createdAt.

Endpoints:
- POST/GET/PUT/DELETE /books and /books/:id with full validation.
- GET /books supports ?q= (case-insensitive search in title and author), ?tag=, ?minPrice= and ?maxPrice=, and pagination ?page=&limit= (default limit 20, max 100). The response is {data, page, limit, total}.
- POST /orders checks stock for every item and decrements it all-or-nothing.
- PATCH /orders/:id/status enforces transitions pending→paid→shipped and pending→cancelled; cancelling restores stock. Invalid transitions return 409.
- GET /orders/:id.
- GET /reports/inventory returns low-stock books (stock < 5) and the total inventory value.

Persistence: a JSON file at data/db.json (path overridable with the DB_PATH env var), written atomically (temp file + rename).
Errors: always {"error": {"code", "message"}} with correct status codes (400 validation, 404 not found, 409 conflict, 405 method not allowed).
Structure: src/server.js (createServer, exported for tests), src/router.js, src/store.js, src/validation.js, src/handlers/books.js, src/handlers/orders.js, src/handlers/reports.js, and index.js that starts the server on PORT (default 3000). `npm start` runs index.js, `npm test` runs the tests.
Tests: node:test tests that start the server on a random port with a temporary DB_PATH and exercise every endpoint, the filters, pagination and every error case. Make all tests pass."""),
    ("2-auth", """Add API-key authentication. Keys live in config/keys.json as an array of {key, role}, where role is "reader" or "admin" (path overridable with the KEYS_PATH env var). Clients send the header `x-api-key`. A missing or unknown key gets 401; a reader calling any non-GET endpoint gets 403. Create a sample config/keys.json with one reader key and one admin key. Update the existing tests to use an admin key, add tests for 401, 403 and reader access, and make the full suite pass."""),
    ("3-observability", """Add a request-logging middleware that appends one JSON line per request to logs/access.log (path overridable with ACCESS_LOG_PATH) with method, path, status and durationMs. Also add GET /health, which requires no API key and returns {status: "ok", uptimeSeconds, books, orders}. Add tests for both, including a check that a request actually produces a log line, and run the full suite."""),
    ("4-bugfix", """A customer reported two problems: (1) ordering a book when the requested quantity exactly equals the remaining stock fails, and (2) it seems possible to cancel an order that has already been paid. Write regression tests that reproduce both reports first, run them, then fix whatever is actually wrong (if a report turns out not to be a bug, keep the test and say so). Run the full suite at the end."""),
    ("5-docs", """Write docs/API.md documenting every endpoint: method, path, auth requirement, query parameters, request body, example curl command and example response, plus the error format and status codes. Also add a CHANGELOG.md summarising what was built in this session. Then run the full test suite one final time and report the number of tests."""),
    ("6-review", """Without changing any files: what are the three most important things you would improve in this codebase before putting it in production, and why? Keep it under 150 words."""),
]

EXTRA_PHASES = [
    ("7-async-store", """Refactor src/store.js to use node:fs/promises instead of synchronous file access, with a serialized write queue so concurrent requests can never interleave writes or lose updates. Keep the API behaviour identical. Add a test that fires 50 concurrent POST /books requests and then checks that all 50 books exist. Run the full suite."""),
    ("8-csv", """Add CSV import and export for books: GET /books/export.csv (admin only) returns all books as CSV with a header row (tags joined with `|`, proper quoting of commas and quotes), and POST /books/import (admin only, body is CSV text with content-type text/csv) validates every row, imports all-or-nothing, and returns {imported, errors:[{line, message}]} with 400 if any row is invalid. Add tests, including a round trip export → import into an empty store, and run the suite."""),
    ("9-rate-limit", """Add per-API-key rate limiting: a fixed window of 60 seconds, with limits configurable in config/keys.json per key (field `rateLimit`, default 120 requests per window). Responses carry X-RateLimit-Limit, X-RateLimit-Remaining and X-RateLimit-Reset headers; over the limit you get 429 with a Retry-After header and the standard error body. /health is not rate limited. Make the clock injectable so tests don't need to wait. Add tests and run the suite."""),
    ("10-etag", """Add ETag support for GET /books/:id and GET /books: a strong ETag computed from the response body; If-None-Match returns 304 with no body when it matches. PUT /books/:id honours If-Match (412 Precondition Failed on mismatch) to prevent lost updates. Add tests and run the suite."""),
    ("11-openapi", """Write an OpenAPI 3.1 document at docs/openapi.json describing every endpoint, parameter, request body, response and the error schema. Serve it at GET /openapi.json (no auth). Add a test that loads the document and checks that every route the router knows about is documented and that every documented path exists in the router (no drift). Run the suite."""),
    ("12-dashboard", """Add a small dashboard: GET / serves public/index.html (no auth), a single static page with plain JS that asks for an API key, then shows the inventory report, a searchable paginated book table and the 10 most recent orders, using the existing API. Serve files from public/ safely (no path traversal; add a test that ../ attempts get 404). Keep it dependency-free. Run the suite."""),
    ("13-loadtest-cleanup", """Write scripts/loadtest.js (node, no deps) that starts the server on a random port with a temp DB, creates 20 books and then sends 500 mixed requests with concurrency 20, printing throughput and p50/p95 latency. Run it once and report the numbers. Then do a final cleanup pass: remove dead code, make error messages consistent, update docs/API.md and CHANGELOG.md for everything added since the last docs update, and run the full test suite one last time."""),
]

stop_monitor = threading.Event()


def ts():
    return time.strftime("%H:%M:%S")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def rows_since(db, last):
    if not os.path.exists(db):
        return []
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute("SELECT rowid, * FROM request_log WHERE rowid > ? ORDER BY rowid", (last,))]
    except sqlite3.Error:
        return []
    finally:
        con.close()


def monitor(db, sink):
    last = 0
    while not stop_monitor.is_set():
        for r in rows_since(db, last):
            last = r["rowid"]
            sink.append(r)
            print(f"[{ts()}]   api #{r['rowid']:<3} depth={r['checkpoint_hit_depth'] or 0:<3} +{r['appended_messages'] or 0:<2} "
                  f"{(r['finish_reason'] or r['error_class'] or '-'):<10} in={r['tokens_in'] or 0:>6} "
                  f"cache={r['cache_read'] or 0:>6} out={r['tokens_out'] or 0:>5} ttfb={r['ttfb_ms'] or 0:>6}ms "
                  f"total={r['total_ms'] or 0:>6}ms", flush=True)
        time.sleep(1.0)


def run_phase(oc, project, prompt, session):
    cmd = ["opencode", "run", "--standalone", "--format", "json", "--model", oc.model, "--auto"]
    if session:
        cmd += ["--session", session]
    cmd.append(prompt)
    env = oc.env()
    env["PWD"] = project
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=project, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         stdin=subprocess.DEVNULL)
    out = {"session": session, "tools": [], "text": "", "errors": [], "raw": ""}
    assert p.stdout
    for line in p.stdout:
        out["raw"] += line
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        out["session"] = out["session"] or e.get("sessionID")
        part = e.get("part") or {}
        if part.get("type") == "tool":
            st = part.get("state", {})
            if st.get("status") in ("completed", "error"):
                inp = st.get("input") or {}
                desc = inp.get("command") or inp.get("path") or inp.get("filePath") or inp.get("pattern") or json.dumps(inp)[:80]
                desc = str(desc).replace("\n", " ")[:100]
                out["tools"].append((part.get("tool"), st.get("status")))
                print(f"[{ts()}] tool {part.get('tool'):<8} {st.get('status'):<9} {desc}", flush=True)
        elif part.get("type") == "text" and part.get("text"):
            out["text"] += part["text"]
        elif e.get("type") == "error":
            out["errors"].append(json.dumps(e.get("error"))[:300])
            print(f"[{ts()}] OPENCODE ERROR {out['errors'][-1]}", flush=True)
    p.wait()
    out["exit"] = p.returncode
    out["secs"] = round(time.time() - t0, 1)
    return out


# ------------------------------------------------------------------ independent verification

def http(method, url, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    h = {"content-type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            txt = r.read().decode()
            return r.status, (json.loads(txt) if txt else None)
    except urllib.error.HTTPError as e:
        txt = e.read().decode()
        try:
            return e.code, json.loads(txt)
        except json.JSONDecodeError:
            return e.code, txt


def find(obj, key):
    """First value for `key` anywhere in a JSON object."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find(v, key)
            if r is not None:
                return r
    return None


def raw(method, url, headers=None):
    req = urllib.request.Request(url, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode(errors="replace")


def verify(project, long_run=False):
    checks = []
    p = subprocess.run("npm test 2>&1 | tail -12", cwd=project, shell=True, capture_output=True, text=True, timeout=300)
    m = re.search(r"# pass (\d+)|ℹ pass (\d+)", p.stdout)
    fails = re.search(r"# fail (\d+)|ℹ fail (\d+)", p.stdout)
    n_pass = int(next(g for g in m.groups() if g)) if m else 0
    n_fail = int(next(g for g in fails.groups() if g)) if fails else -1
    checks.append((f"npm test: {n_pass} passed, {n_fail} failed", p.returncode == 0 and n_fail == 0 and n_pass >= 20))

    keys_path = os.path.join(project, "config", "keys.json")
    try:
        keys = json.load(open(keys_path))
        admin = next(k["key"] for k in keys if k.get("role") == "admin")
        reader = next(k["key"] for k in keys if k.get("role") == "reader")
    except Exception as e:  # noqa: BLE001
        checks.append((f"config/keys.json readable ({e})", False))
        return checks
    tmp = tempfile.mkdtemp()
    port = free_port()
    env = dict(os.environ, PORT=str(port), DB_PATH=os.path.join(tmp, "db.json"),
               ACCESS_LOG_PATH=os.path.join(tmp, "access.log"), KEYS_PATH=keys_path)
    srv = subprocess.Popen(["node", "index.js"], cwd=project, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(50):
            try:
                http("GET", base + "/health")
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.2)
        A, R = {"x-api-key": admin}, {"x-api-key": reader}
        s, b = http("GET", base + "/health")
        checks.append(("GET /health without key → 200 ok", s == 200 and find(b, "status") == "ok"))
        s, _ = http("GET", base + "/books")
        checks.append(("GET /books without key → 401", s == 401))
        s, b = http("POST", base + "/books", {"title": "X", "author": "Y", "isbn": "9780306406158", "price": 10, "stock": 3, "tags": []}, A)
        checks.append(("bad ISBN checksum → 400 with error.code", s == 400 and isinstance(find(b, "error"), dict)))
        book = {"title": "Dune", "author": "Frank Herbert", "isbn": "9780306406157", "price": 12.5, "stock": 4, "tags": ["scifi"]}
        s, _ = http("POST", base + "/books", book, R)
        checks.append(("reader POST /books → 403", s == 403))
        s, b = http("POST", base + "/books", book, A)
        bid = find(b, "id")
        checks.append(("admin POST /books → 201/200 with id", s in (200, 201) and bid is not None))
        s, b = http("GET", base + "/books?q=dune&tag=scifi&page=1&limit=10", None, R)
        checks.append(("search + tag + pagination → total 1", s == 200 and find(b, "total") == 1))
        s, b = http("POST", base + "/orders", {"items": [{"bookId": bid, "qty": 4}]}, A)
        oid = find(b, "id")
        checks.append(("order exactly all stock → success", s in (200, 201) and oid is not None))
        s, b = http("POST", base + "/orders", {"items": [{"bookId": bid, "qty": 1}]}, A)
        checks.append(("order with no stock left → 409/400", s in (400, 409)))
        s, _ = http("PATCH", base + f"/orders/{oid}/status", {"status": "paid"}, A)
        checks.append(("pending → paid", s == 200))
        s, _ = http("PATCH", base + f"/orders/{oid}/status", {"status": "cancelled"}, A)
        checks.append(("paid → cancelled rejected with 409", s == 409))
        s, b = http("GET", base + "/reports/inventory", None, A)
        checks.append(("inventory report lists the low-stock book", s == 200 and "Dune" in json.dumps(b)))
        if long_run:
            st, hd, body = raw("GET", base + "/books/export.csv", A)
            checks.append(("CSV export (admin) has header + Dune row", st == 200 and "Dune" in body and "isbn" in body.lower()))
            hl = {k.lower(): v for k, v in hd.items()}
            st, hd, _ = raw("GET", base + "/books", R)
            hl = {k.lower(): v for k, v in hd.items()}
            checks.append(("rate-limit headers present", st == 200 and "x-ratelimit-remaining" in hl))
            st, hd, _ = raw("GET", base + f"/books/{bid}", A)
            etag = {k.lower(): v for k, v in hd.items()}.get("etag")
            st2, _, body2 = raw("GET", base + f"/books/{bid}", {**A, "If-None-Match": etag or "x"})
            checks.append(("ETag + If-None-Match → 304", bool(etag) and st2 == 304 and body2 == ""))
            st, _, body = raw("GET", base + "/openapi.json")
            checks.append(("GET /openapi.json is OpenAPI 3.1", st == 200 and '"3.1' in body))
            st, hd, body = raw("GET", base + "/")
            checks.append(("dashboard served at /", st == 200 and "<html" in body.lower()))
            st, _, _ = raw("GET", base + "/..%2f..%2fpackage.json")
            st3, _, _ = raw("GET", base + "/../package.json")
            checks.append(("path traversal blocked", st in (400, 401, 403, 404) and st3 in (400, 401, 403, 404)))
        log_ok = os.path.exists(env["ACCESS_LOG_PATH"]) and len(open(env["ACCESS_LOG_PATH"]).read().splitlines()) >= 5
        checks.append(("access log written as JSON lines", log_ok))
    finally:
        os.killpg(srv.pid, signal.SIGTERM)
        shutil.rmtree(tmp, ignore_errors=True)
    checks.append(("docs/API.md written", os.path.exists(os.path.join(project, "docs", "API.md"))))
    checks.append(("CHANGELOG.md written", os.path.exists(os.path.join(project, "CHANGELOG.md"))))
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", default="claude-sonnet")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--long", action="store_true", help="add phases 7-13 (about 10+ minutes in total)")
    args = ap.parse_args()
    args.bin = os.path.abspath(args.bin)
    phase_list = PHASES[:5] + (EXTRA_PHASES if args.long else []) + PHASES[5:]

    root = tempfile.mkdtemp(prefix="cb_big_")
    project = os.path.join(root, "project")
    os.makedirs(project)
    subprocess.run(["git", "init", "-q"], cwd=project)
    json.dump({"name": "bookstore-api", "version": "0.1.0", "private": True}, open(os.path.join(project, "package.json"), "w"), indent=2)

    port, token = oe2e.free_port(), "big-" + os.urandom(8).hex()
    bridge = oe2e.Bridge(args.bin, root, port, token)
    oc = oe2e.OpenCode(root, port, token, f"bridge/{args.model}")
    print(f"[{ts()}] workspace {root}  model bridge/{args.model}")
    rows = []
    phases = []
    t_start = time.time()
    mon = None
    try:
        bridge.start()
        mon = threading.Thread(target=monitor, args=(os.path.join(bridge.workdir, "state", "bridge.db"), rows), daemon=True)
        mon.start()
        session = None
        for name, prompt in phase_list:
            print(f"\n[{ts()}] ===================== phase {name} =====================", flush=True)
            out = run_phase(oc, project, prompt, session)
            session = session or out["session"]
            phases.append((name, out))
            print(f"[{ts()}] phase {name}: exit={out['exit']} {out['secs']}s, {len(out['tools'])} tool calls, errors={len(out['errors'])}")
            print(f"[{ts()}] model says: " + out["text"][-600:].replace("\n", " | "), flush=True)
    finally:
        time.sleep(2)
        stop_monitor.set()
        if mon:
            mon.join(5)
        bridge.stop()

    total_s = time.time() - t_start
    print("\n" + "=" * 100 + "\nLIFECYCLE SUMMARY\n" + "=" * 100)
    print(f"{'phase':<16} {'secs':>6} {'tools':>6} {'exit':>5} {'errors':>6}")
    for name, o in phases:
        print(f"{name:<16} {o['secs']:>6} {len(o['tools']):>6} {o['exit']:>5} {len(o['errors']):>6}")
    tool_counts = {}
    for _, o in phases:
        for t, st in o["tools"]:
            tool_counts[t] = tool_counts.get(t, 0) + 1
    failed_tools = sum(1 for _, o in phases for _, st in o["tools"] if st == "error")
    turns = [r for r in rows if (r["tokens_in"] or 0) >= 3000]
    hits = [r for r in turns if (r["checkpoint_hit_depth"] or 0) > 0]
    tin = sum(r["tokens_in"] or 0 for r in rows)
    cread = sum(r["cache_read"] or 0 for r in rows)
    tout = sum(r["tokens_out"] or 0 for r in rows)
    ttfb = sorted(r["ttfb_ms"] for r in turns if r["ttfb_ms"])
    tot = sorted(r["total_ms"] for r in turns if r["total_ms"])
    errs = [r["error_class"] for r in rows if r["error_class"]]
    print(f"\nwall time: {total_s / 60:.1f} min")
    print(f"tool calls: {sum(tool_counts.values())} {dict(sorted(tool_counts.items(), key=lambda x: -x[1]))}, failed tool runs: {failed_tools}")
    print(f"API requests: {len(rows)} ({len(turns)} model turns, {len(rows) - len(turns)} side requests), errors: {len(errs)} {errs}")
    exact = [r for r in turns if (r["checkpoint_hit_depth"] or 0) > 0 and (r["appended_messages"] or 0) <= 2]
    print(f"exact continuation (resumed from the immediately preceding checkpoint): {len(exact)}/{len(turns)} turns "
          f"({100 * len(exact) / max(len(turns), 1):.0f}%)")
    print(f"checkpoint reuse: {len(hits)}/{len(turns)} turns ({100 * len(hits) / max(len(turns), 1):.0f}%), max depth {max((r['checkpoint_hit_depth'] or 0) for r in rows) if rows else 0}")
    print(f"tokens: in {tin:,}, cache read {cread:,} ({100 * cread / max(tin, 1):.0f}%), out {tout:,}")
    if ttfb:
        print(f"time to first token: p50 {ttfb[len(ttfb) // 2]} ms, p90 {ttfb[int(len(ttfb) * 0.9)]} ms, max {ttfb[-1]} ms")
        print(f"request duration:    p50 {tot[len(tot) // 2]} ms, p90 {tot[int(len(tot) * 0.9)]} ms, max {tot[-1]} ms")
    leak_hits = sorted({m for _, o in phases for m in oe2e.LEAK_MARKERS if m in o["raw"]})
    print(f"leak markers in opencode output: {leak_hits or 'none'}")
    warn = subprocess.run(f"grep -cE ' WARN | ERROR ' {bridge.logfile}", shell=True, capture_output=True, text=True).stdout.strip()
    print(f"bridge log warnings/errors: {warn}")
    print(subprocess.run(f"sed 's/\\x1b\\[[0-9;]*m//g' {bridge.logfile} | grep -E ' WARN | ERROR ' | tail -8", shell=True,
                         capture_output=True, text=True).stdout)

    print("INDEPENDENT VERIFICATION OF THE BUILT SERVICE")
    checks = verify(project, args.long)
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{sum(ok for _, ok in checks)}/{len(checks)} checks passed")
    files = subprocess.run("find . -path ./node_modules -prune -o -path ./.git -prune -o -type f -print | sort", cwd=project,
                           shell=True, capture_output=True, text=True).stdout
    loc = subprocess.run("cat $(find src test* -name '*.js' 2>/dev/null) 2>/dev/null | wc -l", cwd=project, shell=True,
                         capture_output=True, text=True).stdout.strip()
    print(f"\nproject files:\n{files}lines of JS in src/ and tests: {loc}")
    print(f"{'kept ' + root if args.keep else ''}")
    if not args.keep:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
