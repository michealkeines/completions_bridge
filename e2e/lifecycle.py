#!/usr/bin/env python3
"""Run one realistic multi-step coding task through real opencode → bridge → claude and
report the full lifecycle: every API request (checkpoint reuse, appended messages, tokens,
cache reads, latency), every tool call opencode executed, the final project state, and
leak checks.

usage: lifecycle.py [--bin PATH] [--model claude-sonnet] [--keep]
"""
import argparse
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("oe2e", os.path.join(HERE, "opencode_e2e.py"))
oe2e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oe2e)

TASKS = [
    ("build",
     "Build a small command-line expense tracker in Node.js with no dependencies.\n"
     "- `node expenses.js add <amount> <category> [note...]` adds an expense (amount must be a positive number, "
     "store it with an auto-incrementing id and today's date in YYYY-MM-DD).\n"
     "- `node expenses.js list` prints all expenses, one per line.\n"
     "- `node expenses.js summary` prints the total per category and the grand total, with two decimals.\n"
     "- `node expenses.js delete <id>` removes an expense; unknown ids print an error and exit with code 1.\n"
     "Store data in `expenses.json` in the current working directory (create it if missing). "
     "Put the logic in `src/` (pure functions plus a small storage module) and keep `expenses.js` thin. "
     "Write tests with `node:test` covering every command and the error cases (invalid amount, unknown id, "
     "empty store). Add an `npm test` script and a short README. Run the tests and make sure they pass."),
    ("feature",
     "Add a `--month YYYY-MM` option to `list` and `summary` that only includes expenses from that month, "
     "with validation of the format (bad format → error, exit code 1). Add tests for it and run the whole test "
     "suite again."),
    ("review",
     "Without changing any files: give me a short review of the code you wrote. What are the two weakest spots, "
     "and how would you fix them? Keep it under 120 words."),
]


def sh(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True, timeout=120)
    return p.returncode, (p.stdout + p.stderr)


def request_rows(workdir):
    db = os.path.join(workdir, "state", "bridge.db")
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute("SELECT rowid, * FROM request_log ORDER BY rowid")]
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", default="claude-sonnet")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    args.bin = os.path.abspath(args.bin)

    root = tempfile.mkdtemp(prefix="cb_life_")
    project = os.path.join(root, "project")
    os.makedirs(project)
    subprocess.run(["git", "init", "-q"], cwd=project)
    with open(os.path.join(project, "package.json"), "w") as f:
        json.dump({"name": "expenses", "version": "0.1.0", "private": True, "type": "commonjs"}, f, indent=2)

    port, token = oe2e.free_port(), "life-" + os.urandom(8).hex()
    bridge = oe2e.Bridge(args.bin, root, port, token)
    oc = oe2e.OpenCode(root, port, token, f"bridge/{args.model}")
    print(f"workspace: {root}\nmodel: bridge/{args.model}\n")

    phases = []
    session = None
    try:
        t0 = time.time()
        bridge.start()
        print(f"bridge up in {time.time() - t0:.1f}s (includes self-test on first start)\n")
        for name, prompt in TASKS:
            before = len(request_rows(bridge.workdir))
            print(f"=== phase '{name}' ...", flush=True)
            out = oc.run(project, prompt, session=session, timeout=1800)
            session = session or out["session"]
            rows = request_rows(bridge.workdir)[before:]
            phases.append((name, out, rows))
            print(f"    opencode exit={out['exit']} in {out['secs']}s, {len(out['tools'])} tool calls, "
                  f"{len(rows)} API requests, errors={out['errors'][:1]}")
    finally:
        bridge.stop()

    # ------------------------------------------------------------- report
    print("\n" + "=" * 100)
    print("LIFECYCLE REPORT")
    print("=" * 100)
    all_rows = []
    for name, out, rows in phases:
        print(f"\n## phase: {name}  (opencode {out['secs']}s, exit {out['exit']})")
        print("\nTool calls executed by opencode:")
        for t in out["tools"]:
            print(f"  {t[0]:<10} {t[1]:<10} {t[2][:110]}")
        print("\nAPI requests seen by the bridge:")
        print(f"  {'#':>3} {'kind':<11} {'ckpt depth':>10} {'appended':>8} {'finish':<10} {'in tok':>7} "
              f"{'cache rd':>8} {'out tok':>7} {'ttfb ms':>7} {'total ms':>8} error")
        for r in rows:
            kind = "side" if (r["checkpoint_hit_depth"] or 0) == 0 and (r["appended_messages"] or 0) == 0 \
                and r["finish_reason"] == "stop" and (r["tokens_in"] or 0) < 3000 else "turn"
            print(f"  {r['rowid']:>3} {kind:<11} {r['checkpoint_hit_depth'] or 0:>10} {r['appended_messages'] or 0:>8} "
                  f"{(r['finish_reason'] or '-'):<10} {r['tokens_in'] or 0:>7} {r['cache_read'] or 0:>8} "
                  f"{r['tokens_out'] or 0:>7} {r['ttfb_ms'] or 0:>7} {r['total_ms'] or 0:>8} {r['error_class'] or ''}")
        all_rows += rows
        print("\nFinal text from the model:")
        print("  " + out["text"][-900:].replace("\n", "\n  "))

    turns = [r for r in all_rows if (r["tokens_in"] or 0) >= 3000]
    hits = [r for r in turns if (r["checkpoint_hit_depth"] or 0) > 0]
    tin = sum(r["tokens_in"] or 0 for r in all_rows)
    cread = sum(r["cache_read"] or 0 for r in all_rows)
    tout = sum(r["tokens_out"] or 0 for r in all_rows)
    errs = [r for r in all_rows if r["error_class"]]
    ttfb = sorted(r["ttfb_ms"] for r in turns if r["ttfb_ms"])
    print("\n" + "-" * 100)
    print("TOTALS")
    print(f"  API requests: {len(all_rows)} ({len(turns)} model turns, {len(all_rows) - len(turns)} side requests)")
    print(f"  checkpoint reuse: {len(hits)}/{len(turns)} turns resumed from a saved session")
    print(f"  tokens: in {tin:,} (cache read {cread:,} = {100 * cread / max(tin, 1):.0f}%), out {tout:,}")
    if ttfb:
        print(f"  time to first token: median {ttfb[len(ttfb) // 2]} ms, max {ttfb[-1]} ms")
    print(f"  errors: {len(errs)} {[r['error_class'] for r in errs]}")
    print(f"  wall time: {sum(o['secs'] for _, o, _ in phases):.0f}s across {len(phases)} phases")

    print("\nPROJECT VERIFICATION (independent of the model)")
    checks = []
    code, out = sh("npm test 2>&1 | tail -15", project)
    checks.append(("npm test passes", code == 0 and "fail 0" in out))
    tmp = tempfile.mkdtemp()
    run = lambda c: sh(f"node {os.path.join(project, 'expenses.js')} {c}", tmp)  # noqa: E731
    c1, _ = run("add 12.5 food lunch with team")
    c2, _ = run("add 7 transport bus")
    c3, o3 = run("summary")
    c4, o4 = run("delete 999")
    c5, o5 = run("add -3 food")
    month = time.strftime("%Y-%m")
    c6, o6 = run(f"list --month {month}")
    c7, _ = run("list --month 2026-13x")
    checks.append(("add works", c1 == 0 and c2 == 0))
    checks.append(("summary shows totals", c3 == 0 and "19.50" in o3))
    checks.append(("delete unknown id → exit 1", c4 == 1))
    checks.append(("negative amount rejected", c5 != 0))
    checks.append(("--month filter works", c6 == 0 and "lunch" in o6))
    checks.append(("bad --month rejected", c7 == 1))
    checks.append(("README exists", os.path.exists(os.path.join(project, "README.md"))))
    checks.append(("logic in src/", os.path.isdir(os.path.join(project, "src"))))
    leak_hits = sorted({m for _, o, _ in phases for m in oe2e.LEAK_MARKERS if m in o["raw"]})
    checks.append(("no internal leaks in opencode output", not leak_hits))
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if leak_hits:
        print("  leak markers:", leak_hits)
    print(f"\n  npm test tail:\n    " + out.strip().replace("\n", "\n    "))
    print(f"\n{sum(ok for _, ok in checks)}/{len(checks)} project checks passed")
    if args.keep:
        print(f"kept {root}")
    else:
        shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
