#!/usr/bin/env bash
# Stop completions-bridge started by scripts/start.sh (graceful: in-flight turns finish saving).
set -euo pipefail
HOME_DIR="${BRIDGE_HOME:-$HOME/.completions-bridge}"
PIDFILE="$HOME_DIR/bridge.pid"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  pid="$(cat "$PIDFILE")"
  kill -TERM "$pid"
  for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -0 "$pid" 2>/dev/null && kill -KILL "$pid"
  rm -f "$PIDFILE"
  echo "stopped (pid $pid)"
else
  rm -f "$PIDFILE"
  echo "not running"
fi
