#!/usr/bin/env bash
# Build (if needed) and start completions-bridge in the background, then print the API URL,
# token and model names. Safe to run again: if it is already running, it just prints them.
#
# Environment:
#   BRIDGE_HOME   state + config directory (default: ~/.completions-bridge)
#   BRIDGE_PORT   port to use when creating a new config (default: 4141, or the next free one)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOME_DIR="${BRIDGE_HOME:-$HOME/.completions-bridge}"
CONFIG="$HOME_DIR/config.yaml"
PIDFILE="$HOME_DIR/bridge.pid"
LOG="$HOME_DIR/bridge.log"
BIN="$REPO/target/release/completions-bridge"

die() { echo "error: $*" >&2; exit 1; }

command -v cargo >/dev/null || die "cargo not found: install Rust from https://rustup.rs"
command -v claude >/dev/null || die "claude not found: install Claude Code and log in (run 'claude' once)"
command -v curl >/dev/null || die "curl not found"

mkdir -p "$HOME_DIR"
chmod 700 "$HOME_DIR"

port_free() { ! (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

# 1. Config: create once, with a random token and a free port.
if [ ! -f "$CONFIG" ]; then
  port="${BRIDGE_PORT:-4141}"
  while ! port_free "$port"; do port=$((port + 1)); done
  token="cb-$(od -An -tx1 -N20 /dev/urandom | tr -d ' \n')"
  sed -e "s#^  listen: .*#  listen: 127.0.0.1:$port#" \
      -e "s#^  token: .*#  token: $token#" \
      -e "s#^  workdir: .*#  workdir: $HOME_DIR#" \
      "$REPO/config.example.yaml" > "$CONFIG"
  chmod 600 "$CONFIG"
  echo "created $CONFIG (port $port)"
fi

listen="$(sed -n 's/^  listen: *//p' "$CONFIG" | head -1)"
token="$(sed -n 's/^  token: *//p' "$CONFIG" | head -1)"
[ -n "$listen" ] || die "server.listen missing in $CONFIG"
base="http://$listen/v1"
auth=()
[ -n "$token" ] && [ "$token" != "null" ] && auth=(-H "Authorization: Bearer $token")

models_json() { curl -fsS --max-time 3 "${auth[@]}" "$base/models" 2>/dev/null; }

# 2. Already running?
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null && models_json >/dev/null; then
  echo "already running (pid $(cat "$PIDFILE"))"
else
  # 3. Build.
  if [ ! -x "$BIN" ] || [ -n "$(find "$REPO/crates" "$REPO/Cargo.toml" -newer "$BIN" -print -quit 2>/dev/null)" ]; then
    echo "building (cargo build --release) ..."
    (cd "$REPO" && cargo build --release --quiet) || die "build failed"
  fi
  # 4. Start.
  echo "starting ..."
  nohup "$BIN" serve --config "$CONFIG" >>"$LOG" 2>&1 &
  echo $! >"$PIDFILE"
  # 5. Wait (the first start runs a short self-test against claude).
  for _ in $(seq 1 180); do
    models_json >/dev/null && break
    kill -0 "$(cat "$PIDFILE")" 2>/dev/null || die "server exited; last log lines:
$(tail -20 "$LOG")"
    sleep 1
  done
  models_json >/dev/null || die "server did not become ready; see $LOG"
fi

# 6. Report.
status="$(curl -fsS --max-time 3 "${auth[@]}" "http://$listen/admin/status" 2>/dev/null || true)"
mode="$(printf '%s' "$status" | sed -n 's/.*"mode":"\([a-z]*\)".*/\1/p' | head -1)"
echo
echo "completions-bridge is running"
echo "  base URL : $base"
echo "  api key  : ${token:-<none>}"
echo "  backend  : claude (mode: ${mode:-unknown})"
echo "  models   :"
models_json | tr ',' '\n' | sed -n 's/.*"id":"\([^"]*\)".*/    - \1/p'
echo "  logs     : $LOG"
echo "  stop     : $REPO/scripts/stop.sh"
if [ "$mode" = "disabled" ]; then
  echo
  echo "warning: the claude backend is disabled (not installed or not logged in). Run 'claude' once, then restart."
fi
