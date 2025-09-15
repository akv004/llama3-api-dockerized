#!/bin/sh
set -eu

APP_DIR="/home/amit/projects/PycharmProjects/llama3-api"
PID_FILE="$APP_DIR/api.pid"
PORT="8002"

stop_pid() {
  PID="$1"
  OWNER="$(ps -o user= -p "$PID" 2>/dev/null || true)"
  [ -z "$OWNER" ] && { echo "No process $PID"; return; }
  if [ "$OWNER" = "$(id -un)" ]; then
    kill "$PID" || true
    sleep 1
    ps -p "$PID" >/dev/null 2>&1 && kill -9 "$PID" || true
  else
    sudo kill "$PID" || true
    sleep 1
    ps -p "$PID" >/dev/null 2>&1 && sudo kill -9 "$PID" || true
  fi
  echo "Stopped PID $PID (owner: $OWNER)"
}

if [ -f "$PID_FILE" ]; then
  stop_pid "$(cat "$PID_FILE")"
  rm -f "$PID_FILE"
else
  PORT_PID="$(lsof -t -i:"$PORT" || true)"
  if [ -n "${PORT_PID:-}" ]; then
    stop_pid "$PORT_PID"
  else
    echo "No running Llama3 API found."
  fi
fi
