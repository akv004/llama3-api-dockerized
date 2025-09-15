#!/bin/sh
set -eu

APP_DIR="/home/amit/projects/PycharmProjects/synapse-serve"
LOG_DIR="/home/amit/projects/logs"
PID_FILE="$APP_DIR/api.pid"
PY="/home/amit/miniconda3/envs/synapse-serve-llama3/bin/python"
PORT="8002"
LOG_FILE="$LOG_DIR/llama3_api.log"

# Runtime env (no installs here)
export BASE_DIR="/home/amit/projects"
export HF_HOME="/home/amit/projects/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/home/amit/projects/.cache/huggingface/hub"
# Redis already runs as a service per your note; no systemctl start here
export REDIS_URL="redis://localhost:6379/0"
# Change FAISS_DIR to enable RAG
export FAISS_DIR="/home/amit/projects/_no_rag"
export EMBED_MODEL="BAAI/bge-small-en-v1.5"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "$LOG_DIR" "$APP_DIR"

# Already running?
if [ -f "$PID_FILE" ] && ps -p "$(cat "$PID_FILE")" >/dev/null 2>&1; then
  echo "API already running (PID $(cat "$PID_FILE"))."
  exit 0
fi

# Refuse if port in use
if command -v lsof >/dev/null 2>&1; then
  PORT_PID="$(lsof -t -i:"$PORT" || true)"
  if [ -n "${PORT_PID:-}" ]; then
    echo "Port $PORT in use by PID ${PORT_PID}; not starting." >&2
    exit 1
  fi
fi

cd "$APP_DIR"
nohup "$PY" -m uvicorn llama3_api_langchain_rag_redis_sse:app \
  --host 0.0.0.0 --port "$PORT" --workers 1 \
  >> "$LOG_FILE" 2>&1 &

API_PID=$!
echo "$API_PID" > "$PID_FILE"
echo "Started Llama3 API PID $API_PID (logs: $LOG_FILE)"
