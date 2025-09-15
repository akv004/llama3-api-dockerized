#!/bin/sh
set -eu

APP_DIR="/home/amit/projects/PycharmProjects/synapse-serve"
PY="/home/amit/miniconda3/envs/synapse-serve-llama3/bin/python"
PORT="8002"

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

# Refuse if port in use
if command -v lsof >/dev/null 2>&1; then
  PORT_PID="$(lsof -t -i:"$PORT" || true)"
  if [ -n "${PORT_PID:-}" ]; then
    echo "Port $PORT in use by PID ${PORT_PID}; not starting." >&2
    exit 1
  fi
fi

cd "$APP_DIR"

echo "Starting Llama3 API in foreground on port $PORT..."
echo "Press Ctrl+C to stop the server."

"$PY" -m uvicorn llama3_api_langchain_rag_redis_sse:app --host 0.0.0.0 --port "$PORT" --workers 1