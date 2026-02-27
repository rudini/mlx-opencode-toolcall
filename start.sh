#!/usr/bin/env bash
#
# start.sh — Launch the full MLX + OpenCode tool-calling stack
#
# Usage: ./start.sh
#
# Starts:
#   1. mlx-openai-server on port 8000 (the MLX inference backend)
#   2. proxy_server.py on port 5001 (tool-call middleware)
#
# Then run OpenCode in another terminal:
#   opencode --provider mlx-local
#

set -euo pipefail

VENV_DIR="${MLX_VENV:-$HOME/mlx-env}"
MODEL="mlx-community/Qwen3.5-35B-A3B-4bit"
BACKEND_PORT=8000
PROXY_PORT=5001
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Activate venv -----------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "ERROR: Python venv not found at $VENV_DIR"
  echo "Create it first:  python3 -m venv $VENV_DIR"
  exit 1
fi

echo "Activating venv: $VENV_DIR"
source "$VENV_DIR/bin/activate"

# --- Cleanup on exit ---------------------------------------------------------
cleanup() {
  echo ""
  echo "Shutting down..."
  [ -n "${BACKEND_PID:-}" ] && kill "$BACKEND_PID" 2>/dev/null && echo "Stopped mlx-openai-server (PID $BACKEND_PID)"
  [ -n "${PROXY_PID:-}" ] && kill "$PROXY_PID" 2>/dev/null && echo "Stopped proxy (PID $PROXY_PID)"
  wait 2>/dev/null
  echo "Done."
}
trap cleanup EXIT INT TERM

# --- Start mlx-openai-server -------------------------------------------------
echo "Starting mlx-openai-server on port $BACKEND_PORT..."
echo "Model: $MODEL"
echo "(First run will download ~19 GB from HuggingFace)"
echo ""

python -m mlx_vlm.server \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port "$BACKEND_PORT" &
BACKEND_PID=$!

# Wait for the backend to become ready
echo "Waiting for backend to start..."
for i in $(seq 1 120); do
  if curl -sf "http://localhost:$BACKEND_PORT/v1/models" >/dev/null 2>&1; then
    echo "Backend is ready on port $BACKEND_PORT"
    break
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "ERROR: Backend process exited unexpectedly"
    exit 1
  fi
  sleep 1
done

if ! curl -sf "http://localhost:$BACKEND_PORT/v1/models" >/dev/null 2>&1; then
  echo "ERROR: Backend did not start within 120 seconds"
  exit 1
fi

# --- Start proxy server ------------------------------------------------------
echo ""
echo "Starting tool-call proxy on port $PROXY_PORT..."
python "$SCRIPT_DIR/proxy_server.py" &
PROXY_PID=$!
sleep 1

if ! kill -0 "$PROXY_PID" 2>/dev/null; then
  echo "ERROR: Proxy process exited unexpectedly"
  exit 1
fi

# --- Ready --------------------------------------------------------------------
echo ""
echo "============================================"
echo "  Stack is running!"
echo "  Backend:  http://localhost:$BACKEND_PORT"
echo "  Proxy:    http://localhost:$PROXY_PORT"
echo "============================================"
echo ""
echo "Open another terminal and run:"
echo "  opencode --provider mlx-local"
echo ""
echo "Press Ctrl+C to stop everything."

# Wait for background processes
wait
