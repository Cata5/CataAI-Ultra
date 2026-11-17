#!/usr/bin/env bash
# LLM Maker — Studio Launcher (Linux / macOS / WSL)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================="
echo "  LLM Maker — NEURAL RESEARCH WORKSTATION"
echo "  Frontend: Next.js App Router (http://localhost:8080)"
echo "  Backend : PyTorch CUDA Engine (http://localhost:8081)"
echo "  Compute : Dynamic GPU / CUDA VRAM Protection"
echo "========================================================================="

# Resolve Python environment
PYTHON_BIN="python3"
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif [ -f "$SCRIPT_DIR/.venv/bin/python3" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python3"
fi

# Check CUDA availability
echo "[1/3] Checking PyTorch environment..."
$PYTHON_BIN -c "import torch; print(f'PyTorch CUDA Active: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Launch Backend in background
echo "[2/3] Launching Python Decoupled ML Backend on 127.0.0.1:8081..."
$PYTHON_BIN dashboard.py --port 8081 "$@" &
BACKEND_PID=$!

cleanup() {
    echo ""
    echo "Stopping LLM Maker backend (PID: $BACKEND_PID)..."
    kill $BACKEND_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 3

# Launch Frontend
echo "[3/3] Starting Next.js Web UI on http://localhost:8080..."
cd frontend
if [ ! -d ".next" ]; then
    echo "Building Next.js bundle..."
    npm run build
fi

npm run start -- -p 8080
