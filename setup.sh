#!/usr/bin/env bash
# LLM Maker — Automated Setup & Installation (Linux / macOS / WSL)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================="
echo "  LLM Maker — Setup & Installation Assistant"
echo "========================================================================="

# Verify Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 was not found in PATH. Please install Python 3.10+."
    exit 1
fi

# Verify Node.js / npm
if ! command -v npm &> /dev/null; then
    echo "[ERROR] npm was not found in PATH. Please install Node.js 18+."
    exit 1
fi

# Create virtual environment if needed
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating virtual environment (.venv)..."
    python3 -m venv .venv
else
    echo "[1/4] Existing virtual environment detected (.venv)."
fi

PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
PIP_BIN="$SCRIPT_DIR/.venv/bin/pip"

PLAN_ARGS=()
if [ "${LLMMAKER_CPU:-0}" = "1" ]; then
    PLAN_ARGS+=(--cpu)
elif [ -n "${LLMMAKER_CUDA_TAG:-}" ]; then
    PLAN_ARGS+=(--cuda "$LLMMAKER_CUDA_TAG")
fi

echo "[2/4] Detecting hardware and selecting PyTorch runtime..."
if ! python3 -m installer plan "${PLAN_ARGS[@]}"; then
    echo "[ERROR] No compatible PyTorch runtime. Update the NVIDIA driver or run with LLMMAKER_CPU=1."
    exit 1
fi

echo "[3/4] Installing PyTorch (this can download several GB)..."
"$PYTHON_BIN" -m installer torch --venv "$SCRIPT_DIR/.venv" --manifest "$SCRIPT_DIR/.venv/runtime.json" "${PLAN_ARGS[@]}"
"$PIP_BIN" install -r requirements.txt

echo "[4/4] Installing Frontend dependencies..."
cd frontend
npm install
cd ..

echo "========================================================================="
echo "  Setup Complete!"
echo "  Run \"chmod +x run_studio.sh && ./run_studio.sh\" to start the workstation."
echo "========================================================================="
