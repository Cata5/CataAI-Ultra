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
    echo "[1/3] Creating virtual environment (.venv)..."
    python3 -m venv .venv
else
    echo "[1/3] Existing virtual environment detected (.venv)."
fi

PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
PIP_BIN="$SCRIPT_DIR/.venv/bin/pip"

echo "[2/3] Installing Python dependencies..."
$PIP_BIN install --upgrade pip
$PIP_BIN install -r requirements.txt

echo "[3/3] Installing Frontend dependencies..."
cd frontend
npm install
cd ..

echo "========================================================================="
echo "  Setup Complete!"
echo "  Run \"chmod +x run_studio.sh && ./run_studio.sh\" to start the workstation."
echo "========================================================================="
