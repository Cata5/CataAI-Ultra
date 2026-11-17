"""
dashboard.py
Launch script for RoCanon-48.5M Training WebUI v2.
Runs the aiohttp WebUI server and coordinates the decoupled training worker,
hardware telemetry, and SQLite persistence.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import torch
from aiohttp import web

import src.dashboard.server as server_module
from src.dashboard.server import (
    STATE,
    WORKER_MGR,
    create_app,
    start_dashboard_server,
)

logger = logging.getLogger("RoCanonDashboard")


def main():
    venv_py_win = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
    venv_py_nix = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
    venv_py = venv_py_win if venv_py_win.exists() else (venv_py_nix if venv_py_nix.exists() else None)
    if venv_py and venv_py.exists() and Path(sys.executable).resolve() != venv_py.resolve():
        print(f"[Launcher] Switching to project virtualenv: {venv_py}")
        os.execv(str(venv_py), [str(venv_py)] + sys.argv)

    parser = argparse.ArgumentParser(description="RoCanon-48.5M Training WebUI v2")
    parser.add_argument("--port", type=int, default=8080, help="Web UI port (default: 8080)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--train", action="store_true", help="Auto-start training worker on launch")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'cuda')")
    args = parser.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Only"

    print("=" * 75)
    print("      RoCanon-48.5M TRAINING WEBUI v2 — EXPERIMENTATION PLATFORM")
    print("=" * 75)
    print(f" Web Interface: http://{args.host}:{args.port}")
    print(f" Target Compute: {device_str} ({gpu_name})")
    print(" Architecture: Decoupled Multiprocessing Worker + NVML Telemetry + SQLite")
    print(" VRAM Protection: Active ceiling < 5.0 GB on 6GB RTX 3060 Laptop")
    print("=" * 75)

    WORKER_MGR.device_str = device_str

    if args.train:
        print("[Launcher] Auto-start training requested via --train flag.")
        WORKER_MGR.start_run({
            "phase": "pretrain",
            "dataset_source": "tech_wiki_ro",
        })

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_module.MAIN_EVENT_LOOP = loop
    app = create_app()

    STATE.set_event_loop(loop)

    try:
        web.run_app(app, host=args.host, port=args.port, loop=loop)
    except (KeyboardInterrupt, SystemExit):
        print("\n[Launcher] Shutting down studio...")
    finally:
        WORKER_MGR.stop()
        print("[Launcher] Worker cleanly stopped. Goodbye!")


if __name__ == "__main__":
    main()
