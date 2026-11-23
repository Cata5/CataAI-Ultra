"""
src/worker/standalone_runner.py
CLI entrypoint for decoupled training worker when supervised by native desktop runtime (Tauri 2 / Rust).
Takes JSON payload from CLI or stdin, runs TrainingWorkerProcess in isolation,
and emits single-line JSON metrics to stdout for zero-overhead Rust IPC event broadcasting.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from pathlib import Path

# Configure minimal stdout logging so JSON metrics remain parseable
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", stream=sys.stderr)
logger = logging.getLogger("StandaloneWorker")


def main():
    parser = argparse.ArgumentParser(description="Standalone PyTorch Training Worker for Tauri Native Desktop")
    parser.add_argument("--payload", type=str, default=None, help="JSON string or file path containing run configuration")
    parser.add_argument("--device", type=str, default=None, help="CUDA device override (e.g. cuda:0, cpu)")
    parser.add_argument("--db-path", type=str, default="studio.db", help="Path to SQLite database")
    args = parser.parse_args()

    payload = {}
    if args.payload:
        if os.path.exists(args.payload):
            with open(args.payload, "r", encoding="utf-8") as f:
                payload = json.load(f)
        else:
            payload = json.loads(args.payload)
    elif not sys.stdin.isatty():
        raw = sys.stdin.read()
        if raw.strip():
            payload = json.loads(raw)

    if not payload:
        logger.error("No training payload provided. Exiting.")
        sys.exit(1)

    # Lazy import PyTorch and training worker ONLY NOW when ML job is active
    logger.info("Initializing PyTorch and CUDA runtime in isolated worker...")
    from src.worker.protocol import CommandType, EventType, WorkerCommand, WorkerEvent
    from src.worker.training_worker import TrainingWorkerProcess

    cmd_queue = mp.Queue()
    event_queue = mp.Queue()

    worker = TrainingWorkerProcess(
        cmd_queue=cmd_queue,
        event_queue=event_queue,
        device_str=args.device,
        db_path=args.db_path,
    )

    # Signal handlers for graceful desktop shutdown
    def handle_signal(sig, frame):
        logger.info("Shutdown signal received (%s), stopping training...", sig)
        cmd_queue.put(WorkerCommand(command_type=CommandType.STOP_RUN).to_dict())

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Background thread consuming worker events and printing JSON to stdout
    def event_emitter():
        while True:
            try:
                event_data = event_queue.get()
                if isinstance(event_data, dict):
                    # Direct dictionary from worker
                    et = event_data.get("event_type")
                    payload = event_data.get("payload", {})

                    if et == "METRICS_UPDATED" and payload:
                        print(json.dumps(payload), flush=True)
                    else:
                        print(json.dumps(event_data), flush=True)

                    if et in ("RUN_STOPPED", "RUN_FAILED", "RUN_COMPLETED"):
                        logger.info("Terminal run event received (%s), exiting standalone worker.", et)
                        time.sleep(0.5)
                        os._exit(0)
                elif hasattr(event_data, "event_type"):
                    et_val = event_data.event_type.value if hasattr(event_data.event_type, "value") else str(event_data.event_type)
                    if et_val == "METRICS_UPDATED" and getattr(event_data, "payload", None):
                        print(json.dumps(event_data.payload), flush=True)
                    else:
                        print(json.dumps(event_data.to_dict() if hasattr(event_data, "to_dict") else {"event": et_val}), flush=True)

                    if et_val in ("RUN_STOPPED", "RUN_FAILED", "RUN_COMPLETED"):
                        logger.info("Terminal run event received (%s), exiting standalone worker.", et_val)
                        time.sleep(0.5)
                        os._exit(0)
            except Exception as e:
                logger.debug("Emitter exception: %s", e)
                break

    emitter_thread = threading.Thread(target=event_emitter, daemon=True)
    emitter_thread.start()

    # Background thread consuming worker commands from stdin
    def stdin_listener():
        try:
            for line in iter(sys.stdin.readline, ""):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    ct_str = data.get("command_type")
                    if ct_str:
                        logger.info("Worker received command from stdin: %s", ct_str)
                        cmd = WorkerCommand(
                            command_type=CommandType(ct_str),
                            payload=data.get("payload", {}),
                        )
                        cmd_queue.put(cmd.to_dict())
                except Exception as ex:
                    logger.warning("Error parsing stdin command: %s", ex)
        except Exception as e:
            logger.debug("stdin_listener ended: %s", e)

    listener_thread = threading.Thread(target=stdin_listener, daemon=True)
    listener_thread.start()

    # Dispatch START_RUN command
    cmd_queue.put(WorkerCommand(command_type=CommandType.START_RUN, payload=payload).to_dict())

    try:
        worker.run_loop()
    except Exception as e:
        logger.exception("Worker process encountered fatal exception: %s", e)
        print(json.dumps({"error": str(e), "event": "RUN_FAILED"}), flush=True)
        sys.exit(1)

    logger.info("Worker process completed cleanly.")
    sys.exit(0)


if __name__ == "__main__":
    main()
