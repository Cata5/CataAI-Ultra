"""Headless DDP launcher for `torchrun`.

Example (single node, two GPUs):
    torchrun --standalone --nproc_per_node=2 -m src.worker.ddp_runner \
        --payload-file run_payload.json

Every rank executes the same training session; rank 0 owns SQLite writes,
checkpoints and event persistence. Events are emitted as JSON lines on stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading

from .training_worker import TrainingWorkerProcess


def _load_payload(path: str | None, inline: str | None) -> dict:
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if inline:
        return json.loads(inline)
    raise SystemExit("Pass --payload-file or --payload")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a training payload under torchrun/DDP")
    parser.add_argument("--payload-file")
    parser.add_argument("--payload")
    parser.add_argument("--db-path")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [DDP-%(process)d] %(message)s",
    )
    payload = _load_payload(args.payload_file, args.payload)
    payload.setdefault("is_cluster", True)

    events: queue.Queue = queue.Queue()
    commands: queue.Queue = queue.Queue()
    worker = TrainingWorkerProcess(
        cmd_queue=commands,
        event_queue=events,
        device_str=args.device,
        db_path=args.db_path or None,
    )
    stop = threading.Event()

    def emit() -> None:
        while not stop.is_set() or not events.empty():
            try:
                event = events.get(timeout=0.2)
            except queue.Empty:
                continue
            sys.stdout.write(json.dumps(event) + "\n")
            sys.stdout.flush()

    writer = threading.Thread(target=emit, daemon=True)
    writer.start()
    try:
        worker._execute_training_session(payload)
    finally:
        worker.session_done.wait(timeout=10)
        stop.set()
        writer.join(timeout=5)
        try:
            import torch
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
