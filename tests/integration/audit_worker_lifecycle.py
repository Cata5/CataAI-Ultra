"""
End-to-end lifecycle check for the isolated training worker.

Runs the real worker entrypoint three times against a throwaway SQLite database:
  1. stop a live run and verify the STOPPED row plus the resumable snapshot,
  2. resume that snapshot and let the run complete,
  3. resume from a corrupted checkpoint and verify the run fails loudly.

Usage:  .venv\\Scripts\\python.exe tests\\integration\\audit_worker_lifecycle.py
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def base_payload(**overrides):
    payload = {
        "workspace_id": "default",
        "config_arch": {
            "architecture_family": "rocanon_native",
            "vocab_size": 12288,
            "d_model": 128,
            "n_layers": 2,
            "n_q": 4,
            "n_kv": 2,
            "d_ff": 256,
            "max_seq_len": 128,
        },
        "dataset_id": "synthetic",
        "task": "causal_lm",
        "duration_mode": "steps",
        "total_steps": 200,
        "micro_batch_size": 2,
        "grad_accum_steps": 1,
        "seq_len": 64,
        "muon_lr": 0.02,
        "adamw_lr": 0.001,
        "min_lr": 0.0001,
        "warmup_steps": 2,
        "stable_steps": 2,
        "checkpoint_interval": 0,
        "val_interval": 0,
        "early_stopping": False,
        "device": "cuda:0",
    }
    payload.update(overrides)
    return payload


def run_session(workdir: Path, db_path: Path, payload: dict, stop_after_metrics: int = 0, timeout: float = 420.0):
    """Start the worker, optionally stop it mid-run, and collect its event stream."""
    log_path = workdir / f"stderr_{int(time.time() * 1000)}.log"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [
        str(PYTHON), "-u", "-m", "src.worker.standalone_runner",
        "--payload", json.dumps(payload),
        "--db-path", str(db_path),
    ]
    started = time.time()
    metrics = []
    events = []
    raw_lines = []
    with open(log_path, "w", encoding="utf-8", errors="replace") as err_file:
        proc = subprocess.Popen(
            cmd, cwd=str(workdir), env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=err_file, text=True, encoding="utf-8", errors="replace",
        )
        stop_sent = False
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                raw_lines.append(line)
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event_type = data.get("event_type") or data.get("event")
                if event_type == "STEP_COMPLETED":
                    metrics.append(data.get("payload", {}))
                if event_type:
                    events.append({"type": event_type, "payload": data.get("payload", {})})
                if stop_after_metrics and not stop_sent and len(metrics) >= stop_after_metrics:
                    assert proc.stdin is not None
                    proc.stdin.write(json.dumps({"command_type": "STOP_RUN"}) + "\n")
                    proc.stdin.flush()
                    stop_sent = True
                if time.time() - started > timeout:
                    proc.kill()
                    raise TimeoutError(f"worker did not exit within {timeout:.0f}s")
            proc.wait(timeout=60)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=30)
    (workdir / f"stdout_{int(started)}.log").write_text("\n".join(raw_lines), encoding="utf-8")
    return {
        "code": proc.returncode,
        "metrics": metrics,
        "events": events,
        "elapsed": round(time.time() - started, 1),
        "log": log_path,
    }


def db_row(db_path: Path, run_id: str):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None


def latest_run_id(db_path: Path):
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT id FROM runs ORDER BY created_at DESC LIMIT 1").fetchone()
        return row[0] if row else None


def alerts(db_path: Path, run_id: str):
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT category, level FROM alerts WHERE run_id = ?", (run_id,)).fetchall()
        return rows


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="llmmaker-e2e-"))
    db_path = tmp / "e2e.db"
    failures = []

    def check(condition, message):
        print(("PASS  " if condition else "FAIL  ") + message)
        if not condition:
            failures.append(message)

    try:
        print(f"workdir: {tmp}")
        first = run_session(tmp, db_path, base_payload(), stop_after_metrics=4)
        print(f"session 1: exit={first['code']} metrics={len(first['metrics'])} elapsed={first['elapsed']}s")
        check(first["code"] == 0, "stopped worker exits with code 0")
        check(len(first["metrics"]) >= 4, "worker streamed metrics before the stop")
        check(any(m.get("tflops", 0) > 0 for m in first["metrics"]), "throughput metric is populated")

        stopped_types = [e["type"] for e in first["events"]]
        check("RUN_STOPPED" in stopped_types, "RUN_STOPPED event was emitted")

        run1 = latest_run_id(db_path)
        row1 = db_row(db_path, run1) if run1 else None
        check(bool(row1), "run row exists")
        if row1:
            check(row1["status"] == "STOPPED", f"run status persisted as STOPPED (got {row1['status']})")
            check(row1["final_loss"] is not None, "final_loss persisted on stop")
            if first["metrics"]:
                check(row1["total_steps"] == first["metrics"][-1]["step"], "run step matches the last streamed step")

        snapshots = sorted((tmp / "checkpoints").glob("stopped_snapshot*.pt"))
        check(len(snapshots) == 1, f"exactly one resumable stop snapshot (got {len(snapshots)})")
        snapshot = snapshots[-1] if snapshots else None

        if snapshot is not None and row1 is not None:
            stop_step = int(row1["total_steps"])
            second = run_session(
                tmp,
                db_path,
                base_payload(
                    total_steps=stop_step + 6,
                    start_fresh=False,
                    resume_checkpoint=str(snapshot),
                    val_interval=3,
                ),
            )
            print(f"session 2: exit={second['code']} metrics={len(second['metrics'])} elapsed={second['elapsed']}s")
            check(second["code"] == 0, "resumed worker exits with code 0")
            resumed_steps = [m.get("step", 0) for m in second["metrics"]]
            check(bool(resumed_steps) and min(resumed_steps) > stop_step, "resume continues past the stopped step")
            check(any("val_loss" in m for m in second["metrics"]), "validation ran after resume")
            run2 = latest_run_id(db_path)
            row2 = db_row(db_path, run2) if run2 else None
            check(bool(row2) and row2["status"] == "COMPLETED", "resumed run completes")

            corrupt = tmp / "corrupt.pt"
            corrupt.write_bytes(b"not a checkpoint")
            third = run_session(
                tmp,
                db_path,
                base_payload(total_steps=20, start_fresh=False, resume_checkpoint=str(corrupt)),
            )
            print(f"session 3: exit={third['code']} metrics={len(third['metrics'])} elapsed={third['elapsed']}s")
            third_types = [e["type"] for e in third["events"]]
            check("RUN_FAILED" in third_types, "corrupted resume emits RUN_FAILED")
            run3 = latest_run_id(db_path)
            row3 = db_row(db_path, run3) if run3 else None
            check(bool(row3) and row3["status"] == "FAILED", "corrupted resume marks the run FAILED")
            check(any(category == "CHECKPOINT_LOAD_FAILED" for category, _ in alerts(db_path, run3)), "corrupted resume logs an alert")
            check(len(third.get("metrics", [])) == 0, "no training steps run with unreadable weights")

        for log in tmp.glob("stderr_*.log"):
            text = log.read_text(encoding="utf-8", errors="replace")
            for marker in ("Traceback", "CUDA error", "UnpicklingError"):
                if marker in text:
                    print(f"NOTE  {log.name} contains {marker!r}")
    finally:
        if failures:
            print(f"workdir kept for inspection: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print("")
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        return 1
    print("All lifecycle checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())