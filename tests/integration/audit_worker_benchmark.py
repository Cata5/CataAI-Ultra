"""
Measures real training throughput of the isolated worker on a small RoCanon model.

Usage:  .venv\\Scripts\\python.exe tests\\integration\\audit_worker_benchmark.py [steps]
"""

from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 120


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="llmmaker-bench-"))
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
        "total_steps": STEPS,
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
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [
        str(PYTHON), "-u", "-m", "src.worker.standalone_runner",
        "--payload", json.dumps(payload),
        "--db-path", str(tmp / "bench.db"),
    ]
    metrics = []
    started = time.time()
    with open(tmp / "stderr.log", "w", encoding="utf-8", errors="replace") as err:
        proc = subprocess.Popen(cmd, cwd=str(tmp), env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=err, text=True, encoding="utf-8", errors="replace")
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("event_type") == "STEP_COMPLETED":
                    metrics.append(data.get("payload", {}))
            proc.wait(timeout=120)
        finally:
            if proc.poll() is None:
                proc.kill()
    elapsed = time.time() - started

    if not metrics:
        print("No metrics collected.")
        print((tmp / "stderr.log").read_text(encoding="utf-8", errors="replace")[-3000:])
        return 1

    warm = metrics[5:] or metrics
    step_ms = [m.get("step_time_ms", 0) for m in warm]
    tok_s = [m.get("tok_per_sec", 0) for m in warm]
    tflops = [m.get("tflops", 0) for m in warm]
    vram = [m.get("vram_used_gb", 0) for m in metrics]
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f"steps: {len(metrics)} (warm {len(warm)}) in {elapsed:.1f}s wall")
    print(f"step_time_ms: median {statistics.median(step_ms):.2f}  min {min(step_ms):.2f}  max {max(step_ms):.2f}")
    print(f"tokens_per_sec: median {statistics.median(tok_s):.1f}  max {max(tok_s):.1f}")
    print(f"tflops_estimate: median {statistics.median(tflops):.3f}")
    print(f"vram_peak_gb: {max(vram):.2f}")
    print(f"final_loss: {metrics[-1].get('loss')}")
    stderr_text = (tmp / "stderr.log").read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Trainable parameters: (\d+)", stderr_text)
    print(f"trainable_parameters: {match.group(1) if match else 'unknown'}")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())