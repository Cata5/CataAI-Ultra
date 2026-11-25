"""Persistent hyperparameter sweeps and the automatic run queue.

Trial generation is deterministic. The coordinator listens to worker events and
launches the next trial only when the worker reaches a terminal state. Sweeps
are stored as JSON so this feature needs no database or Rust migration.
"""

from __future__ import annotations

import copy
import itertools
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from ..utils.paths import data_dir

logger = logging.getLogger(__name__)

MAX_TRIALS = 50
ALLOWED_AXES = {
    "learning_rate",
    "peak_lr",
    "muon_lr",
    "micro_batch_size",
    "grad_accum_steps",
    "warmup_steps",
    "warmup_steps_pct",
    "stable_steps_pct",
    "weight_decay",
    "dropout",
    "seq_len",
    "optimizer",
    "scheduler",
}
TERMINAL_TRIAL_STATES = {"completed", "failed", "stopped", "pruned"}


def _now() -> float:
    return time.time()


def _as_number(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return text
    return value


class SweepError(ValueError):
    pass


class SweepManager:
    def __init__(self, worker_mgr: Any, root: Optional[Path] = None) -> None:
        self.worker_mgr = worker_mgr
        self.root = Path(root or (data_dir() / "sweeps"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.store_path = self.root / "sweeps.json"
        self._lock = threading.RLock()
        self._sweeps: Dict[str, Dict[str, Any]] = {}
        self._load()
        subscribe = getattr(worker_mgr, "subscribe", None)
        if callable(subscribe):
            subscribe(self.on_worker_event)

    # ---------- persistence ----------
    def _load(self) -> None:
        if not self.store_path.is_file():
            return
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._sweeps = {str(k): v for k, v in data.items() if isinstance(v, dict)}
        except Exception as exc:
            logger.warning("Could not read sweep store: %s", exc)
        changed = False
        for sweep in self._sweeps.values():
            if sweep.get("status") in ("running", "queued", "waiting"):
                sweep["status"] = "interrupted"
                changed = True
                for trial in sweep.get("trials", []):
                    if trial.get("status") in ("running", "queued"):
                        trial["status"] = "interrupted"
        if changed:
            self._save()

    def _save(self) -> None:
        tmp = self.store_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._sweeps, indent=2), encoding="utf-8")
        tmp.replace(self.store_path)

    # ---------- creation ----------
    def _expand_grid(self, axes: Dict[str, Iterable[Any]], max_trials: int) -> List[Dict[str, Any]]:
        if not isinstance(axes, dict) or not axes:
            raise SweepError("axes must contain at least one hyperparameter list")
        clean: Dict[str, List[Any]] = {}
        for key, values in axes.items():
            if key not in ALLOWED_AXES:
                raise SweepError(f"Axis {key} is not allowed; choose {sorted(ALLOWED_AXES)}")
            if not isinstance(values, (list, tuple)) or not values:
                raise SweepError(f"Axis {key} must be a non-empty list")
            clean[key] = [_as_number(value) for value in values]
        total = 1
        for values in clean.values():
            total *= len(values)
        if total > max_trials:
            raise SweepError(f"Grid expands to {total} trials, above max_trials={max_trials}")
        keys = list(clean)
        combos = []
        for values in itertools.product(*(clean[key] for key in keys)):
            combos.append(dict(zip(keys, values)))
        return combos

    def create_sweep(
        self,
        name: str,
        base_payload: Dict[str, Any],
        axes: Dict[str, Iterable[Any]],
        *,
        max_trials: int = MAX_TRIALS,
        early_stop: bool = True,
        early_stop_ratio: float = 0.05,
        workspace_id: str = "default",
    ) -> Dict[str, Any]:
        if not isinstance(base_payload, dict) or not base_payload:
            raise SweepError("base_payload is required")
        for required in ("network_id", "dataset_id", "total_steps"):
            if not base_payload.get(required):
                raise SweepError(f"base_payload.{required} is required")
        try:
            max_trials = max(1, min(int(max_trials), MAX_TRIALS))
        except (TypeError, ValueError):
            raise SweepError("max_trials must be an integer")
        combos = self._expand_grid(axes, max_trials)
        sweep_id = f"sweep_{uuid4().hex[:10]}"
        now = _now()
        base = copy.deepcopy(base_payload)
        base["workspace_id"] = workspace_id
        base["start_fresh"] = True
        sweep = {
            "id": sweep_id,
            "name": str(name or f"Sweep {sweep_id[-5:]}"),
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "max_trials": max_trials,
            "early_stop": bool(early_stop),
            "early_stop_ratio": max(0.0, float(early_stop_ratio)),
            "axes": {key: list(values) for key, values in axes.items()},
            "base_payload": base,
            "best_trial": None,
            "active_trial": None,
            "trials": [],
        }
        for index, combo in enumerate(combos):
            payload = copy.deepcopy(base)
            payload.update(combo)
            total_steps = int(payload.get("total_steps") or 0)
            sweep["trials"].append({
                "index": index,
                "run_id": f"{sweep_id}_t{index:02d}",
                "params": combo,
                "status": "queued",
                "val_loss": None,
                "tokens_per_sec": None,
                "vram_gb": None,
                "steps": 0,
                "total_steps": total_steps,
                "started_at": None,
                "finished_at": None,
                "error": None,
                "payload": payload,
            })
        with self._lock:
            self._sweeps[sweep_id] = sweep
        self._save()
        self._maybe_start_next(sweep_id)
        with self._lock:
            return self._public(sweep_id)

    # ---------- read ----------
    def _public(self, sweep_id: str, include_payload: bool = False) -> Dict[str, Any]:
        sweep = copy.deepcopy(self._sweeps[sweep_id])
        if not include_payload:
            sweep.pop("base_payload", None)
            for trial in sweep.get("trials", []):
                trial.pop("payload", None)
        return sweep

    def list_sweeps(self) -> List[Dict[str, Any]]:
        with self._lock:
            ids = sorted(self._sweeps, key=lambda key: self._sweeps[key].get("created_at", 0), reverse=True)
            return [self._public(sweep_id) for sweep_id in ids]

    def get_sweep(self, sweep_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if sweep_id not in self._sweeps:
                return None
            return self._public(sweep_id)

    def start_sweep(self, sweep_id: str) -> bool:
        """Explicitly start (or resume) a sweep. Reads must never start training."""
        with self._lock:
            sweep = self._sweeps.get(sweep_id)
            if not sweep:
                return False
            if sweep.get("active_trial") is not None:
                return True
            if sweep.get("status") in ("completed", "failed", "stopped"):
                raise SweepError("Sweep is already finished; create a new sweep")
            if sweep.get("status") == "interrupted":
                for trial in sweep.get("trials", []):
                    if trial.get("status") == "interrupted":
                        trial["status"] = "queued"
            sweep["status"] = "queued"
            sweep["updated_at"] = _now()
        self._save()
        return self._maybe_start_next(sweep_id)

    def delete_sweep(self, sweep_id: str) -> bool:
        with self._lock:
            sweep = self._sweeps.get(sweep_id)
            if not sweep:
                return False
            if sweep.get("status") == "running":
                raise SweepError("Stop the sweep before deleting it")
            del self._sweeps[sweep_id]
        self._save()
        return True

    # ---------- lifecycle ----------
    def stop_sweep(self, sweep_id: str) -> bool:
        with self._lock:
            sweep = self._sweeps.get(sweep_id)
            if not sweep:
                return False
            active = sweep.get("active_trial")
            sweep["status"] = "stopped"
            sweep["updated_at"] = _now()
            if active is not None:
                trial = sweep["trials"][active]
                if trial.get("status") == "running":
                    trial["status"] = "stopped"
                    trial["finished_at"] = _now()
            for trial in sweep["trials"]:
                if trial.get("status") == "queued":
                    trial["status"] = "stopped"
            sweep["active_trial"] = None
        self._save()
        if active is not None:
            self.worker_mgr.stop_run()
        return True

    def _maybe_start_next(self, sweep_id: str) -> bool:
        from ..worker.protocol import RunState

        with self._lock:
            sweep = self._sweeps.get(sweep_id)
            if not sweep or sweep.get("status") not in ("queued", "waiting", "running"):
                return False
            if sweep.get("active_trial") is not None:
                return False
            state = getattr(self.worker_mgr, "current_state", RunState.IDLE)
            if state in (RunState.RUNNING, RunState.STARTING, RunState.PAUSED, RunState.STOPPING):
                sweep["status"] = "waiting"
                # A stop is asynchronous: retry once the worker has actually
                # reached IDLE/STOPPED instead of leaving the sweep parked.
                if state == RunState.STOPPING:
                    timer = threading.Timer(0.5, self._maybe_start_next, args=(sweep_id,))
                    timer.daemon = True
                    timer.start()
                return False
            next_index = next(
                (index for index, trial in enumerate(sweep["trials"]) if trial.get("status") == "queued"),
                None,
            )
            if next_index is None:
                sweep["status"] = "completed" if all(t.get("status") == "completed" for t in sweep["trials"]) else "stopped"
                return False
            trial = sweep["trials"][next_index]
            trial["status"] = "running"
            trial["started_at"] = _now()
            sweep["active_trial"] = next_index
            sweep["status"] = "running"
            sweep["updated_at"] = _now()
            payload = copy.deepcopy(trial["payload"])
            payload["run_id"] = trial["run_id"]
            payload["sweep_id"] = sweep_id
        try:
            started = bool(self.worker_mgr.start_run(payload))
        except Exception as exc:
            logger.exception("Could not start sweep trial %s", trial.get("run_id"))
            trial["error"] = str(exc)
            started = False
        with self._lock:
            if not started:
                trial["status"] = "queued"
                sweep["active_trial"] = None
                sweep["status"] = "waiting"
            self._save()
        return started

    # ---------- worker events ----------
    def on_worker_event(self, event: Any) -> None:
        try:
            self._handle_event(event)
        except Exception:
            logger.exception("Sweep coordinator failed to handle worker event")

    def _sweep_for_run(self, run_id: Optional[str]) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        if not run_id:
            return None, None
        for sweep_id, sweep in self._sweeps.items():
            active = sweep.get("active_trial")
            if active is None:
                continue
            trial = sweep["trials"][active]
            if trial.get("run_id") == run_id:
                return sweep_id, trial
        return None, None

    def _handle_event(self, event: Any) -> None:
        from ..worker.protocol import EventType

        event_type = getattr(event, "event_type", None)
        run_id = getattr(event, "run_id", None)
        payload = getattr(event, "payload", {}) or {}
        with self._lock:
            sweep_id, trial = self._sweep_for_run(run_id)
            if not sweep_id or not trial:
                return
            sweep = self._sweeps[sweep_id]
            if event_type in (EventType.METRICS_UPDATED, EventType.STEP_COMPLETED, EventType.VALIDATION_COMPLETED):
                step = int(payload.get("step") or trial.get("steps") or 0)
                trial["steps"] = step
                loss = payload.get("loss")
                if isinstance(loss, (int, float)):
                    trial["last_loss"] = float(loss)
                if payload.get("val_loss") is not None:
                    try:
                        trial["val_loss"] = float(payload["val_loss"])
                    except (TypeError, ValueError):
                        pass
                if payload.get("tokens_per_sec") is not None:
                    trial["tokens_per_sec"] = payload.get("tokens_per_sec")
                if payload.get("vram_gb") is not None:
                    trial["vram_gb"] = payload.get("vram_gb")
                if sweep.get("early_stop") and not trial.get("prune_requested"):
                    # Compare validation to validation. Comparing a 15%-step
                    # training loss with a finished trial's validation loss
                    # prunes healthy trials; this only fires after the worker
                    # has actually reported a validation loss.
                    best_loss = self._best_loss(sweep)
                    total_steps = int(trial.get("total_steps") or 0)
                    min_steps = max(20, int(total_steps * 0.15)) if total_steps else 20
                    current = trial.get("val_loss")
                    if (
                        best_loss is not None
                        and isinstance(current, (int, float))
                        and step >= min_steps
                        and current > best_loss * (1.0 + float(sweep.get("early_stop_ratio", 0.05)))
                    ):
                        trial["prune_requested"] = True
                        trial["pruned"] = True
                        self._save()
                        threading.Thread(target=self.worker_mgr.stop_run, daemon=True).start()
                return
            if event_type not in (EventType.RUN_STOPPED, EventType.RUN_COMPLETED, EventType.RUN_FAILED):
                return
            if event_type == EventType.RUN_COMPLETED:
                trial["status"] = "completed"
            elif event_type == EventType.RUN_FAILED:
                trial["status"] = "failed"
                trial["error"] = str(payload.get("error") or "worker reported failure")
            else:
                trial["status"] = "pruned" if trial.get("pruned") else "stopped"
            trial["finished_at"] = _now()
            trial["steps"] = int(payload.get("step") or trial.get("steps") or 0)
            if payload.get("tokens_per_sec") is not None:
                trial["tokens_per_sec"] = payload.get("tokens_per_sec")
            if payload.get("vram_gb") is not None:
                trial["vram_gb"] = payload.get("vram_gb")
            if payload.get("val_loss") is not None:
                trial["val_loss"] = float(payload["val_loss"])
            sweep["active_trial"] = None
            sweep["updated_at"] = _now()
            self._save()
        self._refresh_trial_metrics(sweep_id, trial)
        self._update_best(sweep_id)
        self._maybe_start_next(sweep_id)

    def _refresh_trial_metrics(self, sweep_id: str, trial: Dict[str, Any]) -> None:
        db = getattr(self.worker_mgr, "db", None)
        if db is None:
            return
        try:
            validation = db.get_latest_validation(trial["run_id"])
            if validation and validation.get("val_loss") is not None:
                trial["val_loss"] = float(validation["val_loss"])
            run = db.get_run(trial["run_id"])
            if run:
                if trial.get("val_loss") is None and run.get("val_loss") is not None:
                    trial["val_loss"] = float(run["val_loss"])
                if trial.get("tokens_per_sec") is None and run.get("tokens_per_sec") is not None:
                    trial["tokens_per_sec"] = run.get("tokens_per_sec")
                if trial.get("vram_gb") is None and run.get("vram_gb") is not None:
                    trial["vram_gb"] = run.get("vram_gb")
        except Exception as exc:
            logger.debug("Could not refresh trial metrics: %s", exc)

    def _best_loss(self, sweep: Dict[str, Any]) -> Optional[float]:
        losses = [
            float(trial["val_loss"])
            for trial in sweep.get("trials", [])
            if trial.get("val_loss") is not None and trial.get("status") == "completed"
        ]
        return min(losses) if losses else None

    def _update_best(self, sweep_id: str) -> None:
        with self._lock:
            sweep = self._sweeps.get(sweep_id)
            if not sweep:
                return
            ranked = [
                trial for trial in sweep["trials"]
                if trial.get("status") == "completed" and trial.get("val_loss") is not None
            ]
            ranked.sort(key=lambda t: (float(t["val_loss"]), -float(t.get("tokens_per_sec") or 0)))
            sweep["best_trial"] = ranked[0]["index"] if ranked else None
            if sweep.get("active_trial") is None and all(t.get("status") in TERMINAL_TRIAL_STATES for t in sweep["trials"]):
                sweep["status"] = "completed" if any(t.get("status") == "completed" for t in sweep["trials"]) else "failed"
            self._save()
