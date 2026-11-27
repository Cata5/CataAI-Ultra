"""Persistent progress reporting for dataset cleaning, tokenization, and audits.

The desktop shell and the browser dashboard both poll these small JSON files so
progress survives route changes and page reloads while a Python worker keeps
running. Writes are atomic and best-effort: progress must never break the
actual dataset operation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..utils.paths import cache_dir

_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
ProgressCallback = Callable[[str, int, int | None, str], None]


def normalize_job_id(job_id: str | None) -> str:
    raw = str(job_id or "").strip()
    if _JOB_ID_RE.match(raw):
        return raw
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:24]
    return f"job_{digest}"


def job_status_path(job_id: str | None) -> Path:
    return cache_dir() / "dataset_jobs" / f"{normalize_job_id(job_id)}.json"


def read_job_status(job_id: str | None) -> dict[str, Any] | None:
    path = job_status_path(job_id)
    try:
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


class DatasetProgress:
    """Write a small, atomically replaced status file for one dataset job."""

    def __init__(
        self,
        job_id: str | None,
        kind: str,
        dataset_id: str | None = None,
        dataset_name: str | None = None,
    ) -> None:
        self.job_id = normalize_job_id(job_id)
        self.kind = kind
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name or dataset_id
        self.path = job_status_path(self.job_id)
        self._last_write = 0.0
        self._payload: dict[str, Any] = {}
        self.update("queued", 0, None, "Queued", force=True)

    def update(
        self,
        phase: str,
        current: int = 0,
        total: int | None = None,
        message: str = "",
        *,
        percent: float | None = None,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        if (
            not force
            and self._payload.get("phase") != "queued"
            and now - self._last_write < 0.2
        ):
            return
        if percent is None and total and total > 0:
            percent = min(100.0, max(0.0, current / total * 100.0))
        payload = {
            "job_id": self.job_id,
            "kind": self.kind,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "status": "running",
            "phase": phase,
            "message": message,
            "current": int(current or 0),
            "total": int(total) if total is not None else None,
            "percent": round(float(percent), 2) if percent is not None else None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._payload = payload
        self._write(payload)
        self._last_write = now

    def complete(self, result: dict[str, Any]) -> None:
        payload = {
            **self._payload,
            "status": "completed",
            "phase": "completed",
            "message": "Completed",
            "percent": 100.0,
            "result": result,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._payload = payload
        self._write(payload)

    def fail(self, error: str) -> None:
        payload = {
            **self._payload,
            "status": "failed",
            "phase": "failed",
            "message": str(error),
            "error": str(error),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._payload = payload
        self._write(payload)

    def __call__(self, phase: str, current: int = 0, total: int | None = None, message: str = "") -> None:
        self.update(phase, current, total, message)

    def _write(self, payload: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_name(f"{self.path.name}.tmp-{os.getpid()}")
            temp.write_text(
                json.dumps(payload, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            os.replace(temp, self.path)
        except OSError:
            pass