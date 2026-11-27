"""
Database manager for RoCanon-48.5M Studio.
Provides SQLite persistence for runs, metrics, checkpoints, alerts, pipelines, datasets, and benchmark results.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..security.secrets import SECRET_KEYS, REDACTED, SecretStore, SecretsKeyError
from ..utils.paths import default_db_path
from .migrations import apply_migrations

logger = logging.getLogger(__name__)


class DatabaseIntegrityError(RuntimeError):
    """Raised when SQLite reports a corrupt or unreadable database."""


# One writable database per installation. LLMMAKER_DB_PATH / LLMMAKER_DATA_DIR
# override it; source checkouts keep their historical project-root location.
DEFAULT_DB_PATH = default_db_path()


class StudioDatabase:
    """Thread-safe SQLite database manager for training metadata and metrics."""

    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        secret_key_path: Path | str | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        key_path = Path(secret_key_path) if secret_key_path else self.db_path.parent / ".secrets.key"
        if not key_path.exists() and self._has_encrypted_secrets():
            raise SecretsKeyError(
                f"The encryption key for {self.db_path} is missing ({key_path}). "
                "Restore the matching .secrets.key file; encrypted credentials cannot be "
                "recovered without it. A new key is not generated to avoid silently losing them."
            )
        self._secret_store = SecretStore(key_path)
        self._lock = threading.Lock()
        self._init_db()

    def _has_encrypted_secrets(self) -> bool:
        """Checks for encrypted rows without creating a replacement key."""
        if not self.db_path.exists():
            return False
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=5.0)
            try:
                rows = conn.execute(
                    "SELECT value_json FROM settings WHERE key IN (?, ?)",
                    tuple(SECRET_KEYS),
                ).fetchall()
            finally:
                conn.close()
        except sqlite3.Error:
            return False
        for (raw,) in rows:
            try:
                value = json.loads(raw)
            except Exception:
                continue
            if SecretStore.is_encrypted(value):
                return True
        return False

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        try:
            conn = self._get_connection()
        except sqlite3.DatabaseError as exc:
            raise DatabaseIntegrityError(
                f"Cannot open database {self.db_path}: {exc}. Restore a backup from the backups/ folder."
            ) from exc
        with self._lock, conn:
            # Refuse to migrate a database SQLite already considers damaged: a
            # backup of a corrupt file is not a recovery point.
            try:
                row = conn.execute("PRAGMA quick_check").fetchone()
            except sqlite3.DatabaseError as exc:
                raise DatabaseIntegrityError(
                    f"Cannot read database {self.db_path}: {exc}. Restore a backup from the backups/ folder."
                ) from exc
            result = str(row[0]) if row else "unknown"
            if result.lower() != "ok":
                raise DatabaseIntegrityError(
                    f"Database integrity check failed for {self.db_path}: {result}. "
                    "Stop the app and restore a snapshot from backups/."
                )

            applied = apply_migrations(conn, self.db_path)
            if applied:
                logger.info("Database %s upgraded to schema version %s", self.db_path, applied[-1])

            # Ensure default workspace exists
            cur = conn.execute("SELECT id FROM workspaces WHERE id = 'default'")
            if not cur.fetchone():
                now = time.time()
                conn.execute(
                    """
                    INSERT INTO workspaces (id, name, description, root_path, is_active, config_json, created_at, updated_at)
                    VALUES ('default', 'Default Workspace', 'Primary default experimentation workspace', '.', 1, '{}', ?, ?)
                    """,
                    (now, now),
                )

            self._encrypt_legacy_secrets(conn)

    # ------------------- Runs -------------------
    def create_run(
        self,
        run_id: str,
        name: str,
        current_phase: str = "pretrain",
        dataset_source: str = "",
        config: Optional[Dict[str, Any]] = None,
        workspace_id: str = "default",
    ) -> None:
        now = time.time()
        cfg_str = json.dumps(config or {})
        with self._lock, self._get_connection() as conn:
            # A duplicate id is a caller error, not a request to replace a run.
            # INSERT OR REPLACE silently deleted the old row while leaving its
            # metrics orphaned; reject the collision so the caller can pick a
            # new id and the existing history stays intact.
            try:
                conn.execute(
                    """
                    INSERT INTO runs
                    (id, name, status, current_phase, dataset_source, total_steps, total_tokens, config_json, workspace_id, created_at, updated_at)
                    VALUES (?, ?, 'STARTING', ?, ?, 0, 0, ?, ?, ?, ?)
                    """,
                    (run_id, name, current_phase, dataset_source, cfg_str, workspace_id, now, now),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Run id '{run_id}' already exists; refusing to overwrite its history.") from exc

    def update_run_status(
        self,
        run_id: str,
        status: str,
        step: Optional[int] = None,
        tokens: Optional[int] = None,
        best_val_loss: Optional[float] = None,
        final_loss: Optional[float] = None,
    ) -> None:
        now = time.time()
        updates = ["status = ?", "updated_at = ?"]
        params: List[Any] = [status, now]
        if step is not None:
            updates.append("total_steps = ?")
            params.append(step)
        if tokens is not None:
            updates.append("total_tokens = ?")
            params.append(tokens)
        if best_val_loss is not None:
            updates.append("best_val_loss = ?")
            params.append(best_val_loss)
        if final_loss is not None:
            updates.append("final_loss = ?")
            params.append(final_loss)
        params.append(run_id)

        sql = f"UPDATE runs SET {', '.join(updates)} WHERE id = ?"
        with self._lock, self._get_connection() as conn:
            conn.execute(sql, params)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
            row = cur.fetchone()
            if not row:
                return None
            d = dict(row)
            if d.get("config_json"):
                d["config"] = json.loads(d["config_json"])
            return d

    def list_runs(self, limit: int = 50, workspace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            if workspace_id:
                cur = conn.execute(
                    "SELECT * FROM runs WHERE workspace_id = ? ORDER BY created_at DESC LIMIT ?",
                    (workspace_id, limit),
                )
            else:
                cur = conn.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
            result = []
            for r in rows:
                d = dict(r)
                if d.get("config_json"):
                    try:
                        d["config"] = json.loads(d["config_json"])
                    except Exception:
                        d["config"] = {}
                result.append(d)
            return result

    # ------------------- Metrics -------------------
    def log_metric(self, run_id: str, data: Dict[str, Any]) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO metrics
                (run_id, step, tokens, train_loss, val_loss, generalization_gap,
                 grad_norm, muon_lr, adamw_lr, vram_used_gb, tok_per_sec, tflops, timestamp,
                 current_epoch, epoch_pct, tokens_per_epoch, tokens_in_epoch, step_time_ms,
                 duration_mode, target_epochs, sched_phase, sched_factor, eval_interval, next_val_step)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    data.get("step", 0),
                    data.get("tokens", 0),
                    data.get("train_loss", 0.0),
                    data.get("val_loss"),
                    data.get("generalization_gap"),
                    data.get("grad_norm", 0.0),
                    data.get("muon_lr", 0.0),
                    data.get("adamw_lr", 0.0),
                    data.get("vram_used_gb", 0.0),
                    data.get("tok_per_sec", 0.0),
                    data.get("tflops", 0.0),
                    data.get("timestamp", time.time()),
                    data.get("current_epoch"),
                    data.get("epoch_pct"),
                    data.get("tokens_per_epoch", 0),
                    data.get("tokens_in_epoch", 0),
                    data.get("step_time_ms", 0.0),
                    data.get("duration_mode"),
                    data.get("target_epochs"),
                    data.get("sched_phase"),
                    data.get("sched_factor"),
                    data.get("eval_interval"),
                    data.get("next_val_step"),
                ),
            )

    def log_metrics_batch(self, run_id: str, batch: List[Dict[str, Any]]) -> None:
        if not batch:
            return
        rows = [
            (
                run_id,
                d.get("step", 0),
                d.get("tokens", 0),
                d.get("train_loss", 0.0),
                d.get("val_loss"),
                d.get("generalization_gap"),
                d.get("grad_norm", 0.0),
                d.get("muon_lr", 0.0),
                d.get("adamw_lr", 0.0),
                d.get("vram_used_gb", 0.0),
                d.get("tok_per_sec", 0.0),
                d.get("tflops", 0.0),
                d.get("timestamp", time.time()),
                d.get("current_epoch"),
                d.get("epoch_pct"),
                d.get("tokens_per_epoch", 0),
                d.get("tokens_in_epoch", 0),
                d.get("step_time_ms", 0.0),
                d.get("duration_mode"),
                d.get("target_epochs"),
                d.get("sched_phase"),
                d.get("sched_factor"),
                d.get("eval_interval"),
                d.get("next_val_step"),
            )
            for d in batch
        ]
        with self._lock, self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO metrics
                (run_id, step, tokens, train_loss, val_loss, generalization_gap,
                 grad_norm, muon_lr, adamw_lr, vram_used_gb, tok_per_sec, tflops, timestamp,
                 current_epoch, epoch_pct, tokens_per_epoch, tokens_in_epoch, step_time_ms,
                 duration_mode, target_epochs, sched_phase, sched_factor, eval_interval, next_val_step)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def delete_metrics_after(self, run_id: str, step: int) -> int:
        """Remove metrics newer than a checkpoint before branching a resume.

        Continuing from an older checkpoint makes the old future points stale;
        keeping them would interleave two branches under the same run id.
        """
        with self._lock, self._get_connection() as conn:
            cur = conn.execute(
                "DELETE FROM metrics WHERE run_id = ? AND step > ?",
                (run_id, step),
            )
            return int(cur.rowcount or 0)

    def get_metrics(
        self,
        run_id: str,
        start_step: int = 0,
        end_step: Optional[int] = None,
        max_points: int = 5000,
    ) -> List[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            if end_step is not None:
                cur = conn.execute(
                    "SELECT * FROM metrics WHERE run_id = ? AND step >= ? AND step <= ? ORDER BY step ASC LIMIT ?",
                    (run_id, start_step, end_step, max_points),
                )
            else:
                cur = conn.execute(
                    "SELECT * FROM metrics WHERE run_id = ? AND step >= ? ORDER BY step ASC LIMIT ?",
                    (run_id, start_step, max_points),
                )
            return [dict(r) for r in cur.fetchall()]

    def get_metrics_history(self, run_id: str, limit: int = 3000) -> List[Dict[str, Any]]:
        return self.get_metrics(run_id=run_id, start_step=0, max_points=limit)

    def get_latest_metric(self, run_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            if run_id:
                cur = conn.execute(
                    "SELECT * FROM metrics WHERE run_id = ? ORDER BY step DESC, id DESC LIMIT 1",
                    (run_id,),
                )
            else:
                cur = conn.execute(
                    "SELECT * FROM metrics ORDER BY id DESC LIMIT 1"
                )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_latest_validation(self, run_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            if run_id:
                cur = conn.execute(
                    "SELECT val_loss, step, timestamp FROM metrics WHERE run_id = ? AND val_loss IS NOT NULL ORDER BY step DESC, id DESC LIMIT 1",
                    (run_id,),
                )
            else:
                cur = conn.execute(
                    "SELECT val_loss, step, timestamp FROM metrics WHERE val_loss IS NOT NULL ORDER BY id DESC LIMIT 1"
                )
            row = cur.fetchone()
            if row:
                return {"val_loss": row["val_loss"], "step": row["step"], "timestamp": row["timestamp"]}
            return None

    # ------------------- Checkpoints -------------------
    def log_checkpoint(
        self,
        checkpoint_id: str,
        run_id: Optional[str],
        step: int,
        tokens: int,
        train_loss: Optional[float],
        val_loss: Optional[float],
        file_path: str,
        file_size_bytes: int,
        phase: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        workspace_id: str = "default",
    ) -> None:
        now = time.time()
        meta_str = json.dumps(metadata or {})
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints
                (id, run_id, step, tokens, train_loss, val_loss, file_path, file_size_bytes, phase, metadata_json, workspace_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    run_id,
                    step,
                    tokens,
                    train_loss,
                    val_loss,
                    file_path,
                    file_size_bytes,
                    phase,
                    meta_str,
                    workspace_id,
                    now,
                ),
            )

    def list_checkpoints(
        self, run_id: Optional[str] = None, limit: int = 100, workspace_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            if run_id:
                cur = conn.execute(
                    "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY step DESC LIMIT ?",
                    (run_id, limit),
                )
            elif workspace_id:
                cur = conn.execute(
                    "SELECT * FROM checkpoints WHERE workspace_id = ? ORDER BY created_at DESC LIMIT ?",
                    (workspace_id, limit),
                )
            else:
                cur = conn.execute(
                    "SELECT * FROM checkpoints ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
            results = []
            for r in cur.fetchall():
                d = dict(r)
                if d.get("metadata_json"):
                    try:
                        d["metadata"] = json.loads(d["metadata_json"])
                    except Exception:
                        d["metadata"] = {}
                results.append(d)
            return results

    def delete_checkpoint(self, checkpoint_id: str) -> Optional[str]:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT file_path FROM checkpoints WHERE id = ?", (checkpoint_id,))
            row = cur.fetchone()
            if not row:
                return None
            file_path = row["file_path"]
            conn.execute("DELETE FROM checkpoints WHERE id = ?", (checkpoint_id,))
            return file_path

    # ------------------- Alerts -------------------
    def log_alert(
        self,
        level: str,
        category: str,
        message: str,
        run_id: Optional[str] = None,
        step: int = 0,
    ) -> None:
        now = time.time()
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO alerts (run_id, level, category, message, step, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, level, category, message, step, now),
            )

        if level.upper() in ("WARNING", "CRITICAL", "ERROR"):
            try:
                from ..notifications.webhook import get_notifier
                get_notifier(db=self).notify_alert(
                    category=category,
                    level=level,
                    message=message,
                    step=step,
                    run_id=run_id,
                )
            except Exception:
                pass

    def list_alerts(self, run_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            if run_id:
                cur = conn.execute(
                    "SELECT * FROM alerts WHERE run_id = ? ORDER BY id DESC LIMIT ?",
                    (run_id, limit),
                )
            else:
                cur = conn.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    def get_metrics_history(self, run_id: str, limit: int = 3000) -> List[Dict[str, Any]]:
        return self.get_metrics(run_id=run_id, start_step=0, max_points=limit)

    def get_latest_metric(self, run_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            if run_id:
                cur = conn.execute(
                    "SELECT * FROM metrics WHERE run_id = ? ORDER BY step DESC, id DESC LIMIT 1",
                    (run_id,),
                )
            else:
                cur = conn.execute(
                    "SELECT * FROM metrics ORDER BY id DESC LIMIT 1"
                )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_latest_validation(self, run_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            if run_id:
                cur = conn.execute(
                    "SELECT val_loss, step, timestamp FROM metrics WHERE run_id = ? AND val_loss IS NOT NULL ORDER BY step DESC, id DESC LIMIT 1",
                    (run_id,),
                )
            else:
                cur = conn.execute(
                    "SELECT val_loss, step, timestamp FROM metrics WHERE val_loss IS NOT NULL ORDER BY id DESC LIMIT 1"
                )
            row = cur.fetchone()
            if row:
                return {"val_loss": row["val_loss"], "step": row["step"], "timestamp": row["timestamp"]}
            return None

    # ------------------- Pipelines -------------------
    def save_pipeline(
        self,
        pipeline_id: str,
        name: str,
        status: str,
        phases: List[Dict[str, Any]],
        current_phase_idx: int = 0,
        state: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = time.time()
        phases_str = json.dumps(phases)
        state_str = json.dumps(state or {})
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pipelines
                (id, name, status, current_phase_idx, phases_json, state_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (pipeline_id, name, status, current_phase_idx, phases_str, state_str, now, now),
            )

    def get_pipeline(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM pipelines WHERE id = ?", (pipeline_id,))
            row = cur.fetchone()
            if not row:
                return None
            d = dict(row)
            d["phases"] = json.loads(d["phases_json"])
            d["state"] = json.loads(d["state_json"]) if d.get("state_json") else {}
            return d

    def list_pipelines(self) -> List[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM pipelines ORDER BY updated_at DESC")
            results = []
            for r in cur.fetchall():
                d = dict(r)
                d["phases"] = json.loads(d["phases_json"])
                d["state"] = json.loads(d["state_json"]) if d.get("state_json") else {}
                results.append(d)
            return results

    # ------------------- Datasets -------------------
    def save_dataset(
        self,
        dataset_id: str,
        name: str,
        file_path: str,
        file_size_bytes: int,
        format_type: str,
        revision: str,
        token_count: int = 0,
        sample_count: int = 0,
        stats: Optional[Dict[str, Any]] = None,
        workspace_id: str = "default",
    ) -> None:
        now = time.time()
        stats_str = json.dumps(stats or {})
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO datasets
                (id, name, file_path, file_size_bytes, format, token_count, sample_count, revision, stats_json, workspace_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    name,
                    file_path,
                    file_size_bytes,
                    format_type,
                    token_count,
                    sample_count,
                    revision,
                    stats_str,
                    workspace_id,
                    now,
                ),
            )

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
            row = cur.fetchone()
            if not row:
                return None
            d = dict(row)
            d["stats"] = json.loads(d["stats_json"]) if d.get("stats_json") else {}
            return d

    def delete_dataset(self, dataset_id: str) -> bool:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            return cur.rowcount > 0

    def list_datasets(self, workspace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            if workspace_id:
                cur = conn.execute(
                    "SELECT * FROM datasets WHERE workspace_id = ? ORDER BY created_at DESC",
                    (workspace_id,),
                )
            else:
                cur = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC")
            results = []
            for r in cur.fetchall():
                d = dict(r)
                d["stats"] = json.loads(d["stats_json"]) if d.get("stats_json") else {}
                results.append(d)
            return results

    # ------------------- Benchmarks -------------------
    def log_benchmark_result(
        self,
        category: str,
        score: float,
        run_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        step: int = 0,
        results: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = time.time()
        res_str = json.dumps(results or {})
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO benchmarks
                (run_id, checkpoint_id, step, category, score, results_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, checkpoint_id, step, category, score, res_str, now),
            )

    def list_benchmarks(self, run_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            if run_id:
                cur = conn.execute(
                    "SELECT * FROM benchmarks WHERE run_id = ? ORDER BY id DESC LIMIT ?",
                    (run_id, limit),
                )
            else:
                cur = conn.execute("SELECT * FROM benchmarks ORDER BY id DESC LIMIT ?", (limit,))
            res = []
            for r in cur.fetchall():
                d = dict(r)
                d["results"] = json.loads(d["results_json"]) if d.get("results_json") else {}
                res.append(d)
            return res

    # ------------------- Persistent Settings -------------------
    def _encrypt_legacy_secrets(self, conn: sqlite3.Connection) -> None:
        """Upgrades plaintext credentials written by older builds in place."""
        now = time.time()
        rows = conn.execute("SELECT key, value_json FROM settings").fetchall()
        for row in rows:
            key = row["key"]
            if key not in SECRET_KEYS:
                continue
            try:
                value = json.loads(row["value_json"])
            except Exception:
                continue
            if not isinstance(value, str) or not value or SecretStore.is_encrypted(value):
                continue
            conn.execute(
                "UPDATE settings SET value_json = ?, updated_at = ? WHERE key = ?",
                (json.dumps(self._secret_store.encrypt(value)), now, key),
            )
            logger.info("Encrypted legacy plaintext setting %s", key)

    def save_settings(self, settings_dict: Dict[str, Any]) -> None:
        now = time.time()
        with self._lock, self._get_connection() as conn:
            for k, v in settings_dict.items():
                key = str(k)
                # Presence flags are derived on read and must never be persisted.
                if key.endswith("_configured"):
                    continue
                if key in SECRET_KEYS:
                    # The redacted sentinel means "keep the stored credential".
                    if v == REDACTED:
                        continue
                    text = "" if v is None else str(v)
                    v = self._secret_store.encrypt(text)
                conn.execute(
                    """
                    INSERT INTO settings (key, value_json, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, updated_at = excluded.updated_at
                    """,
                    (key, json.dumps(v), now),
                )

    def get_settings(self, redact: bool = False) -> Dict[str, Any]:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT key, value_json FROM settings")
            res: Dict[str, Any] = {}
            for row in cur.fetchall():
                key = row["key"]
                try:
                    value = json.loads(row["value_json"])
                except Exception:
                    value = row["value_json"]
                if key in SECRET_KEYS and isinstance(value, str) and value:
                    try:
                        value = self._secret_store.decrypt(value)
                    except SecretsKeyError as exc:
                        # Never present an unreadable credential as an empty one:
                        # the UI could then save over a value it never decrypted.
                        logger.error("Cannot decrypt setting %s: %s", key, exc)
                        raise
                res[key] = value
        if redact:
            from ..security.secrets import redact_settings
            return redact_settings(res)
        return res

    # ------------------- Workspaces -------------------
    def create_workspace(
        self,
        workspace_id: str,
        name: str,
        description: str = "",
        root_path: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = time.time()
        cfg_str = json.dumps(config or {})
        with self._lock, self._get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO workspaces (id, name, description, root_path, is_active, config_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 0, ?, ?, ?)
                    """,
                    (workspace_id, name, description, root_path, cfg_str, now, now),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"Workspace id '{workspace_id}' already exists; refusing to overwrite it."
                ) from exc
        return self.get_workspace(workspace_id)

    def list_workspaces(self) -> List[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM workspaces ORDER BY created_at ASC")
            res = []
            for r in cur.fetchall():
                d = dict(r)
                d["config"] = json.loads(d["config_json"]) if d.get("config_json") else {}
                d["is_active"] = bool(d.get("is_active", 0))
                # Add counts
                c_runs = conn.execute("SELECT COUNT(*) FROM runs WHERE workspace_id = ?", (d["id"],)).fetchone()[0]
                c_chk = conn.execute("SELECT COUNT(*) FROM checkpoints WHERE workspace_id = ?", (d["id"],)).fetchone()[0]
                c_data = conn.execute("SELECT COUNT(*) FROM datasets WHERE workspace_id = ?", (d["id"],)).fetchone()[0]
                d["run_count"] = c_runs
                d["checkpoint_count"] = c_chk
                d["dataset_count"] = c_data
                res.append(d)
            return res

    def get_workspace(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM workspaces WHERE id = ?", (workspace_id,))
            row = cur.fetchone()
            if not row:
                return None
            d = dict(row)
            d["config"] = json.loads(d["config_json"]) if d.get("config_json") else {}
            d["is_active"] = bool(d.get("is_active", 0))
            return d

    def get_active_workspace(self) -> Dict[str, Any]:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM workspaces WHERE is_active = 1 LIMIT 1")
            row = cur.fetchone()
            if row:
                d = dict(row)
                d["config"] = json.loads(d["config_json"]) if d.get("config_json") else {}
                d["is_active"] = True
                return d
            # Fallback to default
            cur = conn.execute("SELECT * FROM workspaces WHERE id = 'default'")
            row = cur.fetchone()
            if row:
                d = dict(row)
                d["config"] = json.loads(d["config_json"]) if d.get("config_json") else {}
                d["is_active"] = True
                return d
            return {
                "id": "default",
                "name": "Default Workspace",
                "description": "Primary workspace",
                "root_path": ".",
                "is_active": True,
                "config": {},
            }

    def set_active_workspace(self, workspace_id: str) -> bool:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT id FROM workspaces WHERE id = ?", (workspace_id,))
            if not cur.fetchone():
                return False
            now = time.time()
            conn.execute("UPDATE workspaces SET is_active = 0")
            conn.execute("UPDATE workspaces SET is_active = 1, updated_at = ? WHERE id = ?", (now, workspace_id))
            return True

    def update_workspace(
        self,
        workspace_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        ws = self.get_workspace(workspace_id)
        if not ws:
            return None
        now = time.time()
        new_name = name if name is not None else ws["name"]
        new_desc = description if description is not None else ws["description"]
        new_cfg = config if config is not None else ws["config"]

        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                UPDATE workspaces SET name = ?, description = ?, config_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_name, new_desc, json.dumps(new_cfg), now, workspace_id),
            )
        return self.get_workspace(workspace_id)

    def delete_workspace(self, workspace_id: str) -> bool:
        if workspace_id == "default":
            return False  # Cannot delete default workspace
        ws = self.get_workspace(workspace_id)
        if not ws:
            return False
        if ws["is_active"]:
            self.set_active_workspace("default")

        with self._lock, self._get_connection() as conn:
            # Delete children first: with foreign keys enabled the order matters,
            # and metrics must not survive their run.
            conn.execute(
                "DELETE FROM metrics WHERE run_id IN (SELECT id FROM runs WHERE workspace_id = ?)",
                (workspace_id,),
            )
            conn.execute("DELETE FROM runs WHERE workspace_id = ?", (workspace_id,))
            conn.execute("DELETE FROM checkpoints WHERE workspace_id = ?", (workspace_id,))
            conn.execute("DELETE FROM datasets WHERE workspace_id = ?", (workspace_id,))
            conn.execute("DELETE FROM workspaces WHERE id = ?", (workspace_id,))
        return True


# Backward compatibility alias
Database = StudioDatabase
