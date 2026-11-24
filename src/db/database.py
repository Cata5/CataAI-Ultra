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

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/rocanon_studio.db")


class StudioDatabase:
    """Thread-safe SQLite database manager for training metadata and metrics."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._get_connection() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                current_phase TEXT NOT NULL,
                dataset_source TEXT,
                total_steps INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                best_val_loss REAL,
                config_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                tokens INTEGER NOT NULL,
                train_loss REAL NOT NULL,
                val_loss REAL,
                generalization_gap REAL,
                grad_norm REAL DEFAULT 0.0,
                muon_lr REAL DEFAULT 0.0,
                adamw_lr REAL DEFAULT 0.0,
                vram_used_gb REAL DEFAULT 0.0,
                tok_per_sec REAL DEFAULT 0.0,
                tflops REAL DEFAULT 0.0,
                timestamp REAL NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_metrics_run_step ON metrics(run_id, step);

            CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY,
                run_id TEXT,
                step INTEGER NOT NULL,
                tokens INTEGER DEFAULT 0,
                train_loss REAL,
                val_loss REAL,
                file_path TEXT NOT NULL,
                file_size_bytes INTEGER DEFAULT 0,
                phase TEXT,
                metadata_json TEXT,
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_checkpoints_step ON checkpoints(step);

            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                level TEXT NOT NULL,
                category TEXT NOT NULL,
                message TEXT NOT NULL,
                step INTEGER DEFAULT 0,
                timestamp REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pipelines (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                current_phase_idx INTEGER DEFAULT 0,
                phases_json TEXT NOT NULL,
                state_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size_bytes INTEGER DEFAULT 0,
                format TEXT NOT NULL,
                token_count INTEGER DEFAULT 0,
                sample_count INTEGER DEFAULT 0,
                revision TEXT NOT NULL,
                stats_json TEXT,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                checkpoint_id TEXT,
                step INTEGER DEFAULT 0,
                category TEXT NOT NULL,
                score REAL DEFAULT 0.0,
                results_json TEXT,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS workspaces (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                root_path TEXT NOT NULL,
                is_active INTEGER DEFAULT 0,
                config_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            """)

            # Safe column migrations for workspace_id
            for tbl in ("runs", "datasets", "checkpoints", "benchmarks"):
                try:
                    conn.execute(f"ALTER TABLE {tbl} ADD COLUMN workspace_id TEXT DEFAULT 'default';")
                except sqlite3.OperationalError:
                    pass  # column already exists

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
            conn.execute(
                """
                INSERT OR REPLACE INTO runs 
                (id, name, status, current_phase, dataset_source, total_steps, total_tokens, config_json, workspace_id, created_at, updated_at)
                VALUES (?, ?, 'STARTING', ?, ?, 0, 0, ?, ?, ?, ?)
                """,
                (run_id, name, current_phase, dataset_source, cfg_str, workspace_id, now, now),
            )

    def update_run_status(
        self,
        run_id: str,
        status: str,
        step: Optional[int] = None,
        tokens: Optional[int] = None,
        best_val_loss: Optional[float] = None,
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
                 grad_norm, muon_lr, adamw_lr, vram_used_gb, tok_per_sec, tflops, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            )
            for d in batch
        ]
        with self._lock, self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO metrics
                (run_id, step, tokens, train_loss, val_loss, generalization_gap,
                 grad_norm, muon_lr, adamw_lr, vram_used_gb, tok_per_sec, tflops, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

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
    def save_settings(self, settings_dict: Dict[str, Any]) -> None:
        now = time.time()
        with self._lock, self._get_connection() as conn:
            for k, v in settings_dict.items():
                conn.execute(
                    """
                    INSERT INTO settings (key, value_json, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, updated_at = excluded.updated_at
                    """,
                    (str(k), json.dumps(v), now),
                )

    def get_settings(self) -> Dict[str, Any]:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT key, value_json FROM settings")
            res = {}
            for row in cur.fetchall():
                try:
                    res[row["key"]] = json.loads(row["value_json"])
                except Exception:
                    res[row["key"]] = row["value_json"]
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
            conn.execute(
                """
                INSERT OR REPLACE INTO workspaces (id, name, description, root_path, is_active, config_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, 0, ?, ?, ?)
                """,
                (workspace_id, name, description, root_path, cfg_str, now, now),
            )
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
            conn.execute("DELETE FROM workspaces WHERE id = ?", (workspace_id,))
            conn.execute("DELETE FROM runs WHERE workspace_id = ?", (workspace_id,))
            conn.execute("DELETE FROM checkpoints WHERE workspace_id = ?", (workspace_id,))
            conn.execute("DELETE FROM datasets WHERE workspace_id = ?", (workspace_id,))
        return True


# Backward compatibility alias
Database = StudioDatabase

