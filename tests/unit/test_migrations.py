"""Tests for the shared schema migration and backup layer."""

import sqlite3
import time
from pathlib import Path

import pytest

from src.db.database import DEFAULT_DB_PATH, StudioDatabase
from src.db.migrations import MIGRATIONS, apply_migrations, current_version
from src.plugins.manager import PluginManager


def _legacy_db(path: Path) -> None:
    """Creates a pre-workspace, pre-final_loss database with real rows."""
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
            CREATE TABLE workspaces (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT,
                root_path TEXT NOT NULL, is_active INTEGER DEFAULT 0,
                config_json TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL
            );
            CREATE TABLE runs (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL,
                current_phase TEXT NOT NULL, dataset_source TEXT,
                total_steps INTEGER DEFAULT 0, total_tokens INTEGER DEFAULT 0,
                best_val_loss REAL, config_json TEXT,
                created_at REAL NOT NULL, updated_at REAL NOT NULL
            );
            """
        )
        now = time.time()
        conn.execute(
            "INSERT INTO workspaces VALUES ('default', 'Default', 'legacy', '.', 1, '{}', ?, ?)",
            (now, now),
        )
        conn.execute(
            "INSERT INTO runs (id, name, status, current_phase, total_steps, total_tokens, created_at, updated_at) "
            "VALUES ('run_legacy', 'Legacy run', 'STOPPED', 'pretrain', 42, 1234, ?, ?)",
            (now, now),
        )
        conn.commit()
    finally:
        conn.close()


def test_legacy_database_is_upgraded_and_backed_up(tmp_path: Path) -> None:
    db_path = tmp_path / "studio.db"
    _legacy_db(db_path)

    db = StudioDatabase(db_path)
    run = db.get_run("run_legacy")
    assert run is not None
    assert run["total_steps"] == 42
    assert run["final_loss"] is None
    assert run["workspace_id"] == "default"

    conn = sqlite3.connect(str(db_path))
    try:
        assert current_version(conn) == MIGRATIONS[-1][0]
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"runs", "metrics", "checkpoints", "datasets", "schema_migrations"} <= tables
    finally:
        conn.close()

    backups = list((tmp_path / "backups").glob("studio-*.db"))
    assert len(backups) == 1
    backup = sqlite3.connect(str(backups[0]))
    try:
        # The backup predates the upgrade: original columns and row are intact.
        cols = {row[1] for row in backup.execute("PRAGMA table_info(runs)")}
        assert "final_loss" not in cols
        assert backup.execute("SELECT total_steps FROM runs WHERE id='run_legacy'").fetchone()[0] == 42
    finally:
        backup.close()


def test_migrations_are_idempotent_and_do_not_create_repeat_backups(tmp_path: Path) -> None:
    db_path = tmp_path / "studio.db"
    _legacy_db(db_path)
    StudioDatabase(db_path)
    backups_after_first = list((tmp_path / "backups").glob("studio-*.db"))

    StudioDatabase(db_path)
    backups_after_second = list((tmp_path / "backups").glob("studio-*.db"))
    assert len(backups_after_second) == len(backups_after_first) == 1

    conn = sqlite3.connect(str(db_path))
    try:
        applied = conn.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
        assert [row[0] for row in applied] == [1, 2, 3]
    finally:
        conn.close()


def test_fresh_database_needs_no_backup(tmp_path: Path) -> None:
    db_path = tmp_path / "fresh.db"
    db = StudioDatabase(db_path)
    assert db.get_run("missing") is None
    assert not (tmp_path / "backups").exists() or not list((tmp_path / "backups").glob("*.db"))
    conn = sqlite3.connect(str(db_path))
    try:
        assert current_version(conn) == MIGRATIONS[-1][0]
    finally:
        conn.close()


def test_default_database_path_is_absolute() -> None:
    assert DEFAULT_DB_PATH.is_absolute()
    assert DEFAULT_DB_PATH.parent.name == "data"


def test_migration_refuses_without_a_backup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "studio.db"
    _legacy_db(db_path)
    monkeypatch.setattr("src.db.migrations.backup_database", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="backup"):
        StudioDatabase(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        # The failed safety check must not leave a half-migrated schema behind.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
        assert "final_loss" not in cols
    finally:
        conn.close()


def test_database_from_a_newer_build_is_refused(tmp_path: Path) -> None:
    db_path = tmp_path / "studio.db"
    StudioDatabase(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT INTO schema_migrations (version, name, applied_at) VALUES (99, 'future.sql', 0)"
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(RuntimeError, match="newer"):
        StudioDatabase(db_path)


def test_safe_mode_disables_plugins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMMAKER_SAFE_MODE", "1")
    db = StudioDatabase(tmp_path / "studio.db")
    manager = PluginManager(tmp_path / "plugins", db=db)
    assert manager.discover_plugins() == []
    ok, message = manager.set_plugin_enabled("vision_adapter", True)
    assert not ok and "Safe mode" in message
    assert not (tmp_path / "plugins").exists()


def test_apply_migrations_returns_empty_when_current(tmp_path: Path) -> None:
    db_path = tmp_path / "fresh.db"
    StudioDatabase(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        assert apply_migrations(conn, db_path) == []
    finally:
        conn.close()