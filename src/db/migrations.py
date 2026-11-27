"""Shared, versioned schema migrations for the SQLite database.

The Rust desktop shell embeds the same `.sql` files at compile time; both layers
apply them in order and record progress in `schema_migrations`. This is the single
contract that prevents the Python and Rust schemas from drifting apart.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

# (version, filename). Versions must only ever be appended.
MIGRATIONS: Tuple[Tuple[int, str], ...] = (
    (1, "001_baseline.sql"),
    (2, "002_final_loss.sql"),
    (3, "003_workspace_columns.sql"),
    (4, "004_metrics_progress.sql"),
)

BACKUP_RETENTION = 10


def _split_statements(sql: str) -> Iterable[str]:
    """Splits a migration file into statements.

    The migration files are intentionally simple (no semicolons inside string
    literals), so stripping `--` comments and splitting on `;` is sufficient.
    """
    cleaned: List[str] = []
    for line in sql.splitlines():
        if line.strip().startswith("--"):
            continue
        cleaned.append(line)
    for statement in "\n".join(cleaned).split(";"):
        statement = statement.strip()
        if statement:
            yield statement


def _has_user_tables(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name IN ('runs', 'workspaces') LIMIT 1"
    ).fetchone()
    return row is not None


def current_version(conn: sqlite3.Connection) -> int:
    """Returns the highest applied migration version (0 if none)."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'"
    ).fetchone()
    if row is None:
        return 0
    row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def backup_database(conn: sqlite3.Connection, db_path: Path, keep: int = BACKUP_RETENTION) -> Optional[Path]:
    """Writes a consistent snapshot next to the database and prunes old ones.

    Uses SQLite's online backup API so a WAL-mode database is copied safely.
    Returns the new backup path, or None when there is nothing to back up.
    """
    try:
        if not db_path.exists() or db_path.stat().st_size == 0:
            return None
        backups_dir = db_path.parent / "backups"
        backups_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = backups_dir / f"{db_path.stem}-{stamp}.db"
        counter = 1
        while target.exists():
            target = backups_dir / f"{db_path.stem}-{stamp}-{counter}.db"
            counter += 1
        dest = sqlite3.connect(str(target))
        try:
            conn.backup(dest)
        finally:
            dest.close()
        backups = sorted(backups_dir.glob(f"{db_path.stem}-*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in backups[keep:]:
            try:
                stale.unlink()
            except OSError as exc:
                logger.warning("Could not prune old backup %s: %s", stale, exc)
        logger.info("Database backup written: %s", target)
        return target
    except Exception as exc:  # A backup failure must not block schema creation on a fresh DB.
        logger.error("Database backup failed for %s: %s", db_path, exc)
        return None


def apply_migrations(conn: sqlite3.Connection, db_path: Optional[Path] = None) -> List[int]:
    """Applies pending migrations and returns the versions applied in this call."""
    latest = MIGRATIONS[-1][0]
    version = current_version(conn)
    if version > latest:
        # A newer build upgraded this database. Opening it with older code would
        # write against a schema it does not understand; refuse instead.
        raise RuntimeError(
            f"Database schema version {version} is newer than this build supports ({latest}). "
            "Update LLM Maker or restore a backup made by this version."
        )
    if version == latest:
        return []

    backup_path = None
    if _has_user_tables(conn):
        target = db_path or Path(str(conn.execute("PRAGMA database_list").fetchone()[2] or ""))
        backup_path = backup_database(conn, target)
        if backup_path is None:
            # Never upgrade a populated database without a recovery point.
            raise RuntimeError(
                f"Refusing to migrate {target}: pre-migration backup could not be written."
            )

    conn.execute(
        """CREATE TABLE IF NOT EXISTS schema_migrations (
               version INTEGER PRIMARY KEY,
               name TEXT NOT NULL,
               applied_at REAL NOT NULL
           )"""
    )

    applied: List[int] = []
    for migration_version, filename in MIGRATIONS:
        if migration_version <= version:
            continue
        sql = (MIGRATIONS_DIR / filename).read_text(encoding="utf-8")
        for statement in _split_statements(sql):
            try:
                conn.execute(statement)
            except sqlite3.OperationalError as exc:
                # The baseline already contains columns that later ALTERs add;
                # "duplicate column name" therefore means the step is satisfied.
                if statement.lstrip().upper().startswith("ALTER") and "duplicate column name" in str(exc).lower():
                    continue
                raise
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
            (migration_version, filename, time.time()),
        )
        applied.append(migration_version)
        logger.info("Applied database migration %s (%s)", migration_version, filename)

    conn.commit()
    if applied and backup_path:
        logger.info("Schema upgraded to version %s; previous data backed up to %s", latest, backup_path)
    return applied