"""Prints a migration/backup summary for a database (smoke-test helper)."""

import json
import sqlite3
import sys
from pathlib import Path

path = Path(sys.argv[1])
conn = sqlite3.connect(str(path))
conn.row_factory = sqlite3.Row
result = {
    "schema_version": None,
    "columns": {},
    "run": None,
    "backups": [],
}
try:
    result["schema_version"] = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
except sqlite3.OperationalError:
    pass
for table in ("runs", "checkpoints", "datasets", "benchmarks"):
    try:
        result["columns"][table] = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    except sqlite3.OperationalError:
        result["columns"][table] = []
row = conn.execute("SELECT id, status, total_steps FROM runs WHERE id = 'run_legacy'").fetchone()
result["run"] = dict(row) if row else None
conn.close()
result["backups"] = [p.name for p in sorted((path.parent / "backups").glob("*.db"))]
print(json.dumps(result, indent=2))