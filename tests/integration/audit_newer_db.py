"""Creates a database stamped with a future schema version (smoke helper)."""

import sqlite3
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(str(path))
conn.execute("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at REAL NOT NULL)")
conn.execute("INSERT INTO schema_migrations VALUES (99, 'future.sql', 0)")
conn.commit()
conn.close()
print(f"future-version database created at {path}")