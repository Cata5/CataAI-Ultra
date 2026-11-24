"""Small SQLite helper used by the manual desktop-startup smoke check."""

from __future__ import annotations

import sqlite3
import sys


def main() -> int:
    action, db_path = sys.argv[1], sys.argv[2]
    with sqlite3.connect(db_path) as conn:
        if action == "seed":
            run_id = conn.execute("SELECT id FROM runs ORDER BY created_at DESC LIMIT 1").fetchone()
            if not run_id:
                print("no runs to seed")
                return 1
            conn.execute("UPDATE runs SET status = ? WHERE id = ?", ("RUNNING", run_id[0]))
            conn.commit()
            print(f"seeded {run_id[0]} as RUNNING")
        elif action == "tables":
            names = [row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )]
            for name in names:
                count = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                print(f"{name}: {count}")
        else:
            rows = conn.execute("SELECT id, status FROM runs ORDER BY created_at DESC LIMIT 3").fetchall()
            print(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())