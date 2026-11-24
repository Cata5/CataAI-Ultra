"""Builds a legacy (pre-migration) database for upgrade smoke tests."""

import sqlite3
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(str(path))
conn.executescript("""
CREATE TABLE workspaces (id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT,
  root_path TEXT NOT NULL, is_active INTEGER DEFAULT 0, config_json TEXT,
  created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE runs (id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL,
  current_phase TEXT NOT NULL, dataset_source TEXT, total_steps INTEGER DEFAULT 0,
  total_tokens INTEGER DEFAULT 0, best_val_loss REAL, config_json TEXT,
  created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
  step INTEGER NOT NULL, tokens INTEGER NOT NULL, train_loss REAL NOT NULL,
  val_loss REAL, timestamp REAL NOT NULL);
""")
now = time.time()
conn.execute("INSERT INTO workspaces VALUES ('default','Default','legacy','.',1,'{}',?,?)", (now, now))
conn.execute("INSERT INTO runs (id,name,status,current_phase,total_steps,created_at,updated_at) VALUES ('run_legacy','Legacy run','STOPPED','pretrain',99,?,?)", (now, now))
conn.commit()
conn.close()
print(f"legacy database created at {path}")