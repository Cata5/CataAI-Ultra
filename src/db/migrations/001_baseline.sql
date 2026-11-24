-- LLM Maker schema, version 1 (baseline).
-- Shared by the Python control plane and the Rust desktop shell; both runners
-- execute this file and record the version in schema_migrations.
-- Every statement must be safe to re-run on an existing database.

CREATE TABLE IF NOT EXISTS workspaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    root_path TEXT NOT NULL,
    is_active INTEGER DEFAULT 0,
    config_json TEXT DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    current_phase TEXT NOT NULL,
    dataset_source TEXT,
    workspace_id TEXT DEFAULT 'default',
    total_steps INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    best_val_loss REAL,
    final_loss REAL,
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
    timestamp REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_metrics_run_step ON metrics(run_id, step);

CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER DEFAULT 0,
    format TEXT NOT NULL,
    workspace_id TEXT DEFAULT 'default',
    token_count INTEGER DEFAULT 0,
    sample_count INTEGER DEFAULT 0,
    revision TEXT NOT NULL DEFAULT '1',
    stats_json TEXT,
    created_at REAL NOT NULL
);

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
    workspace_id TEXT DEFAULT 'default',
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

CREATE TABLE IF NOT EXISTS benchmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    checkpoint_id TEXT,
    step INTEGER DEFAULT 0,
    category TEXT NOT NULL,
    score REAL DEFAULT 0.0,
    results_json TEXT,
    workspace_id TEXT DEFAULT 'default',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);