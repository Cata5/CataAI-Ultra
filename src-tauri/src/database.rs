//! src-tauri/src/database.rs
//! Native SQLite Database Manager for LLM Maker in Rust.
//! Provides instant (<1ms) workspace, run, and metric queries without launching Python.

use crate::secrets::{is_secret_key, SecretStore, REDACTED, SECRET_KEYS};
use rusqlite::{params, Connection, OptionalExtension, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceItem {
    pub id: String,
    pub name: String,
    pub description: String,
    pub root_path: String,
    pub is_active: bool,
    pub config: serde_json::Value,
    pub run_count: Option<i64>,
    pub checkpoint_count: Option<i64>,
    pub dataset_count: Option<i64>,
    pub created_at: f64,
    pub updated_at: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub run_id: String,
    pub name: String,
    pub current_phase: String,
    pub status: String,
    pub final_loss: Option<f64>,
    pub total_steps: Option<i64>,
    pub created_at: f64,
    pub workspace_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointRecord {
    pub id: String,
    pub run_id: Option<String>,
    pub step: i64,
    pub tokens: i64,
    pub train_loss: Option<f64>,
    pub val_loss: Option<f64>,
    pub file_path: String,
    pub file_size_bytes: i64,
    pub phase: Option<String>,
    pub workspace_id: String,
    pub created_at: f64,
}

pub struct DatabaseManager {
    db_path: String,
    secrets: SecretStore,
}


/// Checks for encrypted rows without creating a replacement key.
fn encrypted_secret_exists(db_path: &str) -> Result<bool> {
    let path = Path::new(db_path);
    if !path.exists() {
        return Ok(false);
    }
    let conn = Connection::open(path)?;
    let table: Option<String> = conn
        .query_row(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'settings'",
            [],
            |row| row.get(0),
        )
        .optional()?;
    if table.is_none() {
        return Ok(false);
    }
    let mut stmt = conn.prepare("SELECT value_json FROM settings WHERE key IN (?1, ?2)")?;
    let rows = stmt.query_map(params![SECRET_KEYS[0], SECRET_KEYS[1]], |row| {
        let value: String = row.get(0)?;
        Ok(value)
    })?;
    for row in rows.flatten() {
        if let Ok(serde_json::Value::String(value)) =
            serde_json::from_str::<serde_json::Value>(&row)
        {
            if SecretStore::is_encrypted(&value) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

impl DatabaseManager {
    /// Opens the database and applies pending migrations.
    ///
    /// Initialization errors are returned instead of ignored: silently running
    /// against an unknown or half-migrated schema is how data gets corrupted.
    pub fn new(db_path: &str) -> Result<Self> {
        let key_path = Path::new(db_path)
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(".secrets.key");
        // Never replace a missing key while encrypted rows exist: doing so would
        // silently make every stored credential undecryptable.
        if !key_path.exists() && encrypted_secret_exists(db_path)? {
            return Err(rusqlite::Error::InvalidParameterName(format!(
                "the encryption key {} is missing; restore the matching .secrets.key file",
                key_path.display()
            )));
        }
        let secrets = SecretStore::load_or_create(&key_path)
            .map_err(rusqlite::Error::InvalidParameterName)?;
        let mgr = Self {
            db_path: db_path.to_string(),
            secrets,
        };
        mgr.init_tables()?;
        Ok(mgr)
    }

    fn open(&self) -> Result<Connection> {
        let p = Path::new(&self.db_path);
        if let Some(parent) = p.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let conn = Connection::open(&self.db_path)?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        Ok(conn)
    }

    pub fn init_tables(&self) -> Result<()> {
        let conn = self.open()?;

        // quick_check is cheap on startup and catches a damaged file before the
        // migration runner takes a backup of it or writes new tables.
        let check: String = conn.query_row("PRAGMA quick_check", [], |row| row.get(0))?;
        if !check.eq_ignore_ascii_case("ok") {
            log::error!("Database integrity check failed for {:?}: {}", self.db_path, check);
            return Err(rusqlite::Error::InvalidParameterName(format!(
                "database integrity check failed: {}; stop the app and restore a backup",
                check
            )));
        }

        self.ensure_schema(&conn)?;
        self.migrate_plaintext_secrets(&conn)?;

        // Ensure default workspace exists
        let count: i64 = conn
            .query_row("SELECT count(*) FROM workspaces", [], |r| r.get(0))
            .unwrap_or(0);

        if count == 0 {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();
            let _ = conn.execute(
                "INSERT INTO workspaces (id, name, description, root_path, is_active, config_json, created_at, updated_at) VALUES ('default', 'Default Workspace', 'Primary workspace for default checkpoints and datasets', '.', 1, '{}', ?1, ?1)",
                params![now],
            );
        }

        // Ensure at least one active workspace
        let active_count: i64 = conn
            .query_row("SELECT count(*) FROM workspaces WHERE is_active = 1", [], |r| r.get(0))
            .unwrap_or(0);
        if active_count == 0 {
            let _ = conn.execute("UPDATE workspaces SET is_active = 1 WHERE id = 'default'", []);
        }

        Ok(())
    }

    /// Applies the shared SQL migrations embedded at compile time.
    ///
    /// The Python control plane executes the same files, so the schema has exactly
    /// one definition and one version history.
    fn ensure_schema(&self, conn: &Connection) -> Result<()> {
        const MIGRATIONS: &[(i64, &str, &str)] = &[
            (1, "001_baseline.sql", include_str!("../../src/db/migrations/001_baseline.sql")),
            (2, "002_final_loss.sql", include_str!("../../src/db/migrations/002_final_loss.sql")),
            (3, "003_workspace_columns.sql", include_str!("../../src/db/migrations/003_workspace_columns.sql")),
            (4, "004_metrics_progress.sql", include_str!("../../src/db/migrations/004_metrics_progress.sql")),
        ];

        let latest = MIGRATIONS.last().map(|m| m.0).unwrap_or(0);
        let current = self.schema_version(conn)?;
        if current > latest {
            // Written by a newer build; do not write against an unknown schema.
            log::error!(
                "Database schema version {} is newer than supported version {}",
                current, latest
            );
            return Err(rusqlite::Error::InvalidParameterName(format!(
                "database schema version {} is newer than this build supports ({})",
                current, latest
            )));
        }
        if current == latest {
            return Ok(());
        }

        if self.has_user_tables(conn) {
            match self.backup_database(conn) {
                Some(path) => {
                    crate::log_trace(&format!("Pre-migration database backup written: {:?}", path));
                }
                None => {
                    // Never upgrade a populated database without a recovery point.
                    log::error!("Pre-migration backup failed; refusing to migrate {:?}", self.db_path);
                    return Err(rusqlite::Error::InvalidParameterName(
                        "pre-migration database backup failed; refusing to upgrade".to_string(),
                    ));
                }
            }
        }

        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at REAL NOT NULL)",
            [],
        )?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        for &(version, filename, sql) in MIGRATIONS {
            if version <= current {
                continue;
            }
            for statement in split_sql_statements(sql) {
                match conn.execute(&statement, []) {
                    Ok(_) => {}
                    // The baseline creates columns that later ALTERs add; on an
                    // existing database "duplicate column name" means satisfied.
                    Err(err)
                        if statement.trim_start().to_uppercase().starts_with("ALTER")
                            && err.to_string().to_lowercase().contains("duplicate column name") => {}
                    Err(err) => return Err(err),
                }
            }
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations (version, name, applied_at) VALUES (?1, ?2, ?3)",
                params![version, filename, now],
            )?;
            crate::log_trace(&format!("Applied database migration {} ({})", version, filename));
            log::info!("Applied database migration {} ({})", version, filename);
        }
        Ok(())
    }

    fn schema_version(&self, conn: &Connection) -> Result<i64> {
        let exists: i64 = conn.query_row(
            "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'",
            [],
            |row| row.get(0),
        )?;
        if exists == 0 {
            return Ok(0);
        }
        Ok(conn
            .query_row("SELECT COALESCE(MAX(version), 0) FROM schema_migrations", [], |row| row.get(0))
            .unwrap_or(0))
    }

    fn has_user_tables(&self, conn: &Connection) -> bool {
        conn.query_row(
            "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name IN ('runs', 'workspaces')",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map(|count| count > 0)
        .unwrap_or(false)
    }

    /// Copies the database with SQLite's online backup API before a schema change.
    fn backup_database(&self, conn: &Connection) -> Option<std::path::PathBuf> {
        use rusqlite::backup::Backup;
        use std::time::Duration;

        let db = Path::new(&self.db_path);
        if !db.exists() {
            return None;
        }
        let dir = db.parent()?.join("backups");
        fs::create_dir_all(&dir).ok()?;
        let stem = db.file_stem()?.to_string_lossy().to_string();
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_secs();
        let mut target = dir.join(format!("{}-{}.db", stem, secs));
        let mut suffix = 1;
        while target.exists() {
            target = dir.join(format!("{}-{}-{}.db", stem, secs, suffix));
            suffix += 1;
        }
        let mut dest = Connection::open(&target).ok()?;
        {
            let backup = Backup::new(conn, &mut dest).ok()?;
            backup
                .run_to_completion(5, Duration::from_millis(50), None)
                .ok()?;
        }
        Self::prune_backups(&dir, &stem);
        Some(target)
    }

    fn prune_backups(dir: &Path, stem: &str) {
        let mut backups: Vec<_> = fs::read_dir(dir)
            .map(|entries| {
                entries
                    .flatten()
                    .filter(|entry| {
                        entry
                            .file_name()
                            .to_string_lossy()
                            .starts_with(&format!("{}-", stem))
                    })
                    .collect()
            })
            .unwrap_or_default();
        backups.sort_by_key(|entry| entry.metadata().and_then(|m| m.modified()).ok());
        if backups.len() > 10 {
            for stale in &backups[..backups.len() - 10] {
                let _ = fs::remove_file(stale.path());
            }
        }
    }
    /// Marks runs left in a non-terminal state by a crash/forced exit as stopped.
    ///
    /// Without this, a killed desktop session leaves runs permanently displayed as
    /// RUNNING/STOPPING, which also makes the UI believe a worker is active.
    pub fn recover_stale_runs(&self) -> Result<usize> {
        let conn = self.open()?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let updated = conn.execute(
            "UPDATE runs SET status = 'STOPPED', updated_at = ?1 WHERE status IN ('STARTING','RUNNING','PAUSING','PAUSED','STOPPING')",
            params![now],
        )?;
        Ok(updated)
    }

    pub fn list_workspaces(&self) -> Result<Vec<WorkspaceItem>> {
        let conn = self.open()?;
        let mut stmt = conn.prepare(
            "SELECT id, name, description, root_path, is_active, config_json, created_at, updated_at FROM workspaces ORDER BY created_at ASC",
        )?;
        let rows = stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let name: String = row.get(1)?;
            let description: String = row.get(2).unwrap_or_default();
            let root_path: String = row.get(3).unwrap_or_else(|_| ".".to_string());
            let is_active = row.get::<_, i64>(4).unwrap_or(0) == 1;
            let config_raw: String = row.get(5).unwrap_or_else(|_| "{}".to_string());
            let config: serde_json::Value = serde_json::from_str(&config_raw).unwrap_or(serde_json::json!({}));
            let created_at: f64 = row.get(6).unwrap_or(0.0);
            let updated_at: f64 = row.get(7).unwrap_or(created_at);

            Ok(WorkspaceItem {
                id,
                name,
                description,
                root_path,
                is_active,
                config,
                run_count: None,
                checkpoint_count: None,
                dataset_count: None,
                created_at,
                updated_at,
            })
        })?;

        let mut res = Vec::new();
        for r in rows.flatten() {
            res.push(r);
        }
        Ok(res)
    }

    pub fn get_active_workspace(&self) -> Result<WorkspaceItem> {
        let conn = self.open()?;
        let mut stmt = conn.prepare(
            "SELECT id, name, description, root_path, is_active, config_json, created_at, updated_at FROM workspaces WHERE is_active = 1 LIMIT 1",
        )?;
        let res = stmt.query_row([], |row| {
            let id: String = row.get(0)?;
            let name: String = row.get(1)?;
            let description: String = row.get(2).unwrap_or_default();
            let root_path: String = row.get(3).unwrap_or_else(|_| ".".to_string());
            let is_active = row.get::<_, i64>(4).unwrap_or(0) == 1;
            let config_raw: String = row.get(5).unwrap_or_else(|_| "{}".to_string());
            let config: serde_json::Value = serde_json::from_str(&config_raw).unwrap_or(serde_json::json!({}));
            let created_at: f64 = row.get(6).unwrap_or(0.0);
            let updated_at: f64 = row.get(7).unwrap_or(created_at);

            Ok(WorkspaceItem {
                id,
                name,
                description,
                root_path,
                is_active,
                config,
                run_count: None,
                checkpoint_count: None,
                dataset_count: None,
                created_at,
                updated_at,
            })
        });

        match res {
            Ok(item) => Ok(item),
            Err(_) => {
                // Fallback to default
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64();
                Ok(WorkspaceItem {
                    id: "default".to_string(),
                    name: "Default Workspace".to_string(),
                    description: "Primary workspace".to_string(),
                    root_path: ".".to_string(),
                    is_active: true,
                    config: serde_json::json!({}),
                    run_count: None,
                    checkpoint_count: None,
                    dataset_count: None,
                    created_at: now,
                    updated_at: now,
                })
            }
        }
    }

    pub fn set_active_workspace(&self, id: &str) -> Result<()> {
        let conn = self.open()?;
        conn.execute("UPDATE workspaces SET is_active = 0", [])?;
        conn.execute("UPDATE workspaces SET is_active = 1 WHERE id = ?1", params![id])?;
        Ok(())
    }

    pub fn create_workspace(&self, name: &str, description: &str) -> Result<WorkspaceItem> {
        let conn = self.open()?;
        let mut raw_id = name
            .trim()
            .to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
            .collect::<String>();

        if raw_id.is_empty() {
            raw_id = "workspace".to_string();
        }

        let count: i64 = conn
            .query_row("SELECT count(*) FROM workspaces WHERE id = ?1", params![&raw_id], |r| r.get(0))
            .unwrap_or(0);

        let final_id = if count > 0 {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            format!("{}_{}", raw_id, ts)
        } else {
            raw_id
        };

        let root_path = format!("workspaces/{}", final_id);
        let _ = fs::create_dir_all(format!("{}/datasets", root_path));
        let _ = fs::create_dir_all(format!("{}/checkpoints", root_path));
        let _ = fs::create_dir_all(format!("{}/models", root_path));
        let _ = fs::create_dir_all(format!("{}/runs", root_path));

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        conn.execute(
            "INSERT INTO workspaces (id, name, description, root_path, is_active, config_json, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, 0, '{}', ?5, ?5)",
            params![final_id, name, description, root_path, now],
        )?;

        Ok(WorkspaceItem {
            id: final_id,
            name: name.to_string(),
            description: description.to_string(),
            root_path,
            is_active: false,
            config: serde_json::json!({}),
            run_count: Some(0),
            checkpoint_count: Some(0),
            dataset_count: Some(0),
            created_at: now,
            updated_at: now,
        })
    }

    pub fn delete_workspace(&self, id: &str) -> Result<()> {
        if id == "default" {
            return Err(rusqlite::Error::InvalidQuery);
        }
        let conn = self.open()?;

        // If deleting active workspace, switch to default
        let is_active: bool = conn
            .query_row("SELECT is_active FROM workspaces WHERE id = ?1", params![id], |r| r.get::<_, i64>(0))
            .map(|v| v == 1)
            .unwrap_or(false);

        if is_active {
            let _ = conn.execute("UPDATE workspaces SET is_active = 1 WHERE id = 'default'", []);
        }

        // Children before the parent; metrics would otherwise outlive their run.
        conn.execute(
            "DELETE FROM metrics WHERE run_id IN (SELECT id FROM runs WHERE workspace_id = ?1)",
            params![id],
        )?;
        conn.execute("DELETE FROM runs WHERE workspace_id = ?1", params![id])?;
        conn.execute("DELETE FROM datasets WHERE workspace_id = ?1", params![id])?;
        conn.execute("DELETE FROM checkpoints WHERE workspace_id = ?1", params![id])?;
        conn.execute("DELETE FROM workspaces WHERE id = ?1", params![id])?;

        let ws_dir = format!("workspaces/{}", id);
        let _ = fs::remove_dir_all(&ws_dir);

        Ok(())
    }

    pub fn list_runs(&self, workspace_id: Option<&str>) -> Result<Vec<RunSummary>> {
        let conn = self.open()?;
        // Parameterised: workspace ids are user-controlled and must never be interpolated.
        let query = "SELECT id, name, current_phase, status, COALESCE(final_loss, best_val_loss), total_steps, created_at, workspace_id FROM runs WHERE (?1 IS NULL OR workspace_id = ?1) ORDER BY created_at DESC";
        let mut stmt = conn.prepare(query)?;
        let rows = stmt.query_map(params![workspace_id], |row| {
            Ok(RunSummary {
                run_id: row.get(0)?,
                name: row.get(1)?,
                current_phase: row.get(2).unwrap_or_else(|_| "pretrain".to_string()),
                status: row.get(3).unwrap_or_else(|_| "IDLE".to_string()),
                final_loss: row.get(4).ok(),
                total_steps: row.get(5).ok(),
                created_at: row.get(6)?,
                workspace_id: row.get(7).unwrap_or_default(),
            })
        })?;

        let mut res = Vec::new();
        for r in rows.flatten() {
            res.push(r);
        }
        Ok(res)
    }

    pub fn list_checkpoints(&self, workspace_id: Option<&str>) -> Result<Vec<CheckpointRecord>> {
        let conn = self.open()?;
        let query = "SELECT id, run_id, step, tokens, train_loss, val_loss, file_path, file_size_bytes, phase, workspace_id, created_at FROM checkpoints WHERE (?1 IS NULL OR workspace_id = ?1) ORDER BY step DESC";
        let mut stmt = conn.prepare(query)?;
        let rows = stmt.query_map(params![workspace_id], |row| {
            Ok(CheckpointRecord {
                id: row.get(0)?,
                run_id: row.get(1).ok(),
                step: row.get(2)?,
                tokens: row.get(3).unwrap_or(0),
                train_loss: row.get(4).ok(),
                val_loss: row.get(5).ok(),
                file_path: row.get(6)?,
                file_size_bytes: row.get(7).unwrap_or(0),
                phase: row.get(8).ok(),
                workspace_id: row.get(9).unwrap_or_else(|_| "default".to_string()),
                created_at: row.get(10).unwrap_or(0.0),
            })
        })?;

        let mut res = Vec::new();
        for r in rows.flatten() {
            res.push(r);
        }
        Ok(res)
    }

    pub fn delete_checkpoint(&self, id: &str) -> Result<()> {
        let conn = self.open()?;
        conn.execute(
            "DELETE FROM checkpoints WHERE id = ?1 OR file_path LIKE ?2",
            params![id, format!("%{}%", id)],
        )?;
        Ok(())
    }

    pub fn get_metrics(&self, run_id: &str, start_step: i64, limit: i64) -> Result<Vec<serde_json::Value>> {
        let conn = self.open()?;
        let mut stmt = conn.prepare(
            "SELECT step, tokens, train_loss, val_loss, generalization_gap, grad_norm, muon_lr, adamw_lr,
                    vram_used_gb, tok_per_sec, tflops, timestamp, current_epoch, epoch_pct, tokens_per_epoch,
                    tokens_in_epoch, step_time_ms, duration_mode, target_epochs, sched_phase, sched_factor,
                    eval_interval, next_val_step
             FROM metrics WHERE run_id = ?1 AND step >= ?2 ORDER BY step ASC LIMIT ?3",
        )?;
        let rows = stmt.query_map(params![run_id, start_step, limit], |row| {
            let tokens = row.get::<_, i64>(1)?;
            let tok_per_sec = row.get::<_, f64>(9).unwrap_or(0.0);
            let vram = row.get::<_, f64>(8).unwrap_or(0.0);
            Ok(serde_json::json!({
                "step": row.get::<_, i64>(0)?,
                "tokens": tokens,
                "tokens_seen": tokens,
                "loss": row.get::<_, f64>(2)?,
                "train_loss": row.get::<_, f64>(2)?,
                "val_loss": row.get::<_, Option<f64>>(3)?,
                "generalization_gap": row.get::<_, Option<f64>>(4)?,
                "grad_norm": row.get::<_, f64>(5).unwrap_or(0.0),
                "muon_lr": row.get::<_, f64>(6).unwrap_or(0.0),
                "adamw_lr": row.get::<_, f64>(7).unwrap_or(0.0),
                "learning_rate": row.get::<_, f64>(6).unwrap_or(0.0),
                "vram_used_gb": vram,
                "memory_allocated_mb": vram * 1024.0,
                "tok_per_sec": tok_per_sec,
                "tokens_per_second": tok_per_sec,
                "tflops": row.get::<_, f64>(10).unwrap_or(0.0),
                "timestamp": row.get::<_, f64>(11)?,
                "current_epoch": row.get::<_, Option<i64>>(12)?,
                "epoch_pct": row.get::<_, Option<f64>>(13)?,
                "tokens_per_epoch": row.get::<_, Option<i64>>(14)?,
                "tokens_in_epoch": row.get::<_, Option<i64>>(15)?,
                "step_time_ms": row.get::<_, Option<f64>>(16)?.unwrap_or(0.0),
                "duration_mode": row.get::<_, Option<String>>(17)?,
                "target_epochs": row.get::<_, Option<i64>>(18)?,
                "sched_phase": row.get::<_, Option<String>>(19)?,
                "sched_factor": row.get::<_, Option<f64>>(20)?,
                "eval_interval": row.get::<_, Option<i64>>(21)?,
                "next_val_step": row.get::<_, Option<i64>>(22)?,
            }))
        })?;

        let mut res = Vec::new();
        for r in rows.flatten() {
            res.push(r);
        }
        Ok(res)
    }

    pub fn save_metric(&self, run_id: &str, val: &serde_json::Value) -> Result<()> {
        let conn = self.open()?;
        let step = val.get("step").and_then(|v| v.as_i64()).unwrap_or(0);
        let tokens = val.get("tokens").and_then(|v| v.as_i64()).unwrap_or(0);
        let train_loss = val.get("loss").or_else(|| val.get("train_loss")).and_then(|v| v.as_f64()).unwrap_or(0.0);
        let val_loss = val.get("val_loss").and_then(|v| v.as_f64());
        let gen_gap = val.get("generalization_gap").and_then(|v| v.as_f64());
        let grad_norm = val.get("grad_norm").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let muon_lr = val.get("muon_lr").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let adamw_lr = val.get("adamw_lr").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let vram = val.get("vram_used_gb").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let tok_per_sec = val.get("tok_per_sec").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let tflops = val.get("tflops").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let ts = val.get("timestamp").and_then(|v| v.as_f64()).unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64()
        });
        conn.execute(
            "INSERT INTO metrics (run_id, step, tokens, train_loss, val_loss, generalization_gap, grad_norm, muon_lr, adamw_lr, vram_used_gb, tok_per_sec, tflops, timestamp)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![run_id, step, tokens, train_loss, val_loss, gen_gap, grad_norm, muon_lr, adamw_lr, vram, tok_per_sec, tflops, ts],
        )?;
        Ok(())
    }

    pub fn get_db_path(&self) -> &str {
        &self.db_path
    }

    pub fn mark_dataset_tokenized(
        &self,
        id: &str,
        token_count: i64,
        train_tokens: i64,
        val_tokens: i64,
    ) -> Result<()> {
        let conn = self.open()?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let stats = serde_json::json!({
            "is_tokenized": true,
            "token_count": token_count,
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "tokenized_at": now
        });

        conn.execute(
            "INSERT INTO datasets (id, name, file_path, format, token_count, sample_count, revision, stats_json, created_at)
             VALUES (?1, ?1, '', 'text', ?2, 0, '1', ?3, ?4)
             ON CONFLICT(id) DO UPDATE SET token_count = ?2, stats_json = ?3",
            params![id, token_count, stats.to_string(), now],
        )?;

        // Also write a marker under the writable data root for instant
        // zero-DB redundancy. Never use a path relative to the executable:
        // packaged builds do not run from the project directory.
        let cache_dir = crate::data_dir().join("data").join("cache");
        let _ = fs::create_dir_all(&cache_dir);
        let cache_marker = cache_dir.join(format!("{}.tokenized", id));
        let _ = fs::write(cache_marker, stats.to_string());

        Ok(())
    }

    pub fn get_dataset_meta(&self, id: &str) -> Option<(i64, String)> {
        let cache_dir = crate::data_dir().join("data").join("cache");
        let cache_marker = cache_dir.join(format!("{}.tokenized", id));
        let cache_bin = cache_dir.join(format!("{}_full_tokens.bin", id));

        // The binary shard is authoritative. A stale marker used to say
        // 1,144,246 tokens for every dataset; heal it from the real shard size.
        if let Ok(meta) = fs::metadata(&cache_bin) {
            let actual = (meta.len() / 8) as i64;
            if actual > 0 {
                let mut stats = fs::read_to_string(&cache_marker)
                    .ok()
                    .and_then(|content| serde_json::from_str::<serde_json::Value>(&content).ok())
                    .filter(|value| value.is_object())
                    .unwrap_or_else(|| serde_json::json!({}));
                let train = (actual as f64 * 0.9) as i64;
                stats["is_tokenized"] = serde_json::json!(true);
                stats["token_count"] = serde_json::json!(actual);
                stats["train_tokens"] = serde_json::json!(train);
                stats["val_tokens"] = serde_json::json!(actual - train);
                let content = stats.to_string();
                let _ = fs::write(&cache_marker, &content);
                return Some((actual, content));
            }
        }

        if let Ok(content) = fs::read_to_string(&cache_marker) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(tc) = val.get("token_count").and_then(|v| v.as_i64()) {
                    if tc > 0 {
                        return Some((tc, content));
                    }
                }
            }
        }

        let conn = self.open().ok()?;
        conn.query_row(
            "SELECT token_count, stats_json FROM datasets WHERE id = ?1 OR id LIKE ?1 || '_%' ORDER BY created_at DESC LIMIT 1",
            params![id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .ok()
    }

    pub fn get_settings(&self) -> Result<serde_json::Value> {
        let conn = self.open()?;
        let mut stmt = conn.prepare("SELECT key, value_json FROM settings")?;
        let rows = stmt.query_map([], |row| {
            let k: String = row.get(0)?;
            let v: String = row.get(1)?;
            Ok((k, v))
        })?;

        let mut map = serde_json::Map::new();
        for r in rows.flatten() {
            let mut val: serde_json::Value =
                serde_json::from_str(&r.1).unwrap_or(serde_json::Value::String(r.1));
            if is_secret_key(&r.0) {
                if let Some(stored) = val.as_str() {
                    let plaintext = if stored.is_empty() {
                        String::new()
                    } else {
                        self.secrets.decrypt(stored).map_err(|err| {
                            rusqlite::Error::InvalidParameterName(format!(
                                "cannot decrypt setting {}: {}",
                                r.0, err
                            ))
                        })?
                    };
                    map.insert(
                        format!("{}_configured", r.0),
                        serde_json::Value::Bool(!plaintext.is_empty()),
                    );
                    val = if plaintext.is_empty() {
                        serde_json::Value::String(String::new())
                    } else {
                        serde_json::Value::String(REDACTED.to_string())
                    };
                }
            }
            map.insert(r.0, val);
        }
        Ok(serde_json::Value::Object(map))
    }

    pub fn save_settings(&self, settings: &serde_json::Value) -> Result<()> {
        let conn = self.open()?;
        if let Some(obj) = settings.as_object() {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();
            for (k, v) in obj {
                // Presence flags are derived on read and are never persisted.
                if k.ends_with("_configured") {
                    continue;
                }
                let mut value = v.clone();
                if is_secret_key(k) {
                    let plaintext = match &value {
                        serde_json::Value::Null => "",
                        serde_json::Value::String(s) => s.as_str(),
                        _ => {
                            return Err(rusqlite::Error::InvalidParameterName(format!(
                                "setting {} must be a string",
                                k
                            )))
                        }
                    };
                    if plaintext == REDACTED {
                        continue;
                    }
                    let sealed = self.secrets.encrypt(plaintext).map_err(|err| {
                        rusqlite::Error::InvalidParameterName(format!(
                            "cannot encrypt setting {}: {}",
                            k, err
                        ))
                    })?;
                    value = serde_json::Value::String(sealed);
                }
                let v_str = serde_json::to_string(&value)
                    .map_err(|e| rusqlite::Error::InvalidParameterName(e.to_string()))?;
                conn.execute(
                    "INSERT OR REPLACE INTO settings (key, value_json, updated_at) VALUES (?1, ?2, ?3)",
                    params![k, v_str, now],
                )?;
            }
        }
        Ok(())
    }

    fn migrate_plaintext_secrets(&self, conn: &Connection) -> Result<()> {
        let mut stmt = conn.prepare("SELECT key, value_json FROM settings")?;
        let rows = stmt.query_map([], |row| {
            let k: String = row.get(0)?;
            let v: String = row.get(1)?;
            Ok((k, v))
        })?;
        let mut updates = Vec::new();
        for (key, raw) in rows.flatten() {
            if !is_secret_key(&key) {
                continue;
            }
            let Ok(serde_json::Value::String(value)) =
                serde_json::from_str::<serde_json::Value>(&raw)
            else {
                continue;
            };
            if value.is_empty() || SecretStore::is_encrypted(&value) {
                continue;
            }
            updates.push((key, value));
        }
        drop(stmt);
        for (key, plaintext) in updates {
            let sealed = self.secrets.encrypt(&plaintext).map_err(|err| {
                rusqlite::Error::InvalidParameterName(format!(
                    "cannot encrypt legacy setting {}: {}",
                    key, err
                ))
            })?;
            conn.execute(
                "UPDATE settings SET value_json = ?1 WHERE key = ?2",
                params![serde_json::to_string(&sealed).unwrap_or_default(), key],
            )?;
            log::info!("Encrypted legacy plaintext setting {}", key);
        }
        Ok(())
    }

    pub fn list_alerts(&self, run_id: Option<&str>, limit: i64) -> Result<Vec<serde_json::Value>> {
        let conn = self.open()?;
        if let Some(rid) = run_id {
            let mut stmt = conn.prepare("SELECT id, run_id, level, category, message, step, timestamp FROM alerts WHERE run_id = ?1 ORDER BY id DESC LIMIT ?2")?;
            let rows = stmt.query_map(params![rid, limit], |row| {
                let id: i64 = row.get(0)?;
                let r_id: Option<String> = row.get(1)?;
                let level: String = row.get(2)?;
                let category: String = row.get(3)?;
                let message: String = row.get(4)?;
                let step: i64 = row.get(5)?;
                let timestamp: f64 = row.get(6)?;
                Ok(serde_json::json!({
                    "id": id,
                    "run_id": r_id,
                    "level": level,
                    "category": category,
                    "message": message,
                    "step": step,
                    "timestamp": timestamp,
                    "created_at": timestamp,
                }))
            })?;
            Ok(rows.filter_map(|r| r.ok()).collect())
        } else {
            let mut stmt = conn.prepare("SELECT id, run_id, level, category, message, step, timestamp FROM alerts ORDER BY id DESC LIMIT ?1")?;
            let rows = stmt.query_map(params![limit], |row| {
                let id: i64 = row.get(0)?;
                let r_id: Option<String> = row.get(1)?;
                let level: String = row.get(2)?;
                let category: String = row.get(3)?;
                let message: String = row.get(4)?;
                let step: i64 = row.get(5)?;
                let timestamp: f64 = row.get(6)?;
                Ok(serde_json::json!({
                    "id": id,
                    "run_id": r_id,
                    "level": level,
                    "category": category,
                    "message": message,
                    "step": step,
                    "timestamp": timestamp,
                    "created_at": timestamp,
                }))
            })?;
            Ok(rows.filter_map(|r| r.ok()).collect())
        }
    }

    pub fn list_benchmarks(&self, run_id: Option<&str>, limit: i64) -> Result<Vec<serde_json::Value>> {
        let conn = self.open()?;
        if let Some(rid) = run_id {
            let mut stmt = conn.prepare("SELECT id, run_id, checkpoint_id, step, category, score, results_json, created_at FROM benchmarks WHERE run_id = ?1 ORDER BY id DESC LIMIT ?2")?;
            let rows = stmt.query_map(params![rid, limit], |row| {
                let id: i64 = row.get(0)?;
                let r_id: Option<String> = row.get(1)?;
                let chk_id: Option<String> = row.get(2)?;
                let step: i64 = row.get(3)?;
                let category: String = row.get(4)?;
                let score: f64 = row.get(5)?;
                let results_json: Option<String> = row.get(6)?;
                let created_at: f64 = row.get(7)?;
                let details: serde_json::Value = results_json
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or(serde_json::json!({}));
                Ok(serde_json::json!({
                    "id": id,
                    "run_id": r_id,
                    "checkpoint_id": chk_id,
                    "task_name": category.clone(),
                    "category": category,
                    "score": score,
                    "metric_name": "overall_score",
                    "step": step,
                    "details": details,
                    "created_at": created_at,
                }))
            })?;
            Ok(rows.filter_map(|r| r.ok()).collect())
        } else {
            let mut stmt = conn.prepare("SELECT id, run_id, checkpoint_id, step, category, score, results_json, created_at FROM benchmarks ORDER BY id DESC LIMIT ?1")?;
            let rows = stmt.query_map(params![limit], |row| {
                let id: i64 = row.get(0)?;
                let r_id: Option<String> = row.get(1)?;
                let chk_id: Option<String> = row.get(2)?;
                let step: i64 = row.get(3)?;
                let category: String = row.get(4)?;
                let score: f64 = row.get(5)?;
                let results_json: Option<String> = row.get(6)?;
                let created_at: f64 = row.get(7)?;
                let details: serde_json::Value = results_json
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or(serde_json::json!({}));
                Ok(serde_json::json!({
                    "id": id,
                    "run_id": r_id,
                    "checkpoint_id": chk_id,
                    "task_name": category.clone(),
                    "category": category,
                    "score": score,
                    "metric_name": "overall_score",
                    "step": step,
                    "details": details,
                    "created_at": created_at,
                }))
            })?;
            Ok(rows.filter_map(|r| r.ok()).collect())
        }
    }

    pub fn list_pipelines(&self) -> Result<Vec<serde_json::Value>> {
        let conn = self.open()?;
        let mut stmt = conn.prepare("SELECT id, name, status, current_phase_idx, phases_json, state_json, created_at, updated_at FROM pipelines ORDER BY updated_at DESC")?;
        let rows = stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let name: String = row.get(1)?;
            let status: String = row.get(2)?;
            let current_phase_idx: i64 = row.get(3)?;
            let phases_str: String = row.get(4)?;
            let state_str: Option<String> = row.get(5)?;
            let created_at: f64 = row.get(6)?;
            let updated_at: f64 = row.get(7)?;

            let phases: serde_json::Value = serde_json::from_str(&phases_str).unwrap_or(serde_json::json!([]));
            let state: serde_json::Value = state_str
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or(serde_json::json!({}));

            Ok(serde_json::json!({
                "id": id,
                "name": name,
                "status": status,
                "current_phase_index": current_phase_idx,
                "phases": phases,
                "state": state,
                "created_at": created_at,
                "updated_at": updated_at,
            }))
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    pub fn get_pipeline(&self, id: &str) -> Result<Option<serde_json::Value>> {
        let conn = self.open()?;
        let mut stmt = conn.prepare("SELECT id, name, status, current_phase_idx, phases_json, state_json, created_at, updated_at FROM pipelines WHERE id = ?1")?;
        let mut rows = stmt.query_map(params![id], |row| {
            let pid: String = row.get(0)?;
            let name: String = row.get(1)?;
            let status: String = row.get(2)?;
            let current_phase_idx: i64 = row.get(3)?;
            let phases_str: String = row.get(4)?;
            let state_str: Option<String> = row.get(5)?;
            let created_at: f64 = row.get(6)?;
            let updated_at: f64 = row.get(7)?;

            let phases: serde_json::Value = serde_json::from_str(&phases_str).unwrap_or(serde_json::json!([]));
            let state: serde_json::Value = state_str
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or(serde_json::json!({}));

            Ok(serde_json::json!({
                "id": pid,
                "name": name,
                "status": status,
                "current_phase_index": current_phase_idx,
                "phases": phases,
                "state": state,
                "created_at": created_at,
                "updated_at": updated_at,
            }))
        })?;
        if let Some(r) = rows.next() {
            Ok(Some(r?))
        } else {
            Ok(None)
        }
    }

    pub fn create_pipeline(&self, name: &str, phases: &serde_json::Value) -> Result<serde_json::Value> {
        let conn = self.open()?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        let id = format!("pipe_{}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs());
        let phases_str = serde_json::to_string(phases).unwrap_or("[]".to_string());
        conn.execute(
            "INSERT INTO pipelines (id, name, status, current_phase_idx, phases_json, state_json, created_at, updated_at) VALUES (?1, ?2, 'IDLE', 0, ?3, '{}', ?4, ?4)",
            params![id, name, phases_str, now],
        )?;
        Ok(serde_json::json!({
            "pipeline_id": id,
            "id": id,
            "name": name,
            "status": "IDLE",
            "current_phase_index": 0,
            "phases": phases,
        }))
    }
}

fn split_sql_statements(sql: &str) -> Vec<String> {
    let without_comments: String = sql
        .lines()
        .filter(|line| !line.trim_start().starts_with("--"))
        .collect::<Vec<_>>()
        .join("\n");
    without_comments
        .split(';')
        .map(|statement| statement.trim().to_string())
        .filter(|statement| !statement.is_empty())
        .collect()
}
