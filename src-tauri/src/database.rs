//! src-tauri/src/database.rs
//! Native SQLite Database Manager for LLM Maker in Rust.
//! Provides instant (<1ms) workspace, run, and metric queries without launching Python.

use rusqlite::{params, Connection, Result};
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
}

impl DatabaseManager {
    pub fn new(db_path: &str) -> Self {
        let mgr = Self {
            db_path: db_path.to_string(),
        };
        let _ = mgr.init_tables();
        mgr
    }

    fn open(&self) -> Result<Connection> {
        let p = Path::new(&self.db_path);
        if let Some(parent) = p.parent() {
            let _ = fs::create_dir_all(parent);
        }
        Connection::open(&self.db_path)
    }

    pub fn init_tables(&self) -> Result<()> {
        let conn = self.open()?;
        conn.execute_batch(
            r#"
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
                stats_json TEXT DEFAULT '{}',
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
                metadata_json TEXT DEFAULT '{}',
                created_at REAL NOT NULL
            );

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
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            "#,
        )?;

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

        conn.execute("DELETE FROM workspaces WHERE id = ?1", params![id])?;
        conn.execute("DELETE FROM runs WHERE workspace_id = ?1", params![id])?;
        conn.execute("DELETE FROM datasets WHERE workspace_id = ?1", params![id])?;
        conn.execute("DELETE FROM checkpoints WHERE workspace_id = ?1", params![id])?;

        let ws_dir = format!("workspaces/{}", id);
        let _ = fs::remove_dir_all(&ws_dir);

        Ok(())
    }

    pub fn list_runs(&self, workspace_id: Option<&str>) -> Result<Vec<RunSummary>> {
        let conn = self.open()?;
        let query = if let Some(wid) = workspace_id {
            format!("SELECT id, name, current_phase, status, final_loss, total_steps, created_at, workspace_id FROM runs WHERE workspace_id = '{}' ORDER BY created_at DESC", wid)
        } else {
            "SELECT id, name, current_phase, status, final_loss, total_steps, created_at, workspace_id FROM runs ORDER BY created_at DESC".to_string()
        };

        let mut stmt = conn.prepare(&query)?;
        let rows = stmt.query_map([], |row| {
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
        let query = if let Some(wid) = workspace_id {
            format!("SELECT id, run_id, step, tokens, train_loss, val_loss, file_path, file_size_bytes, phase, workspace_id, created_at FROM checkpoints WHERE workspace_id = '{}' ORDER BY step DESC", wid)
        } else {
            "SELECT id, run_id, step, tokens, train_loss, val_loss, file_path, file_size_bytes, phase, workspace_id, created_at FROM checkpoints ORDER BY step DESC".to_string()
        };

        let mut stmt = conn.prepare(&query)?;
        let rows = stmt.query_map([], |row| {
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
            "SELECT step, tokens, train_loss, val_loss, generalization_gap, grad_norm, muon_lr, adamw_lr, vram_used_gb, tok_per_sec, tflops, timestamp
             FROM metrics WHERE run_id = ?1 AND step >= ?2 ORDER BY step ASC LIMIT ?3",
        )?;
        let rows = stmt.query_map(params![run_id, start_step, limit], |row| {
            Ok(serde_json::json!({
                "step": row.get::<_, i64>(0)?,
                "tokens": row.get::<_, i64>(1)?,
                "loss": row.get::<_, f64>(2)?,
                "train_loss": row.get::<_, f64>(2)?,
                "val_loss": row.get::<_, Option<f64>>(3)?,
                "generalization_gap": row.get::<_, Option<f64>>(4)?,
                "grad_norm": row.get::<_, f64>(5).unwrap_or(0.0),
                "muon_lr": row.get::<_, f64>(6).unwrap_or(0.0),
                "adamw_lr": row.get::<_, f64>(7).unwrap_or(0.0),
                "learning_rate": row.get::<_, f64>(6).unwrap_or(0.0),
                "vram_used_gb": row.get::<_, f64>(8).unwrap_or(0.0),
                "tok_per_sec": row.get::<_, f64>(9).unwrap_or(0.0),
                "tflops": row.get::<_, f64>(10).unwrap_or(0.0),
                "timestamp": row.get::<_, f64>(11)?,
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

        // Also write file marker in data/cache/{id}.tokenized for instant zero-DB redundancy
        let _ = fs::create_dir_all("data/cache");
        let cache_marker = format!("data/cache/{}.tokenized", id);
        let _ = fs::write(&cache_marker, stats.to_string());

        Ok(())
    }

    pub fn get_dataset_meta(&self, id: &str) -> Option<(i64, String)> {
        // Fast disk cache check
        let cache_marker = format!("data/cache/{}.tokenized", id);
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
            let val: serde_json::Value = serde_json::from_str(&r.1).unwrap_or(serde_json::Value::String(r.1));
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
                let v_str = serde_json::to_string(v).unwrap_or_default();
                conn.execute(
                    "INSERT OR REPLACE INTO settings (key, value_json, updated_at) VALUES (?1, ?2, ?3)",
                    params![k, v_str, now],
                )?;
            }
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
                    "created_at": format!("{}", timestamp as i64),
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
                    "created_at": format!("{}", timestamp as i64),
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
                    "created_at": format!("{}", created_at as i64),
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
                    "created_at": format!("{}", created_at as i64),
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
