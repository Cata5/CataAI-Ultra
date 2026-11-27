//! src-tauri/src/lib.rs
//! LLM Maker Tauri 2 Application Entrypoint

pub mod commands;
pub mod database;
pub mod hardware;
pub mod secrets;
pub mod worker;

use commands::*;
use database::DatabaseManager;
use hardware::HardwareMonitor;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use worker::WorkerSupervisor;

use tauri::Manager;

const ICON_RGBA_128: &[u8] = include_bytes!("../icons/icon_128.rgba");

/// Writable per-user data root. Packaged builds never write into Program Files.
pub fn data_dir() -> PathBuf {
    if let Ok(explicit) = std::env::var("LLMMAKER_DATA_DIR") {
        if !explicit.trim().is_empty() {
            let path = PathBuf::from(explicit);
            let _ = std::fs::create_dir_all(&path);
            return path;
        }
    }
    if cfg!(debug_assertions) {
        return worker::project_root();
    }
    #[cfg(target_os = "windows")]
    {
        if let Ok(base) = std::env::var("LOCALAPPDATA") {
            if !base.trim().is_empty() {
                let path = PathBuf::from(base).join("LLMMaker");
                let _ = std::fs::create_dir_all(&path);
                return path;
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(home) = std::env::var_os("HOME") {
            let path = PathBuf::from(home).join("Library/Application Support/LLMMaker");
            let _ = std::fs::create_dir_all(&path);
            return path;
        }
    }
    let path = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".local/share"))
        .join("llm-maker");
    let _ = std::fs::create_dir_all(&path);
    path
}

fn log_path() -> &'static PathBuf {
    static LOG_PATH: OnceLock<PathBuf> = OnceLock::new();
    LOG_PATH.get_or_init(|| {
        let dir = data_dir().join("logs");
        let _ = std::fs::create_dir_all(&dir);
        dir.join("tauri_debug.log")
    })
}

fn migrate_legacy_database(target: &Path) {
    if target.exists() {
        return;
    }
    let root = worker::project_root();
    for candidate in [root.join("studio.db"), root.join("data").join("rocanon_studio.db")] {
        if candidate.exists() && candidate != target {
            if let Some(parent) = target.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            match std::fs::copy(&candidate, target) {
                Ok(_) => {
                    let _ = std::fs::write(
                        target.with_extension("migrated-from.txt"),
                        candidate.to_string_lossy().as_bytes(),
                    );
                    log_trace(&format!("Migrated legacy database {:?} -> {:?}", candidate, target));
                }
                Err(err) => log_trace(&format!("Legacy database migration failed: {}", err)),
            }
            return;
        }
    }
}

pub fn log_trace(msg: &str) {
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(log_path()) {
        let _ = writeln!(f, "[TRACE] {}", msg);
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Keep Python from writing __pycache__ into the read-only packaged resource
    // tree; NSIS cannot remove files it did not install.
    std::env::set_var("PYTHONDONTWRITEBYTECODE", "1");
    let root = worker::project_root();
    let run_data_dir = data_dir();
    let _ = std::fs::create_dir_all(run_data_dir.join("logs"));
    let debug_log = log_path().clone();
    if debug_log.exists() {
        let _ = std::fs::rename(&debug_log, debug_log.with_extension("prev.log"));
    }
    std::panic::set_hook(Box::new(|info| {
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(log_path()) {
            let _ = writeln!(f, "[PANIC] {:?}", info);
        }
    }));

    log_trace("Application starting...");
    log_trace(&format!("Resolved project root: {:?}", root));
    log_trace(&format!("Resolved data root: {:?}", run_data_dir));
    let _ = env_logger::try_init();

    // Absolute database path: a relative one silently created a fresh database
    // (losing every run) whenever the process working directory changed.
    let db_path = std::env::var("LLMMAKER_DB_PATH")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| run_data_dir.join("studio.db"));
    migrate_legacy_database(&db_path);
    log_trace(&format!("Initializing DatabaseManager at {:?}...", db_path));
    let db = DatabaseManager::new(&db_path.to_string_lossy()).unwrap_or_else(|err| {
        log_trace(&format!("FATAL: database initialization failed: {}", err));
        panic!("database initialization failed: {}", err);
    });

    match db.recover_stale_runs() {
        Ok(0) => {}
        Ok(count) => log_trace(&format!("Recovered {} stale run(s) from a previous session", count)),
        Err(err) => log_trace(&format!("Stale run recovery failed: {}", err)),
    }

    log_trace("Initializing HardwareMonitor...");
    let hw = Mutex::new(HardwareMonitor::new());

    log_trace("Initializing WorkerSupervisor...");
    let sup = WorkerSupervisor::new(db_path, run_data_dir.clone());

    let state = AppState {
        db,
        hardware: hw,
        supervisor: sup,
    };

    log_trace("Building Tauri application...");
    let builder = tauri::Builder::default()
        .setup(|app| {
            if let Some(window) = app.get_webview_window("main") {
                let img = tauri::image::Image::new(ICON_RGBA_128, 128, 128);
                let _ = window.set_icon(img);
            }
            Ok(())
        })
        .manage(state)
        .invoke_handler(tauri::generate_handler![
            get_hardware_telemetry,
            get_runtime_status,
            verify_runtime,
            run_setup,
            cancel_setup,
            list_workspaces,
            get_active_workspace,
            set_active_workspace,
            create_workspace,
            delete_workspace,
            list_runs,
            get_run_metrics,
            get_worker_status,
            start_training,
            stop_training,
            pause_run,
            resume_run,
            checkpoint_run,
            validate_run,
            update_lr,
            list_datasets,
            upload_dataset,
            tokenize_dataset,
            curate_dataset,
            clean_dataset,
            audit_dataset_quality,
            get_dataset_operation,
            list_models,
            export_gguf,
            delete_dataset,
            get_dataset_stats,
            inspect_tokenizer,
            list_checkpoints,
            delete_checkpoint,
            list_networks,
            get_network,
            get_network_templates,
            clone_network,
            delete_network,
            save_network,
            estimate_network,
            validate_network,
            get_foundation_presets,
            get_compute_topology,
            get_settings,
            save_settings,
            save_local_model,
            get_scheduler_curve,
            generate_evaluation,
            compare_evaluation,
            get_alerts,
            list_benchmarks,
            run_benchmark_suite,
            list_pipelines,
            get_pipeline,
            create_pipeline,
            get_custom_modules,
            recommend_budget,
            import_foundation_model,
            export_ollama,
            test_notification,
            get_system_logs,
            clear_system_logs
        ]);

    log_trace("Running Tauri context...");
    let app = match builder.build(tauri::generate_context!()) {
        Ok(app) => app,
        Err(e) => {
            log_trace(&format!("Tauri build error: {:?}", e));
            return;
        }
    };

    // Never leave an orphaned Python/CUDA process behind when the window closes.
    app.run(|app_handle, event| {
        if matches!(
            event,
            tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit
        ) {
            if let Some(state) = app_handle.try_state::<AppState>() {
                state.supervisor.shutdown();
            }
        }
    });
}
