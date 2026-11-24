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
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use worker::WorkerSupervisor;

use tauri::Manager;

const ICON_RGBA_128: &[u8] = include_bytes!("../icons/icon_128.rgba");

fn log_path() -> &'static PathBuf {
    static LOG_PATH: OnceLock<PathBuf> = OnceLock::new();
    LOG_PATH.get_or_init(|| worker::project_root().join("tauri_debug.log"))
}

pub fn log_trace(msg: &str) {
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(log_path()) {
        let _ = writeln!(f, "[TRACE] {}", msg);
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let root = worker::project_root();
    let debug_log = root.join("tauri_debug.log");
    if debug_log.exists() {
        let _ = std::fs::rename(&debug_log, root.join("tauri_debug.prev.log"));
    }
    std::panic::set_hook(Box::new(|info| {
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(log_path()) {
            let _ = writeln!(f, "[PANIC] {:?}", info);
        }
    }));

    log_trace("Application starting...");
    log_trace(&format!("Resolved project root: {:?}", root));
    let _ = env_logger::try_init();

    // Absolute database path: a relative one silently created a fresh database
    // (losing every run) whenever the process working directory changed.
    let db_path = root.join("studio.db");
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
    let sup = WorkerSupervisor::new(db_path);

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
