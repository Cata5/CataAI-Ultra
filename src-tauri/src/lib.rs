//! src-tauri/src/lib.rs
//! LLM Maker Tauri 2 Application Entrypoint

pub mod commands;
pub mod database;
pub mod hardware;
pub mod worker;

use commands::*;
use database::DatabaseManager;
use hardware::HardwareMonitor;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::Mutex;
use worker::WorkerSupervisor;

use tauri::Manager;

const ICON_RGBA_128: &[u8] = include_bytes!("../icons/icon_128.rgba");

pub fn log_trace(msg: &str) {
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open("tauri_debug.log") {
        let _ = writeln!(f, "[TRACE] {}", msg);
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    if std::path::Path::new("tauri_debug.log").exists() {
        let _ = std::fs::rename("tauri_debug.log", "tauri_debug.prev.log");
    }
    std::panic::set_hook(Box::new(|info| {
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open("tauri_debug.log") {
            let _ = writeln!(f, "[PANIC] {:?}", info);
        }
    }));

    log_trace("Application starting...");
    let _ = env_logger::try_init();

    log_trace("Initializing DatabaseManager...");
    let db = DatabaseManager::new("studio.db");

    log_trace("Initializing HardwareMonitor...");
    let hw = Mutex::new(HardwareMonitor::new());

    log_trace("Initializing WorkerSupervisor...");
    let sup = WorkerSupervisor::new();

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
    if let Err(e) = builder.run(tauri::generate_context!()) {
        log_trace(&format!("Tauri run error: {:?}", e));
    }
}
