//! src-tauri/src/worker.rs
//! Dedicated Lazy Python Worker Lifecycle Supervisor.
//! Guarantees 0 MB idle Python/CUDA memory by keeping the ML worker dormant until explicitly triggered.

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use tauri::{AppHandle, Emitter};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStatus {
    pub state: String,
    pub run_id: Option<String>,
    pub current_step: i64,
    pub current_loss: Option<f64>,
    pub is_running: bool,
}

pub struct WorkerSupervisor {
    child: Arc<Mutex<Option<Child>>>,
    stdin: Arc<Mutex<Option<ChildStdin>>>,
    status: Arc<Mutex<WorkerStatus>>,
    python_path: PathBuf,
}

impl WorkerSupervisor {
    pub fn new() -> Self {
        let mut candidates = Vec::new();

        // 1. Project-local .venv relative to current directory
        if cfg!(target_os = "windows") {
            candidates.push(PathBuf::from(".venv/Scripts/python.exe"));
        } else {
            candidates.push(PathBuf::from(".venv/bin/python"));
        }

        // 2. Relative to executable location
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                if cfg!(target_os = "windows") {
                    candidates.push(parent.join(".venv/Scripts/python.exe"));
                    candidates.push(parent.join("../.venv/Scripts/python.exe"));
                } else {
                    candidates.push(parent.join(".venv/bin/python"));
                    candidates.push(parent.join("../.venv/bin/python"));
                }
            }
        }

        // 3. Known project workspace paths
        candidates.push(PathBuf::from(r"D:\proiecte\CataAI Ultra\.venv\Scripts\python.exe"));
        candidates.push(PathBuf::from(r"C:\proiecte\CataAI Ultra\.venv\Scripts\python.exe"));

        let python_path = candidates
            .into_iter()
            .find(|p| p.exists())
            .unwrap_or_else(|| PathBuf::from("python"));

        crate::log_trace(&format!("WorkerSupervisor initialized with python_path: {:?}", python_path));

        Self {
            child: Arc::new(Mutex::new(None)),
            stdin: Arc::new(Mutex::new(None)),
            status: Arc::new(Mutex::new(WorkerStatus {
                state: "IDLE".to_string(),
                run_id: None,
                current_step: 0,
                current_loss: None,
                is_running: false,
            })),
            python_path,
        }
    }

    pub fn get_status(&self) -> WorkerStatus {
        self.status.lock().unwrap().clone()
    }

    pub fn send_command(&self, cmd: &serde_json::Value) -> Result<(), String> {
        let mut stdin_guard = self.stdin.lock().unwrap();
        if let Some(ref mut stdin) = *stdin_guard {
            let line = format!("{}\n", serde_json::to_string(cmd).map_err(|e| e.to_string())?);
            stdin.write_all(line.as_bytes()).map_err(|e| format!("Failed to write to worker stdin: {}", e))?;
            stdin.flush().map_err(|e| format!("Failed to flush worker stdin: {}", e))?;
            crate::log_trace(&format!("Dispatched command to Python worker: {}", line.trim()));
            log::info!("Dispatched command to Python worker: {}", line.trim());
            Ok(())
        } else {
            Err("Worker is not running or stdin is unavailable".to_string())
        }
    }

    pub fn start_training(
        &self,
        app: AppHandle,
        payload_json: String,
    ) -> Result<String, String> {
        crate::log_trace("start_training command invoked in WorkerSupervisor");

        let mut child_guard = self.child.lock().unwrap();
        if let Some(ref mut c) = *child_guard {
            if let Ok(Some(_)) = c.try_wait() {
                // Previous worker exited; clear guard
                crate::log_trace("Clearing previous terminated worker child process");
                *child_guard = None;
            } else {
                crate::log_trace("Rejected start_training: Training worker is already active");
                return Err("Training worker is already active".to_string());
            }
        }

        let run_id = format!(
            "run_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );

        crate::log_trace(&format!("Starting Python training worker for run {}", run_id));
        log::info!("Starting Python training worker for run {}", run_id);

        // Resolve project root directory
        let project_dir = if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                if parent.join("src").exists() {
                    parent.to_path_buf()
                } else if parent.join("../src").exists() {
                    parent.join("..").canonicalize().unwrap_or_else(|_| parent.to_path_buf())
                } else {
                    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
                }
            } else {
                std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
            }
        } else {
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        };

        crate::log_trace(&format!("Project root directory resolved to: {:?}", project_dir));

        // Write payload to a temporary file to prevent Windows CLI argument escaping issues
        let payload_file = project_dir.join(format!(".tmp_run_{}.json", run_id));
        if let Err(e) = std::fs::write(&payload_file, &payload_json) {
            crate::log_trace(&format!("Failed to write payload file: {}", e));
            return Err(format!("Failed to write run payload file: {}", e));
        }

        let payload_file_str = payload_file.to_string_lossy().to_string();

        let mut cmd = Command::new(&self.python_path);
        cmd.current_dir(&project_dir);
        cmd.args(&[
            "-u",
            "-m",
            "src.worker.standalone_runner",
            "--payload",
            &payload_file_str,
        ]);

        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            let _ = std::fs::remove_file(&payload_file);
            let err_msg = format!("Failed to spawn Python worker with {:?}: {}", self.python_path, e);
            crate::log_trace(&err_msg);
            err_msg
        })?;

        let stdin = child.stdin.take();
        let stdout = child.stdout.take().ok_or("Failed to capture worker stdout")?;
        let stderr = child.stderr.take();

        *self.stdin.lock().unwrap() = stdin;
        *child_guard = Some(child);

        {
            let mut st = self.status.lock().unwrap();
            st.state = "RUNNING".to_string();
            st.run_id = Some(run_id.clone());
            st.is_running = true;
            st.current_step = 0;
            st.current_loss = None;
        }

        // Emit initial STATE_CHANGED event to Tauri webview immediately
        let _ = app.emit(
            "tauri://training-metrics",
            serde_json::json!({
                "event_type": "STATE_CHANGED",
                "run_id": run_id,
                "payload": {
                    "state": "RUNNING",
                    "new_state": "RUNNING",
                    "run_id": run_id,
                }
            }),
        );

        // Spawn background reader thread for stderr to prevent pipe buffer deadlocks and log all issues
        if let Some(err_pipe) = stderr {
            thread::spawn(move || {
                let reader = BufReader::new(err_pipe);
                for line_res in reader.lines() {
                    if let Ok(line) = line_res {
                        crate::log_trace(&format!("[PYTHON STDERR] {}", line));
                        log::warn!("[PYTHON STDERR] {}", line);
                    }
                }
            });
        }

        // Spawn background reader thread for streaming telemetry from worker stdout to Tauri UI
        let status_arc = Arc::clone(&self.status);
        let child_arc = Arc::clone(&self.child);
        let stdin_arc = Arc::clone(&self.stdin);
        let payload_file_cleanup = payload_file.clone();

        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line_res in reader.lines() {
                if let Ok(line) = line_res {
                    let trimmed = line.trim();
                    if trimmed.starts_with('{') && trimmed.ends_with('}') {
                        if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
                            if let Some(step) = val.get("step").and_then(|s| s.as_i64()) {
                                if let Ok(mut st) = status_arc.lock() {
                                    st.current_step = step;
                                    st.current_loss = val.get("loss").or_else(|| val.get("train_loss")).and_then(|l| l.as_f64());
                                    st.state = "RUNNING".to_string();
                                    st.is_running = true;
                                }
                            }
                            let event_type = val.get("event_type").or_else(|| val.get("type")).and_then(|s| s.as_str());
                            if let Some(et) = event_type {
                                if et == "STATE_CHANGED" || et == "RUN_STARTED" {
                                    let new_state = val.get("payload")
                                        .and_then(|p| p.get("state").or_else(|| p.get("new_state")))
                                        .and_then(|s| s.as_str())
                                        .unwrap_or("RUNNING");
                                    if let Ok(mut st) = status_arc.lock() {
                                        st.state = new_state.to_string();
                                        st.is_running = new_state == "RUNNING";
                                    }
                                } else if et == "RUN_COMPLETED" || et == "RUN_STOPPED" || et == "RUN_FAILED" {
                                    if let Ok(mut st) = status_arc.lock() {
                                        st.state = if et == "RUN_COMPLETED" {
                                            "COMPLETED".to_string()
                                        } else if et == "RUN_FAILED" {
                                            "FAILED".to_string()
                                        } else {
                                            "IDLE".to_string()
                                        };
                                        st.is_running = false;
                                    }
                                }
                            }
                            let _ = app.emit("tauri://training-metrics", val);
                        }
                    }
                }
            }

            // Cleanup payload temp file
            let _ = std::fs::remove_file(&payload_file_cleanup);

            // When stdout closes, worker has exited
            if let Ok(mut sg) = stdin_arc.lock() {
                *sg = None;
            }
            if let Ok(mut guard) = child_arc.lock() {
                if let Some(mut c) = guard.take() {
                    let _ = c.wait();
                }
            }
            if let Ok(mut st) = status_arc.lock() {
                st.state = "IDLE".to_string();
                st.is_running = false;
            }
            let _ = app.emit("tauri://run-stopped", ());
            crate::log_trace("Python training worker exited. Memory reclaimed.");
            log::info!("Python training worker exited cleanly. Memory reclaimed.");
        });

        Ok(run_id)
    }

    pub fn stop_training(&self) -> Result<(), String> {
        let _ = self.send_command(&serde_json::json!({
            "command_type": "STOP_RUN"
        }));

        {
            let mut stdin_guard = self.stdin.lock().unwrap();
            *stdin_guard = None;
        }

        let mut child_guard = self.child.lock().unwrap();
        if let Some(mut child) = child_guard.take() {
            crate::log_trace("Terminating Python training worker process...");
            log::info!("Terminating Python training worker process...");
            let start = std::time::Instant::now();
            let mut exited = false;
            while start.elapsed().as_millis() < 800 {
                if let Ok(Some(_)) = child.try_wait() {
                    exited = true;
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            if !exited {
                let _ = child.kill();
                let _ = child.wait();
            }
        }

        let mut st = self.status.lock().unwrap();
        st.state = "IDLE".to_string();
        st.is_running = false;
        Ok(())
    }
}
