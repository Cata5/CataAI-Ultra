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
        // Resolve python executable in project .venv
        let venv_py = if cfg!(target_os = "windows") {
            PathBuf::from(".venv/Scripts/python.exe")
        } else {
            PathBuf::from(".venv/bin/python")
        };

        let python_path = if venv_py.exists() {
            venv_py
        } else {
            PathBuf::from("python")
        };

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
        let mut child_guard = self.child.lock().unwrap();
        if child_guard.is_some() {
            return Err("Training worker is already active".to_string());
        }

        let run_id = format!("run_{}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs());

        log::info!("Starting Python training worker for run {}", run_id);

        let mut cmd = Command::new(&self.python_path);
        cmd.args(&[
            "-u",
            "-m",
            "src.worker.standalone_runner",
            "--payload",
            &payload_json,
        ]);

        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| format!("Failed to spawn Python worker: {}", e))?;
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().ok_or("Failed to capture worker stdout")?;

        *self.stdin.lock().unwrap() = stdin;
        *child_guard = Some(child);

        {
            let mut st = self.status.lock().unwrap();
            st.state = "RUNNING".to_string();
            st.run_id = Some(run_id.clone());
            st.is_running = true;
        }

        // Spawn background reader thread for streaming telemetry to Tauri UI
        let status_arc = Arc::clone(&self.status);
        let child_arc = Arc::clone(&self.child);
        let stdin_arc = Arc::clone(&self.stdin);

        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line_res in reader.lines() {
                if let Ok(line) = line_res {
                    if line.starts_with('{') && line.ends_with('}') {
                        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
                            if let Some(step) = val.get("step").and_then(|s| s.as_i64()) {
                                if let Ok(mut st) = status_arc.lock() {
                                    st.current_step = step;
                                    st.current_loss = val.get("loss").and_then(|l| l.as_f64());
                                }
                            }
                            if let Some(event_type) = val.get("event_type").and_then(|s| s.as_str()) {
                                if event_type == "STATE_CHANGED" {
                                    if let Some(new_state) = val.get("payload").and_then(|p| p.get("state")).and_then(|s| s.as_str()) {
                                        if let Ok(mut st) = status_arc.lock() {
                                            st.state = new_state.to_string();
                                        }
                                    }
                                }
                            }
                            let _ = app.emit("tauri://training-metrics", val);
                        }
                    }
                }
            }

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
