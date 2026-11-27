//! src-tauri/src/worker.rs
//! Dedicated Lazy Python Worker Lifecycle Supervisor.
//! Guarantees 0 MB idle Python/CUDA memory by keeping the ML worker dormant until explicitly triggered.

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter};

/// Resolves the application root that contains the Python sources.
///
/// Packaged builds and `cargo run` start with different working directories, so the
/// root is discovered by walking up from the executable and the current directory
/// until `src/worker/standalone_runner.py` is found. Set `LLMMAKER_PROJECT_ROOT` to
/// override (useful for portable installs).
pub fn project_root() -> PathBuf {
    if let Ok(env_root) = std::env::var("LLMMAKER_PROJECT_ROOT") {
        let candidate = PathBuf::from(env_root);
        if candidate.join("src").join("worker").join("standalone_runner.py").is_file() {
            return candidate.canonicalize().unwrap_or(candidate);
        }
        let packaged = candidate.join("resources").join("app");
        if packaged.join("src").join("worker").join("standalone_runner.py").is_file() {
            return packaged.canonicalize().unwrap_or(packaged);
        }
    }
    let mut starts: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            starts.push(parent.to_path_buf());
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        starts.push(cwd);
    }
    for start in starts {
        // Installed builds keep the staged Python app under resources/app next
        // to the executable. Check that layout before walking parents.
        for packaged in [start.join("resources").join("app"), start.join("app")] {
            if packaged.join("src").join("worker").join("standalone_runner.py").is_file() {
                return packaged.canonicalize().unwrap_or(packaged);
            }
        }
        let mut cursor = Some(start);
        for _ in 0..8 {
            let Some(dir) = cursor else { break };
            if dir.join("src").join("worker").join("standalone_runner.py").is_file() {
                return dir.canonicalize().unwrap_or(dir);
            }
            cursor = dir.parent().map(|p| p.to_path_buf());
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStatus {
    pub state: String,
    pub run_id: Option<String>,
    pub current_step: i64,
    pub current_loss: Option<f64>,
    pub current_lr: Option<f64>,
    pub is_running: bool,
}

pub struct WorkerSupervisor {
    child: Arc<Mutex<Option<Child>>>,
    stdin: Arc<Mutex<Option<ChildStdin>>>,
    status: Arc<Mutex<WorkerStatus>>,
    python_path: PathBuf,
    project_root: PathBuf,
    db_path: PathBuf,
    data_dir: PathBuf,
    runtime_manifest: Option<PathBuf>,
}

fn read_runtime_python(data_dir: &Path) -> (Option<PathBuf>, Option<PathBuf>) {
    let manifest = data_dir.join("runtime").join("runtime.json");
    let Ok(raw) = std::fs::read_to_string(&manifest) else {
        return (None, None);
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return (None, Some(manifest));
    };
    if value.get("schema").and_then(|v| v.as_i64()) != Some(1) {
        return (None, Some(manifest));
    }
    let python = value
        .get("python")
        .and_then(|v| v.as_str())
        .map(PathBuf::from)
        .filter(|path| path.is_file());
    (python, Some(manifest))
}

fn preflight_script() -> &'static str {
    r#"
import json, sys
result = {"python": sys.version.split()[0], "executable": sys.executable, "torch": None, "cuda": False, "error": None}
try:
    import torch
    result["torch"] = torch.__version__
    result["cuda"] = bool(torch.cuda.is_available())
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"
print("LLMMAKER_PREFLIGHT=" + json.dumps(result))
"#
}

/// Kills a worker and any grandchildren it may have spawned.
fn kill_process_tree(child: &mut Child) {
    let pid = child.id();
    #[cfg(target_os = "windows")]
    {
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    let _ = child.kill();
    let _ = child.wait();
    crate::log_trace(&format!("Training worker process tree {} terminated", pid));
}

impl WorkerSupervisor {
    pub fn new(db_path: PathBuf, data_dir: PathBuf) -> Self {
        let project_root = project_root();
        let mut candidates: Vec<PathBuf> = Vec::new();

        // 0. Explicit operator override (packaged/portable installs).
        if let Ok(explicit) = std::env::var("LLMMAKER_PYTHON") {
            if !explicit.trim().is_empty() {
                candidates.push(PathBuf::from(explicit));
            }
        }

        // 1. First-run wizard manifest.
        let (runtime_python, runtime_manifest) = read_runtime_python(&data_dir);
        if let Some(ref python) = runtime_python {
            candidates.push(python.clone());
        }

        // 2. Project-local virtual environments
        if cfg!(target_os = "windows") {
            candidates.push(project_root.join(".venv/Scripts/python.exe"));
            candidates.push(project_root.join("venv/Scripts/python.exe"));
        } else {
            candidates.push(project_root.join(".venv/bin/python"));
            candidates.push(project_root.join("venv/bin/python"));
        }

        // 3. PATH fallback is a developer convenience only. A release build must
        // use a venv, runtime.json or LLMMAKER_PYTHON, otherwise a poisoned PATH
        // can select a different interpreter (AUD-034).
        #[cfg(debug_assertions)]
        candidates.push(PathBuf::from(if cfg!(target_os = "windows") { "python" } else { "python3" }));

        let python_path = candidates
            .into_iter()
            .find(|p| !p.is_absolute() || p.exists())
            .unwrap_or_else(|| {
                log::error!("No Python runtime found; set LLMMAKER_PYTHON to an absolute interpreter path");
                PathBuf::from("<python-not-found>")
            });

        crate::log_trace(&format!(
            "WorkerSupervisor initialized: project_root={:?}, data_dir={:?}, python_path={:?}, db_path={:?}, runtime_manifest={:?}",
            project_root, data_dir, python_path, db_path, runtime_manifest
        ));

        Self {
            child: Arc::new(Mutex::new(None)),
            stdin: Arc::new(Mutex::new(None)),
            status: Arc::new(Mutex::new(WorkerStatus {
                state: "IDLE".to_string(),
                run_id: None,
                current_step: 0,
                current_loss: None,
                current_lr: None,
                is_running: false,
            })),
            python_path,
            project_root,
            db_path,
            data_dir,
            runtime_manifest,
        }
    }

    /// Checks the selected interpreter before spawning a worker. A missing torch
    /// used to surface only as a dead child process and a stderr trace.
    pub fn preflight(&self) -> Result<String, String> {
        let python = self.resolve_python();
        if !python.is_file() {
            return Err(format!(
                "No Python runtime found at {:?}. Run the setup wizard or set LLMMAKER_PYTHON.",
                python
            ));
        }
        let output = Command::new(&python)
            .args(["-c", preflight_script()])
            .env("PYTHONNOUSERSITE", "1")
            .output()
            .map_err(|err| format!("Cannot start {:?}: {err}", self.python_path))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Python preflight failed: {}", stderr.trim()));
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let line = stdout
            .lines()
            .find(|line| line.starts_with("LLMMAKER_PREFLIGHT="))
            .ok_or_else(|| "Python preflight returned no result".to_string())?;
        let value: serde_json::Value = serde_json::from_str(line.trim_start_matches("LLMMAKER_PREFLIGHT="))
            .map_err(|err| format!("Invalid preflight output: {err}"))?;
        let version = value.get("python").and_then(|v| v.as_str()).unwrap_or("unknown");
        let major = version.split('.').next().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
        let minor = version.split('.').nth(1).and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
        if (major, minor) < (3, 11) {
            return Err(format!("Python {version} is too old; 3.11+ is required. Run the setup wizard."));
        }
        if let Some(error) = value.get("error").and_then(|v| v.as_str()) {
            return Err(format!(
                "PyTorch is not usable in {:?}: {error}. Run the setup wizard or `python -m installer install`.",
                self.python_path
            ));
        }
        let torch = value.get("torch").and_then(|v| v.as_str()).unwrap_or("unknown");
        let cuda = value.get("cuda").and_then(|v| v.as_bool()).unwrap_or(false);
        Ok(format!("Python {version}, torch {torch}, CUDA {}", if cuda { "available" } else { "CPU" }))
    }

    pub fn get_status(&self) -> WorkerStatus {
        self.status.lock().unwrap().clone()
    }

    /// Resolves the runtime manifest again so a just-completed setup is used
    /// without restarting the desktop shell. Falls back to startup resolution.
    pub fn resolve_python(&self) -> PathBuf {
        read_runtime_python(&self.data_dir)
            .0
            .filter(|path| path.is_file())
            .unwrap_or_else(|| self.python_path.clone())
    }

    pub fn python_path(&self) -> PathBuf {
        self.resolve_python()
    }

    pub fn data_dir(&self) -> PathBuf {
        self.data_dir.clone()
    }

    pub fn runtime_manifest(&self) -> Option<PathBuf> {
        self.runtime_manifest.clone()
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
        self.preflight()?;

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

        // Resolve project root directory (works from src-tauri/target and packaged layouts)
        let project_dir = self.project_root.clone();

        crate::log_trace(&format!("Project root directory resolved to: {:?}", project_dir));

        // Write payload to a temporary file to prevent Windows CLI argument escaping issues.
        // The install directory may be read-only, so use the writable data root.
        let tmp_dir = self.data_dir.join("tmp");
        if let Err(e) = std::fs::create_dir_all(&tmp_dir) {
            return Err(format!("Failed to create worker temp directory: {e}"));
        }
        let payload_file = tmp_dir.join(format!("run_{}.json", run_id));
        if let Err(e) = std::fs::write(&payload_file, &payload_json) {
            crate::log_trace(&format!("Failed to write payload file: {}", e));
            return Err(format!("Failed to write run payload file: {}", e));
        }

        let payload_file_str = payload_file.to_string_lossy().to_string();

        let mut cmd = Command::new(self.resolve_python());
        cmd.current_dir(&project_dir);
        let db_path_str = self.db_path.to_string_lossy().to_string();
        cmd.args(&[
            "-u",
            "-m",
            "src.worker.standalone_runner",
            "--payload",
            &payload_file_str,
            "--db-path",
            &db_path_str,
        ]);
        // Keep the worker importable from the resolved project root regardless of CWD.
        cmd.env("PYTHONPATH", &project_dir);
        cmd.env("PYTHONUNBUFFERED", "1");
        cmd.env("PYTHONNOUSERSITE", "1");
        cmd.env("LLMMAKER_DATA_DIR", &self.data_dir);
        cmd.env("LLMMAKER_DB_PATH", &self.db_path);
        if let Some(ref manifest) = self.runtime_manifest {
            cmd.env("LLMMAKER_RUNTIME_MANIFEST", manifest);
        }

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
            st.current_lr = None;
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
            let app_err = app.clone();
            thread::spawn(move || {
                let reader = BufReader::new(err_pipe);
                for line_res in reader.lines() {
                    if let Ok(line) = line_res {
                        crate::log_trace(&format!("[PYTHON STDERR] {}", line));
                        log::warn!("[PYTHON STDERR] {}", line);
                        let lower = line.to_lowercase();
                        let level = if lower.contains("error") || lower.contains("traceback") || lower.contains("exception") || line.contains("RuntimeError") || line.contains("Error:") {
                            "ERROR"
                        } else if lower.contains("warn") {
                            "WARN"
                        } else {
                            "INFO"
                        };
                        let _ = app_err.emit("tauri://system-log", serde_json::json!({
                            "level": level,
                            "source": "WORKER",
                            "message": line.clone(),
                            "raw": format!("[PYTHON STDERR] {}", line),
                        }));
                    }
                }
            });
        }

        // Spawn background reader thread for streaming telemetry from worker stdout to Tauri UI
        let status_arc = Arc::clone(&self.status);
        let child_arc = Arc::clone(&self.child);
        let stdin_arc = Arc::clone(&self.stdin);
        let payload_file_cleanup = payload_file.clone();
        let reader_run_id = run_id.clone();

        thread::spawn(move || {
            // A previous worker's reader thread may finish after a new run started;
            // only mutate status that still belongs to this run id.
            let is_current = |st: &WorkerStatus| st.run_id.as_deref() == Some(reader_run_id.as_str());
            let reader = BufReader::new(stdout);
            for line_res in reader.lines() {
                if let Ok(line) = line_res {
                    let trimmed = line.trim();
                    if trimmed.starts_with('{') && trimmed.ends_with('}') {
                        if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
                            if let Some(step) = val.get("step").and_then(|s| s.as_i64()) {
                                if let Ok(mut st) = status_arc.lock() {
                                    if is_current(&st) {
                                        st.current_step = step;
                                        st.current_loss = val.get("loss").or_else(|| val.get("train_loss")).and_then(|l| l.as_f64());
                                        st.current_lr = val.get("learning_rate").or_else(|| val.get("lr")).and_then(|l| l.as_f64());
                                        st.state = "RUNNING".to_string();
                                        st.is_running = true;
                                    }
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
                                        if is_current(&st) {
                                            st.state = new_state.to_string();
                                            st.is_running = new_state == "RUNNING";
                                        }
                                    }
                                } else if et == "RUN_COMPLETED" || et == "RUN_STOPPED" || et == "RUN_FAILED" {
                                    if let Ok(mut st) = status_arc.lock() {
                                        if is_current(&st) {
                                            st.state = if et == "RUN_COMPLETED" {
                                                "COMPLETED".to_string()
                                            } else if et == "RUN_FAILED" {
                                                "FAILED".to_string()
                                            } else {
                                                "IDLE".to_string()
                                            };
                                            st.is_running = false;
                                            st.run_id = None;
                                        }
                                    }
                                }
                            }
                            let _ = app.emit("tauri://training-metrics", val);
                        }
                    } else if !trimmed.is_empty() {
                        crate::log_trace(&format!("[PYTHON STDOUT] {}", line));
                        let _ = app.emit("tauri://system-log", serde_json::json!({
                            "level": "INFO",
                            "source": "WORKER",
                            "message": line.clone(),
                            "raw": format!("[PYTHON STDOUT] {}", line),
                        }));
                    }
                }
            }

            // Cleanup payload temp file
            let _ = std::fs::remove_file(&payload_file_cleanup);

            // When stdout closes, worker has exited. Lock order (child, then stdin) must
            // match every other site to avoid deadlocking with start/stop commands.
            if let Ok(mut guard) = child_arc.lock() {
                if let Some(mut c) = guard.take() {
                    let _ = c.wait();
                }
                // Hold the child lock while clearing stdin (child -> stdin order):
                // otherwise a start_training that wins the child lock in between
                // would have its new worker's stdin closed by this cleanup.
                if let Ok(mut sg) = stdin_arc.lock() {
                    *sg = None;
                }
            }
            if let Ok(mut st) = status_arc.lock() {
                if is_current(&st) {
                    st.state = "IDLE".to_string();
                    st.is_running = false;
                    st.run_id = None;
                }
            }
            let _ = app.emit("tauri://run-stopped", ());
            crate::log_trace("Python training worker exited. Memory reclaimed.");
            log::info!("Python training worker exited cleanly. Memory reclaimed.");
        });

        Ok(run_id)
    }

    /// Gracefully stops the run, saving a resumable snapshot inside the worker.
    ///
    /// The worker needs time to write a ~200 MB checkpoint and flush metrics, so a
    /// short wait followed by `Child::kill` (the previous behaviour) truncated the
    /// snapshot and lost the terminal run status.
    pub fn stop_training(&self) -> Result<(), String> {
        let stopping_run_id = self.get_status().run_id;
        let _ = self.send_command(&serde_json::json!({
            "command_type": "STOP_RUN"
        }));

        let mut child_guard = self.child.lock().unwrap();
        if let Some(child) = child_guard.as_mut() {
            crate::log_trace("Waiting for the Python training worker to finalize the stop...");
            log::info!("Waiting for the Python training worker to finalize the stop...");
            let start = Instant::now();
            let mut exited = false;
            while start.elapsed() < Duration::from_secs(20) {
                match child.try_wait() {
                    Ok(Some(_)) => {
                        exited = true;
                        break;
                    }
                    Ok(None) => thread::sleep(Duration::from_millis(100)),
                    Err(_) => break,
                }
            }
            if !exited {
                log::warn!("Training worker did not finish stopping within 20s; terminating it.");
                kill_process_tree(child);
            }
        }
        *child_guard = None;
        // Clear the pipe while still holding the child lock: start_training takes
        // that same lock before installing a new worker's stdin, so doing this
        // afterwards could close the *next* run's stdin.
        {
            let mut stdin_guard = self.stdin.lock().unwrap();
            *stdin_guard = None;
        }
        drop(child_guard);

        // Only clear status that still belongs to the run we stopped; a new run may
        // have been started while this command was waiting for the old process.
        let mut st = self.status.lock().unwrap();
        if st.run_id == stopping_run_id {
            st.state = "IDLE".to_string();
            st.is_running = false;
            st.run_id = None;
        }
        Ok(())
    }

    /// Requests an out-of-band validation sweep on the running worker.
    pub fn validate_run(&self) -> Result<(), String> {
        self.send_command(&serde_json::json!({
            "command_type": "RUN_VALIDATION"
        }))
    }

    /// Best-effort cleanup invoked when the desktop application exits so the GPU is
    /// never left allocated by an orphaned Python process.
    pub fn shutdown(&self) {
        let running = self.child.lock().map(|g| g.is_some()).unwrap_or(false);
        if !running {
            return;
        }
        crate::log_trace("Application exit: stopping the training worker");
        let _ = self.send_command(&serde_json::json!({
            "command_type": "STOP_RUN"
        }));
        let deadline = Instant::now() + Duration::from_secs(8);
        loop {
            let mut guard = self.child.lock().unwrap();
            let finished = match guard.as_mut() {
                None => true,
                Some(child) => match child.try_wait() {
                    Ok(Some(_)) => true,
                    Err(_) => true,
                    Ok(None) => false,
                },
            };
            if finished {
                let _ = guard.take();
                break;
            }
            drop(guard);
            if Instant::now() >= deadline {
                if let Ok(mut g) = self.child.lock() {
                    if let Some(child) = g.as_mut() {
                        kill_process_tree(child);
                    }
                    let _ = g.take();
                }
                break;
            }
            thread::sleep(Duration::from_millis(100));
        }
        if let Ok(mut sg) = self.stdin.lock() {
            *sg = None;
        }
    }
}
