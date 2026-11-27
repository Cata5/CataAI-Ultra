# Desktop Integration Audit (Tauri 2 ↔ Python worker)

Scope: `src-tauri/src/{lib,commands,database,worker,hardware}.rs`, `src-tauri/tauri.conf.json`,
`src-tauri/capabilities/*`, and their contract with `src/worker/*`.

Build and startup verification performed on this machine:

* `cargo check --offline` → `Finished dev profile ... in 24.66s`, no errors.
* `cargo build --offline` → `Finished dev profile ... in 17.66s`, produced a 290 MB debug binary.
* The binary was launched against a **copy** of the production database in a temporary directory;
  logs and host state are shown below (no user data was touched).

```
[TRACE] Application starting...
[TRACE] Resolved project root: "\\?\C:\Users\Cata\AppData\Local\Temp\llmmaker-rust-smoke"
[TRACE] Initializing DatabaseManager at ...\studio.db...
[TRACE] Recovered 12 stale run(s) from a previous session
[TRACE] Initializing HardwareMonitor...
[TRACE] Initializing WorkerSupervisor...
[TRACE] WorkerSupervisor initialized: project_root=..., python_path="python", db_path=...
[TRACE] Building Tauri application...
[TRACE] Running Tauri context...
```

## 1. Startup sequence

1. `worker::project_root()` resolves the installation root: `LLMMAKER_PROJECT_ROOT` (validated by the
   presence of `src/worker/standalone_runner.py`), otherwise it walks up to 8 levels from the executable
   directory and then from the CWD.
2. The previous `tauri_debug.log` is rotated to `tauri_debug.prev.log`; a panic hook appends `[PANIC]`.
3. `DatabaseManager` opens `<root>/studio.db` (absolute), runs the schema migration and
   `recover_stale_runs()`.
4. `HardwareMonitor` (NVML/psutil) and `WorkerSupervisor` are created with the same absolute DB path.
5. The Tauri builder registers 54 commands and a run-event callback that calls `supervisor.shutdown()`
   on `ExitRequested`/`Exit`.

## 2. What was broken and fixed

### AUD-006 — hardcoded Python interpreter (High)

`WorkerSupervisor` spawned `D:\proiecte\CataAI Ultra\.venv\Scripts\python.exe`. The application could
only ever train on the machine where it was built. It now probes `<root>/.venv/Scripts/python.exe`,
`<root>/venv/Scripts/python.exe`, POSIX equivalents, and finally `python`/`python3` from `PATH`
(observed: `python_path="python"` when the venv was absent from the temporary root).

### AUD-008 — relative database path (High)

The shell opened `studio.db` relative to the process CWD while the worker defaulted to its own relative
`studio.db`; launching the binary from Explorer vs a terminal could produce two different databases and
"lost" runs. The shell now resolves the root once and passes `--db-path <absolute>` to the child, which
also sets `current_dir(root)`, `PYTHONPATH=root` and `PYTHONUNBUFFERED=1`.

### AUD-007 — status race between runs (High)

The stdout reader thread wrote `runs.status` for whatever `run_id` came in, including events left in the
pipe by a previous run, and could reset the state to `IDLE` after a new run had started. The reader now
captures the run id it was started for and ignores events for other runs; terminal events clear the
supervisor's current run id under a single lock order (child → stdin). Both `stop_training` and the
reader clear `stdin` while still holding the child lock, because `start_training` installs the next
worker's stdin under that lock — clearing it afterwards could close the pipe of a run that just started.

### AUD-002 — 800 ms kill destroyed graceful stop (Critical)

`stop_training` previously waited 800 ms and then `kill`ed the child, truncating the stop snapshot and
the final DB update. It now polls `try_wait` for up to 20 s, only then kills the process tree
(`taskkill /PID <pid> /T /F`), and clears stdin before releasing the child lock — verified by the lifecycle test
that stops a real run and finds the snapshot and `STOPPED` row.

### AUD-010 — manual validation was not wired (High)

There was no `validate_run` command and `api.validateRun` resolved a fake success. The command now sends
`RUN_VALIDATION` to the worker (which honors it through `manual_val_requested`); the frontend throws when
the worker rejects the request instead of silently pretending.

### AUD-009 / AUD-015 — schema drift and stale runs (High)

`list_runs` selects `COALESCE(final_loss, best_val_loss)`; the Python schema lacked `final_loss`, so the
desktop run list returned an SQL error after any Python-side run. Both schemas now contain the column
(with an `ALTER TABLE` migration for existing databases) and a startup recovery flips `STARTING`,
`RUNNING`, `PAUSING`, `PAUSED`, `STOPPING` rows to `STOPPED` with a fresh `updated_at` timestamp — the
12 stale rows found in the production database copy are exactly this failure mode.

### AUD-011 — SQL injection surface (High)

`list_runs`/`list_checkpoints` built `WHERE workspace_id = '<value>'` by string formatting. Both are now
parameterized (`WHERE (?1 IS NULL OR workspace_id = ?1)`) and covered by the desktop run list working
against the real database.

## 3. Shutdown and process hygiene

* The Tauri run loop handles `RunEvent::ExitRequested`/`Exit` and calls `supervisor.shutdown()`:
  `STOP_RUN` on stdin, up to 8 s wait, then process-tree kill. This prevents orphaned Python processes
  holding the GPU after the window closes (a real symptom observed earlier: `python.exe` surviving the
  app and blocking VRAM).
* `kill_process_tree` uses `taskkill /T /F` on Windows and falls back to `child.kill()` elsewhere.
* The supervisor refuses to start a second run while one is active and returns a clear error instead of
  corrupting state.

## 4. Database layer (`database.rs`)

* Creates/migrates `runs`, `metrics`, `checkpoints`, `datasets`, `workspaces`, `settings`, `alerts`,
  `pipelines`, `benchmarks`.
* `recover_stale_runs()` is idempotent and only touches non-terminal statuses.
* Workspace filters are optional and parameterized; the same method signatures back the 54 commands.
* Writes from Rust and Python target the same file. The database uses WAL (set by the Python layer),
  which allows one writer and concurrent readers. No corruption or `database is locked` error was
  observed, but long training runs write metrics from Python while the UI writes settings from Rust; the
  residual contention window is bounded by short transactions and rusqlite's default 5 s busy timeout.

## 5. Capabilities and security posture

* `capabilities/default.json` grants the webview `core:default` plus the application commands; there is
  no remote origin allowed, so the IPC surface is only reachable from the bundled frontend.
* The dev configuration loads `http://localhost:3000`; a release build uses the embedded `frontendDist`.
* The Python child inherits the user environment; it is not sandboxed. It executes only code shipped
  in the repository, but it does write to the project directory (DB, checkpoints, charts).

## 6. Remaining desktop limitations

1. `.cargo/config.toml` pins `target-dir` to `C:/Users/Cata/.cargo/target_llm_maker` and the GNU target
   triple (AUD-022). Removing it is not safe on this machine because `windres` fails on paths containing
   spaces, which is why the redirect exists.
2. Launching the binary from a directory without a detectable project root falls back to the CWD and can
   create a fresh `studio.db` there. `LLMMAKER_PROJECT_ROOT` overrides this.
3. The shell does not verify that the spawned Python matches the expected version/dependencies; a
   missing `torch` surfaces as a worker that exits immediately with a stderr trace in `tauri_debug.log`
   only after the first start attempt.
4. No automated Rust tests exist; `cargo check`/`cargo build` and the manual startup smoke test are the
   current safety net.

## Verification pass (2026-11-27): Tauri status path

* **Finding:** `frontend/src/services/api.ts` `getSystemStatus` (desktop branch) returned `latest_metrics.learning_rate: 0` on every call, while the HTTP branch returns the real learning rate. The Rust `WorkerStatus` had no learning-rate field.
* **Fix:** `src-tauri/src/worker.rs` adds `current_lr: Option<f64>`, parses `learning_rate` (or `lr`) from the same step events that set `current_step` and `current_loss`, and clears it on run start. The frontend returns `ws.current_lr ?? 0`.
* **Verification:** `cargo check` passes. `tsc --noEmit`, `next build`, and 11 Vitest tests pass. The end-to-end desktop flow has not been run (see `known_issues.md` ENV-001 and AUD-036).
* **Also fixed:** `api.ts` used `const module = await import(...)`, which fails lint (`@next/next/no-assign-module-variable`). Renamed to `tauriCore`.
* **Behavior to know:** every desktop IPC call falls back silently to HTTP after a `console.warn`. If the HTTP server is down too, the UI shows the HTTP error, but the original IPC error is only in the console.
