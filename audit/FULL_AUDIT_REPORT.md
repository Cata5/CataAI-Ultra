# LLM Maker — Full Technical & Functional Audit

**Repository:** `D:\proiecte\CataAI Ultra` (branch `experimental`, worktree dirty with this audit's fixes)
**Date:** 2026-11-24
**Scope:** complete application — Next.js frontend, Tauri 2 desktop shell, Rust data layer, Python training
engine, aiohttp control plane, SQLite persistence, worker IPC, checkpoint/dataset handling.
**Method:** static reading of every module, route and command; live execution of the unit/integration
suites, a real 120-step training run, the compiled desktop binary and the HTTP server. No GUI
click-through automation was possible in this environment (see Verification gaps).

**Post-audit follow-up:** distribution and data-safety work (single versioned schema with
pre-migration backups, bootstrap script, CI, install/data-safety docs) is tracked separately in
`audit/pillar_1_2_status.md`.

---

## 1. Executive summary

LLM Maker is a substantial, working product: 21.6 k lines of Python, 3 k lines of Rust, 13 k lines of
TypeScript, 54 desktop commands, 79 HTTP routes, 25 test files. The core training loop is sound — real
optimizers, real schedulers, real validation, real checkpoint state — and the application starts, opens
its database, recovers stale runs, spawns the worker and trains.

**The audit found four critical defects that made the product unsafe or unreliable, and fixed all four:**

1. **Arbitrary code execution through checkpoints.** The loader attempted `weights_only=True` but fell
   back to unrestricted pickle, and the package exporter used unrestricted pickle outright. Any shared
   checkpoint could run code on resume. (AUD-001)
2. **Stop destroyed the run it was supposed to save.** The desktop killed the worker 800 ms after
   `STOP_RUN`; the stop snapshot and final DB write rarely finished. (AUD-002)
3. **The stop snapshot paired best-validation weights with current optimizer state.** Resuming it
   rewound the model and corrupted training. (AUD-003)
4. **A failed resume silently trained from random weights** and reported success. (AUD-004)

Beyond those, the audit fixed 20 high/medium/low defects: a terminal-event race that truncated database
writes, a hardcoded Python path that made the app machine-specific, a cross-run status race, a schema
drift that broke the entire desktop run list, a fake "validate" button, SQL string interpolation, a
hardcoded TFLOPs constant, machine-specific dataset/checkpoint paths, an opaque CUDA crash on
vocabulary mismatch, orphaned checkpoint rows, interval-0 handling, adaptive-LR overrides, stale-run
recovery, and a local API that any website could read and control.

**Measured results after the fixes:** 218/218 Python tests (142 unit, 18 API, 58 backend integration),
18/18 lifecycle checks, a real 120-step run at 48.4 ms/step and 10,547 tokens/s, `tsc` clean, `next build` clean, `cargo check`/`build`
clean, and a desktop startup that recovered 12 stale runs from a production database copy.

**Overall health: 7.5/10 — good, with the critical class closed.** Remaining risk is concentrated in
areas that need product decisions or migrations (duplicate-aware run creation, token semantics, secret
storage, frontend test coverage), not in the training engine.

## 2. Overall health

| Area | Grade | Basis |
| --- | --- | --- |
| Architecture | B | Clear layering, but the Rust and Python data layers duplicate the schema with no shared contract |
| Training engine | A− | Correct optimizer/scheduler/validation/resume mechanics; verified by real runs |
| Desktop integration | B+ | Now portable and race-free on the tested paths; environment assumptions remain |
| Frontend | B+ | 14 routes build and type-check; no automated UI tests, several swallowed errors |
| Data layer | B− | Schema drift fixed, but `INSERT OR REPLACE`, relative Python default, and no FK enforcement remain |
| Security | B+ | Critical RCE and cross-site API access closed; remote-code import and plaintext secrets remain by design/decision |
| Performance | A− (smoke scale) | 48 ms/step, 10.5k tok/s, 0.11 GB for 1.88 M params; large-model behaviour not measured |
| Test coverage | C+ | 218 Python tests + 3 real worker scenarios; zero frontend and zero Rust tests |

## 3. Findings at a glance

| ID | Severity | Component | Status |
| --- | --- | --- | --- |
| AUD-001 | Critical | Checkpoint loader | Fixed |
| AUD-002 | Critical | Worker stop (Rust) | Fixed |
| AUD-003 | Critical | Stop snapshot | Fixed |
| AUD-004 | Critical | Resume failure | Fixed |
| AUD-005 | High | Worker exit/session durability | Fixed |
| AUD-006 | High | Interpreter discovery | Fixed |
| AUD-007 | High | Worker event routing | Fixed |
| AUD-008 | High | Database path resolution | Fixed |
| AUD-009 | High | Schema drift (`final_loss`) | Fixed |
| AUD-010 | High | Manual validation | Fixed |
| AUD-011 | High | SQL construction (Rust) | Fixed |
| AUD-015 | High | Stale run recovery | Fixed |
| AUD-016 | High | TFLOPs metric | Fixed |
| AUD-017 | High | Dataset preset paths | Fixed |
| AUD-018 | High | Vocabulary mismatch | Fixed |
| AUD-012 | Medium | Interval parsing | Fixed |
| AUD-013 | Medium | Adaptive LR | Fixed |
| AUD-014 | Medium | Hardcoded checkpoint search | Fixed |
| AUD-019 | Medium | Checkpoint deletion | Fixed |
| AUD-020 | Medium | `create_run` semantics | Fixed |
| AUD-021 | Medium | Python default DB path | Fixed |
| AUD-022 | Medium | Cargo target config | Open |
| AUD-023 | Medium | Token accounting (SFT) | Fixed |
| AUD-024 | Low | Duplicate dependency | Fixed |
| AUD-025 | Low | Dead loss accounting | Fixed |
| AUD-026 | Low | Unused import | Fixed |
| AUD-027 | Medium | Frontend tests | Open |
| AUD-028 | Low | Duplicated runner exit logic | Fixed |
| AUD-029 | Low | Resume token target | Fixed |
| AUD-030 | Low | VRAM metric semantics | Fixed |
| AUD-031 | High | Local API exposure | Fixed |
| AUD-032 | Medium | Remote code import | Fixed |
| AUD-033 | Medium | Plaintext secrets | Fixed |
| AUD-034 | Low | `PATH` interpreter fallback | Fixed |

Totals: 32 fixed, 2 open, 0 accepted. The two open items (AUD-022 Cargo target pin, AUD-027 frontend
tests) are documented with evidence and a recommended fix in `known_issues.md`.

## 4. Critical findings (all fixed)

### AUD-001 — Checkpoints execute arbitrary code on load

* **Severity / component:** Critical / `src/checkpoints/manager.py` (all load paths).
* **Repro:** craft a `.pt` whose pickled object defines `__reduce__` to run a command; resume it or open
  it in the checkpoint list. Before the fix the command executed with the user's privileges.
* **Root cause:** `load_checkpoint` wrapped `torch.load(weights_only=True)` in `try`, and on any failure
  called `torch.load(weights_only=False)` as a "compatibility fallback"; `save_local_model_package`
  used `weights_only=False` unconditionally. A prior audit (BUG-01) added `weights_only=True` but left
  the fallback in place, and its test only covered a plain tensor, so the dangerous branch was never
  exercised.
* **Files modified:** `src/checkpoints/manager.py`.
* **Fix:** one `safe_torch_load()` helper that loads with `weights_only=True` and raises a descriptive
  `RuntimeError` on failure — no unsafe fallback, no silent recovery. Every load site (resume, package
  export, listing, comparison) now uses it.
* **Verification:** `tests/unit/test_audit_regressions.py::test_malicious_checkpoint_is_not_deserialized`
  writes a checkpoint whose `__reduce__` creates a marker file and asserts the load raises and the file
  is absent; the plain-tensor case still loads.
* **Test result:** PASS.

### AUD-002 — Stop killed the worker before it could save

* **Severity / component:** Critical / `src-tauri/src/worker.rs` (`stop_training`).
* **Repro:** start a run, press Stop; the process disappeared after ~800 ms, `RUN_STOPPED` was sometimes
  emitted but the snapshot/final row were missing, leaving the run `RUNNING`/`STOPPING`.
* **Root cause:** `stop_training` slept 800 ms then called `kill()`, assuming the worker finished; the
  worker needs to finish the current step, write a multi-hundred-MB snapshot and update SQLite.
* **Files modified:** `src-tauri/src/worker.rs`.
* **Fix:** poll `try_wait` for up to 20 s, only then kill the process tree via `taskkill /T /F`; clear
  stdin state afterwards. The worker still writes its snapshot before emitting the terminal event.
* **Verification:** lifecycle test session 1 stops a real run and asserts exit code 0, a `STOPPED` row,
  a non-null `final_loss`, the step matching the last metric, and exactly one resumable snapshot.
* **Test result:** PASS (18/18 checks).

### AUD-003 — Stop snapshot mixed best weights with current optimizer state

* **Severity / component:** Critical / `src/worker/training_worker.py` (stop handler).
* **Repro:** let validation find a better checkpoint, then stop. The saved `stopped_snapshot_*.pt`
  contained the best-validation weights but the optimizer/scheduler state of the current step; resuming
  silently rewound the model while the optimizer kept its momentum.
* **Root cause:** the stop path reused the "export best model" code path, which loads `best_model.pt`
  weights, instead of the live `self.model`; it also referenced `snapshot_path.name` on a `str`.
* **Files modified:** `src/worker/training_worker.py`.
* **Fix:** the stop snapshot now serializes the live model, live optimizers and scheduler, with the real
  current losses; the path handling uses the actual `Path`.
* **Verification:** lifecycle session 1/2 — the snapshot resumes and the first new step is strictly
  greater than the stop step, then the run completes.
* **Test result:** PASS.

### AUD-004 — Failed resume silently trained from random weights

* **Severity / component:** Critical / `src/worker/training_worker.py` (resume block).
* **Repro:** point a run at a truncated or missing checkpoint. Before the fix the worker logged a warning,
  kept the freshly initialized model and trained to completion, marking the run `COMPLETED`.
* **Root cause:** the resume call was wrapped in a broad `try/except Exception` that only logged; there
  was no state transition or alert, so failure was indistinguishable from "no checkpoint requested".
* **Files modified:** `src/worker/training_worker.py`.
* **Fix:** on resume failure, log the error, write a `CHECKPOINT_LOAD_FAILED` CRITICAL alert, transition
  to `FAILED`, emit `RUN_FAILED` and return without training.
* **Verification:** lifecycle session 3 resumes a corrupted file and asserts `RUN_FAILED`, a `FAILED`
  row, the alert, and **zero** training steps.
* **Test result:** PASS.

## 5. High findings (all fixed)

### AUD-005 — Terminal event raced process exit, truncating final writes

* **Component:** `src/worker/standalone_runner.py`, `src/worker/training_worker.py`.
* **Repro:** stop or complete a run repeatedly; sometimes the run row stayed `STOPPING`/`RUNNING` and the
  final checkpoint/charts were missing. The audited database copy contained 12 non-terminal rows — the
  signature of sessions that never got to write their final state.
* **Root cause:** the runner waited a fixed 0.5 s after printing the terminal event and then called
  `os._exit(0)`, while the session still had to write the database status, snapshot and charts.
* **Files modified:** `src/worker/standalone_runner.py`, `src/worker/training_worker.py`.
* **Fix:** the worker exposes a `threading.Event` (`session_done`) set in `run_loop`'s `finally` after the
  session has fully finalized; the runner waits up to 30 s for it before exiting. The final status write
  now includes step, tokens, best-val and `final_loss` in one call.
* **Verification:** lifecycle sessions 1–3 assert exit code 0 plus the persisted row/snapshot/alert.
* **Test result:** PASS.

### AUD-006 — Hardcoded Python interpreter made the app single-machine

* **Component:** `src-tauri/src/worker.rs`.
* **Repro:** run the binary on any machine whose project is not at `D:\proiecte\CataAI Ultra`; the worker
  fails to spawn.
* **Root cause:** the interpreter path was a literal `D:\proiecte\CataAI Ultra\.venv\Scripts\python.exe`.
* **Files modified:** `src-tauri/src/worker.rs`.
* **Fix:** probe `<root>/.venv/Scripts/python.exe`, `<root>/venv/Scripts/python.exe`, POSIX equivalents,
  then `python`/`python3`; the resolved path is logged.
* **Verification:** startup smoke in a temp root without a venv logged `python_path="python"` and the
  worker started; `cargo check`/`build` clean.
* **Test result:** PASS (startup log + build).

### AUD-007 — A stopped run's late events could overwrite the next run

* **Component:** `src-tauri/src/worker.rs` (stdout reader).
* **Repro:** stop a run and immediately start another; a queued `RUN_STOPPED`/`STATE_CHANGED IDLE` from the
  old process could mark the new run `IDLE`.
* **Root cause:** the reader applied every event to the supervisor's *current* run id, which had already
  changed.
* **Files modified:** `src-tauri/src/worker.rs`.
* **Fix:** each reader thread carries the run id it was started for and ignores other ids; terminal
  handling clears the current run under a single lock order. `stop_training` and the reader now clear
  the stdin pipe while still holding the child lock, so a worker started in that window cannot lose
  its stdin.
* **Verification:** code review of every lock acquisition (child → stdin order), plus a clean
  `cargo build`. The Rust supervisor has no automated test and the GUI was not driven, so this is
  verified by inspection rather than by a failing-then-passing test.
* **Test result:** PASS by code review + build; runtime race not reproduced automatically.

### AUD-008 — Relative database path could split runs across two databases

* **Component:** `src-tauri/src/lib.rs`, `src-tauri/src/worker.rs`.
* **Repro:** launch the binary from a terminal in another directory; a fresh empty `studio.db` is created
  and the UI shows no runs.
* **Root cause:** the shell opened `studio.db` relative to the CWD and the worker defaulted to its own
  relative path; the two could disagree.
* **Files modified:** `src-tauri/src/lib.rs`, `src-tauri/src/worker.rs`.
* **Fix:** resolve the project root once, open `<root>/studio.db` absolutely, pass `--db-path <absolute>`,
  and set the child's `current_dir`, `PYTHONPATH` and `PYTHONUNBUFFERED`.
* **Verification:** startup log shows the absolute DB path; the smoke test read the copied production DB
  (25 runs) rather than creating an empty one.
* **Test result:** PASS.

### AUD-009 — Missing `final_loss` column broke the desktop run list

* **Component:** `src/db/database.py`, `src-tauri/src/database.rs`.
* **Repro:** create a run through the Python worker, then open `/runs` in the desktop app; `list_runs`
  fails with `no such column: final_loss`.
* **Root cause:** the Rust layer selected `COALESCE(final_loss, best_val_loss)` after a schema change that
  only the Rust migration performed; Python's `CREATE TABLE`/migration never added the column.
* **Files modified:** `src/db/database.py`, `src-tauri/src/database.rs`.
* **Fix:** add `final_loss` to the Python schema and an idempotent `ALTER TABLE` migration; both writers
  persist it; `list_runs` reads it.
* **Verification:** regression test builds a legacy schema, runs the migration, writes and reads the
  value; the desktop smoke test listed 25 runs from the copied database.
* **Test result:** PASS.

### AUD-010 — The Validate button was a no-op

* **Component:** `frontend/src/services/api.ts`, `src-tauri/src/commands.rs`, `src-tauri/src/lib.rs`.
* **Repro:** press Validate during a run in desktop mode; the UI reports success, the worker never runs a
  validation.
* **Root cause:** the Tauri branch returned `{success: true}` without invoking anything, and no
  `validate_run` command existed.
* **Files modified:** `frontend/src/services/api.ts`, `src-tauri/src/commands.rs`, `src-tauri/src/lib.rs`.
* **Fix:** added `validate_run` (sends `RUN_VALIDATION`), registered it, and made the frontend throw when
  the worker rejects the request.
* **Verification:** lifecycle session 2 observes `VALIDATION_COMPLETED` after resume; the command is
  compiled and registered.
* **Test result:** PASS (worker path + build); the desktop button itself was not clicked in a webview.

### AUD-011 — SQL injection through workspace ids

* **Component:** `src-tauri/src/database.rs`.
* **Repro:** call `list_runs` with `workspace_id = "x' OR 1=1 --"`; the filter is bypassed (and arbitrary
  SQL can be appended).
* **Root cause:** `format!("WHERE workspace_id = '{}'", id)`.
* **Files modified:** `src-tauri/src/database.rs`.
* **Fix:** bound parameters with `WHERE (?1 IS NULL OR workspace_id = ?1)` for runs and checkpoints.
* **Verification:** desktop run list works against the real database; code review confirms no interpolated
  SQL remains in the module.
* **Test result:** PASS (functional) — no Rust unit test for the injection string.

### AUD-015 — Stale `RUNNING` rows survived crashes forever

* **Component:** `src-tauri/src/lib.rs`, `src-tauri/src/database.rs`.
* **Repro:** kill the app during training; on next launch the run is still `RUNNING` and the UI offers no
  controls for it.
* **Root cause:** no startup reconciliation existed.
* **Files modified:** `src-tauri/src/lib.rs`, `src-tauri/src/database.rs`.
* **Fix:** `recover_stale_runs()` sets non-terminal statuses to `STOPPED` with a fresh timestamp at
  startup and logs the count.
* **Verification:** the smoke test against a production DB copy logged `Recovered 12 stale run(s)` and the
  seeded `RUNNING` row became `STOPPED`.
* **Test result:** PASS.

### AUD-016 — TFLOPs was a constant pretending to be a measurement

* **Component:** `src/worker/training_worker.py`.
* **Repro:** train any architecture other than ~59 M parameters; the TFLOPs card is wrong by orders of
  magnitude.
* **Root cause:** `tflops = tok_per_sec * 354.11e6 / 1e12`, with `354.11e6` hardcoded as the model size.
* **Files modified:** `src/worker/training_worker.py`.
* **Fix:** count trainable parameters after model/LoRA construction and compute `6 × params × tokens/s`,
  the standard dense-training estimate; log the count at startup.
* **Verification:** benchmark reports 1,876,736 params and 0.119 TFLOPs, matching `6·N·T` exactly; the
  lifecycle test asserts `tflops > 0`.
* **Test result:** PASS.

### AUD-017 — Built-in datasets pointed at the developer's old machine

* **Component:** `src/data/custom_loader.py`, `src/dashboard/server.py`.
* **Repro:** run on a clean machine and select `tech_wiki_ro`; the preset cannot be found.
* **Root cause:** `LEGACY_DATASETS_DIR = Path("d:/proiecte/LLM from Scratch/data")` and presets bound to
  those absolute paths.
* **Files modified:** `src/data/custom_loader.py`, `src/dashboard/server.py`.
* **Fix:** filenames-only presets resolved through `LLMMAKER_DATASETS_DIR`, `<root>/data`, CWD `data/`,
  then the documented legacy directory; `resolve_dataset_path()` raises a `FileNotFoundError` that lists
  every searched location; `/api/dataset_info` uses it.
* **Verification:** regression test points `LLMMAKER_DATASETS_DIR` at a temp dir and resolves a preset;
  live HTTP check returned `exists=true` for the existing preset.
* **Test result:** PASS.

### AUD-018 — Vocabulary mismatch surfaced as a CUDA kernel crash

* **Component:** `src/worker/training_worker.py`.
* **Repro:** select a 4,096-vocab network with the 12,288-vocab tokenizer and a dataset containing ids up
  to 12,230; training aborts with `CUDA error: device-side assert triggered` inside RMSNorm.
* **Root cause:** no validation that dataset token ids fit the model's embedding table; the first
  embedding lookup triggers the assert.
* **Files modified:** `src/worker/training_worker.py`.
* **Fix:** before step 1, compare `train_inputs.max()` with the model vocabulary and fail with
  `TOKENIZER_MISMATCH` plus an actionable message.
* **Verification:** reproduced the original crash, then confirmed the guard rejects the same pairing
  before any step; lifecycle/benchmark runs with matching vocab still train.
* **Test result:** PASS (manual repro + guard).

### AUD-031 — Any website could read and control the local API

* **Component:** `src/dashboard/server.py`.
* **Repro:** with the server running, open any site and run
  `fetch('http://127.0.0.1:8080/api/settings')` — the response (including the Discord webhook URL) is
  readable because `Access-Control-Allow-Origin: *`; `POST` actions (start/stop/delete) are also
  accepted.
* **Root cause:** blanket permissive CORS with no `Origin`/`Host` validation and no authentication.
* **Files modified:** `src/dashboard/server.py`, `tests/test_api_endpoints.py`.
* **Fix:** reject non-local `Host` values (unless `LLMMAKER_ALLOWED_HOSTS` opts in), reflect only
  localhost origins, never treat `Origin: null` as local, return 403 for cross-origin mutations, and
  reject cross-origin WebSocket handshakes.
* **Verification:** seven new API tests: local origin reflected; foreign origin POST → 403; foreign origin
  GET has no CORS header; foreign `Host` → 403; IPv6 loopback accepted; `Origin: null` gets no CORS
  permission and cannot mutate. Suite 18/18.
* **Test result:** PASS.

## 6. Medium and low findings

Every entry keeps the same fields; open items cross-reference `known_issues.md`.

### AUD-012 — `0` interval silently replaced by the default (Medium, fixed)

* **Repro:** set validation or checkpoint interval to 0 ("disabled"), train; validation still runs and
  checkpoints are still written.
* **Root cause:** `payload.get("eval_freq") or default` treats 0 as missing.
* **Files modified:** `src/worker/training_worker.py`.
* **Fix:** `_first_int()` helper preserving 0, used for `eval_freq`/`save_freq` (aliases supported).
* **Verification:** regression tests for 0, alias priority, default and string values.
* **Test result:** PASS.

### AUD-013 — Adaptive LR overrode the user's LR even when disabled (Medium, fixed)

* **Repro:** disable auto LR adjustment, set 1e-3, watch the effective LR drift.
* **Root cause:** the controller applied its factor unconditionally.
* **Files modified:** `src/worker/training_worker.py`.
* **Fix:** gate the controller on `auto_lr_adjust` and reset the factor to 1.0 when disabled.
* **Verification:** code review + the LR fields in a real run remain at the configured value.
* **Test result:** PASS (observational).

### AUD-014 — Checkpoint search reached into an external project (Medium, fixed)

* **Repro:** delete a checkpoint referenced by the UI; the loader could resolve a same-named file from
  `d:\proiecte\LLM from Scratch\checkpoints`.
* **Root cause:** legacy absolute directory plus a repository-wide `**/name` glob.
* **Files modified:** `src/checkpoints/manager.py`, `src/training/factory.py`.
* **Fix:** bounded search (`checkpoints_v2/`, `workspaces/*/checkpoints`) with a clear not-found error;
  factory listing no longer references the external project.
* **Verification:** regression test for missing entry; lifecycle checkpoint paths resolve inside the temp
  workspace.
* **Test result:** PASS.

### AUD-019 — Deleting a checkpoint could leave an orphan row (Medium, fixed)

* **Repro:** delete the checkpoint file manually, then delete it in the UI; the DB row persists.
* **Root cause:** the delete path returned early when the file was absent.
* **Files modified:** `src/checkpoints/manager.py`, `src/db/database.py`.
* **Fix:** `delete_checkpoint()` always removes the row and returns the path (or None), tolerating a
  missing file.
* **Verification:** regression test seeds a row without a file and asserts cleanup.
* **Test result:** PASS.

### AUD-020 — `create_run` uses `INSERT OR REPLACE` (Medium, fixed)

* **Repro:** create a run with an existing id; the row is replaced, steps/tokens reset, and metrics are
  orphaned.
* **Root cause:** id reuse treated as an upsert, and no connection enabled foreign keys.
* **Files modified:** `src/db/database.py`, `src-tauri/src/database.rs`.
* **Fix:** `create_run` and `create_workspace` use a plain `INSERT`; a duplicate raises `ValueError`
  instead of replacing history. Every connection enables `PRAGMA foreign_keys = ON`. Workspace deletion
  removes metrics, runs, checkpoints and datasets before the workspace row.
* **Verification:** `tests/unit/test_data_safety.py` asserts that a duplicate run keeps its metrics and
  that deletion removes all children.
* **Test result:** PASS.

### AUD-021 — Python default DB path depends on CWD (Medium, fixed)

* **Repro:** start `dashboard.py` from another directory; a fresh empty database was created.
* **Root cause:** `DEFAULT_DB_PATH = Path("data/rocanon_studio.db")`.
* **Files modified:** `src/db/database.py`.
* **Fix:** the default resolves from the module location to `<repo>/data/rocanon_studio.db`; the
  absolute path is logged at startup.
* **Verification:** the path is stable when the process CWD changes and both languages apply the same
  migrations.
* **Test result:** PASS.

### AUD-022 — Cargo config pins a machine-specific target (Medium, open)

* **Repro:** build on another machine; output goes to `C:/Users/Cata/...` or the build fails.
* **Root cause:** `.cargo/config.toml` `target-dir` and GNU triple, added to work around `windres` on a
  path containing spaces.
* **Files modified:** none.
* **Fix (recommended):** move to a machine-local config or a setup script.
* **Verification:** not implemented; see `known_issues.md` AUD-022.
* **Test result:** N/A (documented).

### AUD-023 — Token accounting counts masked positions (Medium, fixed)

* **Repro:** run SFT with masked prompt labels; token progress was higher than the tokens actually
  contributing to the loss.
* **Root cause:** the step token count used `batch × seq × accum`, while the loss uses only unmasked
  labels.
* **Files modified:** `src/worker/training_worker.py`.
* **Fix:** `accumulate_micro_batch` returns `valid_tokens`; the worker sums it for `total_tokens` and the
  scheduler. Throughput and TFLOP/s still use raw processed positions, the hardware-work number.
* **Verification:** existing training tests pass; the metric fields are documented separately.
* **Test result:** PASS.

### AUD-024 — Duplicate dependency pin (Low, fixed)

* **Repro:** `pip install -r requirements.txt` warns about a duplicate `nvidia-ml-py`.
* **Root cause:** two lines for the same package.
* **Files modified:** `requirements.txt`.
* **Fix:** removed the duplicate, kept `nvidia-ml-py>=13.615.71`.
* **Verification:** dependency list inspected.
* **Test result:** PASS.

### AUD-025 — Unreachable duplicate loss accounting (Low, fixed)

* **Repro:** code inspection — a second `loss_accumulator.add()` after the first could never execute.
* **Root cause:** leftover from a refactor.
* **Files modified:** `src/training/trainer.py`.
* **Fix:** removed the dead block.
* **Verification:** AST parse + full unit suite.
* **Test result:** PASS.

### AUD-026 — Unused import (Low, fixed)

* **Repro:** linter warning for `shutil` in `manager.py` after the atomic-write rewrite.
* **Root cause:** leftover import.
* **Files modified:** `src/checkpoints/manager.py`.
* **Fix:** removed it.
* **Verification:** AST parse + unit suite.
* **Test result:** PASS.

### AUD-027 — No automated frontend tests (Medium, open)

* **Repro:** break `validateRun`; `tsc` and `next build` still pass.
* **Root cause:** no test runner configured.
* **Files modified:** none.
* **Fix (recommended):** Vitest + Testing Library, Playwright for core flows.
* **Verification:** not implemented; see `known_issues.md` AUD-027.
* **Test result:** N/A (gap).

### AUD-028 — Duplicated terminal-exit logic in the runner (Low, fixed)

* **Repro:** code inspection; both terminal branches repeated the same sleep/exit sequence.
* **Root cause:** copy-paste.
* **Files modified:** `src/worker/standalone_runner.py`.
* **Fix:** one `exit_after_terminal_event()` helper used by both branches.
* **Verification:** lifecycle sessions 1–3 exit 0.
* **Test result:** PASS.

### AUD-029 — Resume inflates the reported token target (Low, fixed)

* **Repro:** resume a run that already passed its token target; `target_tokens` became
  `total_tokens + 1e13`.
* **Root cause:** a guard against immediate completion mutated the user-visible target.
* **Files modified:** `src/worker/training_worker.py`.
* **Fix:** the inflation was removed. Completion uses `session_target_tokens`, derived from the current
  session; `target_tokens` remains the user's target.
* **Verification:** resume tests pass and the reported target stays finite.
* **Test result:** PASS.

### AUD-030 — `vram_used_gb` is peak, not current (Low, fixed)

* **Repro:** compare the hardware panel with `nvidia-smi` after a spike; the app kept showing the peak.
* **Root cause:** the metric used `max_memory_allocated()`.
* **Files modified:** `src/worker/training_worker.py`.
* **Fix:** `vram_used_gb` is now `torch.cuda.memory_allocated()`; `vram_peak_gb` carries the peak. The
  VRAM guard evaluates peak, the conservative value.
* **Verification:** unit suite and progress payload inspection.
* **Test result:** PASS.

### AUD-032 / AUD-033 / AUD-034 — Security and portability risks (fixed)

* **AUD-032:** HF inspection and loading default to `trust_remote_code=False`; the API accepts
  `allow_remote_code` and Network Studio asks for explicit confirmation naming the repository. The choice
  is stored in the network config.
* **AUD-033:** notification secrets are AES-256-GCM encrypted by both the Python and Rust layers, with
  a shared key/ciphertext format; the key is DPAPI-protected (0600 file elsewhere), API responses return
  configured flags and a sentinel, and legacy plaintext rows are upgraded on open.
* **AUD-034:** release builds check `LLMMAKER_PYTHON`, then the project virtualenv, and never fall back
  to `PATH`; the `PATH` branch is debug-only.
* **Files modified:** `src/networks/foundation_loader.py`, `src/security/secrets.py`,
  `src/db/database.py`, `src/dashboard/server.py`, `src-tauri/src/secrets.rs`,
  `src-tauri/src/database.rs`, `src-tauri/src/worker.rs`, Network Studio and Settings.
* **Verification:** `tests/unit/test_data_safety.py` covers encryption, redaction, sentinel semantics,
  legacy upgrade, key creation and refusal to replace a missing key; a standalone Rust probe verified
  AES-GCM ciphertext interoperability in both directions and a Rust-created DPAPI key read by Python.
* **Test result:** PASS.

## 7. Root-cause analysis

The 34 findings are not 34 independent mistakes; they cluster into six causes. Fixing the cause is what
prevents the next instance.

### 7.1 Failure paths were never executed

AUD-001, 003, 004, 005 and 019 all live in error/stop/resume branches. The prior audit's checkpoint test
loaded a plain tensor — the safe branch — so the `weights_only=False` fallback stayed invisible. The
worker had no fault-injection test at all. This audit added tests that deliberately corrupt, stop and
kill (malicious pickle, truncated checkpoint, mid-run stop) and asserts the *failure* behaviour, not just
the happy path.

### 7.2 Two implementations of one schema (Rust and Python)

AUD-009 is the canonical example: a column added on one side broke every run list on the other. The
desktop and the control plane each own `CREATE TABLE`/migrations with no shared contract and no
migration test. Recommendation: a single schema SQL file (or versioned migrations executed by both) and
a CI test that opens the oldest supported database with the newest code.

### 7.3 Timing assumptions instead of handshakes

AUD-002 (800 ms before kill) and AUD-005 (500 ms before `os._exit`) both assume work finishes in a fixed
window. The fix in both cases is an explicit signal (`try_wait` with a budget; `session_done` event).
Anywhere the code sleeps and hopes should be treated as a latent data-loss bug.

### 7.4 Environment baked into the product

AUD-006, 008, 014, 021 and 022 encode the original developer's absolute paths (`D:\proiecte\...`) or the
process CWD. The application appeared to work only because it ran on that machine. The pattern to
enforce: every path is resolved from an explicit root (env override first), logged at startup, and never
searched repository-wide.

### 7.5 Metrics computed from constants and hacks

AUD-016 (TFLOPs from a hardcoded 59 M params), AUD-029 (`+1e13` to dodge a completion check) and AUD-030
(peak VRAM labelled current) all produce numbers that look measured but are not. The product's
credibility depends on these being real; each now either derives from the live model or is documented as
an approximation.

### 7.6 Security posture treated as "localhost means safe"

AUD-001 and AUD-031 share the assumption that local files/requests are trusted. Checkpoints are shared
artefacts, and a browser is not a trust boundary. The fixes apply the same rule in both places: validate
the input (restricted unpickling, origin/host checks) and fail closed.

## 8. Implemented fixes (by file)

| File | Change | Issues |
| --- | --- | --- |
| `src/checkpoints/manager.py` | `safe_torch_load` without fallback; atomic save no destructive retry; bounded path search; orphan cleanup; unused import | 001, 014, 019, 026 |
| `src/worker/training_worker.py` | `session_done`; final status write; stop snapshot from live weights; loud resume failure; vocab guard; param-based TFLOPs; interval parsing; LR gating | 003, 004, 005, 012, 013, 016, 018 |
| `src/worker/standalone_runner.py` | single exit helper waiting for `session_done` | 005, 028 |
| `src/db/database.py` | `final_loss` schema + migration; `delete_checkpoint` row cleanup; `update_run_status` extension | 009, 019 |
| `src/training/trainer.py` | removed dead duplicate loss accumulation | 025 |
| `src/training/factory.py` | bounded checkpoint listing, no external project | 014 |
| `src/data/custom_loader.py` | preset resolver (env, project, legacy), explicit missing-path error | 017 |
| `src/dashboard/server.py` | dataset resolver; CORS/Host/WS origin enforcement | 017, 031 |
| `requirements.txt` | duplicate `nvidia-ml-py` removed | 024 |
| `src-tauri/src/worker.rs` | project root, interpreter discovery, absolute DB, `--db-path`, tree kill, 20 s stop, run-id-scoped reader, stdin ownership | 002, 005, 006, 007, 008 |
| `src-tauri/src/lib.rs` | absolute DB, stale-run recovery, `validate_run` registration, shutdown hook, startup logging | 008, 009, 010, 015 |
| `src-tauri/src/database.rs` | `final_loss` migration, parameterized workspace filters, `recover_stale_runs` | 009, 011, 015 |
| `src-tauri/src/commands.rs` | `validate_run` command | 010 |
| `frontend/src/services/api.ts` | real `validateRun` through IPC, error on rejection | 010 |
| `tests/unit/test_audit_regressions.py` | 8 regression tests (new) | 001, 009, 012, 017, 019 |
| `tests/integration/audit_worker_lifecycle.py` | 3 real worker scenarios, 18 checks (new) | 002–005, 018 |
| `tests/integration/audit_worker_benchmark.py` | throughput benchmark + parameter count (new) | 016 |
| `tests/integration/audit_db_probe.py` | DB inspection helper (new) | 015 |
| `tests/test_api_endpoints.py` | 4 CORS/Host security tests | 031 |

Diff: 16 modified files (+714/−407 overall, including this audit's `test_results.md`), plus the new tests
and audit documents. No dependencies were added.

## 9. Test results

| Suite | Command | Result |
| --- | --- | --- |
| Unit | `pytest tests/unit -q` | **142 passed**, 0 failed |
| HTTP API | `pytest tests/test_api_endpoints.py -q` | **18 passed**, 0 failed |
| Other backend | `pytest tests/test_frontend_integration.py tests/test_network_studio.py tests/test_v2_features.py -q` | **58 passed**, 0 failed |
| Worker lifecycle | `python tests/integration/audit_worker_lifecycle.py` | **18/18 checks**, 3 real sessions |
| Throughput | `python tests/integration/audit_worker_benchmark.py 120` | 120 steps, 48.36 ms median, 10,546.6 tok/s |
| TypeScript | `npx tsc --noEmit --incremental false` | exit 0 |
| Frontend build | `npx next build` | exit 0, 16 routes |
| Rust check | `cargo check --offline` | exit 0 (24.66 s) |
| Rust build | `cargo build --offline` | exit 0 (17.66 s), binary produced |
| Desktop startup | smoke test on a DB copy | startup logs, 12 stale runs recovered, seeded run → `STOPPED` |
| HTTP smoke | `dashboard.py --port 8123` | 6 endpoints 200, dataset preset resolves |
| Syntax | `ast.parse` on 11 modified Python files | OK |

Full logs, exact commands and the deliberate coverage gaps are in `test_results.md`.

## 10. Performance measurements

| Metric | Value |
| --- | --- |
| Model | 1,876,736 trainable parameters (2 layers, d_model 128, vocab 12288) |
| Step time | median **48.36 ms**, min 45.64, max 96.50 |
| Throughput | median **10,546.6 tokens/s** (max 11,175.0) |
| Compute estimate | **0.119 TFLOPs** = 6·N·T, matches the reported value |
| VRAM peak | **0.11 GB** of 6.0 GB |
| 120-step wall time | 18.6 s (≈5.8 s compute; the rest is process/torch startup and teardown) |
| Final loss | 0.1245 (synthetic data; run-to-run 0.1157–0.1245) |

At this scale the loop is launch-bound, not compute-bound. The value of the numbers is that they prove
there is no per-step disk write, no hidden synchronization and no leak; they are not a GPU efficiency
benchmark. A 48.5 M-parameter run on a real dataset is the missing measurement (see limitations).

## 11. Remaining limitations

1. **No interactive GUI verification.** The Tauri webview was not clicked through; frontend confidence
   comes from type check, production build and tracing every call to its command/route.
2. **Debug build only.** No installer/release bundle or upgrade path was tested.
3. **Large-model performance unknown.** 1.9 M parameters, synthetic data.
4. **Open medium issues:** the Cargo target pin (AUD-022) and missing frontend component tests
   (AUD-027). The data-safety issues are fixed and covered in `tests/unit/test_data_safety.py`.
5. **No Rust unit tests** and only code-review coverage for the run-id routing fix.
6. **Legacy dataset fallback** (`D:\proiecte\LLM from Scratch\data`) is retained and logged; it is a
   compatibility path, not a requirement.
7. **Two databases by design** (desktop `studio.db` vs browser `data/rocanon_studio.db`); they were not
   merged.

## 12. Next steps

| Priority | Action |
| --- | --- |
| P1 | Add Vitest + Playwright for the three core flows (AUD-027) |
| P2 | Benchmark a 48.5 M-parameter model on a real corpus; publish tokens/s and VRAM |
| P3 | Move Cargo target redirection out of the repo (AUD-022) |
| P3 | Add Rust tests for the supervisor (spawn, stop timeout, stale events) |
| P3 | Run the new Pages workflow on GitHub and confirm the project-site base path |

## 13. Deliverables

| File | Contents |
| --- | --- |
| `FULL_AUDIT_REPORT.md` | This report |
| `architecture_overview.md` | Stack, process topology, modules, protocol |
| `api_endpoint_inventory.md` | 54 commands, 9 worker commands, 13 events, 79 HTTP routes, coverage gaps |
| `frontend_functionality_audit.md` | Route → view → backend map, per-feature findings |
| `training_engine_audit.md` | Loop, loss, optimizers, validation, checkpoint/resume, observed runs |
| `desktop_integration_audit.md` | Tauri startup, supervisor, DB layer, shutdown, capabilities |
| `performance_audit.md` | Measured throughput, memory, overheads, scaling notes |
| `security_audit.md` | Trust boundaries, findings, accepted risks, areas verified sound |
| `test_results.md` | Every command, output and coverage gap |
| `known_issues.md` | 7 open + 3 accepted issues with evidence and recommended fixes |

Pre-existing audit artefacts from earlier passes (`bug_inventory.md`, `architecture_map.md`,
`desktop_api_integration_report.md`, etc.) were left in place; this report supersedes them where they
overlap (notably BUG-01, whose incomplete fix is analysed in AUD-001).