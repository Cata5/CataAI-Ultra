# API & IPC Endpoint Inventory

Counts verified by static extraction on 2026-11-24: **79 HTTP routes + 1 WebSocket endpoint**, **54 Tauri
commands**, **9 worker commands**, **13 worker events**.

## 1. Desktop IPC layer (Tauri commands)

`src-tauri/src/commands.rs` registers 54 commands in `lib.rs::generate_handler!`. The frontend reaches
them through `api.ts` (`invoke()`), and each has an HTTP twin except where noted.

| Domain | Commands |
| --- | --- |
| Hardware & system | `get_hardware_telemetry`, `get_compute_topology`, `get_system_logs`, `clear_system_logs`, `test_notification` |
| Workspaces | `list_workspaces`, `get_active_workspace`, `set_active_workspace`, `create_workspace`, `delete_workspace` |
| Runs | `list_runs`, `get_run_metrics`, `get_worker_status`, `start_training`, `stop_training`, `pause_run`, `resume_run`, `checkpoint_run`, `validate_run`, `update_lr` |
| Datasets | `list_datasets`, `upload_dataset`, `tokenize_dataset`, `curate_dataset`, `delete_dataset`, `get_dataset_stats`, `inspect_tokenizer` |
| Checkpoints & models | `list_checkpoints`, `delete_checkpoint`, `save_local_model`, `import_foundation_model`, `export_ollama`, `get_foundation_presets` |
| Networks | `list_networks`, `get_network`, `get_network_templates`, `clone_network`, `delete_network`, `save_network`, `estimate_network`, `validate_network`, `recommend_budget`, `get_custom_modules` |
| Evaluation & benchmarks | `generate_evaluation`, `compare_evaluation`, `run_benchmark_suite`, `list_benchmarks` |
| Pipelines | `list_pipelines`, `get_pipeline`, `create_pipeline` |
| Settings | `get_settings`, `save_settings`, `get_scheduler_curve` |
| Alerts & notifications | `get_alerts`, `test_notification` |

Behavioural notes relevant to the audit:

* `start_training`, `stop_training`, `pause_run`, `resume_run`, `checkpoint_run`, `validate_run` and
  `update_lr` proxy to `WorkerSupervisor` (JSON lines on the child's stdin). All other commands use
  `DatabaseManager` directly.
* `validate_run` was added during this audit; before it, the desktop had no way to trigger a manual
  validation and the frontend silently faked success.
* `stop_training` blocks up to 20 s waiting for the worker to flush, then kills the process tree.

## 2. Worker IPC protocol

| Direction | Messages |
| --- | --- |
| Rust → Python (stdin) | `START_RUN`, `PAUSE_RUN`, `RESUME_RUN`, `STOP_RUN`, `SAVE_CHECKPOINT`, `RUN_VALIDATION`, `RUN_BENCHMARK`, `LOAD_PIPELINE`, `SET_LR_MULTIPLIER` |
| Python → Rust (stdout) | `RUN_STARTED`, `STEP_COMPLETED`, `METRICS_UPDATED`, `LOSS_SPIKE`, `GRADIENT_WARNING`, `CHECKPOINT_SAVED`, `PHASE_COMPLETED`, `VALIDATION_COMPLETED`, `RUN_PAUSED`, `RUN_STOPPED`, `RUN_FAILED`, `RUN_COMPLETED`, `STATE_CHANGED` |

`src-tauri/src/worker.rs` reads stdout line by line, re-emits every event to the webview as
`training://event`, and mirrors status transitions into the `runs` table. Audit change: the reader now
tags itself with the run id it started (AUD-007), so a late event from a stopped run cannot overwrite
the status of the next one.

## 3. HTTP API (aiohttp, `src/dashboard/server.py`)

### System, hardware and workspaces

| Method | Path | Handler |
| --- | --- | --- |
| GET | `/` | `handle_index` |
| GET | `/api/system/status` | `handle_system_status` |
| GET | `/api/hardware` | `handle_hardware_telemetry` |
| POST | `/api/hardware/profile` | `handle_set_hardware_profile` |
| GET | `/api/tasks` | `handle_list_tasks` |
| GET | `/api/plugins` | `handle_list_plugins` |
| POST | `/api/plugins/{id}/toggle` | `handle_toggle_plugin` |
| GET | `/api/alerts` | `handle_get_alerts` |
| GET | `/api/logs` | `handle_get_logs` |
| POST | `/api/logs/clear` | `handle_clear_logs` |
| POST | `/api/notifications/test` | `handle_test_notification` |
| GET | `/api/state` | `handle_get_state` |
| GET | `/api/config` · POST | `handle_get_config` / `handle_post_config` |
| GET | `/api/dataset_info` | `handle_get_dataset_info` |
| GET | `/api/calculate_params` | `handle_calculate_params` |
| POST | `/api/action` | `handle_action` (generic command dispatch) |
| GET | `/api/workspaces` · POST | `handle_list_workspaces` / `handle_create_workspace` |
| GET | `/api/workspaces/current` · POST | `handle_get_active_workspace` / `handle_set_active_workspace` |
| PUT/DELETE | `/api/workspaces/{id}` | `handle_update_workspace` / `handle_delete_workspace` |
| POST | `/api/workspaces/{id}/export` | `handle_export_workspace` |
| POST | `/api/workspaces/import` | `handle_import_workspace` |

### Runs and training control

| Method | Path | Handler |
| --- | --- | --- |
| GET | `/api/runs` | `handle_list_runs` |
| POST | `/api/runs` and `/api/runs/{id}/start` | `handle_create_run` |
| POST | `/api/runs/{id}/pause` · `/resume` · `/stop` | `handle_run_pause` / `handle_run_resume` / `handle_run_stop` |
| POST | `/api/runs/{id}/checkpoint` | `handle_run_checkpoint` |
| POST | `/api/runs/{id}/validate` | `handle_run_validate` |
| POST | `/api/runs/lr` and `/api/runs/{id}/lr` | `handle_update_lr` |
| GET | `/api/runs/{id}/metrics` | `handle_run_metrics` |
| GET/DELETE | `/api/checkpoints` · `/api/checkpoints/{id}` | `handle_get_checkpoints` / `handle_delete_checkpoint` |
| POST | `/api/models/save-local` | `handle_save_local_model` |
| GET/POST | `/api/pipelines`, `/api/pipelines/{id}`, `/start`, `/advance` | pipeline handlers |
| GET/POST | `/api/settings` | `handle_get_settings` / `handle_save_settings` |
| POST | `/api/scheduler/curve` | `handle_scheduler_curve` |

### Datasets and tokenizer

| Method | Path | Handler |
| --- | --- | --- |
| POST | `/api/datasets/upload` | `handle_upload_dataset` |
| GET | `/api/datasets` | `handle_list_datasets` |
| GET | `/api/datasets/{id}/stats` | `handle_get_dataset_stats` |
| POST | `/api/datasets/{id}/tokenize` and `/api/datasets/tokenize` | `handle_tokenize_dataset` |
| POST | `/api/datasets/{id}/curate` and `/api/datasets/curate` | `handle_curate_dataset` |
| DELETE | `/api/datasets/{id}` | `handle_delete_dataset` |
| POST | `/api/tokenizer/inspect` | `handle_tokenizer_inspect` |

### Evaluation

| Method | Path | Handler |
| --- | --- | --- |
| POST | `/api/evaluation/generate` | `handle_eval_generate` |
| POST | `/api/evaluation/stream` | `handle_eval_stream` (chunked SSE-style response) |
| POST | `/api/evaluation/compare` | `handle_eval_compare` |
| POST | `/api/evaluation/benchmark` | `handle_eval_benchmark` |
| GET | `/api/evaluation/results` | `handle_eval_results` |
| GET/POST | `/api/evaluation/suites` | `handle_list_eval_suites` / `handle_create_eval_suite` |

### Networks and model import/export

| Method | Path | Handler |
| --- | --- | --- |
| GET | `/api/networks` | `handle_list_networks` |
| GET | `/api/networks/templates` | `handle_get_network_templates` |
| GET | `/api/networks/custom_modules` | `handle_custom_modules` |
| POST | `/api/networks/validate` · `/estimate` · `/budget_recommend` | `handle_validate_network` / `handle_estimate_network` / `handle_budget_recommend` |
| GET/POST | `/api/networks/{id}` · `/api/networks` | `handle_get_network` / `handle_save_network` |
| POST | `/api/networks/{id}/clone` | `handle_clone_network` |
| DELETE | `/api/networks/{id}` | `handle_delete_network` |
| GET | `/api/models/foundation_presets` | `handle_get_foundation_presets` |
| POST | `/api/models/import_hf` | `handle_import_hf_model` |
| POST | `/api/models/{id}/export_ollama` | `handle_export_ollama` |
| GET | `/api/compute/topology` | `handle_get_compute_topology` |

### OpenAI-compatible inference

| Method | Path | Handler |
| --- | --- | --- |
| GET | `/v1/models` | `handle_v1_models` |
| POST | `/v1/chat/completions` | `handle_v1_chat_completions` |

### WebSocket

`GET /ws` (`handle_websocket`) pushes `{type: "event"|"telemetry"|"status", payload: …}` frames. The
client (`frontend/src/services/websocket.ts`) reconnects with exponential backoff (1 s → ×1.5, capped) and
falls back to polling `/api/runs/{id}/metrics`.

## 4. Desktop coverage gaps

| HTTP capability | Desktop command | Status |
| --- | --- | --- |
| `/api/evaluation/stream` | — | HTTP only (desktop uses non-streaming `generate_evaluation`) |
| `/api/evaluation/suites` | — | HTTP only |
| `/api/pipelines/*/start`, `/advance` | — | HTTP only; desktop can list/create pipelines |
| `/v1/*` | — | HTTP only by design (local inference endpoint) |
| `/api/workspaces/{id}/export`, `/import` | — | HTTP only |
| `/api/config`, `/api/action` | — | HTTP only (legacy endpoints) |

These gaps are architectural, not regressions: the desktop frontend never calls those paths
(`api.ts` has no corresponding method), except workspace export/import which have no UI entry point.

## Verification pass (2026-11-27): static contract check

* **HTTP:** 96 routes registered in `src/dashboard/*.py` (`add_get/post/put/delete`). 67 `/api/...` references in `frontend/src`. Every frontend call matches a registered route and method, after normalising path parameters and query strings. No frontend call is missing a route.
* **Backend routes with no frontend caller:** 33, mostly intentional: the OpenAI-compatible `/v1/*` endpoints, `/ws`, `/`, and internal or CLI endpoints such as `/api/action`, `/api/config` (POST), `/api/state`, `/api/tasks`, `/api/plugins/*`, `/api/workspaces/import|export`, and `/api/evaluation/stream|suites|attention`. Each is a candidate for removal or a UI hook; none is a broken contract.
* **Tauri IPC:** every `invoke("name")` in `api.ts` (about 60 distinct commands) is registered in `src-tauri/src/lib.rs` `generate_handler!` (64 entries). No missing command.
* **Method:** the check was a script that scanned string literals and the nearest `method:` option. Three apparent mismatches were GET calls the heuristic misread, and were confirmed by hand.
