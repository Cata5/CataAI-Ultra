# LLM Maker — Architecture Overview

Audit date: 2026-11-24 · Repository: `D:\proiecte\CataAI Ultra` · Branch: `experimental` (uncommitted audit changes)

## 1. What the product is

LLM Maker (internal name *RoCanon Studio*) is a Windows desktop application for designing,
training, evaluating and exporting small language models on consumer hardware. It ships three
execution surfaces that share one SQLite data model:

| Surface | Entry point | Transport to the ML engine |
| --- | --- | --- |
| Tauri desktop shell | `src-tauri/src/main.rs` → `lib.rs` | Rust spawns a Python worker child and exchanges JSON lines over stdin/stdout |
| Browser/studio server | `dashboard.py` → `src/dashboard/server.py` | aiohttp REST + WebSocket, in-process training manager |
| Worker CLI | `python -m src.worker.standalone_runner` | stdin commands, stdout events, single-run process |

The frontend is a single Next.js App Router application used by both the desktop webview and a
normal browser. `frontend/src/services/api.ts` is the only data-access layer: when it detects the
Tauri runtime it calls `invoke()`, otherwise it calls the HTTP API. This dual-backend design is the
single most important structural fact in the codebase — every feature must work through two
independent implementations of the same data layer.

## 2. Process and data topology

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Tauri 2 shell (Rust)                                                    │
│  lib.rs ── AppState { db: DatabaseManager, supervisor: WorkerSupervisor }│
│    │            │                                    │                  │
│    │            └── rusqlite ──► <project>/studio.db   │ spawn + JSONL   │
│    ├── 54 #[tauri::command] handlers ◄── invoke() ──┐  ▼                  │
└────┼────────────────────────────────────────────────┼──────────────────┘
     │                                                │
     │ Next.js webview (React Query, Zustand-ish stores)                  │
     │   frontend/src/app/*/page.tsx → features/*/…View.tsx               │
     │                                                │
     └── browser fallback: fetch/WebSocket ───────────┼──► aiohttp server  │
                                                      │    (dashboard.py)  │
                                                      ▼                    │
                          src/worker/standalone_runner.py (Python child)     │
                             │ torch, datasets, optimizers                  │
                             └── SQLite: same <project>/studio.db (or --db-path)
```

Key consequences observed during the audit:

* Two writers, one database. The Rust layer writes run/checkpoint rows for UI-driven actions while
  the Python worker writes metrics, alerts and status transitions. Any column mismatch breaks the
  desktop list screens (this happened with `runs.final_loss`, see AUD-009).
* The Python worker is the only component that touches tensors; Rust never loads a checkpoint.
* The database path is resolved three different ways (Rust project root, Python `--db-path`, Python
  module default `data/rocanon_studio.db`). Before this audit they could disagree, producing
  "runs disappear after restart" symptoms.

## 3. Repository layout

| Path | LOC | Responsibility |
| --- | ---: | --- |
| `src/model/` | ~2.2k | RoCanon transformer (RMSNorm, RoPE, grouped-query attention, SwiGLU), custom network factory |
| `src/training/` | ~3.4k | Trainer, Muon+AdamW optimizer, WSD/linear/cosine schedulers, task adapters, PEFT/LoRA, early stopping, VRAM guard |
| `src/worker/` | ~2.6k | Worker state machine, JSON-lines protocol, standalone runner |
| `src/checkpoints/` | ~1.2k | Save/load, atomic writes, package export, chart generation |
| `src/data/` | ~2.4k | Dataset engine, tokenizer integration, streaming mixer, curation |
| `src/db/` | ~1.3k | SQLite schema, migrations, query helpers |
| `src/dashboard/` | ~5.5k | aiohttp server, 79 HTTP routes, WebSocket hub, evaluation/inference integration |
| `src/networks/`, `src/evaluation/`, `src/export/`, `src/hardware/`, `src/notifications/` | ~2.5k | Supporting services |
| `src-tauri/src/` | 3,017 | Tauri shell: `lib.rs`, `commands.rs` (54 commands), `database.rs`, `worker.rs`, `hardware.rs` |
| `frontend/src/` | 13,004 | Next.js 14 pages, feature views, hooks, API/WS clients, i18n |
| `tests/` | 25 files | 142 unit tests + integration/e2e suites |

## 4. Frontend structure

* 14 routes under `frontend/src/app` (13 pages + the generated `_not-found`): `/`, `/studio`,
  `/training`, `/runs`, `/checkpoints`, `/datasets`, `/evaluation`, `/models`, `/hardware`, `/logs`,
  `/analytics`, `/projects`, `/settings`.
* 9 feature views (`features/*/[X]View.tsx`) hold the page logic; pages are thin shells.
* Data fetching: `@tanstack/react-query` with a shared `QueryClient`; live training data arrives on
  a WebSocket (`services/websocket.ts`), with a 1–2 s polling fallback when the socket is down.
* State: React context + hooks; no global store library.
* i18n: `features/i18n` with English/Romanian dictionaries; language persisted in localStorage.
* Styling: Tailwind CSS v4 with a dark studio theme; no component library.

## 5. Backend (Python) command/event protocol

`src/worker/protocol.py` defines the entire contract between the desktop shell and the ML process:

* Commands (stdin JSON): `START_RUN`, `PAUSE_RUN`, `RESUME_RUN`, `STOP_RUN`, `SAVE_CHECKPOINT`,
  `RUN_VALIDATION`, `RUN_BENCHMARK`, `LOAD_PIPELINE`, `SET_LR_MULTIPLIER`.
* Events (stdout JSON): `RUN_STARTED`, `STEP_COMPLETED`, `METRICS_UPDATED`, `LOSS_SPIKE`,
  `GRADIENT_WARNING`, `CHECKPOINT_SAVED`, `PHASE_COMPLETED`, `VALIDATION_COMPLETED`,
  `RUN_PAUSED`, `RUN_STOPPED`, `RUN_FAILED`, `RUN_COMPLETED`, `STATE_CHANGED`.
* Run states: `IDLE → STARTING → RUNNING ⇄ PAUSING/PAUSED → STOPPING → STOPPED | COMPLETED | FAILED`,
  enforced by `RunStateMachine` with explicit legal transitions.

## 6. Training pipeline (high level)

1. `START_RUN` payload resolves the architecture (`config_arch` or `network_id` → `NetworkRegistry`),
   the dataset (`dataset_id`/`dataset_source`) and the optimizer/scheduler configuration.
2. The dataset engine returns token tensors (real files, cached tokenized binaries, or the explicit
   `synthetic` source) with a 95/5 train/val split.
3. `TrainingWorkerProcess` builds the model, wraps it in `TaskAdapter` (causal LM, masked LM,
   classification, regression, instruction SFT), builds Muon+AdamW and the WSD scheduler, then enters
   the accumulate→optimizer-step loop.
4. Every step emits `STEP_COMPLETED` with ~60 fields (loss, LR, grad-norm, ETA, VRAM, throughput).
   Metrics are batched into SQLite every 5 steps or on validation.
5. Validation runs at `val_interval` steps, keeps the best checkpoint by val loss, and feeds early
   stopping.
6. Stop saves a `stopped_snapshot_*.pt` with live weights **and** optimizer/scheduler state so the run
   can resume; completion saves a final checkpoint and marks the run `COMPLETED`.

## 7. Persistence

`studio.db` (project root, 25 runs / 2,051 metrics / 12 checkpoints / 3 datasets / 2 workspaces at audit
time) contains: `runs`, `metrics`, `checkpoints`, `datasets`, `workspaces`, `settings`, `alerts`,
`pipelines`, `benchmarks`. The browser server keeps its own `data/rocanon_studio.db`. Checkpoints live
in `checkpoints/` (default workspace) and `workspaces/<id>/checkpoints/`.

## 8. Where the architecture hurts

| Tension | Symptom found in this audit |
| --- | --- |
| Duplicated data layer (Rust SQL + Python SQL) | Missing `final_loss` column broke every run list (AUD-009) |
| Reliance on process CWD | DB, checkpoints and dataset paths resolved differently per launcher (AUD-008) |
| Fire-and-forget child process | Terminal events raced process exit, truncating final writes (AUD-005) |
| UI-only validation | A tokenizer/model vocabulary mismatch reached the CUDA kernel and surfaced as `device-side assert` (AUD-018) |