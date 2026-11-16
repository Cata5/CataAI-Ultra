# RoCanon Studio v2 — Architecture Map

Repository root: `D:\proiecte\CataAI Ultra` (Git branch `main`, single commit).
Runtime: Python 3.14.6, `torch 2.11.0+cu128`, aiohttp 3.14.5, `tokenizers`, `pynvml`, SQLite (WAL).
No `requirements.txt`, `pyproject.toml`, `setup.py` or README exists — dependencies are only discoverable from
the checked-in `.venv` and the import graph.

## 1. Entry points

| What | Path | Notes |
|---|---|---|
| Application entry point | `dashboard.py` | Parses `--port/--host/--train/--device`, re-execs into `.venv`, builds the aiohttp app, runs `web.run_app`. |
| Windows launcher | `run_studio.bat` | `.\\.venv\\Scripts\\python.exe dashboard.py %*` |
| Frontend | `src/dashboard/ui.py` (`INDEX_HTML`, 3,222 lines) | Single-page HTML+JS, Tailwind CDN, Chart.js. Served by `GET /`. No build step, no framework, no router. |
| Frontend "routing" | `switchTab()` in `ui.py` | Show/hide of 7 `<div id="tab-*">` panels. Two identical definitions (lines ~37 and ~1464). |
| Legacy CLI trainer | `train.py` + `src/dashboard/state.py` | Standalone training script; shares `src/model`, `src/training`. Not used by the WebUI. |
| Unused export helpers | `src/infer/export.py`, `src/infer/generate.py` | `export_to_safetensors` is imported in `server.py` but never called; no route exposes it. |

## 2. Backend / control plane

`src/dashboard/server.py` (723 lines) is the entire backend: module-level singletons plus ~40 handlers.

Singletons created at import: `DB`, `WORKER_MGR`, `CHECKPOINT_MGR`, `DATASET_MGR`, `PIPELINE_MGR`,
`BENCHMARK_MGR`, `HW_MONITOR`, `NETWORK_REGISTRY`, `CUSTOM_MODULE_REG`, lazy `_INFERENCE_ENGINE`.

| Responsibility | Location | Used by app? |
|---|---|---|
| HTTP routes | `create_app()` lines ~640-712 | Yes |
| WebSocket `/ws` | `handle_websocket` | Yes |
| Worker→WS fan-out | `broadcast_to_websockets` + `WORKER_MGR.subscribe` | Yes |
| Telemetry | `handle_hardware_telemetry` → `HardwareMonitor` | Yes |
| Runs/controls | `handle_create_run`, `handle_action`, `handle_update_lr` | Yes |
| Checkpoints | `handle_get_checkpoints`, `handle_delete_checkpoint` | Yes |
| Datasets | `handle_upload_dataset`, `handle_list_datasets`, `handle_get_dataset_stats`, `handle_tokenizer_inspect` | Yes |
| Evaluation | `handle_eval_generate/compare/benchmark/results` | Yes (UI/generate/compare/benchmark) |
| Network Studio | `handle_*_network*` | Yes |
| Legacy | `/api/state`, `/api/config`, `/api/calculate_params`, `/api/action`, `/api/dataset_info` | Partially (`/api/state`, `/api/config` never called by UI) |

No authentication, no CORS configuration, no CSRF protection. Default bind is `127.0.0.1`.

## 3. Training worker architecture

| Module | Path | Responsibility | In → Out |
|---|---|---|---|
| Protocol | `src/worker/protocol.py` | `RunState`, `CommandType`, `EventType`, `WorkerCommand`, `WorkerEvent` (dataclasses ↔ dict). | — |
| State machine | `src/worker/state_machine.py` | `RunStateMachine` with an explicit transition table; raises `InvalidStateTransitionError`. | protocol |
| Control plane | `src/worker/worker_manager.py` | `WorkerManager`: spawns a `multiprocessing` (`spawn`) process, owns command/event queues, listener thread, subscribers, convenience `start_run/pause_run/...`. | db, protocol, state_machine → server |
| Worker | `src/worker/training_worker.py` (884 lines) | `TrainingWorkerProcess.run_loop` + `_execute_training_session`: model build, optimizer/scheduler, dataset load, step loop, validation, early stopping, checkpointing, VRAM guard, events. | all of `src/*` → queues + SQLite |

Process model: one long-lived spawned worker process. Commands arrive on a `mp.Queue`; events return on a second
queue and are drained by a thread in the parent, which updates `current_state` and broadcasts to WebSockets.
`send_command()` starts the process on demand. The worker opens its **own** `StudioDatabase` on the same file
(WAL), so the parent and child both write.

Note: `dashboard.py --train` calls `WORKER_MGR.start_run({phase, dataset_source})` **before** `create_app()`,
i.e. an unprompted training run starts with default dataset/step settings at launch.

## 4. Model, networks and registry

| Module | Path | Responsibility | Used? |
|---|---|---|---|
| RoCanon model | `src/model/architecture.py` | `RoCanonConfig`, `RoCanonForCausalLM`, GQA+RoPE+QK-Norm, SwiGLU, 4 Canon depthwise convs, tied embeddings. | Yes |
| Canon conv | `src/model/canon.py` | `CausalDepthwiseConv1d`. | Yes |
| Component library | `src/networks/components.py` | Activations, RMSNorm/LayerNorm/BatchNorm/GroupNorm, sinusoidal/learned/RoPE, `GroupedQueryAttention`, `StandardMLP`, `SwiGLUFFN`, `GeGLUFFN`, `CausalDepthwiseConv1d`, `GenericTransformerBlock`. | Yes (by factory) |
| Generic factory | `src/networks/factory.py` | `GenericModularNetwork` (decoder_only / encoder_only / encoder_decoder / hybrid / mlp / cnn), `CnnBlock`, `ModelFactory.create`. | Yes |
| Training-side factory | `src/training/factory.py` | `create_model_from_dict`, `create_optimizers_from_dict`, `calculate_analytical_parameters`, `list_checkpoints`. | Yes |
| Templates | `src/networks/templates.py` | 6 seed specs (`rocanon_48m`, `mini_gpt_15m`, `bert_encoder_25m`, `hybrid_cnn_transformer_20m`, `deep_mlp_5m`, `conv1d_network_4m`). | Yes (seeding) |
| Registry | `src/networks/registry.py` | JSON-per-network CRUD in `networks/registry/`, atomic temp+replace, clone, delete (RoCanon protected). | Yes |
| Estimator | `src/networks/estimator.py` | `ParameterCalculator`, `MemoryEstimator`, `BudgetAssistant`. | Yes |
| Validator | `src/networks/validator.py` | `ArchitectureValidator`: schema checks, CPU instantiate, forward, backward, gradient coverage. | Yes |
| Custom modules | `src/networks/custom_loader.py` | `CustomModuleBase`, `CustomModuleRegistry` scanning `networks/custom/*.py` via `importlib`. | Discovery only — **not** instantiated by any model |

## 5. Data, tokenizer, persistence

| Module | Path | Responsibility | Used? |
|---|---|---|---|
| DB | `src/db/database.py` | `StudioDatabase`: SQLite WAL, tables `runs, metrics, checkpoints, alerts, pipelines, datasets, benchmarks`. Connection per call, `threading.Lock`. `get_latest_metric` is defined twice. | Yes |
| Tokenizer | `src/tokenizer/tokenizer.py` | `RoCanonTokenizer` (HF `tokenizers`), 12,288 vocab, NFC + Romanian cedilla unification, byte-level BPE. | Yes |
| Dataset registry | `src/datasets/manager.py` / `validator.py` / `analyzer.py` | Upload → SHA-256 revision → immutable copy → validate (.txt/.jsonl/.json) → tokenize/analyse → DB row. | Yes |
| Training data engine | `src/data/custom_loader.py` | `CustomDatasetEngine.load_dataset`: resolves `KNOWN_DATASETS` key or a filesystem path, caches tensors in `data/cache/`, 95/5 split, optional SFT label masking. | Yes — this is the dataset→training bridge |
| Unused data modules | `src/data/curation.py`, `dataset.py`, `mixer.py` | Curation/MinHash, packed memmap dataset, mixture sampler. | Only imported by `src/data/__init__.py` and tests — dead in the app |

`CustomDatasetEngine.KNOWN_DATASETS` hardcodes three **absolute paths outside the repository**
(`d:\proiecte\LLM from Scratch\data\...`). They exist on this machine, so preset training works, but the project is
not relocatable and there is no path from an uploaded dataset id to this resolver.

## 6. Training internals

| Module | Path | Responsibility |
|---|---|---|
| Trainer | `src/training/trainer.py` | FP32 master weights + BF16 autocast, `accumulate_micro_batch`, grad-norm clip, dual-optimizer step, scheduler advance, memory budget assertion. |
| Loss | `src/training/loss.py` | `ChunkedLinearCrossEntropyFunction`, `ChunkedCrossEntropyLoss`, `TokenLossAccumulator` (token-normalized). Optional Liger. |
| Optimizers | `src/training/optimizer.py` | Muon (2-D) + AdamW (1-D/embeddings/conv) split. |
| Scheduler | `src/training/scheduler.py` | `WSDScheduler`, `clip_gradient_norm`. |
| Early stopping | `src/dashboard/early_stopping.py` | `OverfittingDetector`, best-weight restore. |
| Memory | `src/utils/memory.py` | `MemoryTracker`, peak/allocated helpers. |

The worker's step loop feeds flat `int64` token tensors and always computes a shifted causal-LM loss. There is no
task adapter, no loss selection and no model-output-shape negotiation.

## 7. Telemetry, evaluation, pipelines, inference

| Module | Path | Responsibility | Used? |
|---|---|---|---|
| Hardware | `src/telemetry/hardware.py` | NVML device identity/memory/util/temp/power/clocks + PyTorch allocator stats; explicit `None`/`error` when unavailable. | Yes |
| VRAM guard | `src/telemetry/vram_guard.py` | Threshold evaluation (600/350/150 MB free, 5 GB PyTorch ceiling). | Yes |
| Inference | `src/eval/inference.py` | `InferenceEngine.generate` (real sampling loop), `compare_checkpoints_sequential`. Always a default `RoCanonConfig()` model. | Yes |
| Benchmarks | `src/eval/benchmarks.py` | 12 built-in Romanian/persona/hardware/code/math items, keyword + repetition + compliance scoring, persists per-category rows. | Yes |
| Pipelines | `src/pipeline/manager.py` | `PhaseConfig`, `PipelineManager.create_pipeline/list/get/verify_phase_readiness/advance_phase`. | Persistence/validation only |
| Schedule math | `src/pipeline/scheduler_math.py` | `calculate_schedule_curve`, `calculate_step_lr`. | Yes (visualizer) |
| State manager | `src/dashboard/state.py` | `TrainingStateManager` singleton `STATE`: config dicts, metric history (capped 2000), broadcast helpers. | Partially — only legacy handlers |

## 8. Integration points (the ones that matter)

```
UI (ui.py)
  ├─ fetch /api/runs ─────────► handle_create_run ─► WORKER_MGR.start_run ─► mp.Queue ─► worker
  ├─ fetch /api/action ───────► handle_action ─────► WORKER_MGR.{pause,resume,stop,...}
  ├─ fetch /api/networks/* ───► NETWORK_REGISTRY / ModelFactory / estimators
  ├─ fetch /api/datasets/* ───► DATASET_MGR ─► SQLite + data/uploads/
  ├─ fetch /api/evaluation/* ─► InferenceEngine / BenchmarkManager
  └─ WebSocket /ws ◄────────── worker events ◄── WORKER_MGR listener thread
```

Registry → training: `network_id` in the start payload → worker reads `networks/registry/<id>.json` →
`create_model_from_dict`. Checkpoint → training: `checkpoint_to_load` → `CheckpointManager.load_checkpoint`.
Dataset → training: `dataset_source` string → `CustomDatasetEngine.load_dataset` (key or path; **no DB lookup**).

## 9. Dead / duplicate / unreachable inventory

- `src/infer/export.py` (`export_to_safetensors`), `src/infer/generate.py` (`RoCanonGenerator`) — never called; no route.
- `src/data/curation.py`, `src/data/dataset.py`, `src/data/mixer.py` — imported only by `src/data/__init__.py`/tests.
- `StudioDatabase.get_latest_metric` defined twice (lines ~283 and ~400).
- `switchTab` defined twice in `ui.py`; duplicate `TrainingStateManager` config surface in `state.py`.
- `/api/config` (GET/POST), `/api/state`, `/api/runs/{id}/start|pause|resume|checkpoint|validate|lr`,
  `/api/runs/{id}/metrics`, `/api/alerts`, `/api/pipelines` (GET/POST), `/api/networks/custom_modules`,
  `/api/evaluation/results` — registered but never called by the UI (most exercised only by tests).
- `CommandType.LOAD_PIPELINE` / `RUN_BENCHMARK` — declared, never handled by `training_worker.py`.
- `PipelineManager.verify_phase_readiness` / `advance_phase` — implemented, never called by server or worker.
- `STATE` (`TrainingStateManager`) metric/config machinery is largely bypassed by the worker; only
  `handle_get_state`/`handle_get_config` read it.
- `networks/registry/*.json` carries stale `total_parameters` values that `list_networks()` trusts verbatim.

## 10. External dependencies actually observed

`torch`, `aiohttp`, `tokenizers`, `pynvml` (deprecated package name, warning emitted; `nvidia-ml-py` recommended),
`numpy` (tests/e2e), `safetensors` (optional load path), `liger_kernel` (optional, absent). Frontend pulls
Tailwind and Chart.js from CDNs — i.e. the "local-first" UI still needs internet for styling/charts on first load.
