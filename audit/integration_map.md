# Frontend-Backend Integration Map

## Architecture Overview
- **Frontend**: Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS, `@xyflow/react`.
- **Backend API**: aiohttp HTTP REST on `http://127.0.0.1:8081`.
- **Telemetry Stream**: WebSocket on `ws://127.0.0.1:8081/ws`.
- **Inference & Execution**: PyTorch CUDA 12.8 on NVIDIA RTX 3060 Laptop (6GB VRAM) with single-owner GPU arbitration.

---

### Core Integration Routes

| Feature / Action | Frontend Component | API Client Method | HTTP / WS Route | Backend Service / Handler | Database / Worker / Model | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **System Status Poll** | `AppShell`, `TopBar` | `api.getSystemStatus` | `GET /api/system/status` | `handle_system_status` | `WorkerManager.current_state`, DB alerts | PASS |
| **Hardware Telemetry** | `StatusBar`, `OverviewView` | `api.getHardwareTelemetry` | `GET /api/hardware` | `handle_hardware_telemetry` | `HardwareMonitor` (NVML / PyTorch CUDA) | PASS |
| **WebSocket Events** | `useWebSocket` hook | `wsService.connect` | `WS /ws` | `handle_websocket` | `WorkerManager._broadcast` (Events/Metrics) | PASS |
| **Start Training** | `TrainingView` | `api.startRun` | `POST /api/runs` | `handle_create_run` | `WorkerManager.start_run` -> `TrainingWorker` | PASS |
| **Pause Run** | `OverviewView`, `TrainingView` | `api.pauseRun` | `POST /api/runs/current/pause` | `handle_run_pause` | `WorkerManager.pause_run` | PASS |
| **Resume Run** | `OverviewView`, `TrainingView` | `api.resumeRun` | `POST /api/runs/current/resume` | `handle_run_resume` | `WorkerManager.resume_run` | PASS |
| **Stop Run** | `OverviewView`, `TrainingView` | `api.stopRun` | `POST /api/runs/current/stop` | `handle_run_stop` | `WorkerManager.stop_run` | PASS |
| **Save Checkpoint** | `OverviewView`, `TrainingView` | `api.manualCheckpoint` | `POST /api/runs/current/checkpoint` | `handle_run_checkpoint` | `WorkerManager.request_checkpoint` | PASS |
| **Dynamic LR Tuning** | `OverviewView` | `api.setLrMultiplier` | `POST /api/runs/lr` | `handle_update_lr` | `WorkerManager.set_lr_multiplier` | PASS |
| **List Checkpoints** | `RunsView`, `EvaluationView` | `api.listCheckpoints` | `GET /api/checkpoints` | `handle_get_checkpoints` | `CheckpointManager.list_checkpoints_on_disk` | PASS |
| **Delete Checkpoint** | `RunsView` | `api.deleteCheckpoint` | `DELETE /api/checkpoints/{id}` | `handle_delete_checkpoint` | `CheckpointManager.delete_checkpoint` | PASS |
| **List Datasets** | `DatasetsView`, `TrainingView` | `api.listDatasets` | `GET /api/datasets` | `handle_list_datasets` | `DatasetManager.list_all_datasets` | PASS |
| **Upload Dataset** | `DatasetsView` | `api.uploadDataset` | `POST /api/datasets/upload` | `handle_upload_dataset` | Direct 8081 multipart streaming (up to 500MB) | PASS |
| **Tokenize Dataset** | `DatasetsView`, `TrainingView` | `api.tokenizeDataset` | `POST /api/datasets/{id}/tokenize` | `handle_tokenize_dataset` | `CustomDatasetEngine.load_dataset` (.pt cache) | PASS |
| **Inspect Tokenizer** | `DatasetsView` | `api.inspectTokenizer` | `POST /api/tokenizer/inspect` | `handle_tokenizer_inspect` | `DatasetManager.inspect_tokenizer_sample` | PASS |
| **List Networks** | `NetworkStudioView` | `api.listNetworks` | `GET /api/networks` | `handle_list_networks` | `NetworkRegistry.list_networks` | PASS |
| **List Templates** | `NetworkStudioView` | `api.getNetworkTemplates` | `GET /api/networks/templates` | `handle_get_network_templates` | `ALL_TEMPLATES` (RoCanon, Transformer, MLP, CNN, etc.) | PASS |
| **Validate Architecture** | `NetworkStudioView` | `api.validateNetwork` | `POST /api/networks/validate` | `handle_validate_network` | `ArchitectureValidator.run_preflight_check` | PASS |
| **Estimate Memory/Params**| `NetworkStudioView` | `api.estimateNetwork` | `POST /api/networks/estimate` | `handle_estimate_network` | `ParameterCalculator` & `MemoryEstimator` | PASS |
| **Save Network** | `NetworkStudioView` | `api.saveNetwork` | `POST /api/networks` | `handle_save_network` | `NetworkRegistry.save_network` | PASS |
| **Evaluation Generate** | `EvaluationView` | `api.generateText` | `POST /api/evaluation/generate` | `handle_eval_generate` | `InferenceEngine.generate` (with GPU arbitration) | PASS |
| **Evaluation Compare** | `EvaluationView` | `api.compareCheckpoints` | `POST /api/evaluation/compare` | `handle_eval_compare` | `InferenceEngine.compare_checkpoints_sequential` | PASS |
| **Run Benchmark Suite** | `EvaluationView` | `api.runBenchmark` | `POST /api/evaluation/benchmark` | `handle_eval_benchmark` | `BenchmarkManager.run_benchmark_suite` | PASS |
| **Get / Save Settings** | `SettingsView` | `api.getSettings` / `api.saveSettings` | `GET|POST /api/settings` | `handle_get_settings` / `handle_save_settings` | `StudioDatabase.get_settings` / `save_settings` | PASS |
