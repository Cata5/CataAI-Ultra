# LLM Maker — Training UX & Platform Upgrade Report

**Author:** Senior Machine Learning Infrastructure Architect & Full-Stack Systems Engineer  
**Date:** November 16, 2026  
**Platform:** LLM Maker v2 (Local Neural Research Workstation)  
**Target Hardware:** NVIDIA GeForce RTX 3060 Laptop GPU (6,144 MiB VRAM), Windows 11  
**Target Baseline:** RoCanon-48.5M (48,531,968 parameters, Hybrid CNN-Transformer)  

---

## 1. Executive Summary

This report documents the architectural remediation and UX modernization of the local neural training workstation, transforming it from a fragile single-experiment runner into **LLM Maker** — a resilient, mathematically transparent, and production-grade neural experimentation console.

Key areas resolved:
1. **Application Rebranding**: Complete transition to **LLM Maker**, retaining RoCanon-48.5M as an individually registered, protected baseline model.
2. **Validation Loss Persistence**: Eradication of metric loss upon tab navigation and browser refreshes (`F5`), with distinct tracking of `latest_val_loss`, `latest_val_step`, and `val_status` ('completed' | 'awaiting').
3. **Validation Frequency Bug Resolution**: Discovered and eliminated the root-cause bug that caused 250-step intervals to evaluate every 10 steps and on step 1.
4. **Dataset Progress & Real Token Accounting**: Clear dual-metric display showing current epoch progress (`Epoch 2 / 5 | 6.4M / 10.0M tokens (64%)`) alongside cumulative global tokens seen.
5. **Duration Mode Support**: Fully integrated `"steps"` vs `"epochs"` training objectives, validated through backend worker process stopping conditions.
6. **Time & Latency Separation**: Distinct display of iteration latency (`step_time_ms`) and cumulative active training duration (excluding pause periods), formatted without 24-hour wrap.
7. **Learning Rate Clarity & Stability Safeguards**: Disentangled Muon Effective LR (matrix weights) from AdamW Effective LR (embeddings & layer norms), exposed schedule phases and scaling multipliers, and introduced interactive visual warnings for excessive learning rates.
8. **100% GPU VRAM Release on Stop**: Solved the lingering CUDA context problem (recovered 3.7+ GB VRAM) by implementing safe tensor disposal, CUDA cache clearing, and aggressive worker process reaping.
9. **Network Studio & Registry Polish**: Resolved parameter calculation race conditions, added safe deletion with training lockouts, and integrated educational architectural documentation for all 16 layer types.

---

## 2. Rebranding to "LLM Maker"

The application has been rebranded as **LLM Maker**:
- **Browser Title**: `LLM Maker — Neural Research Workstation`
- **Sidebar Header**: `LLM Maker v2` with subtitle `MODEL: RoCanon-48.5M`
- **Topbar Banner**: `LLM Maker Console`
- **Package Manifests**: `package.json` (`name: "llm-maker"`), `frontend/package.json` (`name: "llm-maker-frontend"`)
- **Startup Script**: `run_studio_v2.bat` updated with high-tech ASCII banner and environment initialization.
- **Architectural Isolation**: RoCanon-48.5M remains registered in `networks/registry/rocanon_48m.json` with protected status against deletion or parameter mutation.

---

## 3. Validation Loss Persistence & Frequency Fix

### 3.1 The Vanishing Validation Loss Bug
- **Symptom**: Validation loss turned into `--` upon tab navigation and browser refresh.
- **Root Cause**: On non-eval steps, the worker emitted `val_loss: null`. The frontend WebSocket handler directly updated `latestMetric.val_loss = null`, overwriting the display card. Furthermore, switching tabs initialized an empty buffer before database synchronization finished.
- **Remediation**:
  1. Worker maintains persistent internal state: `self.latest_val_loss`, `self.latest_val_step`, and `self.latest_val_timestamp`.
  2. Telemetry payload broadcasts these fields on every step.
  3. `database.py` added `get_latest_validation(run_id)` query.
  4. Frontend `websocket.ts` maintains an independent `validationState` store that persists across tab changes and scans history buffers on initial sync.

### 3.2 The Validation Frequency Bug (Every 10 Steps vs Every 250 Steps)
- **Root Cause**:
  1. Frontend payload sent `validation_interval: 250`.
  2. `training_worker.py` line 434 checked `payload.get("eval_frequency")` or fallback to `config_es.get("eval_frequency", 10)`. Because the key was `validation_interval`, it silently evaluated to `None` and fell back to `10`.
  3. `training_worker.py` line 650 contained `or (step == 1)`, forcing an immediate expensive evaluation on step 1.
- **Remediation**:
  - Unified parameter extraction in `training_worker.py`:
    ```python
    eval_freq = int(
        payload.get("validation_interval")
        or payload.get("val_interval")
        or payload.get("eval_frequency")
        or payload.get("eval_freq")
        or config_es.get("eval_frequency", 250)
    )
    ```
  - Removed `or (step == 1)`.
  - Condition is strictly: `(eval_freq > 0 and (step - start_step) > 0 and (step - start_step) % eval_freq == 0) or self.manual_val_requested`.
  - Verified across intervals 10, 50, 100, 250, and 500 in unit tests.

---

## 4. Training Objectives & Dataset Token Accounting

### 4.1 Steps Mode vs Epochs Mode
- **Steps Mode**: Evaluates `step >= total_steps`.
- **Epochs Mode**: Evaluates `total_tokens >= target_epochs * train_len`.
- In `TrainingView.tsx`, an objective toggle switch enables seamless switching between budgets, with dynamic parameter inputs.

### 4.2 Token Tracking Matrix
- **Tokens Seen**: Cumulative total across the entire lifetime of the run (`formatCompactNumber(tokens_seen)`).
- **Tokens in Epoch**: `total_tokens % train_len`, showing real progression through the current dataset cycle.
- **Epoch Progress**: Displayed as `Epoch X / Y | Z tokens / total tokens (%)`.

---

## 5. Timing & Latency Precision

| Metric | Formatter | Behavior | Example |
| :--- | :--- | :--- | :--- |
| **Iteration Latency** | `formatStepLatency(ms)` | Raw batch execution time (<1000ms in ms, >=1000ms in seconds) | `248 ms` / `1.14 s` |
| **Active Duration** | `formatDuration(s)` | Cumulative worker active processing time, excluding pauses | `18:42` (<1h) / `03:24:15` (>=1h) |

---

## 6. Learning Rate & Scheduler Transparency

- **Disentangled Metrics**:
  - `muon_lr`: Matrix parameter learning rate (Muon optimizer applied to hidden layers).
  - `adamw_lr`: Vector parameter learning rate (AdamW applied to embeddings, biases, RMSNorms).
- **Scheduler Overwrite Bug Fix**:
  - Fixed internal `trainer.optimizer_step(advance_scheduler=False)` call in `training_worker.py` to prevent the internal `WSDScheduler` from stomping on worker-level learning rate multipliers.
- **Safety Safeguard**: Amber alert banner triggers in UI if `muon_lr > 0.05` or `peak_lr > 0.01`.
