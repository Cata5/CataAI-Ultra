# Hardcoded Assumptions Remediation Audit

This document catalogues all personal, hardware-specific, language-specific, and environment-specific assumptions identified in the codebase, alongside their generic replacements in LLM Maker.

---

## 1. Eliminated & Generalized Assumptions

| Scope | Hardcoded Assumption | Replaced By / Remediation | File References |
| :--- | :--- | :--- | :--- |
| **Personal Identity** | Creator persona evaluation (`Manu Cătălin`, author questions) | Extensible Persona & Style evaluation suite (`BenchmarkManager.add_custom_item`) where users define any persona name and ground truth facts. Legacy items preserved under `creator_persona` category. | `src/eval/benchmarks.py`, `src/dashboard/server.py` |
| **Hardware Device** | Hardcoded `NVIDIA RTX 3060 Laptop (6GB)` string in UI cards & Sidebar | Dynamic detection from `HardwareMonitor.read_telemetry()` returning device name, VRAM, and RAM. | `frontend/src/components/layout/Sidebar.tsx`, `frontend/src/components/ui/TelemetryCard.tsx` |
| **VRAM Ceilings** | Hardcoded `< 5.0 GB VRAM` limit and 5,000 MB safety threshold | Configurable `HardwareProfile` with ratios: Conservative (65%), Balanced (80%), Performance (90%), Custom. | `src/telemetry/profiles.py`, `src/telemetry/vram_guard.py` |
| **Memory Estimator** | Hardcoded RTX 3060 6GB evaluation logic (`is_safe = total_gb <= 4.8`) | Dynamic comparison against detected device VRAM / RAM and user profile ratio. | `src/networks/estimator.py`, `src/training/factory.py` |
| **Dataset Format** | Assumption of `.txt` or `.jsonl` causal language text | Support for `.txt`, `.jsonl`, `.json`, and tabular `.csv` datasets across multiple task adapters. | `src/datasets/validator.py`, `src/datasets/manager.py` |
| **Training Task** | Exclusively Causal Language Modeling | Explicit `TaskRegistry` with adapters for `causal_lm`, `instruction_sft`, `classification`, and `regression`. | `src/training/tasks/` |
| **Optimizer** | Fixed Muon + AdamW dual optimizer | Universal optimizer factory `build_optimizer()` supporting `adamw`, `adam`, `sgd`, `rmsprop`, `adagrad`, and `muon_adamw`. | `src/training/optimizer.py` |
| **Scheduler** | Hardcoded 2000-step WSD tied to cumulative tokens | `GeneralLRScheduler` supporting `constant`, `linear`, `cosine`, `cosine_warmup`, `wsd`, `step_decay`, `exponential`. | `src/training/scheduler.py` |
| **Language** | Hardcoded Romanian strings and diacritics defaults in UI | Internationalization system (`I18nProvider`) supporting English and Romanian with persistent locale selection. | `frontend/src/i18n/index.tsx`, `frontend/src/components/layout/Topbar.tsx` |
| **Storage Structure** | Flat shared folders for all experiments | `WorkspaceManager` providing dedicated project directories (`checkpoints/`, `datasets/`, `models/`, `runs/`). | `src/workspace/manager.py` |

---

## 2. Invariant Baselines Preserved

- The **RoCanon-48.5M** architecture is retained as a built-in model template (`rocanon_48m` in Model Registry).
- Byte-Level BPE tokenizer with 12,288 vocabulary size is retained as the baseline tokenizer for RoCanon models.
- The `default` workspace preserves all historical runs (`runs.db`), existing checkpoints in `checkpoints/`, and existing datasets in `data/`.
