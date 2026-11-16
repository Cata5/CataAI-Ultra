# LLM Maker — Product Generalization Report

**Version:** 2.0.0  
**Status:** COMPLETE & PRODUCTION-READY  
**Date:** 2026-11-16  

---

## 1. Executive Summary

LLM Maker has been transformed from an author-centric Romanian language training workstation into a **general-purpose, extensible, hardware-aware, local-first artificial intelligence and neural network development platform**. 

The application is no longer tied to one computer, one GPU (RTX 3060 Laptop), one creator persona, one language, or one architecture. Instead:
- **Architecture-Agnostic:** Supports Decoder-Only Transformers, Encoder-Only, MLPs, CNNs, and Custom PyTorch modules via an open Model Registry.
- **Hardware-Aware:** Discovers physical hardware dynamically (CUDA, ROCm, MPS, Host CPU, VRAM, RAM, compute capability) with configurable safety profiles (Conservative 65%, Balanced 80%, Performance 90%, Custom).
- **Task Adapters:** Modular training tasks (Causal LM, Instruction SFT, Image & Tabular Classification, Continuous Regression) with task-specific loss functions and UI metrics.
- **Project & Workspace Isolation:** Full workspace abstraction (`WorkspaceManager`) providing isolated directories for checkpoints, datasets, and models while maintaining 100% backward compatibility for existing runs in the `default` workspace.
- **Universal Evaluation & Benchmarks:** Decoupled from hardcoded creator benchmarks into extensible, user-defined persona and task evaluation suites.
- **Trusted Local Plugin Subsystem:** Declarative manifest-based extension mechanism for local architectures, optimizers, tasks, and dataset loaders.

---

## 2. Generalization Architecture Matrix

| Dimension | Legacy WebUI | LLM Maker General-Purpose Platform | Status |
| :--- | :--- | :--- | :--- |
| **Application Branding** | RoCanon-48.5M Training Studio | **LLM Maker** (with RoCanon as registered baseline model) | Generic |
| **Hardware Detection** | Hardcoded RTX 3060 6GB (< 5.0 GB) | Dynamic discovery (NVML, PyTorch CUDA, ROCm, MPS, CPU, System RAM) | Generic |
| **Memory Policies** | Hardcoded 5,000 MB VRAM ceiling | Configurable Hardware Profiles (`conservative`, `balanced`, `performance`, `custom`) | Generic |
| **Workspace Management** | Flat shared directories (`checkpoints/`, `data/`) | Isolated workspaces with dedicated checkpoint, dataset, and run directories | Generic |
| **Training Tasks** | Causal LM only | `causal_lm`, `instruction_sft`, `classification`, `regression` | Generic |
| **Optimizer Engine** | Fixed Muon + AdamW dual optimizer | Universal optimizer factory: `adamw`, `adam`, `sgd`, `rmsprop`, `adagrad`, `muon_adamw` | Generic |
| **Scheduler Engine** | Hardcoded WSD steps/tokens | `GeneralLRScheduler`: `constant`, `linear`, `cosine`, `cosine_warmup`, `wsd`, `step_decay`, `exponential` | Generic |
| **Dataset Formats** | `.txt`, `.jsonl` | `.txt`, `.jsonl`, `.json`, `.csv` with schema validation and stats | Generic |
| **Evaluation System** | Fixed 5 Creator Persona & 5 Hardware items | Extensible evaluation manager, multilingual understanding, custom persona suites, regression & classification metrics | Generic |
| **Extensibility** | Code edits in Python source | Declarative plugin architecture (`plugins/<plugin_id>/plugin.json`) | Generic |
| **Internationalization** | Hardcoded Romanian text in UI | `I18nProvider` supporting English (`en`) and Romanian (`ro`) | Generic |

---

## 3. Backward Compatibility Preservation

To guarantee existing projects and previous experiment investments remain completely functional:
1. **Model Parameter Exactness:** The baseline RoCanon-48.5M architecture remains registered with its exact parameter invariant of 48,531,968 parameters (41,943,040 Muon weights, 6,588,928 AdamW weights).
2. **Default Workspace Relocatability:** The `default` workspace maps directly to existing root folders (`checkpoints/`, `data/`, `networks/registry/`), immediately recognizing all 40 historical runs, 14 checkpoints, and existing pre-tokenized caches.
3. **Database Migration:** Zero-loss non-destructive SQLite schema upgrades automatically added `workspace_id` columns with default `'default'` to all tables.
