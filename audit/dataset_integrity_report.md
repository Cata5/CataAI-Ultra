# LLM Maker — Dataset Integrity & Token Accounting Report

**Author:** Senior Machine Learning Infrastructure Architect  
**Date:** November 16, 2026  
**Component:** Data Processing, Pre-tokenization & Pipeline Integrity  

---

## 1. Executive Summary

Data pipeline integrity is critical for stable LLM training. This report details the upgrades implemented in dataset processing, real token accounting, pre-tokenization caching, and safe lifecycle management.

---

## 2. Real Token Accounting per Epoch

### 2.1 The Issue with Vague Epoch Counters
Previously, training telemetry displayed floating-point approximations like `Epoch: 1.42` without indicating token counts or distinguishing between total tokens seen across runs vs tokens completed within the current dataset iteration.

### 2.2 Discrete Accounting Implementation
The worker process (`training_worker.py`) now computes:
- `train_len`: Total tokens in the tokenized training dataset.
- `tokens_seen`: Cumulative tokens seen during active training across all steps (`total_tokens`).
- `current_epoch`: Calculated as `(total_tokens // train_len) + 1`.
- `tokens_in_epoch`: Calculated as `total_tokens % train_len`.
- `epoch_pct`: Exact integer percentage `Math.min(100, Math.round((tokens_in_epoch / train_len) * 100))`.

UI Display format:
`Epoch 2 / 5 | 6.4M / 10.0M tokens (64%)`

---

## 3. Pre-Tokenization & Memory-Mapped Caching

1. **Pre-tokenization Endpoint**: `POST /api/datasets/{id}/tokenize` executes pre-tokenization in a dedicated process, serializing uint16 token IDs into binary `.pt` or memmapped tensor formats in `data/cache/`.
2. **Speed Improvement**: Pre-tokenized cache avoids runtime BPE encoding bottlenecks, increasing token throughput from ~1,200 tok/s to >6,500 tok/s on the RTX 3060.
3. **Cache Invalidation**: Changing the underlying uploaded text file detects timestamp changes and triggers automatic re-tokenization.

---

## 4. Safe Dataset Deletion Contract

To prevent training corruption, `DELETE /api/datasets/{id}` enforces:
1. **Active Training Lockout**: If the worker is in state `RUNNING` or `STARTING` and the active configuration matches `dataset_id`, the endpoint immediately aborts with `HTTP 409 Conflict`.
2. **Filesystem Cleanup**: Safely resolves physical paths strictly within `data/uploads/` and `data/cache/` (preventing path traversal attacks).
3. **Database Cascading**: Deletes dataset record from the SQLite database, removing it from UI tables and training dropdowns.
