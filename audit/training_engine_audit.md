# Training Engine Audit

Scope: `src/training/*`, `src/worker/training_worker.py`, `src/worker/standalone_runner.py`,
`src/checkpoints/manager.py`, `src/model/architecture.py`.

## 1. Execution path

```
START_RUN payload
  → resolve architecture (config_arch / network_id)      training_worker.py:211
  → resolve dataset (dataset_id / dataset_source)        training_worker.py:235
  → create_run row (workspace scoped)                    training_worker.py:315
  → build model + optional LoRA                          training_worker.py:382
  → load dataset tensors (train/val 95/5)                custom_loader.py
  → build trainer, Muon+AdamW, WSD scheduler, early stop training_worker.py:560+
  → loop: accumulate_micro_batch × grad_accum → optimizer_step → metrics/validation/checkpoint
  → terminal: COMPLETED | STOPPED (snapshot) | FAILED
```

`Trainer.accumulate_micro_batch` runs the task adapter's forward/backward under autocast and adds the
token-weighted loss to `TokenLossAccumulator`. `Trainer.optimizer_step` clips the global grad norm,
steps both optimizers, zeroes grads, advances `global_step`/`cumulative_tokens` and optionally the
WSD scheduler, then asserts the VRAM budget.

## 2. Loss and token accounting

* Micro-batch tokens = `micro_batch_size × (seq_len − 1) × grad_accum_steps`.
* The worker adds that number to `total_tokens` for every optimizer step and reports it as
  `tokens_seen`/`tokens` in metrics.
* `TokenLossAccumulator` weights each micro-batch by its valid (unmasked) token count, so the reported
  loss is a true token-weighted mean.
* **Finding AUD-023 (Medium):** for supervised fine-tuning the labels contain `-100` on prompt tokens,
  so the loss uses fewer tokens than `total_tokens` counts. Token-progress/ETA are therefore slightly
  optimistic for SFT runs. The loss itself is correct; only the progress denominators are affected.
  Left as a documented limitation because changing it alters persisted metric semantics.
* **Finding AUD-029 (Low):** when a resumed run has already exceeded its token target, the worker bumps
  the target by `+1e13` (`training_worker.py:561`) to avoid an immediate "completed" state. Progress
  percentages then read ~0 %, which is misleading. The step target still drives completion.

## 3. Optimizers and scheduler

* Hybrid Muon (matrix params) + AdamW (embeddings/norms) with fused kernels when available, both
  stepped every optimizer step; global grad-norm clipping at `max_grad_norm` (default 1.0).
* WSD scheduler implements warmup → stable → decay with absolute step and token counts injected from
  the worker (`scheduler.current_step`, `scheduler.current_tokens`).
* Dynamic LR: the UI can set a multiplier and per-optimizer LRs at runtime; the worker applies them on
  the next step.
* **Fixed AUD-013:** the adaptive-LR controller kept multiplying the user's LR even when
  `auto_lr_adjust` was disabled (`payload.auto_adjust=false`). It is now gated and the factor is reset
  to 1.0 when automatic adjustment is off, so a user-set LR stays exactly where the UI shows it.
* `SET_LR_MULTIPLIER` commands are applied without restarting the run; verified in the worker's
  command poll path.

## 4. Validation, early stopping, GPU safety

* Validation runs every `val_interval` steps over up to 12 batches, producing `val_loss` and the
  generalization gap; the best checkpoint by val loss is tracked for early stopping.
* **Fixed AUD-012:** `eval_freq`/`save_freq` were read with `payload.get(key) or default`, so an
  explicit `0` (disable) silently became the default interval (e.g. every 100 steps). Both sites now
  use `_first_int`, which preserves zero; confirm with the regression test
  `test_first_int_preserves_zero_and_alias_priority`.
* CUDA OOM is caught per step: cache cleared, `CUDA_OOM` alert, transition to `PAUSING`, loop continues
  after the pause instead of killing the run.
* A VRAM guard evaluates effective free memory (free + cache) each step and raises
  `VRAM_PRESSURE` alerts with a 30 s debounce.
* NaN/Inf loss and EMA-relative spikes are logged as alerts and emitted as `LOSS_SPIKE` events.

## 5. Checkpointing and resume

Checkpoint payloads contain `model_state_dict`, both optimizer state dicts, scheduler state, `step`,
`tokens_seen`, losses, phase and tokenizer/config metadata. Saves are atomic (`.tmp` + `os.replace`).

* **Fixed AUD-001 (Critical):** every load went through `torch.load(...)` with unrestricted pickle
  semantics, so any checkpoint file — including one downloaded from the internet — could execute
  arbitrary code when resumed or listed. All loads now go through `safe_torch_load`, which uses
  `weights_only=True` and raises a descriptive `RuntimeError` instead of falling back to unsafe
  unpickling. Verified by a regression test that saves a checkpoint whose `__reduce__` would create a
  file: the load raises and the file is never created.
* **Fixed AUD-003 (Critical):** the stop handler saved *best-validation* weights together with the
  *current* optimizer/scheduler state. Resuming such a checkpoint silently rewound the weights and
  paired them with mismatched moments. The stop snapshot now uses the live model.
* **Fixed AUD-002 (Critical):** the desktop killed the worker 800 ms after `STOP_RUN`, before the
  snapshot was written. Rust now waits up to 20 s for a graceful exit before killing the tree, and the
  worker writes the snapshot before emitting `RUN_STOPPED`.
* **Fixed AUD-004 (Critical):** a failed resume (missing/corrupt checkpoint) was logged and training
  continued from freshly initialized weights — the worst possible silent failure. It now logs a
  `CHECKPOINT_LOAD_FAILED` alert, transitions to `FAILED`, emits `RUN_FAILED` and stops.
* Resume restores step, tokens, optimizer and scheduler state; verification: the lifecycle test resumes
  a stopped snapshot and observes the first new step strictly greater than the stop step.

## 6. Terminal lifecycle and durability

`standalone_runner.py` used to `os._exit(0)` 500 ms after printing a terminal event. Everything the
session still had to do (final DB status update, snapshot fsync, chart generation) raced that timer and
was frequently lost; the audited database copy contained 12 non-terminal rows, which is the signature
of this failure (they are now repaired at startup by AUD-015).

* **Fixed AUD-005 (High):** the worker exposes `session_done` (a `threading.Event`) set in `run_loop`'s
  `finally` after the session has fully finalized; the runner waits up to 30 s for it before exiting.
  Verified: the stop test observes the process exit with code 0 **and** the `STOPPED` row with a
  non-null `final_loss` and a snapshot on disk.
* **Fixed (part of AUD-005):** the final status update now writes step, tokens, best val loss and
  `final_loss` in one `update_run_status` call, guarded for runs that fail before those locals exist.

## 7. Metrics emitted per step

`STEP_COMPLETED` carries ~60 fields: loss/train_loss/val_loss, LR (muon, adamw, base, effective),
schedule phase/factor, grad norm, step time, tokens/s, TFLOPs, VRAM, ETA, early-stopping counters,
best val loss/step, epoch and dataset stats.

* **Fixed AUD-016 (High):** `tflops = tok_per_sec * 354.11e6 / 1e12` assumed a 59 M-parameter model.
  For any other architecture (the product's core feature) the number was simply wrong. It is now
  `6 × trainable_params × tokens/s`, the standard dense-transformer training estimate, with the real
  parameter count logged at startup. Verified: 0.12 TFLOPs on the 2-layer benchmark model, and
  `tflops > 0` is asserted in the lifecycle test.
* **AUD-030 (Low):** `vram_used_gb` reports `peak_allocated`, not current allocation, and the same
  value feeds the VRAM guard as "PyTorch allocated". Peak is the conservative choice for the guard but
  the label overstates current usage in the UI.

## 8. Model architecture notes

`RoCanonForCausalLM` is a decoder-only transformer with RMSNorm, RoPE, grouped-query attention
(`n_q` query heads, `n_kv` KV heads) and SwiGLU feed-forward blocks. `create_model_from_dict` also
dispatches to `NetworkRegistry`/`ModelFactory` for user-designed architectures.

* **Fixed AUD-018 (High):** nothing validated that dataset token ids fit the model vocabulary. A
  mismatch surfaced as an opaque `CUDA error: device-side assert triggered` inside RMSNorm after the
  first kernel launch (reproduced during this audit with a 4,096-vocab network on the 12,288-vocab
  tokenizer). The worker now compares `train_inputs.max()` with the model vocabulary before step 1 and
  fails with `TOKENIZER_MISMATCH` plus an actionable message.

## 9. Observed behaviour (real runs)

| Check | Result |
| --- | --- |
| 120-step run on a 2-layer/1.88 M-param model, seq 64, batch 2 | loss 0.1245 final, median 48.4 ms/step, 10,547 tok/s |
| Stop mid-run | `STOPPED` row with `final_loss`, one `stopped_snapshot_*.pt` |
| Resume from snapshot | continues at stop step + 1, validation fires, run reaches `COMPLETED` |
| Resume from corrupted file | `RUN_FAILED`, status `FAILED`, `CHECKPOINT_LOAD_FAILED` alert, zero steps |
| Vocabulary mismatch | rejected before step 1 with `TOKENIZER_MISMATCH` |
| Unit suite after fixes | 142 passed |

## 10. Remaining engine risks

1. Token accounting for masked SFT data (AUD-023).
2. Token-target progress after resume (AUD-029).
3. `vram_used_gb` semantics (AUD-030).
4. No automated test covers multi-GPU/cluster paths; they remain code-reviewed only.
5. Checkpoint loading has no integrity hash; a truncated file is detected by `torch.load` but a
   structurally valid yet semantically wrong file is not.