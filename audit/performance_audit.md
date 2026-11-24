# Performance Audit

## 1. Test environment

| Component | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060 Laptop, 6.0 GB, compute 8.6 |
| Driver / CUDA | CUDA 12.8 runtime, PyTorch 2.11.0+cu128 |
| CPU / RAM | AMD Ryzen (16 logical cores) |
| OS | Windows 11 |
| Storage | local NVMe, repository on `D:` |

## 2. Training throughput (measured)

Method: `tests/integration/audit_worker_benchmark.py 120` spawns the real worker exactly as the desktop
does (`standalone_runner --payload … --db-path …`), trains 120 optimizer steps on the synthetic dataset,
and reports the median over the last 115 steps (5 warm-up steps excluded).

Configuration: RoCanon, `vocab 12288`, `d_model 128`, `n_layers 2`, `n_q 4`, `n_kv 2`, `d_ff 256`,
`seq_len 64`, `micro_batch 2`, `grad_accum 1`, Muon 0.02 / AdamW 1e-3, CUDA bf16 autocast.

| Metric | Run A | Run B |
| --- | ---: | ---: |
| Trainable parameters | 1,876,736 | 1,876,736 |
| Steps measured | 120 | 120 |
| Wall time | (not recorded) | 18.6 s |
| Step time (median) | 47.97 ms | 48.36 ms |
| Step time (min / max) | (not recorded) | 45.6 / 96.5 ms |
| Tokens/s (median) | 10,632.6 | 10,546.6 |
| TFLOPs estimate (median) | 0.120 | 0.119 |
| VRAM peak | 0.11 GB | 0.11 GB |
| Final loss | 0.1157 | 0.1245 |

Interpretation:

* 48 ms/step for 128 tokens/step (2×64) is dominated by kernel-launch overhead, not arithmetic — the
  model is ~1.9 M parameters and the batch is tiny. This is expected for a smoke-scale configuration.
* The TFLOPs estimate is now derived from the real parameter count:
  `6 × 1,876,736 × 10,546.6 / 1e12 = 0.119`, which matches the reported value. Before AUD-016 the
  numerator was hardcoded to a 59 M-parameter model, so every other architecture reported a wildly
  wrong number.
* Run-to-run variance is ~±1 % in step time; max values are first-step CUDA warm-up and occasional
  Windows scheduling jitter.
* The first benchmark taken before all fixes reported 47.97 ms / 10,632.6 tok/s, i.e. the fixes did not
  regress throughput.

## 3. Memory

| Item | Observed |
| --- | --- |
| Peak PyTorch allocation | 0.11 GB (1.9 M model, batch 2) |
| GPU total | 6.0 GB |
| Checkpoint size (order of magnitude) | ~2× params × 4 bytes for weights + 2 optimizer states |

The VRAM guard evaluates `free + reserved-but-unused` each step and raises a debounced `VRAM_PRESSURE`
alert; it targets 90 % of total by default. The reported `vram_used_gb` is `max_memory_allocated` (peak),
not instantaneous usage (AUD-030) — conservative for the guard, slightly misleading in the chart.

## 4. Database and UI overheads

* Metrics are buffered and flushed every 5 steps (or immediately on validation), not per step, which
  keeps SQLite writes off the hot path.
* `STEP_COMPLETED` is sent to Rust once per optimizer step (~20 Hz at 48 ms/step). Each payload is a
  few KB of JSON; the reader thread parses it and emits a Tauri event. This is negligible at the
  measured rate but scales linearly with step frequency for tiny models.
* The frontend re-renders the loss chart and metric grid on each event; React Query batching keeps this
  to one commit per event. No profiling was done in the browser.
* SQLite runs in WAL mode (`PRAGMA journal_mode=WAL`, `synchronous=NORMAL`, set by the Python layer at
  `src/db/database.py:33`; WAL is a persistent database property, so Rust connections inherit it).
  Concurrent writes from Rust (UI settings) and Python (metrics) are short and rare; no
  `database is locked` error appeared in any test run.

## 5. Startup and shutdown

| Phase | Observation |
| --- | --- |
| Tauri startup (temp DB copy, 12 stale runs) | process reached its event loop and logged all startup stages; stale recovery is a single `UPDATE` |
| Worker spawn + teardown | 120 steps took 18.6 s wall; step compute was ~5.8 s (120 × 48 ms), so Python/torch import, dataset setup and process teardown account for ~12–13 s per run |
| Graceful stop | waits for the session to finalize (snapshot write) instead of the old fixed 800 ms; the lifecycle test observed a clean exit well inside the 20 s budget (not timed precisely) |
| Force kill after timeout | 20 s stop budget, then `taskkill /T /F` |

Python/torch import time is the largest fixed cost of starting a run and is unavoidable with the
current process model. Moving the worker to a long-lived process that loads torch once would remove it,
but that is an architectural change beyond this audit.

## 6. Scalability observations

1. The benchmark model is intentionally tiny; the numbers validate the plumbing (no hidden sync, no
   per-step disk write) rather than GPU efficiency. A 48.5 M-parameter run would be compute-bound and
   is the appropriate next measurement (not executed here).
2. Dataset loading is eager: the whole tokenized tensor is loaded into memory before training. This is
   fine for the bundled corpora (tens of MB) but will not scale to multi-GB datasets without a streaming
   sampler.
3. Checkpoint saves copy the full optimizer state; at 48.5 M parameters that is ~0.6 GB per save. The
   default interval (500 steps) keeps this acceptable, but interval `0` plus manual saves is the
   recommended pattern for slow disks.
4. Chart generation runs in the worker at the end of a run; for very long runs the metric list grows
   linearly and plotting cost grows with it.

## 7. Recommendations

| Priority | Recommendation |
| --- | --- |
| High | Benchmark a 48.5 M-parameter / real-dataset run and record tokens/s and VRAM; the current numbers cannot be extrapolated safely |
| Medium | Stream datasets from disk (memory-mapped token buffers) to bound RAM |
| Medium | Enable SQLite WAL and a busy timeout in both writers to remove the residual lock window |
| Low | Decimate WebSocket metric frames for very fast runs (e.g. every 3rd step) |
| Low | Report both current and peak VRAM in the UI |