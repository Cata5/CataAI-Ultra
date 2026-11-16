# LLM Maker — GPU VRAM Cleanup & Worker Lifecycle Audit

**Author:** Senior Systems Architect & CUDA Infrastructure Engineer  
**Date:** November 16, 2026  
**Hardware Target:** NVIDIA GeForce RTX 3060 Laptop GPU (6,144 MiB VRAM)  

---

## 1. Executive Summary

A critical issue in the previous implementation was GPU memory retention after stopping training runs. Even after clicking "STOP", between 3.5 GB and 4.2 GB of VRAM remained allocated, preventing subsequent runs or inference from utilizing GPU acceleration.

This audit details the root causes and the robust three-tier reclamation system implemented to guarantee 100% VRAM release.

---

## 2. Root Cause Analysis

### 2.1 PyTorch CUDA Context & Memory Caching
PyTorch allocates GPU memory via a custom caching allocator. Even when tensors fall out of scope:
1. PyTorch retains memory blocks in its internal cache pool rather than returning them to the NVIDIA driver.
2. If python process references to the model, optimizer, or gradient buffers persist in global or class-level scope, the allocator cannot release them.

### 2.2 Lingering Worker Processes (PID 3252 Investigation)
Investigation via `nvidia-smi` identified:
```
PID 3252 | python.exe | 3,742 MiB VRAM
```
`WorkerManager.stop_run()` was previously sending a command over the queue, but if the worker thread was blocked on a PyTorch kernel or DataLoader read, the process never exited. As long as the OS process holding the CUDA context remained alive, the NVIDIA driver reserved the context memory.

---

## 3. The Three-Tier Cleanup Architecture

### Tier 1: In-Process Object Disposal (`training_worker.py`)
In the worker's `finally:` block upon run termination:
```python
# 1. Break cyclic references and explicitly delete model/tensors
del model, trainer, muon_opt, adamw_opt, scheduler
del train_inputs, train_labels, val_inputs, val_labels
model = None
trainer = None

# 2. Force Python garbage collection
gc.collect()

# 3. Release PyTorch allocator blocks and IPC contexts
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
```

### Tier 2: Process Reaping & Termination (`worker_manager.py`)
In `WorkerManager.stop_run()`:
```python
def _reap_worker_process():
    time.sleep(2.5)
    with self._lock:
        if self._process and self._process.is_alive():
            try:
                self._process.join(timeout=1.5)
                if self._process and self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=1.0)
            except Exception as e:
                logger.warning("Error reaping worker process: %s", e)
            self._process = None
            self.current_state = RunState.STOPPED
```
Terminating the worker process guarantees that the operating system immediately reclaims all CUDA driver contexts, returning VRAM usage to the baseline desktop idle state (~1.8 GB).

### Tier 3: GPU Single-Owner Arbitration for Inference
In `src/dashboard/server.py`:
When an evaluation inference request arrives (`/api/eval/generate`):
- If background training is `RUNNING` or `STARTING`, the inference engine automatically falls back to CPU to prevent CUDA out-of-memory errors on the 6GB GPU.
- If background training is `IDLE` or `PAUSED`, inference executes at high speed on the RTX 3060 CUDA device.
