# LLM Maker — Learning Rate Dynamics & Scheduler Report

**Author:** Senior PyTorch & Optimization Engineer  
**Date:** November 16, 2026  
**Subject:** Investigation of High Learning Rate (`2.28e-2`), Muon vs AdamW Scaling, and Scheduler Overwrites  

---

## 1. Executive Summary

Users reported confusion seeing learning rates such as `2.28e-2` (`0.0228`) during RoCanon-48.5M pre-training, suspecting an instability bug.

This investigation determined:
1. `2.28e-2` was the **Muon optimizer effective learning rate** (base `0.02` multiplied by gradient stability factor `1.14`), **not** the AdamW learning rate.
2. The UI displayed a single ambiguous card titled "Learning Rate" showing `eff_muon_lr`, misleading users into believing AdamW was operating at an unsafe magnitude.
3. An internal scheduler bug in `trainer.py` was periodically overriding the worker's dynamic multipliers.

---

## 2. Mathematical Basis of Muon vs AdamW

### 2.1 Why Muon Uses Much Higher Nominal Learning Rates
Standard AdamW operates on individual parameter coordinates using coordinate-wise gradient second moments:
$$\theta_{t+1} = \theta_t - \eta_{\text{AdamW}} \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
For AdamW, $\eta_{\text{AdamW}} \approx 10^{-4} \dots 3 \times 10^{-3}$. Higher rates cause catastrophic parameter explosion.

**Muon (Momentum Orthogonal Optimizer)** operates on 2D weight matrices (hidden linear layers, SwiGLU gates, attention projections) by performing Newton-Schulz matrix polar decomposition to project the gradient onto the closest orthogonal matrix:
$$G_{\text{orth}} = \text{NewtonSchulz}(M_t)$$
$$\theta_{t+1} = \theta_t - \eta_{\text{Muon}} G_{\text{orth}}$$
Because $G_{\text{orth}}$ has unit spectral norm, each parameter update is inherently bounded and uniform across all singular directions. Consequently, the optimal nominal learning rate for Muon is **10x to 20x higher** than AdamW (typically `0.01` to `0.03`).

### 2.2 Derivation of `2.28e-2`
In `training_worker.py`:
- User configured base `muon_lr = 0.02`.
- Gradient norm stability multiplier evaluated to `1.14`.
- Calculated effective Muon LR: $0.02 \times 1.14 = 0.0228 = 2.28 \times 10^{-2}$.
- AdamW LR remained appropriately scaled at `0.003`.

---

## 3. Scheduler Overwrite Bug & Fix

### 3.1 Problem
`trainer.py` contained an internal call to its own `WSDScheduler` inside `optimizer_step()`. On every step, `trainer.py` reset the optimizer parameter group `lr` to its internal schedule, erasing the adaptive multipliers and manual scale overrides from `worker_manager.py`.

### 3.2 Remediation
Added `advance_scheduler: bool = True` to `Trainer.optimizer_step()`. In `training_worker.py`, step execution calls:
```python
trainer.optimizer_step(num_tokens_in_step=total_tokens, advance_scheduler=False)
```
This grants `training_worker.py` complete, authoritative control over:
- WSD schedule phase calculations (warmup, stable, decay).
- Dynamic learning rate scaling (`api.updateLr({ multiplier })`).
- Stability-based LR throttling.

---

## 4. UI Disentanglement & High LR Safety Warnings

1. Replaced the single "Learning Rate" card with:
   - **Muon LR (Matrix)**: Matrix parameter learning rate with schedule phase.
   - **AdamW LR (Vector)**: Embedding and normalization learning rate.
2. Introduced interactive safety warning:
   - If `muon_lr > 0.05` or `adamw_lr > 0.01`, display amber warning alert to prompt user verification.
