"""
src/pipeline/scheduler_math.py
Mathematical implementations of learning rate schedules shared directly between the
Visualizer and PyTorch training execution to guarantee 100% mathematical parity.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


def calculate_schedule_curve(
    schedule_type: str,
    total_steps: int,
    peak_lr: float,
    min_lr: float = 0.0,
    warmup_steps: int = 200,
    stable_steps: int = 600,
    decay_style: str = "cosine",
    milestones: List[int] | None = None,
    gamma: float = 0.1,
    num_points: int = 200,
) -> List[Dict[str, float]]:
    """
    Computes (step, lr) points using the exact same formulas as WSDScheduler.
    Used by the UI visualizer to render the real PyTorch curve.
    """
    total_steps = max(10, total_steps)
    step_stride = max(1, total_steps // num_points)
    curve: List[Dict[str, float]] = []

    min_lr_ratio = min_lr / max(1e-12, peak_lr) if peak_lr > 0 else 0.0

    for step in range(0, total_steps + 1, step_stride):
        lr = calculate_step_lr(
            step=step,
            schedule_type=schedule_type,
            total_steps=total_steps,
            peak_lr=peak_lr,
            min_lr_ratio=min_lr_ratio,
            warmup_steps=warmup_steps,
            stable_steps=stable_steps,
            decay_style=decay_style,
            milestones=milestones or [],
            gamma=gamma,
        )
        curve.append({"step": step, "lr": lr})

    if curve[-1]["step"] < total_steps:
        final_lr = calculate_step_lr(
            step=total_steps,
            schedule_type=schedule_type,
            total_steps=total_steps,
            peak_lr=peak_lr,
            min_lr_ratio=min_lr_ratio,
            warmup_steps=warmup_steps,
            stable_steps=stable_steps,
            decay_style=decay_style,
            milestones=milestones or [],
            gamma=gamma,
        )
        curve.append({"step": total_steps, "lr": final_lr})

    return curve


def calculate_step_lr(
    step: int,
    schedule_type: str,
    total_steps: int,
    peak_lr: float,
    min_lr_ratio: float = 0.1,
    warmup_steps: int = 200,
    stable_steps: int = 600,
    decay_style: str = "cosine",
    milestones: List[int] | None = None,
    gamma: float = 0.1,
) -> float:
    """Exact step-by-step mathematical calculation of the learning rate factor."""
    sched = schedule_type.lower()

    if sched == "constant":
        return peak_lr

    if sched == "milestones":
        factor = 1.0
        if milestones:
            for m in sorted(milestones):
                if step >= m:
                    factor *= gamma
        return float(peak_lr * factor)

    # Warmup for all dynamic schedules
    if step < warmup_steps:
        if warmup_steps <= 0:
            return peak_lr
        progress = float(step) / float(warmup_steps)
        return float(peak_lr * progress)

    if sched == "wsd":
        # WSD: Warmup -> Stable -> Decay
        if step <= (warmup_steps + stable_steps):
            return peak_lr

        decay_start = warmup_steps + stable_steps
        decay_span = total_steps - decay_start
        if decay_span <= 0:
            return float(peak_lr * min_lr_ratio)

        progress = max(0.0, min(1.0, (step - decay_start) / float(decay_span)))
        if decay_style == "linear":
            factor = 1.0 - (1.0 - min_lr_ratio) * progress
        else:  # cosine
            cosine_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
            factor = min_lr_ratio + (1.0 - min_lr_ratio) * cosine_mult
        return float(peak_lr * factor)

    if sched == "cosine":
        # Standard cosine decay after warmup
        decay_steps = total_steps - warmup_steps
        if decay_steps <= 0:
            return float(peak_lr * min_lr_ratio)
        progress = max(0.0, min(1.0, (step - warmup_steps) / float(decay_steps)))
        cosine_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
        factor = min_lr_ratio + (1.0 - min_lr_ratio) * cosine_mult
        return float(peak_lr * factor)

    if sched == "linear":
        decay_steps = total_steps - warmup_steps
        if decay_steps <= 0:
            return float(peak_lr * min_lr_ratio)
        progress = max(0.0, min(1.0, (step - warmup_steps) / float(decay_steps)))
        factor = 1.0 - (1.0 - min_lr_ratio) * progress
        return float(peak_lr * factor)

    return peak_lr

