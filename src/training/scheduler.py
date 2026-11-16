"""
Learning rate and stage-wise batch size scheduling for RoCanon-48.5M.

Implements:
1. Warmup-Stable-Decay (WSD):
   - Warmup: 2,000 steps linear increase from 0 to peak LR.
   - Stable plateau: constant at peak LR up to 5.0B cumulative tokens.
   - Decay: from 5.0B to 6.8B cumulative tokens, anneals to 10% peak LR (cosine or linear).
   - Synchronous LR updates for dual optimizers (Muon + AdamW).
2. Stage-Wise Batch Size Scheduling:
   - Stage 1 Early (0 to 3.5B tokens): grad_accum = 8 (65,536 tokens/step).
   - Stage 1 Late (3.5B to 5.0B tokens): grad_accum = 16 (131,072 tokens/step).
   - Stage 2 (5.0B to 6.8B tokens): seq_len = 2048, micro_batch = 4, grad_accum = 16 (131,072 tokens/step).
3. Global gradient clipping utility at max_norm = 1.0.
"""

import math
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


class WSDScheduler:
    """
    Warmup-Stable-Decay (WSD) Learning Rate Scheduler.

    Ties the learning rate to optimizer steps during warmup and to cumulative
    tokens during the stable plateau and decay phases.
    Supports a single optimizer or a tuple/list of optimizers (e.g. Muon + AdamW).
    """

    def __init__(
        self,
        optimizers: Optimizer | Sequence[Optimizer],
        warmup_steps: int = 2000,
        plateau_tokens: int = 5_000_000_000,
        total_tokens: int = 6_800_000_000,
        min_lr_ratio: float = 0.1,
        decay_style: str = "cosine",
    ) -> None:
        """
        Args:
            optimizers: Single optimizer or sequence of optimizers (e.g., (muon_opt, adamw_opt)).
            warmup_steps: Number of linear warmup steps (default: 2,000).
            plateau_tokens: Cumulative tokens at which stable plateau ends (default: 5.0B).
            total_tokens: Cumulative tokens at which decay ends (default: 6.8B).
            min_lr_ratio: Final LR as fraction of peak LR (default: 0.1 = 10%).
            decay_style: 'cosine' (default) or 'linear'.
        """
        if isinstance(optimizers, Optimizer):
            self.optimizers = [optimizers]
        else:
            self.optimizers = list(optimizers)

        self.warmup_steps = warmup_steps
        self.plateau_tokens = plateau_tokens
        self.total_tokens = total_tokens
        self.min_lr_ratio = min_lr_ratio
        self.decay_style = decay_style.lower()

        # Store initial learning rates for each param group of each optimizer
        self.base_lrs: list[list[float]] = [
            [group["lr"] for group in opt.param_groups]
            for opt in self.optimizers
        ]

        self.current_step: int = 0
        self.current_tokens: int = 0

    def get_factor(
        self,
        step: int | None = None,
        tokens: int | None = None,
    ) -> float:
        """
        Calculates the learning rate multiplier factor f in [min_lr_ratio, 1.0].

        Args:
            step: Step index override (default: self.current_step).
            tokens: Cumulative token count override (default: self.current_tokens).

        Returns:
            Scalar learning rate scaling factor.
        """
        curr_step = self.current_step if step is None else step
        curr_tokens = self.current_tokens if tokens is None else tokens

        # 1. Warmup Phase (Linear increase from 0 to 1.0)
        if curr_step < self.warmup_steps:
            if self.warmup_steps <= 0:
                return 1.0
            return float(curr_step) / float(self.warmup_steps)

        # 2. Stable Plateau Phase (Constant at peak LR 1.0)
        if curr_tokens <= self.plateau_tokens:
            return 1.0

        # 3. Decay Phase (from plateau_tokens to total_tokens)
        decay_span = self.total_tokens - self.plateau_tokens
        if decay_span <= 0:
            return self.min_lr_ratio

        progress = (curr_tokens - self.plateau_tokens) / float(decay_span)
        progress = max(0.0, min(1.0, progress))

        if self.decay_style == "linear":
            factor = 1.0 - (1.0 - self.min_lr_ratio) * progress
        else:  # cosine decay
            cosine_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
            factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_mult

        return float(factor)

    def step(
        self,
        step: int | None = None,
        tokens: int | None = None,
        num_tokens_in_batch: int | None = None,
    ) -> list[list[float]]:
        """
        Advances the scheduler state and updates learning rates across all optimizers.

        Args:
            step: Specific step number to advance to. If None, increments by 1.
            tokens: Absolute cumulative tokens trained. If None, adds num_tokens_in_batch.
            num_tokens_in_batch: Incremental tokens processed in current step.

        Returns:
            List of updated learning rates per optimizer per group.
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        if tokens is not None:
            self.current_tokens = tokens
        elif num_tokens_in_batch is not None:
            self.current_tokens += num_tokens_in_batch

        factor = self.get_factor()

        current_lrs: list[list[float]] = []
        for opt_idx, opt in enumerate(self.optimizers):
            opt_lrs: list[float] = []
            for group_idx, group in enumerate(opt.param_groups):
                base_lr = self.base_lrs[opt_idx][group_idx]
                new_lr = base_lr * factor
                group["lr"] = new_lr
                opt_lrs.append(new_lr)
            current_lrs.append(opt_lrs)

        return current_lrs

    def get_last_lr(self) -> list[list[float]]:
        """Returns the current learning rates across all registered optimizers."""
        return [
            [group["lr"] for group in opt.param_groups]
            for opt in self.optimizers
        ]

    def state_dict(self) -> dict[str, Any]:
        """Returns dictionary of scheduler state for checkpointing."""
        return {
            "current_step": self.current_step,
            "current_tokens": self.current_tokens,
            "warmup_steps": self.warmup_steps,
            "plateau_tokens": self.plateau_tokens,
            "total_tokens": self.total_tokens,
            "min_lr_ratio": self.min_lr_ratio,
            "decay_style": self.decay_style,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restores scheduler state from dictionary."""
        self.current_step = state_dict["current_step"]
        self.current_tokens = state_dict["current_tokens"]
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
        self.plateau_tokens = state_dict.get("plateau_tokens", self.plateau_tokens)
        self.total_tokens = state_dict.get("total_tokens", self.total_tokens)
        self.min_lr_ratio = state_dict.get("min_lr_ratio", self.min_lr_ratio)
        self.decay_style = state_dict.get("decay_style", self.decay_style)
        if "base_lrs" in state_dict:
            self.base_lrs = state_dict["base_lrs"]
        # Update optimizers immediately
        factor = self.get_factor()
        for opt_idx, opt in enumerate(self.optimizers):
            for group_idx, group in enumerate(opt.param_groups):
                base_lr = self.base_lrs[opt_idx][group_idx]
                group["lr"] = base_lr * factor


class StageBatchScheduler:
    """
    Stage-wise Batch Size Scheduler for RoCanon-48.5M training.

    Manages sequence length, micro-batch size, and gradient accumulation steps:
    - Stage 1 Early (0 to 3.5B tokens): grad_accum=8, seq_len=1024, micro_batch=8 (65,536 tok/step)
    - Stage 1 Late (3.5B to 5.0B tokens): grad_accum=16, seq_len=1024, micro_batch=8 (131,072 tok/step)
    - Stage 2 (5.0B to 6.8B tokens): grad_accum=16, seq_len=2048, micro_batch=4 (131,072 tok/step)
    """

    def __init__(
        self,
        split_3_5b: int = 3_500_000_000,
        split_5_0b: int = 5_000_000_000,
    ) -> None:
        self.split_3_5b = split_3_5b
        self.split_5_0b = split_5_0b

    def get_stage(self, cumulative_tokens: int) -> str:
        """Returns the training stage identifier based on cumulative tokens."""
        if cumulative_tokens < self.split_3_5b:
            return "stage1_early"
        elif cumulative_tokens < self.split_5_0b:
            return "stage1_late"
        else:
            return "stage2"

    def get_grad_accum_steps(self, cumulative_tokens: int) -> int:
        """
        Gradient accumulation steps:
        - 8 steps before 3.5B tokens
        - 16 steps at and after 3.5B tokens
        """
        return 16 if cumulative_tokens >= self.split_3_5b else 8

    def get_seq_len(self, cumulative_tokens: int) -> int:
        """
        Sequence length:
        - 1024 tokens in Stage 1 (< 5.0B tokens)
        - 2048 tokens in Stage 2 (>= 5.0B tokens)
        """
        return 2048 if cumulative_tokens >= self.split_5_0b else 1024

    def get_micro_batch_size(self, cumulative_tokens: int) -> int:
        """
        Micro-batch size:
        - 8 in Stage 1 (8 * 1024 = 8,192 tokens/micro-batch)
        - 4 in Stage 2 (4 * 2048 = 8,192 tokens/micro-batch)
        """
        return 4 if cumulative_tokens >= self.split_5_0b else 8

    def get_global_batch_size_tokens(self, cumulative_tokens: int) -> int:
        """Returns global batch size in tokens per optimizer step."""
        micro_tokens = self.get_micro_batch_size(cumulative_tokens) * self.get_seq_len(cumulative_tokens)
        return self.get_grad_accum_steps(cumulative_tokens) * micro_tokens


def clip_gradient_norm(
    parameters: nn.Module | Iterable[Tensor],
    max_norm: float = 1.0,
    norm_type: float = 2.0,
) -> float:
    """
    Clips global gradient norm of parameters at threshold max_norm (default: 1.0).

    Args:
        parameters: Model or iterable of parameter tensors.
        max_norm: Maximum allowed norm threshold (default: 1.0).
        norm_type: L-p norm type (default: 2.0).

    Returns:
        Total gradient norm prior to clipping.
    """
    if isinstance(parameters, nn.Module):
        params = [p for p in parameters.parameters() if p.grad is not None]
    else:
        params = [p for p in parameters if p.grad is not None]

    if not params:
        return 0.0

    total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm, norm_type=norm_type)
    return float(total_norm)


class GeneralLRScheduler:
    """
    Universal multi-optimizer learning rate scheduler supporting all standard schedules:
    - constant
    - linear / linear_warmup_decay
    - cosine / cosine_warmup
    - wsd (warmup-stable-decay)
    - step_decay
    - exponential
    """

    def __init__(
        self,
        optimizers: Optimizer | Sequence[Optimizer],
        schedule_type: str = "cosine",
        warmup_steps: int = 100,
        total_steps: int = 1000,
        min_lr_ratio: float = 0.1,
        gamma: float = 0.95,
        step_size: int = 200,
    ) -> None:
        if isinstance(optimizers, Optimizer):
            self.optimizers = [optimizers]
        else:
            self.optimizers = list(optimizers)

        self.schedule_type = schedule_type.lower()
        self.warmup_steps = max(0, warmup_steps)
        self.total_steps = max(1, total_steps)
        self.min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))
        self.gamma = gamma
        self.step_size = max(1, step_size)

        self.base_lrs: list[list[float]] = [
            [group["lr"] for group in opt.param_groups]
            for opt in self.optimizers
        ]
        self.current_step: int = 0

    def get_factor(self, step: int) -> float:
        """Computes learning rate scale multiplier in [min_lr_ratio, 1.0] for step."""
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return float(step) / float(self.warmup_steps)

        # After warmup
        effective_step = step - self.warmup_steps
        effective_total = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, effective_step / effective_total))

        st = self.schedule_type
        if st in ("constant", "none"):
            return 1.0
        elif st in ("linear", "linear_decay", "linear_warmup_decay"):
            return 1.0 - progress * (1.0 - self.min_lr_ratio)
        elif st in ("cosine", "cosine_warmup"):
            return self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
        elif st == "wsd":
            # 80% stable plateau, 20% cosine decay
            if progress < 0.8:
                return 1.0
            decay_prog = (progress - 0.8) / 0.2
            return self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1.0 + math.cos(math.pi * decay_prog))
        elif st == "step_decay":
            num_decays = step // self.step_size
            return max(self.min_lr_ratio, float(self.gamma ** num_decays))
        elif st == "exponential":
            return max(self.min_lr_ratio, float(self.gamma ** step))
        return 1.0

    def step(self) -> None:
        self.current_step += 1
        factor = self.get_factor(self.current_step)
        for opt, base_group in zip(self.optimizers, self.base_lrs):
            for group, base_lr in zip(opt.param_groups, base_group):
                group["lr"] = base_lr * factor

    @staticmethod
    def preview_curve(
        schedule_type: str = "cosine",
        warmup_steps: int = 100,
        total_steps: int = 1000,
        peak_lr: float = 3e-4,
        min_lr_ratio: float = 0.1,
        num_points: int = 100,
    ) -> list[dict[str, float]]:
        """Generates (step, lr) points for UI visualization."""
        total_steps = max(1, total_steps)
        step_interval = max(1, total_steps // num_points)
        points = []
        dummy_opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=peak_lr)
        sched = GeneralLRScheduler(
            optimizers=[dummy_opt],
            schedule_type=schedule_type,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        )
        for s in range(0, total_steps + 1, step_interval):
            f = sched.get_factor(s)
            points.append({"step": s, "lr": round(peak_lr * f, 8), "factor": round(f, 4)})
        if points[-1]["step"] < total_steps:
            f = sched.get_factor(total_steps)
            points.append({"step": total_steps, "lr": round(peak_lr * f, 8), "factor": round(f, 4)})
        return points
