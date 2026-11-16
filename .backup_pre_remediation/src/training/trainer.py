"""
Training Loop Engine for RoCanon-48.5M.

Implements the high-performance training loop specified in doc.md (Sections G & H):
- Mixed precision training in BF16 (autocast) with FP32 master weights and FP32 gradient accumulation.
- Strict omission of GradScaler (BF16 dynamic range matches FP32).
- Seamless device dispatch: CUDA on GPU, CPU fallback for portable testing.
- Token-normalized loss backward pass across gradient accumulation micro-batches.
- Dual optimizer execution: Muon for 2D weights + AdamW for embeddings, norms, and convolutions.
- Global gradient norm clipping threshold at 1.0.
- VRAM hardware monitoring enforcing < 5.0 GB peak allocation.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from ..utils.memory import MemoryTracker
from .loss import ChunkedCrossEntropyLoss, TokenLossAccumulator
from .optimizer import configure_optimizers
from .scheduler import WSDScheduler, clip_gradient_norm

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Hyperparameter configuration for RoCanon-48.5M training execution.
    """
    device: str | None = None
    dtype: torch.dtype = torch.bfloat16
    max_grad_norm: float = 1.0
    grad_accum_steps: int = 8
    micro_batch_size: int = 8
    seq_len: int = 1024
    log_interval: int = 10
    max_budget_gb: float = 5.0
    muon_lr: float = 3e-3
    muon_wd: float = 0.01
    adamw_lr: float = 6e-4
    adamw_wd: float = 0.01
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1e-8


class Trainer:
    """
    RoCanon-48.5M Mixed-Precision Training Engine.

    Coordinates forward passes, chunked cross-entropy evaluation,
    token-normalized gradient accumulation, gradient norm clipping at 1.0,
    dual optimizer steps (Muon + AdamW), and WSD learning rate scheduling.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizers: tuple[Optimizer, Optimizer] | None = None,
        scheduler: WSDScheduler | None = None,
        loss_fn: ChunkedCrossEntropyLoss | None = None,
        memory_tracker: MemoryTracker | None = None,
        config: TrainingConfig | None = None,
    ) -> None:
        self.config = config or TrainingConfig()

        # Determine target device
        if self.config.device is not None:
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to target device (master weights kept in FP32)
        self.model = model.to(device=self.device, dtype=torch.float32)

        # Optimizers (Muon + AdamW)
        if optimizers is not None:
            self.muon_optimizer, self.adamw_optimizer = optimizers
        else:
            self.muon_optimizer, self.adamw_optimizer = configure_optimizers(
                self.model,
                muon_lr=self.config.muon_lr,
                muon_wd=self.config.muon_wd,
                adamw_lr=self.config.adamw_lr,
                adamw_wd=self.config.adamw_wd,
                adamw_betas=self.config.adamw_betas,
                adamw_eps=self.config.adamw_eps,
                fused=True,
            )

        # Scheduler
        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = WSDScheduler(
                optimizers=(self.muon_optimizer, self.adamw_optimizer),
            )

        # Loss function
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = ChunkedCrossEntropyLoss(chunk_size=2048)

        # Memory tracker
        if memory_tracker is not None:
            self.memory_tracker = memory_tracker
        else:
            self.memory_tracker = MemoryTracker(
                device=self.device,
                max_budget_gb=self.config.max_budget_gb,
            )

        self.global_step: int = 0
        self.cumulative_tokens: int = 0
        self.loss_accumulator = TokenLossAccumulator()

    def zero_grad(self) -> None:
        """Clears gradients for all optimizers."""
        self.muon_optimizer.zero_grad(set_to_none=True)
        self.adamw_optimizer.zero_grad(set_to_none=True)

    def accumulate_micro_batch(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
        valid_normalizer: int | None = None,
    ) -> tuple[float, int]:
        """
        Executes a single micro-batch forward and backward pass with BF16 autocast.

        Args:
            input_ids: Input tokens [B, S].
            labels: Optional target tokens [B, S]. If None, shifted input_ids are used.
            cu_seqlens: Optional packed sequence delimiter tensor.
            valid_normalizer: Total valid tokens across all micro-batches for exact normalization.

        Returns:
            Tuple of (unscaled_loss_float, valid_tokens_int).
        """
        input_ids = input_ids.to(device=self.device, non_blocking=True)
        if labels is None:
            # Standard causal LM next-token prediction
            target_ids = input_ids.clone()
        else:
            target_ids = labels.to(device=self.device, non_blocking=True)

        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.to(device=self.device, non_blocking=True)

        # Device type for autocast ('cuda' or 'cpu')
        device_type = self.device.type

        # BF16 Autocast forward pass
        with torch.autocast(device_type=device_type, dtype=self.config.dtype):
            # Forward pass through model backbone up to logits
            logits = self.model(input_ids, cu_seqlens=cu_seqlens)
            if isinstance(logits, tuple):
                logits = logits[0]

            # Shift logits and targets for causal next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = target_ids[..., 1:].contiguous()

            # Compute chunked cross-entropy with token normalization
            loss, valid_tokens = self.loss_fn(
                shift_logits,
                shift_targets,
                valid_normalizer=valid_normalizer,
            )

        # Backward pass (FP32 accumulation)
        loss.backward()

        loss_val = float(loss.item())
        # If valid_normalizer was used, scale back to unnormalized sum for logging
        if valid_normalizer is not None and valid_normalizer > 0:
            loss_sum = loss_val * float(valid_normalizer)
        else:
            loss_sum = loss_val * float(valid_tokens)

        self.loss_accumulator.add(loss_sum, valid_tokens)
        return loss_val, valid_tokens

    def optimizer_step(self, num_tokens_in_step: int | None = None) -> float:
        """
        Executes gradient norm clipping at 1.0, steps both optimizers, and advances scheduler.

        Args:
            num_tokens_in_step: Cumulative tokens processed in this step.

        Returns:
            Gradient norm prior to clipping.
        """
        # 1. Global gradient norm clipping threshold at 1.0
        grad_norm = clip_gradient_norm(self.model, max_norm=self.config.max_grad_norm)

        # 2. Step dual optimizers
        self.muon_optimizer.step()
        self.adamw_optimizer.step()

        # 3. Clear gradients
        self.zero_grad()

        # 4. Advance scheduler
        tokens = num_tokens_in_step if num_tokens_in_step is not None else self.loss_accumulator.total_valid_tokens
        self.cumulative_tokens += tokens
        self.global_step += 1
        self.scheduler.step(tokens=self.cumulative_tokens)

        # 5. Check memory budget
        self.memory_tracker.assert_within_budget()

        return grad_norm

    def training_step(
        self,
        micro_batches: Sequence[dict[str, Tensor] | tuple[Tensor, ...]],
    ) -> dict[str, Any]:
        """
        Executes a complete training step over accumulated micro-batches.

        Args:
            micro_batches: Sequence of micro-batches (dictionaries or tuples).

        Returns:
            Dictionary containing metrics: step, loss, grad_norm, tokens, lrs, memory.
        """
        self.model.train()
        self.zero_grad()
        self.loss_accumulator.reset()

        # Pre-count total valid tokens across micro-batches for exact normalization
        total_valid_tokens = 0
        for batch in micro_batches:
            if isinstance(batch, dict):
                ids = batch.get("labels", batch.get("input_ids"))
            elif isinstance(batch, (tuple, list)):
                ids = batch[1] if len(batch) > 1 else batch[0]
            else:
                ids = batch

            if ids is not None:
                # Account for sequence shifting (-1 token per sequence)
                valid = int((ids != self.loss_fn.ignore_index).sum().item())
                # If sequence length > 1, shift leaves (seq_len - 1) tokens
                total_valid_tokens += max(0, valid - ids.size(0))

        total_valid_tokens = max(total_valid_tokens, 1)

        # Accumulate gradients across micro-batches
        for batch in micro_batches:
            if isinstance(batch, dict):
                input_ids = batch["input_ids"]
                labels = batch.get("labels", None)
                cu_seqlens = batch.get("cu_seqlens", None)
            elif isinstance(batch, (tuple, list)):
                input_ids = batch[0]
                labels = batch[1] if len(batch) > 1 else None
                cu_seqlens = batch[2] if len(batch) > 2 else None
            else:
                input_ids = batch
                labels = None
                cu_seqlens = None

            self.accumulate_micro_batch(
                input_ids=input_ids,
                labels=labels,
                cu_seqlens=cu_seqlens,
                valid_normalizer=total_valid_tokens,
            )

        # Optimizer step
        grad_norm = self.optimizer_step(num_tokens_in_step=total_valid_tokens)
        step_loss = self.loss_accumulator.mean_loss

        last_lrs = self.scheduler.get_last_lr()
        muon_lr = last_lrs[0][0] if len(last_lrs) > 0 and len(last_lrs[0]) > 0 else 0.0
        adamw_lr = last_lrs[1][0] if len(last_lrs) > 1 and len(last_lrs[1]) > 0 else 0.0

        return {
            "step": self.global_step,
            "loss": step_loss,
            "grad_norm": grad_norm,
            "cumulative_tokens": self.cumulative_tokens,
            "muon_lr": muon_lr,
            "adamw_lr": adamw_lr,
            "peak_vram_gb": self.memory_tracker.get_peak_allocated_gb(),
        }
