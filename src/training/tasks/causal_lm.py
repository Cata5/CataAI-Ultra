"""
src/training/tasks/causal_lm.py
Causal Language Modeling (Autoregressive Next-Token Prediction) task adapter.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TaskAdapter, TaskBatch, TaskMetrics


class CausalLMTask(TaskAdapter):
    task_id = "causal_lm"
    task_name = "Causal Language Modeling"
    task_family = "language_model"
    compatible_architecture_families = ["decoder_only", "hybrid", "custom_pytorch"]
    requires_tokenizer = True
    progress_unit = "tokens"

    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0) -> None:
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def prepare_batch(self, raw_data: Any, device: torch.device, dtype: torch.dtype) -> TaskBatch:
        if isinstance(raw_data, (tuple, list)):
            x, y = raw_data[0], raw_data[1]
            return TaskBatch(
                inputs=x.to(device, non_blocking=True),
                targets=y.to(device, non_blocking=True),
            )
        elif isinstance(raw_data, dict):
            input_ids = raw_data.get("input_ids")
            labels = raw_data.get("labels", input_ids)
            return TaskBatch(
                inputs=input_ids.to(device, non_blocking=True),
                targets=labels.to(device, non_blocking=True),
                mask=raw_data.get("attention_mask"),
            )
        else:
            # Single tensor: auto-shift for next-token prediction
            t = raw_data.to(device, non_blocking=True)
            return TaskBatch(inputs=t[:, :-1], targets=t[:, 1:])

    def forward_and_loss(
        self,
        model: nn.Module,
        batch: TaskBatch,
    ) -> Tuple[torch.Tensor, TaskMetrics]:
        # Forward pass
        logits = model(batch.inputs)
        if isinstance(logits, tuple):
            logits = logits[0]

        # Flatten logits and targets
        shift_logits = logits.view(-1, logits.size(-1))
        shift_labels = batch.targets.view(-1)

        loss = F.cross_entropy(
            shift_logits,
            shift_labels,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

        loss_val = float(loss.detach().item())
        try:
            ppl = math.exp(min(loss_val, 20.0))
        except OverflowError:
            ppl = 9999.0

        metrics = TaskMetrics(
            loss=loss_val,
            primary_metric_name="token_loss",
            primary_metric_val=round(loss_val, 4),
            detailed_metrics={
                "perplexity": round(ppl, 2),
            },
        )
        return loss, metrics

    def get_display_metrics(self) -> List[Dict[str, str]]:
        return [
            {"key": "token_loss", "label": "Token Loss", "unit": ""},
            {"key": "perplexity", "label": "Perplexity", "unit": ""},
            {"key": "tokens_per_sec", "label": "Throughput", "unit": "tok/s"},
        ]
