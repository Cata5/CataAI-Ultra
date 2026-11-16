"""
src/training/tasks/instruction_sft.py
Supervised Fine-Tuning (Instruction & Chat SFT) task adapter with prompt loss masking.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TaskAdapter, TaskBatch, TaskMetrics


class InstructionSFTTask(TaskAdapter):
    task_id = "instruction_sft"
    task_name = "Instruction Fine-Tuning (SFT)"
    task_family = "language_model"
    compatible_architecture_families = ["decoder_only", "hybrid", "custom_pytorch"]
    requires_tokenizer = True
    progress_unit = "tokens"

    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0) -> None:
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def prepare_batch(self, raw_data: Any, device: torch.device, dtype: torch.dtype) -> TaskBatch:
        if isinstance(raw_data, dict):
            input_ids = raw_data["input_ids"].to(device, non_blocking=True)
            labels = raw_data.get("labels", input_ids).to(device, non_blocking=True)
            return TaskBatch(inputs=input_ids, targets=labels)
        elif isinstance(raw_data, (tuple, list)):
            return TaskBatch(
                inputs=raw_data[0].to(device, non_blocking=True),
                targets=raw_data[1].to(device, non_blocking=True),
            )
        t = raw_data.to(device, non_blocking=True)
        return TaskBatch(inputs=t[:, :-1], targets=t[:, 1:])

    def forward_and_loss(
        self,
        model: nn.Module,
        batch: TaskBatch,
    ) -> Tuple[torch.Tensor, TaskMetrics]:
        logits = model(batch.inputs)
        if isinstance(logits, tuple):
            logits = logits[0]

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
            primary_metric_name="sft_loss",
            primary_metric_val=round(loss_val, 4),
            detailed_metrics={
                "token_loss": round(loss_val, 4),
                "perplexity": round(ppl, 2),
            },
        )
        return loss, metrics

    def get_display_metrics(self) -> List[Dict[str, str]]:
        return [
            {"key": "sft_loss", "label": "Instruction Loss", "unit": ""},
            {"key": "perplexity", "label": "Perplexity", "unit": ""},
            {"key": "tokens_per_sec", "label": "Tokens / sec", "unit": "tok/s"},
        ]
