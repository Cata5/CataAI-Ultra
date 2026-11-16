"""
src/training/tasks/classification.py
Classification task adapter for Computer Vision, NLP text classification, and Tabular networks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TaskAdapter, TaskBatch, TaskMetrics


class ClassificationTask(TaskAdapter):
    task_id = "classification"
    task_name = "Classification"
    task_family = "computer_vision"  # Also supports tabular_numerical & general_sequence
    compatible_architecture_families = ["cnn", "mlp", "encoder_only", "rnn_lstm_gru", "custom_pytorch"]
    requires_tokenizer = False
    progress_unit = "samples"

    def __init__(self, num_classes: int = 10, label_smoothing: float = 0.0) -> None:
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def prepare_batch(self, raw_data: Any, device: torch.device, dtype: torch.dtype) -> TaskBatch:
        if isinstance(raw_data, (tuple, list)):
            x, y = raw_data[0], raw_data[1]
            return TaskBatch(
                inputs=x.to(device, dtype=dtype if x.is_floating_point() else torch.long, non_blocking=True),
                targets=y.to(device, dtype=torch.long, non_blocking=True),
            )
        elif isinstance(raw_data, dict):
            return TaskBatch(
                inputs=raw_data["inputs"].to(device, non_blocking=True),
                targets=raw_data["targets"].to(device, dtype=torch.long, non_blocking=True),
            )
        raise ValueError("Unsupported batch structure for classification")

    def forward_and_loss(
        self,
        model: nn.Module,
        batch: TaskBatch,
    ) -> Tuple[torch.Tensor, TaskMetrics]:
        logits = model(batch.inputs)
        if isinstance(logits, tuple):
            logits = logits[0]

        targets = batch.targets.view(-1)
        loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)

        preds = torch.argmax(logits, dim=-1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)
        accuracy = (correct / total * 100.0) if total > 0 else 0.0

        loss_val = float(loss.detach().item())
        metrics = TaskMetrics(
            loss=loss_val,
            primary_metric_name="accuracy",
            primary_metric_val=round(accuracy, 2),
            detailed_metrics={
                "classification_loss": round(loss_val, 4),
                "accuracy": round(accuracy, 2),
                "samples_processed": total,
            },
        )
        return loss, metrics

    def get_display_metrics(self) -> List[Dict[str, str]]:
        return [
            {"key": "classification_loss", "label": "Cross-Entropy Loss", "unit": ""},
            {"key": "accuracy", "label": "Accuracy", "unit": "%"},
            {"key": "samples_per_sec", "label": "Speed", "unit": "img/s"},
        ]
