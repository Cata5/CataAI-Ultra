"""
src/training/tasks/regression.py
Regression task adapter for continuous values (Tabular, Time-Series, General Neural Networks).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TaskAdapter, TaskBatch, TaskMetrics


class RegressionTask(TaskAdapter):
    task_id = "regression"
    task_name = "Regression"
    task_family = "tabular_numerical"
    compatible_architecture_families = ["mlp", "cnn", "rnn_lstm_gru", "encoder_only", "custom_pytorch"]
    requires_tokenizer = False
    progress_unit = "samples"

    def __init__(self, loss_type: str = "mse") -> None:
        self.loss_type = loss_type.lower()

    def prepare_batch(self, raw_data: Any, device: torch.device, dtype: torch.dtype) -> TaskBatch:
        if isinstance(raw_data, (tuple, list)):
            x, y = raw_data[0], raw_data[1]
            return TaskBatch(
                inputs=x.to(device, dtype=dtype, non_blocking=True),
                targets=y.to(device, dtype=dtype, non_blocking=True),
            )
        elif isinstance(raw_data, dict):
            return TaskBatch(
                inputs=raw_data["inputs"].to(device, dtype=dtype, non_blocking=True),
                targets=raw_data["targets"].to(device, dtype=dtype, non_blocking=True),
            )
        raise ValueError("Unsupported batch structure for regression")

    def forward_and_loss(
        self,
        model: nn.Module,
        batch: TaskBatch,
    ) -> Tuple[torch.Tensor, TaskMetrics]:
        preds = model(batch.inputs)
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = preds.view_as(batch.targets)

        if self.loss_type == "l1":
            loss = F.l1_loss(preds, batch.targets)
        else:
            loss = F.mse_loss(preds, batch.targets)

        with torch.no_grad():
            mae = float(F.l1_loss(preds, batch.targets).item())
            mse = float(F.mse_loss(preds, batch.targets).item())
            rmse = math.sqrt(mse)

        loss_val = float(loss.detach().item())
        metrics = TaskMetrics(
            loss=loss_val,
            primary_metric_name="rmse",
            primary_metric_val=round(rmse, 4),
            detailed_metrics={
                "regression_loss": round(loss_val, 4),
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "samples_processed": batch.targets.size(0),
            },
        )
        return loss, metrics

    def get_display_metrics(self) -> List[Dict[str, str]]:
        return [
            {"key": "regression_loss", "label": "Regression Loss", "unit": ""},
            {"key": "mae", "label": "Mean Absolute Error (MAE)", "unit": ""},
            {"key": "rmse", "label": "Root Mean Squared Error (RMSE)", "unit": ""},
            {"key": "samples_per_sec", "label": "Samples / sec", "unit": "samples/s"},
        ]
