"""
src/training/tasks/base.py
Base Task Adapter interface for LLM Maker.
Standardizes input formatting, forward execution, loss computation, and task-specific metrics.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class TaskBatch:
    """Standardized batch payload fed into model forward pass."""
    inputs: Any  # torch.Tensor or Dict[str, torch.Tensor]
    targets: Any  # torch.Tensor
    mask: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskMetrics:
    """Standardized metric output from a training or validation step."""
    loss: float
    primary_metric_name: str  # e.g., "token_loss", "accuracy", "rmse"
    primary_metric_val: float
    detailed_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        res = {"loss": self.loss, self.primary_metric_name: self.primary_metric_val}
        res.update(self.detailed_metrics)
        return res


class TaskAdapter(abc.ABC):
    """Abstract base class for all training tasks in LLM Maker."""

    task_id: str
    task_name: str
    task_family: str  # "language_model", "computer_vision", "tabular_numerical", "general_sequence"
    compatible_architecture_families: List[str]
    requires_tokenizer: bool = False
    progress_unit: str = "steps"  # "tokens", "samples", "steps"

    @abc.abstractmethod
    def prepare_batch(self, raw_data: Any, device: torch.device, dtype: torch.dtype) -> TaskBatch:
        """Converts raw data from dataloader into a standardized TaskBatch on the target device."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward_and_loss(
        self,
        model: nn.Module,
        batch: TaskBatch,
    ) -> Tuple[torch.Tensor, TaskMetrics]:
        """Executes forward pass, computes loss, and returns (loss_tensor, TaskMetrics)."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_display_metrics(self) -> List[Dict[str, str]]:
        """Returns list of metric definitions for the UI cards: [{'key': 'accuracy', 'label': 'Accuracy', 'unit': '%'}]"""
        raise NotImplementedError
