"""
src/training/tasks/registry.py
Central registry for Training Tasks in LLM Maker.
Handles task discovery, validation against architecture families, and adapter instantiation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

from .base import TaskAdapter
from .causal_lm import CausalLMTask
from .classification import ClassificationTask
from .instruction_sft import InstructionSFTTask
from .regression import RegressionTask
from .simpo import SimPOTask
from .orpo import ORPOTask
from .grpo import GRPOTask
from .dpo import DPOTask

logger = logging.getLogger(__name__)


class TaskRegistry:
    """Registry managing available training task adapters."""

    _TASKS: Dict[str, Type[TaskAdapter]] = {
        "causal_lm": CausalLMTask,
        "instruction_sft": InstructionSFTTask,
        "classification": ClassificationTask,
        "regression": RegressionTask,
        "simpo": SimPOTask,
        "orpo": ORPOTask,
        "grpo": GRPOTask,
        "dpo": DPOTask,
    }

    @classmethod
    def register(cls, task_id: str, adapter_cls: Type[TaskAdapter]) -> None:
        cls._TASKS[task_id.lower()] = adapter_cls

    @classmethod
    def get(cls, task_id: str, **kwargs) -> TaskAdapter:
        tid = task_id.lower()
        if tid not in cls._TASKS:
            logger.warning("Task '%s' not registered. Falling back to 'causal_lm'.", task_id)
            tid = "causal_lm"
        return cls._TASKS[tid](**kwargs)

    @classmethod
    def list_tasks(cls) -> List[Dict[str, Any]]:
        result = []
        for tid, task_cls in cls._TASKS.items():
            result.append({
                "task_id": task_cls.task_id,
                "task_name": task_cls.task_name,
                "task_family": task_cls.task_family,
                "compatible_architecture_families": task_cls.compatible_architecture_families,
                "requires_tokenizer": task_cls.requires_tokenizer,
                "progress_unit": task_cls.progress_unit,
            })
        return result

    @classmethod
    def validate_compatibility(cls, task_id: str, architecture_family: str) -> bool:
        tid = task_id.lower()
        if tid not in cls._TASKS:
            return False
        task_cls = cls._TASKS[tid]
        arch = architecture_family.lower()
        if "custom_pytorch" in task_cls.compatible_architecture_families:
            return True
        return arch in [f.lower() for f in task_cls.compatible_architecture_families]
