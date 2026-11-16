"""
src/training/tasks/__init__.py
"""
from .base import TaskAdapter, TaskBatch, TaskMetrics
from .causal_lm import CausalLMTask
from .classification import ClassificationTask
from .instruction_sft import InstructionSFTTask
from .regression import RegressionTask
from .registry import TaskRegistry

__all__ = [
    "TaskAdapter",
    "TaskBatch",
    "TaskMetrics",
    "CausalLMTask",
    "InstructionSFTTask",
    "ClassificationTask",
    "RegressionTask",
    "TaskRegistry",
]
