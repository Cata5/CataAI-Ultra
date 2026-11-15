"""
src/networks package.
Network Studio: Architecture Builder, Registry, Parameter Calculator, and Validator.
"""

from .components import (
    CausalDepthwiseConv1d,
    GenericTransformerBlock,
    GroupedQueryAttention,
    LearnedPositionalEmbedding,
    RMSNorm,
    SinusoidalPositionalEncoding,
    StandardMLP,
    SwiGLUFFN,
)
from .custom_loader import CustomModuleBase, CustomModuleRegistry
from .estimator import BudgetAssistant, MemoryEstimator, ParameterCalculator
from .factory import GenericModularNetwork, ModelFactory
from .registry import NetworkRegistry
from .templates import ALL_TEMPLATES, get_template_by_id
from .validator import ArchitectureValidator

__all__ = [
    "GenericModularNetwork",
    "ModelFactory",
    "NetworkRegistry",
    "ParameterCalculator",
    "MemoryEstimator",
    "BudgetAssistant",
    "ArchitectureValidator",
    "CustomModuleBase",
    "CustomModuleRegistry",
    "GenericTransformerBlock",
    "GroupedQueryAttention",
    "RMSNorm",
    "SwiGLUFFN",
    "StandardMLP",
    "ALL_TEMPLATES",
    "get_template_by_id",
]

