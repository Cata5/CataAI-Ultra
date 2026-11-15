"""
RoCanon Model Package.

Exposes RoCanon-48.5M architectural components, configurations, and causal convolutions.
"""

from .architecture import (
    RMSNorm,
    RoCanonAttention,
    RoCanonBlock,
    RoCanonConfig,
    RoCanonForCausalLM,
    RoCanonMLP,
    RotaryEmbedding,
    apply_rope,
)
from .canon import CausalDepthwiseConv1d

__all__ = [
    "CausalDepthwiseConv1d",
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rope",
    "RoCanonAttention",
    "RoCanonMLP",
    "RoCanonBlock",
    "RoCanonConfig",
    "RoCanonForCausalLM",
]
