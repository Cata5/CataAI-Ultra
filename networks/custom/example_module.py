"""
networks/custom/example_module.py
Example developer-contributed custom PyTorch module for Network Studio.
Demonstrates how to register novel components with schemas and forward logic.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.networks.custom_loader import CustomModuleBase


class GatedConvolutionalUnit(CustomModuleBase):
    """
    Combines 1D depthwise convolutions with a SiLU gating mechanism
    for efficient local receptive field expansion.
    """
    module_name = "gated_conv_unit"
    description = "Gated Depthwise 1D Convolution with SiLU elementwise multiplication"
    parameter_schema = {
        "dim": {"type": "integer", "default": 512, "description": "Channel dimension"},
        "kernel_size": {"type": "integer", "default": 5, "description": "Kernel window size"},
    }
    input_shape_requirements = "[Batch, Sequence, Dim]"
    output_shape_rules = "[Batch, Sequence, Dim]"

    def __init__(self, dim: int = 512, kernel_size: int = 5, **kwargs) -> None:
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim * 2,
            kernel_size=kernel_size,
            groups=dim,
            bias=False,
        )
        self.out_proj = nn.Linear(dim, dim, bias=False)

    @classmethod
    def parameter_count(cls, dim: int = 512, kernel_size: int = 5, **kwargs) -> int:
        return (dim * 2 * kernel_size) + (dim * dim)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # x is [B, S, D] -> transpose to [B, D, S]
        b, s, d = x.shape
        xt = x.transpose(1, 2)
        padded = F.pad(xt, (self.pad, 0))
        conv_out = self.conv(padded)[..., :s]  # [B, 2*D, S]

        gate, val = conv_out.chunk(2, dim=1)
        gated = F.silu(gate) * val  # [B, D, S]

        out = gated.transpose(1, 2)
        return self.out_proj(out)

