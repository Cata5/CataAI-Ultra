"""
RoCanon Causal Depthwise 1D Convolutions (Canon Layers).

Provides CausalDepthwiseConv1d as specified in RoCanon-48.5M architectural design
(arXiv:2512.17351 / doc.md Section D).
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalDepthwiseConv1d(nn.Module):
    """
    Causal Depthwise 1D Convolution with zero initialization and residual connection.

    In RoCanon-48.5M, four Canon layers (Canon-A, Canon-B, Canon-C, Canon-D) are
    integrated residually into each transformer block. Because convolution weights are
    strictly initialized to zeros, conv(x) == 0 at initialization, ensuring identity
    residual pass-through (x + conv(x) == x) and preserving transformer training stability.

    Args:
        channels: Number of input and output channels (depthwise: groups == channels).
        kernel_size: Size of the 1D temporal convolution kernel (default: 4).
    """

    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            groups=channels,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize convolution weights strictly to zeros."""
        nn.init.zeros_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal left-padding and residual addition.

        Args:
            x: Input tensor of shape [batch_size, seq_len, channels].

        Returns:
            Output tensor of shape [batch_size, seq_len, channels].
        """
        # x is [B, S, C]
        # Transpose to [B, C, S] for PyTorch Conv1d
        x_trans = x.transpose(1, 2)

        # Causal left-padding: pad k - 1 zeros on the left side of sequence dimension.
        # In F.pad with 3D tensor [B, C, S], pad tuple (pad_left, pad_right) operates on last dim (S).
        padded = F.pad(x_trans, (self.kernel_size - 1, 0), mode="constant", value=0.0)

        # Depthwise 1D convolution produces sequence length (S + k - 1 - k + 1) = S
        conv_out = self.conv(padded)

        # Transpose back to [B, S, C]
        out = conv_out.transpose(1, 2)

        # Residual connection
        return x + out
