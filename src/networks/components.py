"""
src/networks/components.py
Reusable neural network component library for Network Studio.
Provides PyTorch modules for embeddings, attention mechanisms, feed-forward blocks,
normalizations, activations, convolutions, and residual structures.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- Activations ----------------

def get_activation_fn(name: str) -> nn.Module:
    n = name.lower()
    if n in ("relu",):
        return nn.ReLU()
    elif n in ("gelu",):
        return nn.GELU()
    elif n in ("silu", "swish"):
        return nn.SiLU()
    elif n in ("tanh",):
        return nn.Tanh()
    elif n in ("sigmoid",):
        return nn.Sigmoid()
    elif n in ("leaky_relu", "leakyrelu"):
        return nn.LeakyReLU(0.1)
    else:
        return nn.GELU()


# ---------------- Normalization ----------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(-1, keepdim=True)
        normed = x.float() * torch.rsqrt(var + self.eps)
        return (normed * self.weight).type_as(x)


class SequenceBatchNorm1d(nn.BatchNorm1d):
    """
    1D Batch Normalization compatible with sequence tensors of shape [B, S, D].
    Transposes to [B, D, S] before applying BatchNorm1d, then transposes back to [B, S, D].
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return super().forward(x.transpose(1, 2)).transpose(1, 2)
        return super().forward(x)


def get_norm_layer(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
    nt = norm_type.lower()
    if nt == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    elif nt == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    elif nt == "batchnorm":
        return SequenceBatchNorm1d(dim, eps=eps)
    elif nt == "groupnorm":
        num_groups = min(32, max(1, dim // 16))
        return nn.GroupNorm(num_groups=num_groups, num_channels=dim)
    elif nt in ("none", "identity", ""):
        return nn.Identity()
    else:
        return nn.LayerNorm(dim, eps=eps)


# ---------------- Embeddings & Positional Encoding ----------------

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_seq_len: int = 2048) -> None:
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, S, D]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embedding table."""

    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        return x + self.embedding(pos)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int = 64, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(0).unsqueeze(1), emb.sin().unsqueeze(0).unsqueeze(1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k: [B, H, S, D]
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos = cos[..., : q.shape[-2], : q.shape[-1]]
    sin = sin[..., : q.shape[-2], : q.shape[-1]]
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out.type_as(q), k_out.type_as(k)


# ---------------- Attention Mechanisms ----------------

class GroupedQueryAttention(nn.Module):
    """
    Flexible Attention Module supporting:
    - Standard Multi-Head Attention (n_q == n_kv)
    - Grouped-Query Attention (n_q > n_kv > 1)
    - Multi-Query Attention (n_kv == 1)
    - Optional QK-Norm per head dimension
    - Optional RoPE
    """

    def __init__(
        self,
        d_model: int,
        n_q: int = 8,
        n_kv: int = 2,
        d_h: Optional[int] = None,
        causal: bool = True,
        dropout: float = 0.0,
        qk_norm: bool = False,
        bias: bool = False,
        use_rope: bool = False,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_q = n_q
        self.n_kv = n_kv
        self.d_h = d_h if d_h is not None else (d_model // n_q)
        self.causal = causal
        self.dropout_p = dropout
        self.qk_norm_enabled = qk_norm
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_model, n_q * self.d_h, bias=bias)
        self.k_proj = nn.Linear(d_model, n_kv * self.d_h, bias=bias)
        self.v_proj = nn.Linear(d_model, n_kv * self.d_h, bias=bias)
        self.out_proj = nn.Linear(n_q * self.d_h, d_model, bias=bias)

        if qk_norm:
            self.q_norm = RMSNorm(self.d_h)
            self.k_norm = RMSNorm(self.d_h)

        if use_rope:
            self.rotary = RotaryEmbedding(self.d_h, max_seq_len=max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        kv_input = context if context is not None else x
        s_kv = kv_input.shape[1]

        q = self.q_proj(x).view(b, s, self.n_q, self.d_h).transpose(1, 2)
        k = self.k_proj(kv_input).view(b, s_kv, self.n_kv, self.d_h).transpose(1, 2)
        v = self.v_proj(kv_input).view(b, s_kv, self.n_kv, self.d_h).transpose(1, 2)

        if self.qk_norm_enabled:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope and context is None:
            cos, sin = self.rotary(x, s)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand KV if n_q != n_kv (GQA / MQA repeat KV heads)
        if self.n_q != self.n_kv:
            repeat_factor = self.n_q // self.n_kv
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # PyTorch SDPA (scaled dot product attention)
        is_causal = self.causal if context is None else False
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(b, s, self.n_q * self.d_h)
        return self.out_proj(attn_out)


# ---------------- Feed-Forward Networks ----------------

class StandardMLP(nn.Module):
    """Standard 2-layer MLP with configurable activation."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "gelu",
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
        self.act = get_activation_fn(activation)
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class SwiGLUFFN(nn.Module):
    """Gated SwiGLU Feed-Forward Network."""

    def __init__(self, d_model: int, d_ff: int, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GeGLUFFN(nn.Module):
    """Gated GELU Feed-Forward Network."""

    def __init__(self, d_model: int, d_ff: int, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))


# ---------------- Convolutions & Other ----------------

class CausalDepthwiseConv1d(nn.Module):
    """1D Causal Depthwise Convolution for local sequence mixing."""

    def __init__(self, dim: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            groups=dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D] -> transpose to [B, D, S]
        xt = x.transpose(1, 2)
        padded = F.pad(xt, (self.pad, 0))
        out = self.conv(padded)
        return out.transpose(1, 2)


# ---------------- Modular Transformer Block ----------------

class GenericTransformerBlock(nn.Module):
    """
    Highly configurable Transformer Block supporting:
    - Pre-Norm or Post-Norm
    - RMSNorm / LayerNorm
    - GQA / MHA / MQA Attention
    - SwiGLU / Standard MLP
    - Optional depthwise causal convolution (Canon-style)
    - Residual scaling (e.g. 1 / sqrt(2L))
    """

    def __init__(
        self,
        d_model: int,
        n_q: int = 8,
        n_kv: int = 2,
        d_h: Optional[int] = None,
        d_ff: int = 1280,
        attention_type: str = "gqa",
        norm_type: str = "rmsnorm",
        activation: str = "swiglu",
        dropout: float = 0.0,
        pre_norm: bool = True,
        residual_scale: float = 1.0,
        has_conv1d: bool = False,
        conv_kernel_size: int = 4,
        qk_norm: bool = False,
        use_rope: bool = False,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.pre_norm = pre_norm
        self.residual_scale = residual_scale
        self.has_conv1d = has_conv1d

        self.norm1 = get_norm_layer(norm_type, d_model)
        self.norm2 = get_norm_layer(norm_type, d_model)

        # Attention
        self.attention = GroupedQueryAttention(
            d_model=d_model,
            n_q=n_q,
            n_kv=n_kv,
            d_h=d_h,
            causal=True,
            dropout=dropout,
            qk_norm=qk_norm,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
        )

        # Optional Convolutions
        if has_conv1d:
            self.conv = CausalDepthwiseConv1d(d_model, kernel_size=conv_kernel_size)

        # FFN
        act_lower = activation.lower()
        if act_lower == "swiglu":
            self.ffn = SwiGLUFFN(d_model, d_ff)
        elif act_lower == "geglu":
            self.ffn = GeGLUFFN(d_model, d_ff)
        else:
            self.ffn = StandardMLP(d_model, d_ff, activation=activation, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.pre_norm:
            # Pre-Norm attention
            h = self.norm1(x)
            if self.has_conv1d:
                h = h + self.conv(h)
            x = x + (self.attention(h, mask=mask) * self.residual_scale)

            # Pre-Norm FFN
            x = x + (self.ffn(self.norm2(x)) * self.residual_scale)
        else:
            # Post-Norm
            attn_in = x
            if self.has_conv1d:
                attn_in = attn_in + self.conv(attn_in)
            x = self.norm1(x + (self.attention(attn_in, mask=mask) * self.residual_scale))
            x = self.norm2(x + (self.ffn(x) * self.residual_scale))

        return x

