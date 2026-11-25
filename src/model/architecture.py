"""
RoCanon-48.5M Model Architecture.

Implements the decoder-only RoCanon-48.5M Transformer architecture as specified in doc.md
(Sections B, C, D, L, N):
- 16 layers, d_model = 512, d_ff = 1,280 with SwiGLU activation
- Grouped-Query Attention (GQA) with 8 Query heads and 2 Key/Value heads (d_h = 64)
- Rotary Position Embeddings (RoPE) and QK-Norm (RMSNorm per head dimension on Q and K)
- Pre-Norm with RMSNorm (eps = 1e-6), zero-bias linear projections, residual scaling (1/sqrt(2L))
- Four residual causal depthwise 1D convolutions (Canon-A, Canon-B, Canon-C, Canon-D) with k = 4
- Tied input/output embeddings (lm_head.weight = embed_tokens.weight)
- Exact analytical parameter count: 48,531,968 (42,240,512 non-embedding + 6,291,456 tied embedding)
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .canon import CausalDepthwiseConv1d


def configure_sdpa_kernels(
    enable_flash: bool = True,
    enable_mem_efficient: bool = True,
    enable_math: bool = True,
) -> dict[str, bool]:
    """
    Explicitly configures PyTorch Scaled Dot-Product Attention (SDPA) backends,
    enabling FlashAttention-2 and Cutlass Memory-Efficient kernels on CUDA
    while ensuring safe math fallback when needed.
    """
    status: dict[str, bool] = {"flash": False, "mem_efficient": False, "math": True}
    if torch.cuda.is_available():
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(enable_flash)
            status["flash"] = bool(torch.backends.cuda.flash_sdp_enabled())
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(enable_mem_efficient)
            status["mem_efficient"] = bool(torch.backends.cuda.mem_efficient_sdp_enabled())
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(enable_math)
            status["math"] = bool(torch.backends.cuda.math_sdp_enabled())
    return status


@dataclass
class RoCanonConfig:
    """
    Configuration hyperparameters for RoCanon-48.5M model architecture.
    """
    vocab_size: int = 12288
    d_model: int = 512
    n_layers: int = 16
    n_q: int = 8
    n_kv: int = 2
    d_h: int = 64
    d_ff: int = 1280
    kernel_size: int = 4
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_seq_len: int = 2048
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    # Compatibility switches. A standard Llama-style model disables Canon-A..D,
    # QK-Norm and residual scaling so its tensors map 1:1 to a llama.cpp GGUF.
    canon_enabled: bool = True
    qk_norm: bool = True
    residual_scale: bool = True
    architecture: str = "rocanon"


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) with learnable scale parameter.

    Args:
        dim: Dimension over which RMS is computed.
        eps: Small constant epsilon for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(variance + self.eps)).type_as(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x) * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) precomputes cosine and sine frequency matrices
    along head dimension d_h.

    Args:
        dim: Dimension of each head (d_h = 64).
        max_seq_len: Initial maximum sequence length for precomputed tables (default: 2048).
        base: Base frequency theta (default: 10000.0).
    """

    def __init__(self, dim: int = 64, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._init_cos_sin_cache(max_seq_len)

    def _init_cos_sin_cache(self, seq_len: int) -> None:
        device = self.inv_freq.device
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def get_cos_sin(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves precomputed cosine and sine tensors up to seq_len.
        Expands the cache if seq_len exceeds current table.
        """
        if seq_len > self.cos_cached.shape[0]:
            self._init_cos_sin_cache(seq_len)
        cos = self.cos_cached[:seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cached[:seq_len].to(device=device, dtype=dtype)
        return cos, sin

    def forward(
        self,
        x: Optional[torch.Tensor],
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device if x is not None else self.inv_freq.device
        dtype = x.dtype if x is not None else torch.float32
        return self.get_cos_sin(seq_len, device=device, dtype=dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies Rotary Position Embedding (RoPE) to tensor x.

    Supports x with shape [B, H, S, D], [B, S, H, D], or [S, D].
    """
    if cos.ndim == 2:
        s_len = cos.shape[0]
        if x.ndim == 4:
            if x.shape[2] == s_len:  # [B, H, S, D]
                cos = cos.unsqueeze(0).unsqueeze(0)
                sin = sin.unsqueeze(0).unsqueeze(0)
            elif x.shape[1] == s_len:  # [B, S, H, D]
                cos = cos.unsqueeze(0).unsqueeze(2)
                sin = sin.unsqueeze(0).unsqueeze(2)
        elif x.ndim == 3:  # [B, S, D]
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


class RoCanonAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) with Canon-B convolution and QK-Norm.

    Specifications:
    - 4:1 GQA ratio (8 Query heads, 2 Key/Value heads, d_h = 64)
    - W_Q (512 -> 512), W_K (512 -> 128), W_V (512 -> 128), W_O (512 -> 512), bias=False
    - Canon-B: CausalDepthwiseConv1d(channels=768, kernel_size=4) on concatenated [Q, K, V]
    - QK-Norm: RMSNorm(64) on Q and K per head dimension prior to dot-product attention
    - RoPE applied to normalized Q and K
    - Scaled dot-product attention with causal mask
    - Residual scaling: Output multiplied by 1/sqrt(2L)
    """

    def __init__(self, config: RoCanonConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_q = config.n_q
        self.n_kv = config.n_kv
        self.d_h = config.d_h
        self.num_key_value_groups = config.n_q // config.n_kv
        self.scale = (1.0 / math.sqrt(2 * config.n_layers)) if config.residual_scale else 1.0

        self.q_proj = nn.Linear(config.d_model, config.n_q * config.d_h, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv * config.d_h, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv * config.d_h, bias=False)
        self.o_proj = nn.Linear(config.n_q * config.d_h, config.d_model, bias=False)
        self.o_proj._is_residual_proj = True

        # Canon-B channels: 512 (Q) + 128 (K) + 128 (V) = 768
        if config.canon_enabled:
            self.canon_b = CausalDepthwiseConv1d(
                channels=config.d_model + 2 * config.n_kv * config.d_h,
                kernel_size=config.kernel_size,
            )

        # QK-Norm learnable scales per head dimension (64 each)
        if config.qk_norm:
            self.q_norm = RMSNorm(config.d_h, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(config.d_h, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b_sz, seq_len, _ = x.shape

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Canon-B: causal depthwise conv on concatenated [Q, K, V]
        if self.config.canon_enabled:
            qkv = self.canon_b(torch.cat([q, k, v], dim=-1))
            q, k, v = torch.split(
                qkv,
                [self.n_q * self.d_h, self.n_kv * self.d_h, self.n_kv * self.d_h],
                dim=-1,
            )

        # Reshape to [B, S, H, d_h] and apply optional QK-Norm per head dimension
        q = q.view(b_sz, seq_len, self.n_q, self.d_h)
        k = k.view(b_sz, seq_len, self.n_kv, self.d_h)
        if self.config.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.view(b_sz, seq_len, self.n_kv, self.d_h).transpose(1, 2)

        # Apply RoPE to normalized query and key representations
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Expand KV heads for GQA (4:1 repeat)
        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Optional diagnostic attention map. Off by default: training never pays
        # for materialising the [B, H, S, S] scores.
        if getattr(self, "capture_attention", False):
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_h)
            causal = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1,
            )
            if cu_seqlens is not None:
                block_mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype)
                for i in range(len(cu_seqlens) - 1):
                    start = cu_seqlens[i].item()
                    end = cu_seqlens[i + 1].item()
                    block_mask[start:end, start:end] = causal[start:end, start:end]
                scores = scores + block_mask
            else:
                scores = scores + causal
            self.last_attention = torch.softmax(scores, dim=-1).detach()

        # Compute scaled dot-product attention
        if cu_seqlens is not None:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype)
            for i in range(len(cu_seqlens) - 1):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                mask[start:end, start:end] = torch.triu(
                    torch.full(
                        (end - start, end - start),
                        float("-inf"),
                        device=x.device,
                        dtype=x.dtype,
                    ),
                    diagonal=1,
                )
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_out = attn_out.transpose(1, 2).contiguous().view(b_sz, seq_len, self.d_model)
        out = self.o_proj(attn_out)
        return out * self.scale


class RoCanonMLP(nn.Module):
    """
    SwiGLU Feed-Forward Network with Canon-D convolution.

    Specifications:
    - W_gate (512 -> 1280), W_up (512 -> 1280), W_down (1280 -> 512), bias=False
    - Canon-D: CausalDepthwiseConv1d(channels=2560, kernel_size=4) on concatenated [Gate, Up]
    - SwiGLU activation: SiLU(gate) * up
    - Residual scaling: Output multiplied by 1/sqrt(2L)
    """

    def __init__(self, config: RoCanonConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.scale = (1.0 / math.sqrt(2 * config.n_layers)) if config.residual_scale else 1.0

        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.down_proj._is_residual_proj = True

        # Canon-D channels: 1280 (Gate) + 1280 (Up) = 2560
        if config.canon_enabled:
            self.canon_d = CausalDepthwiseConv1d(
                channels=2 * config.d_ff,
                kernel_size=config.kernel_size,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project gate and up, concatenate for optional Canon-D convolution
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.config.canon_enabled:
            gu = self.canon_d(torch.cat([gate, up], dim=-1))
            gate, up = torch.split(gu, [self.d_ff, self.d_ff], dim=-1)

        # SwiGLU activation and down projection
        hidden = F.silu(gate) * up
        out = self.down_proj(hidden)
        return out * self.scale


class RoCanonBlock(nn.Module):
    """
    Transformer block consisting of Pre-Norm RMSNorm, Canon-A, Attention,
    FFN Pre-Norm RMSNorm, Canon-C, and SwiGLU MLP.

    Sublayer 1: Attention branch
        norm_attn = attn_norm(x)
        in_attn = canon_a(norm_attn)
        x = x + attn(in_attn, cos, sin)

    Sublayer 2: MLP branch
        norm_ffn = ffn_norm(x)
        in_mlp = canon_c(norm_ffn)
        x = x + mlp(in_mlp)
    """

    def __init__(self, config: RoCanonConfig):
        super().__init__()
        self.config = config
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        if config.canon_enabled:
            self.canon_a = CausalDepthwiseConv1d(config.d_model, config.kernel_size)
        self.attn = RoCanonAttention(config)

        self.ffn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        if config.canon_enabled:
            self.canon_c = CausalDepthwiseConv1d(config.d_model, config.kernel_size)
        self.mlp = RoCanonMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention sublayer
        norm_attn = self.attn_norm(x)
        in_attn = self.canon_a(norm_attn) if self.config.canon_enabled else norm_attn
        attn_out = self.attn(in_attn, cos, sin, cu_seqlens=cu_seqlens)
        x = x + attn_out

        # MLP sublayer
        norm_ffn = self.ffn_norm(x)
        in_mlp = self.canon_c(norm_ffn) if self.config.canon_enabled else norm_ffn
        mlp_out = self.mlp(in_mlp)
        x = x + mlp_out

        return x


class RoCanonForCausalLM(nn.Module):
    """
    RoCanon-48.5M Causal Language Model.

    Full decoder-only Transformer with 16 blocks, tied embeddings, final RMSNorm,
    and causal language modeling head.
    """

    def __init__(self, config: RoCanonConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.rotary = RotaryEmbedding(
            dim=config.d_h,
            max_seq_len=config.max_seq_len,
            base=config.rope_theta,
        )
        self.layers = nn.ModuleList([RoCanonBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Tie word embeddings
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        """
        Weight initialization according to specifications:
        - Linear layers: truncated normal std = 0.02, residual projections (W_O, W_down)
          scaled by 1 / sqrt(2L) (std = 0.02 / sqrt(32) approx 0.0035355).
        - Embeddings: truncated normal std = 0.02.
        - Conv weights: strictly zeros (nn.init.zeros_).
        - RMSNorm weights: strictly ones (torch.ones).
        """
        if isinstance(module, nn.Linear):
            std = self.config.initializer_range
            if self.config.residual_scale and hasattr(module, "_is_residual_proj") and module._is_residual_proj:
                std = std / math.sqrt(2 * self.config.n_layers)
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = self.config.initializer_range
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        elif isinstance(module, nn.Conv1d):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def tie_weights(self) -> None:
        """Ties lm_head.weight directly to embed_tokens.weight."""
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value
        self.tie_weights()

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Returns total number of unique parameters.
        If non_embedding is True, excludes word embedding weights.
        """
        if non_embedding:
            return sum(
                p.numel()
                for n, p in self.named_parameters()
                if "embed_tokens" not in n and "lm_head" not in n
            )
        return sum(p.numel() for p in set(self.parameters()))

    def forward(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of RoCanonForCausalLM.

        Args:
            input_ids: Input token tensor of shape [batch_size, seq_len] or [seq_len].
            cu_seqlens: Optional 1D int32 cumulative lengths tensor [0, l1, l1+l2, ...] for packed sequences.
            labels: Optional labels tensor of shape [batch_size, seq_len]. If provided, returns (loss, logits).

        Returns:
            If labels is None: logits [batch_size, seq_len, vocab_size]
            If labels is provided: (loss, logits)
        """
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        b_sz, seq_len = input_ids.shape

        # Retrieve rotary embeddings
        cos, sin = self.rotary.get_cos_sin(
            seq_len,
            device=input_ids.device,
            dtype=torch.float32,
        )

        # Word embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Transformer backbone layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cos=cos,
                sin=sin,
                cu_seqlens=cu_seqlens,
            )

        # Final RMSNorm
        hidden_states = self.norm(hidden_states)

        # LM Head logits
        logits = self.lm_head(hidden_states)

        if labels is not None:
            if labels.ndim == 1:
                labels = labels.unsqueeze(0)
            if seq_len > 1:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            else:
                loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    labels.view(-1),
                    ignore_index=-100,
                )
            return loss, logits

        return logits
