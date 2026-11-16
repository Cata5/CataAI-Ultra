"""
src/networks/factory.py
Model Factory and Generic Modular Network for Network Studio.
Dynamically instantiates neural networks across multiple families:
Decoder-only Transformers, Encoder-only, MLPs, CNNs, and Hybrids.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import (
    CausalDepthwiseConv1d,
    GenericTransformerBlock,
    LearnedPositionalEmbedding,
    RMSNorm,
    SinusoidalPositionalEncoding,
    get_activation_fn,
    get_norm_layer,
)

logger = logging.getLogger(__name__)


class CnnBlock(nn.Module):
    """1D Convolutional Block maintaining [Batch, Seq, Channels] dimension."""

    def __init__(self, dim: int, kernel_size: int = 3, act: str = "gelu", norm_type: str = "rmsnorm") -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.act = get_activation_fn(act)
        self.norm = get_norm_layer(norm_type, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, S, D] -> transpose to [B, D, S] for Conv1d then back to [B, S, D]
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(self.act(h))


class GenericModularNetwork(nn.Module):
    """
    General-purpose neural network implementation instantiated from modular architecture JSON.
    Supports Causal LM, Masked LM, Classification, and Sequence Regression tasks.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.family = config.get("architecture_family", "decoder_only").lower()
        self.supported_tasks = config.get("supported_tasks", ["causal_lm"])

        self.vocab_size = int(config.get("vocab_size", 12288))
        self.d_model = int(config.get("d_model", 512))
        self.n_layers = int(config.get("n_layers", 8))
        self.max_seq_len = int(config.get("max_seq_len", 1024))
        self.tie_word_embeddings = bool(config.get("tie_word_embeddings", True))
        self.pos_encoding = config.get("pos_encoding", "rope").lower()
        self.norm_type = config.get("norm_type", "rmsnorm")

        # 1. Input Stage
        if self.family in ("decoder_only", "encoder_only", "encoder_decoder", "hybrid"):
            self.embed_tokens = nn.Embedding(self.vocab_size, self.d_model)
            if self.pos_encoding == "learned":
                self.pos_emb = LearnedPositionalEmbedding(self.max_seq_len, self.d_model)
            elif self.pos_encoding == "sinusoidal":
                self.pos_emb = SinusoidalPositionalEncoding(self.d_model, self.max_seq_len)
            else:
                self.pos_emb = None
        elif self.family in ("mlp", "cnn"):
            in_dim = int(config.get("input_dim", 64))
            self.input_proj = nn.Linear(in_dim, self.d_model)
            self.embed_tokens = None
            self.pos_emb = None
        else:
            self.embed_tokens = nn.Embedding(self.vocab_size, self.d_model)
            self.pos_emb = None

        # 2. Backbone Blocks (Heterogeneous or Uniform)
        layers_cfg = config.get("layers_config", [])
        self.blocks = nn.ModuleList()

        residual_scale = float(config.get("residual_scale", 1.0))
        if config.get("scale_residual_with_layers", False):
            residual_scale = 1.0 / math.sqrt(2.0 * self.n_layers)

        for l_idx in range(self.n_layers):
            # Check for per-layer override (heterogeneous layer support)
            layer_spec = layers_cfg[l_idx] if l_idx < len(layers_cfg) else {}

            b_dim = int(layer_spec.get("d_model", self.d_model))
            b_nq = int(layer_spec.get("n_q", config.get("n_q", 8)))
            b_nkv = int(layer_spec.get("n_kv", config.get("n_kv", 2)))
            b_dff = int(layer_spec.get("d_ff", config.get("d_ff", 1280)))
            b_act = layer_spec.get("activation", config.get("activation", "swiglu"))
            b_has_conv = bool(layer_spec.get("has_conv1d", config.get("has_conv1d", False)))
            b_qk_norm = bool(layer_spec.get("qk_norm", config.get("qk_norm", False)))
            b_use_rope = (self.pos_encoding == "rope")

            custom_mod_name = layer_spec.get("custom_module")
            if custom_mod_name:
                from .custom_loader import CustomModuleRegistry
                custom_cls = CustomModuleRegistry().get_module(custom_mod_name)
                if custom_cls is not None:
                    custom_params = layer_spec.get("custom_params", {})
                    block = custom_cls(dim=b_dim, **custom_params)
                else:
                    logger.warning("Custom module '%s' requested but not registered.", custom_mod_name)
                    block = CnnBlock(dim=b_dim, kernel_size=3, act=b_act, norm_type=self.norm_type)
            elif self.family == "mlp":
                # MLP stack
                block = nn.Sequential(
                    nn.Linear(b_dim, b_dff),
                    get_activation_fn(b_act),
                    nn.Linear(b_dff, b_dim),
                    get_norm_layer(self.norm_type, b_dim),
                )
            elif self.family == "cnn":
                k_size = int(layer_spec.get("kernel_size", 3))
                block = CnnBlock(dim=b_dim, kernel_size=k_size, act=b_act, norm_type=self.norm_type)
            else:
                # Transformer Block
                block = GenericTransformerBlock(
                    d_model=b_dim,
                    n_q=b_nq,
                    n_kv=b_nkv,
                    d_ff=b_dff,
                    norm_type=self.norm_type,
                    activation=b_act,
                    pre_norm=bool(config.get("pre_norm", True)),
                    residual_scale=residual_scale,
                    has_conv1d=b_has_conv,
                    qk_norm=b_qk_norm,
                    use_rope=b_use_rope,
                    max_seq_len=self.max_seq_len,
                )

            self.blocks.append(block)

        # 3. Final Normalization
        self.final_norm = get_norm_layer(self.norm_type, self.d_model)

        # 4. Output Head
        head_spec = config.get("output_head", {"type": "lm_head"})
        head_type = head_spec.get("type", "lm_head").lower()

        if head_type == "lm_head":
            self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
            if self.tie_word_embeddings and self.embed_tokens is not None:
                self.lm_head.weight = self.embed_tokens.weight
        elif head_type == "classifier":
            num_classes = int(head_spec.get("num_classes", 2))
            self.classifier = nn.Linear(self.d_model, num_classes)
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Returns total parameter count, optionally omitting token embeddings."""
        if non_embedding and self.embed_tokens is not None:
            emb_params = sum(p.numel() for p in self.embed_tokens.parameters())
            total = sum(p.numel() for p in self.parameters())
            return total - emb_params
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids_or_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = input_ids_or_features

        # Token embedding / input projection
        if self.embed_tokens is not None and x.dtype in (torch.int64, torch.int32):
            h = self.embed_tokens(x)
            if self.pos_emb is not None:
                h = self.pos_emb(h)
        elif hasattr(self, "input_proj") and self.input_proj is not None:
            h = self.input_proj(x)
        else:
            h = x

        # Backbone pass
        for block in self.blocks:
            h = block(h)

        h = self.final_norm(h)

        # Output projection
        if hasattr(self, "classifier") and self.classifier is not None:
            # Classification pooling (mean over sequence or first token)
            pooled = h.mean(dim=1) if h.dim() == 3 else h
            logits = self.classifier(pooled)
            if labels is not None:
                loss = F.cross_entropy(logits, labels)
                return logits, loss
            return logits

        logits = self.lm_head(h)
        if labels is not None:
            # Standard shifted cross-entropy for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            return logits, loss

        return logits


class ModelFactory:
    """Instantiates and registers models for Network Studio."""

    @staticmethod
    def create(config: Dict[str, Any]) -> nn.Module:
        """Resolves configuration and builds instantiated PyTorch model."""
        # Baseline RoCanon-48.5M architectural invariance
        if config.get("id") == "rocanon_48m":
            from ..model.architecture import RoCanonConfig, RoCanonForCausalLM
            m_cfg = RoCanonConfig(
                vocab_size=int(config.get("vocab_size", 12288)),
                d_model=int(config.get("d_model", 512)),
                n_layers=int(config.get("n_layers", 16)),
                n_q=int(config.get("n_q", 8)),
                n_kv=int(config.get("n_kv", 2)),
                d_h=int(config.get("d_h", 64)),
                d_ff=int(config.get("d_ff", 1280)),
                kernel_size=int(config.get("conv_kernel_size", 4)),
                max_seq_len=int(config.get("max_seq_len", 2048)),
                tie_word_embeddings=bool(config.get("tie_word_embeddings", True)),
            )
            return RoCanonForCausalLM(m_cfg)

        # Sanity check required dimensions
        if "d_model" not in config:
            config["d_model"] = 512
        if "n_layers" not in config:
            config["n_layers"] = 8

        model = GenericModularNetwork(config)
        logger.info(
            "ModelFactory created %s model with %d parameters (%s)",
            config.get("name", "CustomNetwork"),
            model.get_num_params(),
            config.get("architecture_family", "decoder_only"),
        )
        return model
