"""
src/networks/templates.py
Standard architecture templates for Network Studio.
Provides production-ready starting points across multiple neural network families.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Template 1: RoCanon-48.5M Exact Baseline
ROCANON_48M_TEMPLATE: Dict[str, Any] = {
    "id": "rocanon_48m",
    "name": "RoCanon-48.5M Baseline",
    "description": "Deep-Thin Canon Transformer++ with 16 layers, GQA 8:2, tied embeddings, and 4 causal depthwise convolutions.",
    "architecture_family": "decoder_only",
    "vocab_size": 12288,
    "max_seq_len": 2048,
    "d_model": 512,
    "n_layers": 16,
    "n_q": 8,
    "n_kv": 2,
    "d_h": 64,
    "d_ff": 1280,
    "norm_type": "rmsnorm",
    "activation": "swiglu",
    "pos_encoding": "rope",
    "has_conv1d": True,
    "conv_kernel_size": 4,
    "qk_norm": True,
    "pre_norm": True,
    "tie_word_embeddings": True,
    "supported_tasks": ["causal_lm"],
    "version": "1.0",
}

# Template 2: Mini-GPT 15M (Fast iteration)
MINI_GPT_15M_TEMPLATE: Dict[str, Any] = {
    "id": "mini_gpt_15m",
    "name": "Mini-GPT 15M",
    "description": "Lightweight 8-layer causal language model with standard MHA and GELU feed-forward networks.",
    "architecture_family": "decoder_only",
    "vocab_size": 12288,
    "max_seq_len": 1024,
    "d_model": 384,
    "n_layers": 8,
    "n_q": 6,
    "n_kv": 6,
    "d_h": 64,
    "d_ff": 1152,
    "norm_type": "layernorm",
    "activation": "gelu",
    "pos_encoding": "learned",
    "has_conv1d": False,
    "qk_norm": False,
    "pre_norm": True,
    "tie_word_embeddings": True,
    "supported_tasks": ["causal_lm"],
    "version": "1.0",
}

# Template 3: BERT-style Encoder 25M (Masked LM & Classification)
BERT_ENCODER_25M_TEMPLATE: Dict[str, Any] = {
    "id": "bert_encoder_25m",
    "name": "BERT Encoder 25M",
    "description": "Encoder-only bidirectional transformer for masked language modeling and sequence representation.",
    "architecture_family": "encoder_only",
    "vocab_size": 12288,
    "max_seq_len": 512,
    "d_model": 448,
    "n_layers": 10,
    "n_q": 7,
    "n_kv": 7,
    "d_h": 64,
    "d_ff": 1344,
    "norm_type": "layernorm",
    "activation": "gelu",
    "pos_encoding": "learned",
    "has_conv1d": False,
    "qk_norm": False,
    "pre_norm": False,
    "tie_word_embeddings": True,
    "supported_tasks": ["masked_lm", "classification"],
    "output_head": {"type": "classifier", "num_classes": 2},
    "version": "1.0",
}

# Template 4: Hybrid CNN-Transformer 20M
HYBRID_CNN_TRANSFORMER_TEMPLATE: Dict[str, Any] = {
    "id": "hybrid_cnn_transformer_20m",
    "name": "Hybrid CNN-Transformer 20M",
    "description": "Combines local 1D convolutional induction with global GQA multi-head attention.",
    "architecture_family": "hybrid",
    "vocab_size": 12288,
    "max_seq_len": 1024,
    "d_model": 416,
    "n_layers": 10,
    "n_q": 8,
    "n_kv": 2,
    "d_h": 52,
    "d_ff": 1040,
    "norm_type": "rmsnorm",
    "activation": "swiglu",
    "pos_encoding": "rope",
    "has_conv1d": True,
    "conv_kernel_size": 5,
    "qk_norm": True,
    "pre_norm": True,
    "tie_word_embeddings": True,
    "supported_tasks": ["causal_lm"],
    "version": "1.0",
}

# Template 5: Deep MLP Network
DEEP_MLP_TEMPLATE: Dict[str, Any] = {
    "id": "deep_mlp_5m",
    "name": "Deep MLP Network 5M",
    "description": "Multi-layer perceptron for vector classification, tabular representations, or token-free embeddings.",
    "architecture_family": "mlp",
    "input_dim": 256,
    "d_model": 512,
    "n_layers": 6,
    "d_ff": 1024,
    "norm_type": "layernorm",
    "activation": "silu",
    "supported_tasks": ["classification", "regression"],
    "output_head": {"type": "classifier", "num_classes": 10},
    "version": "1.0",
}

# Template 6: 1D CNN Architecture
CONV1D_NETWORK_TEMPLATE: Dict[str, Any] = {
    "id": "conv1d_network_4m",
    "name": "1D Causal CNN Network 4M",
    "description": "Pure stacked causal 1D convolutions for low-latency acoustic or character sequence modeling.",
    "architecture_family": "cnn",
    "input_dim": 128,
    "d_model": 384,
    "n_layers": 8,
    "kernel_size": 5,
    "norm_type": "rmsnorm",
    "activation": "gelu",
    "supported_tasks": ["classification", "regression"],
    "output_head": {"type": "classifier", "num_classes": 2},
    "version": "1.0",
}

ALL_TEMPLATES: List[Dict[str, Any]] = [
    ROCANON_48M_TEMPLATE,
    MINI_GPT_15M_TEMPLATE,
    BERT_ENCODER_25M_TEMPLATE,
    HYBRID_CNN_TRANSFORMER_TEMPLATE,
    DEEP_MLP_TEMPLATE,
    CONV1D_NETWORK_TEMPLATE,
]


def get_template_by_id(template_id: str) -> Optional[Dict[str, Any]]:
    for t in ALL_TEMPLATES:
        if t["id"] == template_id:
            return t.copy()
    return None
