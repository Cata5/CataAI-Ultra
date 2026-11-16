"""
src/training/factory.py
Dynamic factory for RoCanon-48.5M model architecture, optimizers, schedulers,
and analytical parameter calculations for UI control.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ..model.architecture import RoCanonConfig, RoCanonForCausalLM
from .optimizer import configure_optimizers
from .scheduler import WSDScheduler


def calculate_analytical_parameters(
    vocab_size: int = 12288,
    d_model: int = 512,
    n_layers: int = 16,
    n_q: int = 8,
    n_kv: int = 2,
    d_ff: int = 1280,
    kernel_size: int = 4,
    tie_word_embeddings: bool = True,
) -> Dict[str, Any]:
    """
    Computes exact parameter counts and memory estimates according to
    analytical formulas in doc.md Section D.
    """
    d_h = d_model // n_q
    d_kv = n_kv * d_h

    # Embeddings
    emb_params = vocab_size * d_model
    lm_head_params = 0 if tie_word_embeddings else (vocab_size * d_model)

    # Attention & Normalization per layer
    q_params = d_model * d_model
    kv_params = 2 * (d_model * d_kv)
    o_params = d_model * d_model
    qk_norm_params = 2 * d_h
    attn_norm_params = d_model

    # SwiGLU FFN per layer
    ffn_params = 3 * (d_model * d_ff)
    ffn_norm_params = d_model

    # Canon Convolutions per layer
    canon_a_params = d_model * kernel_size
    canon_b_params = (d_model + 2 * d_kv) * kernel_size
    canon_c_params = d_model * kernel_size
    canon_d_params = (2 * d_ff) * kernel_size
    canon_params = canon_a_params + canon_b_params + canon_c_params + canon_d_params

    # Single block total
    params_per_block = (
        q_params + kv_params + o_params + qk_norm_params + attn_norm_params +
        ffn_params + ffn_norm_params + canon_params
    )

    # Final norm
    final_norm_params = d_model

    # Totals
    backbone_params = (n_layers * params_per_block) + final_norm_params
    total_params = emb_params + lm_head_params + backbone_params

    # Muon (2D weights) vs AdamW (1D, Embeddings, Convolutions)
    muon_per_block = q_params + kv_params + o_params + ffn_params
    muon_total = n_layers * muon_per_block

    adamw_total = total_params - muon_total

    # Static memory estimates (BF16 weights + FP32 master + FP32 grads + Muon/AdamW states)
    # Weights BF16 (2B) + FP32 Master (4B) = 6B
    # Gradients FP32 (4B)
    # Muon state (4B) for 2D params, AdamW state (8B) for 1D/emb params
    static_bytes = (
        (total_params * 6) +
        (total_params * 4) +
        (muon_total * 4) +
        (adamw_total * 8)
    )
    static_mib = static_bytes / (1024 * 1024)

    return {
        "total_parameters": total_params,
        "backbone_parameters": backbone_params,
        "embedding_parameters": emb_params,
        "parameters_per_layer": params_per_block,
        "muon_parameters": muon_total,
        "adamw_parameters": adamw_total,
        "muon_percentage": round((muon_total / total_params) * 100, 2),
        "adamw_percentage": round((adamw_total / total_params) * 100, 2),
        "static_memory_mib": round(static_mib, 1),
        "static_memory_gb": round(static_mib / 1024, 2),
        "fits_rtx_3060": (static_mib / 1024) < 4.8,
    }


def create_model_from_dict(cfg: Dict[str, Any]) -> nn.Module:
    """Instantiates a model from a dictionary config, supporting RoCanon and custom architectures."""
    family = cfg.get("architecture_family", "").lower()
    if family and family not in ("rocanon_native",):
        from ..networks.factory import ModelFactory
        return ModelFactory.create(cfg)

    model_cfg = RoCanonConfig(
        vocab_size=int(cfg.get("vocab_size", 12288)),
        d_model=int(cfg.get("d_model", 512)),
        n_layers=int(cfg.get("n_layers", 16)),
        n_q=int(cfg.get("n_q", 8)),
        n_kv=int(cfg.get("n_kv", 2)),
        d_h=int(cfg.get("d_model", 512)) // int(cfg.get("n_q", 8)),
        d_ff=int(cfg.get("d_ff", 1280)),
        kernel_size=int(cfg.get("kernel_size", 4)),
        rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
        rope_theta=float(cfg.get("rope_theta", 10000.0)),
        max_seq_len=int(cfg.get("max_seq_len", 2048)),
        initializer_range=float(cfg.get("initializer_range", 0.02)),
        tie_word_embeddings=bool(cfg.get("tie_word_embeddings", True)),
    )
    return RoCanonForCausalLM(model_cfg)


def create_optimizers_from_dict(
    model: nn.Module,
    opt_cfg: Dict[str, Any],
) -> Tuple[Optimizer, Optimizer]:
    """Configures the hybrid Muon + AdamW optimizer from dictionary config."""
    muon_lr = float(opt_cfg.get("muon_lr", 3.0e-3))
    muon_wd = float(opt_cfg.get("muon_weight_decay", 0.01))
    adamw_lr = float(opt_cfg.get("adamw_lr", 6.0e-4))
    adamw_wd = float(opt_cfg.get("adamw_weight_decay", 0.01))
    beta1 = float(opt_cfg.get("adamw_beta1", 0.9))
    beta2 = float(opt_cfg.get("adamw_beta2", 0.95))
    eps = float(opt_cfg.get("adamw_eps", 1.0e-8))

    return configure_optimizers(
        model=model,
        muon_lr=muon_lr,
        muon_wd=muon_wd,
        adamw_lr=adamw_lr,
        adamw_wd=adamw_wd,
        adamw_betas=(beta1, beta2),
        adamw_eps=eps,
        fused=True,
    )


def list_checkpoints(checkpoint_dir: str = "checkpoints") -> List[Dict[str, Any]]:
    """Returns a list of saved checkpoints with size and modification timestamp across local and project directories."""
    dirs_to_scan = [Path(checkpoint_dir), Path(r"d:\proiecte\LLM from Scratch\checkpoints")]
    files = []
    seen_paths = set()

    for chk_dir in dirs_to_scan:
        if not chk_dir.exists():
            continue
        for f in chk_dir.rglob("*.*"):
            if f.suffix in (".pt", ".bin", ".safetensors"):
                if f.resolve() in seen_paths:
                    continue
                seen_paths.add(f.resolve())
                stat = f.stat()
                files.append({
                    "filename": f.name,
                    "path": str(f.resolve()),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": stat.st_mtime,
                    "is_best": "best" in f.name.lower() or "good" in f.name.lower(),
                })

    # Sort newest first
    files.sort(key=lambda x: x["modified"], reverse=True)
    return files

