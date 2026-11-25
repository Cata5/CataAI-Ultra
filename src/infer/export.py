"""
src/infer/export.py
Exports RoCanon-48.5M weights to safetensors format and Hugging Face layout.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import save_file

from ..model.architecture import RoCanonConfig, RoCanonForCausalLM


def export_to_safetensors(
    model: RoCanonForCausalLM,
    output_dir: str | Path,
    tokenizer_path: Optional[str | Path] = None,
) -> Path:
    """
    Exports model weights to safetensors format along with config.json and tokenizer files.

    Args:
        model: Trained RoCanonForCausalLM model instance.
        output_dir: Destination directory.
        tokenizer_path: Optional path to tokenizer files to copy over.

    Returns:
        Path to output directory.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Save config
    config_dict = {
        "architectures": ["RoCanonForCausalLM"],
        "model_type": "rocanon",
        "vocab_size": model.config.vocab_size,
        "d_model": model.config.d_model,
        "n_layers": model.config.n_layers,
        "n_q": model.config.n_q,
        "n_kv": model.config.n_kv,
        "d_h": model.config.d_h,
        "d_ff": model.config.d_ff,
        "kernel_size": model.config.kernel_size,
        "rms_norm_eps": model.config.rms_norm_eps,
        "rope_theta": model.config.rope_theta,
        "max_seq_len": model.config.max_seq_len,
        "tie_word_embeddings": model.config.tie_word_embeddings,
        "canon_enabled": getattr(model.config, "canon_enabled", True),
        "qk_norm": getattr(model.config, "qk_norm", True),
        "residual_scale": getattr(model.config, "residual_scale", True),
        "architecture": getattr(model.config, "architecture", "rocanon"),
        "torch_dtype": "bfloat16",
    }
    with open(out_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # 2. Extract state dict in bfloat16
    state_dict = {}
    for name, param in model.named_parameters():
        state_dict[name] = param.detach().cpu().to(torch.bfloat16)

    # 3. Save safetensors
    weights_path = out_path / "model.safetensors"
    save_file(state_dict, str(weights_path))

    # 4. Copy tokenizer assets when provided, so the export directory is usable
    # without a separate lookup of the training-time tokenizer.
    if tokenizer_path is not None:
        source = Path(tokenizer_path)
        if source.is_dir():
            for candidate in source.iterdir():
                if candidate.is_file() and candidate.name.startswith(("tokenizer", "vocab", "merges", "special_tokens")):
                    shutil.copy2(candidate, out_path / candidate.name)
        elif source.is_file():
            shutil.copy2(source, out_path / source.name)

    return out_path

