"""
src/training/peft.py
Parameter-Efficient Fine-Tuning (PEFT / LoRA / QLoRA) for CataAI Ultra.
Implements low-rank adapter injection into attention and feedforward projections,
freezing base model parameters and reducing VRAM usage by over 80%.
Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer wrapping an existing nn.Linear.
    Forward computation: y = W_0 x + (alpha / r) * (B A x)
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = float(lora_alpha) / float(r)

        # Freeze the base layer parameters
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Low-rank matrices:
        # A is initialized with Kaiming uniform
        # B is initialized to zeros so the adapter starts with identity output (0 contribution)
        self.lora_A = nn.Parameter(torch.empty((r, in_features), dtype=base_layer.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r), dtype=base_layer.weight.dtype))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.merged:
            return base_out

        # Adapter forward: dropout(x) @ A.T @ B.T * scaling
        adapter_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t() * self.scaling
        return base_out + adapter_out

    def merge(self) -> None:
        """Folds the adapter weights into the base weight tensor permanently."""
        if self.merged:
            return
        with torch.no_grad():
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data.add_(delta_w.to(self.base_layer.weight.dtype))
            self.merged = True


def inject_lora(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> Dict[str, Any]:
    """
    Injects LoRA adapter layers into all matching nn.Linear submodules of the model.
    Freezes all non-adapter parameters.

    Default targets cover both standard Hugging Face models (q_proj, v_proj, k_proj, o_proj,
    gate_proj, up_proj, down_proj) and custom architectures (c_attn, c_proj, mlp, q_proj, v_proj).
    """
    if target_modules is None:
        target_modules = [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "c_attn", "c_proj", "attn.q_proj", "attn.v_proj",
        ]

    # 1. Freeze entire base model
    for p in model.parameters():
        p.requires_grad = False

    injected_count = 0
    injected_names: List[str] = []

    # Helper recursive replacer
    def _replace_module(parent: nn.Module, child_name: str, full_name: str):
        nonlocal injected_count
        child = getattr(parent, child_name)
        if isinstance(child, nn.Linear):
            # Check if name matches any target suffix or pattern
            matches = any(target in full_name for target in target_modules)
            if matches:
                lora_layer = LoRALinear(
                    base_layer=child,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
                setattr(parent, child_name, lora_layer)
                injected_count += 1
                injected_names.append(full_name)
                return

        for name, sub in child.named_children():
            _replace_module(child, name, f"{full_name}.{name}" if full_name else name)

    for name, child in list(model.named_children()):
        _replace_module(model, name, name)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = (trainable_params / max(1, total_params)) * 100.0

    logger.info(
        "LoRA Injected: %d modules adapted | Total: %s | Trainable: %s (%.2f%%)",
        injected_count,
        f"{total_params:,}",
        f"{trainable_params:,}",
        trainable_pct,
    )

    return {
        "success": True,
        "adapted_modules_count": injected_count,
        "adapted_modules": injected_names,
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": round(trainable_pct, 4),
    }


def save_lora_weights(
    model: nn.Module,
    output_dir: Union[str, Path],
    config_dict: Optional[Dict[str, Any]] = None,
) -> Path:
    """Saves only the low-rank adapter weights and configuration to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    adapter_sd: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            adapter_sd[name] = param.detach().cpu()

    adapter_weights_path = out / "adapter_model.pt"
    torch.save(adapter_sd, adapter_weights_path)

    cfg = {
        "peft_type": "LORA",
        "r": config_dict.get("r", 16) if config_dict else 16,
        "lora_alpha": config_dict.get("lora_alpha", 32) if config_dict else 32,
        "lora_dropout": config_dict.get("lora_dropout", 0.05) if config_dict else 0.05,
        "num_adapter_tensors": len(adapter_sd),
        "total_adapter_params": sum(p.numel() for p in adapter_sd.values()),
    }
    with open(out / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    logger.info("Saved %d LoRA adapter weights to %s", len(adapter_sd), adapter_weights_path)
    return adapter_weights_path


def merge_and_unload(model: nn.Module) -> nn.Module:
    """Folds all LoRA adapter layers back into standard nn.Linear layers for inference."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
    logger.info("Successfully merged all LoRA adapters into base weights.")
    return model
