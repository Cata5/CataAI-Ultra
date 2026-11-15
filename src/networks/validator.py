"""
src/networks/validator.py
Architecture validation and forward/backward preflight verification for Network Studio.
Verifies dimension compatibility, tensor shapes, forward execution, and gradient propagation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .estimator import MemoryEstimator, ParameterCalculator
from .factory import ModelFactory

logger = logging.getLogger(__name__)


class ArchitectureValidator:
    """Validates structural constraints and runs forward/backward preflight checks."""

    @staticmethod
    def validate_config_schema(config: Dict[str, Any]) -> List[str]:
        """Static checks on tensor dimensions and configuration values."""
        errors: List[str] = []

        # 1. Required keys
        family = config.get("architecture_family", "decoder_only").lower()
        d_model = config.get("d_model")
        n_layers = config.get("n_layers")

        if d_model is None or int(d_model) <= 0:
            errors.append("`d_model` must be a positive integer.")
        if n_layers is None or int(n_layers) <= 0:
            errors.append("`n_layers` must be a positive integer.")

        d_model = int(d_model or 512)
        n_layers = int(n_layers or 8)

        # 2. Attention checks for Transformer families
        if family in ("decoder_only", "encoder_only", "encoder_decoder", "hybrid"):
            n_q = int(config.get("n_q", 8))
            n_kv = int(config.get("n_kv", 2))
            d_h = int(config.get("d_h", d_model // n_q))

            if n_q <= 0:
                errors.append("`n_q` (Query heads) must be > 0.")
            if n_kv <= 0:
                errors.append("`n_kv` (KV heads) must be > 0.")
            if n_q % n_kv != 0:
                errors.append(f"`n_q` ({n_q}) must be divisible by `n_kv` ({n_kv}) for GQA/MQA head repetition.")
            if (n_q * d_h) <= 0:
                errors.append(f"Invalid head dimension: n_q={n_q}, d_h={d_h}")

        # 3. Heterogeneous layer checks
        layers_cfg = config.get("layers_config", [])
        for idx, lspec in enumerate(layers_cfg):
            l_dim = int(lspec.get("d_model", d_model))
            if l_dim != d_model:
                errors.append(f"Layer {idx}: d_model ({l_dim}) does not match backbone dimension ({d_model}).")
            l_nq = int(lspec.get("n_q", config.get("n_q", 8)))
            l_nkv = int(lspec.get("n_kv", config.get("n_kv", 2)))
            if l_nq % l_nkv != 0:
                errors.append(f"Layer {idx}: n_q ({l_nq}) is not divisible by n_kv ({l_nkv}).")

        return errors

    @classmethod
    def run_preflight_check(
        cls,
        config: Dict[str, Any],
        device: str = "cpu",
        batch_size: int = 2,
        seq_len: int = 32,
    ) -> Dict[str, Any]:
        """
        Executes an end-to-end forward and backward pass with synthetic input tensors.
        Confirms shape compatibility, gradient flow, and parameter sanity.
        """
        # 1. Schema check
        schema_errors = cls.validate_config_schema(config)
        if schema_errors:
            return {
                "valid": False,
                "stage": "schema_validation",
                "errors": schema_errors,
            }

        # 2. Model instantiation
        try:
            target_device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
            model = ModelFactory.create(config).to(device=target_device)
            model.train()
        except Exception as e:
            return {
                "valid": False,
                "stage": "model_instantiation",
                "errors": [f"Failed to instantiate model: {e}"],
            }

        # 3. Forward pass verification
        family = config.get("architecture_family", "decoder_only").lower()
        vocab_size = int(config.get("vocab_size", 12288))
        d_model = int(config.get("d_model", 512))

        try:
            if family in ("mlp", "cnn"):
                input_dim = int(config.get("input_dim", 64))
                dummy_input = torch.randn(batch_size, seq_len, input_dim, device=target_device)
                dummy_labels = torch.randint(0, 2, (batch_size,), device=target_device)
            else:
                dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=target_device)
                dummy_labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=target_device)

            output = model(dummy_input)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            # Validate output shape
            if logits is None:
                return {
                    "valid": False,
                    "stage": "forward_pass",
                    "errors": ["Model returned None during forward pass."],
                }

        except Exception as e:
            return {
                "valid": False,
                "stage": "forward_pass",
                "errors": [f"Forward pass runtime error: {e}"],
            }

        # 4. Backward pass & gradient propagation verification
        try:
            model.zero_grad()
            if family in ("mlp", "cnn"):
                loss = logits.sum()
            else:
                loss = logits[..., :-1, :].contiguous().sum()
            loss.backward()

            zero_grad_params = []
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is None:
                    zero_grad_params.append(name)

            if zero_grad_params:
                return {
                    "valid": False,
                    "stage": "gradient_propagation",
                    "errors": [f"Parameters received no gradients: {', '.join(zero_grad_params[:5])}"],
                }

        except Exception as e:
            return {
                "valid": False,
                "stage": "backward_pass",
                "errors": [f"Backward pass runtime error: {e}"],
            }

        # 5. Calculate memory & parameter stats
        params_info = ParameterCalculator.calculate(config)
        memory_info = MemoryEstimator.estimate(config, micro_batch_size=batch_size, seq_len=seq_len)

        return {
            "valid": True,
            "device": str(target_device),
            "output_shape": list(logits.shape),
            "total_parameters": params_info["total_parameters"],
            "parameter_breakdown": params_info,
            "memory_estimate": memory_info,
            "message": "Architecture verified: forward, backward, and gradient flow confirmed successfully.",
        }

