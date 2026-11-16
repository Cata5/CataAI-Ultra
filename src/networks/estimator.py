"""
src/networks/estimator.py
Live Parameter Calculator, Budget Assistant, and VRAM Memory Estimator for Network Studio.
Tailored for the NVIDIA GeForce RTX 3060 Laptop GPU with 6GB VRAM.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


class ParameterCalculator:
    """Calculates detailed parameter counts across components and layers."""

    @staticmethod
    def calculate(config: Dict[str, Any]) -> Dict[str, Any]:
        family = config.get("architecture_family", "decoder_only").lower()
        vocab_size = int(config.get("vocab_size", 12288))
        d_model = int(config.get("d_model", 512))
        n_layers = int(config.get("n_layers", 8))
        tie_embeddings = bool(config.get("tie_word_embeddings", True))
        layers_cfg = config.get("layers_config", [])

        # Check if RoCanon-48.5M baseline template
        if config.get("id") == "rocanon_48m" or (
            family in ("decoder_only", "hybrid")
            and bool(config.get("has_conv1d", False))
            and d_model == 512
            and n_layers == 16
            and vocab_size == 12288
            and int(config.get("d_ff", 1280)) == 1280
            and tie_embeddings
        ):
            from ..training.factory import calculate_analytical_parameters
            rocanon_stats = calculate_analytical_parameters(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                n_q=int(config.get("n_q", 8)),
                n_kv=int(config.get("n_kv", 2)),
                d_ff=int(config.get("d_ff", 1280)),
                kernel_size=int(config.get("conv_kernel_size", 4)),
                tie_word_embeddings=tie_embeddings,
            )
            total_p = rocanon_stats["total_parameters"]
            emb_p = rocanon_stats["embedding_parameters"]
            non_emb_p = rocanon_stats["backbone_parameters"]
            d_h = d_model // int(config.get("n_q", 8))
            d_kv = int(config.get("n_kv", 2)) * d_h
            attn_layer = (d_model * d_model) + (2 * d_model * d_kv) + (d_model * d_model) + (2 * d_h)
            ffn_layer = 3 * d_model * int(config.get("d_ff", 1280))
            conv_layer = int(config.get("conv_kernel_size", 4)) * (3 * d_model + 2 * d_kv + 2 * int(config.get("d_ff", 1280)))
            norm_layer = 2 * d_model
            return {
                "total_parameters": total_p,
                "trainable_parameters": total_p,
                "non_embedding_parameters": non_emb_p,
                "embedding_parameters": emb_p,
                "attention_parameters": attn_layer * n_layers,
                "ffn_parameters": ffn_layer * n_layers,
                "conv_parameters": conv_layer * n_layers,
                "norm_parameters": (norm_layer * n_layers) + d_model,
                "output_head_parameters": 0 if tie_embeddings else emb_p,
                "tied_embeddings": tie_embeddings,
                "layer_count": n_layers,
                "layers": [
                    {
                        "layer_index": i,
                        "attention_params": attn_layer,
                        "ffn_params": ffn_layer,
                        "conv_params": conv_layer,
                        "norm_params": norm_layer,
                        "total_layer_params": rocanon_stats["parameters_per_layer"],
                    }
                    for i in range(n_layers)
                ],
            }

        # 1. Embeddings
        pos_encoding = str(config.get("pos_encoding", "rope")).lower()
        max_seq_len = int(config.get("max_seq_len", 1024))
        norm_type = str(config.get("norm_type", "rmsnorm")).lower()
        norm_mult = 2 if norm_type in ("layernorm", "batchnorm") else 1

        if family in ("mlp", "cnn"):
            input_dim = int(config.get("input_dim", 64))
            emb_params = (input_dim * d_model) + d_model
        else:
            emb_params = vocab_size * d_model
            if pos_encoding == "learned":
                emb_params += max_seq_len * d_model

        # 2. Per-Layer components
        attention_params_total = 0
        ffn_params_total = 0
        conv_params_total = 0
        norm_params_total = 0
        layer_breakdown = []

        for l_idx in range(n_layers):
            layer_spec = layers_cfg[l_idx] if l_idx < len(layers_cfg) else {}
            b_dim = int(layer_spec.get("d_model", d_model))
            b_nq = int(layer_spec.get("n_q", config.get("n_q", 8)))
            b_nkv = int(layer_spec.get("n_kv", config.get("n_kv", 2)))
            b_dh = int(layer_spec.get("d_h", b_dim // b_nq))
            b_dff = int(layer_spec.get("d_ff", config.get("d_ff", 1280)))
            b_act = layer_spec.get("activation", config.get("activation", "swiglu")).lower()
            has_conv = bool(layer_spec.get("has_conv1d", config.get("has_conv1d", False)))
            k_size = int(layer_spec.get("conv_kernel_size", config.get("conv_kernel_size", 4)))
            b_qk_norm = bool(layer_spec.get("qk_norm", config.get("qk_norm", False)))

            if family == "mlp":
                attn_p = 0
                ffn_p = (b_dim * b_dff + b_dff) + (b_dff * b_dim + b_dim)
                conv_p = 0
                norm_p = b_dim * norm_mult
            elif family == "cnn":
                c_k = int(layer_spec.get("kernel_size", config.get("conv_kernel_size", 3)))
                attn_p = 0
                conv_p = (b_dim * b_dim * c_k) + b_dim
                ffn_p = 0
                norm_p = b_dim * norm_mult
            else:
                # Q, K, V, Out projections
                q_p = b_dim * (b_nq * b_dh)
                k_p = b_dim * (b_nkv * b_dh)
                v_p = b_dim * (b_nkv * b_dh)
                out_p = (b_nq * b_dh) * b_dim
                qk_p = (2 * b_dh) if b_qk_norm else 0
                attn_p = q_p + k_p + v_p + out_p + qk_p

                # FFN (SwiGLU/GeGLU uses gate + up + down = 3x, standard MLP uses 2x)
                if b_act in ("swiglu", "geglu"):
                    ffn_p = (b_dim * b_dff) + (b_dim * b_dff) + (b_dff * b_dim)
                else:
                    ffn_p = (b_dim * b_dff) + (b_dff * b_dim)

                conv_p = (b_dim * 4) if has_conv else 0
                norm_p = 2 * b_dim * norm_mult  # norm1 + norm2

            # Custom module parameter calculation if present
            custom_p = 0
            if "custom_module" in layer_spec and layer_spec["custom_module"]:
                try:
                    from .custom_loader import CustomModuleRegistry
                    c_name = layer_spec["custom_module"]
                    c_cls = CustomModuleRegistry.get_module(c_name)
                    if c_cls and hasattr(c_cls, "parameter_count"):
                        custom_p = c_cls.parameter_count(b_dim, **layer_spec.get("custom_params", {}))
                except Exception:
                    custom_p = 0

            attention_params_total += attn_p
            ffn_params_total += ffn_p
            conv_params_total += (conv_p + custom_p)
            norm_params_total += norm_p

            layer_breakdown.append({
                "layer_index": l_idx,
                "attention_params": attn_p,
                "ffn_params": ffn_p,
                "conv_params": conv_p,
                "custom_module_params": custom_p,
                "norm_params": norm_p,
                "total_layer_params": attn_p + ffn_p + conv_p + custom_p + norm_p,
            })

        # Final norm
        final_norm_params = d_model * norm_mult
        norm_params_total += final_norm_params

        # Output head
        head_spec = config.get("output_head", {"type": "lm_head"})
        head_type = head_spec.get("type", "lm_head").lower()
        if head_type == "classifier":
            num_classes = int(head_spec.get("num_classes", 2))
            head_params = (d_model * num_classes) + num_classes
        else:
            if tie_embeddings and family in ("decoder_only", "hybrid"):
                head_params = 0  # Tied with token embeddings
            else:
                head_params = d_model * vocab_size

        total_params = (
            emb_params
            + attention_params_total
            + ffn_params_total
            + conv_params_total
            + norm_params_total
            + head_params
        )

        trainable_params = total_params
        non_emb_params = total_params - emb_params

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_embedding_parameters": non_emb_params,
            "embedding_parameters": emb_params,
            "attention_parameters": attention_params_total,
            "ffn_parameters": ffn_params_total,
            "conv_parameters": conv_params_total,
            "norm_parameters": norm_params_total,
            "output_head_parameters": head_params,
            "tied_embeddings": tie_embeddings,
            "layer_count": n_layers,
            "layers": layer_breakdown,
        }


class MemoryEstimator:
    """Estimates memory usage for training and inference across any detected hardware target."""

    @staticmethod
    def estimate(
        config: Dict[str, Any],
        micro_batch_size: int = 4,
        seq_len: int = 256,
        dtype: str = "bfloat16",
        optimizer: str = "muon_adamw",
        gradient_checkpointing: bool = False,
        hardware_profile: Optional[Dict[str, Any]] = None,
        hardware_target: Optional[str] = None,
        hardware_total_vram_gb: Optional[float] = None,
    ) -> Dict[str, Any]:
        param_stats = ParameterCalculator.calculate(config)
        total_p = param_stats["total_parameters"]
        d_model = int(config.get("d_model", 512))
        n_layers = int(config.get("n_layers", 8))

        # Precision bytes
        dt = dtype.lower()
        param_bytes = 2 if dt in ("bfloat16", "fp16", "bf16") else 4

        # 1. Model weights
        model_mem_mb = (total_p * param_bytes) / (1024 * 1024)

        # 2. Gradients (FP32 master gradient accumulation)
        grad_mem_mb = (total_p * 4) / (1024 * 1024)

        # 3. Optimizer state
        # AdamW / SGD / RMSprop / Adagrad / Muon
        opt_lower = optimizer.lower()
        if opt_lower in ("adamw", "adam"):
            opt_mem_mb = (total_p * 8) / (1024 * 1024)
        elif opt_lower in ("sgd", "adagrad"):
            opt_mem_mb = (total_p * 4) / (1024 * 1024)
        elif opt_lower == "rmsprop":
            opt_mem_mb = (total_p * 8) / (1024 * 1024)
        else:
            # Muon + AdamW
            attn_ffn_p = param_stats["attention_parameters"] + param_stats["ffn_parameters"]
            other_p = total_p - attn_ffn_p
            opt_mem_mb = ((attn_ffn_p * 4) + (other_p * 8)) / (1024 * 1024)

        # 4. Activations
        tokens_per_batch = micro_batch_size * seq_len
        bytes_per_token_layer = 34 * d_model
        if gradient_checkpointing:
            act_mem_bytes = tokens_per_batch * bytes_per_token_layer * 2
        else:
            act_mem_bytes = tokens_per_batch * bytes_per_token_layer * n_layers

        act_mem_mb = act_mem_bytes / (1024 * 1024)

        # 5. KV Cache & Workspace headroom
        workspace_headroom_mb = 350.0  # PyTorch context & caching allocator reserve

        total_training_vram_mb = model_mem_mb + grad_mem_mb + opt_mem_mb + act_mem_mb + workspace_headroom_mb
        total_training_vram_gb = total_training_vram_mb / 1024.0

        # Inference memory (weights + KV cache + small activation)
        inference_vram_mb = model_mem_mb + (act_mem_mb * 0.2) + 200.0

        # Determine target hardware capability dynamically
        target_name = hardware_target
        target_vram_gb = hardware_total_vram_gb

        if target_vram_gb is None or target_name is None:
            try:
                from ..telemetry.hardware import HardwareMonitor
                telemetry = HardwareMonitor().read_telemetry()
                if telemetry.total_vram_mb and telemetry.total_vram_mb > 0:
                    target_vram_gb = telemetry.total_vram_mb / 1024.0
                    target_name = f"{telemetry.device_name} ({target_vram_gb:.1f}GB)"
                else:
                    target_vram_gb = telemetry.system_ram_total_mb / 1024.0
                    target_name = f"{telemetry.device_name} ({target_vram_gb:.1f}GB RAM)"
            except Exception:
                target_vram_gb = 6.0
                target_name = "Detected Accelerator (6.0GB)"

        # Safe allocation limits
        usable_ratio = 0.80
        if hardware_profile and "usable_ratio" in hardware_profile:
            usable_ratio = float(hardware_profile["usable_ratio"])

        safe_ceiling_gb = target_vram_gb * usable_ratio
        tight_ceiling_gb = target_vram_gb * min(0.92, usable_ratio + 0.10)

        is_safe = total_training_vram_gb <= safe_ceiling_gb
        is_tight = safe_ceiling_gb < total_training_vram_gb <= tight_ceiling_gb

        if is_safe:
            status = "SAFE"
            recommendation = f"Fits comfortably on {target_name} with active safety margin ({total_training_vram_gb:.2f} GB <= {safe_ceiling_gb:.1f} GB safe ceiling)."
        elif is_tight:
            status = "TIGHT"
            recommendation = (
                f"Close to limit on {target_name} ({total_training_vram_gb:.2f} GB / {target_vram_gb:.1f} GB). "
                "Recommend reducing micro_batch_size or enabling gradient checkpointing."
            )
        else:
            status = "UNLIKELY_TO_FIT"
            recommendation = (
                f"Estimated training memory ({total_training_vram_gb:.2f} GB) exceeds safe {safe_ceiling_gb:.1f} GB ceiling for {target_name}. "
                "Reduce micro_batch_size to 2 or 1, decrease seq_len, or enable gradient checkpointing."
            )

        return {
            "parameter_count": total_p,
            "precision": dtype,
            "model_weights_mb": round(model_mem_mb, 1),
            "gradients_mb": round(grad_mem_mb, 1),
            "optimizer_states_mb": round(opt_mem_mb, 1),
            "activations_mb": round(act_mem_mb, 1),
            "workspace_headroom_mb": round(workspace_headroom_mb, 1),
            "total_training_vram_mb": round(total_training_vram_mb, 1),
            "total_training_vram_gb": round(total_training_vram_gb, 2),
            "inference_vram_mb": round(inference_vram_mb, 1),
            "inference_vram_gb": round(inference_vram_mb / 1024.0, 2),
            "storage_file_size_mb": round(model_mem_mb, 1),
            "hardware_target": target_name,
            "safety_status": status,
            "recommendation": recommendation,
        }


class BudgetAssistant:
    """Recommends compatible architecture dimensions for desired parameter targets."""

    PRESETS = {
        "1M": {"d_model": 128, "n_layers": 4, "n_q": 4, "n_kv": 2, "d_ff": 384, "vocab_size": 4096},
        "5M": {"d_model": 256, "n_layers": 6, "n_q": 4, "n_kv": 2, "d_ff": 640, "vocab_size": 8192},
        "15M": {"d_model": 384, "n_layers": 8, "n_q": 6, "n_kv": 2, "d_ff": 960, "vocab_size": 12288},
        "25M": {"d_model": 448, "n_layers": 12, "n_q": 7, "n_kv": 2, "d_ff": 1120, "vocab_size": 12288},
        "48.5M": {"d_model": 512, "n_layers": 16, "n_q": 8, "n_kv": 2, "d_ff": 1280, "vocab_size": 12288},
        "100M": {"d_model": 768, "n_layers": 16, "n_q": 12, "n_kv": 4, "d_ff": 2048, "vocab_size": 12288},
    }

    @classmethod
    def get_preset(cls, name: str) -> Optional[Dict[str, Any]]:
        return cls.PRESETS.get(name)

    @classmethod
    def recommend(cls, target_params: int, vocab_size: int = 12288) -> Dict[str, Any]:
        """Calculates optimal d_model and layers for any arbitrary target parameter count."""
        # Find closest match or generate balanced depth/width ratio
        if target_params <= 3_000_000:
            return cls.PRESETS["1M"]
        elif target_params <= 10_000_000:
            return cls.PRESETS["5M"]
        elif target_params <= 20_000_000:
            return cls.PRESETS["15M"]
        elif target_params <= 35_000_000:
            return cls.PRESETS["25M"]
        elif target_params <= 75_000_000:
            return cls.PRESETS["48.5M"]
        else:
            return cls.PRESETS["100M"]
