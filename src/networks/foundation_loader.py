"""
src/networks/foundation_loader.py
Open-Weights Foundation Model Importer & Loader for CataAI Ultra.
Integrates Hugging Face Hub foundation models (Llama 3.x, Qwen 2.5, DeepSeek-R1 distillations,
Mistral, Gemma 2) directly into the Network Registry and Training Engine.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

FOUNDATION_PRESETS: List[Dict[str, Any]] = [
    {
        "id": "qwen2.5-0.5b",
        "name": "Qwen 2.5 0.5B Instruct",
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "parameters": "490M",
        "params_raw": 490_000_000,
        "context_len": 32768,
        "vram_req_fp16": "1.2 GB",
        "vram_req_4bit": "450 MB",
        "description": "Ultra-lightweight state-of-the-art SLM, runs comfortably on any consumer GPU.",
        "recommended_for": ["Low-VRAM Fine-Tuning", "Rapid Prototyping", "Edge Deployment"],
    },
    {
        "id": "qwen2.5-1.5b",
        "name": "Qwen 2.5 1.5B Instruct",
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "parameters": "1.54B",
        "params_raw": 1_540_000_000,
        "context_len": 32768,
        "vram_req_fp16": "3.4 GB",
        "vram_req_4bit": "1.1 GB",
        "description": "Best-in-class small foundation model with exceptional coding and multilingual capabilities.",
        "recommended_for": ["Instruction Tuning", "Bilingual Dialogue", "Code Synthesis"],
    },
    {
        "id": "deepseek-r1-1.5b",
        "name": "DeepSeek-R1 Distill Qwen 1.5B",
        "repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "parameters": "1.54B",
        "params_raw": 1_540_000_000,
        "context_len": 32768,
        "vram_req_fp16": "3.4 GB",
        "vram_req_4bit": "1.1 GB",
        "description": "Native chain-of-thought reasoning model trained with GRPO and DeepSeek-R1 distillation.",
        "recommended_for": ["Mathematical Reasoning", "Logic Verification", "Chain-of-Thought"],
    },
    {
        "id": "llama3.2-1b",
        "name": "Llama 3.2 1B Instruct",
        "repo_id": "meta-llama/Llama-3.2-1B-Instruct",
        "parameters": "1.24B",
        "params_raw": 1_240_000_000,
        "context_len": 131072,
        "vram_req_fp16": "2.8 GB",
        "vram_req_4bit": "900 MB",
        "description": "Meta's official lightweight Llama architecture with 128k context support.",
        "recommended_for": ["General Fine-Tuning", "Agent Tasks", "Summarization"],
    },
    {
        "id": "llama3.2-3b",
        "name": "Llama 3.2 3B Instruct",
        "repo_id": "meta-llama/Llama-3.2-3B-Instruct",
        "parameters": "3.21B",
        "params_raw": 3_210_000_000,
        "context_len": 131072,
        "vram_req_fp16": "6.8 GB",
        "vram_req_4bit": "2.1 GB",
        "description": "Highly capable 3B model, fits in 6GB VRAM with 4-bit QLoRA.",
        "recommended_for": ["High-Performance Dialogue", "Complex Instructions"],
    },
    {
        "id": "gemma2-2b",
        "name": "Gemma 2 2B IT",
        "repo_id": "google/gemma-2-2b-it",
        "parameters": "2.6B",
        "params_raw": 2_600_000_000,
        "context_len": 8192,
        "vram_req_fp16": "5.6 GB",
        "vram_req_4bit": "1.7 GB",
        "description": "Google DeepMind's efficient Gemma 2 architecture with sliding window attention.",
        "recommended_for": ["Knowledge Tasks", "Reasoning", "Multilingual"],
    },
]


class FoundationModelManager:
    """Manages discovery, inspection, and registry registration of Open-Weights Foundation Models."""

    def __init__(self, registry_dir: Union[str, Path] = "networks/registry") -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def list_presets(self) -> List[Dict[str, Any]]:
        """Returns catalogue of popular open-weights foundation models."""
        return FOUNDATION_PRESETS

    def inspect_repo(
        self,
        repo_id: str,
        hf_token: Optional[str] = None,
        allow_remote_code: bool = False,
    ) -> Dict[str, Any]:
        """Inspects model configuration directly from Hugging Face Hub.

        Remote code is opt-in per import: a repository can ship Python that runs
        with the app's privileges, so the default path never executes it.
        """
        from transformers import AutoConfig

        try:
            cfg = AutoConfig.from_pretrained(
                repo_id, token=hf_token, trust_remote_code=allow_remote_code
            )
            cfg_dict = cfg.to_dict()

            hidden_size = cfg_dict.get("hidden_size") or cfg_dict.get("d_model", 2048)
            num_layers = cfg_dict.get("num_hidden_layers") or cfg_dict.get("n_layer", 24)
            num_heads = cfg_dict.get("num_attention_heads") or cfg_dict.get("n_head", 16)
            vocab_size = cfg_dict.get("vocab_size", 32000)
            max_pos = cfg_dict.get("max_position_embeddings") or cfg_dict.get("max_seq_len", 4096)

            # Estimate total parameter count
            intermediate_size = cfg_dict.get("intermediate_size", hidden_size * 4)
            emb_params = vocab_size * hidden_size
            attn_params = num_layers * (4 * hidden_size * hidden_size)
            mlp_params = num_layers * (3 * hidden_size * intermediate_size)
            est_params = emb_params + attn_params + mlp_params

            return {
                "success": True,
                "repo_id": repo_id,
                "architecture": cfg_dict.get("architectures", ["CausalLM"])[0],
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "vocab_size": vocab_size,
                "max_position_embeddings": max_pos,
                "estimated_parameters": est_params,
                "raw_config": cfg_dict,
            }
        except Exception as e:
            logger.error("Failed inspecting HF repo %s: %s", repo_id, e)
            message = str(e)
            needs_remote_code = (not allow_remote_code) and (
                "trust_remote_code" in message.lower() or "custom code" in message.lower()
            )
            return {
                "success": False,
                "error": message,
                "repo_id": repo_id,
                "needs_remote_code": needs_remote_code,
            }

    def import_to_registry(
        self,
        repo_id: str,
        custom_name: Optional[str] = None,
        workspace_id: str = "default",
        hf_token: Optional[str] = None,
        allow_remote_code: bool = False,
    ) -> Dict[str, Any]:
        """Registers a foundation model in the Network Studio registry as an imported network."""
        info = self.inspect_repo(repo_id, hf_token=hf_token, allow_remote_code=allow_remote_code)
        if not info.get("success"):
            raise ValueError(f"Could not inspect foundation model: {info.get('error')}")

        clean_id = repo_id.replace("/", "_").replace("-", "_").lower()
        target_name = (custom_name or repo_id.split("/")[-1]).strip()

        net_cfg = {
            "id": f"hf_{clean_id}",
            "name": target_name,
            "workspace_id": workspace_id,
            "is_foundation": True,
            "hf_repo_id": repo_id,
            "trust_remote_code": bool(allow_remote_code),
            "family": "decoder_only",
            "description": f"Imported Open-Weights Foundation Model ({repo_id}). Total parameters: ~{info.get('estimated_parameters', 0):,}.",
            "architecture": {
                "type": "foundation_huggingface",
                "repo_id": repo_id,
                "vocab_size": info.get("vocab_size"),
                "hidden_size": info.get("hidden_size"),
                "num_layers": info.get("num_layers"),
                "num_heads": info.get("num_heads"),
                "max_sequence_length": info.get("max_position_embeddings"),
            },
            "total_parameters": info.get("estimated_parameters", 0),
            "created_at": None,
        }

        out_path = self.registry_dir / f"hf_{clean_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(net_cfg, f, indent=2)

        logger.info("Successfully imported foundation model %s into %s", repo_id, out_path)
        return net_cfg


def load_foundation_model(
    repo_id_or_path: str,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
    load_in_4bit: bool = False,
    hf_token: Optional[str] = None,
    trust_remote_code: bool = False,
) -> Tuple[nn.Module, Any]:
    """Loads foundation model and tokenizer using Hugging Face AutoClasses."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Loading foundation model '%s' onto %s (4-bit=%s)...", repo_id_or_path, dev, load_in_4bit)

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id_or_path,
        token=hf_token,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "token": hf_token,
    }

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )
            kwargs["device_map"] = "auto"
        except Exception:
            logger.warning("bitsandbytes not available for 4-bit, loading in half precision on device.")
            kwargs["torch_dtype"] = dtype
    else:
        kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(repo_id_or_path, **kwargs)
    if not load_in_4bit and dev.type != "meta":
        model = model.to(dev)

    return model, tokenizer
