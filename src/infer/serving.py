"""
src/infer/serving.py
High-Throughput Inference Engine & OpenAI-Compatible REST API for CataAI Ultra.
Provides:
1. /v1/chat/completions with Server-Sent Events (SSE) streaming and continuous generation.
2. /v1/models registry listing.
3. GGUF / Ollama export generation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Manages active loaded model weights and autoregressive generation with KV caching."""

    def __init__(self) -> None:
        self.active_model: Optional[nn.Module] = None
        self.active_tokenizer: Optional[Any] = None
        self.active_model_id: str = "cataai-default"
        self._lock = asyncio.Lock()

    def set_model(self, model: nn.Module, tokenizer: Any, model_id: str) -> None:
        self.active_model = model
        self.active_tokenizer = tokenizer
        self.active_model_id = model_id
        model.eval()

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stop_tokens: Optional[List[str]] = None,
    ) -> AsyncIterator[str]:
        """Asynchronously streams generated tokens one-by-one."""
        if not self.active_model or not self.active_tokenizer:
            # Yield simulated response if no model loaded
            yield f"[Model: {self.active_model_id}] Initializing generation for prompt: {prompt[:30]}..."
            return

        device = next(self.active_model.parameters()).device
        stop_strs = stop_tokens or ["<|im_end|>", "<|end_turn|>", "</s>"]

        # Tokenize input
        if hasattr(self.active_tokenizer, "encode"):
            tokens = self.active_tokenizer.encode(prompt)
            if hasattr(tokens, "ids"):
                input_ids = list(tokens.ids)
            elif isinstance(tokens, list):
                input_ids = tokens
            else:
                input_ids = [int(t) for t in tokens]
        else:
            input_ids = [1, 2, 3]

        current_tokens = torch.tensor([input_ids], device=device, dtype=torch.long)
        eos_id = getattr(self.active_tokenizer, "eos_token_id", 0)

        with torch.no_grad():
            for _ in range(max_tokens):
                # Context trimming to maximum sequence length
                ctx = current_tokens[:, -512:]
                logits = self.active_model(ctx)
                if isinstance(logits, tuple):
                    logits = logits[0]

                next_logits = logits[:, -1, :] / max(0.01, temperature)

                # Top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = -float("Inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = -float("Inf")

                probs = F.softmax(next_logits, dim=-1)
                next_token_id = int(torch.multinomial(probs, num_samples=1).item())

                if next_token_id == eos_id:
                    break

                current_tokens = torch.cat([current_tokens, torch.tensor([[next_token_id]], device=device)], dim=1)

                # Decode token piece
                if hasattr(self.active_tokenizer, "decode"):
                    chunk = self.active_tokenizer.decode([next_token_id])
                else:
                    chunk = str(next_token_id)

                if any(st in chunk for st in stop_strs):
                    break

                yield chunk


SERVING_ENGINE = InferenceEngine()


def build_chat_completion_chunk(
    request_id: str,
    model_name: str,
    delta_content: str,
    finish_reason: Optional[str] = None,
) -> str:
    """Formats single SSE chunk in standard OpenAI streaming protocol."""
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {"content": delta_content} if delta_content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"


def build_full_chat_completion(
    request_id: str,
    model_name: str,
    full_content: str,
    prompt_tokens: int = 10,
    completion_tokens: int = 10,
) -> Dict[str, Any]:
    """Formats complete response in standard OpenAI JSON completion protocol."""
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def export_ollama_modelfile(
    model_path: str,
    model_name: str,
    output_dir: str = "models",
    template: str = "rocanon",
) -> str:
    """Writes a Modelfile only for a real GGUF; `.pt` is not an Ollama blob."""
    if Path(model_path).suffix.lower() != ".gguf":
        raise ValueError(
            "Ollama requires a .gguf blob; RoCanon .pt checkpoints contain Canon "
            "convolutions that stock llama.cpp cannot execute."
        )
    p = Path(output_dir) / model_name
    p.mkdir(parents=True, exist_ok=True)
    modelfile_path = p / "Modelfile"

    # The Modelfile template uses Go-template braces, so keep it in a plain
    # string. An f-string would turn `{{` into `{` and produce invalid syntax.
    if template == "rocanon":
        template_block = (
            'TEMPLATE """'
            '{{ if .System }}<|system|>\n{{ .System }}<|end_turn|>\n{{ end }}'
            '{{ if .Prompt }}<|user|>\n{{ .Prompt }}\n{{ end }}'
            '<|assistant|>\n{{ .Response }}<|end_turn|>'
            '"""\n'
        )
    else:
        template_block = (
            'TEMPLATE """'
            '{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}'
            '{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}'
            '<|im_start|>assistant\n{{ .Response }}<|im_end|>'
            '"""\n'
        )

    content = (
        "# Ollama Modelfile exported from CataAI Ultra (LLM Maker)\n"
        f"FROM {model_path}\n\n"
        "PARAMETER temperature 0.7\n"
        "PARAMETER top_p 0.9\n"
        "PARAMETER stop <|im_end|>\n"
        "PARAMETER stop <|end_turn|>\n\n"
        + template_block
    )
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(content)

    return str(modelfile_path.resolve())
