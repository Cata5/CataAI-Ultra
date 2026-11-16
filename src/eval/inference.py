"""
src/eval/inference.py
Live inference and sequential side-by-side checkpoint comparison for RoCanon-48.5M.
Enforces the critical GPU safety rule: NEVER load two model instances simultaneously on the 6GB GPU.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..checkpoints.manager import CheckpointManager
from ..model.architecture import RoCanonConfig, RoCanonForCausalLM
from ..tokenizer.tokenizer import RoCanonTokenizer

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Manages single-instance inference and memory-safe sequential checkpoint comparisons."""

    def __init__(
        self,
        model: Optional[RoCanonForCausalLM] = None,
        tokenizer: Optional[RoCanonTokenizer] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = tokenizer or RoCanonTokenizer()
        self.model = model or RoCanonForCausalLM(RoCanonConfig()).to(device=self.device)
        self.model.eval()
        self.checkpoint_manager = CheckpointManager()

    def load_checkpoint_weights(self, checkpoint_path: str | Path) -> Dict[str, Any]:
        """Loads weights into the single existing model instance without allocating a second model."""
        return self.checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            device=self.device,
            restore_training_state=False,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.15,
        no_repeat_ngram_size: int = 3,
        stop_tokens: Optional[List[int]] = None,
        chat_mode: bool = False,
    ) -> Dict[str, Any]:
        """Executes text generation and returns text along with token throughput metrics."""
        t0 = time.time()

        # ChatML / Instruction formatting
        formatted_prompt = prompt
        if chat_mode and not ("<|user|>" in prompt or "<|im_start|>" in prompt):
            formatted_prompt = f"<|user|>\n{prompt.strip()}\n<|assistant|>\n"

        input_ids = self.tokenizer.encode(formatted_prompt)
        curr_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        default_stops = set()
        if self.tokenizer.eos_token_id is not None:
            default_stops.add(self.tokenizer.eos_token_id)
        for stop_tag in ["<|end_turn|>", "<|im_end|>", "<|eos|>", "<|endoftext|>"]:
            tid = self.tokenizer._tokenizer.token_to_id(stop_tag)
            if tid is not None:
                default_stops.add(tid)

        stops = set(stop_tokens) if stop_tokens is not None else default_stops
        generated_token_ids: List[int] = []

        max_seq_len = getattr(self.model.config, "max_seq_len", 2048)

        for _ in range(max_new_tokens):
            if curr_ids.shape[1] > max_seq_len:
                context_ids = curr_ids[:, -max_seq_len:]
            else:
                context_ids = curr_ids

            logits = self.model(context_ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            next_token_logits = logits[0, -1, :].clone()

            # Repetition penalty
            if repetition_penalty != 1.0:
                for tid in set(curr_ids[0].tolist()):
                    if next_token_logits[tid] > 0:
                        next_token_logits[tid] /= repetition_penalty
                    else:
                        next_token_logits[tid] *= repetition_penalty

            # Ban duplicate n-grams to prevent repetitive loops (e.g. "de exemplu de exemplu...")
            if no_repeat_ngram_size > 0 and len(generated_token_ids) >= no_repeat_ngram_size - 1:
                prefix = tuple(generated_token_ids[-(no_repeat_ngram_size - 1):])
                for i in range(len(generated_token_ids) - no_repeat_ngram_size + 1):
                    if tuple(generated_token_ids[i : i + no_repeat_ngram_size - 1]) == prefix:
                        banned_token = generated_token_ids[i + no_repeat_ngram_size - 1]
                        next_token_logits[banned_token] = -float("Inf")

            # Temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                # Top-K
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[-1]] = -float("Inf")
                # Top-P
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float("Inf")

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(next_token_logits, dim=-1).item()

            if next_token in stops:
                break

            generated_token_ids.append(next_token)
            curr_ids = torch.cat([curr_ids, torch.tensor([[next_token]], device=self.device)], dim=1)

        dt = time.time() - t0
        tok_count = len(generated_token_ids)
        tok_per_sec = tok_count / max(1e-5, dt)
        output_text = self.tokenizer.decode(generated_token_ids)

        for stop_str in ["<|end_turn|>", "<|im_end|>", "<|eos|>", "<|endoftext|>"]:
            if output_text.endswith(stop_str):
                output_text = output_text[:-len(stop_str)].strip()

        return {
            "prompt": prompt,
            "formatted_prompt": formatted_prompt,
            "completion": output_text,
            "generated_text": output_text,
            "full_text": formatted_prompt + output_text,
            "tokens_generated": tok_count,
            "generation_time_sec": round(dt, 3),
            "tokens_per_sec": round(tok_per_sec, 1),
            "total_tokens": len(input_ids) + tok_count,
        }

    def compare_checkpoints_sequential(
        self,
        prompt: str,
        checkpoint_paths: List[str],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[Dict[str, Any]]:
        """
        Sequentially tests multiple checkpoints with identical parameters.
        Reuses the single model instance on the GPU to guarantee 0 risk of CUDA OOM.
        """
        results: List[Dict[str, Any]] = []

        for chk_path in checkpoint_paths:
            try:
                load_meta = self.load_checkpoint_weights(chk_path)
                gen_res = self.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                results.append({
                    "checkpoint_path": chk_path,
                    "checkpoint_name": Path(chk_path).name,
                    "step": load_meta.get("step", 0),
                    "val_loss": load_meta.get("val_loss"),
                    "generated_text": gen_res["generated_text"],
                    "tokens_per_sec": gen_res["tokens_per_sec"],
                    "generation_time_sec": gen_res["generation_time_sec"],
                    "success": True,
                })
            except Exception as e:
                logger.exception("Failed evaluation on checkpoint %s: %s", chk_path, e)
                results.append({
                    "checkpoint_path": chk_path,
                    "checkpoint_name": Path(chk_path).name,
                    "success": False,
                    "error": str(e),
                })

        return results

