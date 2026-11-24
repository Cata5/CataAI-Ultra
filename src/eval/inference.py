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
        """Loads weights into the model, dynamically creating custom architectures if saved."""
        import json
        chk_path = self.checkpoint_manager.resolve_checkpoint_path(checkpoint_path)
        meta_p = chk_path.with_suffix(".json")
        if not meta_p.exists() and (chk_path.parent / "metadata.json").exists():
            meta_p = chk_path.parent / "metadata.json"
        arch_cfg = None
        if meta_p.exists():
            try:
                with open(meta_p, "r", encoding="utf-8") as mf:
                    meta = json.load(mf)
                    arch_cfg = (
                        meta.get("config_arch")
                        or meta.get("arch")
                        or meta.get("training_config", {}).get("arch")
                    )
                    net_id = meta.get("network_id") or meta.get("model_id")
                    if not arch_cfg and net_id:
                        from ..networks.registry import NetworkRegistry
                        arch_cfg = NetworkRegistry().get_network(net_id)
            except Exception:
                pass

        if arch_cfg:
            from ..training.factory import create_model_from_dict
            try:
                self.model = create_model_from_dict(arch_cfg).to(device=self.device)
                self.model.eval()
            except Exception as e:
                logger.warning("Could not instantiate custom architecture for inference: %s", e)

        # Inspect checkpoint state_dict directly to verify and auto-adapt model dimensions
        try:
            if chk_path.suffix == ".safetensors":
                from safetensors.torch import load_file
                chk_sd = load_file(str(chk_path), device="cpu")
            else:
                try:
                    payload = torch.load(chk_path, map_location="cpu", weights_only=True)
                except Exception:
                    payload = torch.load(chk_path, map_location="cpu", weights_only=False)
                chk_sd = payload.get("model_state_dict", payload) if isinstance(payload, dict) else {}

            if isinstance(chk_sd, dict) and any("embed_tokens.weight" in k or "weight" in k for k in chk_sd.keys()):
                import re
                clean_keys = {}
                for k, v in chk_sd.items():
                    ck = k
                    for pfx in ("_orig_mod.", "module.", "model."):
                        if ck.startswith(pfx):
                            ck = ck[len(pfx):]
                    clean_keys[ck] = v

                embed_w = clean_keys.get("embed_tokens.weight")
                if embed_w is None:
                    embed_w = clean_keys.get("tok_embeddings.weight")
                if embed_w is not None and len(embed_w.shape) == 2:
                    chk_vocab = embed_w.shape[0]
                    chk_d_model = embed_w.shape[1]

                    layer_idxs = [int(m.group(1)) for k in clean_keys.keys() for m in [re.search(r"layers\.(\d+)\.", k)] if m]
                    chk_n_layers = (max(layer_idxs) + 1) if layer_idxs else 16

                    gate_w = clean_keys.get("layers.0.mlp.gate_proj.weight")
                    chk_d_ff = gate_w.shape[0] if gate_w is not None else (chk_d_model * 4)

                    q_norm_w = clean_keys.get("layers.0.attn.q_norm.weight")
                    chk_d_h = q_norm_w.shape[0] if q_norm_w is not None else 64

                    q_w = clean_keys.get("layers.0.attn.q_proj.weight")
                    chk_n_q = (q_w.shape[0] // chk_d_h) if q_w is not None and chk_d_h > 0 else 8

                    k_w = clean_keys.get("layers.0.attn.k_proj.weight")
                    chk_n_kv = (k_w.shape[0] // chk_d_h) if k_w is not None and chk_d_h > 0 else 2

                    conv_w = clean_keys.get("layers.0.canon_a.conv.weight")
                    chk_kernel_size = conv_w.shape[-1] if conv_w is not None and len(conv_w.shape) >= 3 else 4

                    # Check if currently active model has matching architecture
                    curr_cfg = getattr(self.model, "config", None)
                    needs_rebuild = (
                        curr_cfg is None
                        or getattr(curr_cfg, "d_model", None) != chk_d_model
                        or getattr(curr_cfg, "n_layers", None) != chk_n_layers
                        or getattr(curr_cfg, "d_ff", None) != chk_d_ff
                        or getattr(curr_cfg, "n_q", None) != chk_n_q
                        or getattr(curr_cfg, "n_kv", None) != chk_n_kv
                    )
                    if needs_rebuild:
                        new_cfg = RoCanonConfig()
                        new_cfg.vocab_size = chk_vocab
                        new_cfg.d_model = chk_d_model
                        new_cfg.n_layers = chk_n_layers
                        new_cfg.d_ff = chk_d_ff
                        new_cfg.d_h = chk_d_h
                        new_cfg.n_q = chk_n_q
                        new_cfg.n_kv = chk_n_kv
                        new_cfg.kernel_size = chk_kernel_size
                        self.model = RoCanonForCausalLM(new_cfg).to(device=self.device)
                        self.model.eval()
                        logger.info("Auto-adapted InferenceEngine model architecture to match checkpoint tensors: d_model=%d, n_layers=%d, d_ff=%d", chk_d_model, chk_n_layers, chk_d_ff)
        except Exception as e:
            logger.warning("Could not auto-inspect checkpoint tensor shapes: %s", e)

        return self.checkpoint_manager.load_checkpoint(
            checkpoint_path=chk_path,
            model=self.model,
            device=self.device,
            restore_training_state=False,
        )

    @torch.inference_mode()
    def attention_map(self, prompt: str, layer: int = 0, head: int = 0, max_tokens: int = 256) -> Dict[str, Any]:
        """Returns a real causal attention matrix for one layer/head.

        Only the selected block materialises scores; every other block keeps the
        normal fused SDPA path. The matrix is [prompt_tokens, prompt_tokens].
        """
        if not prompt.strip():
            raise ValueError("Prompt must not be empty")
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)[-max_tokens:]
        if len(ids) < 1:
            raise ValueError("Prompt produced no tokens")
        blocks = getattr(self.model, "layers", None)
        if blocks is None or not (0 <= layer < len(blocks)):
            raise ValueError(f"Layer {layer} is out of range")
        attn = blocks[layer].attn
        n_heads = int(getattr(attn, "n_q", 1))
        if not (0 <= head < n_heads):
            raise ValueError(f"Head {head} is out of range (0..{n_heads - 1})")

        input_ids = torch.tensor([ids], device=self.device, dtype=torch.long)
        attn.capture_attention = True
        try:
            with torch.no_grad():
                self.model(input_ids)
            tensor = getattr(attn, "last_attention", None)
        finally:
            attn.capture_attention = False
            attn.last_attention = None
        if tensor is None:
            raise RuntimeError("Attention was not captured")
        matrix = tensor[0, head].float().cpu()
        tokens = [self.tokenizer.id_to_token(int(token_id)) or str(token_id) for token_id in ids]
        return {
            "layer": layer,
            "head": head,
            "n_heads": n_heads,
            "tokens": tokens,
            "matrix": [[round(float(value), 6) for value in row] for row in matrix],
        }

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        min_new_tokens: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.15,
        no_repeat_ngram_size: int = 3,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        seed: Optional[int] = None,
        stop_tokens: Optional[List[int]] = None,
        stop_sequences: Optional[List[str]] = None,
        chat_mode: bool = False,
    ) -> Dict[str, Any]:
        """Executes text generation and returns text along with token throughput metrics."""
        t0 = time.time()
        if seed is not None and seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

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
        token_counts: Dict[int, int] = {}

        for _ in range(max_new_tokens):
            if curr_ids.shape[1] > max_seq_len:
                context_ids = curr_ids[:, -max_seq_len:]
            else:
                context_ids = curr_ids

            logits = self.model(context_ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            next_token_logits = logits[0, -1, :].clone()

            # Presence and Frequency penalties
            if presence_penalty != 0.0 or frequency_penalty != 0.0:
                for tid, count in token_counts.items():
                    if count > 0:
                        penalty = (presence_penalty if presence_penalty != 0.0 else 0.0) + (frequency_penalty * count)
                        next_token_logits[tid] -= penalty

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

                # Min-P (filters tokens with prob < min_p * max_prob)
                if min_p > 0.0:
                    probs_all = F.softmax(next_token_logits, dim=-1)
                    p_max = probs_all.max()
                    threshold = p_max * min_p
                    next_token_logits[probs_all < threshold] = -float("Inf")

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(next_token_logits, dim=-1).item()

            if next_token in stops:
                if len(generated_token_ids) >= min_new_tokens:
                    break
                else:
                    for s in stops:
                        next_token_logits[s] = -float("Inf")
                    if temperature > 0:
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).item()
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1).item()
                    if next_token in stops:
                        break

            generated_token_ids.append(next_token)
            token_counts[next_token] = token_counts.get(next_token, 0) + 1
            curr_ids = torch.cat([curr_ids, torch.tensor([[next_token]], device=self.device)], dim=1)

            # Check custom stop sequences
            if stop_sequences:
                curr_text = self.tokenizer.decode(generated_token_ids)
                if any(s in curr_text for s in stop_sequences):
                    break

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

    @torch.inference_mode()
    def stream_generate(
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
    ):
        """Streams generated tokens one by one as they are decoded for interactive typewriter effects."""
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

        prev_decoded = ""
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

            # Ban duplicate n-grams
            if no_repeat_ngram_size > 0 and len(generated_token_ids) >= no_repeat_ngram_size - 1:
                prefix = tuple(generated_token_ids[-(no_repeat_ngram_size - 1):])
                for i in range(len(generated_token_ids) - no_repeat_ngram_size + 1):
                    if tuple(generated_token_ids[i : i + no_repeat_ngram_size - 1]) == prefix:
                        banned_token = generated_token_ids[i + no_repeat_ngram_size - 1]
                        next_token_logits[banned_token] = -float("Inf")

            # Temperature & Sampling
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[-1]] = -float("Inf")
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
                if len(generated_token_ids) >= min_new_tokens:
                    break
                else:
                    for s in stops:
                        next_token_logits[s] = -float("Inf")
                    if temperature > 0:
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).item()
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1).item()
                    if next_token in stops:
                        break

            generated_token_ids.append(next_token)
            curr_ids = torch.cat([curr_ids, torch.tensor([[next_token]], device=self.device)], dim=1)

            # Incremental decoding to yield text diff
            full_decoded = self.tokenizer.decode(generated_token_ids)
            chunk = full_decoded[len(prev_decoded):]
            prev_decoded = full_decoded
            if chunk:
                yield chunk

    def compare_checkpoints_sequential(
        self,
        prompt: str,
        checkpoint_paths: List[str],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.15,
        no_repeat_ngram_size: int = 3,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        chat_mode: bool = False,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Sequentially tests multiple checkpoints with user-configured parameters.
        Reuses the model instance and dynamically handles architectures safely.
        """
        results: List[Dict[str, Any]] = []

        for chk_path in checkpoint_paths:
            try:
                resolved_p = self.checkpoint_manager.resolve_checkpoint_path(chk_path)
                load_meta = self.load_checkpoint_weights(resolved_p)
                gen_res = self.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    min_p=min_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    chat_mode=chat_mode,
                    seed=seed,
                    stop_sequences=stop_sequences,
                )
                chk_name = Path(resolved_p).name
                results.append({
                    "checkpoint": chk_name,
                    "checkpoint_name": chk_name,
                    "checkpoint_path": str(resolved_p),
                    "step": load_meta.get("step", 0),
                    "val_loss": load_meta.get("val_loss"),
                    "generated_text": gen_res.get("generated_text", ""),
                    "completion": gen_res.get("completion", gen_res.get("generated_text", "")),
                    "tokens_per_sec": gen_res.get("tokens_per_sec", 0),
                    "tokens_per_second": gen_res.get("tokens_per_sec", 0),
                    "generation_time_sec": gen_res.get("generation_time_sec", 0),
                    "success": True,
                })
            except Exception as e:
                logger.exception("Failed evaluation on checkpoint %s: %s", chk_path, e)
                fallback_name = Path(chk_path).name
                results.append({
                    "checkpoint": fallback_name,
                    "checkpoint_name": fallback_name,
                    "checkpoint_path": str(chk_path),
                    "step": 0,
                    "val_loss": None,
                    "generated_text": f"[Error: {e}]",
                    "completion": f"[Error: {e}]",
                    "tokens_per_sec": 0,
                    "tokens_per_second": 0,
                    "generation_time_sec": 0,
                    "success": False,
                    "error": str(e),
                })

        return results

