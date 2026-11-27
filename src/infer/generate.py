"""
src/infer/generate.py
Inference & interactive generation pipeline for RoCanon-48.5M.
"""

from __future__ import annotations

import argparse
import sys
from typing import Generator, List, Optional

import torch
import torch.nn.functional as F

from ..model.architecture import RoCanonConfig, RoCanonForCausalLM
from ..tokenizer.tokenizer import RoCanonTokenizer


class RoCanonGenerator:
    """
    Text generator for RoCanon-48.5M supporting KV-cache-friendly generation,
    temperature scaling, top-p nucleus sampling, and streaming tokens.
    """

    def __init__(
        self,
        model: RoCanonForCausalLM,
        tokenizer: RoCanonTokenizer,
        device: Optional[torch.device] = None,
    ):
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(device=self.device)
        self.model.eval()
        self.tokenizer = tokenizer

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        tokenizer_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> RoCanonGenerator:
        """Loads RoCanon-48.5M model from checkpoint file."""
        target_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        config = RoCanonConfig()
        model = RoCanonForCausalLM(config)

        if checkpoint_path.endswith(".pt") or checkpoint_path.endswith(".bin"):
            checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=True)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
        elif checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path, device=str(target_device))
            model.load_state_dict(state_dict)

        tokenizer = RoCanonTokenizer(tokenizer_path) if tokenizer_path else RoCanonTokenizer()
        return cls(model, tokenizer, target_device)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[List[int]] = None,
    ) -> str:
        """Generates text from a single prompt string."""
        tokens = []
        for token_text in self.stream_generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_tokens=stop_tokens,
        ):
            tokens.append(token_text)
        return "".join(tokens)

    @torch.inference_mode()
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[List[int]] = None,
    ) -> Generator[str, None, None]:
        """Streams generated tokens one by one."""
        input_ids = self.tokenizer.encode(prompt)
        curr_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        eos_id = self.tokenizer.eos_token_id
        stops = set(stop_tokens or [eos_id])

        for _ in range(max_new_tokens):
            # Truncate context if exceeding model max context
            if curr_ids.shape[1] > self.model.config.max_seq_len:
                context_ids = curr_ids[:, -self.model.config.max_seq_len:]
            else:
                context_ids = curr_ids

            logits = self.model(context_ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            next_token_logits = logits[0, -1, :].clone()

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(curr_ids[0].tolist()):
                    if next_token_logits[token_id] > 0:
                        next_token_logits[token_id] /= repetition_penalty
                    else:
                        next_token_logits[token_id] *= repetition_penalty

            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above top_p
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

            token_str = self.tokenizer.decode([next_token])
            yield token_str

            curr_ids = torch.cat([curr_ids, torch.tensor([[next_token]], device=self.device)], dim=1)


def generate_text(prompt: str, checkpoint_path: str, max_tokens: int = 128) -> str:
    """Helper functional API to generate text."""
    generator = RoCanonGenerator.from_checkpoint(checkpoint_path)
    return generator.generate(prompt, max_new_tokens=max_tokens)


def interactive_chat():
    """Runs interactive CLI chat in the terminal."""
    parser = argparse.ArgumentParser(description="RoCanon-48.5M Interactive Chat CLI")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    args = parser.parse_args()

    print("=" * 60)
    print("RoCanon-48.5M Interactive Chat CLI (Bilingual RO/EN)")
    print("=" * 60)

    config = RoCanonConfig()
    model = RoCanonForCausalLM(config)
    tokenizer = RoCanonTokenizer()

    if args.checkpoint:
        print(f"Loading checkpoint from: {args.checkpoint}")
        chk = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(chk.get("model_state_dict", chk))

    generator = RoCanonGenerator(model, tokenizer)
    print("Model initialized. Type 'quit' or 'exit' to terminate.\n")

    while True:
        try:
            user_input = input("User > ")
            if user_input.strip().lower() in ("quit", "exit"):
                break
            if not user_input.strip():
                continue

            prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
            sys.stdout.write("RoCanon > ")
            sys.stdout.flush()

            for token_chunk in generator.stream_generate(prompt, max_new_tokens=150):
                sys.stdout.write(token_chunk)
                sys.stdout.flush()
            sys.stdout.write("\n\n")
        except KeyboardInterrupt:
            print("\nExiting chat.")
            break


if __name__ == "__main__":
    interactive_chat()

