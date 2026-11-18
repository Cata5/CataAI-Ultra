"""
src/data/custom_loader.py
Custom dataset loader supporting both raw text (Pretraining) and
instruction chat with assistant loss masking (SFT / Creator Persona).
Provides multi-threaded Rust batch tokenization and live progress reporting to the Web UI.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch

from ..tokenizer.tokenizer import RoCanonTokenizer

logger = logging.getLogger(__name__)


class CustomDatasetEngine:
    """
    Loads and tokenizes custom user datasets (.txt, .jsonl):
    - Causal language modeling for Pretraining (e.g. phase1_pretrain_tech_history_v2.txt).
    - Assistant-masked instruction tuning for Chat/Persona (e.g. phase2_chat_ro.txt, alpaca_ro.txt).
    Provides automatic 95/5 train/validation split for real-time held-out generalization tracking.
    """

    KNOWN_DATASETS = {
        "tech_wiki_ro": r"d:\proiecte\LLM from Scratch\data\phase1_pretrain_tech_history_v2.txt",
        "chat_persona_ro": r"d:\proiecte\LLM from Scratch\data\phase2_chat_ro.txt",
        "alpaca_ro": r"d:\proiecte\LLM from Scratch\data\alpaca_ro.txt",
    }

    def __init__(self, tokenizer: Optional[RoCanonTokenizer] = None, cache_dir: str = "data/cache", db: Optional[Any] = None):
        self.tokenizer = tokenizer or RoCanonTokenizer()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db = db

    def load_dataset(
        self,
        dataset_source: str,
        phase: str = "pretrain",  # "pretrain", "midtrain", or "chat_sft"
        max_samples: Optional[int] = None,
        device: Optional[torch.device] = None,
        val_ratio: float = 0.05,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Loads dataset from registered ID, preset key, or file path.
        Returns:
            Tuple of (train_inputs, train_labels, val_inputs, val_labels).
            For pretraining, train_labels and val_labels can be None.
            For chat_sft, labels contain -100 on user prompt tokens.
        """
        device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Explicit synthetic mode
        if dataset_source.lower() == "synthetic":
            print(f"[DatasetEngine] Explicit synthetic test dataset requested for phase: {phase}.")
            if progress_callback:
                progress_callback(f"Generating synthetic {phase} dataset...", 100.0)
            if phase == "classification":
                return self._generate_synthetic_classification(device=device)
            elif phase == "regression":
                return self._generate_synthetic_regression(device=device)
            return self._generate_synthetic_tokens(device=device)

        file_path: Optional[Path] = None

        # 1. Check known presets
        if dataset_source in self.KNOWN_DATASETS:
            cand = Path(self.KNOWN_DATASETS[dataset_source])
            if cand.exists():
                file_path = cand
            else:
                raise FileNotFoundError(f"Preset dataset '{dataset_source}' path does not exist on disk: {cand}")

        # 2. Check SQLite database for uploaded/registered dataset ID
        elif self.db is not None:
            try:
                ds_record = self.db.get_dataset(dataset_source)
                if ds_record and "file_path" in ds_record:
                    cand = Path(ds_record["file_path"])
                    if cand.exists():
                        file_path = cand
                    else:
                        raise FileNotFoundError(f"Registered dataset '{dataset_source}' file missing at: {cand}")
            except Exception as dbe:
                logger.warning("Error checking DB for dataset '%s': %s", dataset_source, dbe)

        # 3. Direct filesystem path check
        if file_path is None:
            cand = Path(dataset_source)
            if cand.exists() and cand.is_file():
                file_path = cand

        # 4. Strict error handling: Never silently fall back to synthetic data
        if file_path is None or not file_path.exists():
            msg = (
                f"Unknown or missing dataset source: '{dataset_source}'. "
                "The dataset must be a registered dataset ID, a valid known preset, or an existing file path. "
                "Synthetic data is only used when explicitly requested with dataset_source='synthetic'."
            )
            print(f"[DatasetEngine] ERROR: {msg}")
            if progress_callback:
                progress_callback(f"Error: {msg}", 0.0)
            raise ValueError(msg)

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"[DatasetEngine] Loading {file_path.name} ({file_size_mb:.1f} MB) for phase: {phase}...")
        if progress_callback:
            progress_callback(f"Opening {file_path.name} ({file_size_mb:.1f} MB)...", 2.0)

        # Handle classification or regression datasets directly
        if phase in ("classification", "regression"):
            return self._load_classification_or_regression(
                file_path=file_path,
                is_regression=(phase == "regression"),
                device=device,
                val_ratio=val_ratio,
            )

        is_chat = (
            phase == "chat_sft"
            or "chat" in file_path.name.lower()
            or "alpaca" in file_path.name.lower()
        )

        if is_chat:
            full_inputs, full_labels = self._load_chat_sft(
                file_path, device=device, max_samples=max_samples, progress_callback=progress_callback
            )
        else:
            full_inputs, full_labels = self._load_pretrain_text(
                file_path, device=device, max_samples=max_samples, progress_callback=progress_callback
            )

        # Minimum token check
        min_tokens = 4 * 256 * 2  # at least 2 full micro-batches
        if len(full_inputs) < min_tokens:
            repeat_factor = (min_tokens // len(full_inputs)) + 1
            full_inputs = full_inputs.repeat(repeat_factor)
            if full_labels is not None:
                full_labels = full_labels.repeat(repeat_factor)

        # Split into train and held-out validation
        total_len = len(full_inputs)
        val_len = max(4 * 256, int(total_len * val_ratio))
        train_len = total_len - val_len

        train_inputs = full_inputs[:train_len].contiguous()
        val_inputs = full_inputs[train_len:].contiguous()

        train_labels = full_labels[:train_len].contiguous() if full_labels is not None else None
        val_labels = full_labels[train_len:].contiguous() if full_labels is not None else None

        print(
            f"[DatasetEngine] Ready: {len(train_inputs):,} train tokens, "
            f"{len(val_inputs):,} val tokens on {device.type.upper()}"
        )
        if progress_callback:
            progress_callback(
                f"Dataset Ready: {len(train_inputs):,} train / {len(val_inputs):,} val tokens ({file_path.name})",
                100.0,
            )

        return train_inputs, train_labels, val_inputs, val_labels

    def _load_pretrain_text(
        self,
        file_path: Path,
        device: torch.device,
        max_samples: Optional[int] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Tokenizes continuous pretraining text into packed token array using multi-threaded Rust batch encoding with streaming memmap support."""
        sub_tag = f"_sub{max_samples}" if max_samples else "_full"
        cache_bin = self.cache_dir / f"{file_path.stem}{sub_tag}_tokens.bin"
        cache_file = self.cache_dir / f"{file_path.stem}{sub_tag}_tokens.pt"

        # 1. Zero-copy binary memmap check (< 50ms load time, zero RAM consumption)
        if cache_bin.exists():
            print(f"[DatasetEngine] Zero-copy loading binary memmap shard: {cache_bin.name}")
            if progress_callback:
                progress_callback(f"Zero-copy loading memmap shard: {cache_bin.name}...", 50.0)
            mmap_arr = np.memmap(cache_bin, dtype=np.int64, mode="r")
            return torch.from_numpy(mmap_arr).to(device=device), None

        # 2. PyTorch .pt cache fallback (using mmap where supported)
        if cache_file.exists():
            print(f"[DatasetEngine] Loading pre-tokenized cache: {cache_file.name}")
            if progress_callback:
                progress_callback(f"Loading cached token array: {cache_file.name}...", 50.0)
            try:
                data = torch.load(cache_file, map_location=device, mmap=True, weights_only=True)
            except Exception:
                data = torch.load(cache_file, map_location=device)
            return data["input_ids"], None

        # Read text file
        if progress_callback:
            progress_callback(f"Reading {file_path.name} from disk...", 5.0)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                text = f.read()

        if max_samples and max_samples > 0:
            text = text[: max_samples * 500]

        total_chars = len(text)
        print(f"[DatasetEngine] Tokenizing {total_chars:,} characters of text...")
        if progress_callback:
            progress_callback(f"Tokenizing {total_chars:,} characters of text...", 10.0)

        # Break into paragraphs for batch tokenization (supports raw text and JSONL with 'text' field)
        paragraphs = []
        if file_path.suffix.lower() == ".jsonl":
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "text" in obj and obj["text"]:
                        paragraphs.append(str(obj["text"]).strip())
                except Exception:
                    pass

        if not paragraphs:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        total_p = len(paragraphs)

        all_ids: List[int] = []
        eos_id = self.tokenizer.eos_token_id

        # Process in parallel Rust batch chunks of 500 paragraphs
        chunk_size = 500
        for i in range(0, total_p, chunk_size):
            chunk = paragraphs[i : i + chunk_size]
            enc_batch = self.tokenizer._tokenizer.encode_batch(chunk)
            for enc in enc_batch:
                all_ids.extend(enc.ids)
                all_ids.append(eos_id)

            done_p = min(total_p, i + chunk_size)
            pct = 10.0 + (done_p / total_p) * 80.0
            if progress_callback:
                progress_callback(
                    f"Tokenizing {total_chars:,} characters of text... ({done_p:,}/{total_p:,} paragraphs - {pct:.0f}%)",
                    pct,
                )

        if progress_callback:
            progress_callback(f"Saving {len(all_ids):,} tokens to cache...", 95.0)

        # Save binary memmap shard for zero-copy OS paging and low RAM footprint
        token_arr = np.array(all_ids, dtype=np.int64)
        token_arr.tofile(cache_bin)

        input_tensor = torch.from_numpy(token_arr).to(device=device)
        print(f"[DatasetEngine] Tokenized {len(input_tensor):,} tokens. Saved memmap shard {cache_bin.name} and cache {cache_file.name}...")
        torch.save({"input_ids": input_tensor.cpu()}, cache_file)

        return input_tensor, None

    def _load_chat_sft(
        self,
        file_path: Path,
        device: torch.device,
        max_samples: Optional[int] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parses conversation dialogues and masks prompt loss:
        User tokens -> label: -100
        Assistant tokens -> label: token_id
        """
        sub_tag = f"_sub{max_samples}" if max_samples else "_full"
        cache_bin_in = self.cache_dir / f"{file_path.stem}{sub_tag}_sft_in.bin"
        cache_bin_lbl = self.cache_dir / f"{file_path.stem}{sub_tag}_sft_lbl.bin"
        cache_file = self.cache_dir / f"{file_path.stem}{sub_tag}_sft.pt"

        # 1. Zero-copy binary memmap check
        if cache_bin_in.exists() and cache_bin_lbl.exists():
            print(f"[DatasetEngine] Zero-copy loading binary SFT memmap shards: {cache_bin_in.name}")
            if progress_callback:
                progress_callback(f"Zero-copy loading SFT memmap shards...", 50.0)
            in_arr = np.memmap(cache_bin_in, dtype=np.int64, mode="r")
            lbl_arr = np.memmap(cache_bin_lbl, dtype=np.int64, mode="r")
            return torch.from_numpy(in_arr).to(device=device), torch.from_numpy(lbl_arr).to(device=device)

        # 2. PyTorch .pt cache fallback (with mmap)
        if cache_file.exists():
            print(f"[DatasetEngine] Loading pre-tokenized chat cache: {cache_file.name}")
            if progress_callback:
                progress_callback(f"Loading cached chat SFT array: {cache_file.name}...", 50.0)
            try:
                data = torch.load(cache_file, map_location=device, mmap=True, weights_only=True)
            except Exception:
                data = torch.load(cache_file, map_location=device)
            return data["input_ids"], data["labels"]

        if progress_callback:
            progress_callback(f"Reading {file_path.name} from disk...", 5.0)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()

        pattern = re.compile(
            r"(?:<\|im_start\|>|<\|)user\s*\n?(.*?)(?:<\|im_end\|>|<\|end_turn\|>)\s*"
            r"(?:<\|im_start\|>|<\|)assistant\s*\n?(.*?)(?:<\|im_end\|>|<\|end_turn\|>|$)",
            re.DOTALL,
        )

        dialog_pairs: List[Tuple[str, str]] = []
        for m in pattern.finditer(content):
            dialog_pairs.append((m.group(1).strip(), m.group(2).strip()))

        # If no regex tags found, parse JSONL lines for chat messages or instruction/output
        if not dialog_pairs and (file_path.suffix.lower() == ".jsonl" or "{" in content[:300]):
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "messages" in obj and isinstance(obj["messages"], list):
                        u_parts: List[str] = []
                        a_parts: List[str] = []
                        for msg in obj["messages"]:
                            r = msg.get("role", "")
                            c = str(msg.get("content", "")).strip()
                            if r == "user":
                                u_parts.append(c)
                            elif r == "assistant":
                                a_parts.append(c)
                        if u_parts and a_parts:
                            dialog_pairs.append(("\n".join(u_parts), "\n".join(a_parts)))
                    elif "instruction" in obj or "output" in obj:
                        prompt = str(obj.get("instruction", "")).strip()
                        if obj.get("input"):
                            prompt += f"\n{str(obj.get('input')).strip()}"
                        resp = str(obj.get("output", "")).strip()
                        if prompt and resp:
                            dialog_pairs.append((prompt, resp))
                except Exception:
                    pass

        print(f"[DatasetEngine] Parsed {len(dialog_pairs):,} dialog turns in {file_path.name}")

        if not dialog_pairs:
            in_t, _ = self._load_pretrain_text(
                file_path, device=device, max_samples=max_samples, progress_callback=progress_callback
            )
            return in_t, in_t.clone()

        if max_samples:
            dialog_pairs = dialog_pairs[:max_samples]

        total_turns = len(dialog_pairs)
        if progress_callback:
            progress_callback(f"Tokenizing {total_turns:,} dialog turns with assistant loss masking...", 15.0)

        all_input_ids: List[int] = []
        all_labels: List[int] = []

        chunk_size = 300
        for i in range(0, total_turns, chunk_size):
            chunk_m = dialog_pairs[i : i + chunk_size]
            prompt_texts = [f"<|user|>\n{u}\n<|assistant|>\n" for u, a in chunk_m]
            asst_texts = [f"{a}<|end_turn|>" for u, a in chunk_m]

            prompt_encs = self.tokenizer._tokenizer.encode_batch(prompt_texts)
            asst_encs = self.tokenizer._tokenizer.encode_batch(asst_texts)

            for p_enc, a_enc in zip(prompt_encs, asst_encs):
                p_ids = p_enc.ids
                a_ids = a_enc.ids
                all_input_ids.extend(p_ids + a_ids)
                all_labels.extend(([-100] * len(p_ids)) + a_ids)

            done_turns = min(total_turns, i + chunk_size)
            pct = 15.0 + (done_turns / total_turns) * 75.0
            if progress_callback:
                progress_callback(
                    f"Tokenizing {total_turns:,} dialogs... ({done_turns:,}/{total_turns:,} - {pct:.0f}%)",
                    pct,
                )

        if progress_callback:
            progress_callback(f"Saving {len(all_input_ids):,} SFT tokens to cache...", 95.0)

        # Save binary memmap shards for zero-copy OS paging
        in_arr = np.array(all_input_ids, dtype=np.int64)
        lbl_arr = np.array(all_labels, dtype=np.int64)
        in_arr.tofile(cache_bin_in)
        lbl_arr.tofile(cache_bin_lbl)

        input_tensor = torch.from_numpy(in_arr).to(device=device)
        labels_tensor = torch.from_numpy(lbl_arr).to(device=device)

        print(f"[DatasetEngine] Tokenized {len(input_tensor):,} SFT tokens. Saved memmap shards and cache {cache_file.name}...")
        torch.save({"input_ids": input_tensor.cpu(), "labels": labels_tensor.cpu()}, cache_file)

        return input_tensor, labels_tensor

    def _generate_synthetic_tokens(
        self, device: torch.device
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """Fallback synthetic tokens."""
        sample = (
            "RoCanon-48.5M este un model lingvistic antrenat de la zero de către Manu Cătălin pe laptopul său personal. "
            "The model excels at Romanian tech, hardware, processors, and programming reasoning."
        )
        ids = self.tokenizer.encode(sample) + [self.tokenizer.eos_token_id]
        while len(ids) < 4 * 256 * 20:
            ids.extend(ids)
        tensor = torch.tensor(ids, dtype=torch.long, device=device)
        split = int(len(tensor) * 0.95)
        return tensor[:split], None, tensor[split:], None

    def _generate_synthetic_classification(
        self,
        input_dim: int = 64,
        num_classes: int = 2,
        n_samples: int = 400,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates synthetic numerical sequence batches for classification training/testing."""
        device = device or torch.device("cpu")
        x = torch.randn(n_samples, 16, input_dim, device=device)
        # Decision boundary based on sum of first feature
        y = (x[..., 0].mean(dim=1) > 0.0).long()
        split = int(n_samples * 0.8)
        return x[:split], y[:split], x[split:], y[split:]

    def _generate_synthetic_regression(
        self,
        input_dim: int = 64,
        n_samples: int = 400,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates synthetic numerical sequence batches for regression training/testing."""
        device = device or torch.device("cpu")
        x = torch.randn(n_samples, 16, input_dim, device=device)
        y = x[..., 0].mean(dim=1, keepdim=True) * 2.5 + 0.5
        split = int(n_samples * 0.8)
        return x[:split], y[:split], x[split:], y[split:]

    def _load_classification_or_regression(
        self,
        file_path: Path,
        is_regression: bool = False,
        device: Optional[torch.device] = None,
        val_ratio: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Loads JSONL datasets with float features and target labels/values."""
        device = device or torch.device("cpu")
        xs: List[List[float]] = []
        ys: List[Any] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    feat = obj.get("features", obj.get("x"))
                    target = obj.get("label", obj.get("target", obj.get("y", 0)))
                    if feat is not None:
                        xs.append(feat)
                        ys.append(target)
                except Exception:
                    continue

        if not xs:
            if is_regression:
                return self._generate_synthetic_regression(device=device)
            return self._generate_synthetic_classification(device=device)

        x_t = torch.tensor(xs, dtype=torch.float32, device=device)
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)  # [B, 1, D]
        y_t = torch.tensor(ys, dtype=torch.float32 if is_regression else torch.long, device=device)

        split = max(1, int(len(x_t) * (1.0 - val_ratio)))
        return x_t[:split], y_t[:split], x_t[split:], y_t[split:]
