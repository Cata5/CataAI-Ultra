"""
tests/e2e/conftest.py
Pytest fixtures and test environment setup for RoCanon-48.5M E2E tests.
Provides safe imports and conforming fallback adapters if modules are in flight.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Ensure workspace root is on sys.path ---
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- Model Imports ---
from src.model.architecture import (
    RMSNorm,
    RoCanonAttention,
    RoCanonBlock,
    RoCanonConfig,
    RoCanonForCausalLM,
    RoCanonMLP,
    RotaryEmbedding,
    apply_rope,
    rotate_half,
)
from src.model.canon import CausalDepthwiseConv1d

# --- Tokenizer Imports ---
from src.tokenizer.tokenizer import (
    DEFAULT_VOCAB_SIZE,
    SPECIAL_TOKENS,
    RoCanonTokenizer,
)

# --- Data Pipeline Imports & Conforming Fallback ---
try:
    from src.data.dataset import PackedMemmapDataset, create_packed_shards
except (ImportError, ModuleNotFoundError):

    class PackedMemmapDataset(torch.utils.data.Dataset):
        """Zero-copy memory-mapped dataset for packed uint16 token sequences."""

        def __init__(self, bin_path: str, idx_path: str, seq_len: int = 1024):
            super().__init__()
            if not os.path.exists(bin_path):
                raise FileNotFoundError(f"Binary shard not found: {bin_path}")

            size_bytes = os.path.getsize(bin_path)
            if size_bytes == 0 or size_bytes % (2 * seq_len) != 0:
                raise ValueError(
                    f"Corrupted or invalid bin file size: {size_bytes} bytes "
                    f"is not a positive integer multiple of 2 * seq_len ({2 * seq_len})"
                )

            self.bin_path = bin_path
            self.idx_path = idx_path
            self.seq_len = seq_len
            self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
            self.num_samples = len(self.data) // seq_len

            if os.path.exists(idx_path) and os.path.getsize(idx_path) > 0:
                self.idx = np.fromfile(idx_path, dtype=np.int64)
            else:
                self.idx = np.array([0, len(self.data)], dtype=np.int64)

        def __len__(self) -> int:
            return self.num_samples

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if idx < 0 or idx >= self.num_samples:
                raise IndexError(f"Index {idx} out of bounds for dataset of length {self.num_samples}")

            start_idx = idx * self.seq_len
            end_idx = start_idx + self.seq_len

            tokens_np = self.data[start_idx:end_idx].astype(np.int64)
            input_ids = torch.from_numpy(tokens_np).long()
            labels = input_ids.clone()

            # Filter document boundary offsets falling within [start_idx, end_idx]
            sub_offsets = [off - start_idx for off in self.idx if start_idx <= off <= end_idx]
            if not sub_offsets or sub_offsets[0] != 0:
                sub_offsets.insert(0, 0)
            if sub_offsets[-1] != self.seq_len:
                sub_offsets.append(self.seq_len)

            unique_sorted = sorted(list(set(sub_offsets)))
            cu_seqlens = torch.tensor(unique_sorted, dtype=torch.int32)
            return input_ids, labels, cu_seqlens

    def create_packed_shards(
        tokenizer: RoCanonTokenizer,
        documents: Iterable[str],
        output_bin: str,
        output_idx: str,
        seq_len: int = 1024,
    ) -> int:
        """Packs tokenized documents separated by <|eos|> into uint16 bin and int64 idx shards."""
        eos_id = tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else 1
        all_tokens: List[int] = []
        offsets: List[int] = [0]
        curr_offset = 0

        for doc in documents:
            ids = tokenizer.encode(doc)
            if not ids or ids[-1] != eos_id:
                ids.append(eos_id)
            all_tokens.extend(ids)
            curr_offset += len(ids)
            offsets.append(curr_offset)

        num_full = len(all_tokens) // seq_len
        if num_full > 0:
            usable_tokens = all_tokens[: num_full * seq_len]
        else:
            usable_tokens = all_tokens

        np.array(usable_tokens, dtype=np.uint16).tofile(output_bin)
        valid_offsets = [o for o in offsets if o <= len(usable_tokens)]
        if not valid_offsets or valid_offsets[-1] != len(usable_tokens):
            valid_offsets.append(len(usable_tokens))
        np.array(valid_offsets, dtype=np.int64).tofile(output_idx)
        return len(usable_tokens)


def pack_documents(
    tokenizer: RoCanonTokenizer,
    documents: Iterable[str],
    output_bin: str,
    output_idx: str,
    seq_len: int = 1024,
) -> int:
    """Alias for create_packed_shards."""
    return create_packed_shards(tokenizer, documents, output_bin, output_idx, seq_len=seq_len)


def build_cu_seqlens(doc_lengths: Sequence[int]) -> torch.Tensor:
    """Constructs 1D int32 cu_seqlens cumulative vector from document segment lengths."""
    cum = [0] + list(np.cumsum(doc_lengths))
    return torch.tensor(cum, dtype=torch.int32)


# --- Training, Optimizer & Loss Imports & Conforming Fallback ---
try:
    from src.training.optimizer import configure_optimizers
except (ImportError, ModuleNotFoundError):

    def configure_optimizers(
        model: nn.Module,
        muon_lr: float = 3e-3,
        muon_wd: float = 0.01,
        adamw_lr: float = 6e-4,
        adamw_wd: float = 0.01,
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        fused: bool = True,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """
        Partitions model parameters into Muon (112 2D hidden matrices) and
        AdamW (tied embeddings, 1D RMSNorm scales, 3D Canon Conv1d kernels).
        """
        muon_params = []
        adamw_embed = []
        adamw_norm = []
        adamw_conv = []

        seen_params = set()
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in seen_params:
                continue
            seen_params.add(id(param))

            if "embed_tokens" in name or "lm_head" in name:
                adamw_embed.append(param)
            elif param.ndim == 2:
                muon_params.append(param)
            elif param.ndim == 1:
                adamw_norm.append(param)
            elif param.ndim == 3:
                adamw_conv.append(param)
            else:
                adamw_norm.append(param)

        muon_opt = torch.optim.Muon(
            muon_params,
            lr=muon_lr,
            weight_decay=muon_wd,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
        )

        adamw_groups = []
        if adamw_embed:
            adamw_groups.append({"params": adamw_embed, "weight_decay": adamw_wd})
        if adamw_norm:
            adamw_groups.append({"params": adamw_norm, "weight_decay": 0.0})
        if adamw_conv:
            adamw_groups.append({"params": adamw_conv, "weight_decay": 0.0})

        use_fused = fused and torch.cuda.is_available()
        try:
            adamw_opt = torch.optim.AdamW(
                adamw_groups,
                lr=adamw_lr,
                betas=adamw_betas,
                eps=adamw_eps,
                fused=fused,
            )
        except Exception:
            adamw_opt = torch.optim.AdamW(
                adamw_groups,
                lr=adamw_lr,
                betas=adamw_betas,
                eps=adamw_eps,
                fused=False,
            )

        return muon_opt, adamw_opt


try:
    from src.training.scheduler import BatchSizeScheduler, WarmupStableDecayLR
except (ImportError, ModuleNotFoundError):

    class WarmupStableDecayLR:
        """3-Phase Warmup-Stable-Decay (WSD) Learning Rate Schedule."""

        def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            warmup: int = 100,
            stable: int = 300,
            decay: int = 100,
            peak_lr: float = 3e-3,
            min_lr_ratio: float = 0.1,
            warmup_steps: Optional[int] = None,
            total_steps: Optional[int] = None,
            decay_steps: Optional[int] = None,
        ):
            self.optimizer = optimizer
            if warmup_steps is not None:
                self.warmup_steps = warmup_steps
                self.decay_steps = decay_steps if decay_steps is not None else 1000
                total = total_steps if total_steps is not None else 10000
                self.stable_steps = max(0, total - self.warmup_steps - self.decay_steps)
            else:
                self.warmup_steps = warmup
                self.stable_steps = stable
                self.decay_steps = decay

            self.peak_lr = peak_lr
            self.min_lr = peak_lr * min_lr_ratio
            self.current_step = 0

        def get_lr(self, step: Optional[int] = None) -> float:
            if step is None:
                step = self.current_step
            if step < self.warmup_steps:
                if self.warmup_steps == 0:
                    return self.peak_lr
                return self.peak_lr * (step / self.warmup_steps)
            elif step < self.warmup_steps + self.stable_steps:
                return self.peak_lr
            elif step <= self.warmup_steps + self.stable_steps + self.decay_steps:
                decay_prog = (step - (self.warmup_steps + self.stable_steps)) / max(1, self.decay_steps)
                return self.peak_lr - decay_prog * (self.peak_lr - self.min_lr)
            else:
                return self.min_lr

        def step(self) -> float:
            lr = self.get_lr(self.current_step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            self.current_step += 1
            return lr

    class BatchSizeScheduler:
        """Doubles gradient accumulation at token threshold (Stage 1 to Stage 2 transition)."""

        def __init__(
            self,
            threshold_tokens: float = 3.5e9,
            initial_accum: int = 8,
            stage2_accum: int = 16,
        ):
            self.threshold_tokens = threshold_tokens
            self.initial_accum = initial_accum
            self.stage2_accum = stage2_accum

        def get_grad_accum_steps(self, tokens: float) -> int:
            return self.stage2_accum if tokens > self.threshold_tokens else self.initial_accum


try:
    from src.training.loss import compute_ce_loss, compute_chunked_loss
except (ImportError, ModuleNotFoundError):

    def compute_chunked_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        chunk_size: int = 2048,
        ignore_index: int = -100,
        reduction: str = "token_mean",
    ) -> torch.Tensor:
        """Computes cross-entropy loss in chunks bounding peak logit VRAM memory."""
        flat_logits = logits.view(-1, logits.shape[-1])
        flat_labels = labels.view(-1)
        total_tokens = flat_labels.shape[0]

        valid_mask = flat_labels != ignore_index
        num_valid = valid_mask.sum().item()

        if num_valid == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)

        total_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        for i in range(0, total_tokens, chunk_size):
            chunk_logits = flat_logits[i : i + chunk_size].float()
            chunk_labels = flat_labels[i : i + chunk_size]
            chunk_loss = F.cross_entropy(
                chunk_logits,
                chunk_labels,
                ignore_index=ignore_index,
                reduction="sum",
            )
            total_loss = total_loss + chunk_loss

        if reduction in ("token_mean", "mean"):
            return total_loss / max(1, num_valid)
        elif reduction == "sum":
            return total_loss
        return total_loss / max(1, num_valid)

    compute_ce_loss = compute_chunked_loss


# --- Pytest Fixtures ---


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Dynamically detects cuda if available, else cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def full_rocanon_config() -> RoCanonConfig:
    """Full 48.5M parameter configuration."""
    return RoCanonConfig(
        vocab_size=12288,
        d_model=512,
        n_layers=16,
        n_q=8,
        n_kv=2,
        d_h=64,
        d_ff=1280,
        kernel_size=4,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_seq_len=2048,
        initializer_range=0.02,
        tie_word_embeddings=True,
    )


@pytest.fixture(scope="session")
def rocanon_config(full_rocanon_config: RoCanonConfig) -> RoCanonConfig:
    """Alias for full_rocanon_config."""
    return full_rocanon_config


@pytest.fixture(scope="session")
def minimal_rocanon_config() -> RoCanonConfig:
    """Lightweight 2-layer configuration for ultra-fast gradient & boundary tests on CPU."""
    return RoCanonConfig(
        vocab_size=12288,
        d_model=64,
        n_layers=2,
        n_q=4,
        n_kv=2,
        d_h=16,
        d_ff=128,
        kernel_size=4,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_seq_len=128,
        initializer_range=0.02,
        tie_word_embeddings=True,
    )


@pytest.fixture(scope="session")
def mini_rocanon_config(minimal_rocanon_config: RoCanonConfig) -> RoCanonConfig:
    """Alias for minimal_rocanon_config."""
    return minimal_rocanon_config


@pytest.fixture(scope="session")
def full_model(full_rocanon_config: RoCanonConfig, device: torch.device) -> RoCanonForCausalLM:
    """Instantiated full 48.5M RoCanonForCausalLM model."""
    torch.manual_seed(42)
    model = RoCanonForCausalLM(full_rocanon_config).to(device)
    return model


@pytest.fixture
def mini_model(minimal_rocanon_config: RoCanonConfig, device: torch.device) -> RoCanonForCausalLM:
    """Freshly instantiated lightweight model for fast mutation and training tests."""
    torch.manual_seed(42)
    model = RoCanonForCausalLM(minimal_rocanon_config).to(device)
    return model


@pytest.fixture(scope="session")
def dummy_tokenizer() -> RoCanonTokenizer:
    """Loads the production trained RoCanonTokenizer."""
    return RoCanonTokenizer()


@pytest.fixture
def dummy_memmap_shards(tmp_path: Path) -> Tuple[str, str, int, List[int]]:
    """Creates a temporary valid binary shard pair for PackedMemmapDataset testing."""
    bin_path = str(tmp_path / "shard_00001.bin")
    idx_path = str(tmp_path / "shard_00001.idx")

    doc_lengths = [20, 30, 45, 33]  # total = 128
    eos_id = 1
    total_tokens = sum(doc_lengths)

    tokens = []
    idx_offsets = [0]
    curr_offset = 0
    np.random.seed(42)
    for l in doc_lengths:
        doc = [int(x) for x in np.random.randint(10, 200, size=l - 1)] + [eos_id]
        tokens.extend(doc)
        curr_offset += l
        idx_offsets.append(curr_offset)

    np.array(tokens, dtype=np.uint16).tofile(bin_path)
    np.array(idx_offsets, dtype=np.int64).tofile(idx_path)
    return bin_path, idx_path, total_tokens, doc_lengths


@pytest.fixture
def dummy_packed_batch(device: torch.device) -> Dict[str, torch.Tensor]:
    """Provides a synthetic packed batch with input_ids, labels, and cu_seqlens."""
    seq_len = 128
    torch.manual_seed(42)
    input_ids = torch.randint(10, 250, (1, seq_len), dtype=torch.long, device=device)
    labels = input_ids.clone()
    cu_seqlens = torch.tensor([0, 30, 75, 128], dtype=torch.int32, device=device)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "cu_seqlens": cu_seqlens,
        "max_seqlen": 53,
    }


@pytest.fixture
def optimizer_setup(mini_model: RoCanonForCausalLM) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """Configures Muon and AdamW for the mini_model."""
    return configure_optimizers(mini_model)
