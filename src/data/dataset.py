"""Flat uint16 binary serialization, padding-free sequence packing, and zero-copy

memmap dataset for RoCanon-48.5M Bilingual SLM.

Provides:
- Flat uint16 binary shard serialization (.bin) and document boundary index (.idx).
- Sequence packing without <|pad|> padding using <|eos|> delimiters into chunks of seq_len.
- Generation of cu_seqlens: 1D torch.int32 strictly monotonic vector [0, L1, L1+L2, ..., seq_len]
  starting at 0 and ending at seq_len for FlashAttention-2 / SDPA block-diagonal isolation.
- PackedMemmapDataset: Zero-copy memory-mapped dataset reading uint16 flat binaries with
  minimal memory footprint (4-6 GB RAM), returning dict supporting tuple unpacking:
  ('input_ids', 'labels', 'cu_seqlens').
- Token-mean loss normalization helper.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Self

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedBatch(dict):
    """Dictionary container holding 'input_ids', 'labels', 'cu_seqlens'

    that supports both dictionary access and 3-tuple unpacking:
    `input_ids, labels, cu_seqlens = batch`
    """

    def __iter__(self):
        yield self["input_ids"]
        yield self["labels"]
        yield self["cu_seqlens"]


def serialize_tokens_to_shard(
    token_ids_iterable: Iterable[list[int]],
    bin_path: str,
    idx_path: str,
) -> int:
    """Serializes token lists to contiguous flat uint16 bin file and int64 index file.

    Returns total number of tokens written.
    """
    all_tokens: list[int] = []
    offsets: list[int] = [0]
    curr_offset = 0

    for doc_ids in token_ids_iterable:
        if not doc_ids:
            continue
        all_tokens.extend(doc_ids)
        curr_offset += len(doc_ids)
        offsets.append(curr_offset)

    np.array(all_tokens, dtype=np.uint16).tofile(bin_path)
    np.array(offsets, dtype=np.int64).tofile(idx_path)
    return len(all_tokens)


def create_packed_shards(
    tokenizer: Any,
    documents: Iterable[str],
    output_bin: str,
    output_idx: str,
    seq_len: int = 1024,
) -> int:
    """Tokenizes documents, appends <|eos|>, packs without padding into multiples of seq_len,

    and writes contiguous uint16 bin and int64 idx shards.

    Returns the total number of usable packed tokens written.
    """
    eos_id = getattr(tokenizer, "eos_token_id", 1)
    all_tokens: list[int] = []
    offsets: list[int] = [0]
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

    # Ensure strictly increasing offsets without duplicate entries
    dedup_offsets = sorted(set(valid_offsets))
    np.array(dedup_offsets, dtype=np.int64).tofile(output_idx)

    return len(usable_tokens)


def pack_documents(
    tokenizer: Any,
    documents: Iterable[str],
    output_bin: str,
    output_idx: str,
    seq_len: int = 1024,
) -> int:
    """Alias for create_packed_shards."""
    return create_packed_shards(
        tokenizer=tokenizer,
        documents=documents,
        output_bin=output_bin,
        output_idx=output_idx,
        seq_len=seq_len,
    )


def pack_sequences(
    tokenized_docs: Iterable[list[int]],
    seq_len: int = 1024,
    eos_token_id: int = 1,
) -> Iterator[tuple[list[int], list[int]]]:
    """Packs documents into exact chunks of seq_len without padding using eos_token_id delimiters.

    Yields (packed_tokens, doc_boundaries) where doc_boundaries is [0, len1, ..., seq_len].
    """
    buffer: list[int] = []
    boundaries: list[int] = [0]

    for doc in tokenized_docs:
        if not doc:
            continue
        tokens = list(doc)
        if tokens[-1] != eos_token_id:
            tokens.append(eos_token_id)

        remaining_tokens = tokens
        while remaining_tokens:
            space_left = seq_len - len(buffer)
            if len(remaining_tokens) <= space_left:
                buffer.extend(remaining_tokens)
                boundaries.append(len(buffer))
                remaining_tokens = []
                if len(buffer) == seq_len:
                    yield (list(buffer), sorted(set(boundaries)))
                    buffer = []
                    boundaries = [0]
            else:
                # Slice portion to fill the current chunk
                buffer.extend(remaining_tokens[:space_left])
                remaining_tokens = remaining_tokens[space_left:]
                boundaries.append(seq_len)
                yield (list(buffer), sorted(set(boundaries)))
                buffer = []
                boundaries = [0]


def build_cu_seqlens(doc_lengths: Sequence[int]) -> torch.Tensor:
    """Constructs 1D int32 cu_seqlens cumulative vector from document segment lengths."""
    cum = [0] + list(np.cumsum(doc_lengths))
    return torch.tensor(cum, dtype=torch.int32)


class PackedMemmapDataset(Dataset):
    """Zero-copy memory-mapped dataset for packed uint16 token sequences.

    Features:
    - High-throughput zero-copy reading directly via np.memmap.
    - Low memory footprint (< 4-6 GB RAM) independent of corpus size.
    - Emits input_ids [seq_len], labels [seq_len], and strictly monotonic cu_seqlens [num_docs + 1].
    - Item is a PackedBatch (dict subclass) compatible with both dict access and tuple unpacking.
    """

    def __init__(
        self,
        bin_path: str,
        idx_path: str | None = None,
        seq_len: int = 1024,
        eos_token_id: int = 1,
    ):
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
        self.eos_token_id = eos_token_id
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.num_samples = len(self.data) // seq_len

        if idx_path and os.path.exists(idx_path) and os.path.getsize(idx_path) > 0:
            self.idx = np.fromfile(idx_path, dtype=np.int64)
        else:
            self.idx = None

    def close(self) -> None:
        """Closes memory map handle (ensuring file handles can be released on Windows)."""
        if hasattr(self, "data") and self.data is not None:
            try:
                if hasattr(self.data, "_mmap") and self.data._mmap is not None:
                    self.data._mmap.close()
            except (AttributeError, OSError):
                pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> PackedBatch:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(
                f"Index {idx} out of bounds for dataset of length {self.num_samples}"
            )

        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len

        tokens_np = self.data[start_idx:end_idx].astype(np.int64)
        input_ids = torch.from_numpy(tokens_np).long()
        labels = input_ids.clone()

        if self.idx is not None and len(self.idx) > 0:
            sub_offsets = [
                int(off - start_idx) for off in self.idx if start_idx <= off <= end_idx
            ]
            if not sub_offsets or sub_offsets[0] != 0:
                sub_offsets.insert(0, 0)
            if sub_offsets[-1] != self.seq_len:
                sub_offsets.append(self.seq_len)
            unique_sorted = sorted(set(sub_offsets))
        else:
            # Fallback to finding <|eos|> tokens
            eos_positions = (tokens_np == self.eos_token_id).nonzero()[0] + 1
            offsets = [0] + eos_positions.tolist() + [self.seq_len]
            unique_sorted = sorted(set(offsets))

        cu_seqlens = torch.tensor(unique_sorted, dtype=torch.int32)

        return PackedBatch(
            {
                "input_ids": input_ids,
                "labels": labels,
                "cu_seqlens": cu_seqlens,
            }
        )


def compute_token_mean_loss(
    loss: torch.Tensor,
    valid_tokens: int | torch.Tensor | None = None,
    ignore_index: int = -100,
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Computes token-mean normalized loss across globally accumulated tokens.

    If loss is unreduced (per-token), normalizes by valid token count.
    """
    if valid_tokens is not None:
        if isinstance(valid_tokens, torch.Tensor):
            n_tokens = valid_tokens.item()
        else:
            n_tokens = float(valid_tokens)
        if n_tokens > 0:
            return loss.sum() / n_tokens
        return loss.sum()

    if labels is not None:
        valid_mask = labels != ignore_index
        n_valid = valid_mask.sum().item()
        if n_valid > 0:
            return loss.sum() / n_valid
        return loss.sum()

    if loss.numel() > 0:
        return loss.mean()

    return loss
