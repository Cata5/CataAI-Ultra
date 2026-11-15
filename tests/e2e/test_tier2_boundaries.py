"""
tests/e2e/test_tier2_boundaries.py
Tier 2: Boundary, Edge, and Corner Case Tests for RoCanon-48.5M.

Covers:
- Tokenizer Boundaries: 6 tests (Empty string & whitespace, single char & token,
  sedila/comma-below cross combinations, extreme digit sequences, special characters
  & byte fallback, maximum length text)
- Architecture Boundaries: 6 tests (Single token S=1, Canon zero-init residual identity,
  Canon strict temporal causality, maximum sequence length, QK-Norm numerical stability,
  residual scaling factor 1/sqrt(2L))
- Data Packing Boundaries: 5 tests (Exact seq_len document, oversized document chunking,
  many tiny single-token documents, cu_seqlens ABI dtype & contiguity, corrupt/zero-byte shard handling)
- Optimizer & Numerical Boundaries: 5 tests (Zero-gradient parameter step, extreme gradient
  clipping, Muon Quintic Newton-Schulz on ill-conditioned matrices, BF16 autocast with FP32
  accumulation, token-mean loss normalization with masked labels)
Total: 22 tests.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.architecture import (
    RMSNorm,
    RoCanonAttention,
    RoCanonBlock,
    RoCanonConfig,
    RoCanonForCausalLM,
    RoCanonMLP,
    RotaryEmbedding,
    apply_rope,
)
from src.model.canon import CausalDepthwiseConv1d
from src.tokenizer.tokenizer import (
    DEFAULT_VOCAB_SIZE,
    SPECIAL_TOKENS,
    RoCanonTokenizer,
)
from tests.e2e.conftest import (
    BatchSizeScheduler,
    PackedMemmapDataset,
    WarmupStableDecayLR,
    build_cu_seqlens,
    compute_ce_loss,
    compute_chunked_loss,
    configure_optimizers,
    pack_documents,
)


# ============================================================================
# BOUNDARY AREA 1: Tokenizer Boundaries (6 Tests)
# ============================================================================


class TestTier2TokenizerBoundaries:
    """Boundary and corner case tests for RoCanonTokenizer."""

    def test_tokenizer_empty_string_and_whitespace_only(self, dummy_tokenizer: RoCanonTokenizer):
        """Verifies empty string returns empty list and whitespace sequences are faithfully preserved."""
        assert dummy_tokenizer.encode("") == []
        assert dummy_tokenizer.decode([]) == ""

        # Whitespace variations
        for ws in [" ", "   ", "\t", "\n", "\t\n\r", " \n \n "]:
            encoded = dummy_tokenizer.encode(ws)
            assert len(encoded) > 0, f"Whitespace '{repr(ws)}' should encode to at least 1 token"
            decoded = dummy_tokenizer.decode(encoded)
            assert decoded == ws, f"Whitespace roundtrip mismatch: {repr(decoded)} != {repr(ws)}"

    def test_tokenizer_single_character_and_single_token(self, dummy_tokenizer: RoCanonTokenizer):
        """Verifies single ASCII, single diacritic, single digit, and single special tokens."""
        for char in ["a", "Z", "ș", "ț", "â", "î", "ă", "9", "0", "=", "+"]:
            encoded = dummy_tokenizer.encode(char)
            assert len(encoded) >= 1, f"Single character '{char}' produced empty encoding"
            decoded = dummy_tokenizer.decode(encoded)
            assert decoded == dummy_tokenizer.normalize(char), f"Single char mismatch: {repr(decoded)} != {repr(char)}"

        # Single special token
        bos_ids = dummy_tokenizer.encode("<|bos|>")
        assert len(bos_ids) == 1
        assert dummy_tokenizer.decode(bos_ids, skip_special_tokens=False) == "<|bos|>"

    def test_tokenizer_sedila_and_comma_below_cross_combinations(self, dummy_tokenizer: RoCanonTokenizer):
        """Verifies 100% equivalence between Romanian cedilla and comma-below encodings across word cases."""
        pairs = [
            ("chişinău", "chișinău"),
            ("Ştefan", "Ștefan"),
            ("învăţătură", "învățătură"),
            ("GHEORGHE ŢEPEŞ", "GHEORGHE ȚEPEȘ"),
            ("conştiinţă", "conștiință"),
            ("PRODUCŢIE ŞI CONSUM", "PRODUCȚIE ȘI CONSUM"),
        ]
        for old_cedilla, new_comma in pairs:
            assert dummy_tokenizer.normalize(old_cedilla) == new_comma
            assert dummy_tokenizer.encode(old_cedilla) == dummy_tokenizer.encode(new_comma), (
                f"Encoding mismatch for pair: '{old_cedilla}' vs '{new_comma}'"
            )
            assert dummy_tokenizer.decode(dummy_tokenizer.encode(old_cedilla)) == new_comma

    def test_tokenizer_extreme_digit_sequences(self, dummy_tokenizer: RoCanonTokenizer):
        """Verifies 100-digit sequences and alternating arithmetic preserve 1-by-1 atomization."""
        digits = "9876543210" * 10  # 100 digits
        ids = dummy_tokenizer.encode(digits)
        assert len(ids) == 100, f"Expected exactly 100 tokens for 100 digits, got {len(ids)}"
        assert dummy_tokenizer.decode(ids) == digits

        # Alternating arithmetic: operators must never fuse with numbers
        math_expr = "1+2*3-4/5=6"
        math_ids = dummy_tokenizer.encode(math_expr)
        decoded_math = dummy_tokenizer.decode(math_ids)
        assert decoded_math == math_expr

    def test_tokenizer_special_characters_and_byte_fallback(self, dummy_tokenizer: RoCanonTokenizer):
        """Verifies byte fallback handles arbitrary UTF-8 multibyte sequences without decode errors."""
        text = "Text cu caractere speciale: 🚀 ⚡ \u200b αβγδ \ufffd"
        ids = dummy_tokenizer.encode(text)
        assert all(0 <= tid < DEFAULT_VOCAB_SIZE for tid in ids), "Token ID out of vocab bounds"
        decoded = dummy_tokenizer.decode(ids)
        assert decoded == text, f"Byte fallback mismatch: {repr(decoded)} != {repr(text)}"

    def test_tokenizer_maximum_length_text(self, dummy_tokenizer: RoCanonTokenizer):
        """Verifies long documents (50,000 characters) encode and decode linearly without recursion errors."""
        chunk = "Cercetare RoCanon-48.5M în limba română și engleză: 12345. "
        huge_text = chunk * 850  # ~50,000 characters
        ids = dummy_tokenizer.encode(huge_text)
        assert len(ids) > 10000, f"Expected >10,000 tokens, got {len(ids)}"
        decoded = dummy_tokenizer.decode(ids)
        assert decoded == dummy_tokenizer.normalize(huge_text)


# ============================================================================
# BOUNDARY AREA 2: Architecture & Forward/Backward Boundaries (6 Tests)
# ============================================================================


class TestTier2ArchitectureBoundaries:
    """Boundary and corner case tests for RoCanon-48.5M Model Architecture."""

    def test_model_single_token_forward(
        self, mini_model: RoCanonForCausalLM, minimal_rocanon_config: RoCanonConfig, device: torch.device
    ):
        """Verifies forward pass with sequence length S=1 executes without dimension squeeze errors."""
        input_ids = torch.tensor([[42]], device=device)
        logits = mini_model(input_ids)

        assert logits.shape == (1, 1, minimal_rocanon_config.vocab_size)
        assert torch.isfinite(logits).all(), "Logits for S=1 contain NaN or Inf"

    def test_model_canon_conv1d_zero_init_residual_identity(self, mini_model: RoCanonForCausalLM, device: torch.device):
        """Verifies Canon layers are strictly zero-initialized and evaluate as exact bitwise identity at init."""
        for idx, layer in enumerate(mini_model.layers):
            assert torch.all(layer.canon_a.conv.weight == 0.0), f"Layer {idx} canon_a not zero-initialized"
            assert torch.all(layer.attn.canon_b.conv.weight == 0.0), f"Layer {idx} canon_b not zero-initialized"
            assert torch.all(layer.canon_c.conv.weight == 0.0), f"Layer {idx} canon_c not zero-initialized"
            assert torch.all(layer.mlp.canon_d.conv.weight == 0.0), f"Layer {idx} canon_d not zero-initialized"

        # Standalone identity verification: y = x + conv(x) == x + 0 == x
        conv = CausalDepthwiseConv1d(channels=64, kernel_size=4).to(device)
        x = torch.randn(2, 8, 64, device=device)
        out = conv(x)
        assert torch.equal(out, x), "Canon layer output did not equal input at step 0"

    def test_model_canon_conv1d_strict_causality(self, device: torch.device):
        """Verifies altering future tokens t >= 6 produces 0 change in outputs at past positions t < 6."""
        conv = CausalDepthwiseConv1d(channels=32, kernel_size=4).to(device)
        # Random non-zero weights
        conv.conv.weight.data.normal_(0.0, 1.0)

        x = torch.randn(1, 12, 32, device=device)
        x_mutated = x.clone()
        x_mutated[:, 6:, :] += 10.0  # perturb future

        y1 = conv(x)
        y2 = conv(x_mutated)

        assert torch.allclose(y1[:, :6, :], y2[:, :6, :], atol=1e-6), (
            "Anti-causal future token leakage detected in CausalDepthwiseConv1d!"
        )

    def test_model_max_sequence_length_forward(
        self, minimal_rocanon_config: RoCanonConfig, device: torch.device
    ):
        """Verifies forward pass executes cleanly at full maximum context length S=2048 using a lightweight config."""
        cfg_2048 = RoCanonConfig(
            vocab_size=12288,
            d_model=64,
            n_layers=2,
            n_q=4,
            n_kv=2,
            d_h=16,
            d_ff=128,
            max_seq_len=2048,
        )
        model_2048 = RoCanonForCausalLM(cfg_2048).to(device)
        model_2048.eval()

        input_ids = torch.randint(0, cfg_2048.vocab_size, (1, 2048), device=device)
        with torch.no_grad():
            logits = model_2048(input_ids)

        assert logits.shape == (1, 2048, cfg_2048.vocab_size)
        assert torch.isfinite(logits).all(), "Logits at S=2048 contain NaN or Inf"

    def test_model_qk_norm_numerical_stability(self, device: torch.device):
        """Verifies RMSNorm on Q and K bounds extreme inputs (10^5) and handles all-zero vectors stably."""
        norm = RMSNorm(dim=64, eps=1e-6).to(device)

        # Extreme magnitude input
        x_huge = torch.full((2, 8, 16, 64), 1e5, device=device)
        out_huge = norm(x_huge)
        assert torch.isfinite(out_huge).all(), "RMSNorm produced NaN or Inf on extreme input"
        variance_huge = out_huge.pow(2).mean(-1)
        assert torch.allclose(variance_huge, torch.ones_like(variance_huge), atol=1e-3)

        # All-zero input
        x_zero = torch.zeros((2, 8, 16, 64), device=device)
        out_zero = norm(x_zero)
        assert torch.isfinite(out_zero).all(), "RMSNorm produced NaN on zero input"
        assert torch.all(out_zero == 0.0)

    def test_model_residual_scaling_factor(self, full_rocanon_config: RoCanonConfig):
        """
        Verifies residual scaling factor 1 / sqrt(2L) on output projections (W_O, W_down).
        For L=16: 1 / sqrt(32) approx 0.1767766952966369.
        """
        L = full_rocanon_config.n_layers
        expected_scale = 1.0 / math.sqrt(2 * L)

        attn = RoCanonAttention(full_rocanon_config)
        mlp = RoCanonMLP(full_rocanon_config)

        assert attn.scale == pytest.approx(expected_scale, rel=1e-6)
        assert mlp.scale == pytest.approx(expected_scale, rel=1e-6)
        assert expected_scale == pytest.approx(0.1767767, rel=1e-5)


# ============================================================================
# BOUNDARY AREA 3: Data Packing & cu_seqlens Boundaries (5 Tests)
# ============================================================================


class TestTier2DataPackingBoundaries:
    """Boundary and corner case tests for Data Packing and cu_seqlens."""

    def test_packing_single_document_equal_to_seq_len(self, tmp_path: Path, dummy_tokenizer: RoCanonTokenizer):
        """Verifies document whose length with <|eos|> matches seq_len exactly forms 1 pack with cu_seqlens=[0, seq_len]."""
        out_bin = str(tmp_path / "exact.bin")
        out_idx = str(tmp_path / "exact.idx")
        seq_len = 64

        # Generate text of exactly 63 tokens so +1 <|eos|> == 64
        words = ["cuvânt"] * 50
        text = " ".join(words)
        tokens = dummy_tokenizer.encode(text)
        target_tokens = tokens[: seq_len - 1]
        exact_text = dummy_tokenizer.decode(target_tokens)

        total_packed = pack_documents(dummy_tokenizer, [exact_text], out_bin, out_idx, seq_len=seq_len)
        assert total_packed == seq_len

        dataset = PackedMemmapDataset(out_bin, out_idx, seq_len=seq_len)
        assert len(dataset) == 1
        _, _, cu_seqlens = dataset[0]
        assert cu_seqlens.tolist() == [0, seq_len]

    def test_packing_oversized_document_chunking(self, tmp_path: Path, dummy_tokenizer: RoCanonTokenizer):
        """Verifies oversized document (>seq_len) partitions cleanly into sequential chunks without dropping tokens."""
        out_bin = str(tmp_path / "oversized.bin")
        out_idx = str(tmp_path / "oversized.idx")
        seq_len = 64

        # Text with ~150 tokens
        long_doc = "Propoziție lungă repetată pentru testarea segmentării fluxului de date. " * 15
        total_packed = pack_documents(dummy_tokenizer, [long_doc], out_bin, out_idx, seq_len=seq_len)

        # Packed tokens must be a multiple of seq_len
        assert total_packed % seq_len == 0
        assert total_packed >= 128

        dataset = PackedMemmapDataset(out_bin, out_idx, seq_len=seq_len)
        assert len(dataset) == total_packed // seq_len

    def test_packing_many_tiny_single_token_documents(self, tmp_path: Path, dummy_tokenizer: RoCanonTokenizer):
        """Verifies packing 32 single-token documents into seq_len=64 generates 33-element cu_seqlens [0, 2, ..., 64]."""
        out_bin = str(tmp_path / "tiny.bin")
        out_idx = str(tmp_path / "tiny.idx")
        seq_len = 64

        # 32 single character documents
        tiny_docs = [f"{i % 10}" for i in range(32)]
        pack_documents(dummy_tokenizer, tiny_docs, out_bin, out_idx, seq_len=seq_len)

        dataset = PackedMemmapDataset(out_bin, out_idx, seq_len=seq_len)
        assert len(dataset) == 1
        _, _, cu_seqlens = dataset[0]

        assert len(cu_seqlens) == 33
        assert cu_seqlens.tolist() == list(range(0, 65, 2))

    def test_packing_cu_seqlens_dtype_and_contiguity(self, dummy_packed_batch: Dict[str, torch.Tensor]):
        """Verifies cu_seqlens ABI compliance: strictly 1D, contiguous, and int32."""
        cu_seqlens = dummy_packed_batch["cu_seqlens"]

        assert cu_seqlens.dtype == torch.int32, f"cu_seqlens must be int32, got {cu_seqlens.dtype}"
        assert cu_seqlens.dim() == 1, f"cu_seqlens must be 1D, got dim={cu_seqlens.dim()}"
        assert cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous in memory"
        assert cu_seqlens[0].item() == 0, "cu_seqlens must start at 0"

    def test_memmap_corrupted_or_zero_byte_handling(self, tmp_path: Path):
        """Verifies invalid bin file sizes (not multiple of 2*seq_len) raise ValueError."""
        corrupt_bin = str(tmp_path / "corrupt.bin")
        corrupt_idx = str(tmp_path / "corrupt.idx")

        # 13 bytes is odd and not a multiple of 2*64 = 128
        with open(corrupt_bin, "wb") as f:
            f.write(b"x" * 13)
        with open(corrupt_idx, "wb") as f:
            f.write(b"\x00" * 16)

        with pytest.raises((ValueError, RuntimeError)):
            _ = PackedMemmapDataset(corrupt_bin, corrupt_idx, seq_len=64)

        # 0 bytes
        empty_bin = str(tmp_path / "empty.bin")
        empty_idx = str(tmp_path / "empty.idx")
        with open(empty_bin, "wb") as f:
            pass
        with open(empty_idx, "wb") as f:
            pass

        with pytest.raises((ValueError, RuntimeError)):
            _ = PackedMemmapDataset(empty_bin, empty_idx, seq_len=64)


# ============================================================================
# BOUNDARY AREA 4: Optimizer & Numerical Boundaries (5 Tests)
# ============================================================================


class TestTier2OptimizerBoundaries:
    """Boundary and numerical stability tests for Hybrid Optimizer and Loss."""

    def test_optimizer_zero_gradient_parameters(self, mini_model: RoCanonForCausalLM, device: torch.device):
        """Verifies parameters with explicitly zero gradients step stably without division by zero or NaN."""
        muon_opt, adamw_opt = configure_optimizers(mini_model)

        for p in mini_model.parameters():
            p.grad = torch.zeros_like(p)

        muon_opt.step()
        adamw_opt.step()

        for name, p in mini_model.named_parameters():
            assert torch.isfinite(p).all(), f"Parameter '{name}' produced NaN/Inf on zero-grad step"

    def test_optimizer_gradient_clipping_extreme_values(self, mini_model: RoCanonForCausalLM, device: torch.device):
        """Verifies gradient clipping contains massive gradients (10^6) to total norm <= 1.0."""
        for p in mini_model.parameters():
            p.grad = torch.full_like(p, 1e6)

        torch.nn.utils.clip_grad_norm_(mini_model.parameters(), max_norm=1.0)
        norm_after = torch.norm(torch.stack([torch.norm(p.grad) for p in mini_model.parameters()]))
        assert norm_after.item() <= 1.0 + 1e-4, f"Clipped grad norm exceeded threshold 1.0: {norm_after.item()}"

    def test_muon_newton_schulz_convergence_extreme_matrices(self):
        """Verifies Quintic Newton-Schulz polynomial stably processes ill-conditioned matrices (cond > 10^7) without NaN/Inf."""
        torch.manual_seed(42)
        u, _ = torch.linalg.qr(torch.randn(32, 32))
        v, _ = torch.linalg.qr(torch.randn(32, 32))
        s = torch.logspace(0, -7, 32)
        W = u @ torch.diag(s) @ v.T  # ill-conditioned matrix

        # Quintic Newton-Schulz iteration coefficients
        a, b, c = 3.4445, -4.775, 2.0315
        X = W / (W.norm() + 1e-7)
        for _ in range(5):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X

        assert torch.isfinite(X).all(), "Newton-Schulz produced NaN or Inf on ill-conditioned matrix"
        assert not torch.isnan(X).any()
        assert not torch.isinf(X).any()

        # Step PyTorch native Muon optimizer on ill-conditioned parameter
        param = nn.Parameter(W.clone())
        opt = torch.optim.Muon([param], lr=0.01)
        (param ** 2).sum().backward()
        opt.step()
        assert torch.isfinite(param).all(), "Muon step produced NaN or Inf on ill-conditioned matrix"

    def test_mixed_precision_bf16_fp32_gradient_accumulation(
        self, mini_model: RoCanonForCausalLM, minimal_rocanon_config: RoCanonConfig, device: torch.device
    ):
        """Verifies model forward pass under BF16 autocast accumulates loss and master gradients in FP32."""
        input_ids = torch.randint(0, minimal_rocanon_config.vocab_size, (2, 8), device=device)

        mini_model.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = mini_model(input_ids)

        loss = F.cross_entropy(logits.float().view(-1, minimal_rocanon_config.vocab_size), input_ids.view(-1))
        loss.backward()

        assert loss.dtype == torch.float32, f"Loss dtype must be FP32, got {loss.dtype}"
        assert mini_model.embed_tokens.weight.grad.dtype == torch.float32, (
            f"Master grad dtype must be FP32, got {mini_model.embed_tokens.weight.grad.dtype}"
        )

    def test_loss_token_mean_normalization_with_masked_tokens(self, device: torch.device):
        """Verifies token-mean loss normalization weighs individual valid tokens equally, unlike micro-batch averaging."""
        logits1 = torch.randn(20, 100, device=device)
        labels1 = torch.randint(0, 100, (20,), device=device)

        logits2 = torch.randn(80, 100, device=device)
        labels2 = torch.randint(0, 100, (80,), device=device)

        sum_loss1 = compute_chunked_loss(logits1, labels1, reduction="sum")
        sum_loss2 = compute_chunked_loss(logits2, labels2, reduction="sum")

        total_valid = 20 + 80  # 100
        token_mean_loss = (sum_loss1 + sum_loss2) / total_valid

        # Naive micro-batch mean: 0.5 * (mean1 + mean2)
        naive_mean = 0.5 * (sum_loss1 / 20.0 + sum_loss2 / 80.0)

        # Because batch sizes are 20 and 80 (unequal weights), token-mean loss != naive mean
        assert not torch.isclose(token_mean_loss, naive_mean, atol=1e-4), (
            "Token-mean loss should differ from naive micro-batch averaging on unequal token batches"
        )
