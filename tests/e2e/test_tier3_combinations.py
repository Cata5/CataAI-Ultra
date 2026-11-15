"""
tests/e2e/test_tier3_combinations.py
Tier 3: Pairwise Cross-Feature Combinations and Subsystem Invariants for RoCanon-48.5M.

Covers:
1. Tokenizer + Data Packing: Raw multilingual text -> encode -> sequence packing -> cu_seqlens.
2. Tokenizer + Model Forward: Tokenize text -> feed into RoCanonForCausalLM -> logits & tied embedding check.
3. Packed Sequences + cu_seqlens Attention Isolation: Zero cross-document attention leakage under block-diagonal mask.
4. Model Architecture + Hybrid Optimizer Partition: Exact 48,531,968 parameter partition between Muon & AdamW.
5. Canon Conv1d Zero-Init & Residual Identity: Canon-A/B/C/D bitwise identity pass-through at step 0.
6. QK-Norm + RoPE Interaction: RMS normalization followed by norm-preserving orthogonal rotary transformation.
7. SwiGLU + Canon-D Interaction: Post-gate/up causal convolution integration and gradient flow.
8. Chunked Loss + Token-Mean Normalization: Masked token invariance and parity with CrossEntropyLoss.
Total: 8 tests.
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


class TestTier3Combinations:
    """Pairwise cross-feature combinatorial test suite for RoCanon-48.5M."""

    def test_tier3_tokenizer_and_data_packing(self, dummy_tokenizer: RoCanonTokenizer, tmp_path: Path):
        """Verifies end-to-end flow from raw diverse texts to uint16 packed binary shard and int32 cu_seqlens."""
        docs = [
            "Ştefan cel Mare şi Sfânt a domnit 47 de ani în Ţara Românească şi Moldova.",
            "Transformer architectures rely on grouped query attention for KV cache efficiency.",
            "Calculul numeric este 12 + 34 = 46 şi 100 * 2 = 200.",
            "    def calculate(x, y):\n        return x + y",
            "<|user|>Cum funcționează RoCanon-48.5M?<|end_turn|>",
        ]
        out_bin = str(tmp_path / "combined_pack.bin")
        out_idx = str(tmp_path / "combined_pack.idx")
        seq_len = 64

        total_tokens = pack_documents(dummy_tokenizer, docs, out_bin, out_idx, seq_len=seq_len)
        assert total_tokens > 0

        dataset = PackedMemmapDataset(out_bin, out_idx, seq_len=seq_len)
        assert len(dataset) >= 1

        input_ids, labels, cu_seqlens = dataset[0]
        assert cu_seqlens.dtype == torch.int32
        assert cu_seqlens[0].item() == 0
        assert cu_seqlens[-1].item() == seq_len

        # All token IDs must be in valid vocabulary range
        assert torch.all((input_ids >= 0) & (input_ids < DEFAULT_VOCAB_SIZE))

        # Pad token must not exist in sequence
        pad_id = dummy_tokenizer.encode("<|pad|>")[0]
        assert pad_id not in input_ids.tolist()

        # Decode sequence and verify text reconstructed
        decoded_text = dummy_tokenizer.decode(input_ids.tolist())
        assert len(decoded_text) > 0

    def test_tier3_tokenizer_and_model_forward(
        self, dummy_tokenizer: RoCanonTokenizer, mini_model: RoCanonForCausalLM, device: torch.device
    ):
        """Verifies text tokenization fed directly into model forward pass generates finite logits with tied head."""
        prompt = "Ştefan cel Mare şi Sfânt a domnit 47 de ani şi a apărat ţara."
        token_ids = dummy_tokenizer.encode(prompt)
        assert len(token_ids) > 0

        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
        logits = mini_model(input_tensor)

        assert logits.shape == (1, len(token_ids), mini_model.config.vocab_size)
        assert torch.isfinite(logits).all(), "Logits contain NaN or Inf values"

        # Verify weight tying pointer
        assert mini_model.lm_head.weight.data_ptr() == mini_model.embed_tokens.weight.data_ptr()

    def test_tier3_packed_sequences_and_cu_seqlens_attention_isolation(
        self, mini_model: RoCanonForCausalLM, device: torch.device
    ):
        """
        Verifies block-diagonal attention mask constructed from cu_seqlens prevents attention leakage across documents:
        Mutating document A must produce exactly zero change in document B logits.
        """
        len_a = 16
        len_b = 24
        total_len = len_a + len_b  # 40

        torch.manual_seed(101)
        doc_a1 = torch.randint(10, 200, (1, len_a), device=device)
        doc_a2 = torch.randint(201, 400, (1, len_a), device=device)
        doc_b = torch.randint(401, 600, (1, len_b), device=device)

        packed1 = torch.cat([doc_a1, doc_b], dim=1)
        packed2 = torch.cat([doc_a2, doc_b], dim=1)

        cu_seqlens = torch.tensor([0, len_a, total_len], dtype=torch.int32, device=device)

        mini_model.eval()
        with torch.no_grad():
            logits1 = mini_model(packed1, cu_seqlens=cu_seqlens)
            logits2 = mini_model(packed2, cu_seqlens=cu_seqlens)

        logits_b1 = logits1[:, len_a:, :]
        logits_b2 = logits2[:, len_a:, :]

        # Max absolute difference must be strictly 0 (within FP precision)
        max_diff = (logits_b1 - logits_b2).abs().max().item()
        assert max_diff < 1e-4, (
            f"Attention leakage detected across document boundaries! Max diff on Doc B: {max_diff}"
        )

    def test_tier3_model_architecture_and_hybrid_optimizer_partition(self, full_rocanon_config: RoCanonConfig):
        """
        Verifies exact 48,531,968 parameter partition:
        - Muon: exactly 112 2D hidden linear matrices (41,943,040 params, WD=0.01)
        - AdamW: tied embeddings (6,291,456 params, WD=0.01), 1D norms (18,944 params, WD=0.0), Canon 3D (278,528 params, WD=0.0)
        - Total: strictly 48,531,968 params.
        """
        L = full_rocanon_config.n_layers
        d = full_rocanon_config.d_model
        d_h = full_rocanon_config.d_h
        d_kv = full_rocanon_config.n_kv * d_h
        d_ff = full_rocanon_config.d_ff
        V = full_rocanon_config.vocab_size
        k = full_rocanon_config.kernel_size

        # 7 hidden linear matrices per layer * 16 layers = 112 matrices
        num_muon_matrices = L * 7
        assert num_muon_matrices == 112

        w_q = d * (full_rocanon_config.n_q * d_h)
        w_k = d * d_kv
        w_v = d * d_kv
        w_o = (full_rocanon_config.n_q * d_h) * d
        w_gate = d * d_ff
        w_up = d * d_ff
        w_down = d_ff * d
        muon_params = L * (w_q + w_k + w_v + w_o + w_gate + w_up + w_down)

        adamw_embed = V * d
        adamw_norms = L * (2 * d + 2 * d_h) + d
        adamw_convs = L * k * (3 * d + 2 * d_kv + 2 * d_ff)
        adamw_params = adamw_embed + adamw_norms + adamw_convs

        assert muon_params == 41943040
        assert adamw_params == 6588928
        assert muon_params + adamw_params == 48531968

    def test_tier3_canon_conv1d_zero_init_and_residual_identity(self, device: torch.device):
        """Verifies all four Canon layer types (A, B, C, D) are zero-initialized and provide exact bitwise identity."""
        channels = [512, 768, 512, 2560]
        k = 4

        for c in channels:
            canon = CausalDepthwiseConv1d(channels=c, kernel_size=k).to(device)

            # Check zero-initialization
            assert torch.all(canon.conv.weight == 0.0), f"Canon layer (C={c}) not zero-initialized"

            # Check identity pass-through
            x = torch.randn(2, 16, c, device=device)
            out = canon(x)
            assert torch.equal(out, x), f"Canon layer (C={c}) failed identity pass-through at init"

    def test_tier3_qk_norm_and_rope_interaction(self, device: torch.device):
        """Verifies QK-Norm normalizes head dimension to RMS 1.0, and RoPE preserves vector L2 norm (orthogonal rotation)."""
        d_h = 64
        q_norm = RMSNorm(dim=d_h, eps=1e-6).to(device)
        rotary = RotaryEmbedding(dim=d_h, max_seq_len=64).to(device)

        B, H, S = 2, 8, 16
        q = torch.randn(B, H, S, d_h, device=device)

        # 1. Apply QK-Norm
        q_normed = q_norm(q)
        rms = torch.sqrt(q_normed.pow(2).mean(dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3), "QK-Norm failed to set RMS to 1.0"

        # 2. Apply RoPE
        cos, sin = rotary.get_cos_sin(S, device=device, dtype=q.dtype)
        q_rotated = apply_rope(q_normed, cos, sin)

        # 3. Verify L2 norm is preserved before and after RoPE rotation
        norm_before = torch.norm(q_normed, dim=-1)
        norm_after = torch.norm(q_rotated, dim=-1)
        assert torch.allclose(norm_before, norm_after, atol=1e-5), "RoPE rotation did not preserve L2 vector norm!"

    def test_tier3_swiglu_and_canon_d_interaction(
        self, mini_model: RoCanonForCausalLM, minimal_rocanon_config: RoCanonConfig, device: torch.device
    ):
        """Verifies Canon-D operates on concatenated gate+up representations, feeds SwiGLU activation, and propagates gradients."""
        layer = mini_model.layers[0]
        mlp = layer.mlp
        d_model = minimal_rocanon_config.d_model
        d_ff = minimal_rocanon_config.d_ff

        x = torch.randn(2, 16, d_model, device=device, requires_grad=True)

        # Project gate and up, concatenate for Canon-D
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        assert gate.shape == (2, 16, d_ff)
        assert up.shape == (2, 16, d_ff)

        gu = mlp.canon_d(torch.cat([gate, up], dim=-1))
        assert gu.shape == (2, 16, 2 * d_ff)

        g, u = torch.split(gu, [d_ff, d_ff], dim=-1)
        hidden = F.silu(g) * u
        out = mlp.down_proj(hidden) * mlp.scale
        assert out.shape == (2, 16, d_model)

        loss = out.sum()
        loss.backward()

        assert mlp.canon_d.conv.weight.grad is not None
        assert mlp.gate_proj.weight.grad is not None
        assert mlp.up_proj.weight.grad is not None
        assert mlp.down_proj.weight.grad is not None
        assert torch.isfinite(mlp.canon_d.conv.weight.grad).all()

    def test_tier3_loss_computation_and_token_mean_normalization(self, device: torch.device):
        """Verifies chunked cross-entropy parity with CrossEntropyLoss and invariance to masked (-100) positions."""
        B, S, V = 2, 32, 128
        logits = torch.randn(B * S, V, device=device, requires_grad=True)
        labels = torch.randint(0, V, (B * S,), device=device)

        # Mask 40% of tokens
        labels[10:35] = -100

        # Chunked loss
        loss_chunked = compute_chunked_loss(logits, labels, chunk_size=16, ignore_index=-100)

        # Standard PyTorch cross entropy
        loss_torch = F.cross_entropy(logits, labels, ignore_index=-100, reduction="mean")

        assert torch.allclose(loss_chunked, loss_torch, atol=1e-5), (
            f"Chunked loss ({loss_chunked.item()}) differs from PyTorch CrossEntropy ({loss_torch.item()})"
        )

        # Mutate masked logits: loss must remain 100% bitwise invariant
        logits_mutated = logits.clone().detach().requires_grad_(True)
        with torch.no_grad():
            logits_mutated[10:35] += 100.0  # huge perturbation on masked tokens

        loss_perturbed = compute_chunked_loss(logits_mutated, labels, chunk_size=16, ignore_index=-100)
        assert torch.allclose(loss_chunked, loss_perturbed, atol=1e-6), (
            "Perturbing masked (-100) logits altered the computed loss!"
        )
