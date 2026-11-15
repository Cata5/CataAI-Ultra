"""
tests/e2e/test_tier1_features.py
Tier 1: Happy-Path Isolated Feature Coverage for RoCanon-48.5M.

Covers:
- Tokenizer: 6 tests (Special Tokens, Sedila Unification, Single-Digit Atomization,
  Preserved Indentation, UTF-8 Round-Trip, Save & Load Pretrained)
- Architecture: 6 tests (Exact Analytical Parameter Count, Weight Tying Identity,
  Canon Conv1d Layer Shapes, Forward Pass Output Shapes, Backward Pass Gradient Propagation,
  Zero-Bias Projections)
- Data Packing: 5 tests (Padding-Free Packing with <|eos|>, cu_seqlens Construction,
  Memmap Dataset Zero-Copy & dtypes, Epoch Indexing & Bounds, cu_seqlens Spanning)
- Hybrid Optimizer: 6 tests (Parameter Partition Disjointness, Analytical Parameter Counts,
  Weight Decay Differentiation, Step Execution, WSD Schedule Phases, Batch Accumulation)
Total: 23 tests.
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
    compute_chunked_loss,
    configure_optimizers,
    pack_documents,
)


# ============================================================================
# FEATURE 1: Custom Bilingual Tokenizer (6 Tests)
# ============================================================================


class TestTier1Tokenizer:
    """Happy-path feature tests for RoCanonTokenizer."""

    def test_tokenizer_special_tokens_registration_and_ids(self, dummy_tokenizer: RoCanonTokenizer):
        """Verifies all 16 special tokens are registered, unique, round-trippable, and in [0, 12287]."""
        assert len(SPECIAL_TOKENS) == 16, f"Expected 16 special tokens, got {len(SPECIAL_TOKENS)}"

        token_ids: List[int] = []
        for token_str in SPECIAL_TOKENS:
            encoded = dummy_tokenizer.encode(token_str)
            assert len(encoded) == 1, f"Special token '{token_str}' must encode to exactly 1 ID, got {encoded}"
            tid = encoded[0]
            assert 0 <= tid < DEFAULT_VOCAB_SIZE, f"Token ID {tid} for '{token_str}' out of vocabulary range [0, 12287]"
            token_ids.append(tid)

            # Round trip decoding
            decoded = dummy_tokenizer.decode([tid], skip_special_tokens=False)
            assert decoded == token_str, f"Round trip decode for special token failed: '{decoded}' != '{token_str}'"

        # Unique token IDs
        assert len(set(token_ids)) == 16, f"Special token IDs must be unique, got duplicates: {token_ids}"

    def test_tokenizer_sedila_normalization_happy_path(self, dummy_tokenizer: RoCanonTokenizer):
        """Verifies strict Romanian sedila unification (ş/ţ -> ș/ț and Ş/Ţ -> Ș/Ț)."""
        sample_legacy = "oraş şi ţară învaţă Ştiinţă Ţepeş"
        normalized = dummy_tokenizer.normalize(sample_legacy)

        # Legacy cedillas must be completely eliminated
        for char in ["ş", "ţ", "Ş", "Ţ"]:
            assert char not in normalized, f"Legacy cedilla '{char}' still present in normalized string: '{normalized}'"

        # Standard comma-below forms must be present
        for char in ["ș", "ț", "Ș", "Ț"]:
            assert char in normalized, f"Standard comma-below '{char}' missing from normalized string: '{normalized}'"

        # Identical token IDs emitted for cedilla vs comma-below variants
        assert dummy_tokenizer.encode("oraş") == dummy_tokenizer.encode("oraș")
        assert dummy_tokenizer.encode("ţară") == dummy_tokenizer.encode("țară")
        assert dummy_tokenizer.encode("Ştefan") == dummy_tokenizer.encode("Ștefan")
        assert dummy_tokenizer.encode("Ţepeş") == dummy_tokenizer.encode("Țepeș")

    def test_tokenizer_single_digit_atomization(self, dummy_tokenizer: RoCanonTokenizer):
        """Verifies digits are atomized 1-by-1 and mathematical operators do not fuse with digits."""
        # 5-digit number encodes to 5 discrete tokens
        encoded_5 = dummy_tokenizer.encode("12345")
        assert len(encoded_5) == 5, f"Expected 5 tokens for '12345', got {len(encoded_5)} ({encoded_5})"

        # Standalone digits match digits inside multi-digit numbers
        tok_1 = dummy_tokenizer.encode("1")
        tok_0 = dummy_tokenizer.encode("0")
        tok_10 = dummy_tokenizer.encode("10")
        assert len(tok_10) == 2, f"Expected 2 tokens for '10', got {len(tok_10)}"
        assert tok_10[0] == tok_1[0], f"First digit of '10' ({tok_10[0]}) does not match '1' ({tok_1[0]})"
        assert tok_10[1] == tok_0[0], f"Second digit of '10' ({tok_10[1]}) does not match '0' ({tok_0[0]})"

        # Operators do not fuse with neighboring digits in unspaced math
        eq_tokens = dummy_tokenizer.encode("10+20=30")
        assert len(eq_tokens) == 8, f"Expected 8 atomized tokens for '10+20=30', got {len(eq_tokens)}"
        tok_2 = dummy_tokenizer.encode("2")
        tok_3 = dummy_tokenizer.encode("3")
        assert eq_tokens[0] == tok_1[0], "First digit '1' in '10+20=30' does not match individual digit '1'"
        assert eq_tokens[1] == tok_0[0], "Second digit '0' in '10+20=30' does not match individual digit '0'"
        assert eq_tokens[3] == tok_2[0], "Third digit '2' in '10+20=30' does not match individual digit '2'"
        assert eq_tokens[4] == tok_0[0], "Fourth digit '0' in '10+20=30' does not match individual digit '0'"
        assert eq_tokens[6] == tok_3[0], "Fifth digit '3' in '10+20=30' does not match individual digit '3'"
        assert eq_tokens[7] == tok_0[0], "Sixth digit '0' in '10+20=30' does not match individual digit '0'"
        decoded_eq = dummy_tokenizer.decode(eq_tokens)
        assert decoded_eq == "10+20=30"

    def test_tokenizer_preserved_indentation(self, dummy_tokenizer: RoCanonTokenizer):
        """Verifies code indentation whitespace (2, 4, 8 spaces) is 100% preserved."""
        indent_samples = [
            "  x = 1",
            "    def run():",
            "        return True",
            "        for i in range(10):\n            pass",
        ]
        for sample in indent_samples:
            encoded = dummy_tokenizer.encode(sample)
            decoded = dummy_tokenizer.decode(encoded)
            assert decoded == sample, f"Indentation lost in roundtrip:\nExpected: {repr(sample)}\nGot:      {repr(decoded)}"

    def test_tokenizer_utf8_roundtrip_invariance(self, dummy_tokenizer: RoCanonTokenizer):
        """Verifies exact 0-error roundtrip reconstruction across Romanian, English, code, and math."""
        text = "RoCanon-48.5M rezolvă ecuația f(x) = x^2 + 3x - 1; def test_țăran(): return 'succes' # 100%"
        encoded = dummy_tokenizer.encode(text)
        decoded = dummy_tokenizer.decode(encoded)
        assert decoded == dummy_tokenizer.normalize(text), f"Roundtrip failed:\nOriginal: {text}\nDecoded:  {decoded}"

    def test_tokenizer_save_and_from_pretrained_roundtrip(self, dummy_tokenizer: RoCanonTokenizer, tmp_path: Path):
        """Verifies serialization to disk and reload via from_pretrained preserves exact behavior."""
        save_dir = str(tmp_path / "saved_tokenizer")
        dummy_tokenizer.save_pretrained(save_dir)

        assert os.path.exists(save_dir)
        reloaded = RoCanonTokenizer.from_pretrained(save_dir)

        test_text = "Verificare salvare tokenizer bilingv RoCanon 2026 cu diacritice: ș, ț, â, î, ă."
        assert reloaded.encode(test_text) == dummy_tokenizer.encode(test_text)
        assert reloaded.decode(reloaded.encode(test_text)) == dummy_tokenizer.decode(dummy_tokenizer.encode(test_text))


# ============================================================================
# FEATURE 2: RoCanon-48.5M Model Architecture (6 Tests)
# ============================================================================


class TestTier1Architecture:
    """Happy-path feature tests for RoCanon-48.5M Model Architecture."""

    def test_model_exact_parameter_count_analytical(self, full_rocanon_config: RoCanonConfig):
        """
        Verifies exact analytical parameter count specified in doc.md Section D:
        N_total = V*d + L * [2d^2 + 2d*d_kv + 2d_h + 2d + 3d*d_ff + k*(3d + 2d_kv + 2d_ff)] + d
        With V=12288, d=512, L=16, n_q=8, n_kv=2, d_h=64, d_kv=128, d_ff=1280, k=4:
        N_total = 48,531,968 exact parameters.
        """
        V = full_rocanon_config.vocab_size
        d = full_rocanon_config.d_model
        L = full_rocanon_config.n_layers
        d_h = full_rocanon_config.d_h
        d_kv = full_rocanon_config.n_kv * d_h
        d_ff = full_rocanon_config.d_ff
        k = full_rocanon_config.kernel_size

        tied_embedding = V * d  # 6,291,456
        attn_linear = 2 * d * d + 2 * d * d_kv  # 655,360
        qk_norm = 2 * d_h  # 128
        pre_norms = 2 * d  # 1,024
        swiglu = 3 * d * d_ff  # 1,966,080
        canon_convs = k * (3 * d + 2 * d_kv + 2 * d_ff)  # 17,408
        layer_total = attn_linear + qk_norm + pre_norms + swiglu + canon_convs  # 2,640,000

        backbone = L * layer_total  # 42,240,000
        final_norm = d  # 512
        total_non_embedding = backbone + final_norm  # 42,240,512
        total_analytical = tied_embedding + total_non_embedding  # 48,531,968

        assert layer_total == 2640000, f"Expected 2,640,000 params/layer, got {layer_total}"
        assert tied_embedding == 6291456, f"Expected 6,291,456 tied embedding params, got {tied_embedding}"
        assert total_non_embedding == 42240512, f"Expected 42,240,512 non-embedding params, got {total_non_embedding}"
        assert total_analytical == 48531968, f"Expected 48,531,968 total parameters, got {total_analytical}"

        # Verify exact parameter count on an instantiated model instance
        assert sum(p.numel() for p in set(RoCanonForCausalLM(full_rocanon_config).parameters())) == 48531968

    def test_model_weight_tying_identity(self, mini_model: RoCanonForCausalLM):
        """Verifies exact memory pointer identity between embed_tokens.weight and lm_head.weight."""
        assert mini_model.lm_head.weight is mini_model.embed_tokens.weight, (
            "lm_head.weight and embed_tokens.weight must reference the exact same Parameter instance"
        )
        assert mini_model.lm_head.weight.data_ptr() == mini_model.embed_tokens.weight.data_ptr(), (
            "lm_head.weight and embed_tokens.weight must share identical storage data pointer"
        )

        # Mutate in-place to verify dual visibility
        orig_val = mini_model.embed_tokens.weight.data[0, 0].item()
        mini_model.embed_tokens.weight.data[0, 0] += 1.0
        assert mini_model.lm_head.weight.data[0, 0].item() == pytest.approx(orig_val + 1.0)
        mini_model.embed_tokens.weight.data[0, 0] -= 1.0  # restore

    def test_model_canon_conv1d_layers_presence_and_shapes(
        self, mini_model: RoCanonForCausalLM, minimal_rocanon_config: RoCanonConfig
    ):
        """Verifies Canon-A, Canon-B, Canon-C, Canon-D causal conv1d presence and exact tensor shapes."""
        d = minimal_rocanon_config.d_model
        d_kv = minimal_rocanon_config.n_kv * minimal_rocanon_config.d_h
        d_ff = minimal_rocanon_config.d_ff
        k = minimal_rocanon_config.kernel_size

        def get_conv_weight(canon_module):
            if hasattr(canon_module, "weight"):
                return canon_module.weight
            return canon_module.conv.weight

        for idx, layer in enumerate(mini_model.layers):
            # Canon-A: Pre-Attention [d_model, 1, k]
            assert hasattr(layer, "canon_a"), f"Layer {idx} missing canon_a"
            assert get_conv_weight(layer.canon_a).shape == (d, 1, k), (
                f"Canon-A shape mismatch: {get_conv_weight(layer.canon_a).shape}"
            )

            # Canon-B: Post-QKV [d_model + 2*d_kv, 1, k]
            assert hasattr(layer.attn, "canon_b"), f"Layer {idx}.attn missing canon_b"
            assert get_conv_weight(layer.attn.canon_b).shape == (d + 2 * d_kv, 1, k), (
                f"Canon-B shape mismatch: {get_conv_weight(layer.attn.canon_b).shape}"
            )

            # Canon-C: Pre-MLP [d_model, 1, k]
            assert hasattr(layer, "canon_c"), f"Layer {idx} missing canon_c"
            assert get_conv_weight(layer.canon_c).shape == (d, 1, k), (
                f"Canon-C shape mismatch: {get_conv_weight(layer.canon_c).shape}"
            )

            # Canon-D: Post-Gate/Up [2*d_ff, 1, k]
            assert hasattr(layer.mlp, "canon_d"), f"Layer {idx}.mlp missing canon_d"
            assert get_conv_weight(layer.mlp.canon_d).shape == (2 * d_ff, 1, k), (
                f"Canon-D shape mismatch: {get_conv_weight(layer.mlp.canon_d).shape}"
            )

    def test_model_forward_pass_output_shapes(
        self, mini_model: RoCanonForCausalLM, minimal_rocanon_config: RoCanonConfig, device: torch.device
    ):
        """Verifies forward pass produces [B, S, V] logits without NaN or Inf."""
        B, S = 2, 16
        input_ids = torch.randint(0, minimal_rocanon_config.vocab_size, (B, S), device=device)
        logits = mini_model(input_ids)

        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (B, S, minimal_rocanon_config.vocab_size), (
            f"Expected logits shape {(B, S, minimal_rocanon_config.vocab_size)}, got {logits.shape}"
        )
        assert torch.isfinite(logits).all(), "Logits contain NaN or Inf values"

    def test_model_backward_pass_gradients_propagation(
        self, mini_model: RoCanonForCausalLM, minimal_rocanon_config: RoCanonConfig, device: torch.device
    ):
        """Verifies forward + backward pass propagates non-zero gradients to all unique trainable parameters."""
        input_ids = torch.randint(0, minimal_rocanon_config.vocab_size, (2, 8), device=device)
        labels = input_ids.clone()

        mini_model.zero_grad()
        loss, logits = mini_model(input_ids, labels=labels)
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        loss.backward()

        unique_params = set(mini_model.parameters())
        for param in unique_params:
            if param.requires_grad:
                assert param.grad is not None, "Parameter grad is None after backward"
                assert torch.isfinite(param.grad).all(), "Parameter grad contains NaN or Inf"
                assert torch.norm(param.grad) > 0.0, "Parameter grad is completely zero"

    def test_model_zero_bias_linear_projections(self, mini_model: RoCanonForCausalLM):
        """Verifies all linear and conv1d projections strictly enforce bias=False."""
        for name, module in mini_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                assert module.bias is None, f"Module '{name}' ({type(module).__name__}) has a bias parameter!"


# ============================================================================
# FEATURE 3: Data Ingestion & Packing (5 Tests)
# ============================================================================


class TestTier1DataPacking:
    """Happy-path feature tests for Data Ingestion, Padding-Free Packing & cu_seqlens."""

    def test_packing_without_padding_delimiter(self, dummy_tokenizer: RoCanonTokenizer, tmp_path: Path):
        """Verifies documents are packed with <|eos|> delimiters without inserting pad tokens."""
        docs = [
            "Primul document despre RoCanon-48.5M.",
            "Al doilea text explică Grouped-Query Attention.",
            "Al treilea text conține cod Python: def add(a, b): return a + b",
        ]
        out_bin = str(tmp_path / "packed.bin")
        out_idx = str(tmp_path / "packed.idx")

        total_packed = pack_documents(dummy_tokenizer, docs, out_bin, out_idx, seq_len=32)
        assert total_packed > 0
        assert os.path.exists(out_bin)
        assert os.path.exists(out_idx)

        tokens = np.fromfile(out_bin, dtype=np.uint16)
        pad_id = dummy_tokenizer.encode("<|pad|>")[0]
        eos_id = dummy_tokenizer.eos_token_id

        assert pad_id not in tokens, "Padding token was found in packed stream!"
        assert eos_id in tokens, "<|eos|> delimiter missing from packed stream!"

    def test_cu_seqlens_vector_construction(self, dummy_memmap_shards: Tuple[str, str, int, List[int]]):
        """Verifies build_cu_seqlens constructs exact int32 cumulative vector [0, L1, L1+L2, ...]."""
        _, _, total_tokens, doc_lengths = dummy_memmap_shards
        cu = build_cu_seqlens(doc_lengths)

        assert cu.dtype == torch.int32, f"cu_seqlens dtype must be int32, got {cu.dtype}"
        assert cu.dim() == 1
        assert cu[0].item() == 0
        assert cu[-1].item() == total_tokens
        assert len(cu) == len(doc_lengths) + 1
        assert cu.tolist() == [0, 20, 50, 95, 128]

    def test_memmap_dataset_zero_copy_and_dtypes(self, dummy_memmap_shards: Tuple[str, str, int, List[int]]):
        """Verifies PackedMemmapDataset streams uint16 memmap data into typed PyTorch tensors."""
        bin_path, idx_path, total_tokens, _ = dummy_memmap_shards
        seq_len = 64
        dataset = PackedMemmapDataset(bin_path, idx_path, seq_len=seq_len)

        assert dataset.data.dtype == np.uint16
        assert len(dataset) == total_tokens // seq_len

        input_ids, labels, cu_seqlens = dataset[0]
        assert input_ids.dtype == torch.long
        assert labels.dtype == torch.long
        assert cu_seqlens.dtype == torch.int32
        assert len(input_ids) == seq_len
        assert len(labels) == seq_len
        assert cu_seqlens[0].item() == 0
        assert cu_seqlens[-1].item() == seq_len

    def test_memmap_dataset_epoch_indexing_and_bounds(self, dummy_memmap_shards: Tuple[str, str, int, List[int]]):
        """Verifies PackedMemmapDataset length and strict out-of-bounds index protection."""
        bin_path, idx_path, total_tokens, _ = dummy_memmap_shards
        seq_len = 64
        dataset = PackedMemmapDataset(bin_path, idx_path, seq_len=seq_len)

        assert len(dataset) == 128 // 64 == 2
        _ = dataset[0]
        _ = dataset[1]

        with pytest.raises(IndexError):
            _ = dataset[2]
        with pytest.raises(IndexError):
            _ = dataset[-1]

    def test_cu_seqlens_spanning_packed_boundaries(self, dummy_packed_batch: Dict[str, torch.Tensor]):
        """Verifies cu_seqlens strictly demarcates document boundaries with monotonic spans."""
        cu_seqlens = dummy_packed_batch["cu_seqlens"]
        seq_len = dummy_packed_batch["input_ids"].shape[1]

        diffs = torch.diff(cu_seqlens)
        assert torch.all(diffs > 0), "cu_seqlens must be strictly monotonically increasing"
        assert cu_seqlens[0].item() == 0, "cu_seqlens must start at 0"
        assert cu_seqlens[-1].item() == seq_len, f"cu_seqlens must end at seq_len ({seq_len})"


# ============================================================================
# FEATURE 4: Hybrid Optimizer & Schedulers (6 Tests)
# ============================================================================


class TestTier1HybridOptimizer:
    """Happy-path feature tests for Hybrid Optimizer (Muon + AdamW) and Schedulers."""

    def test_optimizer_parameter_partition_types(self, mini_model: RoCanonForCausalLM):
        """Verifies clean, non-overlapping partition: Muon gets 2D weights; AdamW gets 1D, 3D & embeddings."""
        muon_opt, adamw_opt = configure_optimizers(mini_model)

        muon_params = set(p for g in muon_opt.param_groups for p in g["params"])
        adamw_params = set(p for g in adamw_opt.param_groups for p in g["params"])
        all_trainable = set(p for p in mini_model.parameters() if p.requires_grad)

        assert muon_params.isdisjoint(adamw_params), "Muon and AdamW parameter sets must be strictly disjoint!"
        assert muon_params.union(adamw_params) == all_trainable, "All trainable parameters must be covered by optimizers!"

        # Muon params must be strictly 2D
        for p in muon_params:
            assert p.ndim == 2, f"Muon parameter must be 2D, got ndim={p.ndim} (shape={p.shape})"

        # AdamW params must be 1D, 3D, or the tied embedding matrix
        for p in adamw_params:
            assert p.ndim in (1, 3) or p is mini_model.embed_tokens.weight, (
                f"Unexpected tensor in AdamW group with shape {p.shape}"
            )

    def test_optimizer_parameter_partition_counts_full_model(self, full_rocanon_config: RoCanonConfig):
        """
        Verifies mathematical parameter allocation for full RoCanon-48.5M:
        - Muon: 112 2D hidden matrices = 41,943,040 params (86.42%)
        - AdamW: tied embeddings (6,291,456) + 1D norms (18,944) + Canon 3D (278,528) = 6,588,928 params (13.58%)
        - Combined: 48,531,968 params.
        """
        L = full_rocanon_config.n_layers
        d = full_rocanon_config.d_model
        d_h = full_rocanon_config.d_h
        d_kv = full_rocanon_config.n_kv * d_h
        d_ff = full_rocanon_config.d_ff
        V = full_rocanon_config.vocab_size
        k = full_rocanon_config.kernel_size

        # Muon: 7 matrices per layer (W_q, W_k, W_v, W_o, W_gate, W_up, W_down)
        w_q = d * (full_rocanon_config.n_q * d_h)
        w_k = d * d_kv
        w_v = d * d_kv
        w_o = (full_rocanon_config.n_q * d_h) * d
        w_gate = d * d_ff
        w_up = d * d_ff
        w_down = d_ff * d
        muon_per_layer = w_q + w_k + w_v + w_o + w_gate + w_up + w_down  # 2,621,440
        muon_total = L * muon_per_layer  # 41,943,040

        # AdamW:
        tied_embed = V * d  # 6,291,456
        norm_scales = L * (2 * d + 2 * d_h) + d  # 18,944
        canon_weights = L * k * (3 * d + 2 * d_kv + 2 * d_ff)  # 278,528
        adamw_total = tied_embed + norm_scales + canon_weights  # 6,588,928

        assert muon_total == 41943040, f"Expected 41,943,040 Muon params, got {muon_total}"
        assert adamw_total == 6588928, f"Expected 6,588,928 AdamW params, got {adamw_total}"
        assert muon_total + adamw_total == 48531968, "Muon + AdamW total must strictly equal 48,531,968"

    def test_optimizer_weight_decay_differentiation(self, mini_model: RoCanonForCausalLM):
        """Verifies weight decay is 0.01 for Muon & embeddings, and strictly 0.0 for 1D norms and Canon conv1d."""
        muon_opt, adamw_opt = configure_optimizers(mini_model)

        # Muon weight decay
        for g in muon_opt.param_groups:
            assert g["weight_decay"] == 0.01, f"Muon weight decay must be 0.01, got {g['weight_decay']}"

        # AdamW weight decays
        found_embed = False
        found_zero_wd = False
        for g in adamw_opt.param_groups:
            for p in g["params"]:
                if p is mini_model.embed_tokens.weight:
                    assert g["weight_decay"] == 0.01, "Embedding weight decay must be 0.01"
                    found_embed = True
                elif p.ndim in (1, 3):
                    assert g["weight_decay"] == 0.0, f"Norm/Conv weight decay must be 0.0, got {g['weight_decay']}"
                    found_zero_wd = True

        assert found_embed, "Embedding parameter group not found in AdamW"
        assert found_zero_wd, "Zero-decay parameter group not found in AdamW"

    def test_optimizer_step_execution(self, mini_model: RoCanonForCausalLM, device: torch.device):
        """Verifies that executing a step with both Muon and AdamW updates model parameters cleanly."""
        muon_opt, adamw_opt = configure_optimizers(mini_model)

        input_ids = torch.randint(0, mini_model.config.vocab_size, (2, 8), device=device)
        loss, _ = mini_model(input_ids, labels=input_ids)
        loss.backward()

        w_before = {name: p.clone().detach() for name, p in mini_model.named_parameters()}

        muon_opt.step()
        adamw_opt.step()

        for name, p in mini_model.named_parameters():
            if p.requires_grad:
                assert not torch.equal(w_before[name], p), f"Parameter '{name}' was not updated after optimizer step"
                assert torch.isfinite(p).all(), f"Parameter '{name}' contains NaN/Inf after optimizer step"

    def test_wsd_learning_rate_schedule_phases(self):
        """Verifies 3-phase Warmup-Stable-Decay schedule curve with linear warmup, plateau, and 10% floor decay."""
        dummy_p = nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([dummy_p], lr=3e-3)
        sched = WarmupStableDecayLR(opt, warmup=100, stable=300, decay=100, peak_lr=3e-3, min_lr_ratio=0.1)

        assert sched.get_lr(0) == pytest.approx(0.0)
        assert sched.get_lr(50) == pytest.approx(1.5e-3)
        assert sched.get_lr(100) == pytest.approx(3e-3)
        assert sched.get_lr(250) == pytest.approx(3e-3)
        assert sched.get_lr(400) == pytest.approx(3e-3)
        assert sched.get_lr(450) == pytest.approx(3e-3 * 0.55)
        assert sched.get_lr(500) == pytest.approx(3e-4)
        assert sched.get_lr(600) == pytest.approx(3e-4)

    def test_stage_wise_gradient_accumulation(self):
        """Verifies BatchSizeScheduler dynamically doubles accumulation from 8 to 16 at 3.5B token threshold."""
        sched = BatchSizeScheduler(threshold_tokens=3.5e9, initial_accum=8, stage2_accum=16)

        assert sched.get_grad_accum_steps(1.0e9) == 8
        assert sched.get_grad_accum_steps(3.499e9) == 8
        assert sched.get_grad_accum_steps(3.501e9) == 16
        assert sched.get_grad_accum_steps(5.0e9) == 16
