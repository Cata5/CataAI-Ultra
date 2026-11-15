"""
Unit Tests for RoCanon-48.5M Model Architecture.

Tests cover:
- Exact parameter count assertion (48,531,968 total, 42,240,512 backbone, 6,291,456 tied embedding)
- Layer-by-layer shapes
- Forward pass with input tensor [B, S] returning logits [B, S, 12288]
- Backward pass with loss computation verifying non-zero gradients on all unique parameters
- Causality test for CausalDepthwiseConv1d (future perturbation does not alter past outputs)
- Identity test for zero-initialized convolutions (output equals input at init)
- RoPE norm preservation & QK-Norm normalization
- Weight tying verification (lm_head.weight is embed_tokens.weight)
- Sequence packing with cu_seqlens
- Model weight initialization ranges
"""

import math
import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model import (
    CausalDepthwiseConv1d,
    RMSNorm,
    RoCanonAttention,
    RoCanonBlock,
    RoCanonConfig,
    RoCanonForCausalLM,
    RoCanonMLP,
    RotaryEmbedding,
    apply_rope,
)


@pytest.fixture(scope="module")
def default_config() -> RoCanonConfig:
    return RoCanonConfig()


@pytest.fixture(scope="module")
def full_model(default_config: RoCanonConfig) -> RoCanonForCausalLM:
    torch.manual_seed(42)
    return RoCanonForCausalLM(default_config)


class TestRoCanonArchitecture:
    """Comprehensive test suite for RoCanon-48.5M architecture."""

    def test_exact_parameter_count(self, full_model: RoCanonForCausalLM, default_config: RoCanonConfig):
        """
        Verifies the exact analytical parameter count specified in doc.md Section D:
        - Total unique parameters: strictly 48,531,968
        - Non-embedding backbone parameters: strictly 42,240,512
        - Tied embedding parameters: strictly 6,291,456
        - Total layer parameters per block: strictly 2,640,000
        """
        # 1. Total unique parameters (with weight tying deduplication)
        unique_params = sum(p.numel() for p in set(full_model.parameters()))
        assert unique_params == 48531968, (
            f"Expected total unique parameters 48,531,968, got {unique_params}"
        )

        # 2. Non-embedding backbone parameters
        non_emb_params = full_model.get_num_params(non_embedding=True)
        assert non_emb_params == 42240512, (
            f"Expected non-embedding backbone parameters 42,240,512, got {non_emb_params}"
        )

        # 3. Tied embedding parameters
        emb_params = full_model.embed_tokens.weight.numel()
        assert emb_params == 6291456, (
            f"Expected tied embedding parameters 6,291,456, got {emb_params}"
        )
        assert emb_params == default_config.vocab_size * default_config.d_model

        # 4. Final RMSNorm parameter count
        final_norm_params = full_model.norm.weight.numel()
        assert final_norm_params == 512

        # 5. Parameters per block: exactly 2,640,000
        for i, layer in enumerate(full_model.layers):
            block_params = sum(p.numel() for p in layer.parameters())
            assert block_params == 2640000, (
                f"Layer {i} parameter count mismatch: expected 2,640,000, got {block_params}"
            )

        # 6. Breakdown inside each block
        layer0 = full_model.layers[0]
        # Attn Norm: 512
        assert layer0.attn_norm.weight.numel() == 512
        # Canon-A: 512 * 4 = 2,048
        assert layer0.canon_a.conv.weight.numel() == 2048
        # Attention:
        # W_Q (512x512=262,144) + W_K (512x128=65,536) + W_V (512x128=65,536) + W_O (512x512=262,144)
        # + Canon-B (768x4=3,072) + Q-Norm (64) + K-Norm (64) = 658,560
        attn_params = sum(p.numel() for p in layer0.attn.parameters())
        assert attn_params == 658560
        # FFN Norm: 512
        assert layer0.ffn_norm.weight.numel() == 512
        # Canon-C: 512 * 4 = 2,048
        assert layer0.canon_c.conv.weight.numel() == 2048
        # MLP:
        # W_gate (512x1280=655,360) + W_up (512x1280=655,360) + W_down (1280x512=655,360)
        # + Canon-D (2560x4=10,240) = 1,976,320
        mlp_params = sum(p.numel() for p in layer0.mlp.parameters())
        assert mlp_params == 1976320

        # Sum per block check
        block_sum = 512 + 2048 + 658560 + 512 + 2048 + 1976320
        assert block_sum == 2640000

    def test_layer_by_layer_shapes(self, default_config: RoCanonConfig):
        """Verifies tensor shapes at every intermediate stage."""
        b_sz, seq_len = 2, 8
        c = default_config

        # 1. Embedding
        embed = nn.Embedding(c.vocab_size, c.d_model)
        tokens = torch.randint(0, c.vocab_size, (b_sz, seq_len))
        h = embed(tokens)
        assert h.shape == (b_sz, seq_len, c.d_model)

        # 2. RotaryEmbedding
        rotary = RotaryEmbedding(c.d_h, c.max_seq_len, c.rope_theta)
        cos, sin = rotary.get_cos_sin(seq_len)
        assert cos.shape == (seq_len, c.d_h)
        assert sin.shape == (seq_len, c.d_h)

        # 3. Canon-A
        canon_a = CausalDepthwiseConv1d(c.d_model, c.kernel_size)
        h_canon_a = canon_a(h)
        assert h_canon_a.shape == (b_sz, seq_len, c.d_model)

        # 4. Attention
        attn = RoCanonAttention(c)
        attn_out = attn(h_canon_a, cos, sin)
        assert attn_out.shape == (b_sz, seq_len, c.d_model)

        # 5. Canon-C
        canon_c = CausalDepthwiseConv1d(c.d_model, c.kernel_size)
        h_canon_c = canon_c(h)
        assert h_canon_c.shape == (b_sz, seq_len, c.d_model)

        # 6. MLP
        mlp = RoCanonMLP(c)
        mlp_out = mlp(h_canon_c)
        assert mlp_out.shape == (b_sz, seq_len, c.d_model)

        # 7. Block
        block = RoCanonBlock(c)
        block_out = block(h, cos, sin)
        assert block_out.shape == (b_sz, seq_len, c.d_model)

        # 8. Final Norm + LM Head
        norm = RMSNorm(c.d_model, c.rms_norm_eps)
        lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        logits = lm_head(norm(block_out))
        assert logits.shape == (b_sz, seq_len, c.vocab_size)

    def test_forward_pass_logits(self, full_model: RoCanonForCausalLM, default_config: RoCanonConfig):
        """Verifies forward pass producing logits of shape [B, S, V]."""
        b_sz, seq_len = 2, 16
        input_ids = torch.randint(0, default_config.vocab_size, (b_sz, seq_len))
        logits = full_model(input_ids)
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (b_sz, seq_len, default_config.vocab_size)
        assert not torch.isnan(logits).any(), "Logits contain NaN values"
        assert not torch.isinf(logits).any(), "Logits contain Inf values"

    def test_forward_pass_single_token_and_1d(self, full_model: RoCanonForCausalLM, default_config: RoCanonConfig):
        """Verifies single-token (S=1) and 1D tensor inputs work correctly."""
        # Single token: [1, 1]
        single_tok = torch.tensor([[42]])
        logits_single = full_model(single_tok)
        assert logits_single.shape == (1, 1, default_config.vocab_size)

        # 1D tensor: [S] -> auto unsqueezed to [1, S]
        one_d_tokens = torch.tensor([1, 2, 3, 4, 5])
        logits_1d = full_model(one_d_tokens)
        assert logits_1d.shape == (1, 5, default_config.vocab_size)

    def test_backward_pass_non_zero_gradients(self, full_model: RoCanonForCausalLM, default_config: RoCanonConfig):
        """
        Verifies that a backward pass propagates gradients to ALL unique parameters.
        No parameter in the model may have a None or zero gradient.
        """
        full_model.zero_grad()
        b_sz, seq_len = 2, 8
        input_ids = torch.randint(0, default_config.vocab_size, (b_sz, seq_len))
        labels = torch.randint(0, default_config.vocab_size, (b_sz, seq_len))

        loss, logits = full_model(input_ids, labels=labels)
        assert loss is not None
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        loss.backward()

        unique_params = set(full_model.parameters())
        zero_grad_params = []
        none_grad_params = []

        for name, p in full_model.named_parameters():
            if p in unique_params:
                if p.grad is None:
                    none_grad_params.append(name)
                elif p.grad.abs().sum().item() == 0.0:
                    zero_grad_params.append(name)

        assert len(none_grad_params) == 0, f"Parameters with None grad: {none_grad_params}"
        assert len(zero_grad_params) == 0, f"Parameters with zero grad: {zero_grad_params}"

    def test_causal_depthwise_conv_causality(self):
        """
        Verifies that CausalDepthwiseConv1d is strictly causal:
        Modifying future token t' > t NEVER alters the output at past token t.
        """
        channels = 64
        kernel_size = 4
        seq_len = 16
        conv = CausalDepthwiseConv1d(channels=channels, kernel_size=kernel_size)

        # Set non-zero weights to test convolution actively
        nn.init.normal_(conv.conv.weight, mean=0.0, std=1.0)

        # Create base input sequence
        x1 = torch.randn(1, seq_len, channels)
        out1 = conv(x1)

        # For every token index t, modify all tokens after t (t+1 .. end)
        for t in range(seq_len - 1):
            x2 = x1.clone()
            # Mutate all future tokens t+1 onwards
            x2[:, t + 1 :, :] = torch.randn(1, seq_len - (t + 1), channels)
            out2 = conv(x2)

            # Past and current token outputs up to t must be strictly identical
            diff = (out1[:, : t + 1, :] - out2[:, : t + 1, :]).abs().max().item()
            assert diff == 0.0, (
                f"Causality violated at token {t}: future modification altered past output by {diff}"
            )

    def test_full_model_causality(self, default_config: RoCanonConfig):
        """
        Verifies causality on the full model:
        Modifying future token does not alter logits at earlier tokens.
        """
        # Test on a 2-layer instance for fast causality check
        config = RoCanonConfig(n_layers=2)
        model = RoCanonForCausalLM(config)
        model.eval()

        seq_len = 10
        x1 = torch.randint(0, config.vocab_size, (1, seq_len))
        with torch.no_grad():
            logits1 = model(x1)

        # Modify future tokens after index 4
        x2 = x1.clone()
        x2[:, 5:] = torch.randint(0, config.vocab_size, (1, seq_len - 5))
        with torch.no_grad():
            logits2 = model(x2)

        # Tokens 0 through 4 must produce identical logits
        diff = (logits1[:, :5, :] - logits2[:, :5, :]).abs().max().item()
        assert diff < 1e-4, f"Full model causality violation: max diff = {diff}"

    def test_canon_zero_init_identity(self, full_model: RoCanonForCausalLM):
        """
        Verifies that all Canon depthwise convolutions are strictly initialized to zeros,
        ensuring identity residual pass-through: x + conv(x) == x.
        """
        # Verify across all 16 layers, 4 Canon layers per layer (Canon-A, B, C, D)
        canon_count = 0
        for name, module in full_model.named_modules():
            if isinstance(module, CausalDepthwiseConv1d):
                canon_count += 1
                # Weight must be strictly zero
                weight_norm = module.conv.weight.abs().sum().item()
                assert weight_norm == 0.0, f"Conv weight not zero-initialized in {name}: {weight_norm}"

                # Forward pass must be strictly identical to input
                x = torch.randn(2, 8, module.channels)
                out = module(x)
                assert torch.equal(out, x), f"Identity violated at init for {name}"

        # 16 layers * 4 Canon modules = 64 Canon modules
        assert canon_count == 64, f"Expected 64 Canon modules, found {canon_count}"

    def test_rope_and_qk_norm(self, default_config: RoCanonConfig):
        """
        Verifies:
        1. RoPE rotation preserves vector Euclidean norm: ||R*v|| == ||v||.
        2. QK-Norm normalizes query and key vectors along head dimension d_h=64.
        """
        # 1. RoPE Norm Preservation
        rotary = RotaryEmbedding(dim=default_config.d_h, max_seq_len=64)
        cos, sin = rotary.get_cos_sin(16)
        x = torch.randn(2, default_config.n_q, 16, default_config.d_h)
        x_rotated = apply_rope(x, cos, sin)

        norm_before = torch.norm(x, dim=-1)
        norm_after = torch.norm(x_rotated, dim=-1)
        assert torch.allclose(norm_before, norm_after, atol=1e-5), (
            "RoPE rotation did not preserve vector norms"
        )

        # 2. QK-Norm Verification
        q_norm = RMSNorm(default_config.d_h, eps=default_config.rms_norm_eps)
        k_norm = RMSNorm(default_config.d_h, eps=default_config.rms_norm_eps)

        raw_q = torch.randn(2, 8, default_config.n_q, default_config.d_h) * 10.0
        norm_q = q_norm(raw_q)
        # RMS of normalized vector should be approximately 1.0
        q_rms = torch.sqrt(norm_q.pow(2).mean(dim=-1))
        assert torch.allclose(q_rms, torch.ones_like(q_rms), atol=1e-3), (
            f"QK-Norm did not normalize Q to unit RMS: mean {q_rms.mean().item()}"
        )

    def test_weight_tying(self, full_model: RoCanonForCausalLM):
        """
        Verifies that lm_head.weight is strictly the exact same tensor object as
        embed_tokens.weight (memory address match and gradient sharing).
        """
        assert full_model.lm_head.weight is full_model.embed_tokens.weight, (
            "lm_head.weight and embed_tokens.weight are not identical tensor objects"
        )
        assert full_model.lm_head.weight.data_ptr() == full_model.embed_tokens.weight.data_ptr(), (
            "lm_head.weight and embed_tokens.weight do not share the same memory pointer"
        )

        # Modifying embedding weight modifies lm_head weight
        orig_val = full_model.embed_tokens.weight[0, 0].item()
        full_model.embed_tokens.weight.data[0, 0] += 1.0
        assert abs(full_model.lm_head.weight[0, 0].item() - (orig_val + 1.0)) < 1e-6
        # Revert
        full_model.embed_tokens.weight.data[0, 0] -= 1.0

        # Verify named_parameters deduplication when remove_duplicate=True
        param_list_all = list(full_model.named_parameters(remove_duplicate=False))
        param_list_dedup = list(full_model.named_parameters(remove_duplicate=True))
        assert len(param_list_all) == len(param_list_dedup) + 1, (
            f"Expected named_parameters(remove_duplicate=False) to have 1 more element than deduplicated, "
            f"got {len(param_list_all)} vs {len(param_list_dedup)}"
        )

    def test_cu_seqlens_packed_sequences(self, default_config: RoCanonConfig):
        """
        Verifies forward pass with cu_seqlens for variable-length packed sequence training.
        """
        config = RoCanonConfig(n_layers=2)
        model = RoCanonForCausalLM(config)

        # Packed sequence of 3 documents of lengths 3, 5, 4 (total 12 tokens)
        cu_seqlens = torch.tensor([0, 3, 8, 12], dtype=torch.int32)
        input_ids = torch.randint(0, config.vocab_size, (1, 12))

        logits = model(input_ids, cu_seqlens=cu_seqlens)
        assert logits.shape == (1, 12, config.vocab_size)
        assert not torch.isnan(logits).any()

    def test_loss_computation_and_labels(self, default_config: RoCanonConfig):
        """Verifies forward pass with labels computes cross-entropy loss correctly."""
        config = RoCanonConfig(n_layers=2)
        model = RoCanonForCausalLM(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        labels = torch.randint(0, config.vocab_size, (2, 8))

        # Forward with labels returns (loss, logits)
        result = model(input_ids, labels=labels)
        assert isinstance(result, tuple)
        loss, logits = result

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0.0
        assert logits.shape == (2, 8, config.vocab_size)

        # Verify ignore_index=-100 works properly
        labels_with_ignore = labels.clone()
        labels_with_ignore[:, :4] = -100
        loss_ignored, _ = model(input_ids, labels=labels_with_ignore)
        assert loss_ignored.item() > 0.0

    def test_initialization_ranges(self, default_config: RoCanonConfig):
        """
        Verifies that weights are initialized according to specifications:
        - Linear std = 0.02
        - Residual projections (W_O, W_down) std = 0.02 / sqrt(32) approx 0.0035355
        - Embeddings std = 0.02
        - RMSNorm scale = 1.0
        - Conv weights = 0.0
        """
        config = RoCanonConfig(n_layers=16)
        model = RoCanonForCausalLM(config)

        # Embeddings
        assert abs(model.embed_tokens.weight.std().item() - 0.02) < 0.005

        # Layer 0 Attention
        layer0 = model.layers[0]
        # W_Q, W_K, W_V: standard init std ~ 0.02
        assert abs(layer0.attn.q_proj.weight.std().item() - 0.02) < 0.005
        # W_O: scaled init std ~ 0.02 / sqrt(32) ~ 0.0035355
        expected_scaled_std = 0.02 / math.sqrt(32)
        assert abs(layer0.attn.o_proj.weight.std().item() - expected_scaled_std) < 0.001

        # Layer 0 MLP
        # W_gate, W_up: standard init std ~ 0.02
        assert abs(layer0.mlp.gate_proj.weight.std().item() - 0.02) < 0.005
        # W_down: scaled init std ~ 0.02 / sqrt(32) ~ 0.0035355
        assert abs(layer0.mlp.down_proj.weight.std().item() - expected_scaled_std) < 0.001

        # RMSNorm
        assert torch.all(model.norm.weight == 1.0)
        assert torch.all(layer0.attn_norm.weight == 1.0)
        assert torch.all(layer0.attn.q_norm.weight == 1.0)

        # Canon Convs
        assert torch.all(layer0.canon_a.conv.weight == 0.0)
        assert torch.all(layer0.attn.canon_b.conv.weight == 0.0)
