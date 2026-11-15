"""
Unit and integration tests for Network Studio (RoCanon-48.5M Platform v2).
Verifies:
1. Component correctness (Attentions, Normalizations, Embeddings, FFNs, Convs).
2. Parameter calculation precision.
3. RTX 3060 6GB VRAM memory estimation & budget recommendations.
4. Preflight architecture validation (schema, forward, backward pass, gradient flow).
5. Network Registry CRUD operations & versioned JSON persistence.
6. Dynamic custom module loading and execution.
7. ModelFactory instantiation across all architecture families (GPT, BERT, Hybrid, MLP, CNN).
"""

import os
import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.networks.components import (
    RMSNorm, get_norm_layer, RotaryEmbedding, apply_rotary_pos_emb,
    GroupedQueryAttention, StandardMLP, SwiGLUFFN, GeGLUFFN,
    CausalDepthwiseConv1d, GenericTransformerBlock, get_activation_fn
)
from src.networks.factory import ModelFactory, GenericModularNetwork
from src.networks.templates import ALL_TEMPLATES, get_template_by_id
from src.networks.estimator import ParameterCalculator, MemoryEstimator, BudgetAssistant
from src.networks.validator import ArchitectureValidator
from src.networks.registry import NetworkRegistry
from src.networks.custom_loader import CustomModuleRegistry, CustomModuleBase


class TestNetworkStudioComponents:
    """Tests for foundational neural network building blocks."""

    def test_activations(self):
        x = torch.randn(2, 4)
        for act in ["relu", "gelu", "silu", "tanh", "sigmoid", "leaky_relu"]:
            fn = get_activation_fn(act)
            out = fn(x)
            assert out.shape == x.shape

    def test_rms_norm(self):
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == (2, 10, 64)
        var = out.pow(2).mean(-1)
        assert torch.allclose(var, torch.ones_like(var), atol=0.1)

    def test_get_norm_layer(self):
        for n_type in ["rmsnorm", "layernorm", "batchnorm", "groupnorm", "none"]:
            layer = get_norm_layer(n_type, dim=32)
            assert isinstance(layer, nn.Module)

    def test_rotary_embedding(self):
        rope = RotaryEmbedding(dim=32, max_seq_len=128)
        x = torch.randn(2, 16, 64)
        cos, sin = rope(x, seq_len=16)
        q = torch.randn(2, 4, 16, 32)
        k = torch.randn(2, 4, 16, 32)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_grouped_query_attention_mha(self):
        # MHA: n_q == n_kv
        mha = GroupedQueryAttention(d_model=64, n_q=8, n_kv=8, d_h=8, causal=True)
        x = torch.randn(2, 16, 64)
        out = mha(x)
        assert out.shape == (2, 16, 64)

    def test_grouped_query_attention_gqa(self):
        # GQA: n_q = 8, n_kv = 2 (4:1 ratio)
        gqa = GroupedQueryAttention(d_model=64, n_q=8, n_kv=2, d_h=8, causal=True)
        x = torch.randn(2, 16, 64)
        out = gqa(x)
        assert out.shape == (2, 16, 64)

    def test_grouped_query_attention_mqa(self):
        # MQA: n_q = 8, n_kv = 1
        mqa = GroupedQueryAttention(d_model=64, n_q=8, n_kv=1, d_h=8, causal=True)
        x = torch.randn(2, 16, 64)
        out = mqa(x)
        assert out.shape == (2, 16, 64)

    def test_ffn_variants(self):
        x = torch.randn(2, 16, 64)
        mlp = StandardMLP(d_model=64, d_ff=256, activation="gelu")
        swiglu = SwiGLUFFN(d_model=64, d_ff=256)
        geglu = GeGLUFFN(d_model=64, d_ff=256)

        assert mlp(x).shape == (2, 16, 64)
        assert swiglu(x).shape == (2, 16, 64)
        assert geglu(x).shape == (2, 16, 64)

    def test_causal_depthwise_conv1d(self):
        conv = CausalDepthwiseConv1d(dim=64, kernel_size=3)
        x = torch.randn(2, 16, 64)
        out = conv(x)
        assert out.shape == (2, 16, 64)

    def test_transformer_block(self):
        block = GenericTransformerBlock(
            d_model=64,
            n_q=4,
            n_kv=2,
            d_ff=128,
            norm_type="rmsnorm",
            activation="swiglu",
            has_conv1d=True,
            conv_kernel_size=3,
        )
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == (2, 16, 64)


class TestModelFactoryAndTemplates:
    """Verifies that all pre-packaged templates construct valid models with exact parameter math."""

    @pytest.mark.parametrize("template_dict", ALL_TEMPLATES)
    def test_template_instantiation_and_parameter_count(self, template_dict):
        cfg = template_dict.copy()
        model = ModelFactory.create(cfg)
        assert isinstance(model, nn.Module)

        calc = ParameterCalculator.calculate(cfg)
        actual_total = sum(p.numel() for p in model.parameters())
        actual_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert calc["trainable_parameters"] == actual_trainable
        assert calc["total_parameters"] == actual_total
        assert actual_total > 0

    def test_rocanon_48m_exact_math(self):
        """RoCanon-48.5M architectural baseline invariant: exactly 48,531,968 parameters."""
        cfg = get_template_by_id("rocanon_48m")
        assert cfg is not None
        model = ModelFactory.create(cfg)
        actual_total = sum(p.numel() for p in model.parameters())
        assert actual_total == 48_531_968, f"Expected 48,531,968 parameters, got {actual_total}"


class TestForwardBackwardAndGradients:
    """Verifies complete forward and backward propagation without NaN/Inf."""

    def test_causal_lm_forward_backward(self):
        cfg = get_template_by_id("mini_gpt_15m")
        assert cfg is not None
        cfg["n_layers"] = 2
        model = ModelFactory.create(cfg)
        model.train()

        tokens = torch.randint(0, cfg["vocab_size"], (2, 32), dtype=torch.long)
        targets = torch.randint(0, cfg["vocab_size"], (2, 32), dtype=torch.long)

        logits = model(tokens)
        assert logits.shape == (2, 32, cfg["vocab_size"])

        loss = nn.functional.cross_entropy(logits.view(-1, cfg["vocab_size"]), targets.view(-1))
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"
                assert not torch.isnan(param.grad).any(), f"Parameter {name} gradient contains NaN"

    def test_masked_lm_forward_backward(self):
        cfg = get_template_by_id("bert_encoder_25m")
        assert cfg is not None
        cfg["n_layers"] = 2
        model = ModelFactory.create(cfg)
        model.train()

        tokens = torch.randint(0, cfg["vocab_size"], (2, 32), dtype=torch.long)
        logits = model(tokens)
        # BERT template has a 2-class sequence classification head
        assert logits.shape == (2, 2)

        loss = logits.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_mlp_and_cnn_forward_backward(self):
        # MLP continuous feature model
        mlp_cfg = get_template_by_id("deep_mlp_5m")
        assert mlp_cfg is not None
        mlp_model = ModelFactory.create(mlp_cfg)
        x_mlp = torch.randn(4, 32, mlp_cfg["input_dim"])
        out_mlp = mlp_model(x_mlp)
        # Sequence classification pooled or token-level head
        out_mlp.sum().backward()

        # Conv1D continuous sequence model
        cnn_cfg = get_template_by_id("conv1d_network_4m")
        assert cnn_cfg is not None
        cnn_model = ModelFactory.create(cnn_cfg)
        x_cnn = torch.randn(4, 32, cnn_cfg["input_dim"])
        out_cnn = cnn_model(x_cnn)
        out_cnn.sum().backward()


class TestArchitectureValidator:
    """Verifies schema validation and preflight safety checks."""

    def test_valid_config_passes(self):
        cfg = get_template_by_id("mini_gpt_15m")
        assert cfg is not None
        res = ArchitectureValidator.run_preflight_check(cfg, device="cpu", batch_size=2, seq_len=16)
        assert res["valid"] is True
        assert "output_shape" in res

    def test_missing_required_fields_fails(self):
        bad_cfg = {"name": "Broken"}
        errors = ArchitectureValidator.validate_config_schema(bad_cfg)
        assert len(errors) > 0
        assert any("d_model" in err for err in errors)

    def test_dimension_mismatch_fails(self):
        cfg = get_template_by_id("mini_gpt_15m")
        assert cfg is not None
        cfg["n_q"] = 7
        cfg["n_kv"] = 2  # 7 not divisible by 2
        errors = ArchitectureValidator.validate_config_schema(cfg)
        assert len(errors) > 0
        assert any("divisible" in err for err in errors)

    def test_negative_layer_count_fails(self):
        cfg = get_template_by_id("mini_gpt_15m")
        assert cfg is not None
        cfg["n_layers"] = 0
        errors = ArchitectureValidator.validate_config_schema(cfg)
        assert len(errors) > 0


class TestMemoryEstimatorAndBudgetAssistant:
    """Tests RTX 3060 6GB VRAM allocation calculations."""

    def test_rtx3060_memory_estimation(self):
        cfg = get_template_by_id("rocanon_48m")
        assert cfg is not None
        est = MemoryEstimator.estimate(
            cfg,
            micro_batch_size=16,
            seq_len=512,
            dtype="bfloat16",
            optimizer="adamw",
        )

        assert est["hardware_target"] == "NVIDIA RTX 3060 Laptop (6GB)"
        assert est["model_weights_mb"] > 0
        assert est["gradients_mb"] > 0
        assert est["optimizer_states_mb"] > 0
        assert est["total_training_vram_mb"] > 0
        assert est["safety_status"] in ["SAFE", "TIGHT", "UNLIKELY_TO_FIT"]
        assert est["total_training_vram_gb"] < 5.0

    def test_budget_assistant_presets(self):
        for budget in ["1M", "5M", "15M", "25M", "48.5M", "100M"]:
            rec = BudgetAssistant.get_preset(budget)
            assert rec is not None
            assert "d_model" in rec
            assert "n_layers" in rec
            assert "n_q" in rec

    def test_budget_assistant_recommend(self):
        rec = BudgetAssistant.recommend(target_params=20_000_000)
        assert rec is not None
        assert rec["d_model"] == 384


class TestNetworkRegistry:
    """Tests saving, querying, cloning, and deleting versioned network configurations."""

    def test_registry_lifecycle(self, tmp_path):
        registry = NetworkRegistry(registry_dir=tmp_path)

        # Check default template population
        networks = registry.list_networks()
        assert len(networks) >= 6

        # Create custom
        custom_cfg = get_template_by_id("mini_gpt_15m")
        assert custom_cfg is not None
        custom_cfg["id"] = "my_experiment_v1"
        custom_cfg["name"] = "My Experiment Model"
        saved = registry.save_network("my_experiment_v1", custom_cfg)
        assert saved["id"] == "my_experiment_v1"

        # Read back
        loaded = registry.get_network("my_experiment_v1")
        assert loaded is not None
        assert loaded["name"] == "My Experiment Model"

        # Clone
        cloned = registry.clone_network("my_experiment_v1", new_id="my_experiment_v2", new_name="Cloned Model")
        assert cloned is not None
        assert cloned["id"] == "my_experiment_v2"
        assert cloned["name"] == "Cloned Model"

        # Delete
        success = registry.delete_network("my_experiment_v2")
        assert success is True
        assert registry.get_network("my_experiment_v2") is None

        # Check RoCanon cannot be deleted
        assert registry.delete_network("rocanon_48m") is False


class TestCustomModuleRegistry:
    """Tests dynamic discovery and loading of user modules from networks/custom/."""

    def test_load_custom_modules(self):
        registry = CustomModuleRegistry()
        modules = registry.list_modules()
        assert len(modules) >= 1
        assert any(m["module_name"] == "gated_conv_unit" for m in modules)

        mod_class = registry.get_module("gated_conv_unit")
        assert mod_class is not None
        assert issubclass(mod_class, CustomModuleBase)

        instance = mod_class(dim=64, kernel_size=3)
        x = torch.randn(2, 16, 64)
        out = instance(x)
        assert out.shape == (2, 16, 64)
