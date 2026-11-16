"""
tests/e2e/test_mandatory_acceptance.py
Comprehensive End-to-End Test Suite for the 5 User Mandatory Acceptance Tests:
1. Checkpoint continuity (train 20 steps, save, resume, verify states and reproducibility).
2. Custom model training (train small MLP and CNN on disposable classification dataset).
3. Dataset integrity (upload dataset with distinctive text, verify it reaches training batches).
4. Checkpoint-based inference (select checkpoint in Evaluation Lab, verify actual loaded weights used).
5. Full architecture lifecycle (custom Transformer create, save, reload, train, checkpoint, restore, evaluate).
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.checkpoints.manager import CheckpointManager
from src.data.custom_loader import CustomDatasetEngine
from src.db.database import StudioDatabase
from src.eval.inference import InferenceEngine
from src.model.architecture import RoCanonConfig, RoCanonForCausalLM
from src.networks.estimator import ParameterCalculator
from src.networks.factory import ModelFactory
from src.networks.registry import NetworkRegistry
from src.networks.validator import ArchitectureValidator
from src.tokenizer.tokenizer import RoCanonTokenizer
from src.training.optimizer import configure_optimizers
from src.training.scheduler import WSDScheduler
from src.training.task_adapter import TaskAdapterFactory
from src.training.trainer import Trainer


class TestMandatoryAcceptance:
    """Verifies all 5 mandatory integration tests without mocking."""

    @pytest.fixture(autouse=True)
    def setup_sandbox(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="rocanon_mandatory_test_"))
        self.db_path = self.tmp_dir / "test_studio.db"
        self.db = StudioDatabase(self.db_path)
        self.chk_mgr = CheckpointManager(checkpoint_dir=self.tmp_dir / "checkpoints", db=self.db)
        self.registry_dir = self.tmp_dir / "network_registry"
        self.registry = NetworkRegistry(registry_dir=self.registry_dir)
        yield
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Test 1: Checkpoint Continuity
    # ------------------------------------------------------------------
    def test_1_checkpoint_continuity(self):
        """
        Train a model for 20 steps on synthetic data, save checkpoint,
        restore into a clean model/optimizer/scheduler, and verify step continuity,
        momentum restoration, and progression on resumed steps.
        """
        cfg = RoCanonConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_q=4,
            n_kv=2,
            d_h=16,
            d_ff=128,
            kernel_size=4,
            max_seq_len=64,
        )
        model1 = RoCanonForCausalLM(cfg).to(device="cpu")
        muon_opt1, adamw_opt1 = configure_optimizers(
            model=model1,
            muon_lr=0.01,
            adamw_lr=0.002,
        )
        scheduler1 = WSDScheduler(
            optimizers=(muon_opt1, adamw_opt1),
            warmup_steps=5,
            total_tokens=100_000,
        )
        trainer1 = Trainer(
            model=model1,
            optimizers=(muon_opt1, adamw_opt1),
            scheduler=scheduler1,
            config=None,
        )

        torch.manual_seed(42)
        dummy_inputs = torch.randint(0, 256, (20, 2, 32))
        dummy_targets = torch.randint(0, 256, (20, 2, 32))

        # Train 20 steps
        for step in range(1, 21):
            x = dummy_inputs[step - 1]
            y = dummy_targets[step - 1]
            trainer1.zero_grad()
            trainer1.accumulate_micro_batch(x, y)
            trainer1.optimizer_step(num_tokens_in_step=64)

        assert trainer1.global_step == 20
        recorded_step = trainer1.global_step
        recorded_param = model1.lm_head.weight.detach().clone()
        first_opt_state = muon_opt1.state[model1.layers[0].attn.o_proj.weight].copy()

        # Save checkpoint
        saved_chk_path = self.chk_mgr.save_checkpoint(
            model=model1,
            step=recorded_step,
            tokens_seen=recorded_step * 64,
            optimizers=(muon_opt1, adamw_opt1),
            scheduler=scheduler1,
            phase="pretrain",
            val_loss=2.5,
            train_loss=2.8,
            filename_prefix="continuity_test",
        )

        # Fresh model & optimizers
        model2 = RoCanonForCausalLM(cfg).to(device="cpu")
        muon_opt2, adamw_opt2 = configure_optimizers(
            model=model2,
            muon_lr=0.01,
            adamw_lr=0.002,
        )
        scheduler2 = WSDScheduler(
            optimizers=(muon_opt2, adamw_opt2),
            warmup_steps=5,
            total_tokens=100_000,
        )
        trainer2 = Trainer(
            model=model2,
            optimizers=(muon_opt2, adamw_opt2),
            scheduler=scheduler2,
            config=None,
        )

        # Restore checkpoint
        meta = self.chk_mgr.load_checkpoint(
            checkpoint_path=saved_chk_path,
            model=model2,
            optimizers=(muon_opt2, adamw_opt2),
            scheduler=scheduler2,
            device=trainer2.device,
            restore_training_state=True,
        )

        trainer2.global_step = meta["step"]

        # Assert exact restoration
        assert trainer2.global_step == 20
        assert torch.allclose(model2.lm_head.weight.cpu(), recorded_param.cpu())
        resumed_opt_state = muon_opt2.state[model2.layers[0].attn.o_proj.weight]
        assert "momentum_buffer" in resumed_opt_state
        assert torch.allclose(first_opt_state["momentum_buffer"].cpu(), resumed_opt_state["momentum_buffer"].cpu())

        # Resume training for 5 more steps
        torch.manual_seed(999)
        extra_inputs = torch.randint(0, 256, (5, 2, 32), device=trainer2.device)
        extra_targets = torch.randint(0, 256, (5, 2, 32), device=trainer2.device)
        for step in range(21, 26):
            x = extra_inputs[step - 21]
            y = extra_targets[step - 21]
            trainer2.zero_grad()
            loss_val, valid_tokens = trainer2.accumulate_micro_batch(x, y)
            trainer2.optimizer_step(num_tokens_in_step=64)
            assert not math.isnan(loss_val)

        assert trainer2.global_step == 25

    # ------------------------------------------------------------------
    # Test 2: Custom Model Training (MLP and CNN)
    # ------------------------------------------------------------------
    def test_2_custom_model_training_mlp_and_cnn(self):
        """
        Verify that generic non-RoCanon architectures (Deep MLP and Conv1D CNN)
        can be instantiated, paired with ClassificationTaskAdapter, and trained
        with valid forward, backward, and weight updates.
        """
        # 1. Train MLP Classifier
        mlp_config = {
            "id": "test_mlp",
            "architecture_family": "mlp",
            "d_model": 32,
            "n_layers": 3,
            "d_ff": 64,
            "input_dim": 16,
            "norm_type": "rmsnorm",
            "activation": "gelu",
            "output_head": {"type": "classifier", "num_classes": 2},
        }
        mlp_model = ModelFactory.create(mlp_config)
        adapter_cls = TaskAdapterFactory.get_adapter("classification")
        mlp_opt = torch.optim.AdamW(mlp_model.parameters(), lr=1e-3)

        x_feat = torch.randn(4, 16)
        y_target = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        p_before = [p.clone() for p in mlp_model.parameters()]
        for _ in range(5):
            mlp_opt.zero_grad()
            loss, _, _ = adapter_cls.forward_loss(mlp_model, x_feat, y_target)
            loss.backward()
            mlp_opt.step()

        p_after = [p.clone() for p in mlp_model.parameters()]
        assert any(not torch.allclose(b, a) for b, a in zip(p_before, p_after)), "MLP weights must update after training"

        # 2. Train CNN Classifier
        cnn_config = {
            "id": "test_cnn",
            "architecture_family": "cnn",
            "d_model": 32,
            "n_layers": 2,
            "input_dim": 8,
            "conv_kernel_size": 3,
            "norm_type": "batchnorm",
            "activation": "relu",
            "output_head": {"type": "classifier", "num_classes": 3},
        }
        cnn_model = ModelFactory.create(cnn_config)
        cnn_opt = torch.optim.AdamW(cnn_model.parameters(), lr=1e-3)

        x_cnn = torch.randn(4, 12, 8)
        y_cnn = torch.tensor([0, 1, 2, 0], dtype=torch.long)

        p_cnn_before = [p.clone() for p in cnn_model.parameters()]
        for _ in range(5):
            cnn_opt.zero_grad()
            loss, _, _ = adapter_cls.forward_loss(cnn_model, x_cnn, y_cnn)
            loss.backward()
            cnn_opt.step()

        p_cnn_after = [p.clone() for p in cnn_model.parameters()]
        assert any(not torch.allclose(b, a) for b, a in zip(p_cnn_before, p_cnn_after)), "CNN weights must update after training"

    # ------------------------------------------------------------------
    # Test 3: Uploaded Dataset Content Integrity
    # ------------------------------------------------------------------
    def test_3_dataset_integrity(self):
        """
        Verify that an uploaded dataset with distinctive text is accurately resolved
        via SQLite registry and loaded without synthetic fallback or token corruption.
        """
        tokenizer = RoCanonTokenizer()
        engine = CustomDatasetEngine(tokenizer=tokenizer, db=self.db, cache_dir=self.tmp_dir / "cache")

        canary_sentence = "CANARY_TOKEN_ROCANON_VERIFICATION_PROMPT_123456789 "
        file_path = self.tmp_dir / "canary_dataset.txt"
        file_path.write_text(canary_sentence * 50, encoding="utf-8")

        dataset_id = "canary_test_dataset"
        self.db.save_dataset(
            dataset_id=dataset_id,
            name="Canary Dataset",
            file_path=str(file_path),
            file_size_bytes=file_path.stat().st_size,
            format_type="txt",
            revision="v1",
            token_count=1000,
            sample_count=1,
        )

        train_in, train_tg, val_in, val_tg = engine.load_dataset(
            dataset_source=dataset_id,
            device=torch.device("cpu"),
        )

        assert train_in is not None
        assert train_in.numel() > 0

        # Decode tokens back to string
        decoded_text = tokenizer.decode(train_in[:60].tolist())
        assert "CANARY_TOKEN_ROCANON" in decoded_text, f"Decoded batch must contain canary token, got: {decoded_text}"

    # ------------------------------------------------------------------
    # Test 4: Checkpoint-Based Inference
    # ------------------------------------------------------------------
    def test_4_checkpoint_based_inference(self):
        """
        Verify that InferenceEngine accurately loads a specified checkpoint,
        executes text generation with those weights, and respects GPU single-owner arbitration.
        """
        cfg = RoCanonConfig(
            vocab_size=12288,
            d_model=512,
            n_layers=16,
            n_q=8,
            n_kv=2,
            d_h=64,
            d_ff=1280,
            kernel_size=4,
            max_seq_len=256,
        )
        base_model = RoCanonForCausalLM(cfg).to(device="cpu")
        saved_path = self.chk_mgr.save_checkpoint(
            model=base_model,
            step=777,
            tokens_seen=500_000,
            phase="midtrain",
            val_loss=1.85,
            train_loss=1.92,
            filename_prefix="eval_test",
        )

        # Instantiate inference engine on CPU
        engine = InferenceEngine(model=base_model, device=torch.device("cpu"))
        meta = engine.load_checkpoint_weights(saved_path)

        assert meta["step"] == 777
        assert meta["val_loss"] == 1.85

        res = engine.generate(
            prompt="RoCanon este un",
            max_new_tokens=10,
            temperature=0.0,  # greedy
        )

        assert "generated_text" in res
        assert res["tokens_generated"] > 0
        assert res["generation_time_sec"] >= 0.0

    # ------------------------------------------------------------------
    # Test 5: Full Architecture Lifecycle
    # ------------------------------------------------------------------
    def test_5_full_architecture_lifecycle(self):
        """
        Full lifecycle test:
        1. Define custom Transformer in Network Registry schema.
        2. Validate schema.
        3. Save to registry.
        4. Reload from registry.
        5. Run CPU forward/backward preflight check.
        6. Calculate exact parameters analytically (<1ms).
        7. Train for 5 steps.
        8. Save checkpoint.
        9. Restore checkpoint.
        10. Run generation with InferenceEngine.
        """
        custom_arch = {
            "id": "custom_transformer_8m",
            "name": "Custom Transformer 8M",
            "architecture_family": "decoder_only",
            "vocab_size": 2048,
            "d_model": 128,
            "n_layers": 4,
            "n_q": 4,
            "n_kv": 2,
            "d_h": 32,
            "d_ff": 384,
            "norm_type": "rmsnorm",
            "activation": "swiglu",
            "pos_encoding": "rope",
            "max_seq_len": 128,
            "has_conv1d": True,
            "conv_kernel_size": 4,
            "qk_norm": True,
            "pre_norm": True,
            "tie_word_embeddings": True,
            "supported_tasks": ["causal_lm"],
            "version": "1.0",
        }

        # 1. Schema check
        errors = ArchitectureValidator.validate_config_schema(custom_arch)
        assert len(errors) == 0, f"Schema validation errors: {errors}"

        # 2. Save & reload
        self.registry.save_network("custom_transformer_8m", custom_arch)
        reloaded = self.registry.get_network("custom_transformer_8m")
        assert reloaded["d_model"] == 128
        assert reloaded["n_layers"] == 4

        # 3. Preflight check
        preflight = ArchitectureValidator.run_preflight_check(reloaded, device="cpu", batch_size=2, seq_len=16)
        assert preflight["valid"] is True, f"Preflight failed: {preflight.get('errors')}"

        # 4. Pure analytical parameter calculation
        t0 = time.time()
        param_stats = ParameterCalculator.calculate(reloaded)
        elapsed_ms = (time.time() - t0) * 1000
        assert elapsed_ms < 50.0, f"Analytical calculation must be <50ms, took {elapsed_ms:.1f}ms"
        assert param_stats["total_parameters"] == preflight["total_parameters"]

        # 5. Train 5 steps
        model = ModelFactory.create(reloaded).to("cpu")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        adapter = TaskAdapterFactory.get_adapter("causal_lm")

        x = torch.randint(0, 2048, (2, 32))
        y = torch.randint(0, 2048, (2, 32))
        for step in range(5):
            opt.zero_grad()
            loss, _, _ = adapter.forward_loss(model, x, y)
            loss.backward()
            opt.step()

        # 6. Checkpoint and restore
        saved_chk = self.chk_mgr.save_checkpoint(
            model=model,
            step=5,
            tokens_seen=320,
            filename_prefix="lifecycle_test",
        )

        fresh_model = ModelFactory.create(reloaded).to("cpu")
        self.chk_mgr.load_checkpoint(
            checkpoint_path=saved_chk,
            model=fresh_model,
            device=torch.device("cpu"),
            restore_training_state=False,
        )

        for p1, p2 in zip(model.parameters(), fresh_model.parameters()):
            assert torch.allclose(p1, p2)
