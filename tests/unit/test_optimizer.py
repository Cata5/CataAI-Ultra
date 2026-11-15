"""
Unit Tests for RoCanon-48.5M Hybrid Optimizer, Schedules & Training Engine.

Supports both pytest and python -m unittest.

Tests cover:
1. Exact parameter partitioning between Muon and AdamW:
   - 112 2D hidden layer weight matrices in Muon (41,943,040 params = 86.42%)
   - Non-2D hidden layer parameters in AdamW (6,588,928 params = 13.58%)
   - Total exact parameter count: strictly 48,531,968 params
2. Differentiated weight decay configuration:
   - Muon 2D matrices: wd = 0.01 (overriding PyTorch default 0.1)
   - AdamW tied embeddings: wd = 0.01
   - AdamW RMSNorm scales & Canon Conv1d filters: wd = 0.0
   - AdamW betas: (0.9, 0.95), beta2 = 0.95
3. Warmup-Stable-Decay (WSD) schedule trajectory:
   - Linear warmup (0 to 2,000 steps)
   - Stable plateau (2,000 steps up to 5.0B tokens at peak LR)
   - Cosine and linear decay (5.0B to 6.8B tokens down to 10% peak LR)
   - Synchronous LR step across both optimizers
   - State dict serialization and restoration
4. Stage-Wise Batch Size Scheduling:
   - Gradient accumulation doubling (8 to 16) at 3.5B tokens
   - Context window expansion (1024 to 2048) at 5.0B tokens
   - Global batch size calculation (65,536 to 131,072 tokens/step)
5. Gradient clipping threshold at max_norm = 1.0
6. Chunked Cross-Entropy Loss:
   - Exact numerical parity with standard PyTorch cross-entropy on loss and gradients (< 1e-5)
   - Correct ignore_index handling
   - True token normalization across micro-batches: sum(L) / sum(T_valid)
7. Muon Quintic Newton-Schulz optimization:
   - Numerical step execution and parameter updates
   - Rejection of non-2D tensors (raises ValueError)
8. MemoryTracker monitoring and < 5.0 GB ceiling assertion
9. End-to-end Trainer step execution and loss decrease
"""

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model import RoCanonConfig, RoCanonForCausalLM
from src.training import (
    ChunkedCrossEntropyLoss,
    Muon,
    StageBatchScheduler,
    TokenLossAccumulator,
    Trainer,
    TrainingConfig,
    WSDScheduler,
    clip_gradient_norm,
    configure_optimizers,
    partition_parameters,
    zeropower_via_newtonschulz,
)
from src.utils.memory import MemoryTracker


class TestHybridOptimizerPartitioning(unittest.TestCase):
    """Tests exact parameter partitioning between Muon and AdamW."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.config = RoCanonConfig()
        cls.full_model = RoCanonForCausalLM(cls.config)

    def test_parameter_count_partitioning_exactness(self):
        """
        Verifies exact analytical partitioning:
        - Muon: 112 2D hidden layer weight matrices totaling strictly 41,943,040 params (86.42%)
        - AdamW: non-2D hidden layer parameters totaling strictly 6,588,928 params (13.58%)
        - Sum: strictly 48,531,968 params
        """
        muon_params, adamw_decay_params, adamw_no_decay_params = partition_parameters(self.full_model)

        # 1. Tensor counts
        self.assertEqual(len(muon_params), 112, f"Expected exactly 112 Muon tensors, got {len(muon_params)}")
        self.assertEqual(len(adamw_decay_params), 1, f"Expected 1 tied embedding tensor, got {len(adamw_decay_params)}")
        self.assertEqual(len(adamw_no_decay_params), 129, f"Expected 129 AdamW no-decay tensors, got {len(adamw_no_decay_params)}")

        # 2. Parameter counts
        muon_count = sum(p.numel() for p in muon_params)
        adamw_decay_count = sum(p.numel() for p in adamw_decay_params)
        adamw_no_decay_count = sum(p.numel() for p in adamw_no_decay_params)
        adamw_total_count = adamw_decay_count + adamw_no_decay_count
        total_count = muon_count + adamw_total_count

        self.assertEqual(muon_count, 41943040, f"Muon param count expected 41,943,040, got {muon_count}")
        self.assertEqual(adamw_decay_count, 6291456, f"AdamW decay param count expected 6,291,456, got {adamw_decay_count}")
        self.assertEqual(adamw_no_decay_count, 297472, f"AdamW no-decay param count expected 297,472, got {adamw_no_decay_count}")
        self.assertEqual(adamw_total_count, 6588928, f"AdamW total param count expected 6,588,928, got {adamw_total_count}")
        self.assertEqual(total_count, 48531968, f"Total combined parameter count expected 48,531,968, got {total_count}")

        # 3. Shape verification: all Muon parameters must be strictly 2D
        for p in muon_params:
            self.assertEqual(p.ndim, 2, f"Muon parameter with invalid ndim {p.ndim} (shape {p.shape})")

        # 4. Configure optimizers wrapper test
        muon_opt, adamw_opt = configure_optimizers(self.full_model)
        self.assertIsNotNone(muon_opt)
        self.assertIsNotNone(adamw_opt)

    def test_differentiated_weight_decays(self):
        """
        Verifies calibrated weight decays and hyperparameters:
        - Muon: weight_decay = 0.01 (overriding PyTorch default 0.1), lr = 3e-3
        - AdamW embeddings group: weight_decay = 0.01, lr = 6e-4
        - AdamW norms and convs group: weight_decay = 0.0, lr = 6e-4
        - AdamW betas: (0.9, 0.95), beta2 = 0.95 NOT 0.999
        - AdamW eps: 1e-8
        """
        muon_opt, adamw_opt = configure_optimizers(
            self.full_model,
            muon_lr=3e-3,
            muon_wd=0.01,
            adamw_lr=6e-4,
            adamw_wd=0.01,
            adamw_betas=(0.9, 0.95),
            adamw_eps=1e-8,
        )

        # Muon checks
        self.assertEqual(len(muon_opt.param_groups), 1)
        muon_group = muon_opt.param_groups[0]
        self.assertEqual(muon_group["lr"], 3e-3)
        self.assertEqual(muon_group["weight_decay"], 0.01)

        # AdamW checks
        self.assertEqual(len(adamw_opt.param_groups), 2)
        decay_group = adamw_opt.param_groups[0]
        no_decay_group = adamw_opt.param_groups[1]

        self.assertEqual(decay_group["lr"], 6e-4)
        self.assertEqual(decay_group["weight_decay"], 0.01)
        self.assertEqual(decay_group["betas"], (0.9, 0.95))
        self.assertEqual(decay_group["eps"], 1e-8)

        self.assertEqual(no_decay_group["lr"], 6e-4)
        self.assertEqual(no_decay_group["weight_decay"], 0.0)
        self.assertEqual(no_decay_group["betas"], (0.9, 0.95))
        self.assertEqual(no_decay_group["eps"], 1e-8)


class TestWSDSchedulerAndBatchSchedules(unittest.TestCase):
    """Tests WSD learning rate schedules and stage batch schedules."""

    def test_wsd_schedule_phases(self):
        """
        Verifies 3-phase WSD schedule:
        1. Linear warmup: 0 at step 0, 0.5 at step 1000, 1.0 at step 2000
        2. Stable plateau: 1.0 up to 5.0B tokens
        3. Cosine decay: decays from 1.0 to 0.1 at 6.8B tokens
        """
        p1 = nn.Parameter(torch.zeros(2, 2))
        p2 = nn.Parameter(torch.zeros(2, 2))
        opt1 = torch.optim.AdamW([p1], lr=3e-3)
        opt2 = torch.optim.AdamW([p2], lr=6e-4)

        scheduler = WSDScheduler(
            optimizers=(opt1, opt2),
            warmup_steps=2000,
            plateau_tokens=5_000_000_000,
            total_tokens=6_800_000_000,
            min_lr_ratio=0.1,
            decay_style="cosine",
        )

        # 1. Warmup checks
        self.assertEqual(scheduler.get_factor(step=0, tokens=0), 0.0)
        self.assertAlmostEqual(scheduler.get_factor(step=1000, tokens=0), 0.5, places=5)
        self.assertAlmostEqual(scheduler.get_factor(step=2000, tokens=0), 1.0, places=5)

        # 2. Plateau checks
        self.assertAlmostEqual(scheduler.get_factor(step=5000, tokens=1_000_000_000), 1.0, places=5)
        self.assertAlmostEqual(scheduler.get_factor(step=50000, tokens=5_000_000_000), 1.0, places=5)

        # 3. Decay checks
        midpoint_factor = scheduler.get_factor(step=60000, tokens=5_900_000_000)
        self.assertAlmostEqual(midpoint_factor, 0.55, places=3)

        end_factor = scheduler.get_factor(step=70000, tokens=6_800_000_000)
        self.assertAlmostEqual(end_factor, 0.1, places=5)

        # 4. Synchronous LR stepping
        scheduler.step(step=2000, tokens=5_000_000_000)
        lrs = scheduler.get_last_lr()
        self.assertAlmostEqual(lrs[0][0], 3e-3, places=6)
        self.assertAlmostEqual(lrs[1][0], 6e-4, places=6)

        # Step into decay
        scheduler.step(step=70000, tokens=6_800_000_000)
        lrs_decay = scheduler.get_last_lr()
        self.assertAlmostEqual(lrs_decay[0][0], 3e-4, places=6)
        self.assertAlmostEqual(lrs_decay[1][0], 6e-5, places=6)

    def test_wsd_linear_decay_and_state_dict(self):
        """Verifies linear decay option and state dict checkpoint restoration."""
        p = nn.Parameter(torch.zeros(2, 2))
        opt = torch.optim.AdamW([p], lr=1e-3)
        scheduler = WSDScheduler(
            optimizers=opt,
            warmup_steps=1000,
            plateau_tokens=2_000_000,
            total_tokens=4_000_000,
            min_lr_ratio=0.1,
            decay_style="linear",
        )

        f_mid = scheduler.get_factor(step=1500, tokens=3_000_000)
        self.assertAlmostEqual(f_mid, 0.55, places=5)

        scheduler.step(step=1500, tokens=3_000_000)
        state = scheduler.state_dict()

        new_scheduler = WSDScheduler(optimizers=opt, warmup_steps=1000)
        new_scheduler.load_state_dict(state)
        self.assertEqual(new_scheduler.current_step, 1500)
        self.assertEqual(new_scheduler.current_tokens, 3_000_000)
        self.assertAlmostEqual(new_scheduler.get_factor(), 0.55, places=5)

    def test_stage_batch_scheduler(self):
        """
        Verifies stage-wise batch schedule:
        - 0 to 3.5B tokens: grad_accum=8, seq_len=1024, micro_batch=8 (65,536 tok/step)
        - 3.5B to 5.0B tokens: grad_accum=16, seq_len=1024, micro_batch=8 (131,072 tok/step)
        - 5.0B to 6.8B tokens: grad_accum=16, seq_len=2048, micro_batch=4 (131,072 tok/step)
        """
        bss = StageBatchScheduler()

        # Stage 1 Early
        self.assertEqual(bss.get_stage(1_000_000_000), "stage1_early")
        self.assertEqual(bss.get_grad_accum_steps(1_000_000_000), 8)
        self.assertEqual(bss.get_seq_len(1_000_000_000), 1024)
        self.assertEqual(bss.get_micro_batch_size(1_000_000_000), 8)
        self.assertEqual(bss.get_global_batch_size_tokens(1_000_000_000), 65536)

        # Stage 1 Late
        self.assertEqual(bss.get_stage(4_000_000_000), "stage1_late")
        self.assertEqual(bss.get_grad_accum_steps(4_000_000_000), 16)
        self.assertEqual(bss.get_seq_len(4_000_000_000), 1024)
        self.assertEqual(bss.get_micro_batch_size(4_000_000_000), 8)
        self.assertEqual(bss.get_global_batch_size_tokens(4_000_000_000), 131072)

        # Stage 2
        self.assertEqual(bss.get_stage(5_500_000_000), "stage2")
        self.assertEqual(bss.get_grad_accum_steps(5_500_000_000), 16)
        self.assertEqual(bss.get_seq_len(5_500_000_000), 2048)
        self.assertEqual(bss.get_micro_batch_size(5_500_000_000), 4)
        self.assertEqual(bss.get_global_batch_size_tokens(5_500_000_000), 131072)

    def test_gradient_norm_clipping(self):
        """Verifies global gradient norm clipping threshold at 1.0."""
        linear = nn.Linear(32, 32)
        x = torch.randn(8, 32)
        y = linear(x).sum() * 1000.0
        y.backward()

        initial_norm = torch.norm(
            torch.stack([torch.norm(p.grad) for p in linear.parameters()])
        ).item()
        self.assertGreater(initial_norm, 1.0)

        clipped_norm = clip_gradient_norm(linear, max_norm=1.0)
        self.assertEqual(clipped_norm, initial_norm)

        post_norm = torch.norm(
            torch.stack([torch.norm(p.grad) for p in linear.parameters()])
        ).item()
        self.assertAlmostEqual(post_norm, 1.0, places=4)


class TestChunkedCrossEntropyLoss(unittest.TestCase):
    """Tests chunked cross-entropy loss and token normalization."""

    def test_numerical_parity_with_pytorch_ce(self):
        """
        Verifies exact numerical parity between ChunkedCrossEntropyLoss
        and standard PyTorch F.cross_entropy on both loss value and gradients.
        """
        torch.manual_seed(42)
        n_tokens, d_model, vocab_size = 256, 64, 128
        chunk_size = 64

        hidden_ref = torch.randn(n_tokens, d_model, requires_grad=True)
        weight_ref = torch.randn(vocab_size, d_model, requires_grad=True)
        targets = torch.randint(0, vocab_size, (n_tokens,))
        targets[10:30] = -100

        # Reference standard calculation
        logits_ref = F.linear(hidden_ref, weight_ref)
        loss_ref = F.cross_entropy(logits_ref, targets, ignore_index=-100, reduction="mean")
        loss_ref.backward()

        # Chunked calculation
        hidden_chunk = hidden_ref.detach().clone().requires_grad_(True)
        weight_chunk = weight_ref.detach().clone().requires_grad_(True)
        loss_fn = ChunkedCrossEntropyLoss(chunk_size=chunk_size, ignore_index=-100)

        loss_chunk, _valid_tokens = loss_fn(hidden_chunk, targets, weight=weight_chunk)
        loss_chunk.backward()

        # Assert loss numerical parity (< 1e-5)
        diff_loss = abs(loss_ref.item() - loss_chunk.item())
        self.assertLess(diff_loss, 1e-5)

        # Assert gradients numerical parity (< 1e-5)
        diff_grad_h = (hidden_ref.grad - hidden_chunk.grad).abs().max().item()
        diff_grad_w = (weight_ref.grad - weight_chunk.grad).abs().max().item()

        self.assertLess(diff_grad_h, 1e-5)
        self.assertLess(diff_grad_w, 1e-5)

    def test_token_normalized_loss_accumulator(self):
        """
        Verifies token-normalized accumulation across micro-batches:
        mean_loss = sum(L) / sum(T_valid).
        """
        accum = TokenLossAccumulator()
        accum.add(loss_sum_value=150.0, valid_tokens=50)
        accum.add(loss_sum_value=300.0, valid_tokens=150)

        self.assertEqual(accum.total_valid_tokens, 200)
        self.assertAlmostEqual(accum.mean_loss, 2.25, places=5)


class TestMuonOptimizerAlgorithm(unittest.TestCase):
    """Tests authentic Muon optimizer and quintic Newton-Schulz algorithm."""

    def test_newton_schulz_orthogonalization(self):
        """Verifies quintic Newton-Schulz polynomial iteration."""
        g = torch.randn(64, 128)
        ortho = zeropower_via_newtonschulz(g, ns_coefficients=(3.4445, -4.775, 2.0315), ns_steps=5)
        self.assertEqual(ortho.shape, g.shape)
        self.assertFalse(torch.isnan(ortho).any())
        self.assertFalse(torch.isinf(ortho).any())

    def test_muon_step_updates_parameter(self):
        """Verifies Muon optimizer step updates 2D weight matrix."""
        w = nn.Parameter(torch.randn(32, 64))
        w_orig = w.clone()
        opt = Muon([w], lr=3e-3, weight_decay=0.01)

        loss = (w ** 2).sum()
        loss.backward()
        opt.step()

        diff = (w - w_orig).abs().max().item()
        self.assertGreater(diff, 0.0)
        self.assertIn("momentum_buffer", opt.state[w])

    def test_muon_rejects_non_2d_tensors(self):
        """Verifies Muon raises ValueError when passed a 1D vector or 3D tensor."""
        p_1d = nn.Parameter(torch.randn(32))
        p_3d = nn.Parameter(torch.randn(32, 1, 4))

        opt_1d = Muon([p_1d], lr=3e-3)
        (p_1d.sum()).backward()
        with self.assertRaises(ValueError):
            opt_1d.step()

        opt_3d = Muon([p_3d], lr=3e-3)
        (p_3d.sum()).backward()
        with self.assertRaises(ValueError):
            opt_3d.step()


class TestMemoryTrackerAndHardwareBudget(unittest.TestCase):
    """Tests MemoryTracker hardware monitoring and VRAM budget assertion."""

    def test_memory_tracker_metrics_and_budget(self):
        tracker = MemoryTracker(max_budget_gb=5.0)
        summary = tracker.summary()
        self.assertIsInstance(summary, dict)
        self.assertIn("peak_allocated_gb", summary)
        self.assertIn("max_budget_gb", summary)
        self.assertEqual(summary["max_budget_gb"], 5.0)

        # assert_within_budget should pass under normal execution
        self.assertTrue(tracker.assert_within_budget(5.0))

        # Test context manager
        with tracker.track("test_block"):
            _ = torch.randn(100, 100)


class TestTrainerExecutionAndLossDecrease(unittest.TestCase):
    """Tests end-to-end multi-step training execution with loss decrease."""

    def test_trainer_multi_step_loss_decrease(self):
        """
        Executes a multi-step training loop on synthetic data verifying:
        - Loss is strictly positive, finite, and decreases over steps
        - Gradients are clipped and applied without NaN/Inf
        - Dual optimizers and scheduler advance synchronously
        """
        torch.manual_seed(42)
        config = RoCanonConfig(n_layers=2, d_model=128, n_q=4, n_kv=2, d_h=32, d_ff=256)
        model = RoCanonForCausalLM(config)

        train_cfg = TrainingConfig(
            device="cpu",
            muon_lr=5e-3,
            adamw_lr=1e-3,
            max_grad_norm=1.0,
        )
        trainer = Trainer(model=model, config=train_cfg)

        target_tokens = torch.randint(0, config.vocab_size, (2, 16))

        losses = []
        for step in range(5):
            micro_batch = [{"input_ids": target_tokens}]
            metrics = trainer.training_step(micro_batch)
            losses.append(metrics["loss"])
            self.assertFalse(math.isnan(metrics["loss"]))
            self.assertFalse(math.isinf(metrics["loss"]))
            self.assertEqual(metrics["step"], step + 1)

        # Assert downward trajectory
        self.assertLess(losses[-1], losses[0])


if __name__ == "__main__":
    unittest.main()
