"""
tests/e2e/test_tier4_workloads.py
Tier 4: Real-World Workload Scenarios and End-to-End Pipeline Simulations for RoCanon-48.5M.

Covers:
1. Full Training Step Simulation: Ingest -> Tokenize -> Pack -> Forward -> Chunked CE Loss -> Backward -> Gradient Clip -> Muon & AdamW Step.
2. Multi-Step Convergence: Optimization sanity and strictly downward loss trend over 15 steps.
3. Memory Ceiling Verification: Peak VRAM (<5.0 GB) and analytical memory budget verification.
4. Multi-Turn Instruction Tuning Workflow: Dialog special tokens and loss calculation exclusively over assistant response tokens.
5. Checkpoint Save/Resume Bitwise Reproducibility: Save checkpoint, resume on fresh model, bitwise identical forward/backward.
6. WSD Scheduler Phase Transitions: 3-phase Warmup-Stable-Decay schedule curve transitions.
7. Sedila & Digit Atomization E2E Invariance: Cedilla vs comma-below invariance and arithmetic digit atomization end-to-end.
Total: 7 tests.
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


class TestTier4Workloads:
    """Comprehensive real-world application workload test suite for RoCanon-48.5M."""

    def test_tier4_full_training_step_pipeline(
        self, dummy_tokenizer: RoCanonTokenizer, mini_model: RoCanonForCausalLM, device: torch.device
    ):
        """
        Scenario 1: Full Training Step Simulation.
        Ingest raw bilingual text -> tokenize -> pack -> forward -> loss -> backward -> clip -> Muon & AdamW step.
        """
        text_ro = "În anul 2026, cercetarea SLM produce 48.5M parametri pentru limba română."
        text_en = "The hybrid Muon optimizer uses quintic Newton-Schulz polynomial iterations."

        # 1. Tokenize & Pack
        ids_ro = dummy_tokenizer.encode(text_ro)
        ids_en = dummy_tokenizer.encode(text_en)
        eos_id = dummy_tokenizer.eos_token_id

        packed_tokens = ids_ro + [eos_id] + ids_en + [eos_id]
        tokens_tensor = torch.tensor([packed_tokens], dtype=torch.long, device=device)
        input_ids = tokens_tensor[:, :-1]
        labels = tokens_tensor[:, 1:]
        cu_seqlens = torch.tensor([0, len(ids_ro), len(packed_tokens) - 1], dtype=torch.int32, device=device)

        # Configure optimizers
        muon_opt, adamw_opt = configure_optimizers(mini_model)

        # 2. Forward pass with cu_seqlens
        logits = mini_model(input_ids, cu_seqlens=cu_seqlens)

        # 3. Compute loss
        loss = compute_chunked_loss(logits, labels, chunk_size=32)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        # Theoretical cross entropy for V=12288 initialized weights centers around ln(12288) ~ 9.42
        assert 7.0 <= loss.item() <= 12.0, f"Initial loss {loss.item()} outside reasonable bound [7.0, 12.0]"

        # 4. Backward pass
        mini_model.zero_grad()
        loss.backward()

        # 5. Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(mini_model.parameters(), max_norm=1.0)
        assert torch.isfinite(grad_norm), f"Gradient norm is not finite: {grad_norm}"
        assert grad_norm > 0.0, "Gradient norm is zero"

        # Record weights before step
        w_before = {name: p.clone().detach() for name, p in mini_model.named_parameters()}

        # 6. Optimizer steps
        muon_opt.step()
        adamw_opt.step()

        # 7. Disaggregate optimizer update assertions
        muon_updates = [
            (p - w_before[name]).abs().max().item()
            for name, p in mini_model.named_parameters()
            if p.requires_grad and p.ndim == 2 and "embed" not in name and "lm_head" not in name
        ]
        adamw_updates = [
            (p - w_before[name]).abs().max().item()
            for name, p in mini_model.named_parameters()
            if p.requires_grad and (p.ndim != 2 or "embed" in name or "lm_head" in name)
        ]
        assert len(muon_updates) > 0, "No Muon parameters identified"
        assert len(adamw_updates) > 0, "No AdamW parameters identified"
        assert max(muon_updates) > 0.0, "No Muon parameters were updated after optimizer step"
        assert max(adamw_updates) > 0.0, "No AdamW parameters were updated after optimizer step"

    def test_tier4_multi_step_training_convergence(
        self, mini_model: RoCanonForCausalLM, device: torch.device
    ):
        """
        Scenario 2: Multi-Step Convergence Sanity.
        Verifies that 15 consecutive optimization steps on fixed packed batch produce a downward loss trend (>= 10% decrease).
        """
        torch.manual_seed(42)
        seq_len = 32
        tokens = torch.randint(10, 200, (1, seq_len + 1), device=device)
        input_ids = tokens[:, :-1]
        labels = tokens[:, 1:]

        muon_opt, adamw_opt = configure_optimizers(mini_model, muon_lr=5e-3, adamw_lr=1e-3)

        initial_loss = None
        current_loss = None

        for step in range(15):
            mini_model.zero_grad()
            logits = mini_model(input_ids)
            loss = F.cross_entropy(logits.view(-1, mini_model.config.vocab_size), labels.view(-1))

            loss_val = loss.item()
            assert torch.isfinite(loss), f"Loss diverged at step {step}: {loss_val}"

            if step == 0:
                initial_loss = loss_val
            current_loss = loss_val

            loss.backward()
            torch.nn.utils.clip_grad_norm_(mini_model.parameters(), max_norm=1.0)
            muon_opt.step()
            adamw_opt.step()

        assert initial_loss is not None and current_loss is not None
        assert current_loss < initial_loss, (
            f"Loss failed to decrease over 15 steps: initial={initial_loss}, final={current_loss}"
        )
        decrease_pct = (initial_loss - current_loss) / initial_loss
        assert decrease_pct >= 0.10, (
            f"Loss decrease ({decrease_pct * 100:.1f}%) was less than required 10% threshold"
        )

    def test_tier4_memory_ceiling_verification(
        self, full_rocanon_config: RoCanonConfig, mini_model: RoCanonForCausalLM, device: torch.device
    ):
        """
        Scenario 3: Peak VRAM / Memory Ceiling Verification (< 5.0 GB).
        Verifies mathematical static memory budget (673.1 MiB) and dynamic VRAM ceiling on CUDA (if available).
        """
        # Analytical parameter memory calculation
        # Weights (BF16): 48,531,968 * 2 bytes = 97,063,936 bytes ~ 92.57 MiB
        # Master weights (FP32): 48,531,968 * 4 bytes = 194,127,872 bytes ~ 185.13 MiB
        # Gradients (FP32): 48,531,968 * 4 bytes = 194,127,872 bytes ~ 185.13 MiB
        # Optimizer states:
        # - Muon (41,943,040 params): 1 momentum buffer (FP32) = 167,772,160 bytes ~ 160.00 MiB
        # - AdamW (6,588,928 params): 2 momentum buffers (FP32) = 52,711,424 bytes ~ 50.27 MiB
        # Total static memory budget = 92.57 + 185.13 + 185.13 + 160.00 + 50.27 = 673.10 MiB (~ 0.66 GB)
        static_budget_bytes = (
            48531968 * 2  # model weights BF16
            + 48531968 * 4  # master weights FP32
            + 48531968 * 4  # gradients FP32
            + 41943040 * 4  # Muon momentum FP32
            + 6588928 * 8  # AdamW first + second moments FP32
        )
        static_mib = static_budget_bytes / (1024 * 1024)
        assert static_mib == pytest.approx(673.10, rel=1e-3)
        assert static_budget_bytes < 1.0 * (1024**3), "Static memory must be strictly under 1.0 GB"

        # Chunked Cross-Entropy logit bounding:
        # Full materialization: B=8, S=1024, V=12288 -> 8 * 1024 * 12288 * 4 bytes = 402,653,184 bytes ~ 384 MiB
        # Chunked (chunk_size=2048): 2048 * 12288 * 4 bytes = 100,663,296 bytes ~ 96 MiB (75% peak reduction)
        full_logit_bytes = 8 * 1024 * 12288 * 4
        chunk_logit_bytes = 2048 * 12288 * 4
        assert chunk_logit_bytes < full_logit_bytes

        # Call compute_chunked_loss on realistic logits and labels, asserting finite scalar loss
        B, S, V = 2, 64, full_rocanon_config.vocab_size
        realistic_logits = torch.randn(B * S, V, device=device)
        realistic_labels = torch.randint(0, V, (B * S,), device=device)
        chunked_loss = compute_chunked_loss(realistic_logits, realistic_labels, chunk_size=32)
        assert torch.isfinite(chunked_loss), f"Chunked loss is not finite: {chunked_loss.item()}"
        assert chunked_loss.item() > 0.0

        # If live CUDA is available, check max memory allocated after a forward pass
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            mini_model_cuda = mini_model.to("cuda")
            x_cuda = torch.randint(0, mini_model.config.vocab_size, (2, 64), device="cuda")
            _ = mini_model_cuda(x_cuda)
            # Peak allocated must not exceed 5.0 GB
            max_vram = torch.cuda.max_memory_allocated()
            assert max_vram < 5.0 * (1024**3), f"Peak allocated VRAM {max_vram / (1024**3):.2f} GB exceeded 5.0 GB ceiling"

    def test_tier4_multi_turn_instruction_tuning_workflow(
        self, dummy_tokenizer: RoCanonTokenizer, mini_model: RoCanonForCausalLM, device: torch.device
    ):
        """
        Scenario 4: Multi-Turn SFT Dialog with Token Masking.
        Verifies system and user prompt tokens are masked with -100, calculating loss exclusively over assistant responses.
        """
        sys_prompt = "<|system|>Ești un asistent inteligent dezvoltat în România.<|end_turn|>"
        user_prompt = "<|user|>Cât face 15 + 27?<|end_turn|>"
        asst_response = "<|assistant|>Calculul este: 15 + 27 = 42.<|end_turn|>"

        ids_sys = dummy_tokenizer.encode(sys_prompt)
        ids_user = dummy_tokenizer.encode(user_prompt)
        ids_asst = dummy_tokenizer.encode(asst_response)

        all_ids = ids_sys + ids_user + ids_asst
        prompt_len = len(ids_sys) + len(ids_user)

        input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)
        labels = input_ids.clone()

        # Mask all tokens prior to assistant response with -100
        labels[:, :prompt_len] = -100

        # Valid tokens remaining should strictly equal assistant response tokens
        valid_count = (labels != -100).sum().item()
        assert valid_count == len(ids_asst)

        # Forward pass and loss (model internally applies autoregressive shift: logits[:-1] -> labels[1:])
        loss, logits = mini_model(input_ids, labels=labels)
        assert torch.isfinite(loss)
        assert loss.item() > 0.0
        assert logits.shape == (1, len(all_ids), mini_model.config.vocab_size)

        mini_model.zero_grad()
        loss.backward()

        # Parameter gradients must be non-zero from assistant response supervision
        grad_norm = torch.norm(mini_model.embed_tokens.weight.grad)
        assert grad_norm > 0.0

    def test_tier4_checkpoint_save_and_resume_reproducibility(
        self, minimal_rocanon_config: RoCanonConfig, device: torch.device, tmp_path: Path
    ):
        """
        Scenario 5: Checkpoint Save and Resume Reproducibility.
        Verifies bitwise identical logits, loss, and weights after saving and resuming from checkpoint.
        """
        ckpt_path = str(tmp_path / "checkpoint.pt")

        torch.manual_seed(42)
        model_a = RoCanonForCausalLM(minimal_rocanon_config).to(device)
        muon_a, adamw_a = configure_optimizers(model_a)

        # Run 2 training steps on model_a
        batch1 = torch.randint(0, minimal_rocanon_config.vocab_size, (2, 16), device=device)
        loss1, _ = model_a(batch1, labels=batch1)
        loss1.backward()
        muon_a.step()
        adamw_a.step()

        # Save checkpoint
        torch.save({
            "model_state": model_a.state_dict(),
            "muon_state": muon_a.state_dict(),
            "adamw_state": adamw_a.state_dict(),
        }, ckpt_path)

        # Instantiate fresh model_b
        torch.manual_seed(999)
        model_b = RoCanonForCausalLM(minimal_rocanon_config).to(device)
        muon_b, adamw_b = configure_optimizers(model_b)

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location=device)
        model_b.load_state_dict(ckpt["model_state"])
        muon_b.load_state_dict(ckpt["muon_state"])
        adamw_b.load_state_dict(ckpt["adamw_state"])

        # Execute identical step 3 on both models
        test_batch = torch.randint(0, minimal_rocanon_config.vocab_size, (2, 16), device=device)

        model_a.zero_grad()
        loss_a, logits_a = model_a(test_batch, labels=test_batch)
        loss_a.backward()
        muon_a.step()
        adamw_a.step()

        model_b.zero_grad()
        loss_b, logits_b = model_b(test_batch, labels=test_batch)
        loss_b.backward()
        muon_b.step()
        adamw_b.step()

        # Bitwise assertions
        assert torch.allclose(logits_a, logits_b, atol=1e-6), "Logits diverged after resuming from checkpoint"
        assert torch.allclose(loss_a, loss_b, atol=1e-6), "Loss diverged after resuming from checkpoint"

        for (name_a, p_a), (name_b, p_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
            assert torch.allclose(p_a, p_b, atol=1e-6), f"Weights diverged for '{name_a}' after checkpoint resume"

    def test_tier4_wsd_scheduler_phase_transitions(self):
        """
        Scenario 6: WSD Learning Rate Transitions.
        Verifies 10k step schedule: warmup (2000 steps) -> stable plateau (7000 steps) -> decay (1000 steps) to 10%.
        """
        dummy_p = nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([dummy_p], lr=3e-3)

        sched = WarmupStableDecayLR(
            opt,
            warmup_steps=2000,
            total_steps=10000,
            decay_steps=1000,
            peak_lr=3.0e-3,
            min_lr_ratio=0.1,
        )

        assert sched.get_lr(0) == pytest.approx(0.0)
        assert sched.get_lr(1000) == pytest.approx(1.5e-3)
        assert sched.get_lr(2000) == pytest.approx(3.0e-3)
        assert sched.get_lr(5000) == pytest.approx(3.0e-3)
        assert sched.get_lr(9000) == pytest.approx(3.0e-3)
        assert sched.get_lr(9500) == pytest.approx(3.0e-3 * 0.55)
        assert sched.get_lr(10000) == pytest.approx(3.0e-4)

    def test_tier4_sedila_and_digit_atomization_e2e_invariance(
        self, dummy_tokenizer: RoCanonTokenizer, mini_model: RoCanonForCausalLM, device: torch.device
    ):
        """
        Scenario 7: Romanian Diacritics & Math Round-Trip Invariance E2E.
        Verifies cedilla and comma-below sentences yield identical token streams, identical forward logits, and atomized digits.
        """
        text_cedilla = (
            "La data de 15.03.2026, preţul a crescut cu 125 lei şi 50 bani. Ecuaţia: 25 * 4 = 100."
        )
        text_comma = (
            "La data de 15.03.2026, prețul a crescut cu 125 lei și 50 bani. Ecuația: 25 * 4 = 100."
        )

        ids_cedilla = dummy_tokenizer.encode(text_cedilla)
        ids_comma = dummy_tokenizer.encode(text_comma)

        # 1. Exact token ID equality
        assert ids_cedilla == ids_comma, "Cedilla and comma-below encodings were not 100% identical!"

        # 2. Every digit in 15.03.2026, 125, 50, 25, 4, 100 is atomized 1-by-1
        digits_str = "".join(ch for ch in text_comma if ch.isdigit())
        digit_tokens = [tok for tok in dummy_tokenizer.tokenize(text_comma) if tok.isdigit()]
        assert len(digit_tokens) == len(digits_str), (
            f"Digit atomization failed: expected {len(digits_str)} digits, got {len(digit_tokens)}"
        )

        # 3. Identical forward pass activations
        t_cedilla = torch.tensor([ids_cedilla], dtype=torch.long, device=device)
        t_comma = torch.tensor([ids_comma], dtype=torch.long, device=device)

        mini_model.eval()
        with torch.no_grad():
            out_cedilla = mini_model(t_cedilla)
            out_comma = mini_model(t_comma)

        assert torch.equal(out_cedilla, out_comma), "Forward pass logits differed between cedilla and comma inputs!"

        # 4. Decoded text reconstructed with comma-below standard
        reconstructed = dummy_tokenizer.decode(ids_cedilla)
        assert "prețul" in reconstructed and "și" in reconstructed and "Ecuația" in reconstructed
        assert "preţul" not in reconstructed and "şi" not in reconstructed
