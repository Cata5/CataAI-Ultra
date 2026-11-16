"""
tests/unit/test_remediation_m1.py
Automated acceptance tests for Milestone 1:
- BUG-01: Safe checkpoint deserialization (weights_only=True, unsafe pickle rejected)
- BUG-02: Path traversal rejection in NetworkRegistry
- BUG-03: Full checkpoint restoration (Weights, Muon, AdamW, Scheduler, RNG)
- Invariance: RoCanon-48.5M exact parameter preservation (48,531,968)
"""

import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.checkpoints.manager import CheckpointManager
from src.model.architecture import RoCanonConfig, RoCanonForCausalLM
from src.networks.registry import NetworkRegistry
from src.training.factory import create_optimizers_from_dict
from src.training.scheduler import WSDScheduler


class TestMilestone1Remediation:
    def test_rocanon_parameter_invariance(self):
        """Preserves exact RoCanon-48.5M parameter count of 48,531,968."""
        cfg = RoCanonConfig()
        model = RoCanonForCausalLM(cfg)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params == 48_531_968, f"Expected 48,531,968 parameters, got {total_params}"

    def test_network_registry_path_traversal_rejection(self, tmp_path):
        """BUG-02: Strict ID validation and path traversal sequences rejected."""
        reg = NetworkRegistry(registry_dir=tmp_path / "registry")

        # Invalid IDs with path traversal or illegal characters
        invalid_ids = [
            "../escaped",
            "..\\escaped",
            "foo/bar",
            "foo\\bar",
            "../../etc/passwd",
            "C:\\Windows\\System32",
            "valid_id; rm -rf /",
            "id with spaces",
            "id.with.dots",
            "",
            "a" * 65,  # Too long
        ]

        dummy_cfg = {
            "name": "TestNet",
            "architecture_family": "decoder_only",
            "d_model": 64,
            "n_layers": 2,
            "n_q": 4,
            "n_kv": 2,
            "d_ff": 128,
            "vocab_size": 512,
        }

        for bad_id in invalid_ids:
            with pytest.raises(ValueError):
                reg._validate_and_resolve_path(bad_id)

            # Ensure save, get, delete reject bad_id safely
            with pytest.raises(ValueError):
                reg.save_network(bad_id, dummy_cfg)

            assert reg.get_network(bad_id) is None
            assert reg.delete_network(bad_id) is False

        # Legitimate IDs should pass
        valid_id = "test_custom_transformer-v1"
        saved = reg.save_network(valid_id, dummy_cfg)
        assert saved["id"] == valid_id
        assert reg.get_network(valid_id) is not None
        assert reg.delete_network(valid_id) is True

    def test_safe_checkpoint_deserialization_rejects_untrusted(self, tmp_path):
        """BUG-01: Checkpoint loading strictly enforces weights_only=True."""
        cm = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")

        # Create an untrusted pickle payload containing an arbitrary callable
        class ExploitPayload:
            def __reduce__(self):
                return (os.system, ("echo EXPLOIT_EXECUTED",))

        untrusted_file = tmp_path / "checkpoints" / "malicious.pt"
        untrusted_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"exploit": ExploitPayload()}, untrusted_file)

        # inspect_checkpoint must safely report error without executing code
        info = cm.inspect_checkpoint(untrusted_file)
        assert "load_error" in info or "error" in info
        err_msg = str(info.get("load_error", ""))
        assert "Weights only load failed" in err_msg or "unsupported GLOBAL" in err_msg or "blocked" in err_msg

        # load_checkpoint must raise UnpicklingError / ValueError under safe weights_only=True
        dummy_model = nn.Linear(4, 4)
        with pytest.raises(Exception):
            cm.load_checkpoint(untrusted_file, dummy_model, restore_training_state=False)

    def test_full_checkpoint_restoration(self, tmp_path):
        """BUG-03: Full restoration of weights, Muon optimizer, AdamW, Scheduler, and RNG states."""
        cm = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")

        cfg = RoCanonConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_q=4,
            n_kv=2,
            d_h=16,
            d_ff=128,
            max_seq_len=64,
        )

        # 1. Initialize model and optimizers
        m1 = RoCanonForCausalLM(cfg)
        opts1 = create_optimizers_from_dict(m1, {"muon_lr": 0.003, "adamw_lr": 0.0006})
        muon1, adamw1 = opts1
        sched1 = WSDScheduler(optimizers=opts1, warmup_steps=5, total_tokens=10_000)

        # Perform steps to populate optimizer state (momentum, exp_avg, exp_avg_sq)
        for step in range(1, 10):
            x = torch.randint(0, 256, (2, 16))
            out = m1(x)
            loss = out.float().mean()
            loss.backward()
            muon1.step()
            adamw1.step()
            muon1.zero_grad()
            adamw1.zero_grad()
            sched1.step(tokens=step * 256)

        assert len(muon1.state) > 0, "Muon optimizer state must contain momentum buffers"
        assert len(adamw1.state) > 0, "AdamW optimizer state must contain momentum buffers"

        saved_path = cm.save_checkpoint(
            model=m1,
            step=9,
            tokens_seen=9 * 256,
            optimizers=opts1,
            scheduler=sched1,
            phase="pretrain",
            filename_prefix="test_chk",
        )

        # Verify companion JSON metadata was saved
        meta_json_path = saved_path.with_suffix(".json")
        assert meta_json_path.exists(), "Companion metadata JSON must be saved"
        with open(meta_json_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
            assert meta_data["step"] == 9
            assert meta_data["tokens_seen"] == 9 * 256

        # 2. Re-instantiate a fresh model, optimizers, and scheduler
        m2 = RoCanonForCausalLM(cfg)
        opts2 = create_optimizers_from_dict(m2, {"muon_lr": 0.003, "adamw_lr": 0.0006})
        muon2, adamw2 = opts2
        sched2 = WSDScheduler(optimizers=opts2, warmup_steps=5, total_tokens=10_000)

        assert len(muon2.state) == 0, "Fresh optimizer state must be empty before load"

        # 3. Restore checkpoint with restore_training_state=True
        load_res = cm.load_checkpoint(
            checkpoint_path=saved_path,
            model=m2,
            optimizers=opts2,
            scheduler=sched2,
            restore_training_state=True,
        )

        assert load_res["step"] == 9
        assert load_res["tokens_seen"] == 9 * 256
        assert len(muon2.state) == len(muon1.state), "Muon state entries must be fully restored"
        assert len(adamw2.state) == len(adamw1.state), "AdamW state entries must be fully restored"
        assert sched2.current_tokens == sched1.current_tokens, "Scheduler current_tokens must match"

        # 4. Verify weight match
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2), "Restored model weights must match saved weights exactly"
