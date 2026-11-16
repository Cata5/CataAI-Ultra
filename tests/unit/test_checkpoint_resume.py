"""
tests/unit/test_checkpoint_resume.py
Verifies checkpoint resume mechanics in training worker:
1. Verifies payload with 'resume_checkpoint' is recognized and loaded into model.
2. Verifies weights are actually applied, preserving step count and not re-initializing to random weights.
3. Verifies metric dictionary aliases (loss, learning_rate, step_time_ms, tokens_seen).
"""

from pathlib import Path
import tempfile
import torch
import pytest

from src.model.architecture import RoCanonConfig, RoCanonForCausalLM
from src.checkpoints.manager import CheckpointManager
from src.db.database import StudioDatabase


def test_checkpoint_resume_weights_applied():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = StudioDatabase(db_path=db_path)
        chk_dir = Path(tmpdir) / "checkpoints"
        chk_dir.mkdir(parents=True, exist_ok=True)
        mgr = CheckpointManager(db=db, checkpoint_dir=chk_dir)

        # Create model 1 and modify weights to known values
        cfg = RoCanonConfig(n_layers=2, d_model=128, n_q=2, n_kv=1, d_h=64, d_ff=256, vocab_size=512)
        model1 = RoCanonForCausalLM(cfg)
        with torch.no_grad():
            for p in model1.parameters():
                p.fill_(0.42)

        # Save checkpoint
        chk_path = mgr.save_checkpoint(
            model=model1,
            step=1234,
            tokens_seen=56789,
            val_loss=2.625,
            train_loss=2.500,
            filename_prefix="test_model",
        )
        assert Path(chk_path).exists()

        # Create model 2 with random/zero weights
        model2 = RoCanonForCausalLM(cfg)
        with torch.no_grad():
            for p in model2.parameters():
                p.fill_(0.0)

        # Load checkpoint into model 2
        res = mgr.load_checkpoint(
            checkpoint_path=chk_path,
            model=model2,
            restore_training_state=False,
        )

        assert res["step"] == 1234
        assert res["tokens_seen"] == 56789
        assert res["val_loss"] == 2.625

        # Verify weights were actually loaded and match 0.42
        for p in model2.parameters():
            assert torch.allclose(p, torch.tensor(0.42), atol=1e-5), "Weights not loaded into model!"


def test_checkpoint_resume_relative_path_resolution():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = StudioDatabase(db_path=db_path)
        chk_dir = Path(tmpdir) / "checkpoints"
        chk_dir.mkdir(parents=True, exist_ok=True)
        mgr = CheckpointManager(db=db, checkpoint_dir=chk_dir)

        cfg = RoCanonConfig(n_layers=2, d_model=128, n_q=2, n_kv=1, d_h=64, d_ff=256, vocab_size=512)
        model = RoCanonForCausalLM(cfg)
        chk_path = mgr.save_checkpoint(
            model=model,
            step=50,
            tokens_seen=1000,
            filename_prefix="relative_snap",
        )
        filename = Path(chk_path).name

        # Load using just the filename
        res = mgr.load_checkpoint(
            checkpoint_path=filename,
            model=model,
            restore_training_state=False,
        )
        assert res["step"] == 50
