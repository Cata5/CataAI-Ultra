"""
Regression tests for defects fixed during the 2026 technical audit.

Each test pins one root cause that was verified in the running application, so a
future change cannot silently reintroduce it.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest
import torch

from src.checkpoints.manager import CheckpointManager, safe_torch_load
from src.data.custom_loader import CustomDatasetEngine
from src.db.database import StudioDatabase
from src.worker.training_worker import _first_int


class _PickleExploit:
    """Payload whose deserialization would create a file if unpickling were allowed."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return (open, (str(self.marker), "w"))


def test_safe_torch_load_refuses_pickle_without_executing_it(tmp_path: Path) -> None:
    marker = tmp_path / "pwned.txt"
    payload_path = tmp_path / "malicious.pt"
    torch.save({"model_state_dict": {}, "exploit": _PickleExploit(marker)}, payload_path)

    with pytest.raises(RuntimeError, match="Refusing to deserialize"):
        safe_torch_load(payload_path)

    assert not marker.exists(), "restricted unpickling must never execute the payload"


def test_safe_torch_load_accepts_regular_tensor_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "plain.pt"
    torch.save({"step": 7, "model_state_dict": {"w": torch.zeros(2, 2)}}, path)

    loaded = safe_torch_load(path)

    assert loaded["step"] == 7
    assert loaded["model_state_dict"]["w"].shape == (2, 2)


def test_first_int_preserves_zero_and_alias_priority() -> None:
    # A configured interval of 0 disables the feature and must not be treated as "missing".
    assert _first_int({"eval_freq": 0}, ["validation_interval", "eval_freq"], 100) == 0
    assert _first_int({"validation_interval": 25, "eval_freq": 5}, ["validation_interval", "eval_freq"], 100) == 25
    assert _first_int({}, ["validation_interval"], 100) == 100
    assert _first_int({"eval_freq": None}, ["eval_freq"], 100) == 100
    assert _first_int({"eval_freq": "40"}, ["eval_freq"], 100) == 40


def test_update_run_status_persists_final_loss(tmp_path: Path) -> None:
    db = StudioDatabase(db_path=tmp_path / "runs.db")
    db.create_run("run_1", "audit run")

    db.update_run_status("run_1", "STOPPED", step=12, tokens=34, final_loss=1.5, best_val_loss=1.4)

    row = db.get_run("run_1")
    assert row is not None
    assert row["status"] == "STOPPED"
    assert row["final_loss"] == 1.5
    assert row["best_val_loss"] == 1.4
    assert row["total_steps"] == 12


def test_legacy_database_is_migrated_with_final_loss(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.db"
    conn = sqlite3.connect(legacy)
    conn.executescript(
        """
        CREATE TABLE runs (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            status TEXT NOT NULL,
            current_phase TEXT NOT NULL,
            dataset_source TEXT,
            total_steps INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            best_val_loss REAL,
            config_json TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()

    db = StudioDatabase(db_path=legacy)
    db.create_run("old_run", "legacy")
    db.update_run_status("old_run", "COMPLETED", final_loss=2.25)

    assert db.get_run("old_run")["final_loss"] == 2.25


def test_delete_checkpoint_clears_orphaned_database_row(tmp_path: Path) -> None:
    db = StudioDatabase(db_path=tmp_path / "chk.db")
    chk_dir = tmp_path / "checkpoints"
    chk_dir.mkdir()
    mgr = CheckpointManager(db=db, checkpoint_dir=chk_dir)

    # Simulate a checkpoint whose file was deleted outside the application.
    db.log_checkpoint(
        checkpoint_id="orphan_step_10",
        run_id=None,
        step=10,
        tokens=100,
        train_loss=1.0,
        val_loss=1.1,
        file_path=str(chk_dir / "orphan_step_10.pt"),
        file_size_bytes=1,
    )

    assert mgr.delete_checkpoint("orphan_step_10.pt") is True
    with sqlite3.connect(tmp_path / "chk.db") as conn:
        remaining = conn.execute("SELECT COUNT(*) FROM checkpoints WHERE id = ?", ("orphan_step_10",)).fetchone()[0]
    assert remaining == 0


def test_preset_dataset_resolution_uses_configured_directory(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "phase2_chat_ro.txt"
    dataset.write_text("salut", encoding="utf-8")
    monkeypatch.setenv("LLMMAKER_DATASETS_DIR", str(tmp_path))

    assert CustomDatasetEngine.resolve_dataset_path("chat_persona_ro") == dataset
    assert CustomDatasetEngine.resolve_dataset_path("not_a_preset") is None


def test_delete_checkpoint_reports_missing_entry(tmp_path: Path) -> None:
    db = StudioDatabase(db_path=tmp_path / "chk2.db")
    chk_dir = tmp_path / "checkpoints"
    chk_dir.mkdir()
    mgr = CheckpointManager(db=db, checkpoint_dir=chk_dir)

    assert mgr.delete_checkpoint("does_not_exist.pt") is False

def test_token_cache_loader_never_unpickles_payloads(tmp_path: Path) -> None:
    from src.data.custom_loader import _load_token_cache

    marker = tmp_path / "cache_pwned.txt"
    cache_path = tmp_path / "poisoned_tokens.pt"
    torch.save({"input_ids": _PickleExploit(marker)}, cache_path)

    assert _load_token_cache(cache_path, "cpu", ("input_ids",)) is None
    assert not marker.exists(), "token cache loading must not run embedded pickle code"


def test_token_cache_loader_returns_valid_cache_and_rejects_missing_keys(tmp_path: Path) -> None:
    from src.data.custom_loader import _load_token_cache

    good = tmp_path / "good.pt"
    torch.save({"input_ids": torch.arange(4), "labels": torch.arange(4)}, good)
    loaded = _load_token_cache(good, "cpu", ("input_ids", "labels"))
    assert loaded is not None and loaded["input_ids"].shape == (4,)

    partial = tmp_path / "partial.pt"
    torch.save({"input_ids": torch.arange(4)}, partial)
    assert _load_token_cache(partial, "cpu", ("input_ids", "labels")) is None

