"""
tests/test_v2_features.py
Comprehensive automated test suite for RoCanon-48.5M Training WebUI v2 features:
- SQLite persistence (runs, metrics, alerts, checkpoints, benchmarks)
- Run state machine and transitions
- Atomic checkpointing and restoration
- NVML telemetry & VRAMGuard protection logic
- Dataset validation, analysis, and tokenizer inspection
- Learning rate scheduler mathematical curve calculation
- Multi-stage pipeline management and readiness verification
- Benchmark suite and objective response scoring
"""

import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest
import torch

from src.checkpoints.manager import CheckpointManager
from src.datasets.analyzer import DatasetAnalyzer
from src.datasets.manager import DatasetManager
from src.datasets.validator import DatasetValidator
from src.db.database import StudioDatabase
from src.eval.benchmarks import BenchmarkItem, BenchmarkManager
from src.model.architecture import RoCanonConfig, RoCanonForCausalLM
from src.pipeline.manager import PipelineManager
from src.pipeline.scheduler_math import calculate_schedule_curve, calculate_step_lr
from src.telemetry.hardware import HardwareMonitor
from src.telemetry.vram_guard import VRAMAlertLevel, VRAMGuard, VRAMThresholdConfig
from src.tokenizer.tokenizer import RoCanonTokenizer
from src.worker.protocol import CommandType, EventType, RunState, WorkerCommand, WorkerEvent
from src.worker.state_machine import InvalidStateTransitionError, RunStateMachine
from src.worker.training_worker import _checkpoint_retention_limit, _epoch_session_budget


class TestDatabasePersistence:
    """Tests SQLite persistence for runs, metrics, alerts, and benchmarks."""

    def test_run_creation_and_update(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = StudioDatabase(db_path)

        db.create_run("run_101", "Pretraining Test", current_phase="pretrain")
        run = db.get_run("run_101")
        assert run is not None
        assert run["id"] == "run_101"
        assert run["status"] == "STARTING"
        assert run["current_phase"] == "pretrain"

        db.update_run_status("run_101", "RUNNING", step=50, tokens=100000, best_val_loss=3.14)
        updated = db.get_run("run_101")
        assert updated["status"] == "RUNNING"
        assert updated["total_steps"] == 50
        assert updated["total_tokens"] == 100000
        assert updated["best_val_loss"] == 3.14

    def test_metrics_logging_and_query(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = StudioDatabase(db_path)
        db.create_run("run_102", "Metrics Test")

        batch = [
            {"step": i, "tokens": i * 1000, "train_loss": 5.0 - (i * 0.1), "val_loss": 5.2 - (i * 0.1)}
            for i in range(1, 11)
        ]
        db.log_metrics_batch("run_102", batch)

        metrics = db.get_metrics("run_102", start_step=1, end_step=5)
        assert len(metrics) == 5
        assert metrics[0]["step"] == 1
        assert metrics[4]["step"] == 5

    def test_alerts_logging(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = StudioDatabase(db_path)

        db.log_alert("WARNING", "LOSS_SPIKE", "Loss jumped to 12.5", run_id="run_103", step=42)
        alerts = db.list_alerts("run_103")
        assert len(alerts) == 1
        assert alerts[0]["category"] == "LOSS_SPIKE"
        assert alerts[0]["level"] == "WARNING"
        assert alerts[0]["step"] == 42


class TestRunStateMachine:
    """Tests the run state machine transitions and invalid transition rejections."""

    def test_happy_path_lifecycle(self):
        sm = RunStateMachine(initial_state=RunState.IDLE)
        assert sm.current_state == RunState.IDLE

        sm.transition_to(RunState.STARTING)
        assert sm.current_state == RunState.STARTING

        sm.transition_to(RunState.RUNNING)
        assert sm.current_state == RunState.RUNNING

        sm.transition_to(RunState.PAUSING)
        sm.transition_to(RunState.PAUSED)
        assert sm.current_state == RunState.PAUSED

        sm.transition_to(RunState.RUNNING)
        sm.transition_to(RunState.STOPPING)
        sm.transition_to(RunState.STOPPED)
        assert sm.current_state == RunState.STOPPED

    def test_illegal_transition_rejection(self):
        sm = RunStateMachine(initial_state=RunState.IDLE)
        with pytest.raises(InvalidStateTransitionError):
            sm.transition_to(RunState.RUNNING)  # Must go through STARTING first

        with pytest.raises(InvalidStateTransitionError):
            sm.transition_to(RunState.PAUSED)


class TestCheckpointManager:
    """Tests atomic checkpoint saving, loading, metadata extraction, and deletion."""

    def test_atomic_saving_and_loading(self, tmp_path):
        cm = CheckpointManager(checkpoint_dir=tmp_path)
        config = RoCanonConfig()
        model = RoCanonForCausalLM(config)

        # Save
        saved_path = cm.save_checkpoint(
            model=model,
            step=100,
            tokens_seen=250000,
            phase="pretrain",
            val_loss=2.45,
            train_loss=2.50,
            filename_prefix="test_chk",
        )
        assert saved_path.exists()
        assert not (tmp_path / ".tmp*").exists()  # No temp residue

        # Inspect
        info = cm.inspect_checkpoint(saved_path)
        assert info["step"] == 100
        assert info["tokens_seen"] == 250000
        assert info["val_loss"] == 2.45

        # Load
        res = cm.load_checkpoint(saved_path, model=model)
        assert res["exact_match"] is True
        assert res["step"] == 100

        # Delete
        deleted = cm.delete_checkpoint(saved_path)
        assert deleted is True
        assert not saved_path.exists()


class TestPhaseAndRetentionBudgets:
    """Regression tests for checkpoint retention and phase-relative epochs."""

    def test_retention_setting_is_clamped(self):
        assert _checkpoint_retention_limit({"max_checkpoints_kept": "7"}) == 7
        assert _checkpoint_retention_limit({"max_checkpoints_kept": 0}) == 1
        assert _checkpoint_retention_limit({"max_checkpoints_kept": "bad"}) == 4
        assert _checkpoint_retention_limit({}) == 4

    def test_prune_keeps_best_plus_configured_window(self, tmp_path):
        cm = CheckpointManager(checkpoint_dir=tmp_path)
        names = [
            "best_model.pt",
            "checkpoint_step_10.pt",
            "checkpoint_step_20.pt",
            "checkpoint_step_30.pt",
            "checkpoint_step_40.pt",
        ]
        now = time.time()
        for offset, name in enumerate(names):
            path = tmp_path / name
            path.write_bytes(b"checkpoint")
            os.utime(path, (now + offset, now + offset))

        deleted = cm.prune_checkpoints(keep_last_n=2)
        remaining = sorted(p.name for p in tmp_path.glob("*.pt"))
        assert deleted == ["checkpoint_step_10.pt", "checkpoint_step_20.pt"]
        assert remaining == ["best_model.pt", "checkpoint_step_30.pt", "checkpoint_step_40.pt"]

    def test_prune_keeps_an_anchor_for_each_completed_phase(self, tmp_path):
        cm = CheckpointManager(checkpoint_dir=tmp_path)
        now = time.time()
        files = [
            ("best_model.pt", None, 0),
            ("checkpoint_step_10.pt", "pretrain", 10),
            ("checkpoint_step_100.pt", "pretrain", 20),
            ("checkpoint_step_7290.pt", "chat_sft", 30),
            ("checkpoint_step_7500.pt", "chat_sft", 40),
            ("checkpoint_step_8000.pt", "chat_sft", 50),
        ]
        for name, phase, offset in files:
            path = tmp_path / name
            path.write_bytes(b"checkpoint")
            os.utime(path, (now + offset, now + offset))
            if phase is not None:
                path.with_suffix(".json").write_text(
                    json.dumps({"phase": phase, "step": offset}), encoding="utf-8"
                )

        deleted = cm.prune_checkpoints(keep_last_n=2, current_phase="chat_sft")
        remaining = sorted(p.name for p in tmp_path.glob("*.pt"))
        assert deleted == ["checkpoint_step_10.pt", "checkpoint_step_7290.pt"]
        assert remaining == [
            "best_model.pt",
            "checkpoint_step_100.pt",
            "checkpoint_step_7500.pt",
            "checkpoint_step_8000.pt",
        ]
        # The pretrain anchor keeps its metadata so its phase remains visible.
        assert (tmp_path / "checkpoint_step_100.json").is_file()

    def test_new_sft_phase_gets_full_epoch_budget(self):
        global_target, remaining, steps, target_steps, done = _epoch_session_budget(
            phase_start_tokens=77_000_000,
            total_tokens=77_000_000,
            target_epochs=2,
            train_len=1_229_500,
            tokens_per_step=5_071,
            start_step=6_546,
        )
        assert global_target == 79_459_000
        assert remaining == 2_459_000
        assert done == 0
        assert steps == 485
        assert target_steps == 7_031

    def test_same_phase_resume_keeps_epoch_progress(self):
        global_target, remaining, steps, target_steps, done = _epoch_session_budget(
            phase_start_tokens=77_000_000,
            total_tokens=77_614_750,
            target_epochs=2,
            train_len=1_229_500,
            tokens_per_step=5_071,
            start_step=6_546,
        )
        assert global_target == 79_459_000
        assert remaining == 1_844_250
        assert done == 614_750
        assert steps == 364
        assert target_steps == 6_910


class TestHardwareAndVRAMGuard:
    """Tests NVML hardware monitoring and dynamic VRAM threshold evaluation."""

    def test_vram_guard_thresholds(self):
        cfg = VRAMThresholdConfig(warning_mb=1000.0, critical_mb=700.0, emergency_mb=400.0)
        guard = VRAMGuard(config=cfg)

        lvl, msg = guard.evaluate(free_vram_mb=1500.0)
        assert lvl == VRAMAlertLevel.NORMAL

        lvl, msg = guard.evaluate(free_vram_mb=850.0)
        assert lvl == VRAMAlertLevel.WARNING

        lvl, msg = guard.evaluate(free_vram_mb=550.0)
        assert lvl == VRAMAlertLevel.CRITICAL

        lvl, msg = guard.evaluate(free_vram_mb=250.0)
        assert lvl == VRAMAlertLevel.EMERGENCY

    def test_hardware_monitor_telemetry_fields(self):
        hm = HardwareMonitor()
        telem = hm.read_telemetry()
        assert telem is not None
        assert isinstance(telem.device_name, str)
        assert telem.timestamp > 0
        d = telem.to_dict()
        assert "device_name" in d
        assert "pytorch_allocated_mb" in d


class TestDatasetManagementAndTokenizerInspector:
    """Tests file validation, tokenization analysis, and tokenizer inspector."""

    def test_txt_validation(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("Linia 1 de text.\nLinia 2 de text românesc cu ș și ț.\n", encoding="utf-8")

        val = DatasetValidator()
        res = val.validate_file(f)
        assert res["valid"] is True
        assert res["valid_records"] == 2

    def test_jsonl_validation(self, tmp_path):
        f = tmp_path / "instructions.jsonl"
        lines = [
            json.dumps({"instruction": "Traduce în română", "input": "Hello", "output": "Salut"}),
            json.dumps({"instruction": "Calculează", "input": "2 + 2", "output": "4"}),
        ]
        f.write_text("\n".join(lines), encoding="utf-8")

        val = DatasetValidator()
        res = val.validate_file(f)
        assert res["valid"] is True
        assert res["valid_records"] == 2

    def test_tokenizer_inspector(self):
        tok = RoCanonTokenizer()
        analyzer = DatasetAnalyzer(tokenizer=tok)
        res = analyzer.inspect_tokenization("Salut, RoCanon!")
        assert res["exact_match"] is True
        assert res["token_count"] > 0
        assert len(res["tokens"]) == res["token_count"]

    def test_dataset_diacritics_and_sequence_stats(self):
        tok = RoCanonTokenizer()
        analyzer = DatasetAnalyzer(tokenizer=tok)
        samples = [
            "Propoziție cu diacritice românești: ș, ț, ă, î, â.",
            "Altă propoziție de test pentru RoCanon-48.5M.",
        ]
        stats = analyzer.analyze_samples(samples)
        assert stats["total_samples"] == 2
        assert stats["romanian_diacritics"]["total_count"] > 0
        assert "sequence_lengths" in stats
        assert stats["sequence_lengths"]["min"] > 0


class TestSchedulerMathParity:
    """Tests the exact mathematical curve calculations of the scheduler."""

    def test_wsd_curve_calculation(self):
        curve = calculate_schedule_curve(
            schedule_type="wsd",
            total_steps=1000,
            peak_lr=0.003,
            min_lr=0.0003,
            warmup_steps=100,
            stable_steps=500,
            decay_style="cosine",
            num_points=100,
        )
        assert len(curve) > 50
        # Warmup: start at 0
        assert curve[0]["lr"] == 0.0
        # Peak at step 100 to 600
        mid = [p for p in curve if 100 <= p["step"] <= 600]
        for p in mid:
            assert math.isclose(p["lr"], 0.003, rel_tol=1e-3)
        # End should be min_lr
        assert math.isclose(curve[-1]["lr"], 0.0003, rel_tol=1e-3)

    def test_linear_decay_schedule(self):
        lr_50 = calculate_step_lr(
            step=550,
            schedule_type="linear",
            total_steps=1000,
            peak_lr=0.002,
            min_lr_ratio=0.1,
            warmup_steps=100,
        )
        assert 0.0002 < lr_50 < 0.002


class TestPipelineManagement:
    """Tests multi-stage pipeline configuration, transition readiness, and advancing."""

    def test_pipeline_creation_and_readiness(self, tmp_path):
        db = StudioDatabase(tmp_path / "pipe.db")
        pm = PipelineManager(db=db)

        pid = pm.create_pipeline("Bilingual Pipeline")
        pipe = pm.get_pipeline(pid)
        assert pipe is not None
        assert len(pipe["phases"]) == 4

        # Phase 0 readiness (start from scratch)
        res0 = pm.verify_phase_readiness(pid, target_phase_idx=0)
        assert res0["ready"] is True

        # Phase 1 without prior checkpoint should fail verification
        res1 = pm.verify_phase_readiness(pid, target_phase_idx=1, previous_checkpoint_path=None)
        assert res1["ready"] is False
        assert "checkpoint is required" in res1["error"]


class TestBenchmarkEvaluation:
    """Tests benchmark items, category coverage, and objective metric scoring."""

    def test_benchmark_categories_presence(self, tmp_path):
        db = StudioDatabase(tmp_path / "bench.db")
        bm = BenchmarkManager(db=db)
        cats = bm.get_categories()

        assert "creator_persona" in cats
        assert "hardware_knowledge" in cats
        assert "romanian_understanding" in cats

        persona_items = bm.get_items_by_category("creator_persona")
        assert len(persona_items) == 5
        # Verify Manu Cătălin reference fact
        assert any("Manu Cătălin" in it.reference_answer for it in persona_items)

        hw_items = bm.get_items_by_category("hardware_knowledge")
        assert len(hw_items) == 5
        # Verify RTX 3060 6GB reference fact
        assert any("6GB" in it.reference_answer or "6 GB" in it.reference_answer for it in hw_items)

    def test_response_scoring_metrics(self, tmp_path):
        db = StudioDatabase(tmp_path / "bench.db")
        bm = BenchmarkManager(db=db)
        item = BenchmarkItem(
            id="test_1",
            category="creator_persona",
            prompt="Cine a creat modelul?",
            expected_keywords=["Manu Cătălin"],
        )

        # Full match
        good_resp = "Modelul a fost creat de Manu Cătălin."
        eval_good = bm.evaluate_response(item, good_resp)
        assert eval_good["fact_coverage_score"] == 1.0
        assert eval_good["repetition_detected"] is False
        assert eval_good["overall_score"] >= 0.7

        # Repetition loop penalty
        bad_loop_resp = "Manu Cătălin test test test test test test test test test test."
        eval_loop = bm.evaluate_response(item, bad_loop_resp)
        assert eval_loop["repetition_detected"] is True
        assert eval_loop["overall_score"] < eval_good["overall_score"]


class TestWorkerProtocol:
    """Tests command and event serialization."""

    def test_command_roundtrip(self):
        cmd = WorkerCommand(
            command_type=CommandType.START_RUN,
            payload={"phase": "pretrain", "target_tokens": 5000000},
        )
        d = cmd.to_dict()
        cmd2 = WorkerCommand.from_dict(d)
        assert cmd2.command_type == CommandType.START_RUN
        assert cmd2.payload["phase"] == "pretrain"

    def test_event_roundtrip(self):
        evt = WorkerEvent(
            event_type=EventType.STEP_COMPLETED,
            run_id="run_test",
            step=42,
            payload={"train_loss": 3.12, "grad_norm": 0.8},
        )
        d = evt.to_dict()
        evt2 = WorkerEvent.from_dict(d)
        assert evt2.event_type == EventType.STEP_COMPLETED
        assert evt2.step == 42
        assert evt2.payload["train_loss"] == 3.12

