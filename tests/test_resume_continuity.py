"""Regression tests for continuous resume: LR schedule, metrics and run identity."""

from src.dashboard import server
from src.db.database import StudioDatabase
from src.worker.training_worker import (
    _fallback_phase_start,
    _phase_continuation_mismatch,
    _phase_schedule_window,
    _schedule_factor,
)


class TestResumeSchedule:
    def test_global_step_does_not_restart_warmup(self):
        # A run resumed at step 500 of 1000 must continue in the stable phase;
        # the old session-relative math restarted warmup at step 0.
        factor, phase = _schedule_factor("wsd", 500, 1000, 100, 400, 0.1)
        assert phase == "stable"
        assert factor == 1.0

        first_factor, first_phase = _schedule_factor("wsd", 0, 1000, 100, 400, 0.1)
        assert first_phase == "warmup"
        assert first_factor < factor

    def test_decay_continues_from_global_step(self):
        factor, phase = _schedule_factor("wsd", 750, 1000, 100, 400, 0.1)
        assert phase == "decay"
        assert 0.1 <= factor < 1.0

    def test_linear_and_constant_schedules(self):
        linear, linear_phase = _schedule_factor("linear", 750, 1000, 100, 0, 0.1)
        assert linear_phase == "decay"
        assert 0.1 <= linear < 1.0
        constant, constant_phase = _schedule_factor("constant", 750, 1000, 100, 0, 0.1)
        assert (constant, constant_phase) == (1.0, "constant")

    def test_new_phase_gets_its_own_lr_window(self):
        # Pretraining ended at step 6546; SFT runs for 485 steps. The LR must
        # start in SFT warmup, not in the decay tail of the 7031-step global
        # schedule.
        start, total = _phase_schedule_window(7031, 6546)
        assert (start, total) == (6546, 485)

        factor, phase = _schedule_factor("wsd", 0, total, 24, 218, 0.1)
        assert phase == "warmup"
        assert factor < 0.05

        mid, mid_phase = _schedule_factor("wsd", 100, total, 24, 218, 0.1)
        assert mid_phase == "stable"
        assert mid == 1.0

        tail, tail_phase = _schedule_factor("wsd", 400, total, 24, 218, 0.1)
        assert tail_phase == "decay"
        assert 0.1 <= tail < 1.0

    def test_old_pretrain_checkpoint_keeps_phase_origin_at_zero(self):
        assert _fallback_phase_start("pretrain", 6546, 77_031_391) == (0, 0)
        assert _fallback_phase_start("chat_sft", 6546, 77_031_391) == (77_031_391, 6546)

    def test_rebuilt_dataset_starts_a_new_phase(self):
        # The repaired SFT cache has fewer tokens than the corrupted one. Resuming
        # the old checkpoint must not treat the handful of leftover tokens as a
        # full epoch budget for the corrected data.
        assert _phase_continuation_mismatch(
            "chat_sft", "alpaca_ro_clean", 11_489_825,
            "chat_sft", "alpaca_ro_clean", 11_442_393,
        ) == "dataset size changed"
        assert _phase_continuation_mismatch(
            "chat_sft", "alpaca_ro_clean", 11_442_393,
            "chat_sft", "alpaca_ro_clean", 11_442_393,
        ) is None
        assert _phase_continuation_mismatch(
            "pretrain", "alpaca_ro", 1_000,
            "chat_sft", "alpaca_ro_clean", 2_000,
        ) == "phase changed"


class TestResumeMetricsPersistence:
    def test_progress_fields_round_trip(self, tmp_path):
        db = StudioDatabase(tmp_path / "resume.db")
        db.create_run("run_resume", "Resume")
        db.log_metric("run_resume", {
            "step": 500,
            "tokens": 128000,
            "train_loss": 2.5,
            "grad_norm": 1.75,
            "muon_lr": 0.02,
            "adamw_lr": 0.003,
            "current_epoch": 2,
            "epoch_pct": 42.5,
            "tokens_per_epoch": 100000,
            "tokens_in_epoch": 28000,
            "step_time_ms": 123.4,
            "duration_mode": "epochs",
            "target_epochs": 3,
            "sched_phase": "decay",
            "sched_factor": 0.5,
            "eval_interval": 250,
            "next_val_step": 750,
        })

        metric = db.get_metrics("run_resume")[0]
        assert metric["current_epoch"] == 2
        assert metric["epoch_pct"] == 42.5
        assert metric["tokens_per_epoch"] == 100000
        assert metric["tokens_in_epoch"] == 28000
        assert metric["step_time_ms"] == 123.4
        assert metric["duration_mode"] == "epochs"
        assert metric["target_epochs"] == 3
        assert metric["sched_phase"] == "decay"
        assert metric["sched_factor"] == 0.5
        assert metric["eval_interval"] == 250
        assert metric["next_val_step"] == 750

    def test_resume_truncates_stale_future_metrics(self, tmp_path):
        db = StudioDatabase(tmp_path / "branch.db")
        db.create_run("run_branch", "Branch")
        db.log_metrics_batch("run_branch", [
            {"step": step, "tokens": step * 100, "train_loss": 1.0}
            for step in range(1, 6)
        ])
        db.create_run("run_other", "Other")
        db.log_metrics_batch("run_other", [{"step": 9, "tokens": 900, "train_loss": 1.0}])

        removed = db.delete_metrics_after("run_branch", 3)
        assert removed == 2
        assert [m["step"] for m in db.get_metrics("run_branch")] == [1, 2, 3]
        assert [m["step"] for m in db.get_metrics("run_other")] == [9]


class TestResumeRunIdentity:
    def test_uses_originating_run_when_it_exists(self, monkeypatch):
        monkeypatch.setattr(
            server.CHECKPOINT_MGR,
            "inspect_checkpoint",
            lambda _path: {"run_id": "run_old", "workspace_id": "default"},
        )
        monkeypatch.setattr(
            server.DB,
            "get_run",
            lambda run_id: {"id": run_id, "workspace_id": "default"},
        )
        data = {"workspace_id": "default", "resume_checkpoint": "chk.pt"}
        assert server.resolve_resume_run_id(data) == "run_old"

    def test_fresh_start_never_reuses_a_run(self, monkeypatch):
        monkeypatch.setattr(
            server.CHECKPOINT_MGR,
            "inspect_checkpoint",
            lambda _path: {"run_id": "run_old", "workspace_id": "default"},
        )
        monkeypatch.setattr(server.DB, "get_run", lambda _run_id: {"workspace_id": "default"})
        data = {
            "workspace_id": "default",
            "resume_checkpoint": "chk.pt",
            "start_fresh": True,
        }
        assert server.resolve_resume_run_id(data) is None

    def test_workspace_mismatch_keeps_history_isolated(self, monkeypatch):
        monkeypatch.setattr(
            server.CHECKPOINT_MGR,
            "inspect_checkpoint",
            lambda _path: {"run_id": "run_old", "workspace_id": "other"},
        )
        data = {"workspace_id": "default", "resume_checkpoint": "chk.pt"}
        assert server.resolve_resume_run_id(data) is None

    def test_missing_source_run_falls_back_to_new_run(self, monkeypatch):
        monkeypatch.setattr(
            server.CHECKPOINT_MGR,
            "inspect_checkpoint",
            lambda _path: {"run_id": "run_missing", "workspace_id": "default"},
        )
        monkeypatch.setattr(server.DB, "get_run", lambda _run_id: None)
        data = {"workspace_id": "default", "resume_checkpoint": "chk.pt"}
        assert server.resolve_resume_run_id(data) is None
