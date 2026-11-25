import time
from types import SimpleNamespace

from src.training.sweeps import SweepManager
from src.worker.protocol import EventType, RunState


class FakeWorker:
    def __init__(self, state=RunState.IDLE):
        self.current_state = state
        self.started = []
        self.stops = 0
        self.db = None

    def start_run(self, payload):
        self.started.append(payload)
        return True

    def stop_run(self):
        self.stops += 1
        return True


def _event(kind, run_id, **payload):
    return SimpleNamespace(event_type=kind, run_id=run_id, payload=payload)


def _make(tmp_path, worker):
    manager = SweepManager(worker, root=tmp_path / "sweeps")
    sweep = manager.create_sweep(
        "test",
        {"network_id": "n", "dataset_id": "d", "total_steps": 100},
        {"learning_rate": [1e-3, 2e-3]},
        max_trials=2,
        early_stop=True,
        early_stop_ratio=0.05,
    )
    return manager, sweep


def test_reads_do_not_start_training(tmp_path):
    worker = FakeWorker()
    manager, sweep = _make(tmp_path, worker)
    worker.started.clear()

    manager.list_sweeps()
    manager.get_sweep(sweep["id"])

    assert worker.started == []


def test_prune_uses_validation_not_training_loss(tmp_path):
    worker = FakeWorker()
    manager, sweep = _make(tmp_path, worker)
    trial0, trial1 = sweep["trials"]

    manager.on_worker_event(_event(EventType.RUN_COMPLETED, trial0["run_id"], val_loss=0.9))
    assert len(worker.started) == 2

    # A bad training-loss report must not trigger pruning.
    manager.on_worker_event(_event(EventType.METRICS_UPDATED, trial1["run_id"], step=20, loss=99.0))
    time.sleep(0.1)
    assert worker.stops == 0

    # A worse validation loss after the minimum step does.
    manager.on_worker_event(_event(EventType.VALIDATION_COMPLETED, trial1["run_id"], step=20, val_loss=2.0))
    deadline = time.time() + 2
    while worker.stops == 0 and time.time() < deadline:
        time.sleep(0.02)
    assert worker.stops == 1


def test_better_validation_is_not_pruned(tmp_path):
    worker = FakeWorker()
    manager, sweep = _make(tmp_path, worker)
    trial0, trial1 = sweep["trials"]

    manager.on_worker_event(_event(EventType.RUN_COMPLETED, trial0["run_id"], val_loss=0.9))
    manager.on_worker_event(_event(EventType.VALIDATION_COMPLETED, trial1["run_id"], step=20, val_loss=0.8))
    time.sleep(0.1)

    assert worker.stops == 0
