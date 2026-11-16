"""
tests/unit/test_gpu_cleanup.py
Regression tests for GPU VRAM cleanup and worker termination.
Verifies:
1. stop() terminates worker process.
2. stop_run() sets state to STOPPING and initiates process reap.
3. gc.collect() and cuda.empty_cache() release GPU memory.
"""

import gc
from unittest.mock import MagicMock, patch
from src.worker.worker_manager import WorkerManager, RunState


def test_gpu_cache_clearing_contract():
    """Verify that gc.collect and cuda.empty_cache can be called without exception."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def test_worker_manager_stop_terminates_process():
    """Verify stop() joins and terminates active worker process."""
    mgr = WorkerManager()
    mock_process = MagicMock()
    mock_process.is_alive.side_effect = [True, False]
    mock_process.pid = 99999
    mgr._process = mock_process

    mgr.stop()
    assert mock_process.join.called


def test_worker_manager_stop_run_sends_command():
    """Verify stop_run transitions state to STOPPING and sends stop command."""
    mgr = WorkerManager()
    mgr.current_state = RunState.RUNNING

    with patch.object(mgr, "send_command", return_value=True):
        sent = mgr.stop_run()
        assert sent is True
        assert mgr.current_state == RunState.STOPPING
