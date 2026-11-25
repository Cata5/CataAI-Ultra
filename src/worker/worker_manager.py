"""
src/worker/worker_manager.py
Backend Control Plane manager coordinating the decoupled training worker process.
Handles process lifecycle, validates commands, dispatches events, and feeds WebSockets.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from ..db.database import StudioDatabase
from .protocol import CommandType, EventType, RunState, WorkerCommand, WorkerEvent
from .state_machine import RunStateMachine

logger = logging.getLogger(__name__)


class WorkerManager:
    """Backend controller managing the training worker process and event streaming."""

    def __init__(self, db: Optional[StudioDatabase] = None, device_str: Optional[str] = None) -> None:
        self.db = db or StudioDatabase()
        self.device_str = device_str
        self._ctx = mp.get_context("spawn")
        self._cmd_queue: Optional[Any] = None
        self._event_queue: Optional[Any] = None
        self._process: Optional[mp.Process] = None
        self._listener_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        # Incremented whenever a new run is dispatched. A pending stop-reaper
        # only kills the process if no newer run has taken ownership of it.
        self._reap_token = 0

        # State tracking on control plane
        self.current_state: RunState = RunState.IDLE
        self.state_machine = RunStateMachine()
        self.current_run_id: Optional[str] = None
        self.latest_metrics: Dict[str, Any] = {}
        self.latest_alerts: List[Dict[str, Any]] = []

        # Event listeners (e.g. WebSocket broadcasters)
        self._subscribers: Set[Callable[[WorkerEvent], None]] = set()

    def subscribe(self, callback: Callable[[WorkerEvent], None]) -> None:
        with self._lock:
            self._subscribers.add(callback)

    def unsubscribe(self, callback: Callable[[WorkerEvent], None]) -> None:
        with self._lock:
            self._subscribers.discard(callback)

    def _broadcast(self, event: WorkerEvent) -> None:
        with self._lock:
            subs = list(self._subscribers)
        for sub in subs:
            try:
                sub(event)
            except Exception as e:
                logger.debug("Error in event subscriber: %s", e)

    def start(self) -> None:
        """Starts worker process and event listener."""
        from .training_worker import worker_process_entrypoint

        with self._lock:
            if self._process is not None and self._process.is_alive():
                logger.warning("Worker process already running.")
                return

            self._cmd_queue = self._ctx.Queue()
            self._event_queue = self._ctx.Queue()
            self._running = True

            self._process = self._ctx.Process(
                target=worker_process_entrypoint,
                args=(self._cmd_queue, self._event_queue, self.device_str, str(self.db.db_path)),
                daemon=True,
            )
            self._process.start()
            logger.info("Spawned worker process PID: %s", self._process.pid)

            self._listener_thread = threading.Thread(target=self._event_listener_loop, daemon=True)
            self._listener_thread.start()

    def stop(self) -> None:
        """Gracefully stops worker process."""
        with self._lock:
            self._running = False
            if self._cmd_queue:
                try:
                    self._cmd_queue.put(WorkerCommand(command_type=CommandType.STOP_RUN).to_dict())
                except Exception:
                    pass

        if self._process and self._process.is_alive():
            self._process.join(timeout=3.0)
            if self._process.is_alive():
                logger.warning("Worker did not exit within timeout. Terminating...")
                self._process.terminate()

    def _event_listener_loop(self) -> None:
        """Background thread reading events from worker queue and broadcasting."""
        while self._running:
            if not self._event_queue:
                time.sleep(0.1)
                continue

            try:
                event_dict = self._event_queue.get(timeout=0.1)
                event = WorkerEvent.from_dict(event_dict)
                self._process_event(event)
            except Exception:
                # Queue empty or timeout
                pass

    def _process_event(self, event: WorkerEvent) -> None:
        et = event.event_type
        if et == EventType.STATE_CHANGED:
            new_state_str = event.payload.get("new_state", "IDLE")
            try:
                self.current_state = RunState(new_state_str)
                self.state_machine.current_state = self.current_state
            except ValueError:
                pass
        elif et == EventType.STEP_COMPLETED:
            self.latest_metrics.update(event.payload)
        elif et == EventType.METRICS_UPDATED:
            self.latest_metrics.update(event.payload)
        elif et in (EventType.LOSS_SPIKE, EventType.GRADIENT_WARNING):
            alert = {
                "type": et.value,
                "step": event.step,
                "payload": event.payload,
                "timestamp": event.timestamp,
            }
            self.latest_alerts.append(alert)
            if len(self.latest_alerts) > 50:
                self.latest_alerts.pop(0)
        elif et == EventType.RUN_STARTED:
            self.current_run_id = event.run_id
            self.current_state = RunState.RUNNING
            self.state_machine.current_state = self.current_state
            self.latest_metrics.update(event.payload)
        elif et in (EventType.RUN_STOPPED, EventType.RUN_COMPLETED, EventType.RUN_FAILED):
            if et == EventType.RUN_STOPPED:
                self.current_state = RunState.STOPPED
            elif et == EventType.RUN_COMPLETED:
                self.current_state = RunState.COMPLETED
            elif et == EventType.RUN_FAILED:
                self.current_state = RunState.FAILED
            self.state_machine.current_state = self.current_state

        # Broadcast to UI WebSockets
        self._broadcast(event)

    def send_command(self, cmd: WorkerCommand) -> bool:
        """Sends command to worker process after validation."""
        if not self._process or not self._process.is_alive():
            logger.info("Worker process not running; starting it now...")
            self.start()

        if self._cmd_queue:
            self._cmd_queue.put(cmd.to_dict())
            logger.info("Dispatched command %s to worker", cmd.command_type.value)
            return True
        return False

    def start_run(self, payload: Dict[str, Any]) -> bool:
        if self.current_state in (RunState.RUNNING, RunState.STARTING):
            logger.warning("Cannot start run: worker is already %s", self.current_state.value)
            return False
        with self._lock:
            self._reap_token += 1
        cmd = WorkerCommand(command_type=CommandType.START_RUN, payload=payload)
        return self.send_command(cmd)

    def pause_run(self) -> bool:
        cmd = WorkerCommand(command_type=CommandType.PAUSE_RUN)
        return self.send_command(cmd)

    def resume_run(self) -> bool:
        cmd = WorkerCommand(command_type=CommandType.RESUME_RUN)
        return self.send_command(cmd)

    def stop_run(self) -> bool:
        cmd = WorkerCommand(command_type=CommandType.STOP_RUN)
        sent = self.send_command(cmd)
        if sent and self.current_state in (RunState.RUNNING, RunState.PAUSED, RunState.STARTING):
            self.current_state = RunState.STOPPING

        # Ensure worker process exits cleanly and CUDA context is completely released back to the OS
        with self._lock:
            reap_token = self._reap_token

        def _reap_worker_process():
            time.sleep(2.5)
            with self._lock:
                # A sweep stops one trial and immediately starts the next on the
                # same long-lived worker. Do not let the old stop's reaper kill
                # the process that now owns a newer run.
                if reap_token != self._reap_token:
                    return
                if self._process and self._process.is_alive():
                    try:
                        logger.info("Reaping worker process to guarantee 100%% CUDA context release.")
                        self._process.join(timeout=1.5)
                        if self._process and self._process.is_alive():
                            self._process.terminate()
                            self._process.join(timeout=1.0)
                    except Exception as e:
                        logger.warning("Error reaping worker process: %s", e)
                    self._process = None
                    self.current_state = RunState.STOPPED

        threading.Thread(target=_reap_worker_process, daemon=True).start()
        return sent

    def trigger_validation(self) -> bool:
        cmd = WorkerCommand(command_type=CommandType.RUN_VALIDATION)
        return self.send_command(cmd)

    def request_checkpoint(self) -> bool:
        cmd = WorkerCommand(command_type=CommandType.SAVE_CHECKPOINT)
        return self.send_command(cmd)

    def set_lr_multiplier(self, multiplier: float) -> bool:
        cmd = WorkerCommand(command_type=CommandType.SET_LR_MULTIPLIER, payload={"multiplier": multiplier})
        return self.send_command(cmd)
