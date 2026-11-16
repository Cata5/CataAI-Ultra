"""
src/worker/state_machine.py
Run state machine enforcing valid transitions and preventing race conditions or duplicate runs.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, List, Optional, Set

from .protocol import RunState

logger = logging.getLogger(__name__)


class InvalidStateTransitionError(Exception):
    """Raised when an illegal state transition is attempted."""
    pass


class RunStateMachine:
    """Manages run state transitions thread-safely."""

    VALID_TRANSITIONS: Dict[RunState, Set[RunState]] = {
        RunState.IDLE: {RunState.STARTING},
        RunState.STARTING: {RunState.RUNNING, RunState.FAILED, RunState.STOPPED},
        RunState.RUNNING: {RunState.PAUSING, RunState.STOPPING, RunState.STOPPED, RunState.COMPLETED, RunState.FAILED},
        RunState.PAUSING: {RunState.PAUSED, RunState.STOPPING, RunState.STOPPED, RunState.FAILED},
        RunState.PAUSED: {RunState.RUNNING, RunState.STOPPING, RunState.STOPPED, RunState.FAILED},
        RunState.STOPPING: {RunState.STOPPED, RunState.FAILED},
        RunState.STOPPED: {RunState.STARTING, RunState.IDLE},
        RunState.COMPLETED: {RunState.STARTING, RunState.IDLE},
        RunState.FAILED: {RunState.STARTING, RunState.IDLE},
    }

    def __init__(
        self,
        initial_state: RunState = RunState.IDLE,
        on_transition: Optional[Callable[[RunState, RunState], None]] = None,
    ) -> None:
        self._state = initial_state
        self._lock = threading.Lock()
        self._on_transition = on_transition

    @property
    def current_state(self) -> RunState:
        with self._lock:
            return self._state

    def is_active(self) -> bool:
        """Returns True if the run is actively running or transitioning."""
        with self._lock:
            return self._state in {
                RunState.STARTING,
                RunState.RUNNING,
                RunState.PAUSING,
                RunState.PAUSED,
                RunState.STOPPING,
            }

    def can_transition_to(self, new_state: RunState) -> bool:
        with self._lock:
            return new_state in self.VALID_TRANSITIONS.get(self._state, set())

    def transition_to(self, new_state: RunState) -> RunState:
        with self._lock:
            if new_state not in self.VALID_TRANSITIONS.get(self._state, set()):
                msg = f"Cannot transition from {self._state.value} to {new_state.value}"
                logger.error(msg)
                raise InvalidStateTransitionError(msg)
            old_state = self._state
            self._state = new_state
            logger.info("RunState transition: %s -> %s", old_state.value, new_state.value)

        if self._on_transition:
            try:
                self._on_transition(old_state, new_state)
            except Exception as e:
                logger.exception("Error in state transition callback: %s", e)

        return new_state

    def force_state(self, state: RunState) -> None:
        """Forces state, used only for recovery after unhandled worker crashes."""
        with self._lock:
            old_state = self._state
            self._state = state
            logger.warning("Forced state change: %s -> %s", old_state.value, state.value)
        if self._on_transition:
            self._on_transition(old_state, state)

