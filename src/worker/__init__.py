"""Worker package for decoupled RoCanon-48.5M training."""

from .protocol import CommandType, EventType, RunState, WorkerCommand, WorkerEvent
from .state_machine import InvalidStateTransitionError, RunStateMachine
from .training_worker import TrainingWorkerProcess, worker_process_entrypoint
from .worker_manager import WorkerManager

__all__ = [
    "RunState",
    "CommandType",
    "EventType",
    "WorkerCommand",
    "WorkerEvent",
    "RunStateMachine",
    "InvalidStateTransitionError",
    "TrainingWorkerProcess",
    "worker_process_entrypoint",
    "WorkerManager",
]

