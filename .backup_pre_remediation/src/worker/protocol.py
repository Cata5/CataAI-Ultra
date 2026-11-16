"""
src/worker/protocol.py
Commands, events, and state definitions for the decoupled RoCanon-48.5M training worker.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class RunState(str, enum.Enum):
    IDLE = "IDLE"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSING = "PAUSING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CommandType(str, enum.Enum):
    START_RUN = "START_RUN"
    PAUSE_RUN = "PAUSE_RUN"
    RESUME_RUN = "RESUME_RUN"
    STOP_RUN = "STOP_RUN"
    SAVE_CHECKPOINT = "SAVE_CHECKPOINT"
    RUN_VALIDATION = "RUN_VALIDATION"
    RUN_BENCHMARK = "RUN_BENCHMARK"
    LOAD_PIPELINE = "LOAD_PIPELINE"
    SET_LR_MULTIPLIER = "SET_LR_MULTIPLIER"


class EventType(str, enum.Enum):
    RUN_STARTED = "RUN_STARTED"
    STEP_COMPLETED = "STEP_COMPLETED"
    METRICS_UPDATED = "METRICS_UPDATED"
    LOSS_SPIKE = "LOSS_SPIKE"
    GRADIENT_WARNING = "GRADIENT_WARNING"
    CHECKPOINT_SAVED = "CHECKPOINT_SAVED"
    PHASE_COMPLETED = "PHASE_COMPLETED"
    VALIDATION_COMPLETED = "VALIDATION_COMPLETED"
    RUN_PAUSED = "RUN_PAUSED"
    RUN_STOPPED = "RUN_STOPPED"
    RUN_FAILED = "RUN_FAILED"
    RUN_COMPLETED = "RUN_COMPLETED"
    STATE_CHANGED = "STATE_CHANGED"


@dataclass
class WorkerCommand:
    command_type: CommandType
    command_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_type": self.command_type.value,
            "command_id": self.command_id,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkerCommand:
        return cls(
            command_type=CommandType(data["command_type"]),
            command_id=data.get("command_id", uuid.uuid4().hex),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class WorkerEvent:
    event_type: EventType
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    run_id: Optional[str] = None
    step: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "event_id": self.event_id,
            "run_id": self.run_id,
            "step": self.step,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkerEvent:
        return cls(
            event_type=EventType(data["event_type"]),
            event_id=data.get("event_id", uuid.uuid4().hex),
            run_id=data.get("run_id"),
            step=data.get("step", 0),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", time.time()),
        )

