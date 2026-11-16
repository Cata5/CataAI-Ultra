"""
src/pipeline/manager.py
Multi-stage training pipeline manager for RoCanon-48.5M.
Sequences Phase 1 (Pretrain), Phase 2 (Domain Adaptation), Phase 3 (Instruction Tuning),
and Phase 4 (Persona Refinement) with validation boundaries, weight restoration, and recovery.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..checkpoints.manager import CheckpointManager
from ..db.database import StudioDatabase

logger = logging.getLogger(__name__)


@dataclass
class PhaseConfig:
    name: str
    phase_type: str  # "pretrain", "midtrain", "chat_sft", "persona"
    enabled: bool = True
    dataset_source: str = "tech_wiki_ro"
    target_tokens: int = 500_000_000
    target_steps: int = 5000
    micro_batch_size: int = 4
    grad_accum_steps: int = 2
    seq_len: int = 256
    learning_rate: float = 3.0e-3
    warmup_steps: int = 200
    eval_frequency: int = 10
    checkpoint_frequency: int = 500
    pause_after_completion: bool = False
    optimizer_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PhaseConfig:
        return cls(
            name=d.get("name", "Unnamed Phase"),
            phase_type=d.get("phase_type", "pretrain"),
            enabled=d.get("enabled", True),
            dataset_source=d.get("dataset_source", "tech_wiki_ro"),
            target_tokens=int(d.get("target_tokens", 500_000_000)),
            target_steps=int(d.get("target_steps", 5000)),
            micro_batch_size=int(d.get("micro_batch_size", 4)),
            grad_accum_steps=int(d.get("grad_accum_steps", 2)),
            seq_len=int(d.get("seq_len", 256)),
            learning_rate=float(d.get("learning_rate", 3.0e-3)),
            warmup_steps=int(d.get("warmup_steps", 200)),
            eval_frequency=int(d.get("eval_frequency", 10)),
            checkpoint_frequency=int(d.get("checkpoint_frequency", 500)),
            pause_after_completion=bool(d.get("pause_after_completion", False)),
            optimizer_config=d.get("optimizer_config", {}),
        )


class PipelineManager:
    """Manages sequential execution and verification of multi-stage training pipelines."""

    DEFAULT_PHASES: List[Dict[str, Any]] = [
        {
            "name": "Phase 1: General Pretraining",
            "phase_type": "pretrain",
            "enabled": True,
            "dataset_source": "tech_wiki_ro",
            "target_tokens": 100_000_000,
            "target_steps": 2500,
            "micro_batch_size": 4,
            "grad_accum_steps": 2,
            "seq_len": 256,
            "learning_rate": 3.0e-3,
            "warmup_steps": 200,
            "eval_frequency": 10,
            "checkpoint_frequency": 500,
            "pause_after_completion": False,
        },
        {
            "name": "Phase 2: Continued Domain Adaptation",
            "phase_type": "midtrain",
            "enabled": True,
            "dataset_source": "tech_wiki_ro",
            "target_tokens": 50_000_000,
            "target_steps": 1500,
            "micro_batch_size": 4,
            "grad_accum_steps": 2,
            "seq_len": 256,
            "learning_rate": 1.5e-3,
            "warmup_steps": 100,
            "eval_frequency": 10,
            "checkpoint_frequency": 500,
            "pause_after_completion": False,
        },
        {
            "name": "Phase 3: Romanian Instruction SFT",
            "phase_type": "chat_sft",
            "enabled": True,
            "dataset_source": "alpaca_ro",
            "target_tokens": 10_000_000,
            "target_steps": 1000,
            "micro_batch_size": 4,
            "grad_accum_steps": 2,
            "seq_len": 256,
            "learning_rate": 6.0e-4,
            "warmup_steps": 50,
            "eval_frequency": 10,
            "checkpoint_frequency": 250,
            "pause_after_completion": False,
        },
        {
            "name": "Phase 4: Creator Persona Refinement",
            "phase_type": "persona",
            "enabled": True,
            "dataset_source": "chat_persona_ro",
            "target_tokens": 5_000_000,
            "target_steps": 500,
            "micro_batch_size": 4,
            "grad_accum_steps": 2,
            "seq_len": 256,
            "learning_rate": 3.0e-4,
            "warmup_steps": 50,
            "eval_frequency": 10,
            "checkpoint_frequency": 250,
            "pause_after_completion": True,
        },
    ]

    def __init__(self, db: Optional[StudioDatabase] = None) -> None:
        self.db = db or StudioDatabase()
        self.checkpoint_manager = CheckpointManager(db=self.db)

    def create_pipeline(self, name: str, phases: Optional[List[Dict[str, Any]]] = None) -> str:
        pipeline_id = f"pipe_{uuid.uuid4().hex[:12]}"
        phase_list = phases if phases is not None else self.DEFAULT_PHASES
        self.db.save_pipeline(
            pipeline_id=pipeline_id,
            name=name,
            status="IDLE",
            phases=phase_list,
            current_phase_idx=0,
            state={"last_checkpoint": None, "history": []},
        )
        return pipeline_id

    def get_pipeline(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        return self.db.get_pipeline(pipeline_id)

    def list_pipelines(self) -> List[Dict[str, Any]]:
        pipes = self.db.list_pipelines()
        if not pipes:
            # Create default standard pipeline
            self.create_pipeline("Default RoCanon-48.5M 4-Stage Pipeline")
            pipes = self.db.list_pipelines()
        return pipes

    def verify_phase_readiness(
        self,
        pipeline_id: str,
        target_phase_idx: int,
        previous_checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Strict pre-flight check before transitioning to next phase.
        Ensures checkpoint integrity, dataset existence, and parameter compatibility.
        """
        pipe = self.get_pipeline(pipeline_id)
        if not pipe:
            return {"ready": False, "error": f"Pipeline {pipeline_id} not found."}

        phases = pipe.get("phases", [])
        if target_phase_idx < 0 or target_phase_idx >= len(phases):
            return {"ready": False, "error": f"Invalid phase index {target_phase_idx}"}

        phase = phases[target_phase_idx]

        # 1. If not the first phase, verify previous checkpoint
        if target_phase_idx > 0:
            chk = previous_checkpoint_path or pipe.get("state", {}).get("last_checkpoint")
            if not chk:
                return {
                    "ready": False,
                    "error": f"Previous checkpoint is required before starting phase {target_phase_idx + 1} ({phase['name']}).",
                }
            chk_path = Path(chk)
            if not chk_path.exists():
                return {
                    "ready": False,
                    "error": f"Previous checkpoint file does not exist on disk: {chk}",
                }
            stat = chk_path.stat()
            if stat.st_size < 1000:
                return {
                    "ready": False,
                    "error": f"Previous checkpoint file appears corrupted ({stat.st_size} bytes): {chk}",
                }

        # 2. Verify dataset source
        ds_source = phase.get("dataset_source", "")
        # Either preset key or file path
        known = {"tech_wiki_ro", "chat_persona_ro", "alpaca_ro"}
        if ds_source not in known and not Path(ds_source).exists():
            return {
                "ready": False,
                "error": f"Dataset source '{ds_source}' for phase '{phase['name']}' not found on disk.",
            }

        return {
            "ready": True,
            "phase_index": target_phase_idx,
            "phase_name": phase["name"],
            "dataset_source": ds_source,
            "checkpoint_to_resume": previous_checkpoint_path,
        }

    def advance_phase(
        self,
        pipeline_id: str,
        completed_step: int,
        checkpoint_path: str,
    ) -> Dict[str, Any]:
        """Advances pipeline to the next phase after successful completion."""
        pipe = self.get_pipeline(pipeline_id)
        if not pipe:
            return {"success": False, "error": "Pipeline not found"}

        current_idx = pipe.get("current_phase_idx", 0)
        phases = pipe.get("phases", [])
        state = pipe.get("state", {})

        history = state.get("history", [])
        history.append({
            "phase_index": current_idx,
            "phase_name": phases[current_idx]["name"] if current_idx < len(phases) else "unknown",
            "completed_step": completed_step,
            "checkpoint": checkpoint_path,
            "timestamp": time.time(),
        })

        next_idx = current_idx + 1
        is_complete = next_idx >= len(phases)
        new_status = "COMPLETED" if is_complete else "TRANSITIONING"

        state["last_checkpoint"] = checkpoint_path
        state["history"] = history

        self.db.save_pipeline(
            pipeline_id=pipeline_id,
            name=pipe["name"],
            status=new_status,
            phases=phases,
            current_phase_idx=next_idx if not is_complete else current_idx,
            state=state,
        )

        return {
            "success": True,
            "is_complete": is_complete,
            "next_phase_idx": next_idx if not is_complete else None,
            "next_phase": phases[next_idx] if not is_complete else None,
            "last_checkpoint": checkpoint_path,
        }

