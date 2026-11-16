"""
src/pipeline/runner.py
Automated Multi-Stage Pipeline Runner for RoCanon-48.5M.
Executes sequential training phases (Pretrain -> Midtrain -> Chat SFT -> Persona Refinement),
verifies phase boundaries, restores checkpoints safely, and manages phase transitions.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..worker.worker_manager import WorkerManager
from ..worker.state_machine import RunState
from .manager import PipelineManager

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrates automatic multi-phase pipeline execution through WorkerManager."""

    def __init__(
        self,
        pipeline_mgr: Optional[PipelineManager] = None,
        worker_mgr: Optional[WorkerManager] = None,
    ) -> None:
        self.pipeline_mgr = pipeline_mgr or PipelineManager()
        self.worker_mgr = worker_mgr
        self._lock = threading.Lock()
        self._active_pipeline_id: Optional[str] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_requested = False

    def start_pipeline_execution(
        self,
        pipeline_id: str,
        start_from_phase: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Initiates execution of the specified pipeline starting from the current or selected phase."""
        with self._lock:
            pipe = self.pipeline_mgr.get_pipeline(pipeline_id)
            if not pipe:
                return {"success": False, "error": f"Pipeline {pipeline_id} not found."}

            if self.worker_mgr is None:
                from ..dashboard.server import WORKER_MGR
                self.worker_mgr = WORKER_MGR

            current_worker_state = self.worker_mgr.state_machine.current_state
            if current_worker_state in (RunState.RUNNING, RunState.STARTING):
                return {
                    "success": False,
                    "error": f"Cannot start pipeline: training worker is currently {current_worker_state.name}.",
                }

            phase_idx = start_from_phase if start_from_phase is not None else pipe.get("current_phase_idx", 0)
            phases = pipe.get("phases", [])
            if phase_idx >= len(phases):
                return {"success": False, "error": f"Pipeline already completed all {len(phases)} phases."}

            # Check phase readiness
            readiness = self.pipeline_mgr.verify_phase_readiness(pipeline_id, phase_idx)
            if not readiness.get("ready"):
                return {"success": False, "error": readiness.get("error", "Phase not ready.")}

            # Prepare training configuration from phase config
            phase_spec = phases[phase_idx]
            run_config = self._build_run_config_for_phase(pipe, phase_spec, readiness.get("checkpoint_to_resume"))

            # Update pipeline status to RUNNING
            self.pipeline_mgr.db.save_pipeline(
                pipeline_id=pipeline_id,
                name=pipe["name"],
                status="RUNNING",
                phases=phases,
                current_phase_idx=phase_idx,
                state=pipe.get("state", {}),
            )

            self._active_pipeline_id = pipeline_id
            self._stop_requested = False

            # Launch worker
            run_id = self.worker_mgr.start_run(run_config)

            # Start background monitor thread to advance when completed
            if self._monitor_thread is None or not self._monitor_thread.is_alive():
                self._monitor_thread = threading.Thread(
                    target=self._monitor_loop,
                    name="PipelineMonitorThread",
                    daemon=True,
                )
                self._monitor_thread.start()

            return {
                "success": True,
                "pipeline_id": pipeline_id,
                "phase_index": phase_idx,
                "phase_name": phase_spec.get("name"),
                "run_id": run_id,
            }

    def _build_run_config_for_phase(
        self,
        pipe: Dict[str, Any],
        phase_spec: Dict[str, Any],
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Constructs a training configuration dictionary tailored for this phase."""
        chk = checkpoint_path or pipe.get("state", {}).get("last_checkpoint")
        return {
            "name": f"{pipe.get('name', 'Pipeline')} - {phase_spec.get('name')}",
            "phase": phase_spec.get("phase_type", "pretrain"),
            "dataset_source": phase_spec.get("dataset_source", "tech_wiki_ro"),
            "target_steps": int(phase_spec.get("target_steps", 1000)),
            "target_tokens": int(phase_spec.get("target_tokens", 100_000_000)),
            "checkpoint_path": chk,
            "training": {
                "micro_batch_size": int(phase_spec.get("micro_batch_size", 4)),
                "grad_accum_steps": int(phase_spec.get("grad_accum_steps", 2)),
                "seq_len": int(phase_spec.get("seq_len", 256)),
            },
            "optimizer": {
                "muon_lr": float(phase_spec.get("learning_rate", 3e-3)),
                "adamw_lr": float(phase_spec.get("learning_rate", 3e-3)) * 0.2,
                "warmup_steps": int(phase_spec.get("warmup_steps", 100)),
            },
            "eval_frequency": int(phase_spec.get("eval_frequency", 10)),
            "save_frequency": int(phase_spec.get("checkpoint_frequency", 250)),
        }

    def _monitor_loop(self) -> None:
        """Polls active worker and advances pipeline phases upon run completion."""
        logger.info("Pipeline monitor loop started.")
        while not self._stop_requested:
            time.sleep(2.0)
            if not self._active_pipeline_id or self.worker_mgr is None:
                continue

            state = self.worker_mgr.state_machine.current_state

            if state == RunState.COMPLETED:
                with self._lock:
                    pid = self._active_pipeline_id
                    pipe = self.pipeline_mgr.get_pipeline(pid)
                    if not pipe:
                        self._active_pipeline_id = None
                        continue

                    current_idx = pipe.get("current_phase_idx", 0)
                    phases = pipe.get("phases", [])
                    current_phase = phases[current_idx] if current_idx < len(phases) else {}

                    # Locate newest checkpoint from completed run
                    chk_list = self.pipeline_mgr.checkpoint_manager.list_checkpoints()
                    latest_chk = chk_list[0]["path"] if chk_list else None

                    logger.info("Pipeline %s phase %d finished. Advancing phase with checkpoint %s.", pid, current_idx, latest_chk)
                    adv = self.pipeline_mgr.advance_phase(
                        pipeline_id=pid,
                        completed_step=self.worker_mgr.latest_metric.get("step", 0) if self.worker_mgr.latest_metric else 0,
                        checkpoint_path=str(latest_chk) if latest_chk else "",
                    )

                    if adv.get("is_complete"):
                        logger.info("Pipeline %s has completed all phases!", pid)
                        self._active_pipeline_id = None
                        break

                    if current_phase.get("pause_after_completion", False):
                        logger.info("Pipeline %s paused as configured after phase %d.", pid, current_idx)
                        self._active_pipeline_id = None
                        break

                    # Auto-advance to next phase
                    next_idx = adv.get("next_phase_idx")
                    logger.info("Automatically launching next phase %d for pipeline %s", next_idx, pid)
                    self.start_pipeline_execution(pid, start_from_phase=next_idx)

            elif state in (RunState.FAILED, RunState.STOPPED):
                with self._lock:
                    if self._active_pipeline_id:
                        pipe = self.pipeline_mgr.get_pipeline(self._active_pipeline_id)
                        if pipe:
                            self.pipeline_mgr.db.save_pipeline(
                                pipeline_id=self._active_pipeline_id,
                                name=pipe["name"],
                                status="PAUSED" if state == RunState.STOPPED else "FAILED",
                                phases=pipe.get("phases", []),
                                current_phase_idx=pipe.get("current_phase_idx", 0),
                                state=pipe.get("state", {}),
                            )
                        self._active_pipeline_id = None
                        break

    def stop_monitoring(self) -> None:
        self._stop_requested = True
