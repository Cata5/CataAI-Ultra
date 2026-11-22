"""
src/worker/training_worker.py
Decoupled training worker process running RoCanon-48.5M training loop.
Communicates via multiprocessing Queues using WorkerCommand and WorkerEvent.
Guarantees GPU memory safety, loss stability monitoring, and atomic checkpointing.
"""

from __future__ import annotations

import logging
import math
import os
import queue
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..checkpoints.manager import CheckpointManager
from ..dashboard.early_stopping import EarlyStoppingConfig, OverfittingDetector
from ..data.custom_loader import CustomDatasetEngine
from ..db.database import StudioDatabase
from ..model.architecture import RoCanonConfig, RoCanonForCausalLM
from ..telemetry.hardware import HardwareMonitor
from ..telemetry.vram_guard import VRAMAlertLevel, VRAMGuard, VRAMThresholdConfig
from ..tokenizer.tokenizer import RoCanonTokenizer
from ..training.factory import create_model_from_dict, create_optimizers_from_dict
from ..training.loss import ChunkedCrossEntropyLoss
from ..training.scheduler import WSDScheduler
from ..training.task_adapter import BaseTaskAdapter, TaskAdapterFactory
from ..training.trainer import Trainer, TrainingConfig
from .protocol import CommandType, EventType, RunState, WorkerCommand, WorkerEvent
from .state_machine import RunStateMachine

logger = logging.getLogger("RoCanonWorker")


class TrainingWorkerProcess:
    """Manages the full lifecycle of training inside an isolated process."""

    def __init__(
        self,
        cmd_queue: Any,
        event_queue: Any,
        device_str: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> None:
        self.cmd_queue = cmd_queue
        self.event_queue = event_queue
        self.device_str = device_str
        self.db = StudioDatabase(db_path) if db_path else StudioDatabase()
        self.checkpoint_manager = CheckpointManager(db=self.db)
        self.hardware_monitor = HardwareMonitor()
        self.vram_guard = VRAMGuard()

        self.state_machine = RunStateMachine(
            initial_state=RunState.IDLE,
            on_transition=self._handle_state_transition,
        )

        self.current_run_id: Optional[str] = None
        self.current_phase: str = "pretrain"
        self.lr_multiplier: float = 1.0
        self.dynamic_muon_lr: Optional[float] = None
        self.dynamic_adamw_lr: Optional[float] = None
        self.auto_lr_adjust: bool = True
        self.adaptive_lr_factor: float = 1.0
        self.stop_requested: bool = False
        self.manual_val_requested: bool = False
        self.manual_chk_requested: bool = False

        # Moving average of loss for spike detection
        self.loss_ema: Optional[float] = None
        self.ema_alpha: float = 0.05
        self._last_vram_alert_time: float = 0.0
        self._last_emergency_save_time: float = 0.0

    def _handle_state_transition(self, old_state: RunState, new_state: RunState) -> None:
        event = WorkerEvent(
            event_type=EventType.STATE_CHANGED,
            run_id=self.current_run_id,
            payload={"old_state": old_state.value, "new_state": new_state.value},
        )
        self._send_event(event)
        if self.current_run_id:
            self.db.update_run_status(self.current_run_id, new_state.value)

    def _send_event(self, event: WorkerEvent) -> None:
        try:
            self.event_queue.put_nowait(event.to_dict())
        except Exception as e:
            logger.error("Failed to enqueue worker event: %s", e)

    def _poll_commands(self) -> None:
        while True:
            try:
                cmd_dict = self.cmd_queue.get_nowait()
                cmd = WorkerCommand.from_dict(cmd_dict)
                self._handle_command(cmd)
            except queue.Empty:
                break
            except Exception as e:
                logger.error("Error reading command: %s", e)
                break

    def _handle_command(self, cmd: WorkerCommand) -> None:
        logger.info("Worker received command: %s", cmd.command_type.value)
        ct = cmd.command_type

        if ct == CommandType.PAUSE_RUN:
            if self.state_machine.can_transition_to(RunState.PAUSING):
                self.state_machine.transition_to(RunState.PAUSING)
        elif ct == CommandType.RESUME_RUN:
            if self.state_machine.can_transition_to(RunState.RUNNING):
                self.state_machine.transition_to(RunState.RUNNING)
        elif ct == CommandType.STOP_RUN:
            self.stop_requested = True
            if self.state_machine.can_transition_to(RunState.STOPPING):
                self.state_machine.transition_to(RunState.STOPPING)
        elif ct == CommandType.SAVE_CHECKPOINT:
            self.manual_chk_requested = True
        elif ct == CommandType.RUN_VALIDATION:
            self.manual_val_requested = True
        elif ct == CommandType.SET_LR_MULTIPLIER:
            if "multiplier" in cmd.payload and cmd.payload["multiplier"] is not None:
                self.lr_multiplier = float(cmd.payload["multiplier"])
            if "muon_lr" in cmd.payload and cmd.payload["muon_lr"] is not None:
                self.dynamic_muon_lr = float(cmd.payload["muon_lr"])
            if "adamw_lr" in cmd.payload and cmd.payload["adamw_lr"] is not None:
                self.dynamic_adamw_lr = float(cmd.payload["adamw_lr"])
            if "auto_adjust" in cmd.payload:
                self.auto_lr_adjust = bool(cmd.payload["auto_adjust"])
            logger.info(
                "Updated dynamic LR settings: multiplier=%.3f, muon=%s, adamw=%s, auto_adjust=%s",
                self.lr_multiplier,
                self.dynamic_muon_lr,
                self.dynamic_adamw_lr,
                self.auto_lr_adjust,
            )

    def run_loop(self) -> None:
        """Main process loop listening for commands and executing training."""
        logger.info("Training worker process ready.")

        while True:
            try:
                # Check commands
                try:
                    cmd_dict = self.cmd_queue.get(timeout=0.1)
                    cmd = WorkerCommand.from_dict(cmd_dict)
                    if cmd.command_type == CommandType.START_RUN:
                        self._execute_training_session(cmd.payload)
                    else:
                        self._handle_command(cmd)
                except queue.Empty:
                    continue
            except KeyboardInterrupt:
                logger.info("Worker interrupted by user.")
                break
            except Exception as e:
                logger.exception("Unexpected error in worker loop: %s", e)
                time.sleep(0.5)

    def _execute_training_session(self, payload: Dict[str, Any]) -> None:
        run_id = payload.get("run_id", f"run_{int(time.time())}")
        self.current_run_id = run_id
        self.stop_requested = False
        self.manual_val_requested = False
        self.manual_chk_requested = False
        self.loss_ema = None

        task_name = (
            payload.get("task")
            or payload.get("task_type")
            or config_arch.get("default_task")
            or "causal_lm"
        )
        phase = payload.get("phase")
        if not phase:
            if task_name in ("instruction_sft", "chat_sft", "chat", "sft"):
                phase = "chat_sft"
            elif task_name in ("classification", "regression", "masked_lm"):
                phase = task_name
            else:
                phase = "pretrain"
        self.current_phase = phase
        dataset_source = payload.get("dataset_source", "tech_wiki_ro")
        config_arch = payload.get("config_arch", {})
        network_id = payload.get("network_id")
        if network_id:
            from ..networks.registry import NetworkRegistry
            net_reg = NetworkRegistry()
            net_cfg = net_reg.get_network(network_id)
            if net_cfg:
                config_arch = net_cfg
        config_opt = payload.get("config_opt", {})
        config_es = payload.get("config_early_stopping", {})
        max_samples = payload.get("max_samples")
        checkpoint_to_load = (
            payload.get("resume_checkpoint")
            or payload.get("checkpoint_to_load")
            or payload.get("checkpoint_path")
            or payload.get("checkpoint")
        )
        duration_mode = str(payload.get("duration_mode", "steps")).lower()
        target_epochs = int(payload.get("target_epochs") or payload.get("epochs") or 3)
        target_steps = int(payload.get("total_steps") or payload.get("target_steps") or 1000)
        target_tokens = int(payload.get("target_tokens", 10_000_000_000_000))
        eval_freq = int(
            payload.get("validation_interval")
            or payload.get("val_interval")
            or payload.get("eval_frequency")
            or payload.get("eval_freq")
            or config_es.get("eval_frequency")
            or 250
        )
        save_freq = int(
            payload.get("checkpoint_interval")
            or payload.get("save_frequency")
            or payload.get("chk_interval")
            or 50
        )
        es_enabled = bool(payload.get("early_stopping_enabled", False))
        es_patience = int(payload.get("early_stopping_patience", 20))
        es_min_delta = float(payload.get("early_stopping_min_delta", 0.001))

        base_muon = float(payload.get("muon_lr") or config_opt.get("muon_lr", 3e-3))
        base_adamw = float(payload.get("adamw_lr") or payload.get("peak_lr") or config_opt.get("adamw_lr", 6e-4))
        config_opt["muon_lr"] = base_muon
        config_opt["adamw_lr"] = base_adamw
        if "micro_batch_size" in payload:
            config_opt["micro_batch_size"] = int(payload["micro_batch_size"])
        if "grad_accum_steps" in payload:
            config_opt["grad_accum_steps"] = int(payload["grad_accum_steps"])

        self.latest_val_loss = None
        self.latest_val_step = None
        self.latest_val_timestamp = None
        self.cumulative_active_time = 0.0

        self.workspace_id = str(payload.get("workspace_id") or "default")
        self.state_machine.transition_to(RunState.STARTING)
        self.db.create_run(
            run_id=run_id,
            name=payload.get("run_name", f"RoCanon Run {run_id}"),
            current_phase=phase,
            dataset_source=str(dataset_source),
            config={"arch": config_arch, "opt": config_opt, "es": config_es},
            workspace_id=self.workspace_id,
        )

        device = torch.device(
            self.device_str if self.device_str else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        logger.info("Worker bound to compute device: %s", device)

        # 1. Initialize Model
        try:
            model = create_model_from_dict(config_arch).to(device=device)

            # Check for PEFT / LoRA
            use_lora = bool(payload.get("use_lora", False))
            lora_r = int(payload.get("lora_r", 16))
            lora_alpha = int(payload.get("lora_alpha", 32))
            lora_dropout = float(payload.get("lora_dropout", 0.05))
            if use_lora:
                try:
                    from ..training.peft import inject_lora
                    lora_res = inject_lora(model, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                    logger.info("PEFT LoRA active: %s", lora_res)
                except Exception as lora_err:
                    logger.warning("LoRA injection failed: %s", lora_err)

            # Check for Multi-GPU Distributed Training
            use_multi_gpu = bool(payload.get("use_multi_gpu", False))
            device_ids = payload.get("device_ids")
            if use_multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
                try:
                    from ..training.distributed import wrap_model_for_multi_gpu
                    model, bound_devices = wrap_model_for_multi_gpu(model, device_ids=device_ids)
                    logger.info("Multi-GPU training wrapped across %s", bound_devices)
                except Exception as mgpu_err:
                    logger.warning("Multi-GPU wrapping failed: %s", mgpu_err)
        except Exception as e:
            logger.exception("Failed to initialize model: %s", e)
            self._send_event(WorkerEvent(
                event_type=EventType.RUN_FAILED,
                run_id=run_id,
                payload={"error": f"Model initialization failed: {e}"},
            ))
            self.state_machine.transition_to(RunState.FAILED)
            return

        # 2. Configure Optimizers & Scheduler
        try:
            muon_opt, adamw_opt = create_optimizers_from_dict(model, config_opt)
            scheduler = WSDScheduler(
                optimizers=(muon_opt, adamw_opt),
                warmup_steps=int(config_opt.get("warmup_steps", 2000)),
                plateau_tokens=int(config_opt.get("plateau_tokens", 5_000_000_000)),
                total_tokens=int(config_opt.get("total_tokens", 6_800_000_000)),
                min_lr_ratio=float(config_opt.get("min_lr_ratio", 0.1)),
                min_lr_floor=float(config_opt.get("min_lr_floor", config_opt.get("min_lr", 0.0))),
                decay_style=str(config_opt.get("decay_style", "cosine")),
            )

            training_cfg = TrainingConfig(
                device=str(device),
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                max_grad_norm=float(config_opt.get("max_grad_norm", 1.0)),
                grad_accum_steps=int(config_opt.get("grad_accum_steps", 2)),
                micro_batch_size=int(config_opt.get("micro_batch_size", 4)),
                seq_len=int(config_opt.get("seq_len", 256)),
                max_budget_gb=float(config_opt.get("max_budget_gb", 5.0)),
            )

            # Resolve Task Adapter (Causal LM, SFT, Classification, Masked LM, Regression)
            task_name = (
                payload.get("task")
                or payload.get("task_type")
                or config_arch.get("default_task")
                or ("instruction_sft" if phase == "chat_sft" else "causal_lm")
            )
            task_adapter = TaskAdapterFactory.get_adapter(task_name)
            logger.info("Configured task adapter: %s (%s)", task_adapter.task_name, task_adapter.description)

            trainer = Trainer(
                model=model,
                optimizers=(muon_opt, adamw_opt),
                scheduler=scheduler,
                config=training_cfg,
                task_adapter=task_adapter,
            )
        except Exception as e:
            logger.exception("Failed to configure trainer/optimizer: %s", e)
            self._send_event(WorkerEvent(
                event_type=EventType.RUN_FAILED,
                run_id=run_id,
                payload={"error": f"Optimizer setup failed: {e}"},
            ))
            self.state_machine.transition_to(RunState.FAILED)
            return

        # 3. Load Checkpoint (Full State Restoration: Weights + Optimizers + Scheduler + RNG)
        start_step = 0
        total_tokens = 0
        start_tokens = 0
        if checkpoint_to_load:
            try:
                resolved_chk_p = self.checkpoint_manager.resolve_checkpoint_path(checkpoint_to_load)
                checkpoint_to_load = str(resolved_chk_p)

                load_res = self.checkpoint_manager.load_checkpoint(
                    checkpoint_path=resolved_chk_p,
                    model=model,
                    optimizers=(muon_opt, adamw_opt),
                    scheduler=scheduler,
                    device=device,
                    restore_training_state=True,
                )
                start_step = int(load_res.get("step", 0))
                total_tokens = int(load_res.get("tokens_seen", 0))
                start_tokens = total_tokens
                trainer.global_step = start_step
                trainer.cumulative_tokens = total_tokens
                chk_val_loss = load_res.get("val_loss")
                chk_train_loss = load_res.get("train_loss")
                if chk_val_loss is not None:
                    try:
                        self.latest_val_loss = float(chk_val_loss)
                        self.latest_val_step = start_step
                        self.latest_val_timestamp = time.time()
                    except (ValueError, TypeError):
                        pass
                self.cumulative_active_time = float(load_res.get("active_training_time_s") or load_res.get("elapsed_time") or 0.0)
                logger.info(
                    "Fully restored training state from %s (starting step: %d, tokens: %d, val_loss: %s, Muon entries: %d)",
                    checkpoint_to_load,
                    start_step,
                    total_tokens,
                    chk_val_loss,
                    len(muon_opt.state),
                )
                self._send_event(WorkerEvent(
                    event_type=EventType.STATE_CHANGED,
                    run_id=run_id,
                    step=start_step,
                    payload={
                        "message": f"Checkpoint loaded: step {start_step}, tokens {total_tokens}, val_loss {chk_val_loss}",
                        "checkpoint": checkpoint_to_load,
                        "step": start_step,
                        "val_loss": chk_val_loss,
                        "loss": chk_train_loss,
                        "train_loss": chk_train_loss,
                    },
                ))
            except Exception as e:
                logger.warning("Could not fully load checkpoint %s: %s", checkpoint_to_load, e)
                self._send_event(WorkerEvent(
                    event_type=EventType.GRADIENT_WARNING,
                    run_id=run_id,
                    payload={"alert": f"Warning: Failed to restore checkpoint {checkpoint_to_load}: {e}"},
                ))

        # Checkpoint resume & target steps resolution:
        # If user loads a checkpoint at step 10,000 and entered 1,000 steps,
        # target_steps (1000) <= start_step (10000).
        # Set target_steps = start_step + session_steps = 11,000 so training does not stop immediately!
        if start_step > 0 and target_steps <= start_step:
            session_steps = target_steps
            target_steps = start_step + session_steps
            logger.info(
                "Checkpoint resume: target_steps (%d) <= start_step (%d). Adjusted target_steps to %d (+%d steps in this session).",
                session_steps, start_step, target_steps, session_steps,
            )
        else:
            session_steps = max(1, target_steps - start_step)

        if total_tokens > 0 and target_tokens <= total_tokens:
            target_tokens = total_tokens + 10_000_000_000_000

        # 4. Configure Early Stopping
        es_cfg = EarlyStoppingConfig(
            enabled=es_enabled,
            patience=es_patience,
            min_delta=es_min_delta,
            min_steps=int(payload.get("early_stopping_min_steps", config_es.get("min_steps", 500))),
            overfitting_ratio=float(config_es.get("overfitting_ratio", 1.15)),
            checkpoint_dir="checkpoints",
        )
        detector = OverfittingDetector(es_cfg)

        # 5. Load Dataset
        tokenizer = RoCanonTokenizer()
        dataset_engine = CustomDatasetEngine(tokenizer=tokenizer, db=self.db)

        def on_ds_progress(txt: str, pct: float):
            self._send_event(WorkerEvent(
                event_type=EventType.METRICS_UPDATED,
                run_id=run_id,
                payload={
                    "dataset_status": txt,
                    "dataset_progress_pct": round(pct, 1),
                    "dataset_name": dataset_source,
                },
            ))

        try:
            train_inputs, train_labels, val_inputs, val_labels = dataset_engine.load_dataset(
                dataset_source=dataset_source,
                phase=phase,
                max_samples=max_samples,
                device=device,
                progress_callback=on_ds_progress,
            )
        except Exception as e:
            logger.exception("Dataset load failed: %s", e)
            self._send_event(WorkerEvent(
                event_type=EventType.RUN_FAILED,
                run_id=run_id,
                payload={"error": f"Dataset loading failed: {e}"},
            ))
            self.state_machine.transition_to(RunState.FAILED)
            return

        is_seq_tokens = (train_inputs.dim() == 1)
        if is_seq_tokens:
            batch_len = training_cfg.micro_batch_size * training_cfg.seq_len
            train_len = len(train_inputs)
            val_len = len(val_inputs)
        else:
            batch_len = training_cfg.micro_batch_size
            train_len = train_inputs.size(0)
            val_len = val_inputs.size(0)

        val_batches = max(1, min(12, val_len // max(1, batch_len)))

        # Resolve target tokens and session steps per dataset/session
        tokens_per_step = training_cfg.micro_batch_size * max(1, training_cfg.seq_len - 1) * max(1, training_cfg.grad_accum_steps)
        if duration_mode == "epochs":
            session_target_tokens = int(target_epochs * train_len)
            session_steps = max(1, math.ceil(session_target_tokens / max(1, tokens_per_step)))
            target_steps = start_step + session_steps
            target_tokens = start_tokens + session_target_tokens
            logger.info(
                "Epoch mode: target %d epochs on dataset (%d tokens/epoch). Session targets: %d steps, %d tokens. Global target step: %d.",
                target_epochs, train_len, session_steps, session_target_tokens, target_steps,
            )
        else:
            session_target_tokens = max(1, session_steps * tokens_per_step)
            target_tokens = start_tokens + session_target_tokens

        self.state_machine.transition_to(RunState.RUNNING)
        self._send_event(WorkerEvent(
            event_type=EventType.RUN_STARTED,
            run_id=run_id,
            payload={
                "train_tokens": train_len,
                "val_tokens": val_len,
                "dataset_tokens": train_len,
                "dataset_name": dataset_source,
                "dataset_status": f"Ready ({train_len:,} train tokens)",
                "dataset_progress_pct": 100.0,
                "target_steps": target_steps,
                "target_tokens": target_tokens,
                "phase": phase,
                "start_step": start_step,
            },
        ))

        step = start_step
        base_muon = float(payload.get("muon_lr") or config_opt.get("muon_lr", 3e-3))
        base_adamw = float(payload.get("adamw_lr") or payload.get("peak_lr") or config_opt.get("adamw_lr", 6e-4))
        eval_freq = int(
            payload.get("validation_interval")
            or payload.get("val_interval")
            or payload.get("eval_frequency")
            or payload.get("eval_freq")
            or config_es.get("eval_frequency")
            or 250
        )
        save_freq = int(
            payload.get("checkpoint_interval")
            or payload.get("save_frequency")
            or payload.get("chk_interval")
            or 50
        )
        grad_norm: Optional[float] = None

        # Metrics write buffer for batch sqlite commits
        metrics_buffer: List[Dict[str, Any]] = []

        logger.info("Training loop started on %s (%d train tokens)", device, train_len)

        try:
            while self.state_machine.current_state in {RunState.RUNNING, RunState.PAUSING, RunState.PAUSED}:
                self._poll_commands()

                # Handle PAUSE
                if self.state_machine.current_state == RunState.PAUSING:
                    self.state_machine.transition_to(RunState.PAUSED)
                    self._send_event(WorkerEvent(
                        event_type=EventType.RUN_PAUSED,
                        run_id=run_id,
                        step=step,
                    ))

                while self.state_machine.current_state == RunState.PAUSED and not self.stop_requested:
                    self._poll_commands()
                    time.sleep(0.1)

                if self.stop_requested or self.state_machine.current_state == RunState.STOPPING:
                    logger.info("Stopping requested at step %d", step)
                    break

                step += 1
                t0 = time.time()

                # Dynamic Learning Rate Scheduling & Adaptation
                session_total_steps = max(1, target_steps - start_step)
                session_step = max(0, step - start_step)

                # 1. Warmup & Cosine Annealing schedule relative to session steps
                warmup_steps = max(10, min(150, int(session_total_steps * 0.05)))
                if session_step < warmup_steps:
                    sched_factor = 0.1 + 0.9 * (float(session_step) / float(max(1, warmup_steps)))
                    sched_phase = "warmup"
                elif session_step > int(session_total_steps * 0.85):
                    decay_span = max(1, session_total_steps - int(session_total_steps * 0.85))
                    decay_progress = (session_step - int(session_total_steps * 0.85)) / float(decay_span)
                    sched_factor = 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, decay_progress)))
                    sched_phase = "decay"
                else:
                    sched_factor = 1.0
                    sched_phase = "stable"

                # 2. Dynamic Auto-Adjustment based on Gradient Stability
                if self.auto_lr_adjust and step > (start_step + 3) and grad_norm is not None:
                    if grad_norm < 1.0:
                        # Gradients very clean: accelerate convergence
                        self.adaptive_lr_factor = min(1.35, self.adaptive_lr_factor * 1.01)
                    elif grad_norm > 4.0:
                        # High gradient pressure: throttle down sharply to protect stability
                        self.adaptive_lr_factor = max(0.65, self.adaptive_lr_factor * 0.85)
                    elif grad_norm > 2.0:
                        self.adaptive_lr_factor = max(0.85, self.adaptive_lr_factor * 0.96)
                    else:
                        self.adaptive_lr_factor = 0.98 * self.adaptive_lr_factor + 0.02 * 1.0
                else:
                    self.adaptive_lr_factor = 1.0

                # 3. Dynamic Base LR overrides (live adjustment from dashboard)
                active_base_muon = self.dynamic_muon_lr if self.dynamic_muon_lr is not None else base_muon
                active_base_adamw = self.dynamic_adamw_lr if self.dynamic_adamw_lr is not None else base_adamw

                curr_mult = sched_factor * self.adaptive_lr_factor * self.lr_multiplier
                eff_muon_lr = active_base_muon * curr_mult
                eff_adamw_lr = active_base_adamw * curr_mult

                for g in muon_opt.param_groups:
                    g["lr"] = eff_muon_lr
                for g in adamw_opt.param_groups:
                    g["lr"] = eff_adamw_lr

                # Manual Checkpoint requested
                if self.manual_chk_requested:
                    self.manual_chk_requested = False
                    chk_p = self.checkpoint_manager.save_checkpoint(
                        model=model,
                        step=step,
                        tokens_seen=total_tokens,
                        optimizers=(muon_opt, adamw_opt),
                        scheduler=scheduler,
                        phase=phase,
                        run_id=run_id,
                        filename_prefix="manual_snapshot",
                    )
                    self._send_event(WorkerEvent(
                        event_type=EventType.CHECKPOINT_SAVED,
                        run_id=run_id,
                        step=step,
                        payload={"path": str(chk_p), "reason": "manual"},
                    ))

                # Step execution with micro-batch accumulation
                try:
                    trainer.zero_grad()
                    session_curr_step = max(0, step - start_step - 1)
                    for accum_idx in range(training_cfg.grad_accum_steps):
                        start_pos = (session_curr_step * training_cfg.grad_accum_steps + accum_idx) * batch_len
                        if is_seq_tokens:
                            if start_pos + batch_len <= train_len:
                                batch_in = train_inputs[start_pos : start_pos + batch_len].view(
                                    training_cfg.micro_batch_size, training_cfg.seq_len
                                )
                                batch_lbl = (
                                    train_labels[start_pos : start_pos + batch_len].view(
                                        training_cfg.micro_batch_size, training_cfg.seq_len
                                    )
                                    if train_labels is not None
                                    else None
                                )
                            else:
                                idx = (torch.arange(batch_len, device=device) + start_pos) % train_len
                                batch_in = train_inputs[idx].view(training_cfg.micro_batch_size, training_cfg.seq_len)
                                batch_lbl = (
                                    train_labels[idx].view(training_cfg.micro_batch_size, training_cfg.seq_len)
                                    if train_labels is not None
                                    else None
                                )
                        else:
                            # Feature tensor [N, S, D] or [N, D]
                            idx = (torch.arange(batch_len, device=device) + start_pos) % train_len
                            batch_in = train_inputs[idx]
                            batch_lbl = train_labels[idx] if train_labels is not None else None

                        trainer.accumulate_micro_batch(input_ids=batch_in, labels=batch_lbl)

                        self._poll_commands()
                        if self.stop_requested or self.state_machine.current_state == RunState.STOPPING:
                            break

                    if self.stop_requested or self.state_machine.current_state == RunState.STOPPING:
                        logger.info("Stopping requested during micro-batch accumulation at step %d", step)
                        break

                    tokens_in_step = training_cfg.micro_batch_size * (training_cfg.seq_len - 1) * training_cfg.grad_accum_steps
                    total_tokens += tokens_in_step

                    grad_norm = trainer.optimizer_step(num_tokens_in_step=total_tokens, advance_scheduler=False)
                    scheduler.current_step = step
                    scheduler.current_tokens = total_tokens
                    train_loss = trainer.loss_accumulator.reset()
                    dt = time.time() - t0
                    self.cumulative_active_time += dt

                except torch.cuda.OutOfMemoryError as oom_err:
                    logger.critical("CUDA Out of Memory encountered at step %d: %s", step, oom_err)
                    torch.cuda.empty_cache()
                    self.db.log_alert(
                        level="CRITICAL",
                        category="CUDA_OOM",
                        message=f"CUDA Out of Memory caught at step {step}. Training paused safely for recovery.",
                        run_id=run_id,
                        step=step,
                    )
                    self._send_event(WorkerEvent(
                        event_type=EventType.GRADIENT_WARNING,
                        run_id=run_id,
                        step=step,
                        payload={"alert": "CUDA OOM: Cache cleared. Pausing safely."},
                    ))
                    if self.state_machine.can_transition_to(RunState.PAUSING):
                        self.state_machine.transition_to(RunState.PAUSING)
                    continue

                tok_per_sec = tokens_in_step / max(1e-5, dt)
                tflops = (tok_per_sec * 354.11e6) / 1e12

                # Stability checks: NaN / Inf / Spike detection
                if math.isnan(train_loss) or math.isinf(train_loss):
                    self.db.log_alert(
                        level="CRITICAL",
                        category="LOSS_NAN_INF",
                        message=f"Invalid loss value detected at step {step}: {train_loss}",
                        run_id=run_id,
                        step=step,
                    )
                    self._send_event(WorkerEvent(
                        event_type=EventType.LOSS_SPIKE,
                        run_id=run_id,
                        step=step,
                        payload={"loss": train_loss, "level": "CRITICAL"},
                    ))

                # Loss Spike relative to EMA
                if self.loss_ema is None:
                    self.loss_ema = train_loss
                else:
                    if train_loss > (self.loss_ema * 2.5) and step > 10:
                        self.db.log_alert(
                            level="WARNING",
                            category="LOSS_SPIKE",
                            message=f"Sudden loss spike: {train_loss:.4f} vs EMA {self.loss_ema:.4f}",
                            run_id=run_id,
                            step=step,
                        )
                        self._send_event(WorkerEvent(
                            event_type=EventType.LOSS_SPIKE,
                            run_id=run_id,
                            step=step,
                            payload={"loss": train_loss, "ema": self.loss_ema},
                        ))
                    self.loss_ema = (1.0 - self.ema_alpha) * self.loss_ema + self.ema_alpha * train_loss

                # Gradient Norm threshold alert
                if grad_norm > 15.0:
                    self.db.log_alert(
                        level="WARNING",
                        category="HIGH_GRAD_NORM",
                        message=f"High gradient norm: {grad_norm:.2f} at step {step}",
                        run_id=run_id,
                        step=step,
                    )

                # Validation evaluation
                val_loss = None
                gen_gap = None
                is_eval_step = (eval_freq > 0 and (step - start_step) > 0 and (step - start_step) % eval_freq == 0) or self.manual_val_requested
                if is_eval_step:
                    self.manual_val_requested = False
                    model.eval()
                    val_loss_acc = 0.0
                    with torch.no_grad():
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
                            for v_idx in range(val_batches):
                                v_start = (v_idx * batch_len) % val_len
                                v_idx_tensor = (torch.arange(batch_len, device=device) + v_start) % val_len
                                if is_seq_tokens:
                                    v_in_batch = val_inputs[v_idx_tensor].view(training_cfg.micro_batch_size, training_cfg.seq_len)
                                    v_tgt_batch = (
                                        val_labels[v_idx_tensor].view(training_cfg.micro_batch_size, training_cfg.seq_len)
                                        if val_labels is not None
                                        else v_in_batch
                                    )
                                else:
                                    v_in_batch = val_inputs[v_idx_tensor]
                                    v_tgt_batch = val_labels[v_idx_tensor] if val_labels is not None else None

                                v_loss_t, _, _ = task_adapter.forward_loss(
                                    model=model,
                                    inputs=v_in_batch,
                                    targets=v_tgt_batch,
                                    loss_fn=trainer.loss_fn,
                                    autocast_dtype=training_cfg.dtype,
                                )
                                val_loss_acc += float(v_loss_t.item())

                    val_loss = val_loss_acc / val_batches
                    self.latest_val_loss = val_loss
                    self.latest_val_step = step
                    self.latest_val_timestamp = time.time()
                    model.train()

                    val_record = detector.evaluate_step(
                        step=step,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        model=model,
                    )
                    gen_gap = val_record.generalization_gap

                    # Check if new best validation checkpoint
                    if detector.best_step == step:
                        self.checkpoint_manager.save_checkpoint(
                            model=model,
                            step=step,
                            tokens_seen=total_tokens,
                            optimizers=(muon_opt, adamw_opt),
                            scheduler=scheduler,
                            phase=phase,
                            val_loss=val_loss,
                            train_loss=train_loss,
                            run_id=run_id,
                            is_best=True,
                        )
                        try:
                            from ..notifications.webhook import get_notifier
                            get_notifier(db=self.db).notify_best_checkpoint(
                                step=step,
                                val_loss=float(val_loss),
                                checkpoint_path="best_model.pt",
                                run_id=run_id,
                            )
                        except Exception as n_err:
                            logger.debug("Webhook notification error: %s", n_err)
                    elif save_freq > 0 and step % save_freq == 0:
                        self.checkpoint_manager.save_checkpoint(
                            model=model,
                            step=step,
                            tokens_seen=total_tokens,
                            optimizers=(muon_opt, adamw_opt),
                            scheduler=scheduler,
                            phase=phase,
                            val_loss=val_loss,
                            train_loss=train_loss,
                            run_id=run_id,
                            filename_prefix="snapshot",
                        )

                    self._send_event(WorkerEvent(
                        event_type=EventType.VALIDATION_COMPLETED,
                        run_id=run_id,
                        step=step,
                        payload={
                            "val_loss": val_loss,
                            "gen_gap": gen_gap,
                            "best_val_loss": detector.best_val_loss,
                            "best_step": detector.best_step,
                        },
                    ))

                    if detector.stop_requested:
                        logger.warning("Early stopping triggered at step %d", step)
                        detector.restore_best(model)
                        self.checkpoint_manager.save_checkpoint(
                            model=model,
                            step=step,
                            tokens_seen=total_tokens,
                            optimizers=(muon_opt, adamw_opt),
                            scheduler=scheduler,
                            phase=phase,
                            val_loss=detector.best_val_loss,
                            train_loss=train_loss,
                            run_id=run_id,
                            filename_prefix="early_stopped_best",
                        )
                        logger.info(
                            "Early stopping triggered at step %d (patience=%d, best_val_loss=%.4f at step %d)",
                            step, es_patience, detector.best_val_loss, detector.best_step
                        )
                        self.stop_requested = True
                        break

                # Telemetry & Memory
                vram_gb = trainer.memory_tracker.get_peak_allocated_gb() if torch.cuda.is_available() else 0.0

                # Check Dynamic VRAM Protection
                if device.type == "cuda":
                    free_b, total_b = torch.cuda.mem_get_info(device)
                    pt_reserved = torch.cuda.memory_reserved(device)
                    pt_allocated = torch.cuda.memory_allocated(device)
                    pt_cache_free = max(0, pt_reserved - pt_allocated)
                    effective_free_mb = (free_b + pt_cache_free) / (1024 * 1024)

                    alert_lvl, alert_msg = self.vram_guard.evaluate(effective_free_mb, pytorch_allocated_mb=vram_gb * 1024)
                    now_t = time.time()
                    if alert_lvl in (VRAMAlertLevel.CRITICAL, VRAMAlertLevel.EMERGENCY):
                        if (now_t - self._last_vram_alert_time) > 30.0:
                            self._last_vram_alert_time = now_t
                            self.db.log_alert(
                                level=alert_lvl.value,
                                category="VRAM_PRESSURE",
                                message=alert_msg,
                                run_id=run_id,
                                step=step,
                            )
                        if alert_lvl == VRAMAlertLevel.EMERGENCY and (now_t - self._last_emergency_save_time) > 60.0:
                            self._last_emergency_save_time = now_t
                            logger.warning("VRAM Emergency! Saving single protective snapshot...")
                            self.checkpoint_manager.save_checkpoint(
                                model=model,
                                step=step,
                                tokens_seen=total_tokens,
                                optimizers=(muon_opt, adamw_opt),
                                scheduler=scheduler,
                                phase=phase,
                                val_loss=val_loss,
                                train_loss=train_loss,
                                run_id=run_id,
                                filename_prefix="emergency_vram",
                            )
                            if self.state_machine.can_transition_to(RunState.PAUSING):
                                self.state_machine.transition_to(RunState.PAUSING)

                session_tokens = max(0, total_tokens - start_tokens)
                epoch_val = round(session_tokens / train_len, 3) if train_len > 0 else 0.0
                tokens_in_curr_epoch = session_tokens % train_len if train_len > 0 else 0
                epoch_pct = round((tokens_in_curr_epoch / train_len) * 100, 2) if train_len > 0 else 0.0
                session_curr_steps = max(0, step - start_step)
                step_pct = round((session_curr_steps / session_total_steps) * 100, 2) if session_total_steps > 0 else 0.0
                token_pct = round((session_tokens / session_target_tokens) * 100, 2) if session_target_tokens > 0 else 0.0
                rem_tokens = max(0, session_target_tokens - session_tokens)
                eta_s = int(rem_tokens / max(tok_per_sec, 1)) if tok_per_sec > 0 else 0

                metric_dict = {
                    "step": step,
                    "start_step": start_step,
                    "session_step": session_curr_steps,
                    "session_target_steps": session_total_steps,
                    "target_steps": target_steps,
                    "step_pct": step_pct,
                    "tokens": total_tokens,
                    "target_tokens": target_tokens,
                    "token_pct": token_pct,
                    "dataset_tokens": train_len,
                    "val_tokens": val_len,
                    "epoch": epoch_val,
                    "epoch_pct": epoch_pct,
                    "tokens_in_epoch": tokens_in_curr_epoch,
                    "eta_seconds": eta_s,
                    "dataset_name": dataset_source,
                    "dataset_status": "CUDA Streaming & Training",
                    "dataset_progress_pct": 100.0,
                    "loss": round(train_loss, 4),
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4) if val_loss is not None else None,
                    "latest_val_loss": round(self.latest_val_loss, 4) if self.latest_val_loss is not None else None,
                    "latest_val_step": self.latest_val_step,
                    "latest_val_timestamp": self.latest_val_timestamp,
                    "val_status": "completed" if self.latest_val_loss is not None else "awaiting",
                    "eval_interval": eval_freq,
                    "last_val_step": self.latest_val_step,
                    "next_val_step": (start_step + (((step - start_step) // eval_freq) + 1) * eval_freq) if eval_freq > 0 else None,
                    "generalization_gap": round(gen_gap, 4) if gen_gap is not None else None,
                    "grad_norm": round(grad_norm, 4) if grad_norm is not None else 0.0,
                    "learning_rate": eff_muon_lr,
                    "lr": eff_muon_lr,
                    "muon_lr": eff_muon_lr,
                    "adamw_lr": eff_adamw_lr,
                    "base_muon_lr": active_base_muon,
                    "base_adamw_lr": active_base_adamw,
                    "sched_phase": sched_phase,
                    "sched_factor": round(sched_factor, 4),
                    "lr_multiplier": round(curr_mult, 4),
                    "adaptive_lr_factor": round(self.adaptive_lr_factor, 4),
                    "step_time_ms": round(dt * 1000, 2),
                    "active_training_time_s": round(self.cumulative_active_time, 2),
                    "duration_mode": duration_mode,
                    "target_epochs": target_epochs if duration_mode == "epochs" else None,
                    "tokens_per_epoch": train_len,
                    "vram_used_gb": round(vram_gb, 2),
                    "memory_allocated_mb": round(vram_gb * 1024, 1),
                    "tokens_seen": total_tokens,
                    "tok_per_sec": round(tok_per_sec, 1),
                    "tokens_per_second": round(tok_per_sec, 1),
                    "tflops": round(tflops, 3),
                    "phase": phase,
                    "early_stopping_enabled": es_enabled,
                    "early_stopping_patience": es_patience,
                    "early_stopping_counter": detector.counter,
                    "early_stopping_min_delta": es_min_delta,
                    "best_val_loss": round(detector.best_val_loss, 4) if detector.best_val_loss < float("inf") else None,
                    "best_step": detector.best_step,
                    "timestamp": time.time(),
                }

                metrics_buffer.append(metric_dict)
                if len(metrics_buffer) >= 5 or val_loss is not None:
                    self.db.log_metrics_batch(run_id, metrics_buffer)
                    metrics_buffer.clear()

                # Dispatch step event
                self._send_event(WorkerEvent(
                    event_type=EventType.STEP_COMPLETED,
                    run_id=run_id,
                    step=step,
                    payload=metric_dict,
                ))

                # Check completion targets
                is_completed = False
                if duration_mode == "epochs":
                    if session_tokens >= session_target_tokens or session_curr_steps >= session_total_steps:
                        logger.info(
                            "Target epochs reached: %d epochs (%d session tokens, %d steps) at step %d",
                            target_epochs, session_tokens, session_curr_steps, step,
                        )
                        is_completed = True
                else:
                    if step >= target_steps or session_curr_steps >= session_total_steps:
                        logger.info(
                            "Target steps reached: step %d (target %d, session %d/%d steps)",
                            step, target_steps, session_curr_steps, session_total_steps,
                        )
                        is_completed = True

                if is_completed:
                    # Ensure final validation pass runs so the run completes with true final evaluation metrics
                    if val_inputs is not None and self.latest_val_step != step:
                        logger.info("Executing final completion validation sweep at step %d...", step)
                        model.eval()
                        val_loss_acc = 0.0
                        with torch.no_grad():
                            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
                                for v_idx in range(val_batches):
                                    v_start = (v_idx * batch_len) % val_len
                                    v_idx_tensor = (torch.arange(batch_len, device=device) + v_start) % val_len
                                    if is_seq_tokens:
                                        v_in_batch = val_inputs[v_idx_tensor].view(training_cfg.micro_batch_size, training_cfg.seq_len)
                                        v_tgt_batch = (
                                            val_labels[v_idx_tensor].view(training_cfg.micro_batch_size, training_cfg.seq_len)
                                            if val_labels is not None
                                            else v_in_batch
                                        )
                                    else:
                                        v_in_batch = val_inputs[v_idx_tensor]
                                        v_tgt_batch = val_labels[v_idx_tensor] if val_labels is not None else None

                                    v_loss_t, _, _ = task_adapter.forward_loss(
                                        model=model,
                                        inputs=v_in_batch,
                                        targets=v_tgt_batch,
                                        loss_fn=trainer.loss_fn,
                                        autocast_dtype=training_cfg.dtype,
                                    )
                                    val_loss_acc += float(v_loss_t.item())

                        val_loss = val_loss_acc / val_batches
                        self.latest_val_loss = val_loss
                        self.latest_val_step = step
                        self.latest_val_timestamp = time.time()

                        val_record = detector.evaluate_step(
                            step=step,
                            train_loss=train_loss,
                            val_loss=val_loss,
                            model=model,
                        )
                        gen_gap = val_record.generalization_gap

                        if detector.best_step == step:
                            logger.info("Final step %d achieved new best validation loss: %.4f", step, val_loss)
                            self.checkpoint_manager.save_checkpoint(
                                model=model,
                                step=step,
                                tokens_seen=total_tokens,
                                optimizers=(muon_opt, adamw_opt),
                                scheduler=scheduler,
                                phase=phase,
                                val_loss=val_loss,
                                train_loss=train_loss,
                                run_id=run_id,
                                is_best=True,
                            )

                    # Generate training and validation charts in checkpoint folder
                    try:
                        from ..checkpoints.charts import generate_training_charts
                        run_metrics = self.db.get_metrics_history(run_id, limit=3000)
                        generate_training_charts(
                            metrics=run_metrics,
                            output_dir=self.checkpoint_manager.checkpoint_dir,
                            model_id=run_id or "run",
                            best_step=detector.best_step,
                            best_val_loss=detector.best_val_loss if detector.best_val_loss < float("inf") else self.latest_val_loss,
                        )
                    except Exception as ce:
                        logger.warning("Could not auto-generate training charts on completion: %s", ce)

                    self.checkpoint_manager.save_checkpoint(
                        model=model,
                        step=step,
                        tokens_seen=total_tokens,
                        optimizers=(muon_opt, adamw_opt),
                        scheduler=scheduler,
                        phase=phase,
                        val_loss=self.latest_val_loss,
                        train_loss=train_loss,
                        run_id=run_id,
                        filename_prefix="final_phase_completed",
                    )
                    try:
                        from ..notifications.webhook import get_notifier
                        get_notifier(db=self.db).notify_run_completed(
                            run_id=run_id,
                            step=step,
                            final_loss=float(self.latest_val_loss) if self.latest_val_loss is not None else (float(val_loss) if val_loss is not None else None),
                            tokens_seen=total_tokens,
                        )
                    except Exception as n_err:
                        logger.debug("Webhook notification error: %s", n_err)

                    self.state_machine.transition_to(RunState.COMPLETED)
                    self._send_event(WorkerEvent(
                        event_type=EventType.RUN_COMPLETED,
                        run_id=run_id,
                        step=step,
                        payload={
                            "final_val_loss": self.latest_val_loss,
                            "final_train_loss": train_loss,
                            "best_val_loss": detector.best_val_loss,
                            "best_step": detector.best_step,
                        },
                    ))
                    break

                # Brief yield
                time.sleep(0.01)

        except Exception as e:
            logger.exception("Fatal error in training session: %s", e)
            self.db.log_alert(
                level="CRITICAL",
                category="WORKER_CRASH",
                message=str(e),
                run_id=run_id,
                step=step,
            )
            self._send_event(WorkerEvent(
                event_type=EventType.RUN_FAILED,
                run_id=run_id,
                step=step,
                payload={"error": str(e)},
            ))
            if self.state_machine.can_transition_to(RunState.FAILED):
                self.state_machine.transition_to(RunState.FAILED)
        finally:
            if self.stop_requested or self.state_machine.current_state in (RunState.STOPPING, RunState.RUNNING):
                logger.info("Finalizing run stop at step %d", step)
                try:
                    detector.restore_best(model)
                    self.checkpoint_manager.save_checkpoint(
                        model=model,
                        step=step,
                        tokens_seen=total_tokens,
                        optimizers=(muon_opt, adamw_opt),
                        scheduler=scheduler,
                        phase=phase,
                        run_id=run_id,
                        filename_prefix="stopped_snapshot",
                    )
                    try:
                        from ..checkpoints.charts import generate_training_charts
                        run_metrics = self.db.get_metrics_history(run_id, limit=3000)
                        generate_training_charts(
                            metrics=run_metrics,
                            output_dir=self.checkpoint_manager.checkpoint_dir,
                            model_id=run_id or "run",
                            best_step=detector.best_step,
                            best_val_loss=detector.best_val_loss if detector.best_val_loss < float("inf") else self.latest_val_loss,
                        )
                    except Exception as ce:
                        logger.debug("Chart generation on stop: %s", ce)
                except Exception as save_err:
                    logger.warning("Could not save stopped snapshot: %s", save_err)

                if self.state_machine.current_state == RunState.RUNNING:
                    if self.state_machine.can_transition_to(RunState.STOPPING):
                        self.state_machine.transition_to(RunState.STOPPING)
                if self.state_machine.can_transition_to(RunState.STOPPED):
                    self.state_machine.transition_to(RunState.STOPPED)
                else:
                    self.state_machine.force_state(RunState.STOPPED)

                self._send_event(WorkerEvent(
                    event_type=EventType.RUN_STOPPED,
                    run_id=run_id,
                    step=step,
                    payload={"status": "stopped", "step": step},
                ))

            if metrics_buffer:
                self.db.log_metrics_batch(run_id, metrics_buffer)
                metrics_buffer.clear()
            self.db.update_run_status(run_id, self.state_machine.current_state.value, step=step, tokens=total_tokens)
            logger.info("Session %s ended with state %s at step %d", run_id, self.state_machine.current_state.value, step)

            # Comprehensive GPU & CPU resource cleanup
            try:
                del model
                del trainer
                del muon_opt
                del adamw_opt
                del scheduler
                del train_inputs
                del train_labels
                del val_inputs
                del val_labels
            except Exception:
                pass
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


def worker_process_entrypoint(
    cmd_queue: Any,
    event_queue: Any,
    device_str: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:
    """Multiprocessing entrypoint safely initialized via spawn."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [RoCanonWorker] %(message)s",
    )
    worker = TrainingWorkerProcess(
        cmd_queue=cmd_queue,
        event_queue=event_queue,
        device_str=device_str,
        db_path=db_path,
    )
    worker.run_loop()

