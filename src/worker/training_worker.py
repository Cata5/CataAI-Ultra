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

        phase = payload.get("phase", "pretrain")
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
        checkpoint_to_load = payload.get("checkpoint_to_load")
        target_steps = int(payload.get("target_steps", 1000))
        target_tokens = int(payload.get("target_tokens", 10_000_000_000_000))  # Rely exclusively on target steps
        eval_freq = int(payload.get("eval_frequency", 10))
        save_freq = int(payload.get("save_frequency", 50))
        es_enabled = bool(payload.get("early_stopping_enabled", True))
        es_patience = int(payload.get("early_stopping_patience", 5))
        es_min_delta = float(payload.get("early_stopping_min_delta", 0.001))

        if "muon_lr" in payload:
            config_opt["muon_lr"] = float(payload["muon_lr"])
        if "adamw_lr" in payload:
            config_opt["adamw_lr"] = float(payload["adamw_lr"])
        if "micro_batch_size" in payload:
            config_opt["micro_batch_size"] = int(payload["micro_batch_size"])
        if "grad_accum_steps" in payload:
            config_opt["grad_accum_steps"] = int(payload["grad_accum_steps"])

        self.state_machine.transition_to(RunState.STARTING)
        self.db.create_run(
            run_id=run_id,
            name=payload.get("run_name", f"RoCanon Run {run_id}"),
            current_phase=phase,
            dataset_source=str(dataset_source),
            config={"arch": config_arch, "opt": config_opt, "es": config_es},
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
        except Exception as e:
            logger.exception("Failed to initialize model: %s", e)
            self._send_event(WorkerEvent(
                event_type=EventType.RUN_FAILED,
                run_id=run_id,
                payload={"error": f"Model initialization failed: {e}"},
            ))
            self.state_machine.transition_to(RunState.FAILED)
            return

        # 2. Load Checkpoint if provided
        start_step = 0
        total_tokens = 0
        if checkpoint_to_load:
            try:
                load_res = self.checkpoint_manager.load_checkpoint(
                    checkpoint_path=checkpoint_to_load,
                    model=model,
                    device=device,
                    restore_training_state=False,
                )
                start_step = int(load_res.get("step", 0))
                total_tokens = int(load_res.get("tokens_seen", 0))
                logger.info("Loaded weights from %s (starting step: %d, tokens: %d)", checkpoint_to_load, start_step, total_tokens)
            except Exception as e:
                logger.warning("Could not load checkpoint %s: %s", checkpoint_to_load, e)

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

        # 3. Configure Optimizers & Scheduler
        try:
            muon_opt, adamw_opt = create_optimizers_from_dict(model, config_opt)
            scheduler = WSDScheduler(
                optimizers=(muon_opt, adamw_opt),
                warmup_steps=int(config_opt.get("warmup_steps", 2000)),
                plateau_tokens=int(config_opt.get("plateau_tokens", 5_000_000_000)),
                total_tokens=int(config_opt.get("total_tokens", 6_800_000_000)),
                min_lr_ratio=float(config_opt.get("min_lr_ratio", 0.1)),
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

            trainer = Trainer(
                model=model,
                optimizers=(muon_opt, adamw_opt),
                scheduler=scheduler,
                config=training_cfg,
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

        # 4. Configure Early Stopping
        es_cfg = EarlyStoppingConfig(
            enabled=es_enabled,
            patience=es_patience,
            min_delta=es_min_delta,
            overfitting_ratio=float(config_es.get("overfitting_ratio", 1.15)),
            checkpoint_dir="checkpoints",
        )
        detector = OverfittingDetector(es_cfg)

        # 5. Load Dataset
        tokenizer = RoCanonTokenizer()
        dataset_engine = CustomDatasetEngine(tokenizer=tokenizer)

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

        batch_len = training_cfg.micro_batch_size * training_cfg.seq_len
        train_len = len(train_inputs)
        val_len = len(val_inputs)
        val_batches = max(1, min(4, val_len // batch_len))

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
        base_muon = float(config_opt.get("muon_lr", 3e-3))
        base_adamw = float(config_opt.get("adamw_lr", 6e-4))
        eval_freq = int(payload.get("eval_frequency") or config_es.get("eval_frequency", 10))
        save_freq = int(payload.get("save_frequency", 50))
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
                elif session_step > int(session_total_steps * 0.85):
                    decay_span = max(1, session_total_steps - int(session_total_steps * 0.85))
                    decay_progress = (session_step - int(session_total_steps * 0.85)) / float(decay_span)
                    sched_factor = 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, decay_progress)))
                else:
                    sched_factor = 1.0

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
                    for accum_idx in range(training_cfg.grad_accum_steps):
                        start_pos = (step * training_cfg.grad_accum_steps + accum_idx) * batch_len
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

                        trainer.accumulate_micro_batch(input_ids=batch_in, labels=batch_lbl)

                        self._poll_commands()
                        if self.stop_requested or self.state_machine.current_state == RunState.STOPPING:
                            break

                    if self.stop_requested or self.state_machine.current_state == RunState.STOPPING:
                        logger.info("Stopping requested during micro-batch accumulation at step %d", step)
                        break

                    tokens_in_step = training_cfg.micro_batch_size * (training_cfg.seq_len - 1) * training_cfg.grad_accum_steps
                    total_tokens += tokens_in_step

                    grad_norm = trainer.optimizer_step(num_tokens_in_step=total_tokens)
                    train_loss = trainer.loss_accumulator.reset()
                    dt = time.time() - t0

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
                if (step % eval_freq == 0) or self.manual_val_requested or (step == 1):
                    self.manual_val_requested = False
                    model.eval()
                    val_loss_acc = 0.0
                    with torch.no_grad():
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
                            for v_idx in range(val_batches):
                                v_start = (v_idx * batch_len) % val_len
                                v_idx_tensor = (torch.arange(batch_len, device=device) + v_start) % val_len
                                v_in_batch = val_inputs[v_idx_tensor].view(training_cfg.micro_batch_size, training_cfg.seq_len)
                                v_tgt_batch = (
                                    val_labels[v_idx_tensor].view(training_cfg.micro_batch_size, training_cfg.seq_len)
                                    if val_labels is not None
                                    else v_in_batch
                                )

                                v_logits = model(v_in_batch)
                                if isinstance(v_logits, tuple):
                                    v_logits = v_logits[0]
                                v_shift_logits = v_logits[..., :-1, :].contiguous()
                                v_shift_targets = v_tgt_batch[..., 1:].contiguous()
                                v_loss_t, _ = trainer.loss_fn(v_shift_logits, v_shift_targets)
                                val_loss_acc += float(v_loss_t.item())

                    val_loss = val_loss_acc / val_batches
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

                epoch_val = round(total_tokens / train_len, 3) if train_len > 0 else 0.0
                tokens_in_curr_epoch = total_tokens % train_len if train_len > 0 else 0
                epoch_pct = round((tokens_in_curr_epoch / train_len) * 100, 2) if train_len > 0 else 0.0
                session_curr_steps = max(0, step - start_step)
                step_pct = round((session_curr_steps / session_total_steps) * 100, 2) if session_total_steps > 0 else 0.0
                token_pct = round((total_tokens / target_tokens) * 100, 4) if target_tokens > 0 else 0.0
                rem_tokens = max(0, target_tokens - total_tokens)
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
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4) if val_loss is not None else None,
                    "generalization_gap": round(gen_gap, 4) if gen_gap is not None else None,
                    "grad_norm": round(grad_norm, 4) if grad_norm is not None else 0.0,
                    "muon_lr": eff_muon_lr,
                    "adamw_lr": eff_adamw_lr,
                    "lr_multiplier": round(curr_mult, 4),
                    "adaptive_lr_factor": round(self.adaptive_lr_factor, 4),
                    "vram_used_gb": round(vram_gb, 2),
                    "tok_per_sec": round(tok_per_sec, 1),
                    "tflops": round(tflops, 3),
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
                if step >= target_steps or total_tokens >= target_tokens:
                    logger.info("Target steps/tokens reached: step %d (target %d, session %d/%d steps)", step, target_steps, session_curr_steps, session_total_steps)
                    self.checkpoint_manager.save_checkpoint(
                        model=model,
                        step=step,
                        tokens_seen=total_tokens,
                        optimizers=(muon_opt, adamw_opt),
                        scheduler=scheduler,
                        phase=phase,
                        run_id=run_id,
                        filename_prefix="final_phase_completed",
                    )
                    self.state_machine.transition_to(RunState.COMPLETED)
                    self._send_event(WorkerEvent(
                        event_type=EventType.RUN_COMPLETED,
                        run_id=run_id,
                        step=step,
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

