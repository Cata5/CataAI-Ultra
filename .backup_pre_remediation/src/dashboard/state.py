"""
src/dashboard/state.py
Thread-safe training state manager, configuration repository, and event broadcaster
for the RoCanon-48.5M full control cockpit.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from aiohttp import web

from ..training.factory import calculate_analytical_parameters, list_checkpoints


@dataclass
class MetricPoint:
    step: int
    tokens: int
    train_loss: float
    val_loss: Optional[float] = None
    generalization_gap: Optional[float] = None
    grad_norm: float = 0.0
    muon_lr: float = 0.0
    adamw_lr: float = 0.0
    vram_used_gb: float = 0.0
    tok_per_sec: float = 0.0
    tflops: float = 0.0
    timestamp: float = field(default_factory=time.time)


class TrainingStateManager:
    """
    Manages all aspects of model architecture parameters, optimizer settings,
    data mixing ratios, live telemetry, and WebSocket broadcasts.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.metrics_history: List[MetricPoint] = []
        self.sample_generations: List[Dict[str, str]] = []
        self.chat_history: List[Dict[str, str]] = []

        # Training Control flags
        self.is_training: bool = False
        self.is_paused: bool = False
        self.stop_requested: bool = False
        self.start_run_requested: bool = False
        self.trigger_val_requested: bool = False
        self.manual_checkpoint_requested: bool = False
        self.export_requested: bool = False
        self.checkpoint_to_load: Optional[str] = None
        self.lr_multiplier: float = 1.0

        # Current live status
        self.current_step: int = 0
        self.total_tokens_processed: int = 0
        self.current_stage: str = "Stage 1: Pretraining (Tech & History)"
        self.current_phase: str = "pretrain"  # "pretrain", "midtrain", or "chat_sft"
        self.dataset_source: str = "tech_wiki_ro"
        self.resume_checkpoint_path: Optional[str] = None
        self.dataset_max_samples: Optional[int] = None
        self.dataset_status: str = "Ready (No dataset loaded)"
        self.dataset_progress_pct: float = 0.0
        self.dataset_total_tokens: int = 0
        self.dataset_train_tokens: int = 0
        self.dataset_val_tokens: int = 0
        self.active_model: Optional[Any] = None
        self.overfitting_status: str = "HEALTHY (READY)"
        self.best_val_loss: Optional[float] = None
        self.best_val_step: Optional[int] = None
        self.early_stopping_counter: int = 0
        self.early_stopping_patience: int = 5
        self.active_device: str = "cuda" if torch_cuda_available() else "cpu"

        # 1. Architecture Configuration (Full Control)
        self.config_arch: Dict[str, Any] = {
            "vocab_size": 12288,
            "d_model": 512,
            "n_layers": 16,
            "n_q": 8,
            "n_kv": 2,
            "d_ff": 1280,
            "kernel_size": 4,
            "rope_theta": 10000.0,
            "max_seq_len": 2048,
            "tie_word_embeddings": True,
            "rms_norm_eps": 1e-6,
        }

        # 2. Optimizer & Hardware Tuning Configuration
        self.config_opt: Dict[str, Any] = {
            "dtype": "bfloat16" if torch_cuda_available() else "float32",
            "micro_batch_size": 4,
            "seq_len": 256,
            "grad_accum_steps": 2,
            "max_grad_norm": 1.0,
            "muon_lr": 3.0e-3,
            "muon_weight_decay": 0.01,
            "muon_momentum": 0.95,
            "muon_nesterov": True,
            "muon_ns_steps": 5,
            "adamw_lr": 6.0e-4,
            "adamw_weight_decay": 0.01,
            "adamw_beta1": 0.9,
            "adamw_beta2": 0.95,
            "adamw_eps": 1.0e-8,
            "warmup_steps": 2000,
            "plateau_tokens": 5_000_000_000,
            "total_tokens": 6_800_000_000,
            "min_lr_ratio": 0.1,
            "decay_style": "cosine",
            "max_budget_gb": 5.0,
        }

        # 3. Data Mixing & Curation Configuration
        self.config_data: Dict[str, Any] = {
            "romanian_edu": 35,
            "english_edu": 30,
            "cosmopedia_textbooks": 15,
            "finemath_reasoning": 10,
            "stack_code": 10,
            "fasttext_lid_threshold": 0.80,
            "minhash_lsh_threshold": 0.80,
            "mask_pii": True,
            "padding_free_packing": True,
        }

        # 4. Overfitting & Early Stopping Configuration
        self.config_early_stopping: Dict[str, Any] = {
            "patience": 5,
            "min_delta": 0.002,
            "overfitting_ratio": 1.15,
            "restore_best_weights": True,
            "eval_frequency": 10,
        }

        # 5. Inference Sampling Configuration
        self.config_inference: Dict[str, Any] = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "max_new_tokens": 128,
        }

        # Connected WebSocket clients
        self.ws_clients: Set[web.WebSocketResponse] = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    def add_metric(self, point: MetricPoint):
        with self.lock:
            self.metrics_history.append(point)
            self.current_step = point.step
            self.total_tokens_processed = point.tokens

            if len(self.metrics_history) > 2000:
                self.metrics_history = self.metrics_history[-2000:]

        self.broadcast_update({
            "type": "metric_point",
            "data": asdict(point),
            "status": self.get_summary(),
        })

    def add_sample_generation(self, prompt: str, completion: str, step: int):
        with self.lock:
            record = {"step": step, "prompt": prompt, "completion": completion, "time": time.strftime("%H:%M:%S")}
            self.sample_generations.append(record)
            if len(self.sample_generations) > 20:
                self.sample_generations = self.sample_generations[-20:]

        self.broadcast_update({
            "type": "sample_generation",
            "data": record,
        })

    def update_early_stopping(
        self,
        status: str,
        best_val: float,
        best_step: int,
        counter: int,
        patience: int,
    ):
        with self.lock:
            self.overfitting_status = status
            self.best_val_loss = best_val
            self.best_val_step = best_step
            self.early_stopping_counter = counter
            self.early_stopping_patience = patience

        self.broadcast_update({
            "type": "overfitting_update",
            "status": status,
            "best_val": best_val,
            "best_step": best_step,
            "counter": counter,
            "patience": patience,
        })

    def update_dataset_progress(
        self,
        status: str,
        pct: float = 0.0,
        total_tokens: int = 0,
        train_tokens: int = 0,
        val_tokens: int = 0,
    ):
        with self.lock:
            self.dataset_status = status
            self.dataset_progress_pct = round(pct, 1)
            if total_tokens:
                self.dataset_total_tokens = total_tokens
            if train_tokens:
                self.dataset_train_tokens = train_tokens
            if val_tokens:
                self.dataset_val_tokens = val_tokens

        self.broadcast_update({
            "type": "dataset_progress",
            "status": self.dataset_status,
            "progress_pct": self.dataset_progress_pct,
            "total_tokens": self.dataset_total_tokens,
            "train_tokens": self.dataset_train_tokens,
            "val_tokens": self.dataset_val_tokens,
        })

    def get_full_config(self) -> Dict[str, Any]:
        """Returns the complete configuration dictionary across all modules."""
        with self.lock:
            param_stats = calculate_analytical_parameters(
                vocab_size=int(self.config_arch["vocab_size"]),
                d_model=int(self.config_arch["d_model"]),
                n_layers=int(self.config_arch["n_layers"]),
                n_q=int(self.config_arch["n_q"]),
                n_kv=int(self.config_arch["n_kv"]),
                d_ff=int(self.config_arch["d_ff"]),
                kernel_size=int(self.config_arch["kernel_size"]),
                tie_word_embeddings=bool(self.config_arch["tie_word_embeddings"]),
            )
            return {
                "arch": dict(self.config_arch),
                "opt": dict(self.config_opt),
                "data": dict(self.config_data),
                "early_stopping": dict(self.config_early_stopping),
                "inference": dict(self.config_inference),
                "calculated_params": param_stats,
                "checkpoints": list_checkpoints(),
                "status": self.get_summary(),
            }

    def update_config_section(self, section: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Updates a specific section of the configuration."""
        with self.lock:
            if section == "arch":
                self.config_arch.update(updates)
            elif section == "opt":
                self.config_opt.update(updates)
                if "lr_multiplier" in updates:
                    self.lr_multiplier = float(updates["lr_multiplier"])
            elif section == "data":
                self.config_data.update(updates)
            elif section == "early_stopping":
                self.config_early_stopping.update(updates)
            elif section == "inference":
                self.config_inference.update(updates)
            elif section == "phase":
                self.current_phase = updates.get("phase", self.current_phase)
                self.dataset_source = updates.get("dataset_source", self.dataset_source)
                self.resume_checkpoint_path = updates.get("resume_checkpoint_path", self.resume_checkpoint_path)
                if "max_samples" in updates:
                    val = updates["max_samples"]
                    self.dataset_max_samples = int(val) if val else None
                if self.current_phase == "chat_sft":
                    self.current_stage = "Stage 3: SFT Chat & Creator Persona (CataAI)"
                elif self.current_phase == "midtrain":
                    self.current_stage = "Stage 2: Mid-Training (Reasoning & Alpaca)"
                else:
                    self.current_stage = "Stage 1: Pretraining (Tech & History)"

        full_cfg = self.get_full_config()
        self.broadcast_update({
            "type": "config_updated",
            "config": full_cfg,
        })
        return full_cfg

    def get_summary(self) -> Dict[str, Any]:
        with self.lock:
            last_pt = self.metrics_history[-1] if self.metrics_history else None
            return {
                "is_training": self.is_training,
                "is_paused": self.is_paused,
                "current_step": self.current_step,
                "total_tokens": self.total_tokens_processed,
                "stage": self.current_stage,
                "current_phase": self.current_phase,
                "dataset_source": self.dataset_source,
                "dataset_status": self.dataset_status,
                "dataset_progress_pct": self.dataset_progress_pct,
                "dataset_total_tokens": self.dataset_total_tokens,
                "dataset_train_tokens": self.dataset_train_tokens,
                "dataset_val_tokens": self.dataset_val_tokens,
                "resume_checkpoint_path": self.resume_checkpoint_path,
                "train_loss": last_pt.train_loss if last_pt else None,
                "val_loss": last_pt.val_loss if last_pt else None,
                "generalization_gap": last_pt.generalization_gap if last_pt else None,
                "grad_norm": last_pt.grad_norm if last_pt else None,
                "muon_lr": (last_pt.muon_lr * self.lr_multiplier) if last_pt else float(self.config_opt["muon_lr"]),
                "adamw_lr": (last_pt.adamw_lr * self.lr_multiplier) if last_pt else float(self.config_opt["adamw_lr"]),
                "vram_gb": last_pt.vram_used_gb if last_pt else 0.0,
                "tok_per_sec": last_pt.tok_per_sec if last_pt else 0.0,
                "tflops": last_pt.tflops if last_pt else 0.0,
                "overfitting_status": self.overfitting_status,
                "best_val_loss": self.best_val_loss,
                "best_val_step": self.best_val_step,
                "early_stopping_counter": self.early_stopping_counter,
                "early_stopping_patience": self.config_early_stopping.get("patience", 5),
                "lr_multiplier": self.lr_multiplier,
                "device": self.active_device,
            }

    def broadcast_update(self, payload: Dict[str, Any]):
        if not self.loop or not self.ws_clients:
            return

        msg_str = json.dumps(payload)

        async def _send():
            dead_clients = set()
            for ws in list(self.ws_clients):
                if ws.closed:
                    dead_clients.add(ws)
                    continue
                try:
                    await ws.send_str(msg_str)
                except Exception:
                    dead_clients.add(ws)
            self.ws_clients.difference_update(dead_clients)

        asyncio.run_coroutine_threadsafe(_send(), self.loop)


def torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# Global singleton instance
STATE = TrainingStateManager()
