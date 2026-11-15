"""
src/dashboard/early_stopping.py
Early stopping and overfitting detection engine for RoCanon-48.5M.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class EarlyStoppingConfig:
    """
    Configuration parameters for early stopping and overfitting detection.
    """
    enabled: bool = True               # Enable or disable early stopping
    patience: int = 5                  # Number of validation checks without improvement
    min_delta: float = 0.001           # Minimum decrease in loss to qualify as improvement
    overfitting_ratio: float = 1.15    # Ratio (val_loss / train_loss) that triggers an overfitting warning
    restore_best_weights: bool = True  # Automatically restore weights from lowest val loss checkpoint
    checkpoint_dir: str = "checkpoints"


@dataclass
class ValidationRecord:
    step: int
    train_loss: float
    val_loss: float
    generalization_gap: float
    is_best: bool
    status: str  # "healthy", "improving", "stalling", "overfitting"


class OverfittingDetector:
    """
    Detects the onset of overfitting by monitoring:
    1. Divergence between Train Loss and Validation Loss (Generalization Gap).
    2. Consecutive non-improving validation evaluations.
    3. Sudden increases in validation loss while training loss drops.
    """

    def __init__(self, config: Optional[EarlyStoppingConfig] = None):
        self.config = config or EarlyStoppingConfig()
        self.best_val_loss: float = float("inf")
        self.best_step: int = 0
        self.counter: int = 0
        self.stop_requested: bool = False
        self.history: List[ValidationRecord] = []
        self.status: str = "INITIALIZING"
        self.best_state_dict: Optional[Dict[str, Any]] = None

    def evaluate_step(
        self,
        step: int,
        train_loss: float,
        val_loss: float,
        model: Optional[nn.Module] = None,
    ) -> ValidationRecord:
        """
        Processes a new validation point and updates early stopping state.
        """
        if train_loss is None:
            train_loss = val_loss if val_loss is not None else 0.0
        gap = val_loss - train_loss
        gap_ratio = (val_loss / train_loss) if train_loss > 0 else 1.0

        is_improvement = (self.best_val_loss - val_loss) > self.config.min_delta

        if is_improvement:
            self.best_val_loss = val_loss
            self.best_step = step
            self.counter = 0
            is_best = True
            status = "HEALTHY (IMPROVING)"

            # Save best weights
            if model is not None and self.config.restore_best_weights:
                self.best_state_dict = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
            logger.info(f"New best val loss: {val_loss:.4f} at step {step}. Marked for checkpointing.")
        else:
            self.counter += 1
            is_best = False

            if self.config.enabled and self.counter >= self.config.patience:
                status = "EARLY STOPPING TRIGGERED"
                self.stop_requested = True
                logger.warning(
                    f"Early stopping triggered at step {step}: no improvement for {self.counter} checks. "
                    f"Best val loss: {self.best_val_loss:.4f} at step {self.best_step}."
                )
            elif gap_ratio >= self.config.overfitting_ratio:
                status = "OVERFITTING DETECTED (GAP DIVERGENCE)"
            else:
                status = f"STALLING ({self.counter}/{self.config.patience})" if self.config.enabled else "NO IMPROVEMENT"

        self.status = status
        record = ValidationRecord(
            step=step,
            train_loss=train_loss,
            val_loss=val_loss,
            generalization_gap=gap,
            is_best=is_best,
            status=status,
        )
        self.history.append(record)
        return record

    def restore_best(self, model: nn.Module) -> bool:
        """Restores the best weights to the model."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            logger.info(f"Restored best weights from step {self.best_step} (val loss: {self.best_val_loss:.4f})")
            return True
        return False

