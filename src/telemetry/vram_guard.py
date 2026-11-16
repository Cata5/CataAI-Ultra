"""
src/telemetry/vram_guard.py
Dynamic VRAM safety monitoring and protective threshold enforcement for LLM Maker.
Adapts dynamically to the detected hardware profile and user-selected safety ceiling.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from .profiles import HardwareProfile

logger = logging.getLogger(__name__)


class VRAMAlertLevel(str, enum.Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


@dataclass
class VRAMThresholdConfig:
    warning_mb: float = 600.0   # Free VRAM below warning triggers WARNING
    critical_mb: float = 350.0   # Free VRAM below critical triggers CRITICAL
    emergency_mb: float = 150.0  # Free VRAM below emergency triggers EMERGENCY
    max_pytorch_allocated_gb: float = 5.0  # Ceiling for PyTorch allocated

    @classmethod
    def from_profile(cls, profile: HardwareProfile) -> VRAMThresholdConfig:
        return cls(
            warning_mb=profile.warning_threshold_mb,
            critical_mb=profile.critical_threshold_mb,
            emergency_mb=profile.emergency_threshold_mb,
            max_pytorch_allocated_gb=round(profile.max_memory_allocated_mb / 1024.0, 2),
        )


class VRAMGuard:
    """Evaluates real available VRAM against configurable protection thresholds."""

    def __init__(self, config: Optional[VRAMThresholdConfig] = None) -> None:
        self.config = config or VRAMThresholdConfig()

    def update_from_profile(self, profile: HardwareProfile) -> None:
        self.config = VRAMThresholdConfig.from_profile(profile)

    def evaluate(self, free_vram_mb: Optional[float], pytorch_allocated_mb: Optional[float] = None) -> Tuple[VRAMAlertLevel, str]:
        if free_vram_mb is None:
            return VRAMAlertLevel.NORMAL, "VRAM telemetry unavailable"

        if free_vram_mb < self.config.emergency_mb:
            msg = f"EMERGENCY: Free VRAM is critically low ({free_vram_mb:.1f} MB < {self.config.emergency_mb:.0f} MB). Immediate protective checkpoint requested."
            return VRAMAlertLevel.EMERGENCY, msg

        if free_vram_mb < self.config.critical_mb:
            msg = f"CRITICAL: Low free VRAM ({free_vram_mb:.1f} MB < {self.config.critical_mb:.0f} MB). Prevent further memory allocation."
            return VRAMAlertLevel.CRITICAL, msg

        if free_vram_mb < self.config.warning_mb:
            msg = f"WARNING: Free VRAM decreasing ({free_vram_mb:.1f} MB < {self.config.warning_mb:.0f} MB)."
            return VRAMAlertLevel.WARNING, msg

        # Also check PyTorch allocated ceiling
        if pytorch_allocated_mb is not None:
            max_pt_mb = self.config.max_pytorch_allocated_gb * 1024.0
            if pytorch_allocated_mb > max_pt_mb:
                msg = f"WARNING: PyTorch allocated ({pytorch_allocated_mb:.1f} MB) exceeds budget ({max_pt_mb:.0f} MB)."
                return VRAMAlertLevel.WARNING, msg

        return VRAMAlertLevel.NORMAL, f"VRAM healthy ({free_vram_mb:.1f} MB free)"
