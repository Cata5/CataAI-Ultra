"""
src/telemetry/profiles.py
Hardware capability profiles, memory allocation policies, and dynamic resource budgeting for LLM Maker.
Replaces hardcoded GPU/VRAM assumptions with dynamic, hardware-aware safety policies.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ResourceProfileMode(str, enum.Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    PERFORMANCE = "performance"
    CUSTOM = "custom"


@dataclass
class HardwareProfile:
    """Configurable resource profile that governs safety thresholds, memory ceilings, and defaults."""
    mode: str = "balanced"
    device_type: str = "cuda"  # "cuda", "mps", "cpu", "rocm"
    total_memory_mb: float = 6144.0  # Total GPU VRAM or system RAM
    usable_ratio: float = 0.80  # Default 80% ceiling for balanced
    max_memory_allocated_mb: float = 4915.2
    warning_threshold_mb: float = 614.4   # ~10% free
    critical_threshold_mb: float = 368.6  # ~6% free
    emergency_threshold_mb: float = 184.3 # ~3% free
    recommended_micro_batch: int = 4
    enable_grad_checkpointing: bool = False
    supported_precisions: list[str] = None  # e.g. ["fp32", "fp16", "bfloat16"]
    active_precision: str = "bfloat16"

    def __post_init__(self):
        if self.supported_precisions is None:
            self.supported_precisions = ["fp32"]

    @classmethod
    def create(
        cls,
        mode: str = "balanced",
        total_vram_mb: Optional[float] = None,
        total_sys_ram_mb: Optional[float] = None,
        device_type: str = "cuda",
        bf16_supported: bool = False,
        fp16_supported: bool = False,
        custom_ratio: Optional[float] = None,
        custom_ceiling_mb: Optional[float] = None,
    ) -> HardwareProfile:
        norm_mode = mode.lower()
        if norm_mode not in ("conservative", "balanced", "performance", "custom"):
            norm_mode = "balanced"

        # Determine reference total memory
        if device_type in ("cuda", "rocm", "mps") and total_vram_mb and total_vram_mb > 0:
            total_mem = total_vram_mb
        else:
            total_mem = total_sys_ram_mb or 16384.0  # Fallback to 16 GB RAM if undetected

        # Determine ratio
        if norm_mode == "conservative":
            ratio = 0.65
        elif norm_mode == "performance":
            ratio = 0.90
        elif norm_mode == "custom" and custom_ratio is not None:
            ratio = max(0.20, min(0.98, custom_ratio))
        else:  # balanced
            ratio = 0.80

        if custom_ceiling_mb and custom_ceiling_mb > 0 and norm_mode == "custom":
            max_alloc = min(custom_ceiling_mb, total_mem)
        else:
            max_alloc = round(total_mem * ratio, 1)

        # Dynamic thresholds
        warn = round(total_mem * 0.10, 1)
        crit = round(total_mem * 0.06, 1)
        emerg = round(total_mem * 0.03, 1)

        # Ensure safety floors
        warn = max(warn, 300.0)
        crit = max(crit, 150.0)
        emerg = max(emerg, 80.0)

        # Recommended batch size
        if total_mem < 4096:
            rec_batch = 1
            grad_ckpt = True
        elif total_mem < 8192:
            rec_batch = 2 if norm_mode == "conservative" else 4
            grad_ckpt = (norm_mode == "conservative")
        elif total_mem < 16384:
            rec_batch = 4 if norm_mode == "conservative" else 8
            grad_ckpt = False
        else:
            rec_batch = 8 if norm_mode == "conservative" else 16
            grad_ckpt = False

        # Precision support
        precisions = ["fp32"]
        if fp16_supported:
            precisions.append("fp16")
        if bf16_supported:
            precisions.append("bfloat16")

        active_p = "bfloat16" if bf16_supported else ("fp16" if fp16_supported else "fp32")

        return cls(
            mode=norm_mode,
            device_type=device_type,
            total_memory_mb=round(total_mem, 1),
            usable_ratio=ratio,
            max_memory_allocated_mb=max_alloc,
            warning_threshold_mb=warn,
            critical_threshold_mb=crit,
            emergency_threshold_mb=emerg,
            recommended_micro_batch=rec_batch,
            enable_grad_checkpointing=grad_ckpt,
            supported_precisions=precisions,
            active_precision=active_p,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["max_memory_allocated_gb"] = round(self.max_memory_allocated_mb / 1024.0, 2)
        d["total_memory_gb"] = round(self.total_memory_mb / 1024.0, 2)
        return d
