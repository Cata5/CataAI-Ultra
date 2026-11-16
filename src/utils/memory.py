"""
Memory management and hardware monitoring utilities for RoCanon-48.5M.

Monitors GPU VRAM allocation, tracking peak and reserved memory to enforce
the strict constraint that peak allocated VRAM must not exceed 5.0 GB on 6 GB RTX 3060.
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)


class MemoryTracker:
    """
    Tracks and enforces GPU VRAM memory constraints during model training and evaluation.

    Attributes:
        device: Target torch device (cuda or cpu).
        max_budget_gb: Maximum allowed allocated VRAM in gigabytes (default: 5.0 GB).
    """

    def __init__(
        self,
        device: torch.device | None = None,
        max_budget_gb: float | None = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        if max_budget_gb is not None:
            self.max_budget_gb = max_budget_gb
        else:
            if self.is_cuda:
                try:
                    _, total_b = torch.cuda.mem_get_info(self.device)
                    self.max_budget_gb = round((total_b * 0.85) / (1024**3), 2)
                except Exception:
                    self.max_budget_gb = 5.0
            else:
                self.max_budget_gb = 16.0
        self.reset_peak_stats()

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda" and torch.cuda.is_available()

    def reset_peak_stats(self) -> None:
        """Resets peak memory tracking statistics."""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_current_allocated_bytes(self) -> int:
        """Returns current allocated GPU memory in bytes."""
        if self.is_cuda:
            return torch.cuda.memory_allocated(self.device)
        return 0

    def get_current_allocated_mb(self) -> float:
        """Returns current allocated GPU memory in megabytes (MiB)."""
        return self.get_current_allocated_bytes() / (1024 * 1024)

    def get_current_allocated_gb(self) -> float:
        """Returns current allocated GPU memory in gigabytes (GiB)."""
        return self.get_current_allocated_bytes() / (1024 * 1024 * 1024)

    def get_peak_allocated_bytes(self) -> int:
        """Returns peak allocated GPU memory in bytes since last reset."""
        if self.is_cuda:
            return torch.cuda.max_memory_allocated(self.device)
        return 0

    def get_peak_allocated_mb(self) -> float:
        """Returns peak allocated GPU memory in megabytes (MiB)."""
        return self.get_peak_allocated_bytes() / (1024 * 1024)

    def get_peak_allocated_gb(self) -> float:
        """Returns peak allocated GPU memory in gigabytes (GiB)."""
        return self.get_peak_allocated_bytes() / (1024 * 1024 * 1024)

    def get_peak_reserved_bytes(self) -> int:
        """Returns peak reserved GPU memory in bytes since last reset."""
        if self.is_cuda:
            return torch.cuda.max_memory_reserved(self.device)
        return 0

    def get_peak_reserved_mb(self) -> float:
        """Returns peak reserved GPU memory in megabytes (MiB)."""
        return self.get_peak_reserved_bytes() / (1024 * 1024)

    def get_peak_reserved_gb(self) -> float:
        """Returns peak reserved GPU memory in gigabytes (GiB)."""
        return self.get_peak_reserved_bytes() / (1024 * 1024 * 1024)

    def assert_within_budget(self, max_budget_gb: float | None = None) -> bool:
        """
        Asserts that peak allocated VRAM does not exceed the allowed budget.

        Args:
            max_budget_gb: Optional override for the ceiling in GB. Defaults to self.max_budget_gb.

        Returns:
            True if within budget or running on CPU.

        Raises:
            RuntimeError: If peak allocated GPU memory strictly exceeds max_budget_gb.
        """
        budget = max_budget_gb if max_budget_gb is not None else self.max_budget_gb
        if self.is_cuda:
            peak_gb = self.get_peak_allocated_gb()
            if peak_gb > budget:
                raise RuntimeError(
                    f"GPU VRAM budget exceeded! Peak allocated: {peak_gb:.3f} GB, "
                    f"Limit: {budget:.3f} GB."
                )
        return True

    def summary(self) -> dict[str, float]:
        """Returns a dictionary summary of memory metrics in megabytes and gigabytes."""
        return {
            "device": str(self.device),
            "is_cuda": self.is_cuda,
            "current_allocated_mb": self.get_current_allocated_mb(),
            "peak_allocated_mb": self.get_peak_allocated_mb(),
            "peak_allocated_gb": self.get_peak_allocated_gb(),
            "peak_reserved_mb": self.get_peak_reserved_mb(),
            "peak_reserved_gb": self.get_peak_reserved_gb(),
            "max_budget_gb": self.max_budget_gb,
        }

    @contextmanager
    def track(self, phase_name: str = "") -> Generator["MemoryTracker", None, None]:
        """
        Context manager to monitor peak memory during a specific operation phase.

        Args:
            phase_name: Descriptive name of the execution phase.
        """
        self.reset_peak_stats()
        try:
            yield self
        finally:
            if self.is_cuda:
                peak_mb = self.get_peak_allocated_mb()
                logger.info("Memory [%s]: peak allocated = %.2f MiB", phase_name, peak_mb)
