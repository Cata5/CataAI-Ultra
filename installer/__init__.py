"""First-run installation helpers for LLM Maker.

The package is deliberately standard-library only: it runs before the training
environment exists, detects the machine, and chooses the PyTorch wheel that
matches the installed NVIDIA driver.
"""

from .hardware import Gpu, HardwareProfile, detect_hardware
from .plan import TorchPlan, choose_plan

__all__ = [
    "Gpu",
    "HardwareProfile",
    "TorchPlan",
    "choose_plan",
    "detect_hardware",
]

__version__ = "1.0.0"