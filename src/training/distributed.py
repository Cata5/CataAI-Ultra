"""
src/training/distributed.py
Distributed Multi-GPU Training Topology for CataAI Ultra.
Provides multi-GPU discovery, device selection, and DataParallel / DDP execution
for scaling model training across multiple local or remote CUDA devices.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def detect_available_compute() -> Dict[str, Any]:
    """Scans system for CUDA, MPS (Apple Silicon), or CPU compute topology."""
    has_cuda = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if has_cuda else 0
    devices: List[Dict[str, Any]] = []

    if has_cuda:
        for idx in range(gpu_count):
            props = torch.cuda.get_device_properties(idx)
            total_mb = round(props.total_memory / (1024 * 1024), 2)
            try:
                free_b, total_b = torch.cuda.mem_get_info(idx)
                free_mb = round(free_b / (1024 * 1024), 2)
            except Exception:
                free_mb = total_mb

            devices.append({
                "index": idx,
                "name": props.name,
                "total_vram_mb": total_mb,
                "free_vram_mb": free_mb,
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            })

    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    return {
        "has_cuda": has_cuda,
        "gpu_count": gpu_count,
        "is_multi_gpu": gpu_count > 1,
        "devices": devices,
        "has_mps": has_mps,
        "default_device": "cuda:0" if has_cuda else ("mps" if has_mps else "cpu"),
    }


def wrap_model_for_multi_gpu(
    model: nn.Module,
    device_ids: Optional[List[int]] = None,
) -> Tuple[nn.Module, List[int]]:
    """
    Wraps model in DataParallel across specified or all detected CUDA devices.
    If single GPU or CPU, returns model as-is.
    """
    if not torch.cuda.is_available():
        return model, []

    total_gpus = torch.cuda.device_count()
    if device_ids is None:
        target_devices = list(range(total_gpus))
    else:
        target_devices = [d for d in device_ids if d < total_gpus]

    if len(target_devices) > 1:
        logger.info("Scaling model across %d CUDA devices: %s", len(target_devices), target_devices)
        model = model.to(f"cuda:{target_devices[0]}")
        parallel_model = nn.DataParallel(model, device_ids=target_devices)
        return parallel_model, target_devices
    elif len(target_devices) == 1:
        model = model.to(f"cuda:{target_devices[0]}")
        return model, target_devices

    return model, []


def get_effective_batch_size(
    micro_batch_size: int,
    grad_accum_steps: int,
    num_devices: int = 1,
) -> int:
    """Calculates global effective batch size across distributed devices."""
    return micro_batch_size * max(1, grad_accum_steps) * max(1, num_devices)
