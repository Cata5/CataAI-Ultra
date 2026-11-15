"""
src/telemetry/hardware.py
Dedicated hardware monitoring service reading real NVIDIA GPU metrics via NVML and PyTorch.
Never fabricates telemetry; explicitly reports Unavailable when metrics cannot be queried.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

# Try to initialize NVML safely
_NVML_INITIALIZED = False
_NVML_HANDLE = None
_NVML_ERROR = None

try:
    import pynvml
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    if device_count > 0:
        _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
        _NVML_INITIALIZED = True
        logger.info("NVML initialized successfully for device 0: %s", pynvml.nvmlDeviceGetName(_NVML_HANDLE))
    else:
        _NVML_ERROR = "No NVIDIA devices detected by NVML."
except Exception as e:
    _NVML_INITIALIZED = False
    _NVML_ERROR = str(e)
    logger.warning("NVML initialization failed: %s", e)


@dataclass
class HardwareTelemetry:
    timestamp: float
    is_available: bool
    # Device identity
    device_name: str
    driver_version: Optional[str]
    cuda_available: bool
    # Device-wide memory (from NVML)
    total_vram_mb: Optional[float]
    used_vram_mb: Optional[float]
    free_vram_mb: Optional[float]
    vram_percent: Optional[float]
    # Utilization & thermals
    gpu_util_percent: Optional[int]
    memory_util_percent: Optional[int]
    temperature_c: Optional[int]
    power_draw_watts: Optional[float]
    power_limit_watts: Optional[float]
    sm_clock_mhz: Optional[int]
    memory_clock_mhz: Optional[int]
    # PyTorch allocator statistics (distinct from device-wide NVML)
    pytorch_allocated_mb: Optional[float]
    pytorch_reserved_mb: Optional[float]
    pytorch_max_allocated_mb: Optional[float]
    pytorch_max_reserved_mb: Optional[float]
    pytorch_mem_get_info_free_mb: Optional[float]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HardwareMonitor:
    """Queries NVML and PyTorch for genuine, real-time hardware metrics."""

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index

    def read_telemetry(self) -> HardwareTelemetry:
        now = time.time()
        cuda_avail = torch.cuda.is_available()

        # PyTorch stats
        pt_alloc_mb = None
        pt_res_mb = None
        pt_max_alloc_mb = None
        pt_max_res_mb = None
        pt_free_mb = None

        if cuda_avail:
            try:
                pt_alloc_mb = round(torch.cuda.memory_allocated(self.device_index) / (1024 * 1024), 2)
                pt_res_mb = round(torch.cuda.memory_reserved(self.device_index) / (1024 * 1024), 2)
                pt_max_alloc_mb = round(torch.cuda.max_memory_allocated(self.device_index) / (1024 * 1024), 2)
                pt_max_res_mb = round(torch.cuda.max_memory_reserved(self.device_index) / (1024 * 1024), 2)
                free_b, total_b = torch.cuda.mem_get_info(self.device_index)
                pt_free_mb = round(free_b / (1024 * 1024), 2)
            except Exception as e:
                logger.debug("Failed reading PyTorch CUDA stats: %s", e)

        if not _NVML_INITIALIZED or _NVML_HANDLE is None:
            dev_name = torch.cuda.get_device_name(self.device_index) if cuda_avail else "CPU Only"
            return HardwareTelemetry(
                timestamp=now,
                is_available=cuda_avail,
                device_name=dev_name,
                driver_version=None,
                cuda_available=cuda_avail,
                total_vram_mb=None,
                used_vram_mb=None,
                free_vram_mb=pt_free_mb,
                vram_percent=None,
                gpu_util_percent=None,
                memory_util_percent=None,
                temperature_c=None,
                power_draw_watts=None,
                power_limit_watts=None,
                sm_clock_mhz=None,
                memory_clock_mhz=None,
                pytorch_allocated_mb=pt_alloc_mb,
                pytorch_reserved_mb=pt_res_mb,
                pytorch_max_allocated_mb=pt_max_alloc_mb,
                pytorch_max_reserved_mb=pt_max_res_mb,
                pytorch_mem_get_info_free_mb=pt_free_mb,
                error=_NVML_ERROR or "NVML unavailable; PyTorch fallback active",
            )

        try:
            h = _NVML_HANDLE
            name = pynvml.nvmlDeviceGetName(h)
            try:
                driver = pynvml.nvmlSystemGetDriverVersion()
            except Exception:
                driver = None

            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            total_mb = round(mem.total / (1024 * 1024), 1)
            used_mb = round(mem.used / (1024 * 1024), 1)
            free_mb = round(mem.free / (1024 * 1024), 1)
            pct = round((mem.used / mem.total) * 100.0, 1)

            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                gpu_util = util.gpu
                mem_util = util.memory
            except Exception:
                gpu_util = None
                mem_util = None

            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None

            try:
                power_w = round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 1)
            except Exception:
                power_w = None

            try:
                limit_w = round(pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0, 1)
            except Exception:
                limit_w = None

            try:
                sm_clk = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
            except Exception:
                sm_clk = None

            try:
                mem_clk = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
            except Exception:
                mem_clk = None

            return HardwareTelemetry(
                timestamp=now,
                is_available=True,
                device_name=name,
                driver_version=driver,
                cuda_available=cuda_avail,
                total_vram_mb=total_mb,
                used_vram_mb=used_mb,
                free_vram_mb=free_mb,
                vram_percent=pct,
                gpu_util_percent=gpu_util,
                memory_util_percent=mem_util,
                temperature_c=temp,
                power_draw_watts=power_w,
                power_limit_watts=limit_w,
                sm_clock_mhz=sm_clk,
                memory_clock_mhz=mem_clk,
                pytorch_allocated_mb=pt_alloc_mb,
                pytorch_reserved_mb=pt_res_mb,
                pytorch_max_allocated_mb=pt_max_alloc_mb,
                pytorch_max_reserved_mb=pt_max_res_mb,
                pytorch_mem_get_info_free_mb=pt_free_mb,
            )
        except Exception as e:
            logger.warning("Error reading NVML hardware metrics: %s", e)
            return HardwareTelemetry(
                timestamp=now,
                is_available=False,
                device_name="Unknown GPU",
                driver_version=None,
                cuda_available=cuda_avail,
                total_vram_mb=None,
                used_vram_mb=None,
                free_vram_mb=pt_free_mb,
                vram_percent=None,
                gpu_util_percent=None,
                memory_util_percent=None,
                temperature_c=None,
                power_draw_watts=None,
                power_limit_watts=None,
                sm_clock_mhz=None,
                memory_clock_mhz=None,
                pytorch_allocated_mb=pt_alloc_mb,
                pytorch_reserved_mb=pt_res_mb,
                pytorch_max_allocated_mb=pt_max_alloc_mb,
                pytorch_max_reserved_mb=pt_max_res_mb,
                pytorch_mem_get_info_free_mb=pt_free_mb,
                error=str(e),
            )

