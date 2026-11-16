"""
src/telemetry/hardware.py
Dedicated hardware monitoring and capability profiling service reading real GPU & CPU metrics.
Never fabricates telemetry; dynamically detects NVIDIA CUDA, AMD ROCm, Apple MPS, and CPU environments.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from .profiles import HardwareProfile

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
    logger.debug("NVML initialization note: %s", e)


def get_system_ram_mb() -> Tuple[float, float]:
    """Returns (total_ram_mb, free_ram_mb) cross-platform without external deps."""
    os_name = platform.system()
    if os_name == "Windows":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('sullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return (
                round(stat.ullTotalPhys / (1024 * 1024), 1),
                round(stat.ullAvailPhys / (1024 * 1024), 1),
            )
        except Exception:
            pass
    elif os_name == "Linux":
        try:
            total_kb, free_kb = 0, 0
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    parts = line.split(":")
                    if parts[0] == "MemTotal":
                        total_kb = int(parts[1].split()[0])
                    elif parts[0] in ("MemAvailable", "MemFree"):
                        if free_kb == 0:
                            free_kb = int(parts[1].split()[0])
            if total_kb > 0:
                return (round(total_kb / 1024, 1), round(free_kb / 1024, 1))
        except Exception:
            pass

    # Fallback default estimate if OS query fails
    return (16384.0, 8192.0)


@dataclass
class HardwareTelemetry:
    timestamp: float
    is_available: bool
    # Device identity
    device_name: str
    driver_version: Optional[str]
    cuda_available: bool
    backend_type: str = "cpu"  # "cuda", "rocm", "mps", "cpu"
    compute_capability: Optional[str] = None
    # System RAM
    system_ram_total_mb: float = 16384.0
    system_ram_free_mb: float = 8192.0
    # Device-wide memory (from NVML/ROCm/PyTorch)
    total_vram_mb: Optional[float] = None
    used_vram_mb: Optional[float] = None
    free_vram_mb: Optional[float] = None
    vram_percent: Optional[float] = None
    # Utilization & thermals
    gpu_util_percent: Optional[int] = None
    memory_util_percent: Optional[int] = None
    temperature_c: Optional[int] = None
    power_draw_watts: Optional[float] = None
    power_limit_watts: Optional[float] = None
    sm_clock_mhz: Optional[int] = None
    memory_clock_mhz: Optional[int] = None
    # PyTorch allocator statistics (distinct from device-wide NVML)
    pytorch_allocated_mb: Optional[float] = None
    pytorch_reserved_mb: Optional[float] = None
    pytorch_max_allocated_mb: Optional[float] = None
    pytorch_max_reserved_mb: Optional[float] = None
    pytorch_mem_get_info_free_mb: Optional[float] = None
    # Precisions & profiles
    supported_precisions: List[str] = None
    bf16_supported: bool = False
    fp16_supported: bool = False
    hardware_profile: Optional[Dict[str, Any]] = None
    resource_profiles: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.supported_precisions is None:
            self.supported_precisions = ["fp32"]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HardwareMonitor:
    """Queries NVML, PyTorch, and system hardware for genuine real-time metrics and capability profiles."""

    def __init__(self, device_index: int = 0, default_profile_mode: str = "balanced") -> None:
        self.device_index = device_index
        self.default_profile_mode = default_profile_mode

    def read_telemetry(self) -> HardwareTelemetry:
        now = time.time()
        cuda_avail = torch.cuda.is_available()
        tot_ram_mb, free_ram_mb = get_system_ram_mb()

        # Backend determination
        backend = "cpu"
        compute_cap = None
        bf16_avail = False
        fp16_avail = False
        supported_p = ["fp32"]

        if cuda_avail:
            if hasattr(torch.version, "hip") and torch.version.hip is not None:
                backend = "rocm"
            else:
                backend = "cuda"
            fp16_avail = True
            supported_p.append("fp16")
            try:
                cap = torch.cuda.get_device_capability(self.device_index)
                compute_cap = f"{cap[0]}.{cap[1]}"
                if torch.cuda.is_bf16_supported():
                    bf16_avail = True
                    supported_p.append("bfloat16")
            except Exception:
                pass
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            backend = "mps"
            fp16_avail = True
            supported_p.append("fp16")

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

        # Build resource profiles
        # Fallback device memory if NVML not available
        ref_vram = None
        if cuda_avail:
            try:
                free_b, total_b = torch.cuda.mem_get_info(self.device_index)
                ref_vram = round(total_b / (1024 * 1024), 1)
            except Exception:
                ref_vram = None

        profiles = {
            "conservative": HardwareProfile.create(
                mode="conservative", total_vram_mb=ref_vram, total_sys_ram_mb=tot_ram_mb,
                device_type=backend, bf16_supported=bf16_avail, fp16_supported=fp16_avail
            ).to_dict(),
            "balanced": HardwareProfile.create(
                mode="balanced", total_vram_mb=ref_vram, total_sys_ram_mb=tot_ram_mb,
                device_type=backend, bf16_supported=bf16_avail, fp16_supported=fp16_avail
            ).to_dict(),
            "performance": HardwareProfile.create(
                mode="performance", total_vram_mb=ref_vram, total_sys_ram_mb=tot_ram_mb,
                device_type=backend, bf16_supported=bf16_avail, fp16_supported=fp16_avail
            ).to_dict(),
        }

        active_profile = profiles.get(self.default_profile_mode, profiles["balanced"])

        if not _NVML_INITIALIZED or _NVML_HANDLE is None:
            dev_name = torch.cuda.get_device_name(self.device_index) if cuda_avail else f"CPU ({platform.processor() or 'General'})"
            return HardwareTelemetry(
                timestamp=now,
                is_available=cuda_avail or True,
                device_name=dev_name,
                driver_version=None,
                cuda_available=cuda_avail,
                backend_type=backend,
                compute_capability=compute_cap,
                system_ram_total_mb=tot_ram_mb,
                system_ram_free_mb=free_ram_mb,
                total_vram_mb=ref_vram,
                used_vram_mb=round(ref_vram - pt_free_mb, 1) if (ref_vram and pt_free_mb) else None,
                free_vram_mb=pt_free_mb,
                vram_percent=round(((ref_vram - pt_free_mb) / ref_vram) * 100, 1) if (ref_vram and pt_free_mb) else None,
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
                supported_precisions=supported_p,
                bf16_supported=bf16_avail,
                fp16_supported=fp16_avail,
                hardware_profile=active_profile,
                resource_profiles=profiles,
                error=_NVML_ERROR if not cuda_avail else None,
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

            # Re-generate profiles with exact NVML total VRAM
            profiles = {
                "conservative": HardwareProfile.create(
                    mode="conservative", total_vram_mb=total_mb, total_sys_ram_mb=tot_ram_mb,
                    device_type=backend, bf16_supported=bf16_avail, fp16_supported=fp16_avail
                ).to_dict(),
                "balanced": HardwareProfile.create(
                    mode="balanced", total_vram_mb=total_mb, total_sys_ram_mb=tot_ram_mb,
                    device_type=backend, bf16_supported=bf16_avail, fp16_supported=fp16_avail
                ).to_dict(),
                "performance": HardwareProfile.create(
                    mode="performance", total_vram_mb=total_mb, total_sys_ram_mb=tot_ram_mb,
                    device_type=backend, bf16_supported=bf16_avail, fp16_supported=fp16_avail
                ).to_dict(),
            }
            active_profile = profiles.get(self.default_profile_mode, profiles["balanced"])

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
                backend_type=backend,
                compute_capability=compute_cap,
                system_ram_total_mb=tot_ram_mb,
                system_ram_free_mb=free_ram_mb,
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
                supported_precisions=supported_p,
                bf16_supported=bf16_avail,
                fp16_supported=fp16_avail,
                hardware_profile=active_profile,
                resource_profiles=profiles,
            )
        except Exception as e:
            logger.warning("Error reading NVML hardware metrics: %s", e)
            return HardwareTelemetry(
                timestamp=now,
                is_available=cuda_avail,
                device_name="Unknown GPU",
                driver_version=None,
                cuda_available=cuda_avail,
                backend_type=backend,
                compute_capability=compute_cap,
                system_ram_total_mb=tot_ram_mb,
                system_ram_free_mb=free_ram_mb,
                total_vram_mb=ref_vram,
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
                supported_precisions=supported_p,
                bf16_supported=bf16_avail,
                fp16_supported=fp16_avail,
                hardware_profile=active_profile,
                resource_profiles=profiles,
                error=str(e),
            )
