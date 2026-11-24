"""Hardware detection used before any Python packages are installed."""

from __future__ import annotations

import csv
import ctypes
import io
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
_SMI_TIMEOUT_SECONDS = 8


@dataclass(frozen=True)
class Gpu:
    """One NVIDIA device reported by nvidia-smi."""

    index: int
    name: str
    driver_version: str
    memory_mb: Optional[int] = None
    compute_capability: Optional[str] = None

    @property
    def compute_capability_tuple(self) -> tuple[int, int]:
        return _parse_capability(self.compute_capability)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["compute_capability_tuple"] = list(self.compute_capability_tuple)
        return data


@dataclass(frozen=True)
class HardwareProfile:
    """Everything the installer needs to choose a runtime."""

    os_name: str
    architecture: str
    python_version: str
    python_executable: str
    gpus: tuple[Gpu, ...]
    nvidia_smi: Optional[str]
    nvidia_error: Optional[str]
    free_disk_gb: float
    total_ram_gb: Optional[float]

    @property
    def primary_gpu(self) -> Optional[Gpu]:
        if not self.gpus:
            return None
        # Prefer the newest compute capability, then the first device.
        return max(self.gpus, key=lambda gpu: (gpu.compute_capability_tuple, -gpu.index))

    @property
    def has_nvidia(self) -> bool:
        return bool(self.gpus)

    def to_dict(self) -> dict[str, Any]:
        return {
            "os_name": self.os_name,
            "architecture": self.architecture,
            "python_version": self.python_version,
            "python_executable": self.python_executable,
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "nvidia_smi": self.nvidia_smi,
            "nvidia_error": self.nvidia_error,
            "free_disk_gb": self.free_disk_gb,
            "total_ram_gb": self.total_ram_gb,
        }


def _parse_capability(value: Optional[str]) -> tuple[int, int]:
    if not value:
        return (0, 0)
    try:
        major, minor = value.replace("sm_", "").split(".")[:2]
        return (int(major), int(minor))
    except (TypeError, ValueError):
        return (0, 0)


def find_nvidia_smi() -> Optional[str]:
    """Return an nvidia-smi path even when the tool is not on PATH."""

    candidates: list[Path] = []
    if os.name == "nt":
        system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        candidates.extend(
            [
                system_root / "System32" / "nvidia-smi.exe",
                program_files / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
            ]
        )
    else:
        candidates.extend(
            [Path("/usr/bin/nvidia-smi"), Path("/usr/local/bin/nvidia-smi")]
        )

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    return shutil.which("nvidia-smi")


def _query_smi(smi: str, fields: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [smi, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=_SMI_TIMEOUT_SECONDS,
        check=False,
        creationflags=_CREATE_NO_WINDOW,
    )


def _parse_memory(value: str) -> Optional[int]:
    token = value.strip().split()[0] if value.strip() else ""
    try:
        return int(float(token))
    except ValueError:
        return None


def parse_gpu_lines(text: str, has_compute_capability: bool) -> list[Gpu]:
    """Parse nvidia-smi CSV output. Public for tests and diagnostics."""

    gpus: list[Gpu] = []
    for row in csv.reader(io.StringIO(text)):
        parts = [part.strip() for part in row]
        if not parts or not any(parts):
            continue
        if len(parts) < 4:
            continue

        try:
            index = int(parts[0])
        except ValueError:
            continue

        name = parts[1]
        driver = parts[2]
        memory = _parse_memory(parts[3])
        capability = None
        if has_compute_capability and len(parts) >= 5:
            candidate = parts[4].strip()
            if candidate and candidate.upper() not in {"N/A", "[N/A]", "UNKNOWN"}:
                capability = candidate

        gpus.append(
            Gpu(
                index=index,
                name=name,
                driver_version=driver,
                memory_mb=memory,
                compute_capability=capability,
            )
        )
    return gpus


def detect_nvidia() -> tuple[list[Gpu], Optional[str], Optional[str]]:
    """Return (GPUs, nvidia-smi path, error)."""

    smi = find_nvidia_smi()
    if not smi:
        return [], None, None

    fields_with_cap = ["index", "name", "driver_version", "memory.total", "compute_cap"]
    try:
        result = _query_smi(smi, fields_with_cap)
    except (OSError, subprocess.SubprocessError) as exc:
        return [], smi, f"nvidia-smi failed: {exc}"

    if result.returncode == 0 and result.stdout.strip():
        gpus = parse_gpu_lines(result.stdout, has_compute_capability=True)
        return gpus, smi, None

    # Older drivers do not understand compute_cap. Retry without it.
    try:
        result = _query_smi(smi, ["index", "name", "driver_version", "memory.total"])
    except (OSError, subprocess.SubprocessError) as exc:
        return [], smi, f"nvidia-smi failed: {exc}"

    if result.returncode == 0 and result.stdout.strip():
        return parse_gpu_lines(result.stdout, has_compute_capability=False), smi, None

    error = (result.stderr or result.stdout or "nvidia-smi returned no devices").strip()
    return [], smi, error[:500]


def _total_ram_gb() -> Optional[float]:
    try:
        if os.name == "nt":

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return round(status.total_physical / (1024**3), 1)

        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return round(page_size * pages / (1024**3), 1)
    except (AttributeError, OSError, ValueError):
        return None


def detect_hardware(disk_path: Optional[Path] = None) -> HardwareProfile:
    """Collect OS, Python, disk and NVIDIA data without importing torch."""

    target = Path(disk_path) if disk_path else Path.home()
    try:
        free_disk_gb = round(shutil.disk_usage(str(target)).free / (1024**3), 1)
    except OSError:
        free_disk_gb = 0.0

    gpus, smi, error = detect_nvidia()
    architecture = platform.machine().lower()
    if architecture in {"amd64", "x86_64", "x64"}:
        architecture = "x86_64"
    elif architecture in {"arm64", "aarch64"}:
        architecture = "arm64"

    return HardwareProfile(
        os_name=platform.system().lower(),
        architecture=architecture,
        python_version=platform.python_version(),
        python_executable=sys.executable,
        gpus=tuple(gpus),
        nvidia_smi=smi,
        nvidia_error=error,
        free_disk_gb=free_disk_gb,
        total_ram_gb=_total_ram_gb(),
    )