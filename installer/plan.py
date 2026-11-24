"""Choose a PyTorch wheel channel from the detected hardware."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional

from .hardware import HardwareProfile

# Minimum driver for each CUDA runtime, per NVIDIA's compatibility tables.
# A newer driver always runs an older CUDA runtime, so these are floors.
CUDA_DRIVERS: dict[str, dict[str, Optional[tuple[int, int]]]] = {
    "cu128": {"windows": (570, 65), "linux": (570, 26), "darwin": None},
    "cu126": {"windows": (560, 94), "linux": (560, 28), "darwin": None},
    "cu124": {"windows": (551, 61), "linux": (550, 54), "darwin": None},
    "cu121": {"windows": (531, 14), "linux": (530, 30), "darwin": None},
    "cu118": {"windows": (452, 39), "linux": (450, 36), "darwin": None},
}

CHANNEL_LABELS = {
    "cu128": "CUDA 12.8",
    "cu126": "CUDA 12.6",
    "cu124": "CUDA 12.4",
    "cu121": "CUDA 12.1",
    "cu118": "CUDA 11.8",
    "cpu": "CPU",
}

_CHANNEL_ORDER = ["cu128", "cu126", "cu124", "cu121", "cu118"]
_CHANNEL_PATTERN = re.compile(r"^cu\d{3}$")


@dataclass(frozen=True)
class TorchPlan:
    """A concrete pip channel and the reason it was selected."""

    channel: str
    index_url: Optional[str]
    label: str
    reason: str
    warning: Optional[str]
    blocked: bool
    gpu_name: Optional[str]
    driver_version: Optional[str]
    minimum_driver: Optional[str]
    is_cuda: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _driver_tuple(value: Optional[str]) -> Optional[tuple[int, int]]:
    if not value:
        return None
    match = re.search(r"(\d+)\.(\d+)", value)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)))


def _format_driver(value: tuple[int, int]) -> str:
    return f"{value[0]}.{value[1]}"


def _requirement(channel: str, os_name: str) -> Optional[tuple[int, int]]:
    requirements = CUDA_DRIVERS.get(channel, {})
    return requirements.get(os_name)


def _cpu_index(os_name: str) -> Optional[str]:
    if os_name == "darwin":
        return None  # PyPI wheel supports MPS on Apple silicon.
    return "https://download.pytorch.org/whl/cpu"


def _cpu_plan(
    os_name: str,
    reason: str,
    warning: Optional[str] = None,
    blocked: bool = False,
    gpu_name: Optional[str] = None,
    driver_version: Optional[str] = None,
) -> TorchPlan:
    return TorchPlan(
        channel="cpu",
        index_url=_cpu_index(os_name),
        label="CPU",
        reason=reason,
        warning=warning,
        blocked=blocked,
        gpu_name=gpu_name,
        driver_version=driver_version,
        minimum_driver=None,
        is_cuda=False,
    )


def _disk_warning(profile: HardwareProfile) -> Optional[str]:
    if profile.free_disk_gb and profile.free_disk_gb < 12:
        return (
            f"Only {profile.free_disk_gb:.1f} GB free. A CUDA install needs "
            "roughly 8 GB for wheels, cache and the virtual environment."
        )
    return None


def _merge_warnings(*values: Optional[str]) -> Optional[str]:
    messages = [value for value in values if value]
    return " ".join(messages) if messages else None


def choose_plan(
    profile: HardwareProfile,
    *,
    cpu_only: bool = False,
    cuda_tag: Optional[str] = None,
) -> TorchPlan:
    """Select the newest PyTorch CUDA runtime the driver can run.

    A Blackwell GPU (compute capability 10.0+) requires CUDA 12.8 or newer.
    When the driver is too old the plan is blocked rather than silently
    downloading a CPU wheel; callers can pass ``cpu_only=True`` to override.
    """

    os_name = profile.os_name.lower()
    disk_warning = _disk_warning(profile)

    if cpu_only:
        return _cpu_plan(
            os_name,
            "CPU build requested explicitly.",
            warning=disk_warning,
        )

    if cuda_tag:
        tag = cuda_tag.strip().lower()
        if not _CHANNEL_PATTERN.fullmatch(tag):
            raise ValueError(f"Invalid CUDA tag {cuda_tag!r}; expected a value like cu128.")
        requirement = _requirement(tag, os_name)
        driver = _driver_tuple(profile.primary_gpu.driver_version) if profile.primary_gpu else None
        too_old = bool(requirement and driver and driver < requirement)
        return TorchPlan(
            channel=tag,
            index_url=f"https://download.pytorch.org/whl/{tag}",
            label=CHANNEL_LABELS.get(tag, tag.upper()),
            reason=f"{tag} selected explicitly.",
            warning=_merge_warnings(
                disk_warning,
                (
                    f"Driver {profile.primary_gpu.driver_version} is below the "
                    f"{_format_driver(requirement)} minimum for {tag}; installation may fail."
                )
                if too_old and requirement
                else None,
            ),
            blocked=False,
            gpu_name=profile.primary_gpu.name if profile.primary_gpu else None,
            driver_version=profile.primary_gpu.driver_version if profile.primary_gpu else None,
            minimum_driver=_format_driver(requirement) if requirement else None,
            is_cuda=True,
        )

    if os_name == "darwin":
        return _cpu_plan(
            os_name,
            "macOS uses the default PyTorch wheel with Metal (MPS) support.",
            warning=disk_warning,
        )

    if not profile.gpus:
        detail = "No NVIDIA GPU detected."
        if profile.nvidia_error:
            detail += f" nvidia-smi: {profile.nvidia_error}"
        return _cpu_plan(
            os_name,
            detail + " CPU build selected; training still runs, slower.",
            warning=disk_warning,
        )

    gpu = profile.primary_gpu
    driver = _driver_tuple(gpu.driver_version) if gpu else None
    if not gpu or not driver:
        return _cpu_plan(
            os_name,
            "NVIDIA device found but its driver version could not be read.",
            warning=_merge_warnings(disk_warning, "Update the NVIDIA driver, then re-run setup."),
            blocked=True,
            gpu_name=gpu.name if gpu else None,
            driver_version=gpu.driver_version if gpu else None,
        )

    # Blackwell (RTX 50 / B100 and newer) needs the CUDA 12.8+ wheel.
    if gpu.compute_capability_tuple[0] >= 10:
        requirement = _requirement("cu128", os_name)
        if requirement and driver < requirement:
            return _cpu_plan(
                os_name,
                f"{gpu.name} needs CUDA 12.8, but driver {gpu.driver_version} is too old.",
                warning=(
                    f"Update the NVIDIA driver to {_format_driver(requirement)} or newer. "
                    "Setup will not download a CPU wheel automatically; pass --cpu to "
                    "install a CPU build deliberately."
                ),
                blocked=True,
                gpu_name=gpu.name,
                driver_version=gpu.driver_version,
            )
        channel = "cu128"
        requirement = _requirement(channel, os_name)
    else:
        channel = ""
        requirement = None
        for candidate in _CHANNEL_ORDER:
            candidate_requirement = _requirement(candidate, os_name)
            if candidate_requirement and driver >= candidate_requirement:
                channel = candidate
                requirement = candidate_requirement
                break

        if not channel:
            highest = _requirement("cu118", os_name)
            floor = _format_driver(highest) if highest else "a supported NVIDIA driver"
            return _cpu_plan(
                os_name,
                f"Driver {gpu.driver_version} is older than the CUDA 11.8 floor ({floor}).",
                warning=(
                    f"Update the NVIDIA driver to {floor} or newer, or pass --cpu to "
                    "install a CPU build deliberately."
                ),
                blocked=True,
                gpu_name=gpu.name,
                driver_version=gpu.driver_version,
            )

    return TorchPlan(
        channel=channel,
        index_url=f"https://download.pytorch.org/whl/{channel}",
        label=CHANNEL_LABELS.get(channel, channel.upper()),
        reason=(
            f"{gpu.name} with driver {gpu.driver_version} runs "
            f"{CHANNEL_LABELS.get(channel, channel)}."
        ),
        warning=disk_warning,
        blocked=False,
        gpu_name=gpu.name,
        driver_version=gpu.driver_version,
        minimum_driver=_format_driver(requirement) if requirement else None,
        is_cuda=True,
    )


def format_plan(plan: TorchPlan) -> str:
    lines = [
        f"PyTorch channel: {plan.label} ({plan.channel})",
        f"Index: {plan.index_url or 'default PyPI'}",
        f"Reason: {plan.reason}",
    ]
    if plan.minimum_driver:
        lines.append(f"Minimum NVIDIA driver: {plan.minimum_driver}")
    if plan.warning:
        lines.append(f"Warning: {plan.warning}")
    if plan.blocked:
        lines.append("Status: blocked until the driver is updated (or --cpu is passed).")
    return "\n".join(lines)