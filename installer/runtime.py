"""Create the Python environment and install the selected PyTorch wheel."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .hardware import HardwareProfile
from .plan import TorchPlan

Logger = Callable[[str], None]

_PIP_TIMEOUT_SECONDS = 7200


def _print(message: str) -> None:
    print(message, flush=True)


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run(
    command: Sequence[str],
    *,
    log: Logger = _print,
    dry_run: bool = False,
    capture: bool = False,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess[str]:
    log("$ " + shlex.join(str(part) for part in command))
    if dry_run:
        return subprocess.CompletedProcess(list(command), 0, "", "")

    result = subprocess.run(
        [str(part) for part in command],
        text=True,
        capture_output=capture,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if len(detail) > 1200:
            detail = detail[-1200:]
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: "
            f"{shlex.join(str(part) for part in command)}\n{detail}"
        )
    return result


def ensure_venv(base_python: str, venv_dir: Path, *, log: Logger = _print, dry_run: bool = False) -> Path:
    """Create ``venv_dir`` if needed and return its interpreter."""

    python = venv_python(venv_dir)
    if python.is_file():
        log(f"Reusing virtual environment: {venv_dir}")
        return python

    log(f"Creating virtual environment: {venv_dir}")
    run([base_python, "-m", "venv", str(venv_dir)], log=log, dry_run=dry_run, timeout=300)
    if not dry_run and not python.is_file():
        raise RuntimeError(f"Python venv did not create {python}")
    return python


def _torch_probe_script() -> str:
    return """
import json
try:
    import torch
    try:
        import torchvision
        torchvision_version = torchvision.__version__
    except Exception:
        torchvision_version = None
    devices = []
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append({
                "index": index,
                "name": props.name,
                "capability": f"{props.major}.{props.minor}",
                "vram_gb": round(props.total_memory / (1024 ** 3), 1),
            })
    payload = {
        "available": True,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision_version,
        "torchvision_available": torchvision_version is not None,
        "cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "devices": devices,
    }
except Exception as exc:
    payload = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
print("LLMMAKER_TORCH=" + json.dumps(payload))
"""


def verify_torch(python: Path, *, log: Logger = _print) -> dict[str, Any]:
    """Import torch in the target venv and report CUDA state."""

    try:
        result = run(
            [str(python), "-c", _torch_probe_script()],
            log=lambda _message: None,
            capture=True,
            timeout=120,
        )
    except (OSError, RuntimeError) as exc:
        log(f"PyTorch probe failed: {type(exc).__name__}: {exc}")
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    for line in result.stdout.splitlines():
        if line.startswith("LLMMAKER_TORCH="):
            payload = json.loads(line.split("=", 1)[1])
            log("PyTorch: " + _describe_torch(payload))
            return payload
    raise RuntimeError(f"Could not read PyTorch verification output: {result.stdout[-400:]}")


def _describe_torch(payload: dict[str, Any]) -> str:
    if not payload.get("available"):
        return "not installed (" + str(payload.get("error", "unknown error")) + ")"
    cuda = "CUDA available" if payload.get("cuda_available") else "CPU only"
    return f"{payload.get('torch_version')} ({payload.get('cuda_version') or 'no CUDA'}, {cuda})"


def _channel_cuda_version(channel: str) -> str:
    """Map ``cu128`` to the ``12.8`` string reported by ``torch.version.cuda``."""

    digits = channel[2:]
    if len(digits) < 2:
        return ""
    return f"{digits[:-1]}.{digits[-1]}"


def _torch_pip_args(plan: TorchPlan, *, force: bool) -> list[str]:
    args = [
        "install",
        "--only-binary=:all:",
        "--progress-bar",
        "off",
        "--timeout",
        "60",
        "--retries",
        "5",
    ]
    if force:
        args.extend(["--force-reinstall", "--no-deps"])
    if plan.index_url:
        args.extend(["--index-url", plan.index_url])
    args.extend(["torch", "torchvision"])
    return args


def install_torch(
    python: Path,
    plan: TorchPlan,
    *,
    log: Logger = _print,
    dry_run: bool = False,
    upgrade_pip: bool = True,
) -> dict[str, Any]:
    """Install (or repair) torch/torchvision for ``plan`` and verify CUDA."""

    if plan.blocked:
        raise RuntimeError(plan.warning or "The detected GPU has no compatible PyTorch wheel.")

    current = verify_torch(python, log=log) if not dry_run else {"available": False}
    needs_install = not current.get("available") or not current.get("torchvision_available")
    force = bool(current.get("available") and not current.get("torchvision_available"))
    if current.get("available") and not current.get("torchvision_available"):
        log("torchvision is missing or broken; reinstalling the matched torch pair.")
    if plan.is_cuda and current.get("available"):
        expected_cuda = _channel_cuda_version(plan.channel)
        installed_cuda = str(current.get("cuda_version") or "")
        if not current.get("cuda_available"):
            log("Installed PyTorch has no CUDA support; replacing it with the selected wheel.")
            needs_install = True
            force = True
        elif (
            expected_cuda
            and installed_cuda
            and installed_cuda.split(".")[:2] != expected_cuda.split(".")
        ):
            log(
                f"Installed PyTorch uses CUDA {installed_cuda}, not {expected_cuda}; "
                "replacing it with the selected wheel."
            )
            needs_install = True
            force = True

    if not needs_install:
        log("PyTorch already satisfies the selected runtime.")
        return current

    if upgrade_pip:
        run([str(python), "-m", "pip", "install", "--upgrade", "pip"], log=log, dry_run=dry_run, timeout=600)

    run(
        [str(python), "-m", "pip", *_torch_pip_args(plan, force=force)],
        log=log,
        dry_run=dry_run,
        timeout=_PIP_TIMEOUT_SECONDS,
    )
    if dry_run:
        return {"available": False, "dry_run": True}

    verified = verify_torch(python, log=log)
    if plan.is_cuda and not verified.get("cuda_available"):
        raise RuntimeError(
            "PyTorch installed but CUDA is not available. Check that the NVIDIA "
            f"driver is {plan.minimum_driver or 'new enough'} and that the GPU is visible "
            "to nvidia-smi. Re-run setup, or use --cpu for a CPU-only install."
        )
    return verified


def install_requirements(
    python: Path,
    requirements: Path,
    *,
    log: Logger = _print,
    dry_run: bool = False,
) -> None:
    if not requirements.is_file():
        raise FileNotFoundError(f"Requirements file not found: {requirements}")
    run(
        [str(python), "-m", "pip", "install", "-r", str(requirements)],
        log=log,
        dry_run=dry_run,
        timeout=_PIP_TIMEOUT_SECONDS,
    )


def write_manifest(
    path: Path,
    *,
    venv_dir: Path,
    plan: TorchPlan,
    hardware: HardwareProfile,
    verification: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "venv": str(venv_dir),
        "python": str(venv_python(venv_dir)),
        "plan": plan.to_dict(),
        "hardware": hardware.to_dict(),
        "torch": verification,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return payload


def read_manifest(path: Path) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("schema") != 1 or not data.get("python"):
        return None
    return data


def check_disk(profile: HardwareProfile, plan: TorchPlan, *, log: Logger = _print) -> None:
    if plan.is_cuda and profile.free_disk_gb and profile.free_disk_gb < 8:
        raise RuntimeError(
            f"Only {profile.free_disk_gb:.1f} GB free. Free at least 8 GB before "
            "installing the CUDA build, or pass --cpu."
        )
    log(f"Free disk: {profile.free_disk_gb:.1f} GB")


def install_all(
    *,
    base_python: str,
    venv_dir: Path,
    requirements: Path,
    manifest: Path,
    plan: TorchPlan,
    hardware: HardwareProfile,
    log: Logger = _print,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Full first-run install: venv, torch, requirements, manifest."""

    check_disk(hardware, plan, log=log)
    python = ensure_venv(base_python, venv_dir, log=log, dry_run=dry_run)
    verification = install_torch(python, plan, log=log, dry_run=dry_run)
    install_requirements(python, requirements, log=log, dry_run=dry_run)
    if dry_run:
        return {"dry_run": True, "plan": plan.to_dict()}
    payload = write_manifest(
        manifest,
        venv_dir=venv_dir,
        plan=plan,
        hardware=hardware,
        verification=verification,
    )
    log(f"Runtime manifest: {manifest}")
    return payload