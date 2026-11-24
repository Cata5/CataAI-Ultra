"""Command line entry point: python -m installer detect|plan|torch|install."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

from .hardware import HardwareProfile, detect_hardware
from .plan import TorchPlan, choose_plan, format_plan
from .runtime import (
    check_disk,
    ensure_venv,
    install_all,
    install_torch,
    read_manifest,
    venv_python,
    verify_torch,
    write_manifest,
)


def _hardware_text(profile: HardwareProfile) -> str:
    lines = [
        "Hardware",
        f"  OS: {profile.os_name} {profile.architecture}",
        f"  Python: {profile.python_version} ({profile.python_executable})",
        f"  Free disk: {profile.free_disk_gb:.1f} GB",
    ]
    if profile.total_ram_gb:
        lines.append(f"  RAM: {profile.total_ram_gb:.1f} GB")
    if profile.gpus:
        for gpu in profile.gpus:
            memory = f", {gpu.memory_mb} MB" if gpu.memory_mb else ""
            capability = f", cc {gpu.compute_capability}" if gpu.compute_capability else ""
            lines.append(
                f"  GPU {gpu.index}: {gpu.name} (driver {gpu.driver_version}{memory}{capability})"
            )
    else:
        lines.append("  GPU: no NVIDIA device detected")
    if profile.nvidia_error:
        lines.append(f"  nvidia-smi: {profile.nvidia_error}")
    return "\n".join(lines)


def _resolve_plan(args: argparse.Namespace, profile: HardwareProfile) -> TorchPlan:
    return choose_plan(profile, cpu_only=args.cpu, cuda_tag=args.cuda)


def _add_plan_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cpu", action="store_true", help="Install the CPU build even when CUDA is available.")
    group.add_argument("--cuda", metavar="TAG", help="Force a CUDA channel such as cu128 or cu126.")


def _cmd_detect(args: argparse.Namespace) -> int:
    profile = detect_hardware()
    if args.json:
        print(json.dumps(profile.to_dict(), indent=2))
    else:
        print(_hardware_text(profile))
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    profile = detect_hardware()
    plan = _resolve_plan(args, profile)
    if args.json:
        print(json.dumps({"hardware": profile.to_dict(), "plan": plan.to_dict()}, indent=2))
    else:
        print(_hardware_text(profile))
        print()
        print(format_plan(plan))
    return 2 if plan.blocked else 0


def _cmd_torch(args: argparse.Namespace) -> int:
    profile = detect_hardware()
    plan = _resolve_plan(args, profile)
    if plan.blocked:
        print(format_plan(plan), file=sys.stderr)
        return 2

    check_disk(profile, plan)
    venv_dir = Path(args.venv).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve() if args.manifest else venv_dir / "runtime.json"
    python = ensure_venv(args.python or sys.executable, venv_dir, dry_run=args.dry_run)
    verification = install_torch(
        python,
        plan,
        dry_run=args.dry_run,
        upgrade_pip=not args.no_pip_upgrade,
    )
    if not args.dry_run:
        payload = write_manifest(
            manifest,
            venv_dir=venv_dir,
            plan=plan,
            hardware=profile,
            verification=verification,
        )
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"Runtime manifest: {manifest}")
    return 0


def _cmd_install(args: argparse.Namespace) -> int:
    profile = detect_hardware()
    plan = _resolve_plan(args, profile)
    if plan.blocked:
        print(format_plan(plan), file=sys.stderr)
        return 2

    venv_dir = Path(args.venv).expanduser().resolve()
    requirements = Path(args.requirements).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve() if args.manifest else venv_dir / "runtime.json"
    payload = install_all(
        base_python=args.python or sys.executable,
        venv_dir=venv_dir,
        requirements=requirements,
        manifest=manifest,
        plan=plan,
        hardware=profile,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    venv_dir = Path(args.venv).expanduser().resolve()
    python = venv_python(venv_dir)
    if not python.is_file():
        print(f"Virtual environment not found: {venv_dir}", file=sys.stderr)
        return 1
    verification = verify_torch(python)
    if args.json:
        print(json.dumps(verification, indent=2))
    return 0 if verification.get("available") else 1


def _cmd_status(args: argparse.Namespace) -> int:
    path = Path(args.manifest).expanduser().resolve()
    data = read_manifest(path)
    if not data:
        print(f"No runtime manifest at {path}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        torch = data.get("torch", {})
        plan = data.get("plan", {})
        print(f"Runtime: {data.get('venv')}")
        print(f"Channel: {plan.get('label')} ({plan.get('channel')})")
        print(f"PyTorch: {torch.get('torch_version')} / CUDA {torch.get('cuda_version')}")
        print(f"CUDA available: {torch.get('cuda_available')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m installer",
        description="Detect hardware and install the matching PyTorch runtime.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="Print CPU, RAM, disk and NVIDIA info.")
    detect.add_argument("--json", action="store_true")
    detect.set_defaults(func=_cmd_detect)

    plan = subparsers.add_parser("plan", help="Show the PyTorch wheel that would be installed.")
    plan.add_argument("--json", action="store_true")
    _add_plan_flags(plan)
    plan.set_defaults(func=_cmd_plan)

    torch = subparsers.add_parser("torch", help="Install torch/torchvision into a venv.")
    torch.add_argument("--venv", default=".venv", help="Virtual environment path (default: .venv).")
    torch.add_argument("--manifest", help="Runtime manifest path (default: <venv>/runtime.json).")
    torch.add_argument("--python", help="Base interpreter used when creating the venv.")
    torch.add_argument("--no-pip-upgrade", action="store_true")
    torch.add_argument("--dry-run", action="store_true")
    torch.add_argument("--json", action="store_true")
    _add_plan_flags(torch)
    torch.set_defaults(func=_cmd_torch)

    install = subparsers.add_parser("install", help="Create a venv and install torch plus requirements.")
    install.add_argument("--venv", default=".venv", help="Virtual environment path (default: .venv).")
    install.add_argument("--requirements", default="requirements.txt", help="Requirements file.")
    install.add_argument("--manifest", help="Runtime manifest path (default: <venv>/runtime.json).")
    install.add_argument("--python", help="Base interpreter used when creating the venv.")
    install.add_argument("--dry-run", action="store_true")
    install.add_argument("--json", action="store_true")
    _add_plan_flags(install)
    install.set_defaults(func=_cmd_install)

    verify = subparsers.add_parser("verify", help="Import torch from a venv and check CUDA.")
    verify.add_argument("--venv", default=".venv")
    verify.add_argument("--json", action="store_true")
    verify.set_defaults(func=_cmd_verify)

    status = subparsers.add_parser("status", help="Read a runtime manifest.")
    status.add_argument("--manifest", default=".venv/runtime.json")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=_cmd_status)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, RuntimeError, ValueError, FileNotFoundError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())