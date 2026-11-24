"""Tests for first-run hardware detection and PyTorch wheel selection."""

import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from installer.hardware import Gpu, HardwareProfile, parse_gpu_lines
from installer.plan import choose_plan
from installer.runtime import _channel_cuda_version, install_torch, read_manifest, write_manifest


@pytest.fixture
def workdir() -> Path:
    # pytest's tmp_path uses mkdtemp(chmod=0700), which the Windows sandbox denies.
    path = Path.cwd() / f".installer-test-{uuid.uuid4().hex[:8]}"
    path.mkdir()
    yield path
    shutil.rmtree(path, ignore_errors=True)


def make_profile(
    *,
    driver: str = "621.24",
    os_name: str = "windows",
    capability: str = "8.6",
    gpus: bool = True,
    free_disk_gb: float = 100.0,
    error: str | None = None,
) -> HardwareProfile:
    devices: tuple[Gpu, ...] = ()
    if gpus:
        devices = (Gpu(0, "NVIDIA GeForce RTX 3060", driver, 6144, capability),)
    return HardwareProfile(
        os_name=os_name,
        architecture="x86_64",
        python_version="3.11.9",
        python_executable="python.exe",
        gpus=devices,
        nvidia_smi="nvidia-smi.exe" if gpus else None,
        nvidia_error=error,
        free_disk_gb=free_disk_gb,
        total_ram_gb=16.0,
    )


@pytest.mark.parametrize(
    ("driver", "expected"),
    [
        ("621.24", "cu128"),
        ("570.65", "cu128"),
        ("570.64", "cu126"),
        ("560.94", "cu126"),
        ("560.93", "cu124"),
        ("551.61", "cu124"),
        ("551.60", "cu121"),
        ("531.14", "cu121"),
        ("531.13", "cu118"),
        ("452.39", "cu118"),
    ],
)
def test_windows_driver_thresholds(driver: str, expected: str) -> None:
    plan = choose_plan(make_profile(driver=driver))
    assert plan.channel == expected
    assert plan.is_cuda
    assert not plan.blocked


def test_missing_torchvision_reinstalls_the_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    from installer import runtime

    plan = choose_plan(make_profile())
    calls: list[list[str]] = []

    def fake_verify(*_args, **_kwargs):
        if calls:
            return {
                "available": True,
                "torch_version": "2.11.0+cu128",
                "torchvision_version": "0.26.0+cu128",
                "torchvision_available": True,
                "cuda_version": "12.8",
                "cuda_available": True,
            }
        return {
            "available": True,
            "torch_version": "2.11.0+cu128",
            "torchvision_version": None,
            "torchvision_available": False,
            "cuda_version": "12.8",
            "cuda_available": True,
        }

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runtime, "verify_torch", fake_verify)
    monkeypatch.setattr(runtime, "run", fake_run)

    result = install_torch(Path("python.exe"), plan, upgrade_pip=False)

    assert result["torchvision_available"] is True
    assert any("--force-reinstall" in command for command in calls)


def test_channel_cuda_version_mapping() -> None:
    assert _channel_cuda_version("cu128") == "12.8"
    assert _channel_cuda_version("cu126") == "12.6"
    assert _channel_cuda_version("cu121") == "12.1"
    assert _channel_cuda_version("cu118") == "11.8"


def test_linux_uses_linux_driver_floor() -> None:
    assert choose_plan(make_profile(os_name="linux", driver="570.26")).channel == "cu128"
    assert choose_plan(make_profile(os_name="linux", driver="570.25")).channel == "cu126"


def test_blackwell_requires_cuda_12_8() -> None:
    plan = choose_plan(make_profile(driver="570.65", capability="12.0"))
    assert plan.channel == "cu128"
    assert not plan.blocked


def test_blackwell_blocks_on_old_driver() -> None:
    plan = choose_plan(make_profile(driver="560.94", capability="12.0"))
    assert plan.blocked
    assert plan.channel == "cpu"
    assert "Update the NVIDIA driver" in (plan.warning or "")


def test_old_driver_blocks_instead_of_silent_cpu_download() -> None:
    plan = choose_plan(make_profile(driver="400.00"))
    assert plan.blocked
    assert plan.channel == "cpu"
    assert "--cpu" in (plan.warning or "")


def test_no_nvidia_selects_cpu_automatically() -> None:
    plan = choose_plan(make_profile(gpus=False))
    assert plan.channel == "cpu"
    assert not plan.blocked
    assert plan.index_url == "https://download.pytorch.org/whl/cpu"


def test_macos_uses_pypi_wheel_for_mps() -> None:
    plan = choose_plan(make_profile(os_name="darwin", gpus=False))
    assert plan.channel == "cpu"
    assert plan.index_url is None
    assert "MPS" in plan.reason


def test_cpu_override_wins_even_with_gpu() -> None:
    plan = choose_plan(make_profile(), cpu_only=True)
    assert plan.channel == "cpu"
    assert not plan.is_cuda


def test_explicit_cuda_tag_builds_index() -> None:
    plan = choose_plan(make_profile(), cuda_tag="cu126")
    assert plan.channel == "cu126"
    assert plan.index_url == "https://download.pytorch.org/whl/cu126"


def test_invalid_cuda_tag_is_rejected() -> None:
    with pytest.raises(ValueError):
        choose_plan(make_profile(), cuda_tag="https://evil.example")


def test_low_disk_adds_warning() -> None:
    plan = choose_plan(make_profile(free_disk_gb=4.0))
    assert "Only 4.0 GB free" in (plan.warning or "")


def test_parse_nvidia_smi_csv() -> None:
    text = "0, NVIDIA GeForce RTX 3060 Laptop GPU, 621.24, 6144 MiB, 8.6\n"
    gpus = parse_gpu_lines(text, has_compute_capability=True)
    assert len(gpus) == 1
    assert gpus[0].name == "NVIDIA GeForce RTX 3060 Laptop GPU"
    assert gpus[0].memory_mb == 6144
    assert gpus[0].compute_capability_tuple == (8, 6)


def test_parse_nvidia_smi_without_compute_cap() -> None:
    text = "0, NVIDIA T4, 550.54, 15360\n"
    gpus = parse_gpu_lines(text, has_compute_capability=False)
    assert gpus[0].compute_capability is None
    assert gpus[0].compute_capability_tuple == (0, 0)


def test_parse_nvidia_smi_na_compute_cap() -> None:
    text = "0, NVIDIA T4, 550.54, 15360, N/A\n"
    gpus = parse_gpu_lines(text, has_compute_capability=True)
    assert gpus[0].compute_capability is None


def test_runtime_manifest_round_trip(workdir: Path) -> None:
    profile = make_profile()
    plan = choose_plan(profile)
    manifest = workdir / "runtime.json"
    write_manifest(
        manifest,
        venv_dir=workdir / ".venv",
        plan=plan,
        hardware=profile,
        verification={"available": True, "torch_version": "2.11.0", "cuda_available": True},
    )
    data = read_manifest(manifest)
    assert data is not None
    assert data["schema"] == 1
    assert data["plan"]["channel"] == "cu128"
    assert data["torch"]["cuda_available"] is True
    assert not (workdir / "runtime.json.tmp").exists()


def test_read_manifest_rejects_wrong_schema(workdir: Path) -> None:
    path = workdir / "runtime.json"
    path.write_text('{"schema": 99, "python": "x"}', encoding="utf-8")
    assert read_manifest(path) is None