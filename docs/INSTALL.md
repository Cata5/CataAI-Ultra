# Installing LLM Maker

LLM Maker is a Windows-first desktop application (Tauri shell + Python training
engine + Next.js UI). This page describes the supported portable install and the
unsigned per-user installer. No Authenticode-signed installer is published; see
"Current limits".

## Requirements

| Component | Minimum | Notes |
| --- | --- | --- |
| OS | Windows 10/11 x64 | Linux/macOS are untested |
| Python | 3.10+ (3.11/3.12 recommended) | Must be on `PATH` as `py -3.12`, `py -3.11`, or `python` |
| Node.js | 20+ | Needed to build the UI |
| Rust | stable, MSVC or GNU toolchain | Needed to build the desktop shell |
| GPU | Optional | NVIDIA driver 452.39+ for CUDA 11.8, 570.65+ for CUDA 12.8 |
| Disk | ~8 GB free | Up to ~8 GB is used by a CUDA wheel, pip cache and `.venv`; more for models |

## One-command setup

From the repository root (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
```

The script starts with hardware detection:

1. Finds a suitable Python interpreter.
2. Runs `nvidia-smi` and reads the GPU name, driver version, VRAM and compute
   capability. It also prints CPU architecture, RAM and free disk.
3. Selects the newest PyTorch CUDA runtime the driver supports
   (`cu128`, `cu126`, `cu124`, `cu121`, `cu118`) or the CPU build.
4. Creates `.venv`, installs the matching `torch`/`torchvision` from the
   official PyTorch index, then installs `requirements.txt`.
5. Runs `npm ci` and a Tauri static-export build (`TAURI_BUILD=true`) in
   `frontend/`, producing `frontend/out/`.
6. Runs `cargo build` in `src-tauri/`.
7. Writes `.venv/runtime.json` with the detected hardware and the verified
   PyTorch/CUDA versions.

It is safe to re-run. Existing `.venv` and a PyTorch build that already matches
the selected CUDA runtime are reused. Use `-Check` to print the hardware plan
without changing anything, `-SkipFrontend` or `-SkipRust` to skip a stage, and
`-PythonExe` to pin an interpreter.

To see the plan without installing anything:

```powershell
.venv\Scripts\python.exe -m installer detect
.venv\Scripts\python.exe -m installer plan
```

### GPU wheel selection

The installer chooses the runtime automatically; the old manual `pip install
torch --index-url ...` step is no longer needed. The table is a floor: a newer
driver can run every older CUDA runtime.

| GPU / driver | Wheel |
| --- | --- |
| Blackwell (compute capability 10.0+), driver 570.65+ | `cu128` |
| NVIDIA, driver 570.65+ | `cu128` |
| NVIDIA, driver 560.94+ | `cu126` |
| NVIDIA, driver 551.61+ | `cu124` |
| NVIDIA, driver 531.14+ | `cu121` |
| NVIDIA, driver 452.39+ | `cu118` |
| No NVIDIA GPU | CPU (`--index-url https://download.pytorch.org/whl/cpu`) |
| macOS | Default PyPI wheel with MPS; no CUDA wheels |
| NVIDIA but driver below 452.39 | Blocked until the driver is updated, or `-CpuOnly` |

If an NVIDIA GPU is present but its driver is too old, setup stops with the
required driver version instead of silently downloading a CPU wheel. Either
update the driver, or install a CPU build deliberately:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1 -CpuOnly
powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1 -CudaTag cu126
```

`-CudaTag` is for testing or a pinned enterprise driver. The installer verifies
`torch.cuda.is_available()` after installation and fails with driver guidance if
CUDA cannot initialize.

## Start the app

```powershell
# Native desktop shell (builds the frontend and copies llm-maker.exe)
run_desktop.bat

# Browser UI (serves the API on :8080 and the UI on :3000)
.venv\Scripts\python.exe dashboard.py
npm --prefix frontend run dev
```

The desktop shell locates the project root by walking up from the executable
until `src/worker/standalone_runner.py` is found. For a portable or relocated
install, set `LLMMAKER_PROJECT_ROOT` to that directory. No registry entries or
files outside the project directory are created (Python packages live in
`.venv`, build output in the configured Cargo target directory).

## Verifying an install

```powershell
.venv\Scripts\python.exe -m installer verify --venv .venv
.venv\Scripts\python.exe -m pytest tests/unit -q
.venv\Scripts\python.exe -m pytest tests/test_api_endpoints.py -q
npx --prefix frontend tsc --noEmit --incremental false
cargo check --manifest-path src-tauri/Cargo.toml
```

Then open the app, start a short run, stop it, and confirm the run shows a
final loss and one checkpoint.

## Troubleshooting

- **`No compatible PyTorch runtime`** — the NVIDIA driver is below the CUDA
  11.8 floor, or a Blackwell GPU is below 570.65. Update the driver, or use
  `-CpuOnly`.
- **`PyTorch installed but CUDA is not available`** — `nvidia-smi` sees a GPU
  but PyTorch cannot initialize it. Check the driver version, reboot after a
  driver update, and re-run `.venv\Scripts\python.exe -m installer verify`.
- **Slow first install** — a CUDA wheel is 2–3 GB. PyTorch's CDN is used, not
  PyPI; proxy settings must allow `download.pytorch.org`.
- **Switch to CPU after a CUDA attempt** — a CUDA build still runs on CPU, so
  bootstrap leaves it in place. `-CpuOnly` selects the CPU wheel for a fresh
  install. To replace an existing CUDA build, uninstall `torch` and
  `torchvision` from `.venv`, then re-run with `-CpuOnly`.

## Unsigned installer

The per-user NSIS build bundles a private CPython runtime and the app sources:

```powershell
npm ci --prefix frontend
.\scripts\build_installer.ps1 `
  -PythonArchive C:\downloads\cpython-3.11-install_only.tar.gz `
  -PythonSha256 <sha256>
```

It writes to
`%LOCALAPPDATA%\LLMMaker\cargo-target\release\bundle\nsis`, writes a
`.sha256` beside it, and installs per-user into
`%LOCALAPPDATA%\LLM Maker`, with `uninstall.exe` beside the app.
The installer is unsigned, so SmartScreen warns on a downloaded copy; verify the
artifact SHA-256 before running it. PyTorch is still installed by the first-run
wizard, not by NSIS.

## Current limits

- No Authenticode-signed installer or auto-updater is published. The unsigned
  installer is reproducible locally and from the release workflow; see
  `docs/INSTALLER.md` for the packaging design and signing fallback.
- The portable `.venv` flow still needs a system Python. Only the NSIS payload
  bundles CPython.
- CI (`.github/workflows/ci.yml`) has not been executed on GitHub-hosted
  runners yet.
- Linux and macOS packaging is untested. The detector uses the same CUDA
  driver floors on Linux; macOS installs the default PyPI wheel with MPS.