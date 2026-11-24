# Installing LLM Maker

LLM Maker is a Windows-first desktop application (Tauri shell + Python training
engine + Next.js UI). This page describes the supported development/portable
install. A signed one-click installer is **not** shipped yet; see "Current
limits" at the end.

## Requirements

| Component | Minimum | Notes |
| --- | --- | --- |
| OS | Windows 10/11 x64 | Linux/macOS are untested |
| Python | 3.10+ (3.11/3.12 recommended) | Must be on `PATH` as `py -3.12`, `py -3.11`, or `python` |
| Node.js | 20+ | Needed to build the UI |
| Rust | stable, MSVC or GNU toolchain | Needed to build the desktop shell |
| GPU | Optional | NVIDIA driver 550+ for CUDA 12.x wheels; CPU training works |
| Disk | ~8 GB | Python wheels, Node modules, Rust build cache, model data |

## One-command setup

From the repository root (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
```

The script:

1. Finds a suitable Python interpreter.
2. Creates `.venv` and installs `requirements.txt`.
3. Runs `npm ci` and `npm run build` in `frontend/`.
4. Runs `cargo build` in `src-tauri/`.
5. Prints the commands to start the app.

It is safe to re-run. Use `-Check` to verify prerequisites without changing
anything, `-SkipFrontend` or `-SkipRust` to skip a stage, and `-PythonExe` to
pin an interpreter.

### GPU wheels

`requirements.txt` installs the default PyTorch build. For CUDA, install the
matching wheel first, then bootstrap:

```powershell
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1 -SkipFrontend -SkipRust
```

CUDA 12.4 builds are available at `/cu124`; CPU-only builds need no extra index.

## Start the app

```powershell
# Native desktop shell
cargo run --manifest-path src-tauri/Cargo.toml

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
.venv\Scripts\python.exe -m pytest tests/unit -q
.venv\Scripts\python.exe -m pytest tests/test_api_endpoints.py -q
npx --prefix frontend tsc --noEmit --incremental false
cargo check --manifest-path src-tauri/Cargo.toml
```

Then open the app, start a short run, stop it, and confirm the run shows a
final loss and one checkpoint.

## Current limits

- There is no signed installer or auto-updater yet; the build is a local
  portable install.
- Python is not bundled. `.cargo/config.toml` still contains
  machine-specific toolchain paths (tracked as AUD-022); remove or override
  them for other machines.
- CI (`.github/workflows/ci.yml`) has not been executed on GitHub-hosted
  runners yet.
- Linux and macOS packaging is untested.