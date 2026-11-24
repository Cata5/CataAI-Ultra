# First-run installer design

This document describes how the portable setup becomes a downloadable Windows
installer. The detection and PyTorch installation engine already exists in
`installer/`; the Tauri packaging and setup screen are the remaining work.

## Decision: two phases, not one NSIS script

Do not install PyTorch from the NSIS installer. A CUDA wheel is 2–3 GB, the
installer would need admin rights and a network connection before the app ever
launches, and an interrupted download can leave a half-installed app that
cannot start its own repair UI.

Split the work:

1. **OS installer** — small, signed, per-user. Installs the Tauri executable,
   the static frontend, the Python sources and a pinned standalone CPython.
2. **First-run wizard** — runs inside the app. Detects the GPU, shows the
   selected CUDA runtime and download size, installs PyTorch into an
   app-managed virtual environment, verifies CUDA, then opens the workstation.

This keeps the download small, lets the app explain driver problems, and makes
the CUDA install resumable/retryable without reinstalling the application.

## Install layout

Per-user install (`currentUser`, no UAC):

```text
%LOCALAPPDATA%\Programs\LLMMaker\
  llm-maker.exe
  resources\
    app\src\...                Python worker sources
    app\installer\...          this package
    app\requirements.txt
    python\python.exe          pinned standalone CPython 3.11/3.12

%LOCALAPPDATA%\LLMMaker\
  runtime\
    py311\                     virtual environment
    runtime.json               hardware + wheel + verification
    cache\pip\                 wheel cache (optional)
  data\                        studio.db, checkpoints, secrets
  logs\
```

Never write the database into `Program Files`. The Tauri app must resolve the
data directory with `app.path().app_local_data_dir()` and pass it to the Python
worker. This also fixes the current project-root walk after installation.

## Tauri configuration

Bundle the interpreter and Python sources as resources; keep the venv out of
the bundle.

```json
{
  "bundle": {
    "active": true,
    "targets": ["nsis"],
    "resources": [
      "resources/**/*"
    ],
    "windows": {
      "nsis": {
        "installMode": "currentUser"
      }
    }
  }
}
```

Before `tauri build`, stage everything under `src-tauri/resources/`:

```text
src-tauri/resources/
  app/src/...              Python worker sources
  app/installer/...        this package
  app/requirements.txt
  python/python.exe        pinned standalone CPython
```

`resources` paths in `tauri.conf.json` are relative to `src-tauri/`, so the
single `resources/**/*` entry keeps the layout intact and avoids Tauri's `_up_`
prefix for paths outside the config directory. Populate `resources/python` at
build time from a pinned `python-build-standalone` CPython archive
(install-only, x86_64). Verify the archive SHA-256 in the build script; do not
download it ad hoc on the user's machine. The standalone build includes `pip`,
which the wizard uses to create the venv.

## First-run sequence

1. Tauri starts and reads
   `%LOCALAPPDATA%\LLMMaker\runtime\runtime.json`.
2. If the manifest is missing or invalid, show the setup screen instead of the
   dashboard.
3. Setup invokes the bundled interpreter with the app's resource directory as
   the working directory (or `PYTHONPATH`), so `-m installer` resolves:

   ```text
   cd %LOCALAPPDATA%\Programs\LLMMaker\resources\app
   ..\python\python.exe -m installer plan --json
   ```

   The result includes GPU name, driver, compute capability, free disk and the
   selected channel. Show the channel, download size and CPU fallback before
   starting.
4. On confirmation, run the install with a pipe the UI can read:

   ```text
   ..\python\python.exe -m installer install \
     --venv %LOCALAPPDATA%\LLMMaker\runtime\py311 \
     --requirements requirements.txt \
     --manifest %LOCALAPPDATA%\LLMMaker\runtime\runtime.json
   ```
5. Stream stdout/stderr to the setup screen. `installer` prints each pip
   command, so the UI can show the current stage (venv, torch, requirements,
   verify).
6. On success, `runtime.json` contains:

   ```json
   {
     "schema": 1,
     "venv": "C:\\Users\\...\\LLMMaker\\runtime\\py311",
     "python": "...\\Scripts\\python.exe",
     "plan": { "channel": "cu128", "index_url": "https://download.pytorch.org/whl/cu128" },
     "hardware": { "gpus": [{ "name": "NVIDIA GeForce RTX 3060", "driver_version": "621.24" }] },
     "torch": { "torch_version": "2.11.0+cu128", "cuda_available": true }
   }
   ```
7. Start the Python worker with `runtime.python`; do not search for a system
   Python.

### UI states

| State | UI |
| --- | --- |
| No manifest | Setup wizard, preflight check |
| Plan blocked | GPU name + required driver + link to NVIDIA; buttons `Retry` and `Install CPU build` |
| Installing | Stage list, last pip line, cancel |
| Verification failed | Log tail, `Retry CUDA`, `Switch to CPU` |
| Ready | Dashboard |
| Manifest exists, venv missing | Treat as broken; offer repair |

Cancel must terminate the whole pip process tree, then remove the staging
directory. Install into `runtime\staging-<id>` and rename to `py311` only after
verification, so a crash never leaves a half-valid runtime.

## Hardware matrix

The Python engine uses these NVIDIA driver floors (Windows / Linux):

| Channel | Windows | Linux |
| --- | --- | --- |
| `cu128` | 570.65 | 570.26 |
| `cu126` | 560.94 | 560.28 |
| `cu124` | 551.61 | 550.54 |
| `cu121` | 531.14 | 530.30 |
| `cu118` | 452.39 | 450.36 |

Blackwell (compute capability 10.0+, including RTX 50 series) requires
`cu128`. If the driver is older, block and ask for a driver update rather than
downloading a CPU wheel by surprise.

AMD on Windows has no PyTorch ROCm wheels; select CPU. macOS uses the default
PyPI wheel (MPS). The current product is Windows-first; do not advertise ROCm
until a Linux test target exists.

## Worker resolution

Change `src-tauri/src/worker.rs` to resolve the interpreter in this order:

1. `LLMMAKER_PYTHON` (explicit override / tests).
2. `runtime.json` in the app data directory.
3. `.venv` in the repository (development mode).
4. Clear error: "Run the setup wizard" rather than `<python-not-found>`.

Pass the app data directory to the worker as an environment variable or CLI
argument so SQLite, checkpoints and logs stay out of the install directory.

## Security and operations

- Sign both `llm-maker.exe` and the NSIS installer; unsigned installers trigger
  SmartScreen and make GPU downloads look suspicious.
- Install per-user; the wizard needs no administrator rights.
- Download only from `https://download.pytorch.org/whl/<channel>` and PyPI over
  HTTPS. Do not accept an arbitrary index URL from the UI.
- Keep the pinned standalone Python archive and requirements hashes in the
  repository. The wizard installs a moving `torch` version today; for release,
  generate a lock file per channel and pass it to pip.
- Log the plan, pip exit code, and `torch.cuda.is_available()` result. Do not
  log secrets or the full environment.
- Provide an offline path: a ZIP with the CPU wheel and a prebuilt venv for
  air-gapped labs. Do not make the normal installer multi-gigabyte.

## Testing

Unit tests already cover driver thresholds, Blackwell blocking, CPU/macOS
fallback, CSV parsing and manifest round-trips. Before shipping:

- Windows 10 + GTX 10-series, Windows 11 + RTX 30/40/50, no-GPU VM.
- Driver one version below each floor: setup blocks with the right message.
- Interrupted download, disk full, proxy failure, and cancel mid-install.
- Re-run setup with an existing valid runtime and with a corrupted venv.
- Confirm the installed app starts with no system Python on `PATH`.
- Confirm uninstall leaves or removes `%LOCALAPPDATA%\LLMMaker` according to a
  documented choice.

## Remaining implementation

1. Build script that downloads and verifies the standalone CPython, then stages
   `src-tauri/resources/python` and `src-tauri/resources/app`.
2. Tauri resource configuration and `app_local_data_dir()` plumbing.
3. Setup screen + Rust command that streams installer output.
4. Worker interpreter resolution from `runtime.json`.
5. NSIS signing, versioning, and a release workflow.