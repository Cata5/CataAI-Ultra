"""Portable application paths.

Source checkouts keep their historical project-root layout. Packaged builds set
LLMMAKER_DATA_DIR (Tauri does this automatically) so SQLite, checkpoints and
logs live under the per-user application-data directory instead of Program Files.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

APP_DIR_NAME = "LLMMaker"


def project_root() -> Path:
    """Directory containing src/; used for bundled read-only resources."""
    return Path(__file__).resolve().parents[2]


def _platform_data_root() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / APP_DIR_NAME
        return Path.home() / "AppData" / "Local" / APP_DIR_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    base = os.environ.get("XDG_DATA_HOME")
    if base:
        return Path(base) / "llm-maker"
    return Path.home() / ".local" / "share" / "llm-maker"


def data_dir() -> Path:
    """Writable application-data directory.

    LLMMAKER_DATA_DIR is the explicit override used by the desktop shell and by
    tests. Otherwise a source checkout keeps using its project root, while a
    packaged build uses the platform application-data directory.
    """
    explicit = os.environ.get("LLMMAKER_DATA_DIR", "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    # Running from a source checkout keeps the historical project-root layout,
    # which makes development predictable and keeps generated files out of the
    # user profile. Frozen/packaged builds never write next to the executable.
    if not getattr(sys, "frozen", False):
        return project_root()

    path = _platform_data_root()
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_db_path() -> Path:
    explicit = os.environ.get("LLMMAKER_DB_PATH", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    target = data_dir() / "studio.db"
    legacy = data_dir() / "data" / "rocanon_studio.db"
    if not target.exists() and legacy.is_file():
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy, target)
        except OSError:
            return legacy
    return target


def logs_dir() -> Path:
    path = data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def backups_dir() -> Path:
    path = data_dir() / "backups"
    path.mkdir(parents=True, exist_ok=True)
    return path


def datasets_dir() -> Path:
    path = data_dir() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def uploads_dir() -> Path:
    path = datasets_dir() / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    path = datasets_dir() / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoints_dir() -> Path:
    path = data_dir() / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def workspaces_dir() -> Path:
    path = data_dir() / "workspaces"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def exports_dir() -> Path:
    path = data_dir() / "exports"
    path.mkdir(parents=True, exist_ok=True)
    return path
