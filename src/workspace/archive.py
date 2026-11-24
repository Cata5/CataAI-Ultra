"""Portable workspace archives with checksums and a versioned manifest.

The archive contains database records plus only the files those records
reference; it never zips the repository. Import creates a new workspace and
never overwrites an existing id.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Tuple

from ..db.database import StudioDatabase

logger = logging.getLogger(__name__)

ARCHIVE_SCHEMA = "llmmaker.workspace/1"
MANIFEST_NAME = "manifest.json"
MAX_MANIFEST_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024
MAX_ENTRIES = 20_000


class WorkspaceArchiveError(ValueError):
    """Archive is malformed, unsafe, or fails its checksum."""


def _sha256(path: Path) -> Tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _rows(conn: sqlite3.Connection, sql: str, params: Iterable[Any] = ()) -> List[Dict[str, Any]]:
    return [dict(row) for row in conn.execute(sql, tuple(params)).fetchall()]


class WorkspaceArchive:
    def __init__(self, db: StudioDatabase, base_dir: Path | str = "workspaces") -> None:
        self.db = db
        self.base_dir = Path(base_dir)

    def _safe_id(self, value: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(value).strip().lower())
        return cleaned or "workspace"

    def _unique_id(self, value: str) -> str:
        base = self._safe_id(value)
        if base == "default":
            base = "imported"
        candidate = base
        suffix = 1
        while self.db.get_workspace(candidate) or (self.base_dir / candidate).exists():
            suffix += 1
            candidate = f"{base}_{int(time.time())}_{suffix}"
        return candidate

    def export(self, workspace_id: str) -> Tuple[Path, str]:
        ws = self.db.get_workspace(workspace_id)
        if not ws:
            raise WorkspaceArchiveError(f"Workspace '{workspace_id}' not found.")

        with self.db._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            run_rows = _rows(conn, "SELECT * FROM runs WHERE workspace_id = ? ORDER BY created_at", (workspace_id,))
            run_ids = [row["id"] for row in run_rows]
            metric_rows: List[Dict[str, Any]] = []
            if run_ids:
                marks = ",".join("?" for _ in run_ids)
                metric_rows = _rows(conn, f"SELECT * FROM metrics WHERE run_id IN ({marks}) ORDER BY id", run_ids)
            checkpoint_rows = _rows(conn, "SELECT * FROM checkpoints WHERE workspace_id = ? ORDER BY created_at", (workspace_id,))
            dataset_rows = _rows(conn, "SELECT * FROM datasets WHERE workspace_id = ? ORDER BY created_at", (workspace_id,))

        candidates: List[Tuple[str, Dict[str, Any]]] = []
        for kind, rows, field in (("checkpoints", checkpoint_rows, "file_path"), ("datasets", dataset_rows, "file_path")):
            for row in rows:
                raw = row.get(field)
                if not raw:
                    continue
                path = Path(str(raw))
                if not path.is_absolute():
                    path = Path.cwd() / path
                try:
                    resolved = path.resolve(strict=True)
                except (OSError, RuntimeError):
                    logger.warning("Skipping missing %s file: %s", kind, raw)
                    continue
                if not resolved.is_file():
                    continue
                candidates.append((kind, {"row": row, "path": resolved}))

        if not candidates and not run_rows and not checkpoint_rows and not dataset_rows:
            logger.info("Exporting empty workspace %s", workspace_id)

        exports_dir = Path("data/exports")
        exports_dir.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(prefix=f"workspace_{self._safe_id(workspace_id)}_", suffix=".zip", dir=exports_dir, delete=False)
        archive_path = Path(handle.name)
        handle.close()

        manifest: Dict[str, Any] = {
            "schema": ARCHIVE_SCHEMA,
            "created_at": time.time(),
            "workspace": {
                "id": ws["id"],
                "name": ws["name"],
                "description": ws.get("description", ""),
                "config": ws.get("config", {}),
            },
            "runs": run_rows,
            "metrics": metric_rows,
            "checkpoints": checkpoint_rows,
            "datasets": dataset_rows,
            "files": [],
        }

        total = 0
        try:
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
                for index, (kind, item) in enumerate(candidates):
                    path: Path = item["path"]
                    digest, size = _sha256(path)
                    total += size
                    if total > MAX_ARCHIVE_BYTES:
                        raise WorkspaceArchiveError("Workspace exceeds the 4 GiB archive limit.")
                    suffix = path.suffix.lower()[:16]
                    entry = f"files/{index:04d}{suffix}"
                    zf.write(path, entry)
                    manifest["files"].append({
                        "entry": entry,
                        "kind": kind,
                        "record_id": item["row"]["id"],
                        "original_name": path.name,
                        "sha256": digest,
                        "size": size,
                    })
                zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2, default=str))
        except Exception:
            archive_path.unlink(missing_ok=True)
            raise

        return archive_path, f"{self._safe_id(workspace_id)}-workspace.zip"

    def import_archive(self, archive_path: Path | str) -> Dict[str, Any]:
        source = Path(archive_path)
        if not source.is_file():
            raise WorkspaceArchiveError("Upload is not a file.")
        if source.stat().st_size > MAX_ARCHIVE_BYTES:
            raise WorkspaceArchiveError("Archive exceeds the 4 GiB limit.")

        try:
            zf = zipfile.ZipFile(source)
        except zipfile.BadZipFile as exc:
            raise WorkspaceArchiveError("Upload is not a valid ZIP archive.") from exc

        with zf:
            infos = zf.infolist()
            if len(infos) > MAX_ENTRIES:
                raise WorkspaceArchiveError("Archive contains too many entries.")
            if sum(info.file_size for info in infos) > MAX_ARCHIVE_BYTES:
                raise WorkspaceArchiveError("Archive expands beyond the 4 GiB limit.")
            try:
                manifest_info = zf.getinfo(MANIFEST_NAME)
            except KeyError as exc:
                raise WorkspaceArchiveError("Archive has no manifest.json.") from exc
            if manifest_info.file_size > MAX_MANIFEST_BYTES:
                raise WorkspaceArchiveError("Manifest is too large.")
            try:
                manifest = json.loads(zf.read(manifest_info).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise WorkspaceArchiveError("Manifest is not valid UTF-8 JSON.") from exc
            if not isinstance(manifest, dict) or manifest.get("schema") != ARCHIVE_SCHEMA:
                raise WorkspaceArchiveError("Unsupported archive schema.")
            for key in ("workspace", "runs", "metrics", "checkpoints", "datasets", "files"):
                if not isinstance(manifest.get(key), (dict, list)):
                    raise WorkspaceArchiveError(f"Manifest field '{key}' has the wrong type.")

            if len(manifest["files"]) > MAX_ENTRIES:
                raise WorkspaceArchiveError("Manifest lists too many files.")

            ws_data = manifest["workspace"]
            if not isinstance(ws_data, dict) or not str(ws_data.get("name", "")).strip():
                raise WorkspaceArchiveError("Manifest has no workspace name.")
            ws_id = self._unique_id(ws_data.get("id") or ws_data["name"])
            root = self.base_dir / ws_id
            files_dir = root / "files"
            files_dir.mkdir(parents=True, exist_ok=False)

            file_map: Dict[str, str] = {}
            try:
                for index, entry in enumerate(manifest["files"]):
                    if not isinstance(entry, dict):
                        raise WorkspaceArchiveError("File entry is malformed.")
                    member = str(entry.get("entry", ""))
                    pure = PurePosixPath(member)
                    if (
                        pure.is_absolute()
                        or ".." in pure.parts
                        or len(pure.parts) != 2
                        or pure.parts[0] != "files"
                        or "\\" in member
                        or ":" in member
                        or "\x00" in member
                    ):
                        raise WorkspaceArchiveError(f"Unsafe file path in manifest: {member!r}")
                    try:
                        info = zf.getinfo(member)
                    except KeyError as exc:
                        raise WorkspaceArchiveError(f"Archive is missing {member}.") from exc
                    expected = str(entry.get("sha256", "")).lower()
                    if len(expected) != 64:
                        raise WorkspaceArchiveError(f"Missing checksum for {member}.")
                    digest = hashlib.sha256()
                    # The destination is generated, never derived from the archive name.
                    dest = files_dir / f"{index:04d}.bin"
                    written = 0
                    with zf.open(info) as src, open(dest, "wb") as out:
                        for chunk in iter(lambda: src.read(1024 * 1024), b""):
                            written += len(chunk)
                            if written > MAX_ARCHIVE_BYTES:
                                raise WorkspaceArchiveError("File exceeds the 4 GiB limit.")
                            digest.update(chunk)
                            out.write(chunk)
                    if digest.hexdigest() != expected:
                        raise WorkspaceArchiveError(f"Checksum mismatch for {member}.")
                    if int(entry.get("size", written)) != written:
                        raise WorkspaceArchiveError(f"Size mismatch for {member}.")
                    file_map[f"{entry.get('kind', '')}:{entry.get('record_id', '')}"] = str(dest)
            except Exception:
                shutil.rmtree(root, ignore_errors=True)
                raise

        created = self.db.create_workspace(
            workspace_id=ws_id,
            name=str(ws_data["name"]),
            description=str(ws_data.get("description", "Imported workspace")),
            root_path=str(root),
            config=ws_data.get("config") if isinstance(ws_data.get("config"), dict) else {},
        )
        run_map: Dict[str, str] = {}
        try:
            with self.db._lock, self.db._get_connection() as conn:
                for row in manifest["runs"]:
                    old = str(row.get("id", ""))
                    new_id = f"{ws_id}__{old}"
                    run_map[old] = new_id
                    conn.execute(
                        """INSERT INTO runs
                        (id, name, status, current_phase, dataset_source, workspace_id, total_steps,
                         total_tokens, best_val_loss, final_loss, config_json, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (new_id, row.get("name", old), "STOPPED", row.get("current_phase", "pretrain"),
                         row.get("dataset_source"), ws_id, row.get("total_steps", 0), row.get("total_tokens", 0),
                         row.get("best_val_loss"), row.get("final_loss"), row.get("config_json", "{}"),
                         row.get("created_at", time.time()), row.get("updated_at", time.time())),
                    )
                for row in manifest["metrics"]:
                    mapped = run_map.get(str(row.get("run_id", "")))
                    if not mapped:
                        continue
                    conn.execute(
                        """INSERT INTO metrics
                        (run_id, step, tokens, train_loss, val_loss, generalization_gap, grad_norm,
                         muon_lr, adamw_lr, vram_used_gb, tok_per_sec, tflops, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (mapped, row.get("step", 0), row.get("tokens", 0), row.get("train_loss", 0.0),
                         row.get("val_loss"), row.get("generalization_gap"), row.get("grad_norm", 0.0),
                         row.get("muon_lr", 0.0), row.get("adamw_lr", 0.0), row.get("vram_used_gb", 0.0),
                         row.get("tok_per_sec", 0.0), row.get("tflops", 0.0), row.get("timestamp", time.time())),
                    )
                for row in manifest["checkpoints"]:
                    old = str(row.get("id", ""))
                    conn.execute(
                        """INSERT INTO checkpoints
                        (id, run_id, step, tokens, train_loss, val_loss, file_path, file_size_bytes,
                         phase, workspace_id, metadata_json, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (f"{ws_id}__{old}", run_map.get(str(row.get("run_id", ""))), row.get("step", 0),
                         row.get("tokens", 0), row.get("train_loss"), row.get("val_loss"),
                         file_map.get(f"checkpoints:{old}", ""), row.get("file_size_bytes", 0), row.get("phase"), ws_id,
                         row.get("metadata_json", "{}"), row.get("created_at", time.time())),
                    )
                for row in manifest["datasets"]:
                    old = str(row.get("id", ""))
                    conn.execute(
                        """INSERT INTO datasets
                        (id, name, file_path, file_size_bytes, format, workspace_id, token_count,
                         sample_count, revision, stats_json, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (f"{ws_id}__{old}", row.get("name", old), file_map.get(f"datasets:{old}", ""),
                         row.get("file_size_bytes", 0), row.get("format", "jsonl"), ws_id,
                         row.get("token_count", 0), row.get("sample_count", 0), row.get("revision", "1"),
                         row.get("stats_json", "{}"), row.get("created_at", time.time())),
                    )
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            try:
                self.db.delete_workspace(ws_id)
            except Exception:
                logger.exception("Failed to roll back workspace %s", ws_id)
            raise

        return {
            "success": True,
            "workspace": created,
            "runs_imported": len(run_map),
            "checkpoints_imported": len(manifest["checkpoints"]),
            "datasets_imported": len(manifest["datasets"]),
            "files_imported": len(file_map),
        }
