"""
src/workspace/manager.py
Workspace and Project Isolation Manager for LLM Maker.
Provides isolated directory structures for models, checkpoints, datasets, and run artifacts,
while preserving full backward compatibility for the 'default' workspace.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..db.database import StudioDatabase
from ..utils.paths import data_dir, workspaces_dir

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """Manages workspace lifecycle, directory structure, and resource isolation."""

    def __init__(self, db: Optional[StudioDatabase] = None, base_dir: Optional[Path | str] = None) -> None:
        self.db = db or StudioDatabase()
        self.base_dir = Path(base_dir) if base_dir else workspaces_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_id(self, ws_id: str) -> str:
        s = re.sub(r"[^a-zA-Z0-9_\-]", "_", ws_id.strip().lower())
        return s or "workspace_1"

    def get_workspace_paths(self, workspace_id: str) -> Dict[str, Path]:
        """Returns directory paths for checkpoints, datasets, models, and artifacts for a workspace."""
        if workspace_id == "default":
            # Backward compatibility: use repository root directories
            chk = data_dir() / "checkpoints"
            data = data_dir() / "data"
            models = data_dir() / "networks" / "registry"
            artifacts = data_dir() / "data" / "runs"
        else:
            ws_root = self.base_dir / workspace_id
            chk = ws_root / "checkpoints"
            data = ws_root / "datasets"
            models = ws_root / "models"
            artifacts = ws_root / "runs"

        chk.mkdir(parents=True, exist_ok=True)
        data.mkdir(parents=True, exist_ok=True)
        models.mkdir(parents=True, exist_ok=True)
        artifacts.mkdir(parents=True, exist_ok=True)

        return {
            "root": data_dir() if workspace_id == "default" else (self.base_dir / workspace_id),
            "checkpoints": chk,
            "datasets": data,
            "models": models,
            "artifacts": artifacts,
        }

    def create_workspace(
        self,
        name: str,
        workspace_id: Optional[str] = None,
        description: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ws_id = self._sanitize_id(workspace_id or name)
        # Check existing
        existing = self.db.get_workspace(ws_id)
        if existing:
            # Append timestamp suffix
            ws_id = f"{ws_id}_{int(time.time())}"

        paths = self.get_workspace_paths(ws_id)
        manifest = {
            "id": ws_id,
            "name": name,
            "description": description,
            "created_at": time.time(),
            "config": config or {},
        }

        # Write manifest file inside workspace root
        if ws_id != "default":
            (paths["root"] / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        res = self.db.create_workspace(
            workspace_id=ws_id,
            name=name,
            description=description,
            root_path=str(paths["root"]),
            config=config or {},
        )
        return res

    def list_workspaces(self) -> List[Dict[str, Any]]:
        return self.db.list_workspaces()

    def get_active_workspace(self) -> Dict[str, Any]:
        return self.db.get_active_workspace()

    def set_active_workspace(self, workspace_id: str) -> bool:
        return self.db.set_active_workspace(workspace_id)

    def update_workspace(
        self,
        workspace_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        res = self.db.update_workspace(workspace_id, name, description, config)
        if res and workspace_id != "default":
            paths = self.get_workspace_paths(workspace_id)
            manifest_p = paths["root"] / "manifest.json"
            if manifest_p.exists():
                try:
                    data = json.loads(manifest_p.read_text(encoding="utf-8"))
                    if name is not None:
                        data["name"] = name
                    if description is not None:
                        data["description"] = description
                    if config is not None:
                        data["config"] = config
                    manifest_p.write_text(json.dumps(data, indent=2), encoding="utf-8")
                except Exception as e:
                    logger.warning("Failed updating workspace manifest.json: %s", e)
        return res

    def delete_workspace(self, workspace_id: str, delete_files: bool = False) -> bool:
        if workspace_id == "default":
            return False
        # If deleting the active workspace, switch to default first
        active = self.get_active_workspace()
        if active and active.get("id") == workspace_id:
            self.set_active_workspace("default")

        success = self.db.delete_workspace(workspace_id)
        if success and delete_files:
            ws_root = self.base_dir / workspace_id
            if ws_root.exists() and ws_root.is_dir():
                shutil.rmtree(ws_root, ignore_errors=True)
            # Remove custom network configs associated with this workspace
            try:
                reg_dir = data_dir() / "networks" / "registry"
                if reg_dir.exists():
                    for p in reg_dir.glob("*.json"):
                        try:
                            cfg = json.loads(p.read_text(encoding="utf-8"))
                            if cfg.get("workspace_id") == workspace_id:
                                p.unlink(missing_ok=True)
                        except Exception:
                            pass
            except Exception as e:
                logger.warning("Could not clean up network files for workspace %s: %s", workspace_id, e)
        return success

    def export_workspace(self, workspace_id: str) -> Dict[str, Any]:
        ws = self.db.get_workspace(workspace_id)
        if not ws:
            raise ValueError(f"Workspace '{workspace_id}' not found.")
        paths = self.get_workspace_paths(workspace_id)

        # Collect checkpoints and datasets metadata
        checkpoints = self.db.list_checkpoints(limit=100)
        ws_checkpoints = [c for c in checkpoints if c.get("workspace_id") == workspace_id]

        datasets = self.db.list_datasets()
        ws_datasets = [d for d in datasets if d.get("workspace_id") == workspace_id]

        runs = self.db.list_runs(limit=100)
        ws_runs = [r for r in runs if r.get("workspace_id") == workspace_id]

        return {
            "schema_version": "1.0",
            "workspace": ws,
            "export_timestamp": time.time(),
            "checkpoints_count": len(ws_checkpoints),
            "datasets_count": len(ws_datasets),
            "runs_count": len(ws_runs),
            "manifest": {
                "checkpoints": ws_checkpoints,
                "datasets": ws_datasets,
                "runs": ws_runs,
            },
        }

    def import_workspace(self, export_data: Dict[str, Any]) -> Dict[str, Any]:
        ws_data = export_data.get("workspace")
        if not ws_data or "name" not in ws_data:
            raise ValueError("Invalid workspace export payload.")

        raw_id = ws_data.get("id", ws_data["name"])
        ws_id = self._sanitize_id(raw_id)
        if ws_id == "default":
            ws_id = f"imported_{int(time.time())}"

        created = self.create_workspace(
            name=ws_data["name"],
            workspace_id=ws_id,
            description=ws_data.get("description", "Imported workspace"),
            config=ws_data.get("config", {}),
        )
        return created
