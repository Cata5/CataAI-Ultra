"""
src/plugins/manager.py
Trusted Local Plugin Manager for LLM Maker.
Discovers local plugins from 'plugins/' directory, validates manifests,
and manages enable/disable state safely without executing untrusted remote code.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..db.database import StudioDatabase
from .manifest import PluginCategory, PluginManifest

logger = logging.getLogger(__name__)


class PluginManager:
    """Manages discovery, validation, and activation of trusted local plugins."""

    def __init__(self, plugins_dir: Path | str = "plugins", db: Optional[StudioDatabase] = None) -> None:
        self.plugins_dir = Path(plugins_dir)
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        self.db = db or StudioDatabase()
        self._ensure_sample_plugins()

    def _ensure_sample_plugins(self) -> None:
        """Seeds standard extension plugin templates if directory is empty."""
        sample_1 = self.plugins_dir / "vision_adapter"
        if not sample_1.exists():
            sample_1.mkdir(parents=True, exist_ok=True)
            manifest = {
                "id": "vision_adapter",
                "name": "Computer Vision & CNN Training Adapter",
                "version": "1.0.0",
                "description": "Enables image-based classification and CNN backbone training pipelines.",
                "category": "task",
                "author": "LLM Maker Core",
                "min_llm_maker_version": "2.0.0",
                "supported_features": ["image_folder", "resnet_backbone", "accuracy_metrics"],
                "config_schema": {
                    "image_size": {"type": "integer", "default": 224},
                    "channels": {"type": "integer", "default": 3},
                },
            }
            (sample_1 / "plugin.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        sample_2 = self.plugins_dir / "lion_optimizer"
        if not sample_2.exists():
            sample_2.mkdir(parents=True, exist_ok=True)
            manifest2 = {
                "id": "lion_optimizer",
                "name": "EvoLved Sign Momentum (Lion) Optimizer",
                "version": "1.0.0",
                "description": "Memory-efficient sign momentum optimizer discovered via program search.",
                "category": "optimizer",
                "author": "LLM Maker Core",
                "min_llm_maker_version": "2.0.0",
                "supported_features": ["fp32_master", "sign_momentum"],
                "config_schema": {
                    "lr": {"type": "number", "default": 1e-4},
                    "beta1": {"type": "number", "default": 0.9},
                    "beta2": {"type": "number", "default": 0.99},
                },
            }
            (sample_2 / "plugin.json").write_text(json.dumps(manifest2, indent=2), encoding="utf-8")

    def _get_enabled_set(self) -> set[str]:
        settings = self.db.get_settings()
        enabled = settings.get("enabled_plugins", ["vision_adapter", "lion_optimizer"])
        return set(enabled) if isinstance(enabled, list) else set()

    def _save_enabled_set(self, enabled: set[str]) -> None:
        self.db.save_settings({"enabled_plugins": sorted(list(enabled))})

    def discover_plugins(self) -> List[Dict[str, Any]]:
        """Scans plugins/ directory and returns list of validated plugin manifests."""
        plugins = []
        enabled_set = self._get_enabled_set()

        for item in self.plugins_dir.iterdir():
            if not item.is_dir():
                continue
            manifest_file = item / "plugin.json"
            if not manifest_file.exists():
                continue

            try:
                data = json.loads(manifest_file.read_text(encoding="utf-8"))
                manifest = PluginManifest(
                    id=data.get("id", item.name),
                    name=data.get("name", item.name),
                    version=data.get("version", "1.0.0"),
                    description=data.get("description", ""),
                    category=data.get("category", "layer"),
                    author=data.get("author", "Community"),
                    min_llm_maker_version=data.get("min_llm_maker_version", "2.0.0"),
                    config_schema=data.get("config_schema", {}),
                    supported_features=data.get("supported_features", []),
                    entry_point=data.get("entry_point"),
                    enabled=(data.get("id", item.name) in enabled_set),
                )
                valid, msg = manifest.validate()
                p_dict = manifest.to_dict()
                p_dict["is_valid"] = valid
                p_dict["validation_error"] = None if valid else msg
                p_dict["plugin_dir"] = str(item)
                plugins.append(p_dict)
            except Exception as e:
                logger.warning("Failed loading plugin at '%s': %s", item, e)
                plugins.append({
                    "id": item.name,
                    "name": item.name,
                    "is_valid": False,
                    "validation_error": str(e),
                    "enabled": False,
                    "plugin_dir": str(item),
                })

        return sorted(plugins, key=lambda x: x.get("name", ""))

    def set_plugin_enabled(self, plugin_id: str, enabled: bool) -> Tuple[bool, str]:
        """Enables or disables a registered local plugin."""
        plugins = {p["id"]: p for p in self.discover_plugins()}
        if plugin_id not in plugins:
            return False, f"Plugin '{plugin_id}' not found."

        if enabled and not plugins[plugin_id].get("is_valid", False):
            return False, f"Cannot enable invalid plugin: {plugins[plugin_id].get('validation_error')}"

        enabled_set = self._get_enabled_set()
        if enabled:
            enabled_set.add(plugin_id)
        else:
            enabled_set.discard(plugin_id)

        self._save_enabled_set(enabled_set)
        return True, f"Plugin '{plugin_id}' {'enabled' if enabled else 'disabled'} successfully."
