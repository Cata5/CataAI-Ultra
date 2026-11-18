"""
src/networks/registry.py
Network Registry managing versioned JSON architecture specifications in `networks/registry/`.
Supports Create, Clone, Edit, Version, Export, Import, and Delete operations.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .estimator import ParameterCalculator
from .templates import ALL_TEMPLATES

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_DIR = Path("networks/registry")


class NetworkRegistry:
    """Manages versioned neural network definitions stored on disk."""

    def __init__(self, registry_dir: Path | str = DEFAULT_REGISTRY_DIR) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._seed_default_templates_if_empty()

    def _seed_default_templates_if_empty(self) -> None:
        """Seeds standard architecture templates if the registry has no files."""
        for tmpl in ALL_TEMPLATES:
            target = self.registry_dir / f"{tmpl['id']}.json"
            if not target.exists():
                try:
                    stats = ParameterCalculator.calculate(tmpl)
                    full_spec = tmpl.copy()
                    full_spec["total_parameters"] = stats["total_parameters"]
                    full_spec["created_at"] = time.time()
                    full_spec["updated_at"] = time.time()
                    with open(target, "w", encoding="utf-8") as f:
                        json.dump(full_spec, f, indent=2)
                except Exception as e:
                    logger.warning("Could not seed template %s: %s", tmpl['id'], e)

    def list_networks(self) -> List[Dict[str, Any]]:
        """Lists all registered networks on disk with calculated parameter counts."""
        results: List[Dict[str, Any]] = []
        for p in self.registry_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                stat = p.stat()
                params = data.get("total_parameters")
                if params is None:
                    try:
                        p_calc = ParameterCalculator.calculate(data)
                        params = p_calc["total_parameters"]
                    except Exception:
                        params = 0

                results.append({
                    "id": data.get("id", p.stem),
                    "name": data.get("name", p.stem),
                    "description": data.get("description", ""),
                    "architecture_family": data.get("architecture_family", "decoder_only"),
                    "architecture_type": data.get("architecture_type", data.get("architecture_family", "decoder_only")),
                    "version": data.get("version", "1.0"),
                    "d_model": data.get("d_model", 512),
                    "n_layers": data.get("n_layers", 8),
                    "total_parameters": params,
                    "total_params": params,
                    "trainable_parameters": params,
                    "trainable_params": params,
                    "parameters": {"total_params": params, "total_parameters": params, "trainable_params": params},
                    "supported_tasks": data.get("supported_tasks", ["causal_lm"]),
                    "file_size_bytes": stat.st_size,
                    "updated_at": data.get("updated_at", stat.st_mtime),
                })
            except Exception as e:
                logger.error("Failed to read network spec %s: %s", p.name, e)

        results.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
        return results

    def _validate_and_resolve_path(self, network_id: str) -> Path:
        """
        Validates network ID and ensures the target file strictly resides
        within the registry directory, preventing path traversal attacks.
        """
        if not network_id or not isinstance(network_id, str):
            raise ValueError("Network ID must be a non-empty string.")

        cleaned = network_id.strip()
        if not re.match(r"^[A-Za-z0-9_-]{1,64}$", cleaned):
            raise ValueError(
                f"Invalid network ID '{network_id}'. IDs must contain only letters, digits, underscores, and hyphens (max 64 chars)."
            )

        target_path = (self.registry_dir / f"{cleaned}.json").resolve()
        registry_root = self.registry_dir.resolve()
        if not target_path.is_relative_to(registry_root):
            raise ValueError(f"Security error: path traversal detected for network ID '{network_id}'")

        return target_path

    def get_network(self, network_id: str) -> Optional[Dict[str, Any]]:
        """Reads full network JSON definition with path traversal protection."""
        try:
            path = self._validate_and_resolve_path(network_id)
        except ValueError as e:
            logger.warning("Rejected invalid network ID in get_network: %s", e)
            return None

        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_network(self, network_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Atomically saves network definition to disk after validation and security checks."""
        target_path = self._validate_and_resolve_path(network_id)
        temp_path = self.registry_dir / f".tmp_{target_path.name}"

        # Validate configuration schema before saving
        from .validator import ArchitectureValidator
        errs = ArchitectureValidator.validate_config_schema(config)
        if errs:
            raise ValueError(f"Architecture validation failed: {'; '.join(errs)}")

        # Calculate exact parameter counts
        stats = ParameterCalculator.calculate(config)
        config["id"] = target_path.stem
        config["total_parameters"] = stats["total_parameters"]
        config["updated_at"] = time.time()
        if "created_at" not in config:
            config["created_at"] = time.time()

        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        os.replace(temp_path, target_path)
        logger.info("Saved network %s to registry (%d parameters)", config["id"], stats["total_parameters"])
        return config

    def clone_network(self, network_id: str, new_id: str, new_name: str) -> Optional[Dict[str, Any]]:
        """Clones an existing network under a new ID and name with path traversal validation."""
        self._validate_and_resolve_path(new_id)
        orig = self.get_network(network_id)
        if not orig:
            return None

        cloned = orig.copy()
        cloned["id"] = new_id
        cloned["name"] = new_name
        cloned["description"] = f"Cloned from {orig.get('name', network_id)}"
        cloned["version"] = "1.0"
        cloned["created_at"] = time.time()
        cloned["updated_at"] = time.time()

        return self.save_network(new_id, cloned)

    def delete_network(self, network_id: str) -> bool:
        """Deletes a network definition from the registry (RoCanon baseline is protected)."""
        try:
            path = self._validate_and_resolve_path(network_id)
        except ValueError as e:
            logger.warning("Rejected invalid network ID in delete_network: %s", e)
            return False

        if path.stem == "rocanon_48m":
            logger.warning("Cannot delete RoCanon-48.5M baseline architecture.")
            return False

        if path.exists():
            path.unlink()
            return True
        return False

