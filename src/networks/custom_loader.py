"""
src/networks/custom_loader.py
Custom PyTorch module loader and developer extension system.
Loads validated custom modules from `networks/custom/` without executing arbitrary browser input.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

CUSTOM_MODULES_DIR = Path("networks/custom")


class CustomModuleBase(nn.Module):
    """
    Stable base interface for custom developer-contributed PyTorch modules.
    Subclasses in `networks/custom/` must implement this interface.
    """
    module_name: str = "custom_module"
    description: str = "Custom developer module"
    parameter_schema: Dict[str, Any] = {}
    input_shape_requirements: str = "[Batch, Sequence, Channels/Dim]"
    output_shape_rules: str = "Same as input or projected dimension"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

    @classmethod
    def parameter_count(cls, dim: int, **kwargs: Any) -> int:
        """Analytical parameter count calculation for Network Studio."""
        return 0

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward()")


class CustomModuleRegistry:
    """Discovers and registers custom PyTorch modules from `networks/custom/`."""

    def __init__(self, directory: Path | str = CUSTOM_MODULES_DIR) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._registry: Dict[str, Type[CustomModuleBase]] = {}
        self.reload()

    def reload(self) -> None:
        """Scans the custom directory and registers valid subclasses."""
        self._registry.clear()
        for py_file in self.directory.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            try:
                mod_name = f"networks.custom.{py_file.stem}"
                spec = importlib.util.spec_from_file_location(mod_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (
                            inspect.isclass(attr)
                            and issubclass(attr, CustomModuleBase)
                            and attr is not CustomModuleBase
                        ):
                            self._registry[attr.module_name] = attr
                            logger.info("Registered custom module '%s' from %s", attr.module_name, py_file.name)
            except Exception as e:
                logger.error("Failed loading custom module from %s: %s", py_file.name, e)

    def get_module(self, name: str) -> Optional[Type[CustomModuleBase]]:
        return self._registry.get(name)

    def list_modules(self) -> List[Dict[str, Any]]:
        """Returns catalog of registered custom modules and their schemas."""
        modules = []
        for name, cls in self._registry.items():
            modules.append({
                "module_name": cls.module_name,
                "description": cls.description,
                "parameter_schema": cls.parameter_schema,
                "input_shape_requirements": cls.input_shape_requirements,
                "output_shape_rules": cls.output_shape_rules,
            })
        return modules

