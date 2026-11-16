"""
src/plugins/manifest.py
Plugin manifest specification and validator for LLM Maker.
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class PluginCategory(str, enum.Enum):
    ARCHITECTURE = "architecture"
    LAYER = "layer"
    TASK = "task"
    DATASET_LOADER = "dataset_loader"
    TOKENIZER = "tokenizer"
    OPTIMIZER = "optimizer"
    SCHEDULER = "scheduler"
    HARDWARE_PROVIDER = "hardware_provider"
    EVAL_METRIC = "eval_metric"
    EXPORT_FORMAT = "export_format"


@dataclass
class PluginManifest:
    id: str
    name: str
    version: str
    description: str
    category: str
    author: str = "Community"
    min_llm_maker_version: str = "2.0.0"
    config_schema: Dict[str, Any] = field(default_factory=dict)
    supported_features: List[str] = field(default_factory=list)
    entry_point: Optional[str] = None
    enabled: bool = False

    def validate(self) -> Tuple[bool, str]:
        if not self.id or not isinstance(self.id, str):
            return False, "Plugin 'id' is required and must be a string."
        if not self.name:
            return False, "Plugin 'name' is required."
        if not self.version:
            return False, "Plugin 'version' is required."
        try:
            PluginCategory(self.category)
        except ValueError:
            valid_cats = [c.value for c in PluginCategory]
            return False, f"Invalid category '{self.category}'. Valid categories: {valid_cats}"
        return True, "Valid"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
