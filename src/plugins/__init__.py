"""
src/plugins/__init__.py
"""
from .manager import PluginManager
from .manifest import PluginCategory, PluginManifest

__all__ = ["PluginManager", "PluginManifest", "PluginCategory"]
