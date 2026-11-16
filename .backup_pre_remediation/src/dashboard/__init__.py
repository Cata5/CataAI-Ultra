"""
RoCanon-48.5M Real-Time Training Dashboard and Overfitting Cockpit.
"""

from .early_stopping import EarlyStoppingConfig, OverfittingDetector, ValidationRecord
from .server import create_app, start_dashboard_server
from .state import STATE, MetricPoint, TrainingStateManager

__all__ = [
    "STATE",
    "MetricPoint",
    "TrainingStateManager",
    "EarlyStoppingConfig",
    "OverfittingDetector",
    "ValidationRecord",
    "create_app",
    "start_dashboard_server",
]

