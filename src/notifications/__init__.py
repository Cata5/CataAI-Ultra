"""
src/notifications package for RoCanon-48.5M Studio.
"""

from .webhook import WebhookNotifier, get_notifier

__all__ = ["WebhookNotifier", "get_notifier"]
