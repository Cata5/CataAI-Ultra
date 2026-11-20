"""
src/notifications/webhook.py
Real-time webhook notification dispatcher supporting Discord Webhooks
and Telegram Bot API messages for model training lifecycle events:
- Training run completion
- Out-of-memory, loss spikes, and critical alerts
- New best checkpoints achieved
Dispatches asynchronously in background threads with zero latency impact on training.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WebhookNotifier:
    """Non-blocking notification engine for Discord and Telegram webhooks."""

    def __init__(
        self,
        discord_webhook_url: Optional[str] = None,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        self.discord_webhook_url = (discord_webhook_url or "").strip()
        self.telegram_bot_token = (telegram_bot_token or "").strip()
        self.telegram_chat_id = (telegram_chat_id or "").strip()
        self.enabled = enabled

    def update_config(
        self,
        discord_webhook_url: Optional[str] = None,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        if discord_webhook_url is not None:
            self.discord_webhook_url = discord_webhook_url.strip()
        if telegram_bot_token is not None:
            self.telegram_bot_token = telegram_bot_token.strip()
        if telegram_chat_id is not None:
            self.telegram_chat_id = telegram_chat_id.strip()
        if enabled is not None:
            self.enabled = enabled

    def _dispatch_async(self, fn, *args, **kwargs) -> threading.Thread:
        """Executes notification in background daemon thread to never block training loops."""
        t = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
        t.start()
        return t

    def _send_discord_sync(
        self,
        webhook_url: str,
        title: str,
        description: str,
        level: str = "INFO",
        fields: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        if not webhook_url or not webhook_url.startswith("http"):
            return False

        # Colors: Green for SUCCESS/BEST, Red for CRITICAL/ERROR/SPIKE, Amber for WARNING, Cyan for INFO
        color_map = {
            "SUCCESS": 0x10B981,
            "BEST": 0x10B981,
            "CRITICAL": 0xEF4444,
            "ERROR": 0xEF4444,
            "SPIKE": 0xEF4444,
            "WARNING": 0xF59E0B,
            "INFO": 0x06B6D4,
        }
        color = color_map.get(level.upper(), 0x06B6D4)

        embed = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "footer": {"text": "CataAI Ultra • Training Cockpit"},
        }
        if fields:
            embed["fields"] = [
                {"name": str(f.get("name", "")), "value": str(f.get("value", "")), "inline": bool(f.get("inline", True))}
                for f in fields
            ]

        payload = {
            "username": "CataAI Cockpit",
            "avatar_url": "https://raw.githubusercontent.com/Cata5/CataAI-Ultra/main/frontend/public/favicon.ico",
            "embeds": [embed],
        }

        try:
            req = urllib.request.Request(
                webhook_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "CataAI-Ultra-Notifier/1.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                return resp.status in (200, 204)
        except Exception as e:
            logger.warning("Discord webhook dispatch failed: %s", e)
            return False

    def _send_telegram_sync(
        self,
        bot_token: str,
        chat_id: str,
        title: str,
        message: str,
        level: str = "INFO",
        fields: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        if not bot_token or not chat_id:
            return False

        emoji_map = {
            "SUCCESS": "✅",
            "BEST": "🏆",
            "CRITICAL": "🚨",
            "ERROR": "🛑",
            "SPIKE": "⚠️",
            "WARNING": "⚠️",
            "INFO": "ℹ️",
        }
        icon = emoji_map.get(level.upper(), "🤖")

        lines = [
            f"*{icon} {title}*",
            message,
        ]
        if fields:
            lines.append("")
            for f in fields:
                lines.append(f"• *{f.get('name', '')}*: `{f.get('value', '')}`")
        lines.append("\n_CataAI Ultra Cockpit_")

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": "\n".join(lines),
            "parse_mode": "Markdown",
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "CataAI-Ultra-Notifier/1.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                return resp.status == 200
        except Exception as e:
            logger.warning("Telegram notification dispatch failed: %s", e)
            return False

    def send_notification(
        self,
        title: str,
        message: str,
        level: str = "INFO",
        fields: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Sends notification to configured Discord and Telegram targets in background."""
        if not self.enabled:
            return

        def task():
            if self.discord_webhook_url:
                self._send_discord_sync(self.discord_webhook_url, title, message, level=level, fields=fields)
            if self.telegram_bot_token and self.telegram_chat_id:
                self._send_telegram_sync(self.telegram_bot_token, self.telegram_chat_id, title, message, level=level, fields=fields)

        self._dispatch_async(task)

    def notify_run_completed(
        self,
        run_id: str,
        step: int,
        final_loss: Optional[float] = None,
        tokens_seen: Optional[int] = None,
        elapsed_sec: Optional[float] = None,
    ) -> None:
        """Fires when training session finishes successfully."""
        fields = [
            {"name": "Final Step", "value": f"{step:,}"},
            {"name": "Run ID", "value": str(run_id)},
        ]
        if final_loss is not None:
            fields.append({"name": "Final Loss", "value": f"{final_loss:.4f}"})
        if tokens_seen is not None:
            fields.append({"name": "Tokens Seen", "value": f"{tokens_seen:,}"})
        if elapsed_sec is not None:
            fields.append({"name": "Total Time", "value": f"{elapsed_sec / 60.0:.1f} min"})

        self.send_notification(
            title="Training Run Completed Successfully! 🎉",
            message=f"Training session for run `{run_id}` completed target steps.",
            level="SUCCESS",
            fields=fields,
        )

    def notify_best_checkpoint(
        self,
        step: int,
        val_loss: float,
        checkpoint_path: str,
        run_id: Optional[str] = None,
    ) -> None:
        """Fires when a new lowest validation loss checkpoint is saved."""
        fields = [
            {"name": "Best Val Loss", "value": f"{val_loss:.4f}"},
            {"name": "Step", "value": f"{step:,}"},
            {"name": "Checkpoint File", "value": str(checkpoint_path)},
        ]
        if run_id:
            fields.append({"name": "Run ID", "value": str(run_id)})

        self.send_notification(
            title="New Best Checkpoint Achieved! 🏆",
            message=f"New optimal validation loss reached: **{val_loss:.4f}** at step {step:,}.",
            level="BEST",
            fields=fields,
        )

    def notify_alert(
        self,
        category: str,
        level: str,
        message: str,
        step: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Fires on loss spikes, OOM warnings, or critical hardware/gradient anomalies."""
        fields = [
            {"name": "Category", "value": str(category)},
            {"name": "Severity", "value": str(level).upper()},
        ]
        if step is not None:
            fields.append({"name": "Step", "value": f"{step:,}"})
        if run_id:
            fields.append({"name": "Run ID", "value": str(run_id)})

        title = "Training Alert: " + category.replace("_", " ").title()
        self.send_notification(
            title=title,
            message=message,
            level="CRITICAL" if level.upper() in ("CRITICAL", "ERROR") else "WARNING",
            fields=fields,
        )

    def test_notification(
        self,
        discord_url: Optional[str] = None,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronously tests notification channels and returns detailed results."""
        results = {"discord": False, "telegram": False, "errors": []}
        d_url = discord_url if discord_url is not None else self.discord_webhook_url
        t_token = telegram_token if telegram_token is not None else self.telegram_bot_token
        t_chat = telegram_chat_id if telegram_chat_id is not None else self.telegram_chat_id

        fields = [
            {"name": "Source", "value": "Test Button in Settings"},
            {"name": "Timestamp", "value": time.strftime("%Y-%m-%d %H:%M:%S")},
        ]

        if d_url:
            success = self._send_discord_sync(
                d_url,
                title="CataAI Ultra - Discord Webhook Connected! 🚀",
                description="Test notification successfully delivered from CataAI Ultra ML Training Studio.",
                level="SUCCESS",
                fields=fields,
            )
            results["discord"] = success
            if not success:
                results["errors"].append("Discord dispatch failed (check webhook URL).")

        if t_token and t_chat:
            success = self._send_telegram_sync(
                t_token,
                t_chat,
                title="CataAI Ultra - Telegram Connected! 🚀",
                message="Test notification successfully delivered from CataAI Ultra ML Training Studio.",
                level="SUCCESS",
                fields=fields,
            )
            results["telegram"] = success
            if not success:
                results["errors"].append("Telegram dispatch failed (check bot token & chat ID).")

        return results


# Global singleton instance
_GLOBAL_NOTIFIER: Optional[WebhookNotifier] = None


def get_notifier(db: Optional[Any] = None) -> WebhookNotifier:
    """Returns singleton WebhookNotifier populated with settings from database."""
    global _GLOBAL_NOTIFIER
    if _GLOBAL_NOTIFIER is None:
        _GLOBAL_NOTIFIER = WebhookNotifier()

    if db is not None:
        try:
            settings = db.get_settings()
            _GLOBAL_NOTIFIER.update_config(
                discord_webhook_url=settings.get("discord_webhook_url"),
                telegram_bot_token=settings.get("telegram_bot_token"),
                telegram_chat_id=settings.get("telegram_chat_id"),
                enabled=bool(settings.get("enable_webhooks", True)),
            )
        except Exception as e:
            logger.debug("Could not read webhook settings from DB: %s", e)

    return _GLOBAL_NOTIFIER
