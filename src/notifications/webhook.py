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
        discord_mention: Optional[str] = None,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        self.discord_webhook_url = (discord_webhook_url or "").strip()
        self.discord_mention = (discord_mention or "").strip()
        self.telegram_bot_token = (telegram_bot_token or "").strip()
        self.telegram_chat_id = (telegram_chat_id or "").strip()
        self.enabled = enabled

    def update_config(
        self,
        discord_webhook_url: Optional[str] = None,
        discord_mention: Optional[str] = None,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        if discord_webhook_url is not None:
            self.discord_webhook_url = discord_webhook_url.strip()
        if discord_mention is not None:
            self.discord_mention = discord_mention.strip()
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
        mention: Optional[str] = None,
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
        mention_str = (mention or "").strip()
        if mention_str:
            payload["content"] = mention_str

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
        include_mention: bool = False,
    ) -> None:
        """Sends notification to configured Discord and Telegram targets in background.
        Mentions/pings are only included if include_mention is True.
        """
        if not self.enabled:
            return

        mention = self.discord_mention if include_mention else None

        def task():
            if self.discord_webhook_url:
                self._send_discord_sync(
                    self.discord_webhook_url,
                    title,
                    message,
                    level=level,
                    fields=fields,
                    mention=mention,
                )
            if self.telegram_bot_token and self.telegram_chat_id:
                self._send_telegram_sync(
                    self.telegram_bot_token,
                    self.telegram_chat_id,
                    title,
                    message,
                    level=level,
                    fields=fields,
                )

        self._dispatch_async(task)

    def notify_live_progress(
        self,
        run_id: str,
        step: int,
        total_steps: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        tok_per_sec: Optional[float] = None,
        tokens_seen: Optional[int] = None,
        vram_used_gb: Optional[float] = None,
        muon_lr: Optional[float] = None,
        adamw_lr: Optional[float] = None,
        grad_norm: Optional[float] = None,
        eta_s: Optional[float] = None,
        elapsed_s: Optional[float] = None,
    ) -> None:
        """Sends rich live training progress update without pinging."""
        pct = (step / max(1, total_steps)) * 100.0 if total_steps > 0 else 0.0
        filled_len = min(10, max(0, int(round(pct / 10.0))))
        bar = "█" * filled_len + "░" * (10 - filled_len)

        def _fmt_dur(secs: Optional[float]) -> str:
            if secs is None or secs < 0:
                return "--:--"
            m, s = divmod(int(secs), 60)
            h, m = divmod(m, 60)
            return f"{h}h {m}m" if h > 0 else f"{m}m {s}s"

        fields = [
            {"name": "Progress", "value": f"[{bar}] **{pct:.1f}%** (`{step:,} / {total_steps:,}`)", "inline": False},
            {"name": "Train Loss", "value": f"`{train_loss:.4f}`", "inline": True},
        ]
        if val_loss is not None:
            fields.append({"name": "Val Loss", "value": f"`{val_loss:.4f}`", "inline": True})
        if tok_per_sec is not None:
            fields.append({"name": "Throughput", "value": f"`{tok_per_sec:,.0f} tok/s`", "inline": True})
        if tokens_seen is not None:
            fields.append({"name": "Tokens Seen", "value": f"`{tokens_seen:,}`", "inline": True})
        if vram_used_gb is not None:
            fields.append({"name": "VRAM Used", "value": f"`{vram_used_gb:.2f} GB`", "inline": True})
        if grad_norm is not None:
            fields.append({"name": "Grad Norm", "value": f"`{grad_norm:.2f}`", "inline": True})
        if muon_lr is not None or adamw_lr is not None:
            lr_str = f"M: `{muon_lr:.1e}`" if muon_lr else ""
            if adamw_lr:
                lr_str += f" | A: `{adamw_lr:.1e}`"
            fields.append({"name": "Learning Rate", "value": lr_str, "inline": True})

        timing_str = f"Elapsed: {_fmt_dur(elapsed_s)}"
        if eta_s is not None and eta_s > 0:
            timing_str += f" | ETA: {_fmt_dur(eta_s)}"
        fields.append({"name": "Timing", "value": timing_str, "inline": False})

        self.send_notification(
            title=f"📊 Live Training Progress • Step {step:,}",
            message=f"Run `{run_id}` training snapshot at step **{step:,}** of {total_steps:,}.",
            level="INFO",
            fields=fields,
            include_mention=False,  # Never ping on periodic live updates
        )

    def notify_run_started(
        self,
        run_id: str,
        model_name: str,
        param_count: Optional[int] = None,
        dataset_name: Optional[str] = None,
        target_steps: Optional[int] = None,
        target_epochs: Optional[int] = None,
        duration_mode: str = "steps",
        batch_size: Optional[int] = None,
        muon_lr: Optional[float] = None,
        adamw_lr: Optional[float] = None,
        device_name: Optional[str] = None,
        resumed_from: Optional[str] = None,
    ) -> None:
        """Fires immediately when training starts (NO PING)."""
        fields = [
            {"name": "Run ID", "value": f"`{run_id}`", "inline": True},
            {"name": "Model", "value": f"**{model_name}**" + (f" ({param_count/1e6:.1f}M)" if param_count else ""), "inline": True},
        ]
        if dataset_name:
            fields.append({"name": "Dataset", "value": f"`{dataset_name}`", "inline": True})
        if duration_mode == "epochs" and target_epochs:
            fields.append({"name": "Target", "value": f"{target_epochs} Epochs", "inline": True})
        elif target_steps:
            fields.append({"name": "Target", "value": f"{target_steps:,} Steps", "inline": True})
        if batch_size:
            fields.append({"name": "Batch Size", "value": f"{batch_size} seqs", "inline": True})
        if muon_lr or adamw_lr:
            lr_s = f"M: `{muon_lr:.1e}`" if muon_lr else ""
            if adamw_lr:
                lr_s += f" | A: `{adamw_lr:.1e}`"
            fields.append({"name": "Learning Rate", "value": lr_s, "inline": True})
        if device_name:
            fields.append({"name": "Device", "value": f"`{device_name}`", "inline": True})
        if resumed_from:
            fields.append({"name": "Mode", "value": f"💾 Resumed from `{resumed_from}`", "inline": False})
        else:
            fields.append({"name": "Mode", "value": "🌱 **Fresh Training (From Scratch)**", "inline": True})

        self.send_notification(
            title="🚀 Training Run Started",
            message=f"Training initialized for run `{run_id}` with **{model_name}**.",
            level="INFO",
            fields=fields,
            include_mention=False,  # NO PING on run start
        )

    def notify_run_stopped(
        self,
        run_id: str,
        step: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        tokens_seen: Optional[int] = None,
        elapsed_sec: Optional[float] = None,
        snapshot_path: Optional[str] = None,
    ) -> None:
        """Fires when training is stopped manually (PINGS configured target)."""
        fields = [
            {"name": "Stopped Step", "value": f"**{step:,}**", "inline": True},
            {"name": "Run ID", "value": f"`{run_id}`", "inline": True},
        ]
        if train_loss is not None:
            fields.append({"name": "Train Loss", "value": f"`{train_loss:.4f}`", "inline": True})
        if val_loss is not None:
            fields.append({"name": "Val Loss", "value": f"`{val_loss:.4f}`", "inline": True})
        if tokens_seen is not None:
            fields.append({"name": "Tokens Seen", "value": f"`{tokens_seen:,}`", "inline": True})
        if elapsed_sec is not None:
            fields.append({"name": "Active Time", "value": f"`{elapsed_sec / 60.0:.1f} min`", "inline": True})
        if snapshot_path:
            fields.append({"name": "Saved Snapshot", "value": f"`{snapshot_path}`", "inline": False})

        self.send_notification(
            title="🛑 Training Run Stopped",
            message=f"Training session for run `{run_id}` was stopped at step **{step:,}**.",
            level="WARNING",
            fields=fields,
            include_mention=True,  # PING on run stop (finish)
        )

    def notify_run_completed(
        self,
        run_id: str,
        step: int,
        final_loss: Optional[float] = None,
        best_val_loss: Optional[float] = None,
        tokens_seen: Optional[int] = None,
        elapsed_sec: Optional[float] = None,
        peak_vram_gb: Optional[float] = None,
    ) -> None:
        """Fires when training session finishes successfully (PINGS configured target)."""
        fields = [
            {"name": "Final Step", "value": f"**{step:,}**", "inline": True},
            {"name": "Run ID", "value": f"`{run_id}`", "inline": True},
        ]
        if final_loss is not None:
            fields.append({"name": "Final Loss", "value": f"`{final_loss:.4f}`", "inline": True})
        if best_val_loss is not None:
            fields.append({"name": "Best Val Loss", "value": f"`{best_val_loss:.4f}`", "inline": True})
        if tokens_seen is not None:
            fields.append({"name": "Total Tokens", "value": f"`{tokens_seen:,}`", "inline": True})
        if elapsed_sec is not None:
            fields.append({"name": "Total Time", "value": f"`{elapsed_sec / 60.0:.1f} min`", "inline": True})
        if peak_vram_gb is not None:
            fields.append({"name": "Peak VRAM", "value": f"`{peak_vram_gb:.2f} GB`", "inline": True})

        self.send_notification(
            title="🏁 Training Run Completed Successfully! 🎉",
            message=f"Training session for run `{run_id}` completed successfully at step **{step:,}**.",
            level="SUCCESS",
            fields=fields,
            include_mention=True,  # PING on finish!
        )

    def notify_run_failed(
        self,
        run_id: str,
        step: int,
        error_message: str,
    ) -> None:
        """Fires when training run encounters an unrecoverable failure (PINGS configured target)."""
        fields = [
            {"name": "Step", "value": f"**{step:,}**", "inline": True},
            {"name": "Run ID", "value": f"`{run_id}`", "inline": True},
            {"name": "Error Details", "value": f"```{error_message[:400]}```", "inline": False},
        ]
        self.send_notification(
            title="🚨 Training Run Aborted / Failed!",
            message=f"Training for run `{run_id}` halted due to error.",
            level="CRITICAL",
            fields=fields,
            include_mention=True,  # PING on failure!
        )

    def notify_best_checkpoint(
        self,
        step: int,
        val_loss: float,
        checkpoint_path: str,
        run_id: Optional[str] = None,
        train_loss: Optional[float] = None,
        tokens_seen: Optional[int] = None,
    ) -> None:
        """Fires when a new lowest validation loss checkpoint is saved (NO PING)."""
        fields = [
            {"name": "Best Val Loss", "value": f"**{val_loss:.4f}**", "inline": True},
            {"name": "Step", "value": f"`{step:,}`", "inline": True},
        ]
        if train_loss is not None:
            fields.append({"name": "Train Loss", "value": f"`{train_loss:.4f}`", "inline": True})
        if tokens_seen is not None:
            fields.append({"name": "Tokens Seen", "value": f"`{tokens_seen:,}`", "inline": True})
        fields.append({"name": "Checkpoint File", "value": f"`{checkpoint_path}`", "inline": False})
        if run_id:
            fields.append({"name": "Run ID", "value": f"`{run_id}`", "inline": True})

        self.send_notification(
            title="🏆 New Best Checkpoint Achieved!",
            message=f"New optimal validation loss reached: **{val_loss:.4f}** at step **{step:,}**.",
            level="BEST",
            fields=fields,
            include_mention=False,  # NO PING for regular best checkpoint
        )

    def notify_alert(
        self,
        category: str,
        level: str,
        message: str,
        step: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Fires on loss spikes, OOM warnings, or critical hardware/gradient anomalies (PINGS configured target)."""
        fields = [
            {"name": "Category", "value": f"`{category}`", "inline": True},
            {"name": "Severity", "value": f"**{level.upper()}**", "inline": True},
        ]
        if step is not None:
            fields.append({"name": "Step", "value": f"`{step:,}`", "inline": True},)
        if run_id:
            fields.append({"name": "Run ID", "value": f"`{run_id}`", "inline": True})

        title = "⚠️ Training Alert: " + category.replace("_", " ").title()
        self.send_notification(
            title=title,
            message=message,
            level="CRITICAL" if level.upper() in ("CRITICAL", "ERROR") else "WARNING",
            fields=fields,
            include_mention=True,  # PING on critical alerts!
        )

    def test_notification(
        self,
        discord_url: Optional[str] = None,
        discord_mention: Optional[str] = None,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronously tests notification channels and returns detailed results."""
        results = {"discord": False, "telegram": False, "errors": []}
        d_url = discord_url if discord_url is not None else self.discord_webhook_url
        d_mention = discord_mention if discord_mention is not None else self.discord_mention
        t_token = telegram_token if telegram_token is not None else self.telegram_bot_token
        t_chat = telegram_chat_id if telegram_chat_id is not None else self.telegram_chat_id

        fields = [
            {"name": "Source", "value": "Test Button in Settings"},
            {"name": "Timestamp", "value": time.strftime("%Y-%m-%d %H:%M:%S")},
        ]
        if d_mention:
            fields.append({"name": "Mention Target", "value": d_mention})

        if d_url:
            success = self._send_discord_sync(
                d_url,
                title="CataAI Ultra - Discord Webhook Connected! 🚀",
                description="Test notification successfully delivered from CataAI Ultra ML Training Studio.",
                level="SUCCESS",
                fields=fields,
                mention=d_mention,
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
                discord_mention=settings.get("discord_mention"),
                telegram_bot_token=settings.get("telegram_bot_token"),
                telegram_chat_id=settings.get("telegram_chat_id"),
                enabled=bool(settings.get("enable_webhooks", True)),
            )
        except Exception as e:
            logger.debug("Could not read webhook settings from DB: %s", e)

    return _GLOBAL_NOTIFIER
