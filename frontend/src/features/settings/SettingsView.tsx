"use client";

import React, { useState, useEffect } from "react";
import { api } from "@/services/api";
import { Alert } from "@/types/api";
import { DataTable, Column } from "@/components/ui/DataTable";
import { Button } from "@/components/ui/Button";
import { InlineAlert } from "@/components/ui/InlineAlert";
import {
  Settings,
  Cpu,
  Save,
  CheckCircle2,
  Shield,
  Bell,
  RefreshCw,
  Send,
  Radio,
  Check,
  AlertCircle,
  Download,
  Upload,
} from "lucide-react";
import { cn, formatDateTime } from "@/lib/utils";

export const SettingsView: React.FC = () => {
  const [settings, setSettings] = useState<Record<string, any>>({
    theme: "minimal-dark-grayscale",
    density: "compact",
    gpu_poll_interval_ms: 2000,
    vram_ceiling_mb: 5000,
    temp_threshold_c: 80,
    max_checkpoints_kept: 4,
    default_micro_batch_size: 4,
    default_seq_len: 256,
  });

  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [secretDrafts, setSecretDrafts] = useState({
    discord_webhook_url: "",
    telegram_bot_token: "",
  });
  const [secretsConfigured, setSecretsConfigured] = useState({
    discord_webhook_url: false,
    telegram_bot_token: false,
  });
  const [activeWorkspaceId, setActiveWorkspaceId] = useState("default");
  const [isExporting, setIsExporting] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isTestingWebhook, setIsTestingWebhook] = useState(false);
  const [testResult, setTestResult] = useState<{
    discord: boolean;
    telegram: boolean;
    errors: string[];
  } | null>(null);

  const fetchSettingsAndAlerts = async () => {
    try {
      const [s, a, ws] = await Promise.all([
        api.getSettings(),
        api.getAlerts(undefined, 50),
        api.getActiveWorkspace().catch(() => null),
      ]);
      if (ws?.id) setActiveWorkspaceId(ws.id);
      if (s && Object.keys(s).length > 0) {
        setSecretsConfigured({
          discord_webhook_url: Boolean(s.discord_webhook_url_configured),
          telegram_bot_token: Boolean(s.telegram_bot_token_configured),
        });
        const { discord_webhook_url, telegram_bot_token, ...safeSettings } = s;
        setSettings((prev) => ({ ...prev, ...safeSettings }));
      }
      setAlerts(a);
      setLoadError(null);
    } catch (e: any) {
      console.error(e);
      setLoadError(e?.message || String(e));
    }
  };

  useEffect(() => {
    fetchSettingsAndAlerts();
  }, []);

  const secretPayload = () => {
    const payload: Record<string, string> = {};
    if (secretDrafts.discord_webhook_url) {
      payload.discord_webhook_url = secretDrafts.discord_webhook_url;
    }
    if (secretDrafts.telegram_bot_token) {
      payload.telegram_bot_token = secretDrafts.telegram_bot_token;
    }
    return payload;
  };

  const handleClearSecret = async (field: "discord_webhook_url" | "telegram_bot_token") => {
    try {
      await api.saveSettings({ [field]: "" });
      setSecretsConfigured((prev) => ({ ...prev, [field]: false }));
      setSecretDrafts((prev) => ({ ...prev, [field]: "" }));
      setFeedback("Stored credential removed.");
    } catch (e: any) {
      alert(`Could not clear credential: ${e.message}`);
    }
  };

  const handleExportWorkspace = async () => {
    setIsExporting(true);
    try {
      const blob = await api.exportWorkspaceArchive(activeWorkspaceId);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${activeWorkspaceId}-workspace.zip`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      setFeedback("Workspace archive downloaded (manifest + checksums included).");
    } catch (e: any) {
      alert(`Export failed: ${e.message}`);
    } finally {
      setIsExporting(false);
    }
  };

  const handleImportWorkspace = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setIsImporting(true);
    try {
      const res = await api.importWorkspaceArchive(file);
      setFeedback(
        `Imported workspace '${res.workspace?.name}' with ${res.runs_imported} runs and ${res.files_imported} files.`,
      );
    } catch (e: any) {
      alert(`Import failed: ${e.message}`);
    } finally {
      setIsImporting(false);
    }
  };

  const handleTestNotification = async () => {
    setIsTestingWebhook(true);
    setTestResult(null);
    try {
      await api.saveSettings({ ...settings, ...secretPayload() });
      setSecretsConfigured((prev) => ({
        discord_webhook_url: prev.discord_webhook_url || Boolean(secretDrafts.discord_webhook_url),
        telegram_bot_token: prev.telegram_bot_token || Boolean(secretDrafts.telegram_bot_token),
      }));
      setFeedback("Settings saved to database.");
      setTimeout(() => setFeedback(null), 3000);

      const res = await api.testNotification({
        ...secretPayload(),
        discord_mention: settings.discord_mention,
        telegram_chat_id: settings.telegram_chat_id,
      });
      setTestResult(res);
    } catch (e: any) {
      setTestResult({
        discord: false,
        telegram: false,
        errors: [e?.message || "Test dispatch failed"],
      });
    } finally {
      setIsTestingWebhook(false);
    }
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await api.saveSettings({ ...settings, ...secretPayload() });
      setSecretsConfigured((prev) => ({
        discord_webhook_url: prev.discord_webhook_url || Boolean(secretDrafts.discord_webhook_url),
        telegram_bot_token: prev.telegram_bot_token || Boolean(secretDrafts.telegram_bot_token),
      }));
      setSecretDrafts({ discord_webhook_url: "", telegram_bot_token: "" });
      setFeedback("Settings successfully persisted to SQLite database.");
      setTimeout(() => setFeedback(null), 4000);
    } catch (e: any) {
      alert(`Save failed: ${e.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  const alertColumns: Column<Alert>[] = [
    {
      header: "LEVEL",
      accessorKey: "level",
      cell: (item) => (
        <span
          className={cn(
            "px-2 py-0.5 rounded text-[10px] font-medium uppercase font-mono border",
            item.level === "CRITICAL" || item.level === "ERROR"
              ? "bg-[#f87171]/10 text-[#f87171] border-[#f87171]/30"
              : item.level === "WARNING"
              ? "bg-[#facc15]/10 text-[#facc15] border-[#facc15]/30"
              : "bg-[#262626] text-[#A3A3A3] border-[#333333]"
          )}
        >
          {item.level}
        </span>
      ),
    },
    {
      header: "CATEGORY",
      accessorKey: "category",
      cell: (item) => <span className="font-medium text-[#F2F2F2]">{item.category}</span>,
    },
    {
      header: "STEP",
      accessorKey: "step",
      sortable: true,
      cell: (item) => (
        <span className="font-mono-num text-[#A3A3A3]">
          {item.step !== undefined ? `Step ${item.step}` : "—"}
        </span>
      ),
    },
    {
      header: "MESSAGE",
      accessorKey: "message",
      cell: (item) => <span className="text-[#F2F2F2] max-w-md truncate">{item.message}</span>,
    },
    {
      header: "TIMESTAMP",
      accessorKey: "created_at",
      cell: (item) => (
        <span className="text-[#737373] text-[10px]">
          {formatDateTime(item.created_at)}
        </span>
      ),
    },
  ];

  return (
    <div className="space-y-4 font-mono text-xs">
      {/* Header */}
      <div className="panel p-3.5 bg-[#202020] border-[#333333] flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <Settings className="w-4 h-4 text-[#A3A3A3]" />
          <div>
            <h3 className="font-semibold text-[#F2F2F2] uppercase tracking-wider text-xs">
              SYSTEM SETTINGS & HARDWARE MONITORING POLICY
            </h3>
            <p className="text-[11px] text-[#737373]">
              Configure VRAM limits, NVML polling frequency, checkpoint policies and workstation preferences
            </p>
          </div>
        </div>

        <Button
          variant="primary"
          size="sm"
          onClick={handleSave}
          loading={isSaving}
          icon={<Save className="w-3.5 h-3.5" />}
        >
          Save Settings
        </Button>
      </div>

      {loadError && (
        <InlineAlert
          tone="error"
          title="Could not load settings"
          message={`${loadError}. Stored credentials are never returned; resolve the error before saving.`}
          onDismiss={() => setLoadError(null)}
        />
      )}

      {feedback && (
        <InlineAlert
          tone="success"
          title="Configuration updated"
          message={feedback}
          onDismiss={() => setFeedback(null)}
        />
      )}

      {/* Settings Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Hardware & GPU Safety Policy */}
        <div className="panel p-4 bg-[#202020] border-[#333333] space-y-3">
          <h4 className="text-xs font-semibold text-[#F2F2F2] uppercase tracking-wider pb-2 border-b border-[#333333] flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-[#A3A3A3]" /> GPU SAFETY & NVML POLICIES
          </h4>

          <div>
            <label className="text-[10px] text-[#A3A3A3] uppercase font-medium block mb-1">
              VRAM Hard Ceiling (MB)
            </label>
            <input
              type="number"
              value={settings.vram_ceiling_mb}
              onChange={(e) =>
                setSettings({ ...settings, vram_ceiling_mb: parseInt(e.target.value) || 5000 })
              }
              className="w-full bg-[#1B1B1B] border border-[#333333] rounded px-2.5 py-1.5 text-[#F2F2F2] focus:outline-none focus:border-[#525252]"
            />
            <span className="text-[10px] text-[#737373] mt-1 block">
              Dynamic accelerator safe allocation ceiling. Triggers protective checkpoint or throttle alert if breached.
            </span>
          </div>

          <div>
            <label className="text-[10px] text-[#A3A3A3] uppercase font-medium block mb-1">
              Thermal Alarm Threshold (°C)
            </label>
            <input
              type="number"
              value={settings.temp_threshold_c}
              onChange={(e) =>
                setSettings({ ...settings, temp_threshold_c: parseInt(e.target.value) || 80 })
              }
              className="w-full bg-[#1B1B1B] border border-[#333333] rounded px-2.5 py-1.5 text-[#F2F2F2] focus:outline-none focus:border-[#525252]"
            />
          </div>

          <div>
            <label className="text-[10px] text-[#A3A3A3] uppercase font-medium block mb-1">
              NVML Telemetry Polling Rate (ms)
            </label>
            <input
              type="number"
              value={settings.gpu_poll_interval_ms}
              onChange={(e) =>
                setSettings({ ...settings, gpu_poll_interval_ms: parseInt(e.target.value) || 2000 })
              }
              className="w-full bg-[#1B1B1B] border border-[#333333] rounded px-2.5 py-1.5 text-[#F2F2F2] focus:outline-none focus:border-[#525252]"
            />
          </div>
        </div>

        {/* Checkpoint & Retention Policies */}
        <div className="panel p-4 bg-[#202020] border-[#333333] space-y-3">
          <h4 className="text-xs font-semibold text-[#F2F2F2] uppercase tracking-wider pb-2 border-b border-[#333333] flex items-center gap-1.5">
            <Shield className="w-3.5 h-3.5 text-[#A3A3A3]" /> CHECKPOINT & PERSISTENCE
          </h4>

          <div>
            <label className="text-[10px] text-[#A3A3A3] uppercase font-medium block mb-1">
              Max Checkpoints Retained
            </label>
            <input
              type="number"
              value={settings.max_checkpoints_kept}
              onChange={(e) =>
                setSettings({ ...settings, max_checkpoints_kept: parseInt(e.target.value) || 4 })
              }
              className="w-full bg-[#1B1B1B] border border-[#333333] rounded px-2.5 py-1.5 text-[#F2F2F2] focus:outline-none focus:border-[#525252]"
            />
            <span className="text-[10px] text-[#737373] mt-1 block">
              Keeps a rolling window of recent step checkpoints plus the best validation checkpoint.
            </span>
          </div>

          <div>
            <label className="text-[10px] text-[#A3A3A3] uppercase font-medium block mb-1">
              Default Micro-Batch Size
            </label>
            <input
              type="number"
              value={settings.default_micro_batch_size}
              onChange={(e) =>
                setSettings({ ...settings, default_micro_batch_size: parseInt(e.target.value) || 4 })
              }
              className="w-full bg-[#1B1B1B] border border-[#333333] rounded px-2.5 py-1.5 text-[#F2F2F2] focus:outline-none focus:border-[#525252]"
            />
          </div>

          <div>
            <label className="text-[10px] text-[#A3A3A3] uppercase font-medium block mb-1">
              Default Sequence Length
            </label>
            <input
              type="number"
              value={settings.default_seq_len}
              onChange={(e) =>
                setSettings({ ...settings, default_seq_len: parseInt(e.target.value) || 256 })
              }
              className="w-full bg-[#1B1B1B] border border-[#333333] rounded px-2.5 py-1.5 text-[#F2F2F2] focus:outline-none focus:border-[#525252]"
            />
          </div>
        </div>
      </div>

      {/* Recovery & Diagnostics */}
      <div className="panel p-4 bg-[#202020] border-[#333333] space-y-3">
        <h4 className="text-xs font-semibold text-[#F2F2F2] uppercase tracking-wider pb-2 border-b border-[#333333] flex items-center gap-1.5">
          <Shield className="w-3.5 h-3.5 text-[#A3A3A3]" /> RECOVERY & DIAGNOSTICS
        </h4>
        <label className="flex items-start gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={settings.safe_mode === true}
            onChange={(e) => setSettings({ ...settings, safe_mode: e.target.checked })}
            className="mt-0.5 rounded bg-[#1B1B1B] border-[#333333] text-[#F2F2F2]"
          />
          <span>
            <span className="text-[11px] text-[#F2F2F2] block">Safe mode</span>
            <span className="text-[10px] text-[#737373] block">
              Disables plugin discovery and execution without changing the enabled set. Intended for
              diagnosing a broken plugin. Saved with the other settings.
            </span>
          </span>
        </label>

        <div className="border-t border-[#333333] pt-3 space-y-2">
          <span className="text-[11px] text-[#F2F2F2] block">Workspace backup</span>
          <p className="text-[10px] text-[#737373]">
            Export a ZIP with the workspace, runs, metrics, checkpoints, datasets, a versioned
            manifest and SHA-256 checksums. Import always creates a new workspace; it never
            overwrites an existing one.
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={handleExportWorkspace}
              loading={isExporting}
              icon={<Download className="w-3.5 h-3.5" />}
            >
              Export workspace
            </Button>
            <label className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded border border-[#333333] bg-[#1B1B1B] text-[11px] text-[#F2F2F2] hover:border-[#525252] cursor-pointer">
              <Upload className="w-3.5 h-3.5" />
              {isImporting ? "Importing…" : "Import archive"}
              <input
                type="file"
                accept=".zip,application/zip"
                onChange={handleImportWorkspace}
                disabled={isImporting}
                className="sr-only"
              />
            </label>
            <span className="text-[10px] text-[#737373]">Active: {activeWorkspaceId}</span>
          </div>
        </div>
      </div>

      {/* Real-Time Webhook Notifications (Discord & Telegram) */}
      <div className="panel p-4 bg-[#202020] border-[#333333] space-y-4">
        <div className="flex items-center justify-between pb-2 border-b border-[#333333]">
          <h4 className="text-xs font-semibold text-[#F2F2F2] uppercase tracking-wider flex items-center gap-1.5">
            <Radio className="w-3.5 h-3.5 text-[#A3A3A3]" /> DISCORD & TELEGRAM REAL-TIME NOTIFICATIONS
          </h4>
          <label className="flex items-center gap-2 cursor-pointer text-xs text-[#A3A3A3]">
            <input
              type="checkbox"
              checked={settings.enable_webhooks !== false}
              onChange={(e) =>
                setSettings({ ...settings, enable_webhooks: e.target.checked })
              }
              className="rounded bg-[#1B1B1B] border-[#333333] text-[#F2F2F2]"
            />
            <span>Enable Webhook Alerts</span>
          </label>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Discord Configuration */}
          <div className="space-y-3 p-3 bg-[#1B1B1B] rounded border border-[#2B2B2B]">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold text-[#F2F2F2] uppercase tracking-wider">
                Discord Webhook Integration
              </span>
              <span className="text-[9px] text-[#737373] font-mono">Real-time embeds</span>
            </div>

            <div>
              <label className="text-[10px] text-[#A3A3A3] uppercase font-medium block mb-1">
                Discord Webhook URL
              </label>
              <div className="flex items-center gap-2">
                <input
                  type="password"
                  placeholder={secretsConfigured.discord_webhook_url ? "Saved encrypted — enter a new URL to replace" : "https://discord.com/api/webhooks/..."}
                  value={secretDrafts.discord_webhook_url}
                  onChange={(e) =>
                    setSecretDrafts({ ...secretDrafts, discord_webhook_url: e.target.value })
                  }
                  className="w-full bg-[#141414] border border-[#333333] rounded px-2.5 py-1.5 text-[#F2F2F2] placeholder-[#525252] focus:outline-none focus:border-[#525252]"
                />
                {secretsConfigured.discord_webhook_url && (
                  <button
                    type="button"
                    onClick={() => handleClearSecret("discord_webhook_url")}
                    className="text-[10px] text-[#A3A3A3] hover:text-[#F2F2F2] underline shrink-0"
                  >
                    Clear
                  </button>
                )}
              </div>
              <span className="text-[10px] text-[#737373] mt-1 block">
                Stored with AES-256-GCM, key protected by Windows DPAPI. Never returned by the API.
              </span>
            </div>

            <div>
              <label className="text-[10px] text-[#A3A3A3] uppercase font-medium block mb-1">
                Discord Mention / Ping Target
              </label>
              <input
                type="text"
                placeholder="e.g. @everyone, @here, <@123456789>, or <@&987654321>"
                value={settings.discord_mention || ""}
                onChange={(e) =>
                  setSettings({ ...settings, discord_mention: e.target.value })
                }
                className="w-full bg-[#141414] border border-[#333333] rounded px-2.5 py-1.5 text-[#F2F2F2] placeholder-[#525252] focus:outline-none focus:border-[#525252]"
              />
              <span className="text-[10px] text-[#737373] mt-1 block">
                Who to ping on alert: <code className="text-[#A3A3A3]">@everyone</code>, <code className="text-[#A3A3A3]">@here</code>, user ID <code className="text-[#A3A3A3]">&lt;@USER_ID&gt;</code>, or role ID <code className="text-[#A3A3A3]">&lt;@&amp;ROLE_ID&gt;</code>.
              </span>
            </div>
          </div>

          {/* Telegram Configuration */}
          <div className="space-y-3 p-3 bg-[#1B1B1B] rounded border border-[#2B2B2B]">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold text-[#F2F2F2] uppercase tracking-wider">
                Telegram Bot Integration
              </span>
              <span className="text-[9px] text-[#737373] font-mono">Bot API messages</span>
            </div>

            <div>
              <label className="text-[10px] text-[#A3A3A3] uppercase font-medium block mb-1">
                Telegram Bot Token
              </label>
              <div className="flex items-center gap-2">
                <input
                  type="password"
                  placeholder={secretsConfigured.telegram_bot_token ? "Saved encrypted — enter a new token to replace" : "123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ"}
                  value={secretDrafts.telegram_bot_token}
                  onChange={(e) =>
                    setSecretDrafts({ ...secretDrafts, telegram_bot_token: e.target.value })
                  }
                  className="w-full bg-[#141414] border border-[#333333] rounded px-2.5 py-1.5 text-[#F2F2F2] placeholder-[#525252] focus:outline-none focus:border-[#525252]"
                />
                {secretsConfigured.telegram_bot_token && (
                  <button
                    type="button"
                    onClick={() => handleClearSecret("telegram_bot_token")}
                    className="text-[10px] text-[#A3A3A3] hover:text-[#F2F2F2] underline shrink-0"
                  >
                    Clear
                  </button>
                )}
              </div>
              <span className="text-[10px] text-[#737373] mt-1 block">
                Created via @BotFather. Stored encrypted and never echoed back.
              </span>
            </div>

            <div>
              <label className="text-[10px] text-[#A3A3A3] uppercase font-medium block mb-1">
                Telegram Chat / Group ID
              </label>
              <input
                type="text"
                placeholder="e.g. 123456789 or -100123456789"
                value={settings.telegram_chat_id || ""}
                onChange={(e) =>
                  setSettings({ ...settings, telegram_chat_id: e.target.value })
                }
                className="w-full bg-[#141414] border border-[#333333] rounded px-2.5 py-1.5 text-[#F2F2F2] placeholder-[#525252] focus:outline-none focus:border-[#525252]"
              />
              <span className="text-[10px] text-[#737373] mt-1 block">
                Your personal User ID or group Chat ID.
              </span>
            </div>
          </div>
        </div>

        {/* Test Ping Dispatcher */}
        <div className="pt-2 flex flex-wrap items-center justify-between gap-3 border-t border-[#2B2B2B]">
          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={handleTestNotification}
              loading={isTestingWebhook}
              icon={<Send className="w-3.5 h-3.5" />}
            >
              Test Notification Ping
            </Button>
            <span className="text-[10px] text-[#737373]">
              Dispatches a test payload with your mention configuration to configured channels.
            </span>
          </div>

          {testResult && (
            <div className="flex items-center gap-2 text-xs">
              {testResult.discord && (
                <span className="px-2 py-0.5 rounded bg-[#10B981]/10 text-[#10B981] border border-[#10B981]/30 flex items-center gap-1 font-mono text-[10px]">
                  <Check className="w-3 h-3" /> Discord OK
                </span>
              )}
              {testResult.telegram && (
                <span className="px-2 py-0.5 rounded bg-[#10B981]/10 text-[#10B981] border border-[#10B981]/30 flex items-center gap-1 font-mono text-[10px]">
                  <Check className="w-3 h-3" /> Telegram OK
                </span>
              )}
              {testResult.errors.length > 0 && (
                <span className="px-2 py-0.5 rounded bg-[#EF4444]/10 text-[#EF4444] border border-[#EF4444]/30 flex items-center gap-1 font-mono text-[10px]">
                  <AlertCircle className="w-3 h-3" /> {testResult.errors.join("; ")}
                </span>
              )}
            </div>
          )}
        </div>
      </div>

      {/* System Alert History Table */}
      <div className="space-y-2 pt-2">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-semibold text-[#F2F2F2] uppercase tracking-wider flex items-center gap-1.5">
            <Bell className="w-3.5 h-3.5 text-[#A3A3A3]" /> SYSTEM ALERTS & EVENT LOG ({alerts.length})
          </h3>
          <Button
            variant="secondary"
            size="sm"
            onClick={fetchSettingsAndAlerts}
            icon={<RefreshCw className="w-3 h-3" />}
          >
            Refresh
          </Button>
        </div>

        <DataTable
          data={alerts}
          columns={alertColumns}
          keyExtractor={(a) => a.id}
          searchPlaceholder="Search alerts by category or message..."
          emptyText="No system alerts registered."
        />
      </div>
    </div>
  );
};
