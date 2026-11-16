"use client";

import React, { useState, useEffect } from "react";
import { api } from "@/services/api";
import { Alert } from "@/types/api";
import { DataTable, Column } from "@/components/ui/DataTable";
import {
  Settings,
  Cpu,
  Save,
  CheckCircle2,
  Shield,
  Bell,
  RefreshCw,
} from "lucide-react";
import { cn } from "@/lib/utils";

export const SettingsView: React.FC = () => {
  const [settings, setSettings] = useState<Record<string, any>>({
    theme: "sci-fi-midnight",
    density: "compact",
    gpu_poll_interval_ms: 2000,
    vram_ceiling_mb: 5000,
    temp_threshold_c: 80,
    max_checkpoints_kept: 4,
    default_micro_batch_size: 4,
    default_seq_len: 256,
  });

  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [isSaving, setIsSaving] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);

  const fetchSettingsAndAlerts = async () => {
    try {
      const [s, a] = await Promise.all([api.getSettings(), api.getAlerts(undefined, 50)]);
      if (s && Object.keys(s).length > 0) {
        setSettings((prev) => ({ ...prev, ...s }));
      }
      setAlerts(a);
    } catch (e: any) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchSettingsAndAlerts();
  }, []);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await api.saveSettings(settings);
      setFeedback("Settings successfully persisted to SQLite database!");
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
            "px-2 py-0.5 rounded text-[10px] font-bold uppercase",
            item.level === "CRITICAL" || item.level === "ERROR"
              ? "bg-rose-950/60 text-rose-400 border border-rose-800"
              : item.level === "WARNING"
              ? "bg-amber-950/60 text-amber-400 border border-amber-800"
              : "bg-cyan-950/60 text-cyan-400 border border-cyan-800"
          )}
        >
          {item.level}
        </span>
      ),
    },
    {
      header: "CATEGORY",
      accessorKey: "category",
      cell: (item) => <span className="font-bold text-slate-200">{item.category}</span>,
    },
    {
      header: "STEP",
      accessorKey: "step",
      sortable: true,
      cell: (item) => (
        <span className="font-mono-num text-slate-300">
          {item.step !== undefined ? `Step ${item.step}` : "—"}
        </span>
      ),
    },
    {
      header: "MESSAGE",
      accessorKey: "message",
      cell: (item) => <span className="text-slate-300 max-w-md truncate">{item.message}</span>,
    },
    {
      header: "TIMESTAMP",
      accessorKey: "created_at",
      cell: (item) => (
        <span className="text-slate-400 text-[10px]">
          {item.created_at ? new Date(item.created_at).toLocaleString() : "—"}
        </span>
      ),
    },
  ];

  return (
    <div className="space-y-4 font-mono text-xs">
      {/* Header */}
      <div className="glass-panel p-3.5 rounded-lg flex items-center justify-between border-l-4 border-l-cyan-400">
        <div className="flex items-center gap-2.5">
          <Settings className="w-5 h-5 text-cyan-400" />
          <div>
            <h3 className="font-bold text-slate-100 uppercase tracking-wider text-xs">
              SYSTEM SETTINGS & HARDWARE MONITORING POLICY
            </h3>
            <p className="text-[10px] text-slate-400">
              Configure VRAM limits, NVML polling frequency, checkpoint policies and user appearance
            </p>
          </div>
        </div>

        <button
          onClick={handleSave}
          disabled={isSaving}
          className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white font-bold transition-all shadow-[0_0_12px_rgba(0,240,255,0.3)]"
        >
          <Save className="w-3.5 h-3.5" />
          {isSaving ? "SAVING..." : "SAVE SETTINGS"}
        </button>
      </div>

      {feedback && (
        <div className="p-3 rounded-lg bg-emerald-950/80 border border-emerald-500/40 text-emerald-300 font-bold flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4" /> {feedback}
        </div>
      )}

      {/* Settings Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Hardware & GPU Safety Policy */}
        <div className="glass-panel p-4 rounded-lg space-y-3">
          <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider pb-2 border-b border-slate-800 flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" /> GPU SAFETY & NVML POLICIES
          </h4>

          <div>
            <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
              VRAM Hard Ceiling (MB)
            </label>
            <input
              type="number"
              value={settings.vram_ceiling_mb}
              onChange={(e) =>
                setSettings({ ...settings, vram_ceiling_mb: parseInt(e.target.value) || 5000 })
              }
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-cyan-400"
            />
            <span className="text-[10px] text-slate-400 mt-1 block">
              RTX 3060 Laptop (6GB VRAM) safe ceiling: 5,000 MB. Triggers automatic throttle alert if breached.
            </span>
          </div>

          <div>
            <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
              Thermal Alarm Threshold (°C)
            </label>
            <input
              type="number"
              value={settings.temp_threshold_c}
              onChange={(e) =>
                setSettings({ ...settings, temp_threshold_c: parseInt(e.target.value) || 80 })
              }
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-cyan-400"
            />
          </div>

          <div>
            <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
              NVML Telemetry Polling Rate (ms)
            </label>
            <input
              type="number"
              value={settings.gpu_poll_interval_ms}
              onChange={(e) =>
                setSettings({ ...settings, gpu_poll_interval_ms: parseInt(e.target.value) || 2000 })
              }
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-cyan-400"
            />
          </div>
        </div>

        {/* Checkpoint & Retention Policies */}
        <div className="glass-panel p-4 rounded-lg space-y-3">
          <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider pb-2 border-b border-slate-800 flex items-center gap-1.5">
            <Shield className="w-3.5 h-3.5 text-cyan-400" /> CHECKPOINT & PERSISTENCE
          </h4>

          <div>
            <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
              Max Checkpoints Retained
            </label>
            <input
              type="number"
              value={settings.max_checkpoints_kept}
              onChange={(e) =>
                setSettings({ ...settings, max_checkpoints_kept: parseInt(e.target.value) || 4 })
              }
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-cyan-400"
            />
            <span className="text-[10px] text-slate-400 mt-1 block">
              Keeps a rolling window of the 4 most recent step checkpoints plus the best validation checkpoint.
            </span>
          </div>

          <div>
            <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
              Default Micro-Batch Size
            </label>
            <input
              type="number"
              value={settings.default_micro_batch_size}
              onChange={(e) =>
                setSettings({ ...settings, default_micro_batch_size: parseInt(e.target.value) || 4 })
              }
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-cyan-400"
            />
          </div>

          <div>
            <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
              Default Sequence Length
            </label>
            <input
              type="number"
              value={settings.default_seq_len}
              onChange={(e) =>
                setSettings({ ...settings, default_seq_len: parseInt(e.target.value) || 256 })
              }
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-cyan-400"
            />
          </div>
        </div>
      </div>

      {/* System Alert History Table */}
      <div className="space-y-2 pt-2">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider flex items-center gap-1.5">
            <Bell className="w-3.5 h-3.5 text-cyan-400" /> SYSTEM ALERTS & EVENT LOG ({alerts.length})
          </h3>
          <button
            onClick={fetchSettingsAndAlerts}
            className="p-1.5 rounded bg-slate-900 border border-slate-800 text-slate-400 hover:text-cyan-400"
            title="Refresh Alerts"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
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
