"use client";

import React, { useState, useEffect } from "react";
import { usePathname } from "next/navigation";
import {
  Search,
  Bell,
  Cpu,
  Wifi,
  WifiOff,
  Terminal,
  Sparkles,
  Globe,
  Briefcase,
  ChevronDown,
} from "lucide-react";
import { StatusIndicator } from "@/components/ui/StatusIndicator";
import { CommandPalette } from "@/components/ui/CommandPalette";
import { SetupWizardModal } from "@/components/onboarding/SetupWizardModal";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useHardwareTelemetry } from "@/hooks/useHardwareTelemetry";
import { useSystemStatus } from "@/hooks/useSystemStatus";
import { useTranslation } from "@/i18n";

const TITLES: Record<string, string> = {
  "/": "Command Center / Overview",
  "/studio": "Network Studio / Architecture Designer",
  "/training": "Experiment Control / Active Training",
  "/datasets": "Data Explorer / Dataset Manager",
  "/evaluation": "Evaluation Lab & Inference Engine",
  "/runs": "Runs, Checkpoints & Analytics",
  "/settings": "System Settings & Hardware Policy",
};

export const Topbar: React.FC = () => {
  const pathname = usePathname();
  const [isCommandOpen, setIsCommandOpen] = useState(false);
  const [isWizardOpen, setIsWizardOpen] = useState(false);
  const [showAlertsDropdown, setShowAlertsDropdown] = useState(false);
  const [showWorkspaceDropdown, setShowWorkspaceDropdown] = useState(false);
  const [workspaces, setWorkspaces] = useState<any[]>([]);
  const [activeWorkspace, setActiveWorkspace] = useState<any>(null);

  const { isConnected, currentState, recentAlerts } = useWebSocket();
  const { status } = useSystemStatus(4000);
  const { telemetry } = useHardwareTelemetry(3000);
  const { locale, setLocale, t } = useTranslation();

  // First-run onboarding check
  useEffect(() => {
    try {
      const completed = localStorage.getItem("llm_maker_onboarding_completed");
      if (!completed) {
        setIsWizardOpen(true);
      }
    } catch {}
  }, []);

  // Fetch workspaces
  const fetchWorkspaces = () => {
    fetch("/api/workspaces")
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data)) setWorkspaces(data);
      })
      .catch(() => {});
    fetch("/api/workspaces/current")
      .then((res) => res.json())
      .then((data) => {
        if (data && data.id) setActiveWorkspace(data);
      })
      .catch(() => {});
  };

  useEffect(() => {
    fetchWorkspaces();
  }, [isWizardOpen]);

  const handleSwitchWorkspace = async (id: string) => {
    try {
      await fetch("/api/workspaces/current", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workspace_id: id }),
      });
      fetchWorkspaces();
      setShowWorkspaceDropdown(false);
    } catch {}
  };

  const pageTitle = TITLES[pathname] || "LLM Maker Console";
  const effectiveState = currentState !== "IDLE" ? currentState : status?.state || "IDLE";

  return (
    <>
      <header className="h-14 border-b border-slate-800/80 glass-panel bg-[#060a14]/90 px-4 flex items-center justify-between select-none z-20">
        {/* Left: Breadcrumbs & Page Identity */}
        <div className="flex items-center gap-3">
          <Terminal className="w-4 h-4 text-cyan-400" />
          <h1 className="font-mono text-xs font-bold text-slate-100 tracking-wider uppercase">
            {pageTitle}
          </h1>
        </div>

        {/* Center: Command Palette Trigger */}
        <button
          onClick={() => setIsCommandOpen(true)}
          className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/80 border border-slate-800 text-slate-400 hover:border-cyan-400/50 hover:text-slate-300 transition-all font-mono text-xs w-64 justify-between"
        >
          <span className="flex items-center gap-1.5 truncate">
            <Search className="w-3.5 h-3.5 text-cyan-400" />
            <span>Search or jump...</span>
          </span>
          <kbd className="px-1.5 py-0.5 rounded bg-slate-800 text-[10px] text-slate-300 border border-slate-700">
            Ctrl+K
          </kbd>
        </button>

        {/* Right: Telemetry, State & Connection Indicators */}
        <div className="flex items-center gap-3">
          {/* Worker Run State */}
          <div className="flex items-center gap-2 px-2.5 py-1 rounded bg-slate-900/70 border border-slate-800">
            <span className="text-[10px] font-mono text-slate-400 uppercase">WORKER:</span>
            <StatusIndicator state={effectiveState} size="sm" />
          </div>

          {/* Quick GPU Pill */}
          {telemetry && (
            <div className="hidden sm:flex items-center gap-2 px-2.5 py-1 rounded bg-slate-900/70 border border-slate-800 text-xs font-mono">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <span className="text-slate-300 font-bold">
                {telemetry.gpu_utilization_pct ?? telemetry.gpu_util_percent ?? 0}%
              </span>
              <span className="text-slate-400">
                {Math.round(telemetry.memory_used_mb ?? telemetry.used_vram_mb ?? 0)} MB
              </span>
            </div>
          )}

          {/* WebSocket Link Status */}
          <div
            className="flex items-center gap-1.5 px-2 py-1 rounded text-xs font-mono border"
            style={{
              borderColor: isConnected ? "rgba(16, 185, 129, 0.3)" : "rgba(239, 68, 68, 0.3)",
              backgroundColor: isConnected ? "rgba(16, 185, 129, 0.08)" : "rgba(239, 68, 68, 0.08)",
            }}
            title={isConnected ? "WebSocket Real-Time Link Active (8081)" : "WebSocket Link Disconnected"}
          >
            {isConnected ? (
              <Wifi className="w-3.5 h-3.5 text-emerald-400" />
            ) : (
              <WifiOff className="w-3.5 h-3.5 text-rose-400" />
            )}
            <span className="text-[10px] uppercase font-bold text-slate-300">
              {isConnected ? "LIVE" : "OFFLINE"}
            </span>
          </div>

          {/* Alerts Bell */}
          <div className="relative">
            <button
              onClick={() => setShowAlertsDropdown(!showAlertsDropdown)}
              className="p-1.5 rounded-lg bg-slate-900/80 border border-slate-800 hover:border-cyan-400/50 text-slate-300 hover:text-cyan-400 transition-colors relative"
              title="Recent Alerts"
            >
              <Bell className="w-4 h-4" />
              {recentAlerts.length > 0 && (
                <span className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-cyan-400 ring-2 ring-[#060a14] animate-pulse" />
              )}
            </button>

            {showAlertsDropdown && (
              <div className="absolute right-0 mt-2 w-80 glass-panel border border-slate-700/80 rounded-lg shadow-2xl p-3 z-50 animate-in fade-in duration-150">
                <div className="flex justify-between items-center pb-2 border-b border-slate-800 text-xs font-mono">
                  <span className="font-bold text-slate-200">RECENT ALERTS</span>
                  <span className="text-slate-400">{recentAlerts.length} events</span>
                </div>
                <div className="max-h-60 overflow-y-auto space-y-1.5 mt-2">
                  {recentAlerts.length === 0 ? (
                    <div className="p-3 text-center text-xs font-mono text-slate-400">
                      No system alerts recorded.
                    </div>
                  ) : (
                    recentAlerts.slice(0, 10).map((alert, idx) => (
                      <div
                        key={idx}
                        className="p-2 rounded bg-slate-900/80 border border-slate-800 text-[11px] font-mono flex flex-col gap-0.5"
                      >
                        <div className="flex justify-between items-center">
                          <span
                            className={
                              alert.level === "CRITICAL" || alert.level === "ERROR"
                                ? "text-rose-400 font-bold"
                                : alert.level === "WARNING"
                                ? "text-amber-400 font-bold"
                                : "text-cyan-400 font-bold"
                            }
                          >
                            [{alert.level}] {alert.category}
                          </span>
                          {alert.step !== undefined && (
                            <span className="text-slate-400 text-[10px]">
                              Step {alert.step}
                            </span>
                          )}
                        </div>
                        <span className="text-slate-300">{alert.message}</span>
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Workspace Switcher */}
          <div className="relative hidden md:block">
            <button
              onClick={() => setShowWorkspaceDropdown(!showWorkspaceDropdown)}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900/80 border border-slate-800 hover:border-cyan-400/50 text-slate-300 font-mono text-xs transition-colors"
              title="Active Workspace"
            >
              <Briefcase className="w-3.5 h-3.5 text-cyan-400" />
              <span className="max-w-[100px] truncate font-bold text-slate-200">
                {activeWorkspace?.name || "DEFAULT"}
              </span>
              <ChevronDown className="w-3 h-3 text-slate-400" />
            </button>

            {showWorkspaceDropdown && (
              <div className="absolute right-0 mt-2 w-56 glass-panel border border-slate-700/80 rounded-lg shadow-2xl p-2 z-50 animate-in fade-in duration-150">
                <div className="px-2 py-1 text-[10px] font-mono font-bold text-slate-400 border-b border-slate-800">
                  WORKSPACES
                </div>
                <div className="max-h-48 overflow-y-auto space-y-1 mt-1 font-mono text-xs">
                  {workspaces.map((ws) => (
                    <button
                      key={ws.id}
                      onClick={() => handleSwitchWorkspace(ws.id)}
                      className={`w-full text-left px-2 py-1.5 rounded flex items-center justify-between transition-colors ${
                        activeWorkspace?.id === ws.id
                          ? "bg-cyan-950/60 text-cyan-300 border border-cyan-500/30"
                          : "text-slate-300 hover:bg-slate-800/60"
                      }`}
                    >
                      <span className="truncate">{ws.name}</span>
                      {activeWorkspace?.id === ws.id && (
                        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                      )}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Setup Wizard Button */}
          <button
            onClick={() => setIsWizardOpen(true)}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-cyan-950/40 border border-cyan-500/40 hover:bg-cyan-900/50 text-cyan-400 font-mono text-xs transition-all shadow-[0_0_8px_rgba(0,240,255,0.15)]"
            title="Open Hardware & Setup Wizard"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span className="hidden lg:inline font-bold">Wizard</span>
          </button>

          {/* Language Toggle (EN / RO) */}
          <button
            onClick={() => setLocale(locale === "en" ? "ro" : "en")}
            className="flex items-center gap-1 px-2 py-1 rounded-lg bg-slate-900/80 border border-slate-800 text-slate-300 hover:text-cyan-400 font-mono text-xs transition-colors"
            title="Toggle Language (English / Română)"
          >
            <Globe className="w-3.5 h-3.5 text-cyan-400" />
            <span className="uppercase font-bold">{locale}</span>
          </button>
        </div>
      </header>

      <CommandPalette isOpen={isCommandOpen} onClose={() => setIsCommandOpen(false)} />
      <SetupWizardModal isOpen={isWizardOpen} onClose={() => setIsWizardOpen(false)} />
    </>
  );
};
