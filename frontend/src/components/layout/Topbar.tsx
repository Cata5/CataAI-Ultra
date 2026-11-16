"use client";

import React, { useState } from "react";
import { usePathname } from "next/navigation";
import {
  Search,
  Bell,
  Cpu,
  Wifi,
  WifiOff,
  Terminal,
} from "lucide-react";
import { StatusIndicator } from "@/components/ui/StatusIndicator";
import { CommandPalette } from "@/components/ui/CommandPalette";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useHardwareTelemetry } from "@/hooks/useHardwareTelemetry";
import { useSystemStatus } from "@/hooks/useSystemStatus";

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
  const [showAlertsDropdown, setShowAlertsDropdown] = useState(false);

  const { isConnected, currentState, recentAlerts } = useWebSocket();
  const { status } = useSystemStatus(4000);
  const { telemetry } = useHardwareTelemetry(3000);

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
        </div>
      </header>

      <CommandPalette isOpen={isCommandOpen} onClose={() => setIsCommandOpen(false)} />
    </>
  );
};
