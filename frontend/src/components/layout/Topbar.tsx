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

const TITLES: Record<string, { section: string; title: string }> = {
  "/": { section: "Workspace", title: "Overview" },
  "/studio": { section: "Development", title: "Architecture Studio" },
  "/training": { section: "Development", title: "Training Console" },
  "/datasets": { section: "Development", title: "Dataset Explorer" },
  "/evaluation": { section: "Development", title: "Evaluation Lab" },
  "/runs": { section: "Management", title: "Runs & Experiments" },
  "/settings": { section: "System", title: "Hardware & Settings" },
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

  const navMeta = TITLES[pathname] || { section: "Console", title: "Neural Workstation" };
  const effectiveState = currentState !== "IDLE" ? currentState : status?.state || "IDLE";

  return (
    <>
      <header className="h-13 min-h-[52px] border-b border-[#273548] bg-[#0E1420]/95 backdrop-blur-md px-4 flex items-center justify-between select-none z-20">
        {/* Left: Refined Breadcrumbs & Page Identity */}
        <div className="flex items-center gap-2 text-xs font-mono">
          <span className="text-[#5E7188] hover:text-[#91A0B4] transition-colors">
            {navMeta.section}
          </span>
          <span className="text-[#384C66]">/</span>
          <h1 className="font-semibold text-[#ECF3FC] tracking-wide flex items-center gap-2">
            {navMeta.title}
          </h1>
        </div>

        {/* Center: Linear/Raycast-grade Command Palette Trigger */}
        <button
          onClick={() => setIsCommandOpen(true)}
          className="hidden md:flex items-center gap-2.5 px-3 py-1.5 rounded-md bg-[#131C29] border border-[#273548] text-[#91A0B4] hover:border-[#384C66] hover:text-[#ECF3FC] transition-all font-mono text-xs w-72 justify-between shadow-inner"
        >
          <span className="flex items-center gap-2 truncate">
            <Search className="w-3.5 h-3.5 text-[#45CFFF]" />
            <span className="truncate">Search commands, models, datasets...</span>
          </span>
          <kbd className="px-1.5 py-0.5 rounded bg-[#182333] text-[10px] text-[#91A0B4] border border-[#273548]">
            Ctrl+K
          </kbd>
        </button>

        {/* Right: Telemetry, State & Connection Indicators */}
        <div className="flex items-center gap-2.5">
          {/* Worker Run State Pill */}
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-md bg-[#131C29] border border-[#273548] text-xs font-mono">
            <span className="text-[10px] text-[#5E7188] uppercase font-semibold">WORKER</span>
            <StatusIndicator state={effectiveState} size="sm" />
          </div>

          {/* Quick GPU Telemetry Pill */}
          {telemetry && (
            <div className="hidden sm:flex items-center gap-2 px-2.5 py-1 rounded-md bg-[#131C29] border border-[#273548] text-xs font-mono">
              <Cpu className="w-3.5 h-3.5 text-[#45CFFF]" />
              <span className="text-[#ECF3FC] font-bold font-mono-num">
                {telemetry.gpu_utilization_pct ?? telemetry.gpu_util_percent ?? 0}%
              </span>
              <span className="text-[#5E7188]">|</span>
              <span className="text-[#91A0B4] font-mono-num">
                {Math.round(telemetry.memory_used_mb ?? telemetry.used_vram_mb ?? 0)} MB
              </span>
            </div>
          )}

          {/* WebSocket Real-Time Link Status */}
          <div
            className="flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-mono border transition-colors"
            style={{
              borderColor: isConnected ? "rgba(53, 212, 154, 0.3)" : "rgba(255, 101, 122, 0.3)",
              backgroundColor: isConnected ? "rgba(53, 212, 154, 0.08)" : "rgba(255, 101, 122, 0.08)",
            }}
            title={isConnected ? "WebSocket Real-Time Link Active (8081)" : "WebSocket Link Disconnected"}
          >
            {isConnected ? (
              <Wifi className="w-3.5 h-3.5 text-[#35D49A]" />
            ) : (
              <WifiOff className="w-3.5 h-3.5 text-[#FF657A]" />
            )}
            <span className="text-[10px] uppercase font-bold text-[#ECF3FC]">
              {isConnected ? "LIVE" : "OFFLINE"}
            </span>
          </div>

          {/* Alerts Bell */}
          <div className="relative">
            <button
              onClick={() => setShowAlertsDropdown(!showAlertsDropdown)}
              className="p-1.5 rounded-md bg-[#131C29] border border-[#273548] hover:border-[#384C66] text-[#91A0B4] hover:text-[#ECF3FC] transition-colors relative"
              title="Recent Alerts"
            >
              <Bell className="w-4 h-4" />
              {recentAlerts.length > 0 && (
                <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-[#45CFFF] ring-2 ring-[#0E1420] animate-pulse" />
              )}
            </button>

            {showAlertsDropdown && (
              <div className="absolute right-0 mt-2 w-80 glass-panel-elevated border border-[#384C66] rounded-lg shadow-2xl p-3 z-50 animate-in fade-in duration-150">
                <div className="flex justify-between items-center pb-2 border-b border-[#273548] text-xs font-mono">
                  <span className="font-bold text-[#ECF3FC]">SYSTEM ALERTS</span>
                  <span className="text-[#91A0B4]">{recentAlerts.length} events</span>
                </div>
                <div className="max-h-60 overflow-y-auto space-y-1.5 mt-2">
                  {recentAlerts.length === 0 ? (
                    <div className="p-3 text-center text-xs font-mono text-[#5E7188]">
                      No system alerts recorded.
                    </div>
                  ) : (
                    recentAlerts.slice(0, 10).map((alert, idx) => (
                      <div
                        key={idx}
                        className="p-2 rounded bg-[#131C29] border border-[#273548] text-[11px] font-mono flex flex-col gap-0.5"
                      >
                        <div className="flex justify-between items-center">
                          <span
                            className={
                              alert.level === "CRITICAL" || alert.level === "ERROR"
                                ? "text-[#FF657A] font-bold"
                                : alert.level === "WARNING"
                                ? "text-[#F2B85B] font-bold"
                                : "text-[#45CFFF] font-bold"
                            }
                          >
                            [{alert.level}] {alert.category}
                          </span>
                          {alert.step !== undefined && (
                            <span className="text-[#5E7188] text-[10px]">
                              Step {alert.step}
                            </span>
                          )}
                        </div>
                        <span className="text-[#91A0B4]">{alert.message}</span>
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
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[#131C29] border border-[#273548] hover:border-[#384C66] text-[#91A0B4] font-mono text-xs transition-colors"
              title="Active Workspace"
            >
              <Briefcase className="w-3.5 h-3.5 text-[#45CFFF]" />
              <span className="max-w-[100px] truncate font-semibold text-[#ECF3FC]">
                {activeWorkspace?.name || "DEFAULT"}
              </span>
              <ChevronDown className="w-3 h-3 text-[#5E7188]" />
            </button>

            {showWorkspaceDropdown && (
              <div className="absolute right-0 mt-2 w-56 glass-panel-elevated border border-[#384C66] rounded-lg shadow-2xl p-2 z-50 animate-in fade-in duration-150">
                <div className="px-2 py-1 text-[10px] font-mono font-bold text-[#5E7188] border-b border-[#273548]">
                  WORKSPACES
                </div>
                <div className="max-h-48 overflow-y-auto space-y-1 mt-1 font-mono text-xs">
                  {workspaces.map((ws) => (
                    <button
                      key={ws.id}
                      onClick={() => handleSwitchWorkspace(ws.id)}
                      className={`w-full text-left px-2 py-1.5 rounded flex items-center justify-between transition-colors ${
                        activeWorkspace?.id === ws.id
                          ? "bg-[#182333] text-[#45CFFF] border border-[#45CFFF]/30"
                          : "text-[#91A0B4] hover:bg-[#182333] hover:text-[#ECF3FC]"
                      }`}
                    >
                      <span className="truncate">{ws.name}</span>
                      {activeWorkspace?.id === ws.id && (
                        <span className="w-1.5 h-1.5 rounded-full bg-[#45CFFF]" />
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
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[#131C29] border border-[#273548] hover:border-[#45CFFF]/50 text-[#45CFFF] font-mono text-xs transition-all shadow-[0_0_8px_rgba(69,207,255,0.1)]"
            title="Open Setup Wizard"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span className="hidden lg:inline font-semibold">Wizard</span>
          </button>

          {/* Language Toggle (EN / RO) */}
          <button
            onClick={() => setLocale(locale === "en" ? "ro" : "en")}
            className="flex items-center gap-1 px-2 py-1 rounded-md bg-[#131C29] border border-[#273548] hover:border-[#384C66] text-[#91A0B4] hover:text-[#ECF3FC] font-mono text-xs transition-colors"
            title="Toggle Language (English / Română)"
          >
            <Globe className="w-3.5 h-3.5 text-[#45CFFF]" />
            <span className="uppercase font-bold">{locale}</span>
          </button>
        </div>
      </header>

      <CommandPalette isOpen={isCommandOpen} onClose={() => setIsCommandOpen(false)} />
      <SetupWizardModal isOpen={isWizardOpen} onClose={() => setIsWizardOpen(false)} />
    </>
  );
};
