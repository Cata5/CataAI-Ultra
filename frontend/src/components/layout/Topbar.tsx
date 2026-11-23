"use client";

import React, { useState, useEffect } from "react";
import { usePathname } from "next/navigation";
import {
  Search,
  Bell,
  Cpu,
  Wifi,
  WifiOff,
  Briefcase,
  ChevronDown,
  Sun,
  Moon,
  Plus,
  X,
  Trash2,
} from "lucide-react";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { StatusIndicator } from "@/components/ui/StatusIndicator";
import { CommandPalette } from "@/components/ui/CommandPalette";
import { SetupWizardModal } from "@/components/onboarding/SetupWizardModal";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useHardwareTelemetry } from "@/hooks/useHardwareTelemetry";
import { useSystemStatus } from "@/hooks/useSystemStatus";
import { api } from "@/services/api";

const TITLES: Record<string, { section: string; title: string }> = {
  "/": { section: "Workspace", title: "Overview" },
  "/projects": { section: "Workspace", title: "Projects" },
  "/models": { section: "Workspace", title: "Models" },
  "/studio": { section: "Development", title: "Network Studio" },
  "/datasets": { section: "Development", title: "Datasets" },
  "/training": { section: "Development", title: "Training" },
  "/evaluation": { section: "Development", title: "Evaluation" },
  "/runs": { section: "Management", title: "Experiments" },
  "/checkpoints": { section: "Management", title: "Checkpoints" },
  "/analytics": { section: "Management", title: "Analytics" },
  "/hardware": { section: "System", title: "Hardware Telemetry" },
  "/settings": { section: "System", title: "Settings" },
};

export const Topbar: React.FC = () => {
  const pathname = usePathname();
  const [isCommandOpen, setIsCommandOpen] = useState(false);
  const [isWizardOpen, setIsWizardOpen] = useState(false);
  const [showAlertsDropdown, setShowAlertsDropdown] = useState(false);
  const [showWorkspaceDropdown, setShowWorkspaceDropdown] = useState(false);
  const [workspaces, setWorkspaces] = useState<any[]>([]);
  const [activeWorkspace, setActiveWorkspace] = useState<any>(null);
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [showCreateWsModal, setShowCreateWsModal] = useState(false);
  const [newWsName, setNewWsName] = useState("");
  const [isCreatingWs, setIsCreatingWs] = useState(false);
  const [workspaceToDelete, setWorkspaceToDelete] = useState<{ id: string; name: string } | null>(null);
  const [isDeletingWs, setIsDeletingWs] = useState(false);

  const { isConnected, currentState, recentAlerts } = useWebSocket();
  const { status } = useSystemStatus(4000);
  const { telemetry } = useHardwareTelemetry(3000);

  useEffect(() => {
    const saved = localStorage.getItem("llm_maker_theme") as "dark" | "light" | null;
    if (saved) {
      setTheme(saved);
      document.documentElement.setAttribute("data-theme", saved);
    }
  }, []);

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    localStorage.setItem("llm_maker_theme", next);
    document.documentElement.setAttribute("data-theme", next);
  };

  // Fetch workspaces
  const fetchWorkspaces = () => {
    api.listWorkspaces()
      .then((data) => {
        if (Array.isArray(data)) setWorkspaces(data);
      })
      .catch(() => {});
    api.getActiveWorkspace()
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
      await api.setActiveWorkspace(id);
      fetchWorkspaces();
      setShowWorkspaceDropdown(false);
      window.location.reload();
    } catch {}
  };

  const handleCreateWorkspace = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newWsName.trim() || isCreatingWs) return;
    setIsCreatingWs(true);
    try {
      const data = await api.createWorkspace(newWsName.trim());
      if (data && data.id) {
        await handleSwitchWorkspace(data.id);
        setShowCreateWsModal(false);
        setNewWsName("");
      }
    } catch (err: any) {
      alert(`Workspace creation failed: ${err.message}`);
    } finally {
      setIsCreatingWs(false);
    }
  };

  const handleDeleteWorkspace = async () => {
    if (!workspaceToDelete || isDeletingWs) return;
    setIsDeletingWs(true);
    try {
      await api.deleteWorkspace(workspaceToDelete.id);
      const wasActive = activeWorkspace?.id === workspaceToDelete.id;
      setWorkspaceToDelete(null);
      await fetchWorkspaces();
      if (wasActive) {
        window.location.reload();
      }
    } catch (err: any) {
      alert(`Workspace deletion failed: ${err.message}`);
    } finally {
      setIsDeletingWs(false);
    }
  };

  const navMeta = TITLES[pathname] || { section: "Workspace", title: "Overview" };
  const effectiveState = currentState !== "IDLE" ? currentState : status?.state || "IDLE";

  return (
    <>
      <header className="h-11 min-h-[44px] border-b border-[#333333] bg-[#171717] px-3.5 flex items-center justify-between select-none z-20">
        {/* Left: Breadcrumbs */}
        <div className="flex items-center gap-2 text-xs">
          <span className="text-[#737373]">
            {navMeta.section}
          </span>
          <span className="text-[#404040]">/</span>
          <h1 className="font-medium text-[#F2F2F2] tracking-tight">
            {navMeta.title}
          </h1>
        </div>

        {/* Center: Command Palette Trigger */}
        <button
          onClick={() => setIsCommandOpen(true)}
          className="hidden md:flex items-center gap-2 px-2.5 py-1 rounded bg-[#1B1B1B] border border-[#333333] text-[#737373] hover:border-[#404040] hover:text-[#A3A3A3] transition-colors text-xs w-64 justify-between"
        >
          <span className="flex items-center gap-1.5 truncate">
            <Search className="w-3.5 h-3.5" />
            <span className="truncate">Search commands, models...</span>
          </span>
          <kbd className="px-1 py-0.2 rounded bg-[#262626] text-[10px] text-[#A3A3A3] border border-[#333333]">
            Ctrl+K
          </kbd>
        </button>

        {/* Right: State, Telemetry & Actions */}
        <div className="flex items-center gap-2">
          {/* Worker Run State Pill */}
          <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-[#1B1B1B] border border-[#333333] text-xs font-mono">
            <span className="text-[10px] text-[#737373] uppercase">STATE</span>
            <StatusIndicator state={effectiveState} size="sm" />
          </div>

          {/* Quick GPU Telemetry Pill */}
          {telemetry && (
            <div className="hidden sm:flex items-center gap-1.5 px-2 py-0.5 rounded bg-[#1B1B1B] border border-[#333333] text-xs font-mono">
              <Cpu className="w-3.5 h-3.5 text-[#A3A3A3]" />
              <span className="text-[#F2F2F2] font-mono-num font-medium">
                {telemetry.gpu_utilization_pct ?? telemetry.gpu_util_percent ?? 0}%
              </span>
              <span className="text-[#404040]">|</span>
              <span className="text-[#A3A3A3] font-mono-num">
                {Math.round(telemetry.memory_used_mb ?? telemetry.used_vram_mb ?? 0)} MB
              </span>
            </div>
          )}

          {/* WebSocket Link Status */}
          <div
            className="flex items-center gap-1.5 px-2 py-0.5 rounded text-xs font-mono border"
            style={{
              borderColor: isConnected ? "rgba(74, 222, 128, 0.25)" : "rgba(248, 113, 113, 0.25)",
              backgroundColor: isConnected ? "rgba(74, 222, 128, 0.06)" : "rgba(248, 113, 113, 0.06)",
            }}
            title={isConnected ? "WebSocket Connected" : "WebSocket Disconnected"}
          >
            {isConnected ? (
              <Wifi className="w-3 h-3 text-[#4ade80]" />
            ) : (
              <WifiOff className="w-3 h-3 text-[#f87171]" />
            )}
            <span className="text-[10px] font-medium text-[#F2F2F2]">
              {isConnected ? "LIVE" : "OFFLINE"}
            </span>
          </div>

          {/* Alerts Bell */}
          <div className="relative">
            <button
              onClick={() => setShowAlertsDropdown(!showAlertsDropdown)}
              className="p-1 rounded bg-[#1B1B1B] border border-[#333333] hover:border-[#404040] text-[#A3A3A3] hover:text-[#F2F2F2] transition-colors relative"
              title="Recent Alerts"
            >
              <Bell className="w-3.5 h-3.5" />
              {recentAlerts.length > 0 && (
                <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 rounded-full bg-[#facc15]" />
              )}
            </button>

            {showAlertsDropdown && (
              <div className="absolute right-0 mt-1.5 w-72 bg-[#202020] border border-[#333333] rounded shadow-lg p-2.5 z-50">
                <div className="flex justify-between items-center pb-1.5 border-b border-[#262626] text-xs font-mono">
                  <span className="font-medium text-[#F2F2F2]">ALERTS</span>
                  <span className="text-[#737373] text-[10px]">{recentAlerts.length} events</span>
                </div>
                <div className="max-h-56 overflow-y-auto space-y-1 mt-1.5">
                  {recentAlerts.length === 0 ? (
                    <div className="p-2 text-center text-xs text-[#737373]">
                      No system alerts.
                    </div>
                  ) : (
                    recentAlerts.slice(0, 8).map((alert, idx) => (
                      <div
                        key={idx}
                        className="p-1.5 rounded bg-[#1B1B1B] border border-[#262626] text-[11px] font-mono flex flex-col gap-0.5"
                      >
                        <div className="flex justify-between items-center">
                          <span
                            className={
                              alert.level === "CRITICAL" || alert.level === "ERROR"
                                ? "text-[#f87171] font-semibold"
                                : alert.level === "WARNING"
                                ? "text-[#facc15] font-semibold"
                                : "text-[#A3A3A3] font-semibold"
                            }
                          >
                            [{alert.level}] {alert.category}
                          </span>
                          {alert.step !== undefined && (
                            <span className="text-[#737373] text-[10px]">
                              Step {alert.step}
                            </span>
                          )}
                        </div>
                        <span className="text-[#A3A3A3]">{alert.message}</span>
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
              className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-[#1B1B1B] border border-[#333333] hover:border-[#404040] text-[#A3A3A3] text-xs transition-colors"
              title="Active Workspace"
            >
              <Briefcase className="w-3 h-3 text-[#737373]" />
              <span className="max-w-[80px] truncate text-[#F2F2F2] font-medium">
                {activeWorkspace?.name || "Default"}
              </span>
              <ChevronDown className="w-3 h-3 text-[#737373]" />
            </button>

            {showWorkspaceDropdown && (
              <div className="absolute right-0 mt-1.5 w-48 bg-[#202020] border border-[#333333] rounded shadow-lg p-1.5 z-50">
                <div className="px-2 py-1 text-[10px] font-mono text-[#737373] border-b border-[#262626]">
                  WORKSPACES
                </div>
                <div className="max-h-48 overflow-y-auto space-y-0.5 mt-1 text-xs">
                  {workspaces.map((ws) => (
                    <div
                      key={ws.id}
                      className={`group w-full px-2 py-1 rounded flex items-center justify-between transition-colors ${
                        activeWorkspace?.id === ws.id
                          ? "bg-[#262626] text-[#F2F2F2] font-medium"
                          : "text-[#A3A3A3] hover:bg-[#262626] hover:text-[#F2F2F2]"
                      }`}
                    >
                      <button
                        type="button"
                        onClick={() => handleSwitchWorkspace(ws.id)}
                        className="truncate flex-1 text-left flex items-center gap-1.5"
                      >
                        <span className="truncate">{ws.name}</span>
                        {activeWorkspace?.id === ws.id && (
                          <span className="w-1.5 h-1.5 rounded-full bg-[#D4D4D4] shrink-0" />
                        )}
                      </button>
                      {ws.id !== "default" && (
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setShowWorkspaceDropdown(false);
                            setWorkspaceToDelete({ id: ws.id, name: ws.name });
                          }}
                          className="opacity-0 group-hover:opacity-100 p-0.5 text-[#737373] hover:text-[#f87171] transition-all ml-1.5 shrink-0"
                          title={`Delete workspace ${ws.name}`}
                          aria-label={`Delete workspace ${ws.name}`}
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      )}
                    </div>
                  ))}
                </div>

                <button
                  type="button"
                  onClick={() => {
                    setShowWorkspaceDropdown(false);
                    setShowCreateWsModal(true);
                  }}
                  className="w-full mt-1.5 pt-1.5 border-t border-[#262626] text-left px-2 py-1 rounded flex items-center gap-1.5 text-xs text-[#A3A3A3] hover:bg-[#262626] hover:text-[#F2F2F2] transition-colors"
                >
                  <Plus className="w-3.5 h-3.5 text-[#D4D4D4]" />
                  <span>New Workspace...</span>
                </button>
              </div>
            )}
          </div>

          {/* White / Dark Theme Switch */}
          <button
            type="button"
            onClick={toggleTheme}
            className="p-1.5 rounded text-[#737373] hover:text-[#F2F2F2] hover:bg-[#202020] transition-colors"
            title={theme === "dark" ? "Switch to White Theme" : "Switch to Dark Theme"}
            aria-label="Toggle theme"
          >
            {theme === "dark" ? (
              <Sun className="w-3.5 h-3.5 text-[#A3A3A3]" />
            ) : (
              <Moon className="w-3.5 h-3.5 text-[#111827]" />
            )}
          </button>
        </div>
      </header>

      {/* Create Workspace Modal */}
      {showCreateWsModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm font-mono text-xs">
          <div className="bg-[#202020] border border-[#333333] rounded-lg w-full max-w-sm p-4 shadow-xl">
            <div className="flex items-center justify-between pb-2 mb-3 border-b border-[#262626]">
              <div className="flex items-center gap-1.5 font-semibold text-[#F2F2F2] uppercase">
                <Briefcase className="w-4 h-4 text-[#A3A3A3]" />
                <span>Create Isolated Workspace</span>
              </div>
              <button
                type="button"
                onClick={() => setShowCreateWsModal(false)}
                className="text-[#737373] hover:text-[#F2F2F2]"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handleCreateWorkspace} className="space-y-3">
              <div>
                <label className="text-[10px] text-[#737373] uppercase font-medium block mb-1">
                  Workspace Name
                </label>
                <input
                  type="text"
                  required
                  autoFocus
                  placeholder="e.g. Experiment_A, Romanian_SFT"
                  value={newWsName}
                  onChange={(e) => setNewWsName(e.target.value)}
                  className="input w-full"
                />
                <span className="text-[10px] text-[#737373] mt-1 block">
                  Isolated environment with zero shared datasets or custom models.
                </span>
              </div>

              <div className="flex justify-end gap-2 pt-2 border-t border-[#262626]">
                <button
                  type="button"
                  onClick={() => setShowCreateWsModal(false)}
                  className="btn btn-secondary btn-sm"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!newWsName.trim() || isCreatingWs}
                  className="btn btn-primary btn-sm flex items-center gap-1"
                >
                  <Plus className="w-3 h-3" />
                  <span>{isCreatingWs ? "Creating..." : "Create & Switch"}</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Confirm Delete Workspace Dialog */}
      <ConfirmDialog
        isOpen={workspaceToDelete !== null}
        title={`Delete workspace "${workspaceToDelete?.name}"?`}
        message={`This will permanently remove workspace "${workspaceToDelete?.name}" along with all of its isolated runs, datasets, models, and checkpoints. This action cannot be undone.`}
        confirmLabel="Delete workspace"
        cancelLabel="Cancel"
        variant="danger"
        loading={isDeletingWs}
        onConfirm={handleDeleteWorkspace}
        onCancel={() => setWorkspaceToDelete(null)}
      />

      <CommandPalette isOpen={isCommandOpen} onClose={() => setIsCommandOpen(false)} />
      <SetupWizardModal isOpen={isWizardOpen} onClose={() => setIsWizardOpen(false)} />
    </>
  );
};
