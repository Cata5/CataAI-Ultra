"use client";

import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { api, isTauri } from "@/services/api";
import { SystemLogEntry } from "@/types/api";
import {
  Terminal,
  Search,
  Filter,
  RotateCcw,
  Trash2,
  Download,
  Pause,
  Play,
  ArrowDown,
  Copy,
  Check,
  AlertCircle,
  AlertTriangle,
  Info,
  Clock,
  Layers,
  ChevronDown,
  ChevronRight,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";

type LogLevelFilter = "ALL" | "ERROR" | "WARN" | "INFO" | "DEBUG";
type LogSourceFilter = "ALL" | "WORKER" | "TAURI" | "SYSTEM" | "DATABASE";

export const LogsView: React.FC = () => {
  const [logs, setLogs] = useState<SystemLogEntry[]>([]);
  const [filterText, setFilterText] = useState<string>("");
  const [selectedLevel, setSelectedLevel] = useState<LogLevelFilter>("ALL");
  const [selectedSource, setSelectedSource] = useState<LogSourceFilter>("ALL");
  const [autoScroll, setAutoScroll] = useState<boolean>(true);
  const [isLive, setIsLive] = useState<boolean>(true);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [expandedLogId, setExpandedLogId] = useState<number | null>(null);
  const [isCopied, setIsCopied] = useState<boolean>(false);

  const terminalEndRef = useRef<HTMLDivElement>(null);
  const terminalContainerRef = useRef<HTMLDivElement>(null);

  const fetchLogs = useCallback(async () => {
    try {
      const data = await api.getSystemLogs(2500);
      setLogs(data);
    } catch (err) {
      console.error("Failed to fetch system logs:", err);
    }
  }, []);

  // Initial load
  useEffect(() => {
    setIsLoading(true);
    fetchLogs().finally(() => setIsLoading(false));
  }, [fetchLogs]);

  // Periodic polling for live logs
  useEffect(() => {
    if (!isLive) return;
    const interval = setInterval(() => {
      fetchLogs();
    }, 2000);
    return () => clearInterval(interval);
  }, [isLive, fetchLogs]);

  // Listen to live Tauri events if available
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    if (isTauri()) {
      import("@tauri-apps/api/event").then(({ listen }) => {
        listen<any>("tauri://system-log", (event) => {
          const payload = event.payload;
          if (payload) {
            setLogs((prev) => [
              ...prev,
              {
                id: prev.length + 1,
                timestamp: payload.timestamp ? new Date(payload.timestamp * 1000).toLocaleTimeString() : new Date().toLocaleTimeString(),
                level: payload.level || "INFO",
                source: payload.source || "WORKER",
                message: payload.message || "",
                raw: payload.raw || payload.message || "",
              },
            ]);
          }
        }).then((fn) => {
          unlisten = fn;
        }).catch(console.error);
      });
    }
    return () => {
      if (unlisten) unlisten();
    };
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    if (autoScroll && terminalContainerRef.current) {
      terminalContainerRef.current.scrollTop = terminalContainerRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  const handleClearLogs = async () => {
    if (window.confirm("Are you sure you want to clear all logs?")) {
      try {
        await api.clearSystemLogs();
        setLogs([]);
      } catch (err) {
        console.error("Failed to clear logs:", err);
      }
    }
  };

  const handleCopyLogs = () => {
    const text = filteredLogs.map((l) => `[${l.timestamp || "—"}] [${l.level}] [${l.source}] ${l.raw || l.message}`).join("\n");
    navigator.clipboard.writeText(text);
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2000);
  };

  const handleDownloadLogs = () => {
    const text = filteredLogs.map((l) => `[${l.timestamp || "—"}] [${l.level}] [${l.source}] ${l.raw || l.message}`).join("\n");
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `cata_ai_system_logs_${new Date().toISOString().replace(/[:.]/g, "-")}.txt`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Filtered entries
  const filteredLogs = useMemo(() => {
    const query = filterText.toLowerCase().trim();
    return logs.filter((log) => {
      if (selectedLevel !== "ALL" && log.level !== selectedLevel) return false;
      if (selectedSource !== "ALL" && log.source !== selectedSource) return false;
      if (query) {
        return (
          log.message.toLowerCase().includes(query) ||
          log.raw.toLowerCase().includes(query) ||
          log.source.toLowerCase().includes(query) ||
          (log.timestamp && log.timestamp.toLowerCase().includes(query))
        );
      }
      return true;
    });
  }, [logs, selectedLevel, selectedSource, filterText]);

  // Counts
  const errorCount = useMemo(() => logs.filter((l) => l.level === "ERROR").length, [logs]);
  const warnCount = useMemo(() => logs.filter((l) => l.level === "WARN").length, [logs]);
  const infoCount = useMemo(() => logs.filter((l) => l.level === "INFO").length, [logs]);
  const debugCount = useMemo(() => logs.filter((l) => l.level === "DEBUG").length, [logs]);

  return (
    <div className="space-y-3 font-mono text-xs max-w-[1600px] mx-auto animate-in fade-in duration-150">
      {/* Top Header Card */}
      <div className="panel p-3.5 bg-[#202020] border-[#333333] flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded bg-[#1A1A1A] border border-[#333333] flex items-center justify-center shrink-0 text-[#F2F2F2]">
            <Terminal className="w-4 h-4 text-[#A3A3A3]" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-xs font-semibold text-[#F2F2F2] uppercase tracking-wide">
                System & Training Logs
              </h2>
              <span className="px-1.5 py-0.5 rounded text-[10px] bg-[#141414] border border-[#333333] text-[#A3A3A3] font-mono-num">
                {filteredLogs.length} of {logs.length} events
              </span>
            </div>
            <p className="text-[10px] text-[#737373]">
              Execution trace, PyTorch CUDA worker stdout/stderr, supervisor commands, and runtime diagnostics.
            </p>
          </div>
        </div>

        {/* Global Action Toolbar */}
        <div className="flex items-center gap-2 flex-wrap">
          <button
            onClick={() => setAutoScroll(!autoScroll)}
            className={cn(
              "px-2.5 py-1.5 rounded text-[11px] font-medium border transition-colors flex items-center gap-1.5",
              autoScroll
                ? "bg-[#1A1A1A] border-[#4ADE80]/40 text-[#4ADE80]"
                : "bg-[#1A1A1A] border-[#333333] text-[#737373] hover:text-[#A3A3A3]"
            )}
            title="Toggle Auto-Scroll to bottom on new log arrival"
          >
            <ArrowDown className="w-3.5 h-3.5" />
            <span>Auto-Scroll {autoScroll ? "ON" : "OFF"}</span>
          </button>

          <button
            onClick={() => setIsLive(!isLive)}
            className={cn(
              "px-2.5 py-1.5 rounded text-[11px] font-medium border transition-colors flex items-center gap-1.5",
              isLive
                ? "bg-[#1A1A1A] border-[#38BDF8]/40 text-[#38BDF8]"
                : "bg-[#1A1A1A] border-[#333333] text-[#737373] hover:text-[#A3A3A3]"
            )}
            title="Toggle live log polling"
          >
            {isLive ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            <span>Live {isLive ? "Active" : "Paused"}</span>
          </button>

          <button
            onClick={() => fetchLogs()}
            disabled={isLoading}
            className="px-2.5 py-1.5 rounded bg-[#1A1A1A] border border-[#333333] hover:border-[#525252] text-[#A3A3A3] hover:text-[#F2F2F2] text-[11px] font-medium transition-colors flex items-center gap-1.5"
            title="Refresh logs immediately"
          >
            <RotateCcw className={cn("w-3.5 h-3.5", isLoading && "animate-spin")} />
            <span>Refresh</span>
          </button>

          <button
            onClick={handleCopyLogs}
            className="px-2.5 py-1.5 rounded bg-[#1A1A1A] border border-[#333333] hover:border-[#525252] text-[#A3A3A3] hover:text-[#F2F2F2] text-[11px] font-medium transition-colors flex items-center gap-1.5"
            title="Copy filtered logs to clipboard"
          >
            {isCopied ? <Check className="w-3.5 h-3.5 text-[#4ADE80]" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{isCopied ? "Copied" : "Copy"}</span>
          </button>

          <button
            onClick={handleDownloadLogs}
            className="px-2.5 py-1.5 rounded bg-[#1A1A1A] border border-[#333333] hover:border-[#525252] text-[#A3A3A3] hover:text-[#F2F2F2] text-[11px] font-medium transition-colors flex items-center gap-1.5"
            title="Export filtered logs to .txt file"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export</span>
          </button>

          <button
            onClick={handleClearLogs}
            className="px-2.5 py-1.5 rounded bg-[#1A1A1A] border border-[#333333] hover:border-[#F87171]/50 text-[#737373] hover:text-[#F87171] text-[11px] font-medium transition-colors flex items-center gap-1.5"
            title="Clear all logs"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear</span>
          </button>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="panel p-2.5 bg-[#202020] border-[#333333] flex flex-col md:flex-row items-stretch md:items-center justify-between gap-2.5">
        {/* Search input */}
        <div className="relative flex-1 min-w-[240px]">
          <Search className="w-3.5 h-3.5 text-[#737373] absolute left-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
          <input
            type="text"
            value={filterText}
            onChange={(e) => setFilterText(e.target.value)}
            placeholder="Search log messages, tracebacks, devices, steps..."
            className="input w-full pl-8 pr-7 py-1 text-xs font-mono"
          />
          {filterText && (
            <button
              onClick={() => setFilterText("")}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-[#737373] hover:text-[#F2F2F2]"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {/* Level Filters */}
        <div className="flex items-center gap-1 flex-wrap">
          <button
            onClick={() => setSelectedLevel("ALL")}
            className={cn(
              "px-2 py-1 rounded text-[10px] font-medium transition-colors border",
              selectedLevel === "ALL"
                ? "bg-[#2A2A2A] border-[#D4D4D4] text-[#F2F2F2]"
                : "bg-[#171717] border-[#2A2A2A] text-[#737373] hover:text-[#A3A3A3]"
            )}
          >
            ALL ({logs.length})
          </button>

          <button
            onClick={() => setSelectedLevel("ERROR")}
            className={cn(
              "px-2 py-1 rounded text-[10px] font-medium transition-colors border flex items-center gap-1",
              selectedLevel === "ERROR"
                ? "bg-rose-950/50 border-rose-500 text-rose-300 font-semibold"
                : errorCount > 0
                ? "bg-rose-950/20 border-rose-900/40 text-rose-400 hover:bg-rose-950/30"
                : "bg-[#171717] border-[#2A2A2A] text-[#737373] hover:text-[#A3A3A3]"
            )}
          >
            <AlertCircle className="w-3 h-3 text-rose-400" />
            <span>ERROR ({errorCount})</span>
          </button>

          <button
            onClick={() => setSelectedLevel("WARN")}
            className={cn(
              "px-2 py-1 rounded text-[10px] font-medium transition-colors border flex items-center gap-1",
              selectedLevel === "WARN"
                ? "bg-amber-950/50 border-amber-500 text-amber-300 font-semibold"
                : warnCount > 0
                ? "bg-amber-950/20 border-amber-900/40 text-amber-400 hover:bg-amber-950/30"
                : "bg-[#171717] border-[#2A2A2A] text-[#737373] hover:text-[#A3A3A3]"
            )}
          >
            <AlertTriangle className="w-3 h-3 text-amber-400" />
            <span>WARN ({warnCount})</span>
          </button>

          <button
            onClick={() => setSelectedLevel("INFO")}
            className={cn(
              "px-2 py-1 rounded text-[10px] font-medium transition-colors border flex items-center gap-1",
              selectedLevel === "INFO"
                ? "bg-sky-950/50 border-sky-500 text-sky-300 font-semibold"
                : "bg-[#171717] border-[#2A2A2A] text-[#737373] hover:text-[#A3A3A3]"
            )}
          >
            <Info className="w-3 h-3 text-sky-400" />
            <span>INFO ({infoCount})</span>
          </button>

          <button
            onClick={() => setSelectedLevel("DEBUG")}
            className={cn(
              "px-2 py-1 rounded text-[10px] font-medium transition-colors border",
              selectedLevel === "DEBUG"
                ? "bg-[#2A2A2A] border-[#D4D4D4] text-[#F2F2F2]"
                : "bg-[#171717] border-[#2A2A2A] text-[#737373] hover:text-[#A3A3A3]"
            )}
          >
            DEBUG ({debugCount})
          </button>
        </div>

        {/* Source Dropdown Filter */}
        <div className="flex items-center gap-1.5 shrink-0">
          <Filter className="w-3.5 h-3.5 text-[#737373]" />
          <select
            value={selectedSource}
            onChange={(e) => setSelectedSource(e.target.value as LogSourceFilter)}
            className="input text-[11px] py-1 font-mono bg-[#171717] border-[#2A2A2A]"
          >
            <option value="ALL">All Sources</option>
            <option value="WORKER">Worker (PyTorch/CUDA)</option>
            <option value="TAURI">Tauri (Rust Supervisor)</option>
            <option value="SYSTEM">System & Panic</option>
            <option value="DATABASE">Database & Storage</option>
          </select>
        </div>
      </div>

      {/* Terminal Log Console */}
      <div className="panel bg-[#101010] border-[#2A2A2A] rounded-lg overflow-hidden flex flex-col h-[650px] shadow-2xl">
        {/* Terminal Header Bar */}
        <div className="bg-[#171717] px-3.5 py-2 border-b border-[#2A2A2A] flex items-center justify-between text-[11px] select-none">
          <div className="flex items-center gap-2">
            <div className="flex gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full bg-rose-500/80" />
              <div className="w-2.5 h-2.5 rounded-full bg-amber-500/80" />
              <div className="w-2.5 h-2.5 rounded-full bg-emerald-500/80" />
            </div>
            <span className="text-[#737373] font-mono text-[10px] ml-2">
              cata_ai@workstation: ~/logs/tauri_debug.log
            </span>
          </div>

          <div className="flex items-center gap-3 text-[10px] text-[#737373]">
            {isLive && (
              <span className="flex items-center gap-1 text-[#4ADE80]">
                <span className="w-1.5 h-1.5 rounded-full bg-[#4ADE80] animate-pulse" />
                STREAMING
              </span>
            )}
            <span>{filteredLogs.length} lines</span>
          </div>
        </div>

        {/* Terminal Body */}
        <div
          ref={terminalContainerRef}
          className="flex-1 overflow-y-auto p-3 space-y-1 font-mono text-[11px] leading-relaxed select-text"
        >
          {filteredLogs.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center p-8 text-[#525252]">
              <Terminal className="w-8 h-8 mb-2 text-[#404040]" />
              <p className="text-xs text-[#737373]">No logs matching current filter criteria.</p>
              <p className="text-[10px] mt-1 text-[#525252]">
                {filterText || selectedLevel !== "ALL" || selectedSource !== "ALL"
                  ? "Try resetting filters or clearing the search query."
                  : "Worker output and system traces will appear here automatically."}
              </p>
            </div>
          ) : (
            filteredLogs.map((log) => {
              const isExpanded = expandedLogId === log.id;
              const isError = log.level === "ERROR";
              const isWarn = log.level === "WARN";

              return (
                <div
                  key={log.id}
                  onClick={() => setExpandedLogId(isExpanded ? null : log.id)}
                  className={cn(
                    "group px-2 py-1 rounded transition-colors cursor-pointer border",
                    isError
                      ? "bg-rose-950/20 border-rose-900/30 hover:bg-rose-950/30 text-rose-300"
                      : isWarn
                      ? "bg-amber-950/15 border-amber-900/30 hover:bg-amber-950/25 text-amber-200"
                      : "bg-[#141414] border-transparent hover:border-[#262626] hover:bg-[#1A1A1A] text-[#D4D4D4]"
                  )}
                >
                  <div className="flex items-start gap-2.5">
                    {/* Line Number */}
                    <span className="text-[#525252] text-[10px] select-none font-mono-num shrink-0 w-11 text-right">
                      #{String(log.id).padStart(4, "0")}
                    </span>

                    {/* Timestamp */}
                    <span className="text-[#737373] text-[10px] select-none shrink-0 font-mono-num">
                      {log.timestamp || "—"}
                    </span>

                    {/* Level Badge */}
                    <span
                      className={cn(
                        "px-1.5 py-0.2 rounded text-[9px] font-bold select-none shrink-0 uppercase tracking-wider",
                        isError
                          ? "bg-rose-500/20 text-rose-400 border border-rose-500/30"
                          : isWarn
                          ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                          : log.level === "DEBUG"
                          ? "bg-slate-800 text-slate-400 border border-slate-700"
                          : "bg-sky-500/15 text-sky-300 border border-sky-500/25"
                      )}
                    >
                      {log.level}
                    </span>

                    {/* Source Badge */}
                    <span
                      className={cn(
                        "px-1.5 py-0.2 rounded text-[9px] select-none shrink-0 uppercase",
                        log.source === "WORKER"
                          ? "bg-purple-950/40 text-purple-300 border border-purple-800/40"
                          : log.source === "DATABASE"
                          ? "bg-amber-950/40 text-amber-300 border border-amber-800/40"
                          : log.source === "SYSTEM"
                          ? "bg-cyan-950/40 text-cyan-300 border border-cyan-800/40"
                          : "bg-emerald-950/40 text-emerald-300 border border-emerald-800/40"
                      )}
                    >
                      {log.source}
                    </span>

                    {/* Message */}
                    <div className="flex-1 break-all whitespace-pre-wrap font-mono">
                      {log.message}
                    </div>

                    {/* Expand icon */}
                    <span className="text-[#525252] group-hover:text-[#A3A3A3] shrink-0 pt-0.5">
                      {isExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                    </span>
                  </div>

                  {/* Expanded Raw Details */}
                  {isExpanded && (
                    <div className="mt-2 pt-2 border-t border-[#333333] pl-14 text-[10px] space-y-1 text-[#A3A3A3]">
                      <div className="flex items-center justify-between text-[#737373]">
                        <span>RAW ENTRY:</span>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            navigator.clipboard.writeText(log.raw || log.message);
                          }}
                          className="hover:text-[#F2F2F2] flex items-center gap-1"
                        >
                          <Copy className="w-2.5 h-2.5" /> Copy Raw
                        </button>
                      </div>
                      <pre className="bg-[#0D0D0D] p-2 rounded border border-[#262626] whitespace-pre-wrap select-text text-[#E5E5E5]">
                        {log.raw || log.message}
                      </pre>
                    </div>
                  )}
                </div>
              );
            })
          )}
          <div ref={terminalEndRef} />
        </div>

        {/* Terminal Footer Info */}
        <div className="bg-[#141414] px-3.5 py-1.5 border-t border-[#262626] flex items-center justify-between text-[10px] text-[#737373]">
          <div className="flex items-center gap-3">
            <span>Filter: {selectedLevel} / {selectedSource}</span>
            {filterText && <span>Query: &quot;{filterText}&quot;</span>}
          </div>
          <div>
            Click any row to inspect raw payload and stack traces.
          </div>
        </div>
      </div>
    </div>
  );
};
