"use client";

import React, { useState, useEffect, useMemo } from "react";
import { api } from "@/services/api";
import { RunItem, CheckpointItem, MetricPoint } from "@/types/api";
import { DataTable, Column } from "@/components/ui/DataTable";
import { ChartPanel } from "@/components/ui/ChartPanel";
import { StatusIndicator } from "@/components/ui/StatusIndicator";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { Button } from "@/components/ui/Button";
import { formatCompactNumber, formatNumber, formatDateTime } from "@/lib/utils";
import {
  History,
  Bookmark,
  Trash2,
  LineChart,
  RefreshCw,
  ExternalLink,
  Layers,
  Play,
  TrendingDown,
  HardDrive,
  Activity,
  CheckCircle2,
  Clock,
  Sparkles,
  BarChart2,
  Terminal,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { LogsView } from "@/features/logs/LogsView";

export type RunManagementTab = "experiments" | "checkpoints" | "analytics" | "logs" | "all";

export interface RunsViewProps {
  initialTab?: RunManagementTab | string;
}

export const RunsView: React.FC<RunsViewProps> = ({ initialTab = "experiments" }) => {
  const [activeTab, setActiveTab] = useState<RunManagementTab>(() => {
    if (initialTab === "checkpoints" || initialTab === "analytics" || initialTab === "logs" || initialTab === "all") {
      return initialTab;
    }
    return "experiments";
  });

  const [runs, setRuns] = useState<RunItem[]>([]);
  const [checkpoints, setCheckpoints] = useState<CheckpointItem[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [runMetrics, setRunMetrics] = useState<MetricPoint[]>([]);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Delete checkpoint state
  const [deleteTargetId, setDeleteTargetId] = useState<string | null>(null);

  // Sync tab if prop changes (e.g. navigation across sidebar routes)
  useEffect(() => {
    if (initialTab && (initialTab === "checkpoints" || initialTab === "analytics" || initialTab === "logs" || initialTab === "all" || initialTab === "experiments")) {
      setActiveTab(initialTab as RunManagementTab);
    }
  }, [initialTab]);

  const fetchHistory = async () => {
    setIsRefreshing(true);
    try {
      const [r, c] = await Promise.all([api.listRuns(100), api.listCheckpoints()]);
      setRuns(r);
      setCheckpoints(c);
      if (r.length > 0 && !selectedRunId) {
        setSelectedRunId(r[0].id);
      }
    } catch (e: any) {
      console.error(e);
    } finally {
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  // Fetch metrics for selected historical run
  useEffect(() => {
    if (!selectedRunId) return;
    api
      .getRunMetrics(selectedRunId, 0, 3000)
      .then((m) => {
        setRunMetrics(m);
      })
      .catch(console.error);
  }, [selectedRunId]);

  const handleDeleteCheckpoint = async () => {
    if (!deleteTargetId) return;
    try {
      await api.deleteCheckpoint(deleteTargetId);
      setCheckpoints((prev) => prev.filter((c) => c.id !== deleteTargetId && c.filename !== deleteTargetId));
      setDeleteTargetId(null);
    } catch (e: any) {
      alert(`Delete error: ${e.message}`);
    }
  };

  // Metrics summary calculations
  const totalTokensProcessed = useMemo(() => {
    return runs.reduce((acc, r) => acc + (r.total_tokens || 0), 0);
  }, [runs]);

  const bestOverallValLoss = useMemo(() => {
    const valid = checkpoints
      .map((c) => c.val_loss)
      .filter((v): v is number => typeof v === "number" && !isNaN(v) && v > 0);
    return valid.length > 0 ? Math.min(...valid) : null;
  }, [checkpoints]);

  const totalDiskUsageMb = useMemo(() => {
    return checkpoints.reduce((acc, c) => acc + (c.size_mb || 0), 0);
  }, [checkpoints]);

  const selectedRun = useMemo(() => {
    return runs.find((r) => r.id === selectedRunId) || runs[0] || null;
  }, [runs, selectedRunId]);

  // Columns definition
  const runColumns: Column<RunItem>[] = [
    {
      header: "RUN ID",
      accessorKey: "id",
      sortable: true,
      cell: (item) => (
        <button
          onClick={() => {
            setSelectedRunId(item.id);
            if (activeTab === "experiments") {
              setActiveTab("analytics");
            }
          }}
          className={cn(
            "text-left font-medium transition-colors flex items-center gap-1.5 group",
            selectedRunId === item.id ? "text-[#F2F2F2] font-semibold" : "text-[#A3A3A3] hover:text-[#F2F2F2]"
          )}
          title="Click to view analytics"
        >
          <History className="w-3.5 h-3.5 text-[#737373] group-hover:text-[#F2F2F2]" />
          <span className="truncate max-w-[160px] font-mono underline decoration-dotted decoration-[#525252]">
            {item.id}
          </span>
        </button>
      ),
    },
    {
      header: "NAME / TASK",
      accessorKey: "name",
      cell: (item) => (
        <div className="flex flex-col">
          <span className="font-medium text-[#F2F2F2] truncate max-w-[180px]">{item.name || item.id}</span>
          <span className="text-[10px] text-[#737373]">
            {item.config?.task || (item as any).current_phase || "causal_lm"}
          </span>
        </div>
      ),
    },
    {
      header: "STATUS",
      accessorKey: "status",
      cell: (item) => <StatusIndicator state={item.status} size="sm" />,
    },
    {
      header: "TOTAL STEPS",
      accessorKey: "total_steps",
      sortable: true,
      cell: (item) => (
        <span className="font-mono-num font-medium text-[#F2F2F2]">
          {item.total_steps?.toLocaleString() || 0}
        </span>
      ),
    },
    {
      header: "TOKENS SEEN",
      accessorKey: "total_tokens",
      sortable: true,
      cell: (item) => (
        <span className="font-mono-num text-[#A3A3A3]">
          {formatCompactNumber(item.total_tokens || 0)}
        </span>
      ),
    },
    {
      header: "BEST VAL LOSS",
      accessorKey: "best_val_loss",
      sortable: true,
      cell: (item) => (
        <span className="font-mono-num font-medium text-[#4ade80]">
          {item.best_val_loss ? formatNumber(item.best_val_loss, 4) : "—"}
        </span>
      ),
    },
    {
      header: "STARTED AT",
      accessorKey: "started_at",
      cell: (item) => (
        <span className="text-[10px] text-[#737373]">
          {formatDateTime(item.started_at)}
        </span>
      ),
    },
    {
      header: "ACTIONS",
      cell: (item) => (
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => {
              setSelectedRunId(item.id);
              setActiveTab("analytics");
            }}
            className="px-2 py-1 rounded bg-[#262626] border border-[#333333] hover:border-[#525252] text-[#A3A3A3] hover:text-[#F2F2F2] transition-colors flex items-center gap-1 text-[10px]"
            title="Inspect Analytics"
          >
            <LineChart className="w-3 h-3 text-[#A3A3A3]" />
            <span>Analytics</span>
          </button>
        </div>
      ),
    },
  ];

  const checkpointColumns: Column<CheckpointItem>[] = [
    {
      header: "CHECKPOINT FILE",
      accessorKey: "filename",
      sortable: true,
      cell: (item) => (
        <div className="flex items-center gap-2">
          <Bookmark className="w-3.5 h-3.5 text-[#A3A3A3] shrink-0" />
          <div className="flex flex-col">
            <div className="flex items-center gap-1.5">
              <span className="font-medium text-[#F2F2F2]">{item.model_name || item.filename}</span>
              {item.filename !== item.model_name && (
                <span className="text-[10px] text-[#737373] font-mono">({item.filename})</span>
              )}
              {item.run_id && (
                <span className="text-[9px] px-1.5 py-0.2 rounded bg-[#262626] text-[#A3A3A3] border border-[#333333] font-mono">
                  {item.run_id}
                </span>
              )}
            </div>
            <span className="text-[10px] text-[#737373] truncate max-w-xs">{item.path}</span>
          </div>
        </div>
      ),
    },
    {
      header: "STEP",
      accessorKey: "step",
      sortable: true,
      cell: (item) => (
        <span className="font-mono-num font-medium text-[#F2F2F2]">
          Step {item.step?.toLocaleString() || 0}
        </span>
      ),
    },
    {
      header: "VAL LOSS",
      accessorKey: "val_loss",
      sortable: true,
      cell: (item) => (
        <span className="font-mono-num text-[#4ade80] font-medium">
          {item.val_loss ? formatNumber(item.val_loss, 4) : "—"}
        </span>
      ),
    },
    {
      header: "SIZE (MB)",
      accessorKey: "size_mb",
      sortable: true,
      cell: (item) => (
        <span className="font-mono-num text-[#A3A3A3]">
          {item.size_mb ? `${item.size_mb.toFixed(1)} MB` : "—"}
        </span>
      ),
    },
    {
      header: "ACTIONS",
      cell: (item) => (
        <div className="flex items-center gap-1.5">
          <a
            href={`/evaluation?checkpoint=${encodeURIComponent(item.id || item.filename)}`}
            className="px-2 py-1 rounded bg-[#262626] border border-[#333333] hover:border-[#525252] text-[#A3A3A3] hover:text-[#F2F2F2] transition-colors flex items-center gap-1 text-[10px]"
            title="Load in Evaluation Lab"
          >
            <ExternalLink className="w-3 h-3" />
            <span>Eval</span>
          </a>
          <a
            href={`/training?resume_checkpoint=${encodeURIComponent(item.id || item.filename)}`}
            className="px-2 py-1 rounded bg-[#262626] border border-[#333333] hover:border-[#525252] text-[#A3A3A3] hover:text-[#F2F2F2] transition-colors flex items-center gap-1 text-[10px]"
            title="Resume Training from this checkpoint"
          >
            <Play className="w-3 h-3 text-[#4ade80]" />
            <span>Resume</span>
          </a>
          <button
            onClick={() => setDeleteTargetId(item.id || item.filename)}
            className="p-1.5 rounded bg-[#262626] border border-[#333333] hover:border-[#f87171]/50 hover:text-[#f87171] text-[#737373] transition-colors"
            title="Delete Checkpoint"
          >
            <Trash2 className="w-3 h-3" />
          </button>
        </div>
      ),
    },
  ];

  const chartSteps = runMetrics.map((m) => m.step);
  const lossSeries = runMetrics.map((m) => (m.loss !== undefined ? m.loss : (m as any).train_loss ?? 0));
  const valLossSeries = runMetrics.map((m) =>
    m.val_loss !== null && m.val_loss !== undefined ? Number(m.val_loss) : null
  );
  const hasValLoss = runMetrics.some((m) => m.val_loss !== null && m.val_loss !== undefined);
  const lrSeries = runMetrics.map((m) => (m.learning_rate !== undefined ? m.learning_rate : (m as any).lr ?? (m as any).muon_lr ?? 0));

  return (
    <div className="space-y-4 font-mono text-xs">
      {/* Top Header & Tab Switcher */}
      <div className="panel p-3.5 bg-[#202020] border-[#333333] flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          {activeTab === "experiments" && <Layers className="w-4 h-4 text-[#A3A3A3]" />}
          {activeTab === "checkpoints" && <Bookmark className="w-4 h-4 text-[#A3A3A3]" />}
          {activeTab === "analytics" && <LineChart className="w-4 h-4 text-[#A3A3A3]" />}
          {activeTab === "logs" && <Terminal className="w-4 h-4 text-[#A3A3A3]" />}
          {activeTab === "all" && <History className="w-4 h-4 text-[#A3A3A3]" />}
          <div>
            <h3 className="font-semibold text-[#F2F2F2] uppercase tracking-wider text-xs">
              {activeTab === "experiments" && "EXPERIMENTS & RUNS REPOSITORY"}
              {activeTab === "checkpoints" && "CHECKPOINT STORAGE & WEIGHT MANAGER"}
              {activeTab === "analytics" && "TRAINING RUN ANALYTICS & DYNAMICS"}
              {activeTab === "logs" && "SYSTEM & TRAINING EXECUTION LOGS"}
              {activeTab === "all" && "MANAGEMENT OVERVIEW (RUNS, CHECKPOINTS & ANALYTICS)"}
            </h3>
            <p className="text-[11px] text-[#737373]">
              {activeTab === "experiments" && "Track training run lifecycles, execution status, and token throughput milestones"}
              {activeTab === "checkpoints" && "Inspect saved model weights, safe deserialization (weights_only=True), and export options"}
              {activeTab === "analytics" && "Examine train vs validation loss convergence and adaptive learning rate curves"}
              {activeTab === "logs" && "Filter and inspect worker stdout/stderr, supervisor diagnostics, and CUDA runtime traces"}
              {activeTab === "all" && "Consolidated view of all experiments, stored checkpoints, and telemetry"}
            </p>
          </div>
        </div>

        {/* View Switcher Controls */}
        <div className="flex items-center gap-2 self-stretch sm:self-auto justify-between sm:justify-end">
          <div className="flex items-center gap-1 bg-[#1A1A1A] p-1 rounded-md border border-[#333333]">
            <button
              onClick={() => setActiveTab("experiments")}
              className={cn(
                "px-2.5 py-1 rounded text-[11px] font-medium transition-colors flex items-center gap-1.5",
                activeTab === "experiments"
                  ? "bg-[#2A2A2A] text-[#F2F2F2] shadow-sm border border-[#404040]"
                  : "text-[#737373] hover:text-[#A3A3A3]"
              )}
            >
              <Layers className="w-3 h-3" />
              <span>Experiments ({runs.length})</span>
            </button>
            <button
              onClick={() => setActiveTab("checkpoints")}
              className={cn(
                "px-2.5 py-1 rounded text-[11px] font-medium transition-colors flex items-center gap-1.5",
                activeTab === "checkpoints"
                  ? "bg-[#2A2A2A] text-[#F2F2F2] shadow-sm border border-[#404040]"
                  : "text-[#737373] hover:text-[#A3A3A3]"
              )}
            >
              <Bookmark className="w-3 h-3" />
              <span>Checkpoints ({checkpoints.length})</span>
            </button>
            <button
              onClick={() => setActiveTab("analytics")}
              className={cn(
                "px-2.5 py-1 rounded text-[11px] font-medium transition-colors flex items-center gap-1.5",
                activeTab === "analytics"
                  ? "bg-[#2A2A2A] text-[#F2F2F2] shadow-sm border border-[#404040]"
                  : "text-[#737373] hover:text-[#A3A3A3]"
              )}
            >
              <LineChart className="w-3 h-3" />
              <span>Analytics</span>
            </button>
            <button
              onClick={() => setActiveTab("logs")}
              className={cn(
                "px-2.5 py-1 rounded text-[11px] font-medium transition-colors flex items-center gap-1.5",
                activeTab === "logs"
                  ? "bg-[#2A2A2A] text-[#F2F2F2] shadow-sm border border-[#404040]"
                  : "text-[#737373] hover:text-[#A3A3A3]"
              )}
            >
              <Terminal className="w-3 h-3" />
              <span>Logs</span>
            </button>
            <button
              onClick={() => setActiveTab("all")}
              className={cn(
                "px-2 py-1 rounded text-[11px] font-medium transition-colors hidden md:flex items-center gap-1",
                activeTab === "all"
                  ? "bg-[#2A2A2A] text-[#F2F2F2] shadow-sm border border-[#404040]"
                  : "text-[#737373] hover:text-[#A3A3A3]"
              )}
            >
              <span>All</span>
            </button>
          </div>

          <Button
            variant="secondary"
            size="sm"
            onClick={fetchHistory}
            loading={isRefreshing}
            icon={<RefreshCw className="w-3 h-3" />}
          >
            Refresh
          </Button>
        </div>
      </div>

      {/* =========================================================================
          TAB 1: EXPERIMENTS (RUNS)
          ========================================================================= */}
      {(activeTab === "experiments" || activeTab === "all") && (
        <div className="space-y-4">
          {/* Quick Metrics Bar for Runs */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="panel p-3 bg-[#1C1C1C] border-[#333333]">
              <span className="text-[10px] text-[#737373] uppercase tracking-wider block">Total Recorded Runs</span>
              <span className="text-lg font-bold text-[#F2F2F2] font-mono-num">{runs.length}</span>
            </div>
            <div className="panel p-3 bg-[#1C1C1C] border-[#333333]">
              <span className="text-[10px] text-[#737373] uppercase tracking-wider block">Active / Running</span>
              <span className="text-lg font-bold text-[#4ade80] font-mono-num">
                {runs.filter((r) => r.status === "RUNNING").length}
              </span>
            </div>
            <div className="panel p-3 bg-[#1C1C1C] border-[#333333]">
              <span className="text-[10px] text-[#737373] uppercase tracking-wider block">Completed Sessions</span>
              <span className="text-lg font-bold text-[#60a5fa] font-mono-num">
                {runs.filter((r) => r.status === "COMPLETED").length}
              </span>
            </div>
            <div className="panel p-3 bg-[#1C1C1C] border-[#333333]">
              <span className="text-[10px] text-[#737373] uppercase tracking-wider block">Cumulative Tokens</span>
              <span className="text-lg font-bold text-[#F2F2F2] font-mono-num">
                {formatCompactNumber(totalTokensProcessed)}
              </span>
            </div>
          </div>

          {/* Runs History Table */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold text-[#F2F2F2] uppercase tracking-wider flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-[#A3A3A3]" /> EXPERIMENTS LIST ({runs.length})
              </h3>
              <span className="text-[11px] text-[#737373]">
                Click any Run ID to inspect historical loss curves & learning rate schedule
              </span>
            </div>
            <DataTable
              data={runs}
              columns={runColumns}
              keyExtractor={(r) => r.id}
              searchPlaceholder="Search runs by ID, name or task..."
            />
          </div>
        </div>
      )}

      {/* =========================================================================
          TAB 2: CHECKPOINTS
          ========================================================================= */}
      {(activeTab === "checkpoints" || activeTab === "all") && (
        <div className="space-y-4">
          {/* Quick Metrics Bar for Checkpoints */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="panel p-3 bg-[#1C1C1C] border-[#333333]">
              <span className="text-[10px] text-[#737373] uppercase tracking-wider block">Saved Checkpoints</span>
              <span className="text-lg font-bold text-[#F2F2F2] font-mono-num">{checkpoints.length}</span>
            </div>
            <div className="panel p-3 bg-[#1C1C1C] border-[#333333]">
              <span className="text-[10px] text-[#737373] uppercase tracking-wider block">Best Val Loss</span>
              <span className="text-lg font-bold text-[#4ade80] font-mono-num">
                {bestOverallValLoss ? formatNumber(bestOverallValLoss, 4) : "—"}
              </span>
            </div>
            <div className="panel p-3 bg-[#1C1C1C] border-[#333333]">
              <span className="text-[10px] text-[#737373] uppercase tracking-wider block">Total Disk Storage</span>
              <span className="text-lg font-bold text-[#F2F2F2] font-mono-num">
                {totalDiskUsageMb > 1024
                  ? `${(totalDiskUsageMb / 1024).toFixed(2)} GB`
                  : `${totalDiskUsageMb.toFixed(1)} MB`}
              </span>
            </div>
            <div className="panel p-3 bg-[#1C1C1C] border-[#333333]">
              <span className="text-[10px] text-[#737373] uppercase tracking-wider block">Format Security</span>
              <span className="text-xs font-semibold text-[#a3e635] mt-1 block">
                weights_only=True
              </span>
            </div>
          </div>

          {/* Checkpoints Manager Table */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold text-[#F2F2F2] uppercase tracking-wider flex items-center gap-1.5">
                <Bookmark className="w-3.5 h-3.5 text-[#A3A3A3]" /> SAVED MODEL CHECKPOINTS ({checkpoints.length})
              </h3>
              <span className="text-[11px] text-[#737373]">
                Load into Evaluation lab or resume training directly from any checkpoint
              </span>
            </div>
            <DataTable
              data={checkpoints}
              columns={checkpointColumns}
              keyExtractor={(c) => c.id || c.filename}
              searchPlaceholder="Search checkpoints by name, step, or run ID..."
            />
          </div>
        </div>
      )}

      {/* =========================================================================
          TAB 3: ANALYTICS & DYNAMICS
          ========================================================================= */}
      {(activeTab === "analytics" || activeTab === "all") && (
        <div className="space-y-4">
          {/* Run Picker Bar */}
          <div className="panel p-3 bg-[#1C1C1C] border-[#333333] flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <LineChart className="w-4 h-4 text-[#A3A3A3]" />
              <span className="font-semibold text-[#F2F2F2] uppercase text-xs">Active Analytics Target:</span>
              <select
                value={selectedRunId || ""}
                onChange={(e) => setSelectedRunId(e.target.value)}
                className="bg-[#262626] border border-[#404040] rounded px-2.5 py-1 text-xs text-[#F2F2F2] focus:outline-none focus:border-[#737373]"
              >
                {runs.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.id} {r.name ? `— ${r.name}` : ""} ({r.status})
                  </option>
                ))}
              </select>
            </div>

            {selectedRun && (
              <div className="flex items-center gap-3 text-[11px] text-[#A3A3A3]">
                <div className="flex items-center gap-1.5">
                  <span>Status:</span>
                  <StatusIndicator state={selectedRun.status} size="sm" />
                </div>
                <span>Steps: <strong className="text-[#F2F2F2] font-mono-num">{selectedRun.total_steps || 0}</strong></span>
                <span>Tokens: <strong className="text-[#F2F2F2] font-mono-num">{formatCompactNumber(selectedRun.total_tokens || 0)}</strong></span>
                {selectedRun.best_val_loss && (
                  <span>Best Val: <strong className="text-[#4ade80] font-mono-num">{formatNumber(selectedRun.best_val_loss, 4)}</strong></span>
                )}
              </div>
            )}
          </div>

          {/* Charts Display */}
          {selectedRunId ? (
            <div className="space-y-3">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
                <ChartPanel
                  title="HISTORICAL LOSS CURVE (TRAIN VS VALIDATION)"
                  labels={chartSteps}
                  datasets={[
                    {
                      label: "Train Loss",
                      data: lossSeries,
                      borderColor: "#F2F2F2",
                      backgroundColor: "rgba(242, 242, 242, 0.05)",
                      fill: true,
                    },
                    ...(hasValLoss
                      ? [
                          {
                            label: "Validation Loss",
                            data: valLossSeries,
                            borderColor: "#4ade80",
                            borderDash: [4, 4],
                            pointRadius: 4,
                            spanGaps: true,
                          },
                        ]
                      : []),
                  ]}
                  yAxisLabel="Loss"
                  height={240}
                />

                <ChartPanel
                  title="LEARNING RATE DYNAMICS (WSD / COSINE)"
                  labels={chartSteps}
                  datasets={[
                    {
                      label: "Learning Rate",
                      data: lrSeries,
                      borderColor: "#D4D4D4",
                      backgroundColor: "rgba(212, 212, 212, 0.05)",
                      fill: true,
                    },
                  ]}
                  yAxisLabel="Effective LR"
                  height={240}
                />
              </div>

              {/* Data points count */}
              <div className="flex items-center justify-between text-[11px] text-[#737373] px-1">
                <span>Total telemetry points plotted: {runMetrics.length}</span>
                {runMetrics.length === 0 && (
                  <span className="text-[#facc15]">No metric history recorded yet for run {selectedRunId}.</span>
                )}
              </div>
            </div>
          ) : (
            <div className="panel p-8 text-center text-[#737373]">
              No runs recorded yet. Start training to generate real-time metrics telemetry.
            </div>
          )}
        </div>
      )}

      {/* TAB 4: LOGS */}
      {activeTab === "logs" && (
        <LogsView />
      )}

      {/* Confirm Delete Dialog */}
      <ConfirmDialog
        isOpen={!!deleteTargetId}
        title="Delete Checkpoint"
        message={`Are you sure you want to permanently delete checkpoint '${deleteTargetId}'? This action cannot be undone.`}
        confirmLabel="Delete Checkpoint"
        cancelLabel="Cancel"
        variant="danger"
        onConfirm={handleDeleteCheckpoint}
        onCancel={() => setDeleteTargetId(null)}
      />
    </div>
  );
};
