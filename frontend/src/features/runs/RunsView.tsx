"use client";

import React, { useState, useEffect } from "react";
import { api } from "@/services/api";
import { RunItem, CheckpointItem, MetricPoint } from "@/types/api";
import { DataTable, Column } from "@/components/ui/DataTable";
import { ChartPanel } from "@/components/ui/ChartPanel";
import { StatusIndicator } from "@/components/ui/StatusIndicator";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { formatCompactNumber, formatNumber } from "@/lib/utils";
import {
  History,
  Bookmark,
  Trash2,
  LineChart,
  RefreshCw,
  ExternalLink,
  Layers,
} from "lucide-react";
import { cn } from "@/lib/utils";

export const RunsView: React.FC = () => {
  const [runs, setRuns] = useState<RunItem[]>([]);
  const [checkpoints, setCheckpoints] = useState<CheckpointItem[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [runMetrics, setRunMetrics] = useState<MetricPoint[]>([]);

  // Delete checkpoint state
  const [deleteTargetId, setDeleteTargetId] = useState<string | null>(null);

  const fetchHistory = async () => {
    try {
      const [r, c] = await Promise.all([api.listRuns(50), api.listCheckpoints()]);
      setRuns(r);
      setCheckpoints(c);
      if (r.length > 0 && !selectedRunId) {
        setSelectedRunId(r[0].id);
      }
    } catch (e: any) {
      console.error(e);
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

  const runColumns: Column<RunItem>[] = [
    {
      header: "RUN ID",
      accessorKey: "id",
      sortable: true,
      cell: (item) => (
        <button
          onClick={() => setSelectedRunId(item.id)}
          className={cn(
            "text-left font-bold transition-colors flex items-center gap-1.5",
            selectedRunId === item.id ? "text-cyan-400" : "text-slate-200 hover:text-cyan-300"
          )}
        >
          <History className="w-3.5 h-3.5" />
          <span className="truncate max-w-[140px]">{item.id}</span>
        </button>
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
        <span className="font-mono-num font-bold text-slate-100">
          {item.total_steps?.toLocaleString() || 0}
        </span>
      ),
    },
    {
      header: "TOKENS SEEN",
      accessorKey: "total_tokens",
      sortable: true,
      cell: (item) => (
        <span className="font-mono-num text-slate-300">
          {formatCompactNumber(item.total_tokens || 0)}
        </span>
      ),
    },
    {
      header: "BEST VAL LOSS",
      accessorKey: "best_val_loss",
      sortable: true,
      cell: (item) => (
        <span className="font-mono-num font-bold text-emerald-400">
          {item.best_val_loss ? formatNumber(item.best_val_loss, 4) : "—"}
        </span>
      ),
    },
    {
      header: "STARTED AT",
      accessorKey: "started_at",
      cell: (item) => (
        <span className="text-[10px] text-slate-400">
          {item.started_at ? new Date(item.started_at).toLocaleString() : "—"}
        </span>
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
          <Bookmark className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
          <div className="flex flex-col">
            <span className="font-bold text-slate-200">{item.filename}</span>
            <span className="text-[10px] text-slate-400 truncate max-w-xs">{item.path}</span>
          </div>
        </div>
      ),
    },
    {
      header: "STEP",
      accessorKey: "step",
      sortable: true,
      cell: (item) => (
        <span className="font-mono-num font-bold text-cyan-300">
          Step {item.step?.toLocaleString() || 0}
        </span>
      ),
    },
    {
      header: "VAL LOSS",
      accessorKey: "val_loss",
      sortable: true,
      cell: (item) => (
        <span className="font-mono-num text-emerald-400 font-bold">
          {item.val_loss ? formatNumber(item.val_loss, 4) : "—"}
        </span>
      ),
    },
    {
      header: "SIZE (MB)",
      accessorKey: "size_mb",
      sortable: true,
      cell: (item) => (
        <span className="font-mono-num text-slate-300">
          {item.size_mb ? `${item.size_mb.toFixed(1)} MB` : "—"}
        </span>
      ),
    },
    {
      header: "ACTIONS",
      cell: (item) => (
        <div className="flex items-center gap-2">
          <a
            href="/evaluation"
            className="p-1 rounded bg-slate-900 border border-slate-700 hover:border-cyan-400 text-slate-300 hover:text-cyan-400 transition-colors"
            title="Load in Evaluation Lab"
          >
            <ExternalLink className="w-3 h-3" />
          </a>
          <button
            onClick={() => setDeleteTargetId(item.id || item.filename)}
            className="p-1 rounded bg-slate-900 border border-slate-800 hover:border-rose-500 hover:text-rose-400 text-slate-400 transition-colors"
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
      {/* Top Banner */}
      <div className="glass-panel p-3.5 rounded-lg flex items-center justify-between border-l-4 border-l-cyan-400">
        <div className="flex items-center gap-2.5">
          <History className="w-5 h-5 text-cyan-400" />
          <div>
            <h3 className="font-bold text-slate-100 uppercase tracking-wider text-xs">
              RUNS & CHECKPOINT REPOSITORY
            </h3>
            <p className="text-[10px] text-slate-400">
              Safe deserialization architecture (weights_only=True) with atomic metadata verification
            </p>
          </div>
        </div>
        <button
          onClick={fetchHistory}
          className="inline-flex items-center gap-1.5 px-3 py-1 rounded bg-slate-900 border border-slate-700 hover:border-cyan-400 text-slate-200 transition-colors"
        >
          <RefreshCw className="w-3.5 h-3.5 text-cyan-400" /> REFRESH
        </button>
      </div>

      {/* Selected Run Metrics Visualization */}
      {selectedRunId && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="font-bold text-slate-200 uppercase tracking-wider text-xs flex items-center gap-1.5">
              <LineChart className="w-3.5 h-3.5 text-cyan-400" /> METRIC DYNAMICS FOR RUN:{" "}
              <span className="text-cyan-400">{selectedRunId}</span>
            </span>
            <span className="text-[10px] text-slate-400">
              {runMetrics.length} data points recorded
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
            <ChartPanel
              title="HISTORICAL LOSS CURVE"
              labels={chartSteps}
              datasets={[
                {
                  label: "Train Loss",
                  data: lossSeries,
                  borderColor: "#00f0ff",
                  backgroundColor: "rgba(0, 240, 255, 0.08)",
                  fill: true,
                },
                ...(hasValLoss
                  ? [
                      {
                        label: "Validation Loss",
                        data: valLossSeries,
                        borderColor: "#10b981",
                        borderDash: [4, 4],
                        pointRadius: 4,
                        spanGaps: true,
                      },
                    ]
                  : []),
              ]}
              yAxisLabel="Loss"
              height={220}
            />

            <ChartPanel
              title="LEARNING RATE CURVE"
              labels={chartSteps}
              datasets={[
                {
                  label: "LR",
                  data: lrSeries,
                  borderColor: "#818cf8",
                  backgroundColor: "rgba(129, 140, 248, 0.08)",
                  fill: true,
                },
              ]}
              yAxisLabel="LR"
              height={220}
            />
          </div>
        </div>
      )}

      {/* Runs History Table */}
      <div className="space-y-2">
        <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider flex items-center gap-1.5">
          <Layers className="w-3.5 h-3.5 text-cyan-400" /> TRAINING EXPERIMENTS HISTORY ({runs.length})
        </h3>
        <DataTable
          data={runs}
          columns={runColumns}
          keyExtractor={(r) => r.id}
          searchPlaceholder="Search runs by ID..."
        />
      </div>

      {/* Checkpoints Manager Table */}
      <div className="space-y-2 pt-2">
        <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider flex items-center gap-1.5">
          <Bookmark className="w-3.5 h-3.5 text-cyan-400" /> SAVED CHECKPOINTS ON DISK ({checkpoints.length})
        </h3>
        <DataTable
          data={checkpoints}
          columns={checkpointColumns}
          keyExtractor={(c) => c.id || c.filename}
          searchPlaceholder="Search checkpoints..."
        />
      </div>

      {/* Confirm Delete Dialog */}
      <ConfirmDialog
        isOpen={!!deleteTargetId}
        title="DELETE CHECKPOINT"
        message={`Are you sure you want to permanently delete checkpoint '${deleteTargetId}'? This action cannot be undone.`}
        confirmLabel="DELETE CHECKPOINT"
        cancelLabel="CANCEL"
        variant="danger"
        onConfirm={handleDeleteCheckpoint}
        onCancel={() => setDeleteTargetId(null)}
      />
    </div>
  );
};
