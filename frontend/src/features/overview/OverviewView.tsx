"use client";

import React, { useState, useEffect } from "react";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useHardwareTelemetry } from "@/hooks/useHardwareTelemetry";
import { useSystemStatus } from "@/hooks/useSystemStatus";
import { api } from "@/services/api";
import { MetricCard } from "@/components/ui/MetricCard";
import { TelemetryCard } from "@/components/ui/TelemetryCard";
import { ChartPanel } from "@/components/ui/ChartPanel";
import { StatusIndicator } from "@/components/ui/StatusIndicator";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { formatCompactNumber, formatNumber, formatDuration, formatStepLatency } from "@/lib/utils";
import {
  Activity,
  Play,
  Pause,
  Square,
  Bookmark,
  CheckCircle2,
  TrendingDown,
  Gauge,
  Clock,
  Layers,
  Sparkles,
  Sliders,
} from "lucide-react";

export const OverviewView: React.FC = () => {
  const {
    currentState,
    latestMetric,
    metricsHistory,
    recentAlerts,
    recentEvents,
    loadRunHistory,
    latestValLoss,
    latestValStep,
    latestValTimestamp,
    valStatus,
    activeTrainingTime,
  } = useWebSocket();
  const { status, refresh: refreshStatus } = useSystemStatus(3000);
  const { telemetry } = useHardwareTelemetry(2000);

  const [confirmStop, setConfirmStop] = useState(false);
  const [lrMultiplier, setLrMultiplier] = useState(1.0);
  const [lrFeedback, setLrFeedback] = useState<string | null>(null);

  // Sync historical metrics from database if buffer is currently empty
  useEffect(() => {
    if (status?.run_id && metricsHistory.length === 0) {
      loadRunHistory(status.run_id);
    }
  }, [status?.run_id, metricsHistory.length, loadRunHistory]);

  const effectiveState = currentState !== "IDLE" ? currentState : status?.state || "IDLE";
  const isRunning = effectiveState === "RUNNING";
  const isPaused = effectiveState === "PAUSED";
  const activeMetrics = latestMetric || status?.latest_metrics;

  const currentStep = activeMetrics?.step ?? 0;
  const currentLoss = activeMetrics?.loss ?? (activeMetrics as any)?.train_loss ?? 0;
  const valLoss = activeMetrics?.val_loss ?? null;
  const effectiveValLoss = valLoss !== null ? valLoss : (latestValLoss ?? activeMetrics?.latest_val_loss ?? null);
  const effectiveValStep = activeMetrics?.latest_val_step ?? latestValStep ?? null;

  const muonLr = activeMetrics?.muon_lr ?? activeMetrics?.learning_rate ?? (activeMetrics as any)?.lr ?? 0;
  const adamwLr = activeMetrics?.adamw_lr ?? 0;
  const tokensSeen = activeMetrics?.tokens_seen ?? (activeMetrics as any)?.tokens ?? 0;
  const tokPerSec = activeMetrics?.tokens_per_second ?? (activeMetrics as any)?.tok_per_sec ?? 0;
  const stepTimeMs = activeMetrics?.step_time_ms ?? ((activeMetrics as any)?.dt ? (activeMetrics as any).dt * 1000 : 0);
  const gradNorm = activeMetrics?.grad_norm ?? null;

  // Chart telemetry series
  const chartSteps = metricsHistory.map((m) => m.step);
  const lossSeries = metricsHistory.map((m) => (m.loss !== undefined ? m.loss : (m as any).train_loss ?? 0));
  const valLossSeries = metricsHistory.map((m) =>
    m.val_loss !== null && m.val_loss !== undefined ? Number(m.val_loss) : null
  );
  const hasValLoss = metricsHistory.some((m) => m.val_loss !== null && m.val_loss !== undefined);
  const lrSeries = metricsHistory.map((m) => (m.learning_rate !== undefined ? m.learning_rate : (m as any).lr ?? (m as any).muon_lr ?? 0));
  const stepDurationSeries = metricsHistory.map((m) => (m.step_time_ms !== undefined ? m.step_time_ms : 0));
  const gradNormSeries = metricsHistory.map((m) =>
    m.grad_norm !== null && m.grad_norm !== undefined ? Number(m.grad_norm) : null
  );

  // Quick Action handlers
  const handleTogglePause = async () => {
    try {
      if (isRunning) {
        await api.pauseRun();
      } else if (isPaused) {
        await api.resumeRun();
      }
      refreshStatus();
    } catch (e: any) {
      alert(`Action error: ${e.message}`);
    }
  };

  const handleStopRun = async () => {
    try {
      await api.stopRun();
      setConfirmStop(false);
      refreshStatus();
    } catch (e: any) {
      alert(`Stop error: ${e.message}`);
    }
  };

  const handleManualCheckpoint = async () => {
    try {
      await api.checkpointRun();
      alert("Checkpoint requested! Worker will save upon completing next micro-step.");
    } catch (e: any) {
      alert(`Checkpoint error: ${e.message}`);
    }
  };

  const handleTriggerVal = async () => {
    try {
      await api.validateRun();
      alert("Validation sweep triggered!");
    } catch (e: any) {
      alert(`Validation error: ${e.message}`);
    }
  };

  const handleApplyLrMultiplier = async () => {
    try {
      await api.updateLr({ multiplier: lrMultiplier });
      setLrFeedback(`LR scaled by ${lrMultiplier}x`);
      setTimeout(() => setLrFeedback(null), 3000);
    } catch (e: any) {
      alert(`LR scale error: ${e.message}`);
    }
  };

  return (
    <div className="space-y-4">
      {/* Top Banner: Real-Time Command Bar */}
      <div className="glass-panel p-3.5 rounded-lg flex flex-wrap items-center justify-between gap-3 border-l-4 border-l-cyan-400">
        <div className="flex items-center gap-3">
          <StatusIndicator state={effectiveState} size="lg" />
          <div className="flex flex-col">
            <span className="font-mono text-xs font-bold text-slate-100 flex items-center gap-1.5">
              <span>ACTIVE SESSION:</span>
              <span className="text-cyan-400">{status?.run_id || "No active run"}</span>
            </span>
            <span className="text-[10px] font-mono text-slate-400">
              Target Architecture: RoCanon-48.5M (Strict Invariant: 48,531,968 params)
            </span>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2">
          {(isRunning || isPaused) && (
            <>
              <button
                onClick={handleTogglePause}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-700 hover:border-cyan-400 text-xs font-mono text-slate-200 transition-all hover:shadow-[0_0_10px_rgba(0,240,255,0.2)]"
              >
                {isRunning ? (
                  <>
                    <Pause className="w-3.5 h-3.5 text-amber-400" /> PAUSE
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5 text-cyan-400" /> RESUME
                  </>
                )}
              </button>

              <button
                onClick={handleManualCheckpoint}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-700 hover:border-emerald-400 text-xs font-mono text-slate-200 transition-all"
              >
                <Bookmark className="w-3.5 h-3.5 text-emerald-400" /> CHECKPOINT
              </button>

              <button
                onClick={handleTriggerVal}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-700 hover:border-sky-400 text-xs font-mono text-slate-200 transition-all"
              >
                <CheckCircle2 className="w-3.5 h-3.5 text-sky-400" /> VALIDATE
              </button>

              <button
                onClick={() => setConfirmStop(true)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-950/40 border border-rose-800/80 hover:bg-rose-900/60 text-xs font-mono text-rose-300 transition-all"
              >
                <Square className="w-3.5 h-3.5 text-rose-400" /> STOP
              </button>
            </>
          )}

          {!isRunning && !isPaused && (
            <a
              href="/training"
              className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-mono font-bold shadow-[0_0_12px_rgba(0,240,255,0.3)] transition-all"
            >
              <Play className="w-3.5 h-3.5" /> LAUNCH EXPERIMENT
            </a>
          )}
        </div>
      </div>

      {/* 8 Metric Telemetry Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2.5">
        <MetricCard
          title="TRAIN LOSS"
          value={currentLoss > 0 ? formatNumber(currentLoss, 4) : "—"}
          glowColor="cyan"
          icon={<TrendingDown className="w-3.5 h-3.5" />}
          subtitle="Cross Entropy"
        />
        <MetricCard
          title="VAL LOSS"
          value={effectiveValLoss !== null ? formatNumber(effectiveValLoss, 4) : isRunning ? "Awaiting..." : "—"}
          glowColor="emerald"
          icon={<CheckCircle2 className="w-3.5 h-3.5" />}
          subtitle={effectiveValStep ? `Step ${effectiveValStep}` : isRunning ? `Every ${activeMetrics?.eval_interval || 250} steps` : "Periodic Eval"}
        />
        <MetricCard
          title="STEP"
          value={currentStep.toLocaleString()}
          glowColor="cyan"
          icon={<Layers className="w-3.5 h-3.5" />}
          subtitle="Global"
        />
        <MetricCard
          title="TOKENS SEEN"
          value={formatCompactNumber(tokensSeen)}
          glowColor="violet"
          icon={<Activity className="w-3.5 h-3.5" />}
          subtitle="Cumulative"
        />
        <MetricCard
          title="THROUGHPUT"
          value={tokPerSec > 0 ? Math.round(tokPerSec) : "—"}
          unit="tok/s"
          glowColor="emerald"
          icon={<Gauge className="w-3.5 h-3.5" />}
          subtitle="Token Velocity"
        />
        <MetricCard
          title="LEARNING RATE"
          value={muonLr > 0 ? muonLr.toExponential(2) : "—"}
          glowColor="cyan"
          icon={<Sparkles className="w-3.5 h-3.5" />}
          subtitle={adamwLr > 0 ? `AdamW: ${adamwLr.toExponential(1)}` : "Muon Matrix"}
        />
        <MetricCard
          title="GRAD NORM"
          value={gradNorm !== null ? formatNumber(gradNorm, 3) : "—"}
          glowColor="amber"
          icon={<Activity className="w-3.5 h-3.5" />}
          subtitle="L2 Norm"
        />
        <MetricCard
          title="ACTIVE TIME"
          value={formatDuration(activeTrainingTime)}
          glowColor="violet"
          icon={<Clock className="w-3.5 h-3.5" />}
          subtitle={stepTimeMs > 0 ? formatStepLatency(stepTimeMs) : "Total Run Time"}
        />
      </div>

      {/* Charts & Hardware Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3.5">
        {/* Main Loss Plot (2 cols on lg) */}
        <div className="lg:col-span-2">
          <ChartPanel
            title="TRAINING & VALIDATION DYNAMICS"
            labels={chartSteps}
            datasets={[
              {
                label: "Train Loss",
                data: lossSeries,
                borderColor: "#00f0ff",
                backgroundColor: "rgba(0, 240, 255, 0.08)",
                fill: true,
                borderWidth: 2,
              },
              ...(hasValLoss
                ? [
                    {
                      label: "Validation Loss",
                      data: valLossSeries,
                      borderColor: "#10b981",
                      borderDash: [4, 4],
                      borderWidth: 2,
                      pointRadius: 4,
                      spanGaps: true,
                    },
                  ]
                : []),
            ]}
            yAxisLabel="Loss"
            height={260}
          />
        </div>

        {/* Real Hardware NVML Telemetry Card */}
        <div>
          <TelemetryCard telemetry={telemetry} className="h-full" />
        </div>
      </div>

      {/* Secondary Telemetry Plots & Dynamic Controls */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3.5">
        {/* Learning Rate Trajectory */}
        <ChartPanel
          title="LEARNING RATE TRAJECTORY"
          labels={chartSteps}
          datasets={[
            {
              label: "Effective LR",
              data: lrSeries,
              borderColor: "#818cf8",
              backgroundColor: "rgba(129, 140, 248, 0.08)",
              fill: true,
            },
          ]}
          yAxisLabel="LR"
          height={180}
        />

        {/* Gradient Stability */}
        <ChartPanel
          title="GRADIENT STABILITY (L2 NORM)"
          labels={chartSteps}
          datasets={[
            {
              label: "Grad Norm",
              data: gradNormSeries,
              borderColor: "#f59e0b",
              spanGaps: true,
            },
          ]}
          yAxisLabel="Norm"
          height={180}
        />

        {/* Step Latency Duration */}
        <ChartPanel
          title="STEP DURATION (LATENCY)"
          labels={chartSteps}
          datasets={[
            {
              label: "Duration (ms)",
              data: stepDurationSeries,
              borderColor: "#a855f7",
              backgroundColor: "rgba(168, 85, 247, 0.08)",
              fill: true,
            },
          ]}
          yAxisLabel="ms"
          height={180}
        />

        {/* Real-Time Dynamic LR Adjustment & Telemetry Feed */}
        <div className="glass-panel p-3.5 rounded-lg flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-800">
              <span className="font-mono text-xs font-bold text-slate-200 uppercase flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-cyan-400" /> DYNAMIC LR SCALING
              </span>
              {lrFeedback && (
                <span className="text-[10px] font-mono text-emerald-400 font-bold">
                  {lrFeedback}
                </span>
              )}
            </div>
            <p className="text-[11px] font-mono text-slate-400 mb-3">
              Scale Muon & AdamW learning rates in real time without interrupting training.
            </p>
            <div className="space-y-2">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-slate-400">Multiplier:</span>
                <span className="text-cyan-400 font-bold">{lrMultiplier.toFixed(2)}x</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="3.0"
                step="0.05"
                value={lrMultiplier}
                onChange={(e) => setLrMultiplier(parseFloat(e.target.value))}
                className="w-full accent-cyan-400 bg-slate-800 h-1.5 rounded cursor-pointer"
              />
              <div className="flex justify-between text-[10px] font-mono text-slate-400">
                <span>0.1x (Fine)</span>
                <span>1.0x (Default)</span>
                <span>3.0x (Aggressive)</span>
              </div>
            </div>
          </div>

          <div className="pt-3 border-t border-slate-800 flex items-center justify-between">
            <button
              onClick={() => setLrMultiplier(1.0)}
              className="text-xs font-mono text-slate-400 hover:text-slate-200"
            >
              Reset 1.0x
            </button>
            <button
              onClick={handleApplyLrMultiplier}
              disabled={!isRunning}
              className="px-3 py-1 rounded bg-cyan-600 hover:bg-cyan-500 disabled:opacity-40 disabled:hover:bg-cyan-600 text-xs font-mono font-bold text-white transition-all shadow-[0_0_10px_rgba(0,240,255,0.3)]"
            >
              APPLY LR
            </button>
          </div>
        </div>
      </div>

      {/* Activity & Alert Log Feed */}
      <div className="glass-panel p-3.5 rounded-lg">
        <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-3 bg-indigo-400 rounded-sm shadow-[0_0_6px_#818cf8]" />
            <h3 className="font-mono text-xs font-semibold text-slate-200 tracking-wider uppercase">
              ACTIVITY STREAM & RUNTIME EVENTS
            </h3>
          </div>
          <span className="text-[10px] font-mono text-slate-400">
            {recentEvents.length + recentAlerts.length} events logged
          </span>
        </div>

        <div className="max-h-44 overflow-y-auto space-y-1.5 font-mono text-xs">
          {recentAlerts.length === 0 && recentEvents.length === 0 ? (
            <div className="p-4 text-center text-slate-400 text-xs">
              Waiting for worker events and telemetry signals...
            </div>
          ) : (
            <>
              {recentAlerts.map((a, i) => (
                <div
                  key={`alert-${i}`}
                  className="p-2 rounded bg-slate-900/70 border border-slate-800/80 flex items-center justify-between text-xs"
                >
                  <div className="flex items-center gap-2">
                    <span
                      className={`text-[10px] px-1.5 py-0.5 rounded font-bold uppercase ${
                        a.level === "CRITICAL" || a.level === "ERROR"
                          ? "bg-rose-950/60 text-rose-400 border border-rose-800"
                          : a.level === "WARNING"
                          ? "bg-amber-950/60 text-amber-400 border border-amber-800"
                          : "bg-cyan-950/60 text-cyan-400 border border-cyan-800"
                      }`}
                    >
                      {a.level}
                    </span>
                    <span className="text-slate-300">{a.message}</span>
                  </div>
                  <span className="text-slate-400 text-[10px]">
                    {a.step ? `Step ${a.step}` : a.category}
                  </span>
                </div>
              ))}
              {recentEvents.slice(0, 10).map((e, i) => (
                <div
                  key={`ev-${i}`}
                  className="p-1.5 rounded bg-slate-900/40 border border-slate-800/40 flex items-center justify-between text-[11px] text-slate-400"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-cyan-400 font-bold">[{e.event_type}]</span>
                    <span className="truncate max-w-lg text-slate-300">
                      {typeof e.payload === "string"
                        ? e.payload
                        : JSON.stringify(e.payload)}
                    </span>
                  </div>
                  <span className="text-[10px] text-slate-400">
                    {e.timestamp ? new Date(e.timestamp).toLocaleTimeString() : "--:--:--"}
                  </span>
                </div>
              ))}
            </>
          )}
        </div>
      </div>

      {/* Confirm Stop Dialog */}
      <ConfirmDialog
        isOpen={confirmStop}
        title="HALT ACTIVE TRAINING SESSION"
        message="Are you sure you want to stop the active training worker? The current checkpoint and model weights will be safely committed."
        confirmLabel="HALT TRAINING"
        cancelLabel="CONTINUE TRAINING"
        variant="danger"
        onConfirm={handleStopRun}
        onCancel={() => setConfirmStop(false)}
      />
    </div>
  );
};
