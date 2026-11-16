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
  Terminal,
  Cpu,
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
    <div className="space-y-4 max-w-[1600px] mx-auto animate-in fade-in duration-200">
      {/* Top Banner: Real-Time Neural Command Center Bar */}
      <div className="glass-panel p-3.5 rounded-lg flex flex-wrap items-center justify-between gap-3 border-l-2 border-l-[#45CFFF] relative overflow-hidden">
        <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-[#45CFFF]/30 via-transparent to-transparent pointer-events-none" />

        <div className="flex items-center gap-3.5 min-w-0">
          <StatusIndicator state={effectiveState} size="lg" />
          <div className="flex flex-col min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-mono text-xs font-bold text-[#ECF3FC] tracking-wider uppercase flex items-center gap-1.5">
                SESSION: <span className="text-[#45CFFF] font-mono-num">{status?.run_id || "IDLE / NO RUN ACTIVE"}</span>
              </span>
              <span className="tech-badge tech-badge-violet">
                {status?.active_network || "Neural Architecture"}
              </span>
              <span className="tech-badge">
                LOCAL-FIRST ENGINE
              </span>
            </div>
            <span className="text-[11px] font-mono text-[#91A0B4] mt-0.5 truncate">
              {isRunning
                ? "Active PyTorch execution loop • Hardware NVML & real-time telemetry streaming"
                : "Workstation ready for architecture exploration, dataset staging and model training"}
            </span>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2">
          {(isRunning || isPaused) && (
            <>
              <button
                onClick={handleTogglePause}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-[#182333] border border-[#273548] hover:border-[#384C66] text-xs font-mono text-[#ECF3FC] transition-all shadow-sm"
              >
                {isRunning ? (
                  <>
                    <Pause className="w-3.5 h-3.5 text-[#F2B85B]" /> PAUSE
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5 text-[#45CFFF]" /> RESUME
                  </>
                )}
              </button>

              <button
                onClick={handleManualCheckpoint}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-[#182333] border border-[#273548] hover:border-[#35D49A]/50 text-xs font-mono text-[#ECF3FC] transition-all"
              >
                <Bookmark className="w-3.5 h-3.5 text-[#35D49A]" /> CHECKPOINT
              </button>

              <button
                onClick={handleTriggerVal}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-[#182333] border border-[#273548] hover:border-[#45CFFF]/50 text-xs font-mono text-[#ECF3FC] transition-all"
              >
                <CheckCircle2 className="w-3.5 h-3.5 text-[#45CFFF]" /> VALIDATE
              </button>

              <button
                onClick={() => setConfirmStop(true)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-[#FF657A]/10 border border-[#FF657A]/40 hover:bg-[#FF657A]/20 text-xs font-mono text-[#FF657A] transition-all font-semibold"
              >
                <Square className="w-3.5 h-3.5 text-[#FF657A]" /> STOP
              </button>
            </>
          )}

          {!isRunning && !isPaused && (
            <a
              href="/training"
              className="inline-flex items-center gap-2 px-4 py-1.5 rounded-md bg-[#45CFFF] hover:bg-[#80DEFF] text-[#090D14] text-xs font-mono font-bold shadow-[0_0_12px_rgba(69,207,255,0.25)] transition-all"
            >
              <Play className="w-3.5 h-3.5 fill-current" /> LAUNCH EXPERIMENT
            </a>
          )}
        </div>
      </div>

      {/* 8 Primary Telemetry Cards */}
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
          subtitle="Velocity"
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
          subtitle={stepTimeMs > 0 ? formatStepLatency(stepTimeMs) : "Total Elapsed"}
        />
      </div>

      {/* Main Dynamics & Telemetry Section */}
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
                borderColor: "#45CFFF",
                backgroundColor: "rgba(69, 207, 255, 0.06)",
                fill: true,
                borderWidth: 2,
              },
              ...(hasValLoss
                ? [
                    {
                      label: "Validation Loss",
                      data: valLossSeries,
                      borderColor: "#35D49A",
                      borderDash: [4, 4],
                      borderWidth: 2,
                      pointRadius: 3.5,
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
              borderColor: "#9B8CFF",
              backgroundColor: "rgba(155, 140, 255, 0.06)",
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
              borderColor: "#F2B85B",
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
              borderColor: "#80DEFF",
              backgroundColor: "rgba(128, 222, 255, 0.06)",
              fill: true,
            },
          ]}
          yAxisLabel="ms"
          height={180}
        />

        {/* Real-Time Dynamic LR Adjustment & Telemetry Feed */}
        <div className="glass-panel p-3.5 rounded-lg flex flex-col justify-between relative overflow-hidden">
          <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#45CFFF]/15 to-transparent pointer-events-none" />

          <div>
            <div className="flex items-center justify-between pb-2 mb-2 border-b border-[#273548]">
              <span className="font-mono text-xs font-semibold text-[#ECF3FC] uppercase flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-[#45CFFF]" /> DYNAMIC LR SCALING
              </span>
              {lrFeedback && (
                <span className="text-[10px] font-mono text-[#35D49A] font-bold">
                  {lrFeedback}
                </span>
              )}
            </div>
            <p className="text-[11px] font-mono text-[#91A0B4] mb-3 leading-relaxed">
              Scale optimizer learning rates on-the-fly during active training runs.
            </p>
            <div className="space-y-2">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-[#91A0B4]">Multiplier:</span>
                <span className="text-[#45CFFF] font-bold font-mono-num">{lrMultiplier.toFixed(2)}x</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="3.0"
                step="0.05"
                value={lrMultiplier}
                onChange={(e) => setLrMultiplier(parseFloat(e.target.value))}
                className="w-full accent-[#45CFFF] bg-[#0E1420] h-1.5 rounded cursor-pointer"
              />
              <div className="flex justify-between text-[10px] font-mono text-[#5E7188]">
                <span>0.1x (Fine)</span>
                <span>1.0x (Default)</span>
                <span>3.0x (Aggressive)</span>
              </div>
            </div>
          </div>

          <div className="pt-3 border-t border-[#273548] flex items-center justify-between mt-3">
            <button
              onClick={() => setLrMultiplier(1.0)}
              className="text-xs font-mono text-[#91A0B4] hover:text-[#ECF3FC] transition-colors"
            >
              Reset 1.0x
            </button>
            <button
              onClick={handleApplyLrMultiplier}
              disabled={!isRunning}
              className="px-3 py-1 rounded-md bg-[#45CFFF] hover:bg-[#80DEFF] disabled:opacity-30 disabled:hover:bg-[#45CFFF] text-xs font-mono font-bold text-[#090D14] transition-all shadow-sm"
            >
              APPLY LR
            </button>
          </div>
        </div>
      </div>

      {/* Activity & Alert Log Feed */}
      <div className="glass-panel p-3.5 rounded-lg relative overflow-hidden">
        <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#9B8CFF]/20 to-transparent pointer-events-none" />

        <div className="flex items-center justify-between pb-2 mb-2 border-b border-[#273548]">
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-3 bg-[#9B8CFF] rounded-sm shadow-[0_0_6px_rgba(155,140,255,0.4)]" />
            <h3 className="font-mono text-xs font-semibold text-[#ECF3FC] tracking-wider uppercase">
              ACTIVITY STREAM & RUNTIME SIGNALS
            </h3>
          </div>
          <span className="text-[10px] font-mono text-[#91A0B4]">
            {recentEvents.length + recentAlerts.length} events logged
          </span>
        </div>

        <div className="max-h-44 overflow-y-auto space-y-1.5 font-mono text-xs pr-1">
          {recentAlerts.length === 0 && recentEvents.length === 0 ? (
            <div className="p-4 text-center text-[#5E7188] text-xs">
              Waiting for worker events and telemetry signals...
            </div>
          ) : (
            <>
              {recentAlerts.map((a, i) => (
                <div
                  key={`alert-${i}`}
                  className="p-2 rounded bg-[#0E1420] border border-[#273548] flex items-center justify-between text-xs"
                >
                  <div className="flex items-center gap-2">
                    <span
                      className={`text-[10px] px-1.5 py-0.5 rounded font-bold uppercase ${
                        a.level === "CRITICAL" || a.level === "ERROR"
                          ? "bg-[#FF657A]/15 text-[#FF657A] border border-[#FF657A]/30"
                          : a.level === "WARNING"
                          ? "bg-[#F2B85B]/15 text-[#F2B85B] border border-[#F2B85B]/30"
                          : "bg-[#45CFFF]/15 text-[#45CFFF] border border-[#45CFFF]/30"
                      }`}
                    >
                      {a.level}
                    </span>
                    <span className="text-[#ECF3FC]">{a.message}</span>
                  </div>
                  <span className="text-[#5E7188] text-[10px]">
                    {a.step ? `Step ${a.step}` : a.category}
                  </span>
                </div>
              ))}
              {recentEvents.slice(0, 10).map((e, i) => (
                <div
                  key={`ev-${i}`}
                  className="p-1.5 rounded bg-[#0E1420]/60 border border-[#273548]/60 flex items-center justify-between text-[11px] text-[#91A0B4]"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-[#45CFFF] font-bold">[{e.event_type}]</span>
                    <span className="truncate max-w-lg text-[#ECF3FC]">
                      {typeof e.payload === "string"
                        ? e.payload
                        : JSON.stringify(e.payload)}
                    </span>
                  </div>
                  <span className="text-[10px] text-[#5E7188]">
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