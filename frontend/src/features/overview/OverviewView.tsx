"use client";

import React, { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useHardwareTelemetry } from "@/hooks/useHardwareTelemetry";
import { useSystemStatus } from "@/hooks/useSystemStatus";
import { api } from "@/services/api";
import { MetricCard } from "@/components/ui/MetricCard";
import { TelemetryCard } from "@/components/ui/TelemetryCard";
import { ChartPanel } from "@/components/ui/ChartPanel";
import { StatusIndicator } from "@/components/ui/StatusIndicator";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { SaveModelModal } from "@/components/ui/SaveModelModal";
import { Button, buttonClasses } from "@/components/ui/Button";
import { InlineAlert, AlertTone } from "@/components/ui/InlineAlert";
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
  RefreshCw,
  HardDrive,
} from "lucide-react";

type PendingAction = "pause" | "checkpoint" | "validate" | "stop" | "lr" | null;

interface Feedback {
  tone: AlertTone;
  message: string;
  title?: string;
}

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
    valStatus,
    activeTrainingTime,
  } = useWebSocket();
  const {
    status,
    error: statusError,
    isLoading: statusLoading,
    refresh: refreshStatus,
  } = useSystemStatus(3000);
  const {
    telemetry,
    error: telemetryError,
    isLoading: telemetryLoading,
  } = useHardwareTelemetry(2000);

  const [confirmStop, setConfirmStop] = useState(false);
  const [isSaveModalOpen, setIsSaveModalOpen] = useState(false);
  const [lrMultiplier, setLrMultiplier] = useState(1.0);
  const [lrFeedback, setLrFeedback] = useState<string | null>(null);
  const [pendingAction, setPendingAction] = useState<PendingAction>(null);
  const [feedback, setFeedback] = useState<Feedback | null>(null);
  const feedbackTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Sync historical metrics from database if buffer is currently empty
  useEffect(() => {
    if (status?.run_id && metricsHistory.length === 0) {
      loadRunHistory(status.run_id);
    }
  }, [status?.run_id, metricsHistory.length, loadRunHistory]);

  useEffect(() => {
    return () => {
      if (feedbackTimer.current) clearTimeout(feedbackTimer.current);
    };
  }, []);

  const notify = (tone: AlertTone, message: string, title?: string) => {
    setFeedback({ tone, message, title });
    if (feedbackTimer.current) clearTimeout(feedbackTimer.current);
    if (tone !== "error") {
      feedbackTimer.current = setTimeout(() => setFeedback(null), 6000);
    }
  };

  const runAction = async (
    action: NonNullable<PendingAction>,
    request: () => Promise<unknown>,
    successMessage: string,
    successTitle?: string
  ) => {
    if (pendingAction) return;
    setPendingAction(action);
    setFeedback(null);
    try {
      await request();
      notify("success", successMessage, successTitle);
      await refreshStatus();
    } catch (e: any) {
      notify("error", e?.message || "The request failed. Check the worker logs and try again.");
    } finally {
      setPendingAction(null);
    }
  };

  const effectiveState = currentState !== "IDLE" ? currentState : status?.state || "IDLE";
  const isRunning = effectiveState === "RUNNING";
  const isPaused = effectiveState === "PAUSED";
  const isActive = isRunning || isPaused;
  const activeMetrics = latestMetric || status?.latest_metrics;

  const currentStep = activeMetrics?.step ?? 0;
  const currentLoss = activeMetrics?.loss ?? (activeMetrics as any)?.train_loss ?? 0;
  const valLoss = activeMetrics?.val_loss ?? null;
  const effectiveValLoss = valLoss !== null ? valLoss : latestValLoss ?? activeMetrics?.latest_val_loss ?? null;
  const effectiveValStep = activeMetrics?.latest_val_step ?? latestValStep ?? null;

  const muonLr = activeMetrics?.muon_lr ?? activeMetrics?.learning_rate ?? (activeMetrics as any)?.lr ?? 0;
  const adamwLr = activeMetrics?.adamw_lr ?? 0;
  const tokensSeen = activeMetrics?.tokens_seen ?? (activeMetrics as any)?.tokens ?? 0;
  const tokPerSec = activeMetrics?.tokens_per_second ?? (activeMetrics as any)?.tok_per_sec ?? 0;
  const stepTimeMs = activeMetrics?.step_time_ms ?? ((activeMetrics as any)?.dt ? (activeMetrics as any).dt * 1000 : 0);
  const gradNorm = activeMetrics?.grad_norm ?? null;
  const evalInterval = activeMetrics?.eval_interval || 250;

  // Dataset & Epoch progress
  const activeDurationMode = (activeMetrics as any)?.duration_mode || "steps";
  const currentEpoch = (activeMetrics as any)?.current_epoch || 1;
  const displayTargetEpochs = (activeMetrics as any)?.target_epochs || 2;
  const tokensInEpoch = (activeMetrics as any)?.tokens_in_epoch || 0;
  const tokensPerEpoch = (activeMetrics as any)?.tokens_per_epoch || 0;
  const epochPercent = tokensPerEpoch > 0 ? Math.min(100, Math.round((tokensInEpoch / tokensPerEpoch) * 100)) : 0;

  // Split LR Scheduling & Evaluation Tracking
  const schedPhase = (activeMetrics as any)?.sched_phase || "stable";
  const schedFactor = (activeMetrics as any)?.sched_factor ?? 1.0;
  const nextValStep = (activeMetrics as any)?.next_val_step ?? (currentStep > 0 ? currentStep + (evalInterval - (currentStep % evalInterval)) : evalInterval);

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

  const metricsLoading = statusLoading && !activeMetrics;
  const chartsLoading = statusLoading && metricsHistory.length === 0;
  const anyPending = pendingAction !== null;
  const valSubtitle = effectiveValStep
    ? `Step ${effectiveValStep}`
    : isRunning
    ? valStatus === "evaluating"
      ? "Evaluating…"
      : `Every ${evalInterval} steps`
    : "Periodic evaluation";

  // Quick Action handlers
  const handleTogglePause = () => {
    if (isRunning) {
      void runAction("pause", () => api.pauseRun(), "Training paused. The worker keeps its state and can be resumed.", "Paused");
    } else if (isPaused) {
      void runAction("pause", () => api.resumeRun(), "Training resumed.", "Resumed");
    }
  };

  const handleStopRun = async () => {
    if (pendingAction) return;
    setPendingAction("stop");
    try {
      await api.stopRun();
      setConfirmStop(false);
      notify("success", "Stop requested. The current checkpoint and model weights will be committed.", "Stopping run");
      await refreshStatus();
    } catch (e: any) {
      notify("error", e?.message || "Stop request failed. The run may still be active.");
    } finally {
      setPendingAction(null);
    }
  };

  const handleManualCheckpoint = () => {
    void runAction(
      "checkpoint",
      () => api.checkpointRun(),
      "Checkpoint requested. The worker will save it after the next micro-step.",
      "Checkpoint queued"
    );
  };

  const handleTriggerVal = () => {
    void runAction(
      "validate",
      () => api.validateRun(),
      "Validation sweep triggered. Validation loss will appear on the next eval step.",
      "Validation queued"
    );
  };

  const handleApplyLrMultiplier = async () => {
    if (pendingAction) return;
    setPendingAction("lr");
    setFeedback(null);
    try {
      await api.updateLr({ multiplier: lrMultiplier });
      setLrFeedback(`Applied ${lrMultiplier.toFixed(2)}×`);
      notify("success", `Learning rate scaled by ${lrMultiplier.toFixed(2)}× for the active run.`, "LR updated");
      setTimeout(() => setLrFeedback(null), 4000);
    } catch (e: any) {
      notify("error", e?.message || "Could not scale the learning rate.");
    } finally {
      setPendingAction(null);
    }
  };

  return (
    <div className="space-y-3 max-w-[1600px] mx-auto animate-in fade-in duration-200">
      {/* Session banner */}
      <section className="panel relative overflow-hidden p-3.5 flex flex-wrap items-center justify-between gap-3">
        {isActive && (
          <span
            className="absolute left-0 top-3 bottom-3 w-[2px] rounded-r bg-[#45CFFF]"
            aria-hidden="true"
          />
        )}

        <div className="flex items-center gap-3.5 min-w-0 pl-1">
          <StatusIndicator state={effectiveState} size="lg" />
          <div className="flex flex-col min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-mono text-[11px] font-bold text-[#ECF3FC] tracking-wider uppercase flex items-center gap-1.5">
                Session:
                <span className="text-[#45CFFF] font-mono-num normal-case tracking-normal">
                  {status?.run_id || "no run active"}
                </span>
              </span>
              <span className="tech-badge tech-badge-violet">
                {status?.active_network || "Neural Architecture"}
              </span>
              <span className="tech-badge">Local-first engine</span>
            </div>
            <span className="text-[11px] font-mono text-[#91A0B4] mt-0.5 truncate">
              {isRunning
                ? "Active PyTorch execution loop · Hardware NVML & real-time telemetry streaming"
                : isPaused
                ? "Run paused · worker state preserved, resume when ready"
                : "Workstation ready for architecture exploration, dataset staging and model training"}
            </span>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {isActive ? (
            <>
              <Button
                variant="secondary"
                size="sm"
                onClick={handleTogglePause}
                loading={pendingAction === "pause"}
                disabled={anyPending}
                icon={isRunning ? <Pause className="w-3.5 h-3.5 text-[#F2B85B]" /> : <Play className="w-3.5 h-3.5 text-[#45CFFF]" />}
              >
                {isRunning ? "Pause" : "Resume"}
              </Button>

              <Button
                variant="secondary"
                size="sm"
                onClick={handleManualCheckpoint}
                loading={pendingAction === "checkpoint"}
                disabled={anyPending}
                icon={<Bookmark className="w-3.5 h-3.5 text-[#35D49A]" />}
              >
                Checkpoint
              </Button>

              <Button
                variant="ghost"
                size="sm"
                onClick={handleTriggerVal}
                loading={pendingAction === "validate"}
                disabled={anyPending}
                icon={<CheckCircle2 className="w-3.5 h-3.5 text-[#45CFFF]" />}
              >
                Validate
              </Button>

              <Button
                variant="danger"
                size="sm"
                onClick={() => setConfirmStop(true)}
                disabled={anyPending}
                icon={<Square className="w-3 h-3 fill-current" />}
              >
                Stop
              </Button>

              <Button
                variant="secondary"
                size="sm"
                onClick={() => setIsSaveModalOpen(true)}
                icon={<HardDrive className="w-3.5 h-3.5 text-[#45CFFF]" />}
              >
                Save Local Model
              </Button>
            </>
          ) : (
            <div className="flex items-center gap-2">
              <Button
                variant="secondary"
                size="sm"
                onClick={() => setIsSaveModalOpen(true)}
                icon={<HardDrive className="w-3.5 h-3.5 text-[#45CFFF]" />}
              >
                Save Local Model
              </Button>
              <Link href="/training" className={buttonClasses("primary")}>
                <Play className="w-3.5 h-3.5 fill-current" aria-hidden="true" />
                Launch experiment
              </Link>
            </div>
          )}
        </div>
      </section>

      {/* Action feedback & link health */}
      {statusError && !status && (
        <InlineAlert
          tone="warning"
          title="Status link degraded"
          message={statusError}
          action={
            <Button
              variant="secondary"
              size="sm"
              onClick={() => void refreshStatus()}
              icon={<RefreshCw className="w-3.5 h-3.5 text-[#45CFFF]" />}
            >
              Retry
            </Button>
          }
        />
      )}

      {feedback && (
        <InlineAlert
          tone={feedback.tone}
          title={feedback.title}
          message={feedback.message}
          onDismiss={() => setFeedback(null)}
        />
      )}

      {/* Primary Live Telemetry Strip matching Training Console */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5 select-none font-mono">
        {/* Train Loss */}
        <div className="glass-panel p-2.5 rounded-lg border-l-2 border-l-[#45CFFF]">
          <div className="text-[10px] text-[#91A0B4] uppercase font-semibold flex items-center justify-between">
            <span>TRAIN LOSS</span>
            <TrendingDown className="w-3 h-3 text-[#45CFFF]" />
          </div>
          <div className="text-base font-bold font-mono-num text-[#ECF3FC] mt-1">
            {currentLoss > 0 ? formatNumber(currentLoss, 4) : "—"}
          </div>
          <div className="text-[9px] text-[#5E7188] mt-0.5">
            Step {currentStep.toLocaleString()}
          </div>
        </div>

        {/* Validation Loss */}
        <div className="glass-panel p-2.5 rounded-lg border-l-2 border-l-[#35D49A]">
          <div className="text-[10px] text-[#91A0B4] uppercase font-semibold flex items-center justify-between">
            <span>VAL LOSS</span>
            <CheckCircle2 className="w-3 h-3 text-[#35D49A]" />
          </div>
          <div className="text-base font-bold font-mono-num text-[#35D49A] mt-1">
            {effectiveValLoss !== null ? (
              formatNumber(effectiveValLoss, 4)
            ) : isRunning ? (
              <span className="text-xs text-[#F2B85B] animate-pulse">Awaiting val...</span>
            ) : (
              "—"
            )}
          </div>
          <div className="text-[9px] text-[#5E7188] mt-0.5">
            {effectiveValStep ? `Evaluated at Step ${effectiveValStep}` : `Next eval: Step ${nextValStep}`}
          </div>
        </div>

        {/* Dataset & Objective Progress */}
        <div className="glass-panel p-2.5 rounded-lg border-l-2 border-l-[#9B8CFF]">
          <div className="text-[10px] text-[#91A0B4] uppercase font-semibold flex items-center justify-between">
            <span>DATASET PROGRESS</span>
            <Activity className="w-3 h-3 text-[#9B8CFF]" />
          </div>
          <div className="text-sm font-bold font-mono-num text-[#ECF3FC] mt-1">
            {activeDurationMode === "epochs" ? (
              <span>Epoch {currentEpoch} / {displayTargetEpochs}</span>
            ) : (
              <span>Step {currentStep.toLocaleString()}</span>
            )}
          </div>
          <div className="text-[9px] text-[#5E7188] mt-0.5 truncate">
            {tokensPerEpoch > 0 ? (
              `${formatCompactNumber(tokensInEpoch)} / ${formatCompactNumber(tokensPerEpoch)} tok (${epochPercent}%)`
            ) : (
              `${formatCompactNumber(tokensSeen)} cumulative tokens`
            )}
          </div>
        </div>

        {/* Latency & Total Active Training Duration */}
        <div className="glass-panel p-2.5 rounded-lg border-l-2 border-l-[#80DEFF]">
          <div className="text-[10px] text-[#91A0B4] uppercase font-semibold flex items-center justify-between">
            <span>TIME & LATENCY</span>
            <Clock className="w-3 h-3 text-[#80DEFF]" />
          </div>
          <div className="text-base font-bold font-mono-num text-[#80DEFF] mt-1">
            {formatDuration(activeTrainingTime)}
          </div>
          <div className="text-[9px] text-[#5E7188] mt-0.5 flex justify-between">
            <span>Iter: {formatStepLatency(stepTimeMs)}</span>
            <span>{tokPerSec > 0 ? `${Math.round(tokPerSec)} tok/s` : ""}</span>
          </div>
        </div>

        {/* Muon Effective LR (Matrices) */}
        <div className="glass-panel p-2.5 rounded-lg border-l-2 border-l-[#45CFFF]">
          <div className="text-[10px] text-[#91A0B4] uppercase font-semibold flex items-center justify-between">
            <span>MUON LR (MATRIX)</span>
            <Sparkles className="w-3 h-3 text-[#45CFFF]" />
          </div>
          <div className="text-sm font-bold font-mono-num text-[#45CFFF] mt-1 flex items-center gap-1.5">
            <span>{muonLr > 0 ? muonLr.toExponential(2) : "—"}</span>
          </div>
          <div className="text-[9px] text-[#5E7188] mt-0.5">
            Phase: {schedPhase} ({(schedFactor * 100).toFixed(0)}%)
          </div>
        </div>

        {/* AdamW Effective LR & Next Eval */}
        <div className="glass-panel p-2.5 rounded-lg border-l-2 border-l-[#35D49A]">
          <div className="text-[10px] text-[#91A0B4] uppercase font-semibold flex items-center justify-between">
            <span>ADAMW LR (VECTORS)</span>
            <Gauge className="w-3 h-3 text-[#35D49A]" />
          </div>
          <div className="text-sm font-bold font-mono-num text-[#35D49A] mt-1">
            {adamwLr > 0 ? adamwLr.toExponential(2) : "—"}
          </div>
          <div className="text-[9px] text-[#5E7188] mt-0.5">
            Next eval: Step {nextValStep}
          </div>
        </div>
      </div>

      {/* Main dynamics & hardware */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        <div className="lg:col-span-2">
          <ChartPanel
            title="Training & validation dynamics"
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
            loading={chartsLoading}
            emptyTitle="No metrics recorded"
            emptyHint="Launch an experiment to stream train and validation loss here."
          />
        </div>

        <TelemetryCard
          telemetry={telemetry}
          error={telemetryError}
          isLoading={telemetryLoading}
          className="h-full"
        />
      </div>

      {/* Secondary plots & dynamic controls */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
        <ChartPanel
          title="Learning rate trajectory"
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
          loading={chartsLoading}
          emptyTitle="No LR data"
          emptyHint="The schedule curve appears after the first steps."
        />

        <ChartPanel
          title="Gradient stability"
          labels={chartSteps}
          datasets={[
            {
              label: "Grad Norm",
              data: gradNormSeries,
              borderColor: "#F2B85B",
              spanGaps: true,
            },
          ]}
          yAxisLabel="L2 norm"
          height={180}
          loading={chartsLoading}
          emptyTitle="No gradient data"
          emptyHint="Norm telemetry appears once the worker reports it."
        />

        <ChartPanel
          title="Step duration"
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
          loading={chartsLoading}
          emptyTitle="No latency data"
          emptyHint="Per-step timing appears while training runs."
        />

        {/* Live LR control */}
        <section className="panel p-3.5 flex flex-col justify-between gap-3">
          <div>
            <div className="panel-header">
              <h3 className="panel-title">
                <Sliders className="w-3.5 h-3.5 text-[#45CFFF]" aria-hidden="true" />
                Dynamic LR scaling
              </h3>
              {lrFeedback && (
                <span className="font-mono text-[10px] font-bold text-[#35D49A]">{lrFeedback}</span>
              )}
            </div>

            <p className="text-[11px] font-mono text-[#91A0B4] leading-relaxed">
              {isRunning
                ? "Scale optimizer learning rates on-the-fly for the active run."
                : "Available while a training run is active. Launch an experiment to enable it."}
            </p>

            <div className="space-y-2.5 mt-3">
              <div className="flex justify-between items-baseline font-mono text-[11px]">
                <label htmlFor="lr-multiplier" className="text-[#91A0B4]">
                  Multiplier
                </label>
                <span className="text-[#45CFFF] font-bold font-mono-num text-sm">
                  {lrMultiplier.toFixed(2)}×
                </span>
              </div>
              <input
                id="lr-multiplier"
                type="range"
                min="0.1"
                max="3.0"
                step="0.05"
                value={lrMultiplier}
                disabled={!isRunning}
                onChange={(e) => setLrMultiplier(parseFloat(e.target.value))}
                className="range"
                aria-valuetext={`${lrMultiplier.toFixed(2)} times`}
              />
              <div className="flex justify-between font-mono text-[10px] text-[#5E7188]">
                <span>0.1× fine</span>
                <span>1.0× default</span>
                <span>3.0× aggressive</span>
              </div>
            </div>
          </div>

          <div className="pt-3 border-t border-[var(--border-subtle)] flex items-center justify-between gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setLrMultiplier(1.0)}
              disabled={!isRunning || anyPending}
            >
              Reset 1.0×
            </Button>
            <Button
              variant="primary"
              size="sm"
              onClick={handleApplyLrMultiplier}
              loading={pendingAction === "lr"}
              disabled={!isRunning || anyPending}
            >
              Apply LR
            </Button>
          </div>
        </section>
      </div>

      {/* Activity & alerts */}
      <section className="panel p-3.5">
        <div className="panel-header">
          <h3 className="panel-title">
            <Terminal className="w-3.5 h-3.5 text-[#9B8CFF]" aria-hidden="true" />
            Activity stream & runtime signals
          </h3>
          <span className="font-mono text-[10px] text-[#5E7188]">
            {recentEvents.length + recentAlerts.length} events logged
          </span>
        </div>

        <div className="max-h-44 overflow-y-auto space-y-1.5 font-mono text-xs pr-1">
          {recentAlerts.length === 0 && recentEvents.length === 0 ? (
            <div className="py-6 flex flex-col items-center justify-center gap-1.5 text-center">
              <Terminal className="w-5 h-5 text-[#384C66]" aria-hidden="true" />
              <p className="text-[11px] font-semibold uppercase tracking-wider text-[#91A0B4]">
                No activity yet
              </p>
              <p className="text-[11px] text-[#5E7188]">
                Worker events and runtime signals will appear here once a run starts.
              </p>
            </div>
          ) : (
            <>
              {recentAlerts.map((a, i) => (
                <div
                  key={`alert-${i}`}
                  className="p-2 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-subtle)] flex items-center justify-between gap-3 text-xs"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <span
                      className={`shrink-0 text-[9px] px-1.5 py-0.5 rounded border font-bold uppercase ${
                        a.level === "CRITICAL" || a.level === "ERROR"
                          ? "bg-[rgba(255,101,122,0.12)] text-[#FF657A] border-[rgba(255,101,122,0.3)]"
                          : a.level === "WARNING"
                          ? "bg-[rgba(242,184,91,0.12)] text-[#F2B85B] border-[rgba(242,184,91,0.3)]"
                          : "bg-[rgba(69,207,255,0.12)] text-[#45CFFF] border-[rgba(69,207,255,0.3)]"
                      }`}
                    >
                      {a.level}
                    </span>
                    <span className="text-[#ECF3FC] truncate">{a.message}</span>
                  </div>
                  <span className="shrink-0 text-[#5E7188] text-[10px]">
                    {a.step ? `Step ${a.step}` : a.category}
                  </span>
                </div>
              ))}
              {recentEvents.slice(0, 10).map((e, i) => (
                <div
                  key={`ev-${i}`}
                  className="p-1.5 rounded-md bg-[var(--bg-subtle)]/60 border border-[var(--border-subtle)]/70 flex items-center justify-between gap-3 text-[11px] text-[#91A0B4]"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="shrink-0 text-[#45CFFF] font-bold">[{e.event_type}]</span>
                    <span className="truncate max-w-lg text-[#ECF3FC]">
                      {typeof e.payload === "string" ? e.payload : JSON.stringify(e.payload)}
                    </span>
                  </div>
                  <span className="shrink-0 text-[10px] text-[#5E7188]">
                    {e.timestamp ? new Date(e.timestamp).toLocaleTimeString() : "--:--:--"}
                  </span>
                </div>
              ))}
            </>
          )}
        </div>
      </section>

      <ConfirmDialog
        isOpen={confirmStop}
        title="Halt active training session"
        message="Stopping ends the current run. The last saved checkpoint and model weights are kept; any progress since the last checkpoint is lost."
        confirmLabel="Halt training"
        cancelLabel="Continue training"
        variant="danger"
        loading={pendingAction === "stop"}
        onConfirm={handleStopRun}
        onCancel={() => setConfirmStop(false)}
      />

      <SaveModelModal
        isOpen={isSaveModalOpen}
        onClose={() => setIsSaveModalOpen(false)}
        defaultModelId={status?.active_network || "rocanon_48m"}
        onSaved={(pkg) => {
          notify("success", `Model package '${pkg.name}' saved into models/${pkg.name}/`, "Saved Locally");
          refreshStatus();
        }}
      />
    </div>
  );
};
