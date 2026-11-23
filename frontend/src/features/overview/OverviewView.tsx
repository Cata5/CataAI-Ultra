"use client";

import React, { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useHardwareTelemetry } from "@/hooks/useHardwareTelemetry";
import { useSystemStatus } from "@/hooks/useSystemStatus";
import { api } from "@/services/api";
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

  const chartsLoading = statusLoading && metricsHistory.length === 0;
  const anyPending = pendingAction !== null;

  // Quick Action handlers
  const handleTogglePause = () => {
    if (isRunning) {
      void runAction("pause", () => api.pauseRun(), "Training paused. Worker state preserved.", "Paused");
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
      notify("success", "Stop requested. Final weights saved.", "Stopping run");
      await refreshStatus();
    } catch (e: any) {
      notify("error", e?.message || "Stop request failed.");
    } finally {
      setPendingAction(null);
    }
  };

  const handleManualCheckpoint = () => {
    void runAction(
      "checkpoint",
      () => api.checkpointRun(),
      "Checkpoint requested after next micro-step.",
      "Checkpoint queued"
    );
  };

  const handleTriggerVal = () => {
    void runAction(
      "validate",
      () => api.validateRun(),
      "Validation sweep triggered.",
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
      notify("success", `Learning rate scaled by ${lrMultiplier.toFixed(2)}×.`, "LR updated");
      setTimeout(() => setLrFeedback(null), 4000);
    } catch (e: any) {
      notify("error", e?.message || "Could not scale learning rate.");
    } finally {
      setPendingAction(null);
    }
  };

  return (
    <div className="space-y-3 max-w-[1600px] mx-auto animate-in fade-in duration-150">
      {/* Session Header Panel */}
      <section className="panel p-3 bg-[#202020] border-[#333333] flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3 min-w-0">
          <StatusIndicator state={effectiveState} size="md" />
          <div className="flex flex-col min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-mono text-xs font-semibold text-[#F2F2F2]">
                Session: <span className="text-[#A3A3A3] font-normal">{status?.run_id || "No active run"}</span>
              </span>
              <span className="tech-badge">
                {status?.active_network || "Neural Model"}
              </span>
            </div>
            <span className="text-[11px] font-mono text-[#737373] mt-0.5 truncate">
              {isRunning
                ? "Active PyTorch execution loop · Local telemetry streaming"
                : isPaused
                ? "Run paused · Worker state preserved"
                : "Workstation idle · Ready for architecture design and training"}
            </span>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          {isActive ? (
            <>
              <Button
                variant="secondary"
                size="sm"
                onClick={handleTogglePause}
                loading={pendingAction === "pause"}
                disabled={anyPending}
                icon={isRunning ? <Pause className="w-3 h-3 text-[#facc15]" /> : <Play className="w-3 h-3 text-[#F2F2F2]" />}
              >
                {isRunning ? "Pause" : "Resume"}
              </Button>

              <Button
                variant="secondary"
                size="sm"
                onClick={handleManualCheckpoint}
                loading={pendingAction === "checkpoint"}
                disabled={anyPending}
                icon={<Bookmark className="w-3 h-3 text-[#A3A3A3]" />}
              >
                Checkpoint
              </Button>

              <Button
                variant="ghost"
                size="sm"
                onClick={handleTriggerVal}
                loading={pendingAction === "validate"}
                disabled={anyPending}
                icon={<CheckCircle2 className="w-3 h-3 text-[#A3A3A3]" />}
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
                icon={<HardDrive className="w-3 h-3 text-[#A3A3A3]" />}
              >
                Save Model
              </Button>
            </>
          ) : (
            <div className="flex items-center gap-1.5">
              <Button
                variant="secondary"
                size="sm"
                onClick={() => setIsSaveModalOpen(true)}
                icon={<HardDrive className="w-3 h-3 text-[#A3A3A3]" />}
              >
                Save Model
              </Button>
              <Link href="/training" className={buttonClasses("primary", "sm")}>
                <Play className="w-3 h-3 fill-current" aria-hidden="true" />
                Start Training
              </Link>
            </div>
          )}
        </div>
      </section>

      {/* Action feedback */}
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
              icon={<RefreshCw className="w-3 h-3" />}
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

      {/* Live Metric Cards Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 select-none font-mono">
        {/* Train Loss */}
        <div className="panel p-2.5 bg-[#202020] border-[#333333]">
          <div className="text-[10px] text-[#D4D4D4] uppercase font-semibold flex items-center justify-between">
            <span>Train Loss</span>
            <TrendingDown className="w-3 h-3 text-[#A3A3A3]" />
          </div>
          <div className="text-base font-semibold font-mono-num text-[#F2F2F2] mt-1">
            {currentLoss > 0 ? formatNumber(currentLoss, 4) : "—"}
          </div>
          <div className="text-[10px] text-[#A3A3A3] font-medium mt-0.5">
            Step {currentStep.toLocaleString()}
          </div>
        </div>

        {/* Validation Loss */}
        <div className="panel p-2.5 bg-[#202020] border-[#333333]">
          <div className="text-[10px] text-[#D4D4D4] uppercase font-semibold flex items-center justify-between">
            <span>Val. Loss</span>
            <CheckCircle2 className="w-3 h-3 text-[#A3A3A3]" />
          </div>
          <div className="text-base font-semibold font-mono-num text-[#4ade80] mt-1">
            {effectiveValLoss !== null ? (
              formatNumber(effectiveValLoss, 4)
            ) : isRunning ? (
              <span className="text-xs text-[#facc15]">Evaluating...</span>
            ) : (
              "—"
            )}
          </div>
          <div className="text-[10px] text-[#A3A3A3] font-medium mt-0.5">
            {effectiveValStep ? `Step ${effectiveValStep}` : `Next: Step ${nextValStep}`}
          </div>
        </div>

        {/* Dataset Progress */}
        <div className="panel p-2.5 bg-[#202020] border-[#333333]">
          <div className="text-[10px] text-[#D4D4D4] uppercase font-semibold flex items-center justify-between">
            <span>Dataset Progress</span>
            <Activity className="w-3 h-3 text-[#A3A3A3]" />
          </div>
          <div className="text-sm font-semibold font-mono-num text-[#F2F2F2] mt-1">
            {activeDurationMode === "epochs" ? (
              <span>Epoch {currentEpoch} / {displayTargetEpochs}</span>
            ) : (
              <span>Step {currentStep.toLocaleString()}</span>
            )}
          </div>
          <div className="text-[10px] text-[#A3A3A3] font-medium mt-0.5 truncate">
            {tokensPerEpoch > 0 ? (
              `${formatCompactNumber(tokensInEpoch)} / ${formatCompactNumber(tokensPerEpoch)} (${epochPercent}%)`
            ) : (
              `${formatCompactNumber(tokensSeen)} tokens`
            )}
          </div>
        </div>

        {/* Time & Latency */}
        <div className="panel p-2.5 bg-[#202020] border-[#333333]">
          <div className="text-[10px] text-[#D4D4D4] uppercase font-semibold flex items-center justify-between">
            <span>Time & Latency</span>
            <Clock className="w-3 h-3 text-[#A3A3A3]" />
          </div>
          <div className="text-base font-semibold font-mono-num text-[#F2F2F2] mt-1">
            {formatDuration(activeTrainingTime)}
          </div>
          <div className="text-[10px] text-[#A3A3A3] font-medium mt-0.5 flex justify-between">
            <span>{formatStepLatency(stepTimeMs)}</span>
            <span>{tokPerSec > 0 ? `${Math.round(tokPerSec)} tok/s` : ""}</span>
          </div>
        </div>

        {/* Muon LR */}
        <div className="panel p-2.5 bg-[#202020] border-[#333333]">
          <div className="text-[10px] text-[#D4D4D4] uppercase font-semibold flex items-center justify-between">
            <span>Muon LR (Matrix)</span>
            <Sparkles className="w-3 h-3 text-[#A3A3A3]" />
          </div>
          <div className="text-sm font-semibold font-mono-num text-[#F2F2F2] mt-1">
            {muonLr > 0 ? muonLr.toExponential(2) : "—"}
          </div>
          <div className="text-[10px] text-[#A3A3A3] font-medium mt-0.5">
            Phase: {schedPhase}
          </div>
        </div>

        {/* AdamW LR */}
        <div className="panel p-2.5 bg-[#202020] border-[#333333]">
          <div className="text-[10px] text-[#D4D4D4] uppercase font-semibold flex items-center justify-between">
            <span>AdamW LR (Vectors)</span>
            <Gauge className="w-3 h-3 text-[#A3A3A3]" />
          </div>
          <div className="text-sm font-semibold font-mono-num text-[#A3A3A3] mt-1">
            {adamwLr > 0 ? adamwLr.toExponential(2) : "—"}
          </div>
          <div className="text-[10px] text-[#A3A3A3] font-medium mt-0.5">
            Next eval: Step {nextValStep}
          </div>
        </div>
      </div>

      {/* Main Charts & Hardware */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-2.5">
        <div className="lg:col-span-2">
          <ChartPanel
            title="Training & validation loss"
            labels={chartSteps}
            datasets={[
              {
                label: "Train Loss",
                data: lossSeries,
                borderColor: "#F2F2F2",
                borderWidth: 1.6,
              },
              ...(hasValLoss
                ? [
                    {
                      label: "Validation Loss",
                      data: valLossSeries,
                      borderColor: "#A3A3A3",
                      borderDash: [4, 4],
                      borderWidth: 1.4,
                      pointRadius: 2.5,
                      spanGaps: true,
                    },
                  ]
                : []),
            ]}
            yAxisLabel="Loss"
            height={240}
            loading={chartsLoading}
            emptyTitle="No metrics recorded"
            emptyHint="Start a training run to stream loss metrics here."
          />
        </div>

        <TelemetryCard
          telemetry={telemetry}
          error={telemetryError}
          isLoading={telemetryLoading}
          className="h-full"
        />
      </div>

      {/* Secondary Charts & Controls */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-2.5">
        <ChartPanel
          title="Learning rate"
          labels={chartSteps}
          datasets={[
            {
              label: "Effective LR",
              data: lrSeries,
              borderColor: "#D4D4D4",
              borderWidth: 1.4,
            },
          ]}
          yAxisLabel="LR"
          height={160}
          loading={chartsLoading}
          emptyTitle="No LR data"
          emptyHint="Learning rate schedule displays during runs."
        />

        <ChartPanel
          title="Gradient norm"
          labels={chartSteps}
          datasets={[
            {
              label: "Grad Norm",
              data: gradNormSeries,
              borderColor: "#A3A3A3",
              borderWidth: 1.2,
              spanGaps: true,
            },
          ]}
          yAxisLabel="L2 norm"
          height={160}
          loading={chartsLoading}
          emptyTitle="No gradient data"
          emptyHint="Gradient norm reports during active training."
        />

        <ChartPanel
          title="Step latency"
          labels={chartSteps}
          datasets={[
            {
              label: "Step Time (ms)",
              data: stepDurationSeries,
              borderColor: "#8C8C8C",
              borderWidth: 1.2,
            },
          ]}
          yAxisLabel="ms"
          height={160}
          loading={chartsLoading}
          emptyTitle="No latency data"
          emptyHint="Timing metrics record per step."
        />

        {/* Dynamic LR scaling */}
        <section className="panel p-3 bg-[#202020] border-[#333333] flex flex-col justify-between gap-2.5">
          <div>
            <div className="panel-header">
              <h3 className="panel-title">
                <Sliders className="w-3.5 h-3.5 text-[#737373]" aria-hidden="true" />
                Dynamic LR scaling
              </h3>
              {lrFeedback && (
                <span className="font-mono text-[10px] text-[#4ade80]">{lrFeedback}</span>
              )}
            </div>

            <p className="text-[11px] font-mono text-[#737373] leading-relaxed">
              {isRunning
                ? "Scale optimizer learning rates on-the-fly for the active run."
                : "Available during active training runs."}
            </p>

            <div className="space-y-2 mt-2">
              <div className="flex justify-between items-baseline font-mono text-xs">
                <label htmlFor="lr-multiplier" className="text-[#737373]">
                  Multiplier
                </label>
                <span className="text-[#F2F2F2] font-mono-num font-medium text-xs">
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
              />
              <div className="flex justify-between font-mono text-[9px] text-[#525252]">
                <span>0.1× fine</span>
                <span>1.0×</span>
                <span>3.0×</span>
              </div>
            </div>
          </div>

          <div className="pt-2 border-t border-[#262626] flex items-center justify-between gap-1.5">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setLrMultiplier(1.0)}
              disabled={!isRunning || anyPending}
            >
              Reset
            </Button>
            <Button
              variant="primary"
              size="sm"
              onClick={handleApplyLrMultiplier}
              loading={pendingAction === "lr"}
              disabled={!isRunning || anyPending}
            >
              Apply
            </Button>
          </div>
        </section>
      </div>

      {/* Activity log */}
      <section className="panel p-3 bg-[#202020] border-[#333333]">
        <div className="panel-header">
          <h3 className="panel-title">
            <Terminal className="w-3.5 h-3.5 text-[#737373]" aria-hidden="true" />
            Activity & runtime signals
          </h3>
          <span className="font-mono text-[10px] text-[#525252]">
            {recentEvents.length + recentAlerts.length} events
          </span>
        </div>

        <div className="max-h-36 overflow-y-auto space-y-1 font-mono text-xs pr-1">
          {recentAlerts.length === 0 && recentEvents.length === 0 ? (
            <div className="py-4 flex flex-col items-center justify-center gap-1 text-center">
              <Terminal className="w-4 h-4 text-[#404040]" aria-hidden="true" />
              <p className="text-[10px] uppercase tracking-wider text-[#737373]">
                No recent activity
              </p>
              <p className="text-[10px] text-[#525252]">
                Events and signals stream here during runs.
              </p>
            </div>
          ) : (
            <>
              {recentAlerts.map((a, i) => (
                <div
                  key={`alert-${i}`}
                  className="p-1.5 rounded bg-[#1B1B1B] border border-[#262626] flex items-center justify-between gap-2 text-xs"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <span
                      className={`shrink-0 text-[9px] px-1 py-0.2 rounded font-medium uppercase ${
                        a.level === "CRITICAL" || a.level === "ERROR"
                          ? "bg-[rgba(248,113,113,0.1)] text-[#f87171] border border-[rgba(248,113,113,0.25)]"
                          : a.level === "WARNING"
                          ? "bg-[rgba(250,204,21,0.08)] text-[#facc15] border border-[rgba(250,204,21,0.25)]"
                          : "bg-[#262626] text-[#A3A3A3] border border-[#333333]"
                      }`}
                    >
                      {a.level}
                    </span>
                    <span className="text-[#F2F2F2] truncate">{a.message}</span>
                  </div>
                  <span className="shrink-0 text-[#525252] text-[10px]">
                    {a.step ? `Step ${a.step}` : a.category}
                  </span>
                </div>
              ))}
              {recentEvents.slice(0, 8).map((e, i) => (
                <div
                  key={`ev-${i}`}
                  className="p-1 rounded bg-[#1B1B1B] border border-[#262626] flex items-center justify-between gap-2 text-[11px] text-[#A3A3A3]"
                >
                  <div className="flex items-center gap-1.5 min-w-0">
                    <span className="shrink-0 text-[#737373] font-medium">[{e.event_type}]</span>
                    <span className="truncate max-w-lg text-[#F2F2F2]">
                      {typeof e.payload === "string" ? e.payload : JSON.stringify(e.payload)}
                    </span>
                  </div>
                  <span className="shrink-0 text-[10px] text-[#525252]">
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
        message="Stopping ends the current run. Saved checkpoints and weights are kept."
        confirmLabel="Halt training"
        cancelLabel="Continue"
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
          notify("success", `Model package '${pkg.name}' saved.`, "Saved Locally");
          refreshStatus();
        }}
      />
    </div>
  );
};
