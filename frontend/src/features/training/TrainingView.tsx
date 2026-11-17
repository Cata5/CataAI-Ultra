"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useSystemStatus } from "@/hooks/useSystemStatus";
import { useHardwareTelemetry } from "@/hooks/useHardwareTelemetry";
import { api } from "@/services/api";
import { ChartPanel } from "@/components/ui/ChartPanel";
import { StatusIndicator } from "@/components/ui/StatusIndicator";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { DatasetItem, NetworkConfig, CheckpointItem } from "@/types/api";
import {
  Play,
  Pause,
  Square,
  Bookmark,
  CheckCircle2,
  Sliders,
  Layers,
  Sparkles,
  RefreshCw,
  Clock,
  TrendingDown,
  Gauge,
  Activity,
  AlertTriangle,
  RotateCcw,
  Calculator,
  Info,
} from "lucide-react";
import {
  cn,
  formatCompactNumber,
  formatNumber,
  formatDuration,
  formatStepLatency,
} from "@/lib/utils";

export interface TrainingFormConfig {
  selectedNetwork: string;
  selectedDataset: string;
  selectedTask: string;
  optimizer: string;
  schedulerType: string;
  resumeCheckpoint: string;
  durationMode: "steps" | "epochs";
  targetEpochs: number;
  precision: string;
  peakLr: number;
  muonLr: number;
  minLr: number;
  microBatchSize: number;
  gradAccumSteps: number;
  seqLen: number;
  totalSteps: number;
  warmupSteps: number;
  stableSteps: number;
  gradClip: number;
  chkInterval: number;
  valInterval: number;
  earlyStoppingEnabled: boolean;
  patience: number;
  minSteps: number;
}

const STORAGE_KEY = "llm_maker_training_config_v2";

const DEFAULT_TRAINING_CONFIG: TrainingFormConfig = {
  selectedNetwork: "rocanon_48_5m",
  selectedDataset: "tech_wiki_ro",
  selectedTask: "causal_lm",
  optimizer: "muon_adamw",
  schedulerType: "wsd",
  resumeCheckpoint: "",
  durationMode: "steps",
  targetEpochs: 3,
  precision: "bf16",
  peakLr: 0.003,
  muonLr: 0.02,
  minLr: 0.0001,
  microBatchSize: 4,
  gradAccumSteps: 8,
  seqLen: 256,
  totalSteps: 10000,
  warmupSteps: 500,
  stableSteps: 7500,
  gradClip: 1.0,
  chkInterval: 500,
  valInterval: 250,
  earlyStoppingEnabled: false,
  patience: 20,
  minSteps: 500,
};

const getSavedConfig = (): TrainingFormConfig => {
  if (typeof window === "undefined") return DEFAULT_TRAINING_CONFIG;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_TRAINING_CONFIG;
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_TRAINING_CONFIG, ...parsed };
  } catch (e) {
    console.warn("Failed to load training form config from localStorage:", e);
    return DEFAULT_TRAINING_CONFIG;
  }
};

export const TrainingView: React.FC = () => {
  const {
    currentState,
    latestMetric,
    metricsHistory,
    loadRunHistory,
    latestValLoss,
    latestValStep,
    latestValTimestamp,
    valStatus,
    activeTrainingTime,
  } = useWebSocket();
  const { status, refresh: refreshStatus } = useSystemStatus(3000);
  const { telemetry } = useHardwareTelemetry(5000);
  const computeLabel = telemetry?.device_name || "Detected Hardware Accelerator";

  // Available options from backend
  const [networks, setNetworks] = useState<NetworkConfig[]>([]);
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [checkpoints, setCheckpoints] = useState<CheckpointItem[]>([]);
  const [schedulerCurve, setSchedulerCurve] = useState<{ steps: number[]; lrs: number[] }>({
    steps: [],
    lrs: [],
  });

  // Persistent Form State
  const [form, setForm] = useState<TrainingFormConfig>(DEFAULT_TRAINING_CONFIG);
  const [isLoadedFromStorage, setIsLoadedFromStorage] = useState(false);
  const [isTokenizingDataset, setIsTokenizingDataset] = useState<boolean>(false);

  // Dialogs & Notifications
  const [confirmStop, setConfirmStop] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);

  // Load configuration from localStorage on mount and check URL params
  useEffect(() => {
    const saved = getSavedConfig();
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      const netParam = params.get("network") || params.get("model");
      if (netParam) {
        saved.selectedNetwork = netParam;
      }
    }
    setForm(saved);
    setIsLoadedFromStorage(true);
  }, []);

  // Update a form field and automatically persist to localStorage
  const updateForm = useCallback(
    <K extends keyof TrainingFormConfig>(key: K, value: TrainingFormConfig[K]) => {
      setForm((prev) => {
        const next = { ...prev, [key]: value };
        if (typeof window !== "undefined") {
          try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
          } catch (e) {
            console.warn("Failed to persist training config:", e);
          }
        }
        return next;
      });
    },
    []
  );

  const handleResetDefaults = () => {
    if (window.confirm("Reset all training settings and hyperparameters to default values?")) {
      setForm(DEFAULT_TRAINING_CONFIG);
      if (typeof window !== "undefined") {
        try {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(DEFAULT_TRAINING_CONFIG));
        } catch (e) {
          console.warn("Failed to clear config in localStorage:", e);
        }
      }
      setFeedback("All training parameters reset to defaults.");
      setTimeout(() => setFeedback(null), 4000);
    }
  };

  const effectiveState = currentState !== "IDLE" ? currentState : status?.state || "IDLE";
  const isRunning = effectiveState === "RUNNING";
  const isPaused = effectiveState === "PAUSED";

  // Fetch backend options on mount
  useEffect(() => {
    api.listNetworks().then(setNetworks).catch(console.error);
    api.listDatasets().then(setDatasets).catch(console.error);
    api.listCheckpoints().then(setCheckpoints).catch(console.error);
  }, []);

  // Sync historical telemetry across tabs if buffer was empty
  useEffect(() => {
    if (status?.run_id && metricsHistory.length === 0) {
      loadRunHistory(status.run_id);
    }
  }, [status?.run_id, metricsHistory.length, loadRunHistory]);

  // Find currently selected dataset object
  const currentDataset = useMemo(
    () => datasets.find((d) => d.id === form.selectedDataset),
    [datasets, form.selectedDataset]
  );

  // Dataset tokens count
  const datasetTokens = currentDataset?.train_tokens || currentDataset?.total_tokens || 0;

  // Geometry calculations
  const effectiveBatchSize = form.microBatchSize * form.gradAccumSteps;
  const tokensPerStep = effectiveBatchSize * form.seqLen;

  // Estimation of steps per epoch & total steps
  const stepsPerEpoch = useMemo(() => {
    if (tokensPerStep <= 0 || datasetTokens <= 0) return 0;
    return Math.ceil(datasetTokens / tokensPerStep);
  }, [datasetTokens, tokensPerStep]);

  const estimatedEpochSteps = stepsPerEpoch > 0 ? stepsPerEpoch * form.targetEpochs : 0;

  // Effective steps to display and plan for
  const plannedSteps = form.durationMode === "epochs" && estimatedEpochSteps > 0 ? estimatedEpochSteps : form.totalSteps;

  // Update schedule preview curve
  useEffect(() => {
    let active = true;
    api
      .getSchedulerCurve({
        schedule_type: form.schedulerType,
        total_steps: plannedSteps,
        peak_lr: form.peakLr,
        min_lr: form.minLr,
        warmup_steps: form.warmupSteps,
        stable_steps: form.stableSteps,
        decay_style: "cosine",
      })
      .then((res) => {
        if (active && Array.isArray(res)) {
          setSchedulerCurve({
            steps: res.map((r: any) => r.step),
            lrs: res.map((r: any) => r.lr),
          });
        }
      })
      .catch(console.error);
    return () => {
      active = false;
    };
  }, [form.schedulerType, plannedSteps, form.peakLr, form.minLr, form.warmupSteps, form.stableSteps]);

  const handleTokenizeSelectedDataset = async () => {
    if (!form.selectedDataset) return;
    setIsTokenizingDataset(true);
    setFeedback(null);
    try {
      const res = await api.tokenizeDataset(form.selectedDataset);
      setFeedback(
        `Dataset tokenized successfully: ${(res.token_count || res.train_tokens || 0).toLocaleString()} tokens cached.`
      );
      // Refresh datasets list
      const dsList = await api.listDatasets();
      setDatasets(dsList);
      setTimeout(() => setFeedback(null), 5000);
    } catch (e: any) {
      alert(`Tokenization failed: ${e.message}`);
    } finally {
      setIsTokenizingDataset(false);
    }
  };

  const handleStartTraining = async () => {
    setIsSubmitting(true);
    setFeedback(null);
    try {
      const payload: Record<string, any> = {
        network_id: form.selectedNetwork,
        dataset_source: form.selectedDataset,
        task_type: form.selectedTask,
        optimizer: form.optimizer,
        scheduler_type: form.schedulerType,
        duration_mode: form.durationMode,
        target_epochs: form.targetEpochs,
        peak_lr: form.peakLr,
        muon_lr: form.muonLr,
        min_lr: form.minLr,
        micro_batch_size: form.microBatchSize,
        gradient_accumulation_steps: form.gradAccumSteps,
        seq_len: form.seqLen,
        total_steps: plannedSteps,
        warmup_steps: form.warmupSteps,
        stable_steps: form.stableSteps,
        grad_clip: form.gradClip,
        checkpoint_interval: form.chkInterval,
        validation_interval: form.valInterval,
        early_stopping: {
          enabled: form.earlyStoppingEnabled,
          patience: form.patience,
          min_steps: form.minSteps,
        },
      };

      if (form.resumeCheckpoint) {
        payload.resume_checkpoint = form.resumeCheckpoint;
        payload.checkpoint_to_load = form.resumeCheckpoint;
        payload.checkpoint_path = form.resumeCheckpoint;
      }

      await api.startRun(payload);
      setFeedback(
        `Training run initiated successfully (${form.durationMode.toUpperCase()} mode)!`
      );
      refreshStatus();
    } catch (e: any) {
      alert(`Start failed: ${e.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleTogglePause = async () => {
    try {
      if (isRunning) {
        await api.pauseRun();
      } else if (isPaused) {
        await api.resumeRun();
      }
      refreshStatus();
    } catch (e: any) {
      alert(`Error: ${e.message}`);
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

  const handleCheckpoint = async () => {
    try {
      await api.checkpointRun();
      alert("Manual checkpoint requested.");
    } catch (e: any) {
      alert(`Checkpoint error: ${e.message}`);
    }
  };

  const handleValidate = async () => {
    try {
      await api.validateRun();
      alert("Validation triggered.");
    } catch (e: any) {
      alert(`Validation error: ${e.message}`);
    }
  };

  const chartSteps = metricsHistory.map((m) => m.step);
  const lossSeries = metricsHistory.map((m) => (m.loss !== undefined ? m.loss : (m as any).train_loss ?? 0));
  const valLossSeries = metricsHistory.map((m) =>
    m.val_loss !== null && m.val_loss !== undefined ? Number(m.val_loss) : null
  );
  const hasValLoss = metricsHistory.some((m) => m.val_loss !== null && m.val_loss !== undefined);

  // Derived Telemetry State
  const currentStep = latestMetric?.step ?? status?.latest_metrics?.step ?? 0;
  const currentLoss = latestMetric?.loss ?? (latestMetric as any)?.train_loss ?? status?.latest_metrics?.loss ?? 0;
  const effectiveValLoss = latestValLoss ?? latestMetric?.val_loss ?? status?.latest_metrics?.latest_val_loss ?? null;
  const effectiveValStep = latestValStep ?? latestMetric?.latest_val_step ?? status?.latest_metrics?.latest_val_step ?? null;

  // Split Learning Rates & Scheduling
  const muonEffLr = latestMetric?.muon_lr ?? latestMetric?.learning_rate ?? form.muonLr;
  const adamwEffLr = latestMetric?.adamw_lr ?? form.peakLr;
  const schedPhase = latestMetric?.sched_phase || "stable";
  const schedFactor = latestMetric?.sched_factor ?? 1.0;
  const isHighLr = form.muonLr > 0.05 || form.peakLr > 0.01;

  // Dataset & Epoch Progress Tracking
  const activeDurationMode = latestMetric?.duration_mode || form.durationMode;
  const currentEpoch = latestMetric?.current_epoch || 1;
  const displayTargetEpochs = latestMetric?.target_epochs || form.targetEpochs;
  const tokensInEpoch = latestMetric?.tokens_in_epoch || 0;
  const tokensPerEpoch = latestMetric?.tokens_per_epoch || datasetTokens || 0;
  const epochPercent = tokensPerEpoch > 0 ? Math.min(100, Math.round((tokensInEpoch / tokensPerEpoch) * 100)) : 0;
  const tokensSeen = latestMetric?.tokens_seen ?? (latestMetric as any)?.tokens ?? status?.latest_metrics?.tokens_seen ?? 0;
  const tokPerSec = latestMetric?.tokens_per_second ?? (latestMetric as any)?.tok_per_sec ?? status?.latest_metrics?.tokens_per_second ?? 0;
  const stepTimeMs = latestMetric?.step_time_ms ?? ((latestMetric as any)?.dt ? (latestMetric as any).dt * 1000 : 0);

  // Scheduled validation tracking
  const scheduledInterval = latestMetric?.eval_interval || form.valInterval;
  const nextValStep = latestMetric?.next_val_step ?? (currentStep > 0 ? currentStep + (scheduledInterval - (currentStep % scheduledInterval)) : scheduledInterval);

  return (
    <div className="space-y-4 font-mono text-xs">
      {/* Top Controls Header */}
      {/* Header Controller Bar */}
      <div className="glass-panel p-3.5 rounded-lg flex flex-wrap items-center justify-between gap-3 border-l-2 border-l-[#45CFFF] relative overflow-hidden">
        <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-[#45CFFF]/30 via-transparent to-transparent pointer-events-none" />

        <div className="flex items-center gap-3">
          <StatusIndicator state={effectiveState} size="lg" />
          <div className="flex flex-col">
            <span className="text-xs font-bold text-[#ECF3FC] flex items-center gap-1.5 font-mono uppercase tracking-wider">
              <span>EXPERIMENT CONTROLLER</span>
              {status?.run_id && <span className="text-[#45CFFF] font-mono-num font-semibold">({status.run_id})</span>}
            </span>
            <span className="text-[10px] font-mono text-[#91A0B4] mt-0.5">
              Precision: {form.precision.toUpperCase()} Mixed | Compute: {computeLabel} | Safety: Dynamic Guard
            </span>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2">
          {!isRunning && !isPaused ? (
            <button
              onClick={handleStartTraining}
              disabled={isSubmitting}
              className="inline-flex items-center gap-2 px-4 py-1.5 rounded-md bg-[#45CFFF] hover:bg-[#80DEFF] disabled:opacity-40 text-[#090D14] text-xs font-mono font-bold transition-all shadow-[0_0_12px_rgba(69,207,255,0.25)]"
            >
              <Play className="w-3.5 h-3.5 fill-current" /> START TRAINING
            </button>
          ) : (
            <>
              <button
                onClick={handleTogglePause}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-[#182333] border border-[#273548] hover:border-[#384C66] text-xs font-mono text-[#ECF3FC] transition-all"
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
                onClick={handleCheckpoint}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-[#182333] border border-[#273548] hover:border-[#35D49A]/50 text-xs font-mono text-[#ECF3FC] transition-all"
              >
                <Bookmark className="w-3.5 h-3.5 text-[#35D49A]" /> SAVE CHECKPOINT
              </button>

              <button
                onClick={handleValidate}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-[#182333] border border-[#273548] hover:border-[#45CFFF]/50 text-xs font-mono text-[#ECF3FC] transition-all"
              >
                <CheckCircle2 className="w-3.5 h-3.5 text-[#45CFFF]" /> RUN VALIDATION
              </button>

              <button
                onClick={() => setConfirmStop(true)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-[#FF657A]/15 border border-[#FF657A]/30 hover:bg-[#FF657A]/25 text-xs font-mono text-[#FF657A] font-semibold transition-all"
              >
                <Square className="w-3.5 h-3.5 text-[#FF657A]" /> STOP RUN
              </button>
            </>
          )}
        </div>
      </div>

      {feedback && (
        <div className="p-2.5 rounded-md bg-[#35D49A]/15 border border-[#35D49A]/30 text-[#35D49A] font-bold text-xs flex items-center gap-2 font-mono">
          <CheckCircle2 className="w-4 h-4" /> {feedback}
        </div>
      )}

      {/* Live Telemetry & Progress Strip */}
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

        {/* Validation Loss with Persistence */}
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
            {effectiveValStep ? `Evaluated at Step ${effectiveValStep}` : `Interval: every ${scheduledInterval} steps`}
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
              <span>Step {currentStep.toLocaleString()} / {form.totalSteps.toLocaleString()}</span>
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
            <span>{muonEffLr > 0 ? muonEffLr.toExponential(2) : "—"}</span>
            {isHighLr && (
              <span className="p-0.5 rounded bg-[#F2B85B]/15 text-[#F2B85B] border border-[#F2B85B]/40" title="High LR: check stability">
                <AlertTriangle className="w-2.5 h-2.5" />
              </span>
            )}
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
            {adamwEffLr > 0 ? adamwEffLr.toExponential(2) : "—"}
          </div>
          <div className="text-[9px] text-[#5E7188] mt-0.5">
            Next eval: Step {nextValStep}
          </div>
        </div>
      </div>

      {/* Main Grid: Experiment Configuration (Left 2 cols) + Plots & Schedulers (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left Column: Model, Dataset, & Hyperparams */}
        <div className="lg:col-span-2 space-y-4">
          {/* Section 1: Architecture & Dataset Selection */}
          <div className="glass-panel p-4 rounded-lg space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-[#273548]">
              <h3 className="text-xs font-bold text-[#ECF3FC] uppercase tracking-wider flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-[#45CFFF]" /> ARCHITECTURE & DATASET SELECTION
              </h3>
              <button
                type="button"
                onClick={handleResetDefaults}
                disabled={isRunning || isPaused}
                className="inline-flex items-center gap-1 text-[10px] text-[#91A0B4] hover:text-[#F2B85B] transition-colors disabled:opacity-40"
                title="Reset hyperparameters to defaults"
              >
                <RotateCcw className="w-3 h-3" /> Reset Defaults
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div>
                <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                  Architecture Model
                </label>
                <div className="flex items-center gap-1.5">
                  <select
                    value={form.selectedNetwork}
                    onChange={(e) => updateForm("selectedNetwork", e.target.value)}
                    disabled={isRunning || isPaused}
                    className="flex-1 bg-[#0E1420] border border-[#273548] rounded px-2.5 py-1.5 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF] truncate"
                  >
                    {networks.map((n) => (
                      <option key={n.id} value={n.id}>
                        {n.name} ({((n.total_params || 48531968) / 1_000_000).toFixed(1)}M)
                      </option>
                    ))}
                  </select>
                  <a
                    href={`/studio?network=${encodeURIComponent(form.selectedNetwork)}`}
                    className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-md bg-[#131C29] border border-[#273548] hover:border-[#45CFFF] hover:text-[#45CFFF] text-[#91A0B4] text-xs font-mono transition-colors shrink-0"
                    title="Edit this model in Network Studio"
                  >
                    <Sliders className="w-3.5 h-3.5 text-[#45CFFF]" />
                    <span>EDIT</span>
                  </a>
                </div>
              </div>

              <div>
                <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                  Dataset Source
                </label>
                <div className="flex items-center gap-1.5">
                  <select
                    value={form.selectedDataset}
                    onChange={(e) => updateForm("selectedDataset", e.target.value)}
                    disabled={isRunning || isPaused}
                    className="flex-1 bg-[#0E1420] border border-[#273548] rounded px-2.5 py-1.5 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF] truncate"
                  >
                    {datasets.map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.name} {d.is_uploaded ? "(Uploaded)" : ""}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    onClick={handleTokenizeSelectedDataset}
                    disabled={isTokenizingDataset || !form.selectedDataset || isRunning || isPaused}
                    className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-md bg-[#45CFFF]/10 border border-[#45CFFF]/30 hover:bg-[#45CFFF]/20 text-[#45CFFF] text-xs font-mono transition-colors disabled:opacity-50 shrink-0"
                    title="Pre-tokenize selected dataset"
                  >
                    {isTokenizingDataset ? (
                      <RefreshCw className="w-3.5 h-3.5 animate-spin text-[#45CFFF]" />
                    ) : (
                      <Sparkles className="w-3.5 h-3.5 text-[#45CFFF]" />
                    )}
                    TOKENIZE
                  </button>
                </div>
              </div>

              <div>
                <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                  Training Task
                </label>
                <select
                  value={form.selectedTask}
                  onChange={(e) => updateForm("selectedTask", e.target.value)}
                  disabled={isRunning || isPaused}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded px-2.5 py-1.5 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF]"
                >
                  <option value="causal_lm">Causal LM (Autoregressive Next-Token)</option>
                  <option value="instruction_sft">Instruction Fine-Tuning (SFT & Chat)</option>
                  <option value="masked_lm">Masked LM (Bidirectional MLM)</option>
                  <option value="classification">Classification (Vision & Tabular)</option>
                  <option value="regression">Regression (Continuous Numerical)</option>
                </select>
              </div>
            </div>

            {/* Resume from checkpoint */}
            <div>
              <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                Resume From Existing Checkpoint (Optional)
              </label>
              <select
                value={form.resumeCheckpoint}
                onChange={(e) => updateForm("resumeCheckpoint", e.target.value)}
                disabled={isRunning || isPaused}
                className="w-full bg-[#0E1420] border border-[#273548] rounded px-2.5 py-1.5 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF]"
              >
                <option value="">Start Fresh (No checkpoint restore)</option>
                {checkpoints.map((c) => (
                  <option key={c.id || c.path || c.filename} value={c.path || c.filename}>
                    {c.filename} {c.run_id ? `[${c.run_id}]` : ""} (Step {c.step}
                    {c.val_loss ? ` | Val Loss: ${c.val_loss.toFixed(4)}` : ""})
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Section 2: Optimizer, Duration Mode & Learning Rates */}
          <div className="glass-panel p-4 rounded-lg space-y-3">
            <h3 className="text-xs font-bold text-[#ECF3FC] uppercase tracking-wider flex items-center gap-1.5 pb-2 border-b border-[#273548]">
              <Sparkles className="w-3.5 h-3.5 text-[#45CFFF]" /> OBJECTIVE, OPTIMIZER & LEARNING RATE SCHEDULE
            </h3>

            {/* Training Duration Mode Selector */}
            <div className="p-3 rounded-lg bg-[#0E1420]/80 border border-[#273548] space-y-2.5">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
                <div>
                  <span className="text-xs text-[#ECF3FC] font-bold uppercase tracking-wider flex items-center gap-1.5">
                    <Activity className="w-3.5 h-3.5 text-[#45CFFF]" />
                    TRAINING DURATION MODE
                  </span>
                  <span className="text-[11px] text-[#91A0B4] block mt-0.5">
                    Choose between training for full dataset epochs or a fixed budget of optimization steps.
                  </span>
                </div>
                <div className="flex items-center gap-1.5 bg-[#090D14] p-1 rounded-lg border border-[#273548] self-stretch sm:self-auto">
                  <button
                    type="button"
                    onClick={() => updateForm("durationMode", "steps")}
                    disabled={isRunning || isPaused}
                    className={cn(
                      "flex-1 sm:flex-initial px-4 py-1.5 rounded-md text-xs font-bold transition-all flex items-center justify-center gap-1.5 cursor-pointer",
                      form.durationMode === "steps"
                        ? "bg-[#45CFFF] text-[#090D14] shadow-[0_0_12px_rgba(69,207,255,0.35)]"
                        : "text-[#91A0B4] hover:text-[#ECF3FC] hover:bg-[#182333]"
                    )}
                  >
                    STEPS BUDGET
                  </button>
                  <button
                    type="button"
                    onClick={() => updateForm("durationMode", "epochs")}
                    disabled={isRunning || isPaused}
                    className={cn(
                      "flex-1 sm:flex-initial px-4 py-1.5 rounded-md text-xs font-bold transition-all flex items-center justify-center gap-1.5 cursor-pointer",
                      form.durationMode === "epochs"
                        ? "bg-[#45CFFF] text-[#090D14] shadow-[0_0_12px_rgba(69,207,255,0.35)]"
                        : "text-[#91A0B4] hover:text-[#ECF3FC] hover:bg-[#182333]"
                    )}
                  >
                    EPOCHS DURATION
                  </button>
                </div>
              </div>

              {/* Mode-Specific Parameter & Estimator Banner */}
              {form.durationMode === "epochs" ? (
                <div className="p-3 rounded-lg bg-[#45CFFF]/10 border border-[#45CFFF]/30 space-y-2">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <label className="text-[11px] text-[#45CFFF] font-bold uppercase">
                        Target Dataset Epochs:
                      </label>
                      <input
                        type="number"
                        min="1"
                        max="100"
                        value={form.targetEpochs}
                        onChange={(e) => updateForm("targetEpochs", Math.max(1, parseInt(e.target.value) || 1))}
                        disabled={isRunning || isPaused}
                        className="w-20 bg-[#0E1420] border border-[#45CFFF]/50 rounded px-2.5 py-1 text-[#ECF3FC] font-bold text-center text-sm focus:outline-none focus:border-[#45CFFF]"
                      />
                      <div className="flex items-center gap-1">
                        {[1, 2, 3, 5, 10].map((preset) => (
                          <button
                            key={preset}
                            type="button"
                            onClick={() => updateForm("targetEpochs", preset)}
                            disabled={isRunning || isPaused}
                            className={cn(
                              "px-2 py-0.5 rounded text-[10px] font-bold border transition-colors",
                              form.targetEpochs === preset
                                ? "bg-[#45CFFF]/20 text-[#45CFFF] border-[#45CFFF]"
                                : "bg-[#0E1420] text-[#91A0B4] border-[#273548] hover:border-[#384C66] hover:text-[#ECF3FC]"
                            )}
                          >
                            {preset}x
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="flex items-center gap-1.5 text-xs text-[#91A0B4]">
                      <Calculator className="w-3.5 h-3.5 text-[#45CFFF]" />
                      <span>
                        Batch Tokens / Step: <strong className="text-[#80DEFF]">{tokensPerStep.toLocaleString()}</strong>
                      </span>
                    </div>
                  </div>

                  <div className="text-[11px] text-[#91A0B4] flex flex-wrap items-center justify-between gap-2 pt-1 border-t border-[#45CFFF]/20">
                    <div>
                      {datasetTokens > 0 ? (
                        <span>
                          Dataset Size: <strong className="text-[#ECF3FC]">{datasetTokens.toLocaleString()} tokens</strong> (~
                          <strong className="text-[#45CFFF]">{stepsPerEpoch.toLocaleString()} steps / epoch</strong>)
                        </span>
                      ) : (
                        <span className="text-[#F2B85B] flex items-center gap-1">
                          <Info className="w-3 h-3" />
                          Dataset not pre-tokenized yet. Click <strong>TOKENIZE</strong> above to calculate exact step count.
                        </span>
                      )}
                    </div>
                    <div>
                      Total Run Target:{" "}
                      <strong className="text-[#45CFFF] font-bold">
                        {estimatedEpochSteps > 0
                          ? `~${estimatedEpochSteps.toLocaleString()} steps (${(datasetTokens * form.targetEpochs).toLocaleString()} tokens)`
                          : `${form.targetEpochs} full passes over dataset`}
                      </strong>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="p-3 rounded-lg bg-[#090D14]/70 border border-[#273548] space-y-2">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <label className="text-[11px] text-[#ECF3FC] font-bold uppercase">
                        Total Training Steps:
                      </label>
                      <input
                        type="number"
                        min="100"
                        step="500"
                        value={form.totalSteps}
                        onChange={(e) => updateForm("totalSteps", Math.max(10, parseInt(e.target.value) || 1000))}
                        disabled={isRunning || isPaused}
                        className="w-28 bg-[#0E1420] border border-[#273548] rounded px-2.5 py-1 text-[#ECF3FC] font-bold text-center text-sm focus:outline-none focus:border-[#45CFFF]"
                      />
                      <div className="flex items-center gap-1">
                        {[2000, 5000, 10000, 25000].map((preset) => (
                          <button
                            key={preset}
                            type="button"
                            onClick={() => updateForm("totalSteps", preset)}
                            disabled={isRunning || isPaused}
                            className={cn(
                              "px-2 py-0.5 rounded text-[10px] font-bold border transition-colors",
                              form.totalSteps === preset
                                ? "bg-[#45CFFF]/20 text-[#45CFFF] border-[#45CFFF]"
                                : "bg-[#0E1420] text-[#91A0B4] border-[#273548] hover:border-[#384C66] hover:text-[#ECF3FC]"
                            )}
                          >
                            {(preset / 1000)}k
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="text-xs text-[#91A0B4]">
                      Total Token Budget:{" "}
                      <strong className="text-[#80DEFF]">
                        {((form.totalSteps * tokensPerStep) / 1_000_000).toFixed(2)}M tokens
                      </strong>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div>
                <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                  Optimizer
                </label>
                <select
                  value={form.optimizer}
                  onChange={(e) => updateForm("optimizer", e.target.value)}
                  disabled={isRunning || isPaused}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded px-2.5 py-1.5 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF]"
                >
                  <option value="muon_adamw">Muon + AdamW (Fast)</option>
                  <option value="adamw">AdamW Standard</option>
                  <option value="sgd">SGD with Momentum</option>
                </select>
              </div>

              <div>
                <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                  LR Schedule
                </label>
                <select
                  value={form.schedulerType}
                  onChange={(e) => updateForm("schedulerType", e.target.value)}
                  disabled={isRunning || isPaused}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded px-2.5 py-1.5 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF]"
                >
                  <option value="wsd">Warmup-Stable-Decay (WSD)</option>
                  <option value="cosine">Cosine Annealing</option>
                  <option value="linear">Linear Decay</option>
                  <option value="constant">Constant LR</option>
                </select>
              </div>

              <div>
                <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                  AdamW LR (Embed/Norm)
                </label>
                <input
                  type="number"
                  step="0.0001"
                  value={form.peakLr}
                  onChange={(e) => updateForm("peakLr", parseFloat(e.target.value) || 0.003)}
                  disabled={isRunning || isPaused}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded px-2.5 py-1.5 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF]"
                />
              </div>

              <div>
                <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                  Muon LR (Linear Weights)
                </label>
                <input
                  type="number"
                  step="0.001"
                  value={form.muonLr}
                  onChange={(e) => updateForm("muonLr", parseFloat(e.target.value) || 0.02)}
                  disabled={isRunning || isPaused}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded px-2.5 py-1.5 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF]"
                />
              </div>
            </div>

            {(form.muonLr > 0.05 || form.peakLr > 0.01) && (
              <div className="p-2 rounded bg-[#F2B85B]/15 border border-[#F2B85B]/30 text-[#F2B85B] text-[10px] flex items-center gap-2">
                <AlertTriangle className="w-3.5 h-3.5 text-[#F2B85B] shrink-0" />
                <span>
                  High Learning Rate Warning: Muon LR &gt; 0.05 or AdamW LR &gt; 0.01 may cause gradient instability or loss divergence.
                </span>
              </div>
            )}

            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 pt-1">
              <div>
                <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                  Warmup Steps
                </label>
                <input
                  type="number"
                  value={form.warmupSteps}
                  onChange={(e) => updateForm("warmupSteps", parseInt(e.target.value) || 500)}
                  disabled={isRunning || isPaused}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded px-2.5 py-1.5 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF]"
                />
              </div>

              <div>
                <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                  Stable Steps
                </label>
                <input
                  type="number"
                  value={form.stableSteps}
                  onChange={(e) => updateForm("stableSteps", parseInt(e.target.value) || 7500)}
                  disabled={isRunning || isPaused}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded px-2.5 py-1.5 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF]"
                />
              </div>

              <div>
                <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                  Min LR
                </label>
                <input
                  type="number"
                  step="0.00001"
                  value={form.minLr}
                  onChange={(e) => updateForm("minLr", parseFloat(e.target.value) || 0.0)}
                  disabled={isRunning || isPaused}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded px-2.5 py-1.5 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF]"
                />
              </div>
            </div>
          </div>

          {/* Section 3: Batch Geometry & VRAM Constraints */}
          <div className="glass-panel p-4 rounded-lg space-y-3">
            <div className="flex justify-between items-center pb-2 border-b border-[#273548]">
              <h3 className="text-xs font-bold text-[#ECF3FC] uppercase tracking-wider flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-[#45CFFF]" /> BATCH GEOMETRY & GPU ALLOCATION
              </h3>
              <span className="text-[#45CFFF] font-bold">
                Effective Batch: {effectiveBatchSize} sequences ({tokensPerStep.toLocaleString()} tokens/step)
              </span>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div>
                <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                  Micro-Batch Size
                </label>
                <input
                  type="number"
                  value={form.microBatchSize}
                  onChange={(e) => updateForm("microBatchSize", parseInt(e.target.value) || 4)}
                  disabled={isRunning || isPaused}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded px-2.5 py-1.5 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF]"
                />
              </div>

              <div>
                <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                  Grad Accum Steps
                </label>
                <input
                  type="number"
                  value={form.gradAccumSteps}
                  onChange={(e) => updateForm("gradAccumSteps", parseInt(e.target.value) || 8)}
                  disabled={isRunning || isPaused}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded px-2.5 py-1.5 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF]"
                />
              </div>

              <div>
                <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                  Seq Length
                </label>
                <input
                  type="number"
                  value={form.seqLen}
                  onChange={(e) => updateForm("seqLen", parseInt(e.target.value) || 256)}
                  disabled={isRunning || isPaused}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded px-2.5 py-1.5 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF]"
                />
              </div>

              <div>
                <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                  Grad Clip Norm
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={form.gradClip}
                  onChange={(e) => updateForm("gradClip", parseFloat(e.target.value) || 1.0)}
                  disabled={isRunning || isPaused}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded px-2.5 py-1.5 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF]"
                />
              </div>
            </div>

            {/* Intervals & Early Stopping */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-1">
              <div>
                <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                  Checkpoint Every
                </label>
                <input
                  type="number"
                  value={form.chkInterval}
                  onChange={(e) => updateForm("chkInterval", parseInt(e.target.value) || 500)}
                  disabled={isRunning || isPaused}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded px-2.5 py-1.5 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF]"
                />
              </div>

              <div>
                <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                  Validation Every
                </label>
                <input
                  type="number"
                  value={form.valInterval}
                  onChange={(e) => updateForm("valInterval", parseInt(e.target.value) || 250)}
                  disabled={isRunning || isPaused}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded px-2.5 py-1.5 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF]"
                />
              </div>

              <div className="col-span-2 flex flex-col gap-2 pt-2">
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={() => updateForm("earlyStoppingEnabled", !form.earlyStoppingEnabled)}
                    disabled={isRunning || isPaused}
                    className={cn(
                      "px-3 py-1.5 rounded text-xs border font-bold transition-colors cursor-pointer",
                      form.earlyStoppingEnabled
                        ? "bg-[#F2B85B]/20 text-[#F2B85B] border-[#F2B85B]/40"
                        : "bg-[#0E1420] text-[#91A0B4] border-[#273548]"
                    )}
                  >
                    Early Stopping: {form.earlyStoppingEnabled ? "ENABLED" : "DISABLED (Pretraining default)"}
                  </button>
                  <span className="text-[10px] text-[#5E7188]">
                    Prevents premature halt on long pre-training runs.
                  </span>
                </div>
                {form.earlyStoppingEnabled && (
                  <div className="grid grid-cols-2 gap-2 pt-1">
                    <div>
                      <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-0.5">
                        Patience (Evals)
                      </label>
                      <input
                        type="number"
                        value={form.patience}
                        onChange={(e) => updateForm("patience", parseInt(e.target.value) || 20)}
                        disabled={isRunning || isPaused}
                        className="w-full bg-[#0E1420] border border-[#273548] rounded px-2 py-1 text-[#ECF3FC] text-xs focus:outline-none focus:border-[#45CFFF]"
                      />
                    </div>
                    <div>
                      <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-0.5">
                        Min Steps Before Stop
                      </label>
                      <input
                        type="number"
                        value={form.minSteps}
                        onChange={(e) => updateForm("minSteps", parseInt(e.target.value) || 500)}
                        disabled={isRunning || isPaused}
                        className="w-full bg-[#0E1420] border border-[#273548] rounded px-2 py-1 text-[#ECF3FC] text-xs focus:outline-none focus:border-[#45CFFF]"
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Mathematical Schedule Visualizer & Live Loss */}
        <div className="space-y-4">
          <ChartPanel
            title="ANALYTICAL SCHEDULE PREVIEW"
            labels={schedulerCurve.steps.filter((_, i) => i % 10 === 0)}
            datasets={[
              {
                label: "Planned LR",
                data: schedulerCurve.lrs.filter((_, i) => i % 10 === 0),
                borderColor: "#45CFFF",
                backgroundColor: "rgba(69, 207, 255, 0.1)",
                fill: true,
              },
            ]}
            yAxisLabel="LR"
            height={220}
          />

          <ChartPanel
            title="ACTIVE RUN TELEMETRY"
            labels={chartSteps}
            datasets={[
              {
                label: "Train Loss",
                data: lossSeries,
                borderColor: "#80DEFF",
              },
              ...(hasValLoss
                ? [
                    {
                      label: "Val Loss",
                      data: valLossSeries,
                      borderColor: "#35D49A",
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
        </div>
      </div>

      {/* Confirm Stop Dialog */}
      <ConfirmDialog
        isOpen={confirmStop}
        title="HALT TRAINING RUN"
        message="Are you sure you want to stop the active training worker? State will be saved to disk."
        confirmLabel="HALT TRAINING"
        cancelLabel="CONTINUE"
        variant="danger"
        onConfirm={handleStopRun}
        onCancel={() => setConfirmStop(false)}
      />
    </div>
  );
};
