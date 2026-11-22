"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useSystemStatus } from "@/hooks/useSystemStatus";
import { useHardwareTelemetry } from "@/hooks/useHardwareTelemetry";
import { api } from "@/services/api";
import { ChartPanel } from "@/components/ui/ChartPanel";
import { StatusIndicator } from "@/components/ui/StatusIndicator";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { SaveModelModal } from "@/components/ui/SaveModelModal";
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
  HardDrive,
} from "lucide-react";
import {
  cn,
  formatCompactNumber,
  formatNumber,
  formatDuration,
  formatStepLatency,
  formatParamCount,
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
  useLora: boolean;
  loraR: number;
  loraAlpha: number;
  useMultiGpu: boolean;
  selectedGpuIds: number[];
}

const STORAGE_KEY = "llm_maker_training_config_v2";

const DEFAULT_TRAINING_CONFIG: TrainingFormConfig = {
  selectedNetwork: "rocanon_48m",
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
  useLora: false,
  loraR: 16,
  loraAlpha: 32,
  useMultiGpu: false,
  selectedGpuIds: [0],
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
    activeTrainingTime,
  } = useWebSocket();
  const { status, refresh: refreshStatus } = useSystemStatus(3000);
  const { telemetry } = useHardwareTelemetry(5000);
  const computeLabel = telemetry?.device_name || "Hardware Accelerator";

  // Available options from backend
  const [networks, setNetworks] = useState<NetworkConfig[]>([]);
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [checkpoints, setCheckpoints] = useState<CheckpointItem[]>([]);
  const [computeTopology, setComputeTopology] = useState<any>(null);
  const [schedulerCurve, setSchedulerCurve] = useState<{ steps: number[]; lrs: number[] }>({
    steps: [],
    lrs: [],
  });

  // Persistent Form State
  const [form, setForm] = useState<TrainingFormConfig>(DEFAULT_TRAINING_CONFIG);
  const [isTokenizingDataset, setIsTokenizingDataset] = useState<boolean>(false);

  // Dialogs & Notifications
  const [confirmStop, setConfirmStop] = useState(false);
  const [isSaveModalOpen, setIsSaveModalOpen] = useState(false);
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
  }, []);

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
    if (window.confirm("Reset all training hyperparameters to defaults?")) {
      setForm(DEFAULT_TRAINING_CONFIG);
      if (typeof window !== "undefined") {
        try {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(DEFAULT_TRAINING_CONFIG));
        } catch (e) {
          console.warn("Failed to clear config in localStorage:", e);
        }
      }
      setFeedback("Hyperparameters reset to defaults.");
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
    api.getComputeTopology().then(setComputeTopology).catch(console.error);
  }, []);

  // Sync historical telemetry across tabs if buffer was empty
  useEffect(() => {
    if (status?.run_id && metricsHistory.length === 0) {
      loadRunHistory(status.run_id);
    }
  }, [status?.run_id, metricsHistory.length, loadRunHistory]);

  const currentDataset = useMemo(
    () => datasets.find((d) => d.id === form.selectedDataset),
    [datasets, form.selectedDataset]
  );

  const datasetTokens = currentDataset?.train_tokens || currentDataset?.total_tokens || 0;

  const effectiveBatchSize = form.microBatchSize * form.gradAccumSteps;
  const tokensPerStep = effectiveBatchSize * form.seqLen;

  const stepsPerEpoch = useMemo(() => {
    if (tokensPerStep <= 0 || datasetTokens <= 0) return 0;
    return Math.ceil(datasetTokens / tokensPerStep);
  }, [datasetTokens, tokensPerStep]);

  const estimatedEpochSteps = stepsPerEpoch > 0 ? stepsPerEpoch * form.targetEpochs : 0;
  const plannedSteps = form.durationMode === "epochs" && estimatedEpochSteps > 0 ? estimatedEpochSteps : form.totalSteps;

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

  const handleStartTraining = async () => {
    setIsSubmitting(true);
    try {
      const payload: any = {
        network_id: form.selectedNetwork,
        dataset_id: form.selectedDataset,
        task: form.selectedTask,
        optimizer: form.optimizer,
        scheduler: form.schedulerType,
        duration_mode: form.durationMode,
        target_epochs: form.targetEpochs,
        micro_batch_size: form.microBatchSize,
        grad_accum_steps: form.gradAccumSteps,
        seq_len: form.seqLen,
        learning_rate: form.peakLr,
        muon_lr: form.muonLr,
        min_lr: form.minLr,
        warmup_steps: form.warmupSteps,
        stable_steps: form.stableSteps,
        precision: form.precision,
        grad_clip: form.gradClip,
        checkpoint_interval: form.chkInterval,
        val_interval: form.valInterval,
        early_stopping: form.earlyStoppingEnabled,
        patience: form.patience,
        min_steps: form.minSteps,
      };

      if (form.durationMode === "epochs") {
        payload.total_steps = plannedSteps;
      } else {
        payload.total_steps = form.totalSteps;
      }

      if (form.resumeCheckpoint) {
        payload.resume_checkpoint = form.resumeCheckpoint;
      }

      if (form.useLora) {
        payload.use_lora = true;
        payload.lora_r = form.loraR;
        payload.lora_alpha = form.loraAlpha;
      }

      if (form.useMultiGpu) {
        payload.use_multi_gpu = true;
        payload.device_ids = form.selectedGpuIds;
      }

      await api.startRun(payload);
      setFeedback("Training initiated successfully.");
      await refreshStatus();
      setTimeout(() => setFeedback(null), 4000);
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
        setFeedback("Training paused.");
      } else {
        await api.resumeRun();
        setFeedback("Training resumed.");
      }
      await refreshStatus();
      setTimeout(() => setFeedback(null), 3000);
    } catch (e: any) {
      alert(`Pause/Resume failed: ${e.message}`);
    }
  };

  const handleCheckpoint = async () => {
    try {
      await api.checkpointRun();
      setFeedback("Checkpoint requested.");
      await refreshStatus();
      setTimeout(() => setFeedback(null), 3000);
    } catch (e: any) {
      alert(`Checkpoint failed: ${e.message}`);
    }
  };

  const handleValidate = async () => {
    try {
      await api.validateRun();
      setFeedback("Validation sweep triggered.");
      await refreshStatus();
      setTimeout(() => setFeedback(null), 3000);
    } catch (e: any) {
      alert(`Validation failed: ${e.message}`);
    }
  };

  const handleStopRun = async () => {
    try {
      await api.stopRun();
      setConfirmStop(false);
      setFeedback("Training stopped.");
      await refreshStatus();
      setTimeout(() => setFeedback(null), 3000);
    } catch (e: any) {
      alert(`Stop failed: ${e.message}`);
    }
  };

  const handleTokenizeSelectedDataset = async () => {
    if (!form.selectedDataset) return;
    setIsTokenizingDataset(true);
    setFeedback(`Tokenizing dataset ${form.selectedDataset}...`);
    try {
      const res = await api.tokenizeDataset(form.selectedDataset);
      setFeedback(`Pre-tokenized! ${res.token_count?.toLocaleString() || "All"} tokens ready.`);
      const updated = await api.listDatasets();
      setDatasets(updated);
      setTimeout(() => setFeedback(null), 4000);
    } catch (e: any) {
      alert(`Tokenization failed: ${e.message}`);
      setFeedback(null);
    } finally {
      setIsTokenizingDataset(false);
    }
  };

  const chartSteps = metricsHistory.map((m) => m.step);
  const lossSeries = metricsHistory.map((m) => (m.loss !== undefined ? m.loss : (m as any).train_loss ?? 0));
  const valLossSeries = metricsHistory.map((m) =>
    m.val_loss !== null && m.val_loss !== undefined ? Number(m.val_loss) : null
  );
  const hasValLoss = metricsHistory.some((m) => m.val_loss !== null && m.val_loss !== undefined);

  const currentStep = latestMetric?.step ?? status?.latest_metrics?.step ?? 0;
  const currentLoss = latestMetric?.loss ?? (latestMetric as any)?.train_loss ?? status?.latest_metrics?.loss ?? 0;
  const valLoss = latestMetric?.val_loss ?? null;
  const effectiveValLoss = valLoss !== null ? valLoss : latestValLoss ?? status?.latest_metrics?.val_loss ?? null;
  const effectiveValStep = latestMetric?.latest_val_step ?? latestValStep ?? null;

  const muonEffLr = latestMetric?.muon_lr ?? 0;
  const adamwEffLr = latestMetric?.adamw_lr ?? latestMetric?.learning_rate ?? 0;
  const schedPhase = (latestMetric as any)?.sched_phase || "stable";
  const schedFactor = (latestMetric as any)?.sched_factor ?? 1.0;
  const isHighLr = muonEffLr > 0.04 || adamwEffLr > 0.008;

  const activeDurationMode = latestMetric?.duration_mode || form.durationMode;
  const currentEpoch = latestMetric?.current_epoch || 1;
  const displayTargetEpochs = latestMetric?.target_epochs || form.targetEpochs;
  const tokensInEpoch = latestMetric?.tokens_in_epoch || 0;
  const tokensPerEpoch = latestMetric?.tokens_per_epoch || datasetTokens || 0;
  const epochPercent = tokensPerEpoch > 0 ? Math.min(100, Math.round((tokensInEpoch / tokensPerEpoch) * 100)) : 0;
  const tokensSeen = latestMetric?.tokens_seen ?? (latestMetric as any)?.tokens ?? status?.latest_metrics?.tokens_seen ?? 0;
  const tokPerSec = latestMetric?.tokens_per_second ?? (latestMetric as any)?.tok_per_sec ?? status?.latest_metrics?.tokens_per_second ?? 0;
  const stepTimeMs = latestMetric?.step_time_ms ?? ((latestMetric as any)?.dt ? (latestMetric as any).dt * 1000 : 0);

  const scheduledInterval = latestMetric?.eval_interval || form.valInterval;
  const nextValStep = latestMetric?.next_val_step ?? (currentStep > 0 ? currentStep + (scheduledInterval - (currentStep % scheduledInterval)) : scheduledInterval);

  return (
    <div className="space-y-3 font-mono text-xs">
      {/* Top Header & Actions Controller Bar */}
      <div className="panel p-3 bg-[#202020] border-[#333333] flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <StatusIndicator state={effectiveState} size="md" />
          <div className="flex flex-col">
            <span className="text-xs font-semibold text-[#F2F2F2] flex items-center gap-1.5 uppercase">
              <span>Training Controller</span>
              {status?.run_id && <span className="text-[#A3A3A3] normal-case font-mono-num">({status.run_id})</span>}
            </span>
            <span className="text-[10px] text-[#737373] mt-0.5">
              Precision: {form.precision.toUpperCase()} | Compute: {computeLabel}
            </span>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-1.5">
          {!isRunning && !isPaused ? (
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() => setIsSaveModalOpen(true)}
                className="btn btn-secondary btn-sm"
              >
                <HardDrive className="w-3 h-3 text-[#A3A3A3]" /> Save Model
              </button>
              <button
                onClick={handleStartTraining}
                disabled={isSubmitting || datasets.length === 0 || !form.selectedDataset}
                className="btn btn-primary btn-sm flex items-center gap-1"
                title={datasets.length === 0 ? "Upload a dataset in this workspace first" : undefined}
              >
                <Play className="w-3 h-3 fill-current" /> Start Training
              </button>
            </div>
          ) : (
            <>
              <button
                onClick={handleTogglePause}
                className="btn btn-secondary btn-sm"
              >
                {isRunning ? (
                  <>
                    <Pause className="w-3 h-3 text-[#facc15]" /> Pause
                  </>
                ) : (
                  <>
                    <Play className="w-3 h-3 text-[#F2F2F2]" /> Resume
                  </>
                )}
              </button>

              <button
                onClick={handleCheckpoint}
                className="btn btn-secondary btn-sm"
              >
                <Bookmark className="w-3 h-3 text-[#A3A3A3]" /> Checkpoint
              </button>

              <button
                type="button"
                onClick={() => setIsSaveModalOpen(true)}
                className="btn btn-secondary btn-sm"
              >
                <HardDrive className="w-3 h-3 text-[#A3A3A3]" /> Save Model
              </button>

              <button
                onClick={handleValidate}
                className="btn btn-ghost btn-sm"
              >
                <CheckCircle2 className="w-3 h-3 text-[#A3A3A3]" /> Validate
              </button>

              <button
                onClick={() => setConfirmStop(true)}
                className="btn btn-danger btn-sm"
              >
                <Square className="w-3 h-3 fill-current" /> Stop
              </button>
            </>
          )}
        </div>
      </div>

      {feedback && (
        <div className="p-2 rounded bg-[#262626] border border-[#333333] text-[#F2F2F2] text-xs flex items-center gap-2 font-mono">
          <CheckCircle2 className="w-3.5 h-3.5 text-[#4ade80]" /> {feedback}
        </div>
      )}

      {/* Live Telemetry & Progress Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 select-none font-mono">
        {/* Train Loss */}
        <div className="panel p-2.5 bg-[#202020] border-[#333333]">
          <div className="text-[10px] text-[#737373] uppercase font-medium flex items-center justify-between">
            <span>Train Loss</span>
            <TrendingDown className="w-3 h-3 text-[#A3A3A3]" />
          </div>
          <div className="text-base font-semibold font-mono-num text-[#F2F2F2] mt-1">
            {currentLoss > 0 ? formatNumber(currentLoss, 4) : "—"}
          </div>
          <div className="text-[10px] text-[#525252] mt-0.5">
            Step {currentStep.toLocaleString()}
          </div>
        </div>

        {/* Validation Loss */}
        <div className="panel p-2.5 bg-[#202020] border-[#333333]">
          <div className="text-[10px] text-[#737373] uppercase font-medium flex items-center justify-between">
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
          <div className="text-[10px] text-[#525252] mt-0.5 flex justify-between">
            <span>{effectiveValStep ? `Step ${effectiveValStep}` : "Awaiting"}</span>
            <span>Next: Step {nextValStep}</span>
          </div>
        </div>

        {/* Dataset Progress */}
        <div className="panel p-2.5 bg-[#202020] border-[#333333]">
          <div className="text-[10px] text-[#737373] uppercase font-medium flex items-center justify-between">
            <span>Dataset Progress</span>
            <Activity className="w-3 h-3 text-[#A3A3A3]" />
          </div>
          <div className="text-sm font-semibold font-mono-num text-[#F2F2F2] mt-1">
            {activeDurationMode === "epochs" ? (
              <span>Epoch {currentEpoch} / {displayTargetEpochs}</span>
            ) : (
              <span>Step {currentStep.toLocaleString()} / {form.totalSteps.toLocaleString()}</span>
            )}
          </div>
          <div className="text-[10px] text-[#525252] mt-0.5 truncate">
            {tokensPerEpoch > 0 ? (
              `${formatCompactNumber(tokensInEpoch)} / ${formatCompactNumber(tokensPerEpoch)} (${epochPercent}%)`
            ) : (
              `${formatCompactNumber(tokensSeen)} tokens`
            )}
          </div>
        </div>

        {/* Time & Latency */}
        <div className="panel p-2.5 bg-[#202020] border-[#333333]">
          <div className="text-[10px] text-[#737373] uppercase font-medium flex items-center justify-between">
            <span>Time & Latency</span>
            <Clock className="w-3 h-3 text-[#A3A3A3]" />
          </div>
          <div className="text-base font-semibold font-mono-num text-[#F2F2F2] mt-1">
            {formatDuration(activeTrainingTime)}
          </div>
          <div className="text-[10px] text-[#525252] mt-0.5 flex justify-between">
            <span>{formatStepLatency(stepTimeMs)}</span>
            <span>{tokPerSec > 0 ? `${Math.round(tokPerSec)} tok/s` : ""}</span>
          </div>
        </div>

        {/* Muon LR */}
        <div className="panel p-2.5 bg-[#202020] border-[#333333]">
          <div className="text-[10px] text-[#737373] uppercase font-medium flex items-center justify-between">
            <span>Muon LR (Matrix)</span>
            <Sparkles className="w-3 h-3 text-[#A3A3A3]" />
          </div>
          <div className="text-sm font-semibold font-mono-num text-[#F2F2F2] mt-1 flex items-center gap-1.5">
            <span>{muonEffLr > 0 ? muonEffLr.toExponential(2) : "—"}</span>
            {isHighLr && (
              <span className="text-[#facc15]" title="High LR">
                <AlertTriangle className="w-3 h-3" />
              </span>
            )}
          </div>
          <div className="text-[10px] text-[#525252] mt-0.5">
            Phase: {schedPhase} ({(schedFactor * 100).toFixed(0)}%)
          </div>
        </div>

        {/* AdamW LR */}
        <div className="panel p-2.5 bg-[#202020] border-[#333333]">
          <div className="text-[10px] text-[#737373] uppercase font-medium flex items-center justify-between">
            <span>AdamW LR (Vectors)</span>
            <Gauge className="w-3 h-3 text-[#A3A3A3]" />
          </div>
          <div className="text-sm font-semibold font-mono-num text-[#A3A3A3] mt-1">
            {adamwEffLr > 0 ? adamwEffLr.toExponential(2) : "—"}
          </div>
          <div className="text-[10px] text-[#525252] mt-0.5">
            Floor: {form.minLr.toExponential(1)}
          </div>
        </div>
      </div>

      {/* Main Grid: Config + Telemetry */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        {/* Left Column: Model, Dataset, & Hyperparams */}
        <div className="lg:col-span-2 space-y-3">
          {/* Section 1: Architecture & Dataset Selection */}
          <div className="panel p-3.5 bg-[#202020] border-[#333333] space-y-2.5">
            <div className="flex items-center justify-between pb-1.5 border-b border-[#262626]">
              <h3 className="panel-title">
                <Layers className="w-3.5 h-3.5 text-[#737373]" /> Architecture & Dataset
              </h3>
              <button
                type="button"
                onClick={handleResetDefaults}
                disabled={isRunning || isPaused}
                className="inline-flex items-center gap-1 text-[10px] text-[#737373] hover:text-[#F2F2F2] transition-colors disabled:opacity-40"
                title="Reset hyperparameters to defaults"
              >
                <RotateCcw className="w-3 h-3" /> Reset Defaults
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
              <div>
                <label className="text-[10px] text-[#737373] uppercase font-medium block mb-1">
                  Architecture Model
                </label>
                <div className="flex items-center gap-1.5">
                  <select
                    value={form.selectedNetwork}
                    onChange={(e) => updateForm("selectedNetwork", e.target.value)}
                    disabled={isRunning || isPaused}
                    className="input flex-1"
                  >
                    {networks.map((n) => {
                      const pCount = n.total_params ?? (n as any).total_parameters;
                      return (
                        <option key={n.id} value={n.id}>
                          {n.name} {pCount ? `(${formatParamCount(pCount)})` : ""}
                        </option>
                      );
                    })}
                  </select>
                  <a
                    href={`/studio?network=${encodeURIComponent(form.selectedNetwork)}`}
                    className="btn btn-secondary btn-sm px-2 text-[10px]"
                    title="Edit in Network Studio"
                  >
                    <Sliders className="w-3 h-3 text-[#737373]" />
                    <span>Studio</span>
                  </a>
                </div>
              </div>

              <div>
                <label className="text-[10px] text-[#737373] uppercase font-medium block mb-1">
                  Dataset Source
                </label>
                <div className="flex items-center gap-1.5">
                  <select
                    value={form.selectedDataset}
                    onChange={(e) => updateForm("selectedDataset", e.target.value)}
                    disabled={isRunning || isPaused || datasets.length === 0}
                    className="input flex-1"
                  >
                    {datasets.length === 0 ? (
                      <option value="">No datasets in workspace (Upload in Datasets)</option>
                    ) : (
                      datasets.map((d) => (
                        <option key={d.id} value={d.id}>
                          {d.name} {d.is_uploaded ? "(Uploaded)" : ""}
                        </option>
                      ))
                    )}
                  </select>
                  <button
                    type="button"
                    onClick={handleTokenizeSelectedDataset}
                    disabled={isTokenizingDataset || !form.selectedDataset || isRunning || isPaused}
                    className="btn btn-secondary btn-sm px-2 text-[10px]"
                    title="Pre-tokenize selected dataset"
                  >
                    {isTokenizingDataset ? (
                      <RefreshCw className="w-3 h-3 animate-spin" />
                    ) : (
                      <Sparkles className="w-3 h-3 text-[#737373]" />
                    )}
                    Tokenize
                  </button>
                </div>
              </div>

              <div>
                <label className="text-[10px] text-[#737373] uppercase font-medium block mb-1">
                  Training Task
                </label>
                <select
                  value={form.selectedTask}
                  onChange={(e) => updateForm("selectedTask", e.target.value)}
                  disabled={isRunning || isPaused}
                  className="input"
                >
                  <optgroup label="Language Modeling">
                    <option value="causal_lm">Causal LM (Autoregressive)</option>
                    <option value="instruction_sft">Instruction Fine-Tuning (SFT)</option>
                    <option value="masked_lm">Masked LM (Bidirectional)</option>
                  </optgroup>
                  <optgroup label="Post-Training Alignment & Reasoning">
                    <option value="simpo">SimPO (Simple Preference Optimization - Ref Free)</option>
                    <option value="orpo">ORPO (Odds Ratio Preference Optimization)</option>
                    <option value="grpo">GRPO (DeepSeek-R1 Style Reasoning RL)</option>
                    <option value="dpo">DPO (Direct Preference Optimization)</option>
                  </optgroup>
                  <optgroup label="Supervised Heads">
                    <option value="classification">Classification</option>
                    <option value="regression">Regression</option>
                  </optgroup>
                </select>
              </div>
            </div>

            {/* Resume from checkpoint */}
            <div>
              <label className="text-[10px] text-[#737373] uppercase font-medium block mb-1">
                Resume From Checkpoint (Optional)
              </label>
              <select
                value={form.resumeCheckpoint}
                onChange={(e) => updateForm("resumeCheckpoint", e.target.value)}
                disabled={isRunning || isPaused || checkpoints.length === 0}
                className="input"
              >
                <option value="">{checkpoints.length === 0 ? "No checkpoints in workspace (Start Fresh)" : "Start Fresh (No checkpoint restore)"}</option>
                {checkpoints.map((c) => {
                  const valStr = c.val_loss !== null && c.val_loss !== undefined ? ` | Val: ${Number(c.val_loss).toFixed(4)}` : "";
                  const prefix = c.is_saved_model ? "📦 Model:" : "💾 Checkpoint:";
                  const name = c.model_name || c.filename;
                  return (
                    <option key={c.id || c.path || c.filename} value={c.path || c.filename}>
                      {prefix} {name} (Step {c.step}{valStr})
                    </option>
                  );
                })}
              </select>
            </div>
          </div>

          {/* Section 2: Optimizer, Duration Mode & Learning Rates */}
          <div className="panel p-3.5 bg-[#202020] border-[#333333] space-y-2.5">
            <h3 className="panel-title pb-1.5 border-b border-[#262626]">
              <Sparkles className="w-3.5 h-3.5 text-[#737373]" /> Optimizer & Schedule
            </h3>

            {/* Duration Mode Selector */}
            <div className="p-2.5 rounded bg-[#1B1B1B] border border-[#262626] space-y-2">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
                <div>
                  <span className="text-xs text-[#F2F2F2] font-medium uppercase">
                    Duration Mode
                  </span>
                  <span className="text-[10px] text-[#737373] block mt-0.5">
                    Train for dataset epochs or fixed optimization steps.
                  </span>
                </div>
                <div className="flex items-center gap-1 bg-[#202020] p-0.5 rounded border border-[#333333]">
                  <button
                    type="button"
                    onClick={() => updateForm("durationMode", "steps")}
                    disabled={isRunning || isPaused}
                    className={cn(
                      "px-3 py-1 rounded text-xs transition-colors",
                      form.durationMode === "steps"
                        ? "bg-[#D4D4D4] text-[#171717] font-semibold"
                        : "text-[#737373] hover:text-[#F2F2F2]"
                    )}
                  >
                    Steps
                  </button>
                  <button
                    type="button"
                    onClick={() => updateForm("durationMode", "epochs")}
                    disabled={isRunning || isPaused}
                    className={cn(
                      "px-3 py-1 rounded text-xs transition-colors",
                      form.durationMode === "epochs"
                        ? "bg-[#D4D4D4] text-[#171717] font-semibold"
                        : "text-[#737373] hover:text-[#F2F2F2]"
                    )}
                  >
                    Epochs
                  </button>
                </div>
              </div>

              {form.durationMode === "epochs" ? (
                <div className="p-2 rounded bg-[#202020] border border-[#262626] space-y-1.5">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <label className="text-[10px] text-[#A3A3A3] uppercase">
                        Epochs:
                      </label>
                      <input
                        type="number"
                        min="1"
                        max="100"
                        value={form.targetEpochs}
                        onChange={(e) => updateForm("targetEpochs", Math.max(1, parseInt(e.target.value) || 1))}
                        disabled={isRunning || isPaused}
                        className="input w-16 text-center"
                      />
                      <div className="flex items-center gap-1">
                        {[1, 2, 3, 5].map((preset) => (
                          <button
                            key={preset}
                            type="button"
                            onClick={() => updateForm("targetEpochs", preset)}
                            disabled={isRunning || isPaused}
                            className={cn(
                              "px-1.5 py-0.5 rounded text-[10px] border transition-colors",
                              form.targetEpochs === preset
                                ? "bg-[#262626] text-[#F2F2F2] border-[#D4D4D4]"
                                : "bg-[#1B1B1B] text-[#737373] border-[#262626]"
                            )}
                          >
                            {preset}x
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="flex items-center gap-1 text-[11px] text-[#737373]">
                      <Calculator className="w-3 h-3" />
                      <span>Tokens/Step: <strong className="text-[#F2F2F2]">{tokensPerStep.toLocaleString()}</strong></span>
                    </div>
                  </div>

                  <div className="text-[10px] text-[#737373] pt-1 border-t border-[#262626] flex justify-between">
                    <span>
                      Dataset: {datasetTokens > 0 ? `${datasetTokens.toLocaleString()} tokens (~${stepsPerEpoch.toLocaleString()} steps/epoch)` : "Tokenize dataset to calculate steps"}
                    </span>
                    <span>
                      Target: {estimatedEpochSteps > 0 ? `~${estimatedEpochSteps.toLocaleString()} steps` : `${form.targetEpochs} passes`}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="p-2 rounded bg-[#202020] border border-[#262626] space-y-1.5">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <label className="text-[10px] text-[#A3A3A3] uppercase">
                        Total Steps:
                      </label>
                      <input
                        type="number"
                        min="100"
                        step="500"
                        value={form.totalSteps}
                        onChange={(e) => updateForm("totalSteps", Math.max(10, parseInt(e.target.value) || 1000))}
                        disabled={isRunning || isPaused}
                        className="input w-24 text-center"
                      />
                      <div className="flex items-center gap-1">
                        {[2000, 5000, 10000, 25000].map((preset) => (
                          <button
                            key={preset}
                            type="button"
                            onClick={() => updateForm("totalSteps", preset)}
                            disabled={isRunning || isPaused}
                            className={cn(
                              "px-1.5 py-0.5 rounded text-[10px] border transition-colors",
                              form.totalSteps === preset
                                ? "bg-[#262626] text-[#F2F2F2] border-[#D4D4D4]"
                                : "bg-[#1B1B1B] text-[#737373] border-[#262626]"
                            )}
                          >
                            {(preset / 1000)}k
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="text-[10px] text-[#737373]">
                      Tokens: <strong className="text-[#F2F2F2]">{((form.totalSteps * tokensPerStep) / 1_000_000).toFixed(2)}M</strong>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
              <div>
                <label className="text-[10px] text-[#737373] uppercase font-medium block mb-1">
                  Optimizer
                </label>
                <select
                  value={form.optimizer}
                  onChange={(e) => updateForm("optimizer", e.target.value)}
                  disabled={isRunning || isPaused}
                  className="input"
                >
                  <option value="muon_adamw">Muon + AdamW</option>
                  <option value="adamw">AdamW Standard</option>
                  <option value="sgd">SGD</option>
                </select>
              </div>

              <div>
                <label className="text-[10px] text-[#737373] uppercase font-medium block mb-1">
                  Schedule
                </label>
                <select
                  value={form.schedulerType}
                  onChange={(e) => updateForm("schedulerType", e.target.value)}
                  disabled={isRunning || isPaused}
                  className="input"
                >
                  <option value="wsd">WSD (Warmup-Stable-Decay)</option>
                  <option value="cosine">Cosine</option>
                  <option value="linear">Linear</option>
                  <option value="constant">Constant</option>
                </select>
              </div>

              <div>
                <label className="text-[10px] text-[#737373] uppercase font-medium block mb-1">
                  AdamW LR
                </label>
                <input
                  type="number"
                  step="0.0001"
                  value={form.peakLr}
                  onChange={(e) => updateForm("peakLr", parseFloat(e.target.value) || 0.003)}
                  disabled={isRunning || isPaused}
                  className="input"
                />
              </div>

              <div>
                <label className="text-[10px] text-[#737373] uppercase font-medium block mb-1">
                  Muon LR
                </label>
                <input
                  type="number"
                  step="0.001"
                  value={form.muonLr}
                  onChange={(e) => updateForm("muonLr", parseFloat(e.target.value) || 0.02)}
                  disabled={isRunning || isPaused}
                  className="input"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 gap-2.5 pt-1">
              <div>
                <label className="text-[10px] text-[#737373] uppercase font-medium block mb-1">
                  Warmup Steps
                </label>
                <input
                  type="number"
                  value={form.warmupSteps}
                  onChange={(e) => updateForm("warmupSteps", parseInt(e.target.value) || 500)}
                  disabled={isRunning || isPaused}
                  className="input"
                />
              </div>

              <div>
                <label className="text-[10px] text-[#737373] uppercase font-medium block mb-1">
                  Stable Steps
                </label>
                <input
                  type="number"
                  value={form.stableSteps}
                  onChange={(e) => updateForm("stableSteps", parseInt(e.target.value) || 7500)}
                  disabled={isRunning || isPaused}
                  className="input"
                />
              </div>

              <div>
                <label className="text-[10px] text-[#737373] uppercase font-medium block mb-1">
                  Min LR
                </label>
                <input
                  type="number"
                  step="0.00001"
                  value={form.minLr}
                  onChange={(e) => updateForm("minLr", parseFloat(e.target.value) || 0.0)}
                  disabled={isRunning || isPaused}
                  className="input"
                />
              </div>
            </div>
          </div>

          {/* Section 3: Batch Geometry & Hardware */}
          <div className="panel p-3.5 bg-[#202020] border-[#333333] space-y-2.5">
            <div className="flex justify-between items-center pb-1.5 border-b border-[#262626]">
              <h3 className="panel-title">
                <Sliders className="w-3.5 h-3.5 text-[#737373]" /> Batch Geometry & Hardware
              </h3>
              <span className="text-[10px] text-[#A3A3A3]">
                Effective: {effectiveBatchSize} seqs ({tokensPerStep.toLocaleString()} tok/step)
              </span>
            </div>

            {/* PEFT / LoRA & Multi-GPU Topology Controls */}
            <div className="p-2.5 rounded bg-[#1B1B1B] border border-[#262626] space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <span className="text-xs text-[#F2F2F2] font-medium uppercase">
                    Compute & PEFT Adaptation
                  </span>
                  <span className="text-[10px] text-[#737373] block mt-0.5">
                    LoRA low-rank adapter injection (&gt;80% VRAM saving) &amp; multi-GPU distribution.
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <label className="flex items-center gap-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={form.useLora}
                      onChange={(e) => updateForm("useLora", e.target.checked)}
                      disabled={isRunning || isPaused}
                      className="rounded border-[#333333] bg-[#202020] text-[#D4D4D4] focus:ring-0 w-3.5 h-3.5"
                    />
                    <span className="text-xs text-[#F2F2F2] font-mono">LoRA Adapter</span>
                  </label>

                  {computeTopology?.gpu_count > 1 && (
                    <label className="flex items-center gap-1.5 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={form.useMultiGpu}
                        onChange={(e) => updateForm("useMultiGpu", e.target.checked)}
                        disabled={isRunning || isPaused}
                        className="rounded border-[#333333] bg-[#202020] text-[#D4D4D4] focus:ring-0 w-3.5 h-3.5"
                      />
                      <span className="text-xs text-[#F2F2F2] font-mono">Multi-GPU ({computeTopology.gpu_count}x)</span>
                    </label>
                  )}
                </div>
              </div>

              {form.useLora && (
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 p-2 rounded bg-[#202020] border border-[#262626]">
                  <div>
                    <label className="text-[10px] text-[#A3A3A3] uppercase block mb-1">
                      LoRA Rank (r)
                    </label>
                    <select
                      value={form.loraR}
                      onChange={(e) => updateForm("loraR", parseInt(e.target.value) || 16)}
                      disabled={isRunning || isPaused}
                      className="input text-xs"
                    >
                      <option value="8">r = 8 (Minimal VRAM)</option>
                      <option value="16">r = 16 (Standard)</option>
                      <option value="32">r = 32 (High Capacity)</option>
                      <option value="64">r = 64 (Dense Adapter)</option>
                    </select>
                  </div>

                  <div>
                    <label className="text-[10px] text-[#A3A3A3] uppercase block mb-1">
                      LoRA Alpha (α)
                    </label>
                    <select
                      value={form.loraAlpha}
                      onChange={(e) => updateForm("loraAlpha", parseInt(e.target.value) || 32)}
                      disabled={isRunning || isPaused}
                      className="input text-xs"
                    >
                      <option value="16">α = 16</option>
                      <option value="32">α = 32 (Standard 2x)</option>
                      <option value="64">α = 64</option>
                      <option value="128">α = 128</option>
                    </select>
                  </div>

                  <div className="col-span-2 sm:col-span-1 flex flex-col justify-center text-[10px] text-[#737373]">
                    <span>Scaling factor: <strong className="text-[#F2F2F2]">{(form.loraAlpha / (form.loraR || 1)).toFixed(2)}x</strong></span>
                    <span className="text-[9px] text-[#4ade80]">Freezes base model &gt;80% VRAM saving</span>
                  </div>
                </div>
              )}

              {form.useMultiGpu && computeTopology?.devices?.length > 1 && (
                <div className="p-2 rounded bg-[#202020] border border-[#262626] space-y-1.5">
                  <span className="text-[10px] text-[#A3A3A3] uppercase block">
                    Active CUDA Devices
                  </span>
                  <div className="flex flex-wrap gap-2">
                    {computeTopology.devices.map((dev: any) => {
                      const isSelected = form.selectedGpuIds.includes(dev.index);
                      return (
                        <button
                          key={dev.index}
                          type="button"
                          disabled={isRunning || isPaused}
                          onClick={() => {
                            const current = form.selectedGpuIds;
                            const next = isSelected
                              ? current.filter((id) => id !== dev.index)
                              : [...current, dev.index];
                            updateForm("selectedGpuIds", next.length > 0 ? next : [0]);
                          }}
                          className={cn(
                            "px-2 py-1 rounded text-xs border font-mono flex items-center gap-1.5 transition-colors",
                            isSelected
                              ? "bg-[#262626] text-[#F2F2F2] border-[#D4D4D4]"
                              : "bg-[#1B1B1B] text-[#737373] border-[#262626]"
                          )}
                        >
                          <HardDrive className="w-3 h-3" />
                          <span>GPU {dev.index}: {dev.name} ({Math.round(dev.total_vram_mb / 1024)}GB)</span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
              <div>
                <label className="text-[10px] text-[#737373] uppercase font-medium block mb-1">
                  Micro-Batch Size
                </label>
                <input
                  type="number"
                  value={form.microBatchSize}
                  onChange={(e) => updateForm("microBatchSize", parseInt(e.target.value) || 4)}
                  disabled={isRunning || isPaused}
                  className="input"
                />
              </div>

              <div>
                <label className="text-[10px] text-[#737373] uppercase font-medium block mb-1">
                  Grad Accum Steps
                </label>
                <input
                  type="number"
                  value={form.gradAccumSteps}
                  onChange={(e) => updateForm("gradAccumSteps", parseInt(e.target.value) || 8)}
                  disabled={isRunning || isPaused}
                  className="input"
                />
              </div>

              <div>
                <label className="text-[10px] text-[#737373] uppercase font-medium block mb-1">
                  Seq Length
                </label>
                <input
                  type="number"
                  value={form.seqLen}
                  onChange={(e) => updateForm("seqLen", parseInt(e.target.value) || 256)}
                  disabled={isRunning || isPaused}
                  className="input"
                />
              </div>

              <div>
                <label className="text-[10px] text-[#737373] uppercase font-medium block mb-1">
                  Grad Clip Norm
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={form.gradClip}
                  onChange={(e) => updateForm("gradClip", parseFloat(e.target.value) || 1.0)}
                  disabled={isRunning || isPaused}
                  className="input"
                />
              </div>
            </div>

            {/* Intervals & Early Stopping */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5 pt-1">
              <div>
                <label className="text-[10px] text-[#737373] uppercase font-medium block mb-1">
                  Checkpoint Every
                </label>
                <input
                  type="number"
                  value={form.chkInterval}
                  onChange={(e) => updateForm("chkInterval", parseInt(e.target.value) || 500)}
                  disabled={isRunning || isPaused}
                  className="input"
                />
              </div>

              <div>
                <label className="text-[10px] text-[#737373] uppercase font-medium block mb-1">
                  Validation Every
                </label>
                <input
                  type="number"
                  value={form.valInterval}
                  onChange={(e) => updateForm("valInterval", parseInt(e.target.value) || 250)}
                  disabled={isRunning || isPaused}
                  className="input"
                />
              </div>

              <div className="col-span-2 flex flex-col gap-1.5 pt-1">
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => updateForm("earlyStoppingEnabled", !form.earlyStoppingEnabled)}
                    disabled={isRunning || isPaused}
                    className={cn(
                      "btn btn-sm text-[10px]",
                      form.earlyStoppingEnabled ? "btn-secondary border-[#D4D4D4]" : "btn-ghost border-[#333333]"
                    )}
                  >
                    Early Stopping: {form.earlyStoppingEnabled ? "Enabled" : "Disabled"}
                  </button>
                  <span className="text-[10px] text-[#525252]">
                    Pre-training default: disabled.
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Analytical Schedule & Live Loss */}
        <div className="space-y-3">
          <ChartPanel
            title="Schedule curve"
            labels={schedulerCurve.steps.filter((_, i) => i % 10 === 0)}
            datasets={[
              {
                label: "Planned LR",
                data: schedulerCurve.lrs.filter((_, i) => i % 10 === 0),
                borderColor: "#D4D4D4",
                borderWidth: 1.4,
              },
            ]}
            yAxisLabel="LR"
            height={200}
          />

          <ChartPanel
            title="Loss curve"
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
                      label: "Val Loss",
                      data: valLossSeries,
                      borderColor: "#A3A3A3",
                      borderDash: [4, 4],
                      pointRadius: 2.5,
                      spanGaps: true,
                    },
                  ]
                : []),
            ]}
            yAxisLabel="Loss"
            height={200}
          />
        </div>
      </div>

      {/* Confirm Stop Dialog */}
      <ConfirmDialog
        isOpen={confirmStop}
        title="Halt training run"
        message="Are you sure you want to stop training? State will be saved to disk."
        confirmLabel="Halt training"
        cancelLabel="Continue"
        variant="danger"
        onConfirm={handleStopRun}
        onCancel={() => setConfirmStop(false)}
      />

      {/* Save Local Model Modal */}
      <SaveModelModal
        isOpen={isSaveModalOpen}
        onClose={() => setIsSaveModalOpen(false)}
        defaultModelId={form.selectedNetwork || "rocanon_48m"}
        onSaved={(pkg) => setFeedback(`Saved local package '${pkg.name}' in models/${pkg.name}/`)}
      />
    </div>
  );
};
