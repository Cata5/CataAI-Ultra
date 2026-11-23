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
  warmupPct: number;
  stablePct: number;
  gradClip: number;
  chkInterval: number;
  valInterval: number;
  earlyStoppingEnabled: boolean;
  patience: number;
  minSteps: number;
  useLora: boolean;
  loraR: number;
  loraAlpha: number;
  deviceMode: string;
  clusterMasterAddr: string;
  clusterMasterPort: number;
  clusterWorldSize: number;
  clusterRank: number;
  useMultiGpu: boolean;
  selectedGpuIds: number[];
}

const STORAGE_KEY = "llm_maker_training_config_v4";

export interface TaskPreset {
  label: string;
  optimizer: string;
  schedulerType: string;
  peakLr: number;
  muonLr: number;
  minLr: number;
  warmupPct: number;
  stablePct: number;
}

export const TASK_PRESETS: Record<string, TaskPreset> = {
  causal_lm: {
    label: "Pre-training (Causal LM)",
    optimizer: "muon_adamw",
    schedulerType: "wsd",
    peakLr: 0.003,
    muonLr: 0.02,
    minLr: 0.0001,
    warmupPct: 3,
    stablePct: 82,
  },
  instruction_sft: {
    label: "Instruction SFT",
    optimizer: "muon_adamw",
    schedulerType: "wsd",
    peakLr: 0.0003,
    muonLr: 0.005,
    minLr: 0.00001,
    warmupPct: 5,
    stablePct: 45,
  },
  masked_lm: {
    label: "Masked LM",
    optimizer: "muon_adamw",
    schedulerType: "wsd",
    peakLr: 0.002,
    muonLr: 0.015,
    minLr: 0.0001,
    warmupPct: 5,
    stablePct: 75,
  },
  simpo: {
    label: "SimPO Alignment",
    optimizer: "adamw",
    schedulerType: "wsd",
    peakLr: 0.00005,
    muonLr: 0.001,
    minLr: 0.000005,
    warmupPct: 10,
    stablePct: 30,
  },
  orpo: {
    label: "ORPO Alignment",
    optimizer: "adamw",
    schedulerType: "wsd",
    peakLr: 0.00005,
    muonLr: 0.001,
    minLr: 0.000005,
    warmupPct: 10,
    stablePct: 30,
  },
  grpo: {
    label: "GRPO Reasoning",
    optimizer: "adamw",
    schedulerType: "wsd",
    peakLr: 0.00002,
    muonLr: 0.0005,
    minLr: 0.000001,
    warmupPct: 10,
    stablePct: 40,
  },
  dpo: {
    label: "DPO Alignment",
    optimizer: "adamw",
    schedulerType: "wsd",
    peakLr: 0.00002,
    muonLr: 0.0005,
    minLr: 0.000002,
    warmupPct: 10,
    stablePct: 20,
  },
  classification: {
    label: "Classification Head",
    optimizer: "adamw",
    schedulerType: "cosine",
    peakLr: 0.0001,
    muonLr: 0.002,
    minLr: 0.00001,
    warmupPct: 10,
    stablePct: 40,
  },
  regression: {
    label: "Regression Head",
    optimizer: "adamw",
    schedulerType: "cosine",
    peakLr: 0.0001,
    muonLr: 0.002,
    minLr: 0.00001,
    warmupPct: 10,
    stablePct: 40,
  },
};

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
  warmupSteps: 300,
  stableSteps: 8200,
  warmupPct: 3,
  stablePct: 82,
  gradClip: 1.0,
  chkInterval: 500,
  valInterval: 250,
  earlyStoppingEnabled: false,
  patience: 20,
  minSteps: 500,
  useLora: false,
  loraR: 16,
  loraAlpha: 32,
  deviceMode: "cuda:0",
  clusterMasterAddr: "127.0.0.1",
  clusterMasterPort: 29500,
  clusterWorldSize: 2,
  clusterRank: 0,
  useMultiGpu: false,
  selectedGpuIds: [0],
};

const getSavedConfig = (): TrainingFormConfig => {
  if (typeof window === "undefined") return DEFAULT_TRAINING_CONFIG;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_TRAINING_CONFIG;
    const parsed = JSON.parse(raw);
    const merged = { ...DEFAULT_TRAINING_CONFIG, ...parsed };
    // Guard against corrupt/stale percentages from older versions
    if (typeof merged.warmupPct !== "number" || merged.warmupPct < 1 || merged.warmupPct > 40) {
      merged.warmupPct = 3;
    }
    if (typeof merged.stablePct !== "number" || merged.stablePct < 15 || merged.stablePct > 95) {
      merged.stablePct = 82;
    }
    return merged;
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

  // Percentage-based warmup & stable calculation
  const calcWarmupSteps = useMemo(() => {
    return Math.max(1, Math.round((plannedSteps * (form.warmupPct || 3)) / 100));
  }, [plannedSteps, form.warmupPct]);

  const calcStableSteps = useMemo(() => {
    return Math.max(1, Math.round((plannedSteps * (form.stablePct || 82)) / 100));
  }, [plannedSteps, form.stablePct]);

  const calcDecaySteps = useMemo(() => {
    return Math.max(0, plannedSteps - calcWarmupSteps - calcStableSteps);
  }, [plannedSteps, calcWarmupSteps, calcStableSteps]);

  const decayPct = Math.max(0, 100 - (form.warmupPct || 3) - (form.stablePct || 82));

  const handleSelectTask = useCallback(
    (newTask: string) => {
      const preset = TASK_PRESETS[newTask];
      setForm((prev) => {
        const next: TrainingFormConfig = {
          ...prev,
          selectedTask: newTask,
          ...(preset
            ? {
                optimizer: preset.optimizer,
                schedulerType: preset.schedulerType,
                peakLr: preset.peakLr,
                muonLr: preset.muonLr,
                minLr: preset.minLr,
                warmupPct: preset.warmupPct,
                stablePct: preset.stablePct,
              }
            : {}),
        };
        if (typeof window !== "undefined") {
          try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
          } catch (e) {
            console.warn("Failed to persist training config:", e);
          }
        }
        return next;
      });

      if (preset) {
        setFeedback(
          `Applied ${preset.label} preset: Peak LR ${preset.peakLr}, Muon LR ${preset.muonLr}, Warmup ${preset.warmupPct}%, Stable ${preset.stablePct}%`
        );
        setTimeout(() => setFeedback(null), 4000);
      }
    },
    []
  );

  useEffect(() => {
    let active = true;
    api
      .getSchedulerCurve({
        schedule_type: form.schedulerType,
        total_steps: plannedSteps,
        peak_lr: form.peakLr,
        min_lr: form.minLr,
        warmup_steps: calcWarmupSteps,
        stable_steps: calcStableSteps,
        decay_style: "cosine",
      })
      .then((res) => {
        if (!active || !res) return;
        let points: Array<{ step: number; lr: number }> = [];
        if (Array.isArray(res)) {
          points = res;
        } else if (Array.isArray((res as any).curve)) {
          points = (res as any).curve;
        }
        if (points.length > 0) {
          setSchedulerCurve({
            steps: points.map((r: any) => r.step),
            lrs: points.map((r: any) => r.lr),
          });
        }
      })
      .catch(console.error);
    return () => {
      active = false;
    };
  }, [form.schedulerType, plannedSteps, form.peakLr, form.minLr, calcWarmupSteps, calcStableSteps]);

  const handleStartTraining = async () => {
    setIsSubmitting(true);
    try {
      const payload: any = {
        network_id: form.selectedNetwork,
        dataset_id: form.selectedDataset,
        dataset_source: form.selectedDataset,
        task: form.selectedTask,
        optimizer: form.optimizer,
        scheduler: form.schedulerType,
        duration_mode: form.durationMode,
        target_epochs: form.targetEpochs,
        micro_batch_size: form.microBatchSize,
        grad_accum_steps: form.gradAccumSteps,
        seq_len: form.seqLen,
        learning_rate: form.peakLr,
        peak_lr: form.peakLr,
        muon_lr: form.muonLr,
        min_lr: form.minLr,
        warmup_steps: calcWarmupSteps,
        stable_steps: calcStableSteps,
        warmup_steps_pct: form.warmupPct,
        stable_steps_pct: form.stablePct,
        precision: form.precision,
        grad_clip: form.gradClip,
        checkpoint_interval: form.chkInterval,
        val_interval: form.valInterval,
        early_stopping: form.earlyStoppingEnabled,
        patience: form.patience,
        min_steps: form.minSteps,
        device: form.deviceMode,
        use_multi_gpu: form.deviceMode === "multi_gpu" || form.useMultiGpu,
        device_ids: form.selectedGpuIds,
        is_cluster: form.deviceMode === "cluster",
        master_addr: form.clusterMasterAddr,
        master_port: form.clusterMasterPort,
        world_size: form.clusterWorldSize,
        rank: form.clusterRank,
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
          <div className="text-[10px] text-[#A3A3A3] font-medium mt-0.5 flex justify-between">
            <span>{effectiveValStep ? `Step ${effectiveValStep}` : "Awaiting"}</span>
            <span>Next: Step {nextValStep}</span>
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
              <span>Step {currentStep.toLocaleString()} / {form.totalSteps.toLocaleString()}</span>
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
          <div className="text-sm font-semibold font-mono-num text-[#F2F2F2] mt-1 flex items-center gap-1.5">
            <span>{muonEffLr > 0 ? muonEffLr.toExponential(2) : "—"}</span>
            {isHighLr && (
              <span className="text-[#facc15]" title="High LR">
                <AlertTriangle className="w-3 h-3" />
              </span>
            )}
          </div>
          <div className="text-[10px] text-[#A3A3A3] font-medium mt-0.5">
            Phase: {schedPhase} ({(schedFactor * 100).toFixed(0)}%)
          </div>
        </div>

        {/* AdamW LR */}
        <div className="panel p-2.5 bg-[#202020] border-[#333333]">
          <div className="text-[10px] text-[#D4D4D4] uppercase font-semibold flex items-center justify-between">
            <span>AdamW LR (Vectors)</span>
            <Gauge className="w-3 h-3 text-[#A3A3A3]" />
          </div>
          <div className="text-sm font-semibold font-mono-num text-[#A3A3A3] mt-1">
            {adamwEffLr > 0 ? adamwEffLr.toExponential(2) : "—"}
          </div>
          <div className="text-[10px] text-[#A3A3A3] font-medium mt-0.5">
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
                <label className="text-[10px] text-[#A3A3A3] uppercase font-semibold block mb-1">
                  Training Task
                </label>
                <select
                  value={form.selectedTask}
                  onChange={(e) => handleSelectTask(e.target.value)}
                  disabled={isRunning || isPaused}
                  className="input font-semibold"
                >
                  <optgroup label="Language Modeling">
                    <option value="causal_lm">Causal LM (Autoregressive Pretraining)</option>
                    <option value="instruction_sft">Instruction Fine-Tuning (SFT)</option>
                    <option value="masked_lm">Masked LM (Bidirectional)</option>
                  </optgroup>
                  <optgroup label="Post-Training Alignment &amp; Reasoning">
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
              <label className="text-[10px] text-[#A3A3A3] uppercase font-semibold block mb-1">
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
            <div className="flex items-center justify-between pb-1.5 border-b border-[#262626]">
              <h3 className="panel-title">
                <Sparkles className="w-3.5 h-3.5 text-[#A3A3A3]" /> Optimizer &amp; Schedule
              </h3>
              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  onClick={() => handleSelectTask("causal_lm")}
                  disabled={isRunning || isPaused}
                  className={cn(
                    "px-2 py-0.5 rounded text-[10px] font-semibold border transition-colors flex items-center gap-1",
                    form.selectedTask === "causal_lm"
                      ? "bg-[#262626] text-[#4ade80] border-[#4ade80]/40"
                      : "bg-[#1B1B1B] text-[#A3A3A3] border-[#333333] hover:text-[#F2F2F2]"
                  )}
                  title="Optimal Pretrain: AdamW 3e-3, Muon 0.02, Min 1e-4, 3% Warmup, 82% Stable, 15% Decay"
                >
                  <Sparkles className="w-2.5 h-2.5 text-[#4ade80]" />
                  <span>⚡ Pretrain Preset</span>
                </button>
                <button
                  type="button"
                  onClick={() => handleSelectTask("instruction_sft")}
                  disabled={isRunning || isPaused}
                  className={cn(
                    "px-2 py-0.5 rounded text-[10px] font-semibold border transition-colors flex items-center gap-1",
                    form.selectedTask === "instruction_sft"
                      ? "bg-[#262626] text-[#38bdf8] border-[#38bdf8]/40"
                      : "bg-[#1B1B1B] text-[#A3A3A3] border-[#333333] hover:text-[#F2F2F2]"
                  )}
                  title="Optimal SFT: AdamW 3e-4, Muon 0.005, Min 1e-5, 5% Warmup, 45% Stable, 50% Decay"
                >
                  <Sliders className="w-2.5 h-2.5 text-[#38bdf8]" />
                  <span>⚡ SFT Preset</span>
                </button>
              </div>
            </div>

            {/* Duration Mode Selector */}
            <div className="p-2.5 rounded bg-[#1B1B1B] border border-[#262626] space-y-2">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
                <div>
                  <span className="text-xs text-[#F2F2F2] font-semibold uppercase">
                    Duration Mode
                  </span>
                  <span className="text-[10px] text-[#A3A3A3] block mt-0.5">
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
                        : "text-[#A3A3A3] hover:text-[#F2F2F2]"
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
                        : "text-[#A3A3A3] hover:text-[#F2F2F2]"
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
                      <label className="text-[10px] text-[#D4D4D4] font-semibold uppercase">
                        Epochs:
                      </label>
                      <input
                        type="number"
                        min="1"
                        max="100"
                        value={form.targetEpochs}
                        onChange={(e) => updateForm("targetEpochs", Math.max(1, parseInt(e.target.value) || 1))}
                        disabled={isRunning || isPaused}
                        className="input w-16 text-center font-mono-num font-semibold"
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
                                : "bg-[#1B1B1B] text-[#A3A3A3] border-[#262626]"
                            )}
                          >
                            {preset}x
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="flex items-center gap-1 text-[11px] text-[#A3A3A3]">
                      <Calculator className="w-3 h-3 text-[#A3A3A3]" />
                      <span>Tokens/Step: <strong className="text-[#F2F2F2]">{tokensPerStep.toLocaleString()}</strong></span>
                    </div>
                  </div>

                  <div className="text-[10px] text-[#A3A3A3] pt-1 border-t border-[#262626] flex justify-between">
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
                      <label className="text-[10px] text-[#D4D4D4] font-semibold uppercase">
                        Total Steps:
                      </label>
                      <input
                        type="number"
                        min="100"
                        step="500"
                        value={form.totalSteps}
                        onChange={(e) => updateForm("totalSteps", Math.max(10, parseInt(e.target.value) || 1000))}
                        disabled={isRunning || isPaused}
                        className="input w-24 text-center font-mono-num font-semibold"
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
                                : "bg-[#1B1B1B] text-[#A3A3A3] border-[#262626]"
                            )}
                          >
                            {(preset / 1000)}k
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="text-[10px] text-[#A3A3A3]">
                      Tokens: <strong className="text-[#F2F2F2]">{((form.totalSteps * tokensPerStep) / 1_000_000).toFixed(2)}M</strong>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
              <div>
                <label className="text-[10px] text-[#A3A3A3] uppercase font-semibold block mb-1">
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
                <label className="text-[10px] text-[#A3A3A3] uppercase font-semibold block mb-1">
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
                <label className="text-[10px] text-[#A3A3A3] uppercase font-semibold block mb-1">
                  AdamW LR
                </label>
                <input
                  type="number"
                  step="any"
                  value={form.peakLr}
                  onChange={(e) => updateForm("peakLr", isNaN(parseFloat(e.target.value)) ? 0 : parseFloat(e.target.value))}
                  disabled={isRunning || isPaused}
                  className="input font-mono-num font-semibold"
                />
              </div>

              <div>
                <label className="text-[10px] text-[#A3A3A3] uppercase font-semibold block mb-1">
                  Muon LR
                </label>
                <input
                  type="number"
                  step="any"
                  value={form.muonLr}
                  onChange={(e) => updateForm("muonLr", isNaN(parseFloat(e.target.value)) ? 0 : parseFloat(e.target.value))}
                  disabled={isRunning || isPaused}
                  className="input font-mono-num font-semibold"
                />
              </div>
            </div>

            {/* Percentage-based Warmup, Stable & Decay Schedule */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 pt-1">
              {/* Warmup % */}
              <div className="p-2.5 rounded bg-[#1B1B1B] border border-[#262626] space-y-1.5">
                <div className="flex items-center justify-between">
                  <label className="text-[10px] text-[#A3A3A3] uppercase font-semibold">
                    Warmup Ratio
                  </label>
                  <span className="text-[10px] text-[#D4D4D4] font-mono-num font-semibold">
                    ~{calcWarmupSteps.toLocaleString()} steps
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <input
                    type="number"
                    min="0"
                    max="30"
                    step="1"
                    value={form.warmupPct}
                    onChange={(e) => updateForm("warmupPct", Math.max(0, Math.min(30, parseInt(e.target.value) || 0)))}
                    disabled={isRunning || isPaused}
                    className="input w-20 text-center font-mono-num font-semibold"
                  />
                  <span className="text-xs text-[#A3A3A3] font-medium">%</span>
                  <div className="flex items-center gap-1 ml-auto">
                    {[2, 3, 5].map((pct) => (
                      <button
                        key={pct}
                        type="button"
                        onClick={() => updateForm("warmupPct", pct)}
                        disabled={isRunning || isPaused}
                        className={cn(
                          "px-1.5 py-0.5 rounded text-[10px] border transition-colors",
                          form.warmupPct === pct
                            ? "bg-[#262626] text-[#F2F2F2] border-[#D4D4D4]"
                            : "bg-[#1B1B1B] text-[#737373] border-[#262626]"
                        )}
                      >
                        {pct}%
                      </button>
                    ))}
                  </div>
                </div>
                <span className="text-[9px] text-[#737373] block">
                  Pretrain: 3% | SFT: 5%
                </span>
              </div>

              {/* Stable % */}
              <div className="p-2.5 rounded bg-[#1B1B1B] border border-[#262626] space-y-1.5">
                <div className="flex items-center justify-between">
                  <label className="text-[10px] text-[#A3A3A3] uppercase font-semibold">
                    Stable Ratio (Plateau)
                  </label>
                  <span className="text-[10px] text-[#D4D4D4] font-mono-num font-semibold">
                    ~{calcStableSteps.toLocaleString()} steps
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <input
                    type="number"
                    min="10"
                    max="95"
                    step="1"
                    value={form.stablePct}
                    onChange={(e) => updateForm("stablePct", Math.max(10, Math.min(95, parseInt(e.target.value) || 10)))}
                    disabled={isRunning || isPaused}
                    className="input w-20 text-center font-mono-num font-semibold"
                  />
                  <span className="text-xs text-[#A3A3A3] font-medium">%</span>
                  <div className="flex items-center gap-1 ml-auto">
                    {[45, 75, 82].map((pct) => (
                      <button
                        key={pct}
                        type="button"
                        onClick={() => updateForm("stablePct", pct)}
                        disabled={isRunning || isPaused}
                        className={cn(
                          "px-1.5 py-0.5 rounded text-[10px] border transition-colors",
                          form.stablePct === pct
                            ? "bg-[#262626] text-[#F2F2F2] border-[#D4D4D4]"
                            : "bg-[#1B1B1B] text-[#737373] border-[#262626]"
                        )}
                      >
                        {pct}%
                      </button>
                    ))}
                  </div>
                </div>
                <span className="text-[9px] text-[#737373] block">
                  Pretrain: 82% | SFT: 45%
                </span>
              </div>

              {/* Decay (Auto-computed) & Min LR Floor */}
              <div className="p-2.5 rounded bg-[#1B1B1B] border border-[#262626] space-y-1.5">
                <div className="flex items-center justify-between">
                  <label className="text-[10px] text-[#A3A3A3] uppercase font-semibold">
                    Decay Anneal (~{calcDecaySteps.toLocaleString()} st.)
                  </label>
                  <span className="text-[10px] text-[#4ade80] font-mono-num font-semibold">
                    {decayPct}% remaining
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="flex-1">
                    <label className="text-[9px] text-[#737373] uppercase block mb-0.5">
                      Min LR Floor
                    </label>
                    <input
                      type="number"
                      step="any"
                      value={form.minLr}
                      onChange={(e) => updateForm("minLr", isNaN(parseFloat(e.target.value)) ? 0 : parseFloat(e.target.value))}
                      disabled={isRunning || isPaused}
                      className="input text-xs font-mono-num font-semibold"
                    />
                  </div>
                  <div className="text-right">
                    <span className="text-[9px] text-[#737373] block mb-0.5">Style</span>
                    <span className="text-[10px] text-[#D4D4D4] font-medium">Cosine</span>
                  </div>
                </div>
                <span className="text-[9px] text-[#737373] block">
                  Pretrain: 15% (1e-4) | SFT: 50% (1e-5)
                </span>
              </div>
            </div>
          </div>

          {/* Section 3: Batch Geometry & Hardware */}
          <div className="panel p-3.5 bg-[#202020] border-[#333333] space-y-2.5">
            <div className="flex justify-between items-center pb-1.5 border-b border-[#262626]">
              <h3 className="panel-title">
                <Sliders className="w-3.5 h-3.5 text-[#A3A3A3]" /> Batch Geometry &amp; Hardware
              </h3>
              <span className="text-[10px] text-[#A3A3A3] font-medium">
                Effective: {effectiveBatchSize} seqs ({tokensPerStep.toLocaleString()} tok/step)
              </span>
            </div>

            {/* Compute Target Selection */}
            <div className="p-2.5 rounded bg-[#1B1B1B] border border-[#262626] space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <span className="text-xs text-[#F2F2F2] font-semibold uppercase">
                    Compute Target &amp; Device Topology
                  </span>
                  <span className="text-[10px] text-[#A3A3A3] block mt-0.5">
                    Select local GPU, CPU fallback, multi-GPU DataParallel, or remote network cluster (PyTorch DDP).
                  </span>
                </div>
                {computeTopology?.cluster_detected && (
                  <span className="px-2 py-0.5 rounded text-[10px] bg-[#14532d] text-[#86efac] border border-[#22c55e]/30 font-semibold flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#4ade80] animate-pulse"></span>
                    Cluster Detected ({computeTopology.cluster_nodes?.length || 1} Nodes)
                  </span>
                )}
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                <div>
                  <label className="text-[10px] text-[#A3A3A3] uppercase font-medium block mb-1">
                    Execution Device
                  </label>
                  <select
                    value={form.deviceMode}
                    onChange={(e) => {
                      const mode = e.target.value;
                      updateForm("deviceMode", mode);
                      if (mode === "multi_gpu") {
                        updateForm("useMultiGpu", true);
                      } else {
                        updateForm("useMultiGpu", false);
                      }
                    }}
                    disabled={isRunning || isPaused}
                    className="input font-mono text-xs"
                  >
                    <option value="auto">Auto (Detect Fastest Accelerator)</option>
                    {computeTopology?.devices && computeTopology.devices.length > 0 ? (
                      computeTopology.devices.map((d: any) => (
                        <option key={d.index} value={`cuda:${d.index}`}>
                          GPU {d.index}: {d.name} ({Math.round(d.total_vram_mb / 1024)}GB VRAM)
                        </option>
                      ))
                    ) : (
                      <option value="cuda:0">GPU 0 (CUDA Default)</option>
                    )}
                    <option value="cpu">CPU (Pure CPU Fallback)</option>
                    <option value="multi_gpu">
                      Multi-GPU Parallel ({computeTopology?.gpu_count || 2} GPUs)
                    </option>
                    <option value="cluster">
                      Remote Node Cluster (PyTorch DDP)
                    </option>
                  </select>
                </div>

                <div className="flex flex-col justify-center text-[10px] text-[#A3A3A3]">
                  <span className="font-semibold text-[#D4D4D4]">
                    {form.deviceMode === "cluster"
                      ? "Distributed Multi-Node PyTorch DDP"
                      : form.deviceMode === "multi_gpu"
                      ? `Local DataParallel across ${computeTopology?.gpu_count || 2} GPUs`
                      : form.deviceMode === "cpu"
                      ? "System RAM execution (slow, debugging only)"
                      : `Dedicated Hardware: ${computeTopology?.devices?.[0]?.name || "NVIDIA GPU"}`}
                  </span>
                  <span className="text-[9px] text-[#737373]">
                    {form.deviceMode === "cluster"
                      ? "Synchronizes gradients across nodes via TCP/IP sockets"
                      : "Zero-paging memory protection active"}
                  </span>
                </div>
              </div>

              {/* Cluster Parameters when "cluster" is selected */}
              {form.deviceMode === "cluster" && (
                <div className="p-2.5 rounded bg-[#202020] border border-[#333333] space-y-2 mt-2">
                  <div className="flex items-center justify-between border-b border-[#262626] pb-1">
                    <span className="text-[10px] font-semibold text-[#F2F2F2] uppercase">
                      Cluster Network Configuration (PyTorch DDP)
                    </span>
                    <span className="text-[9px] text-[#A3A3A3] font-mono">
                      TCP/IP Socket Sync
                    </span>
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                    <div>
                      <label className="text-[10px] text-[#A3A3A3] uppercase block mb-1">
                        Master Node IP
                      </label>
                      <input
                        type="text"
                        value={form.clusterMasterAddr}
                        onChange={(e) => updateForm("clusterMasterAddr", e.target.value)}
                        disabled={isRunning || isPaused}
                        placeholder="127.0.0.1 or 192.168.1.100"
                        className="input text-xs font-mono"
                      />
                    </div>

                    <div>
                      <label className="text-[10px] text-[#A3A3A3] uppercase block mb-1">
                        Master Port
                      </label>
                      <input
                        type="number"
                        value={form.clusterMasterPort}
                        onChange={(e) => updateForm("clusterMasterPort", parseInt(e.target.value) || 29500)}
                        disabled={isRunning || isPaused}
                        className="input text-xs font-mono text-center"
                      />
                    </div>

                    <div>
                      <label className="text-[10px] text-[#A3A3A3] uppercase block mb-1">
                        World Size (Nodes)
                      </label>
                      <input
                        type="number"
                        min="1"
                        value={form.clusterWorldSize}
                        onChange={(e) => updateForm("clusterWorldSize", Math.max(1, parseInt(e.target.value) || 1))}
                        disabled={isRunning || isPaused}
                        className="input text-xs font-mono text-center"
                      />
                    </div>

                    <div>
                      <label className="text-[10px] text-[#A3A3A3] uppercase block mb-1">
                        Node Rank
                      </label>
                      <input
                        type="number"
                        min="0"
                        value={form.clusterRank}
                        onChange={(e) => updateForm("clusterRank", Math.max(0, parseInt(e.target.value) || 0))}
                        disabled={isRunning || isPaused}
                        className="input text-xs font-mono text-center"
                      />
                    </div>
                  </div>
                  <div className="text-[10px] text-[#737373]">
                    Rank 0 is the coordinator/master node. Worker nodes should set Rank &ge; 1 pointing to the Master IP.
                  </div>
                </div>
              )}
            </div>

            {/* PEFT / LoRA Controls */}
            <div className="p-2.5 rounded bg-[#1B1B1B] border border-[#262626] space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <span className="text-xs text-[#F2F2F2] font-semibold uppercase">
                    PEFT Adaptation (LoRA)
                  </span>
                  <span className="text-[10px] text-[#A3A3A3] block mt-0.5">
                    LoRA low-rank adapter injection (&gt;80% VRAM saving).
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
                <label className="text-[10px] text-[#A3A3A3] uppercase font-semibold block mb-1">
                  Micro-Batch Size
                </label>
                <input
                  type="number"
                  value={form.microBatchSize}
                  onChange={(e) => updateForm("microBatchSize", parseInt(e.target.value) || 4)}
                  disabled={isRunning || isPaused}
                  className="input font-mono-num font-semibold"
                />
              </div>

              <div>
                <label className="text-[10px] text-[#A3A3A3] uppercase font-semibold block mb-1">
                  Grad Accum Steps
                </label>
                <input
                  type="number"
                  value={form.gradAccumSteps}
                  onChange={(e) => updateForm("gradAccumSteps", parseInt(e.target.value) || 8)}
                  disabled={isRunning || isPaused}
                  className="input font-mono-num font-semibold"
                />
              </div>

              <div>
                <label className="text-[10px] text-[#A3A3A3] uppercase font-semibold block mb-1">
                  Seq Length
                </label>
                <input
                  type="number"
                  value={form.seqLen}
                  onChange={(e) => updateForm("seqLen", parseInt(e.target.value) || 256)}
                  disabled={isRunning || isPaused}
                  className="input font-mono-num font-semibold"
                />
              </div>

              <div>
                <label className="text-[10px] text-[#A3A3A3] uppercase font-semibold block mb-1">
                  Grad Clip Norm
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={form.gradClip}
                  onChange={(e) => updateForm("gradClip", parseFloat(e.target.value) || 1.0)}
                  disabled={isRunning || isPaused}
                  className="input font-mono-num font-semibold"
                />
              </div>
            </div>

            {/* Intervals & Early Stopping */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5 pt-1">
              <div>
                <label className="text-[10px] text-[#A3A3A3] uppercase font-semibold block mb-1">
                  Checkpoint Every
                </label>
                <input
                  type="number"
                  value={form.chkInterval}
                  onChange={(e) => updateForm("chkInterval", parseInt(e.target.value) || 500)}
                  disabled={isRunning || isPaused}
                  className="input font-mono-num font-semibold"
                />
              </div>

              <div>
                <label className="text-[10px] text-[#A3A3A3] uppercase font-semibold block mb-1">
                  Validation Every
                </label>
                <input
                  type="number"
                  value={form.valInterval}
                  onChange={(e) => updateForm("valInterval", parseInt(e.target.value) || 250)}
                  disabled={isRunning || isPaused}
                  className="input font-mono-num font-semibold"
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
                  <span className="text-[10px] text-[#A3A3A3]">
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
            labels={(schedulerCurve?.steps || []).filter((_, i) => i % 10 === 0)}
            datasets={[
              {
                label: "Planned LR",
                data: (schedulerCurve?.lrs || []).filter((_, i) => i % 10 === 0),
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
