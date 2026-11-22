"use client";

import React, { useState, useEffect, useCallback } from "react";
import { api } from "@/services/api";
import { CheckpointItem, BenchmarkResult } from "@/types/api";
import { DataTable, Column } from "@/components/ui/DataTable";
import {
  FlaskConical,
  Play,
  Columns,
  Sparkles,
  Cpu,
  Gauge,
  Clock,
  Bookmark,
  Award,
  RotateCcw,
  RefreshCw,
  Sliders,
  AlertCircle,
  X,
  CheckCircle2,
  Package,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface SamplingConfig {
  maxTokens: number;
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  repPenalty: number;
  noRepeatNgram: number;
  presencePenalty: number;
  frequencyPenalty: number;
  seed: string;
  stopSequences: string;
}

const DEFAULT_CONFIG: SamplingConfig = {
  maxTokens: 128,
  temperature: 0.7,
  topP: 0.9,
  topK: 50,
  minP: 0.0,
  repPenalty: 1.15,
  noRepeatNgram: 3,
  presencePenalty: 0.0,
  frequencyPenalty: 0.0,
  seed: "",
  stopSequences: "",
};

export const EvaluationView: React.FC = () => {
  const [checkpoints, setCheckpoints] = useState<CheckpointItem[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>("");
  const [benchmarks, setBenchmarks] = useState<BenchmarkResult[]>([]);
  const [isLoadingCheckpoints, setIsLoadingCheckpoints] = useState<boolean>(false);

  // Prompt state
  const [prompt, setPrompt] = useState<string>(
    "Inteligența artificială modernă funcționează prin"
  );
  const [isChatMode, setIsChatMode] = useState<boolean>(true);

  // Granular Sampling Parameters (Fully configurable, NO PRESETS)
  const [config, setConfig] = useState<SamplingConfig>(DEFAULT_CONFIG);

  // Single inference result & error feedback
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [generationResult, setGenerationResult] = useState<any | null>(null);
  const [evalError, setEvalError] = useState<string | null>(null);

  // Side-by-side comparison
  const [isCompareMode, setIsCompareMode] = useState<boolean>(false);
  const [compareCheckpoints, setCompareCheckpoints] = useState<string[]>([]);
  const [compareResults, setCompareResults] = useState<any[]>([]);
  const [isComparing, setIsComparing] = useState<boolean>(false);

  // Benchmark suite
  const [isRunningBenchmarks, setIsRunningBenchmarks] = useState<boolean>(false);

  const fetchCheckpoints = useCallback(async () => {
    setIsLoadingCheckpoints(true);
    try {
      const list = await api.listCheckpoints();
      setCheckpoints(list);
      if (list.length > 0 && !selectedCheckpoint) {
        setSelectedCheckpoint(list[0].path || list[0].filename);
      }
      if (list.length >= 2 && compareCheckpoints.length === 0) {
        setCompareCheckpoints([list[0].path || list[0].filename, list[1].path || list[1].filename]);
      }
    } catch (err: any) {
      console.error("Failed to load checkpoints:", err);
    } finally {
      setIsLoadingCheckpoints(false);
    }
  }, [selectedCheckpoint, compareCheckpoints.length]);

  useEffect(() => {
    fetchCheckpoints();
    api.listBenchmarkResults().then(setBenchmarks).catch(console.error);
  }, [fetchCheckpoints]);

  const updateParam = <K extends keyof SamplingConfig>(key: K, value: SamplingConfig[K]) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  const handleResetDefaults = () => {
    setConfig(DEFAULT_CONFIG);
  };

  const parseStopSequences = (raw: string): string[] | undefined => {
    if (!raw.trim()) return undefined;
    return raw
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
  };

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    setIsGenerating(true);
    setGenerationResult(null);
    setEvalError(null);

    const parsedSeed = config.seed.trim() !== "" ? parseInt(config.seed, 10) : undefined;
    const stopSeqs = parseStopSequences(config.stopSequences);

    try {
      const res = await api.generateEvaluation({
        prompt,
        max_new_tokens: config.maxTokens,
        temperature: config.temperature,
        top_p: config.topP,
        top_k: config.topK,
        min_p: config.minP,
        repetition_penalty: config.repPenalty,
        no_repeat_ngram_size: config.noRepeatNgram,
        presence_penalty: config.presencePenalty,
        frequency_penalty: config.frequencyPenalty,
        seed: isNaN(parsedSeed as number) ? undefined : parsedSeed,
        stop_sequences: stopSeqs,
        checkpoint: selectedCheckpoint || undefined,
        chat_mode: isChatMode,
      });

      if ((res as any).error) {
        setEvalError((res as any).error);
      } else {
        setGenerationResult(res);
      }
    } catch (e: any) {
      setEvalError(e.message || "Unknown inference error occurred.");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleRunComparison = async () => {
    if (!prompt.trim() || compareCheckpoints.length === 0) return;
    setIsComparing(true);
    setCompareResults([]);
    setEvalError(null);

    const parsedSeed = config.seed.trim() !== "" ? parseInt(config.seed, 10) : undefined;
    const stopSeqs = parseStopSequences(config.stopSequences);

    try {
      const res = await api.compareEvaluation({
        prompt,
        checkpoint_paths: compareCheckpoints,
        max_new_tokens: config.maxTokens,
        temperature: config.temperature,
        top_p: config.topP,
        top_k: config.topK,
        min_p: config.minP,
        repetition_penalty: config.repPenalty,
        no_repeat_ngram_size: config.noRepeatNgram,
        presence_penalty: config.presencePenalty,
        frequency_penalty: config.frequencyPenalty,
        seed: isNaN(parsedSeed as number) ? undefined : parsedSeed,
        stop_sequences: stopSeqs,
        chat_mode: isChatMode,
      });

      if (res.results && Array.isArray(res.results)) {
        setCompareResults(res.results);
      } else {
        setCompareResults([]);
      }
    } catch (e: any) {
      setEvalError(`Comparison failed: ${e.message}`);
    } finally {
      setIsComparing(false);
    }
  };

  const handleRunBenchmarkSuite = async () => {
    setIsRunningBenchmarks(true);
    try {
      await api.runBenchmarkSuite();
      const res = await api.listBenchmarkResults();
      setBenchmarks(res);
    } catch (e: any) {
      alert(`Benchmark error: ${e.message}`);
    } finally {
      setIsRunningBenchmarks(false);
    }
  };

  const benchmarkColumns: Column<BenchmarkResult>[] = [
    {
      header: "TASK / SUITE",
      accessorKey: "task_name",
      cell: (b) => (
        <span className="font-bold text-[#ECF3FC] font-mono">{b.task_name}</span>
      ),
    },
    {
      header: "CHECKPOINT",
      accessorKey: "checkpoint_id",
      cell: (b) => (
        <span className="text-[#91A0B4] font-mono">{b.checkpoint_id || "Active Weights"}</span>
      ),
    },
    {
      header: "METRIC",
      accessorKey: "metric_name",
      cell: (b) => (
        <span className="text-[#9B8CFF] font-mono uppercase text-[10px]">{b.metric_name || "Score"}</span>
      ),
    },
    {
      header: "SCORE / RESULT",
      accessorKey: "score",
      sortable: true,
      cell: (b) => (
        <span className="font-mono-num font-bold text-[#45CFFF]">
          {typeof b.score === "number" ? b.score.toFixed(4) : "—"}
        </span>
      ),
    },
    {
      header: "TIMESTAMP",
      accessorKey: "created_at",
      cell: (b) => (
        <span className="text-[#5E7188] text-[10px] font-mono">
          {b.created_at ? new Date(b.created_at).toLocaleString() : "—"}
        </span>
      ),
    },
  ];

  return (
    <div className="space-y-4 font-mono text-xs max-w-[1600px] mx-auto animate-in fade-in duration-200">
      {/* Top Header Mode Toggle */}
      <div className="glass-panel p-3.5 rounded-lg flex flex-wrap items-center justify-between gap-3 border-l-2 border-l-[#45CFFF] relative overflow-hidden">
        <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-[#45CFFF]/30 via-transparent to-transparent pointer-events-none" />

        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-[#131C29] border border-[#45CFFF]/40 flex items-center justify-center shrink-0 shadow-[0_0_12px_rgba(69,207,255,0.2)]">
            <FlaskConical className="w-4 h-4 text-[#45CFFF]" />
          </div>
          <div>
            <h2 className="text-xs font-bold text-[#ECF3FC] uppercase tracking-wider">
              EVALUATION LAB & INFERENCE PLAYGROUND
            </h2>
            <p className="text-[10px] text-[#91A0B4] mt-0.5">
              Granular hyperparameter tuning, multi-model side-by-side matrices, and automated benchmarks.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsCompareMode(false)}
            className={cn(
              "px-3 py-1.5 rounded-md text-xs font-mono font-semibold transition-all",
              !isCompareMode
                ? "bg-[#45CFFF] text-[#090D14] shadow-sm font-bold"
                : "bg-[#131C29] border border-[#273548] text-[#91A0B4] hover:text-[#ECF3FC]"
            )}
          >
            SINGLE INFERENCE
          </button>
          <button
            onClick={() => setIsCompareMode(true)}
            className={cn(
              "px-3 py-1.5 rounded-md text-xs font-mono font-semibold transition-all flex items-center gap-1.5",
              isCompareMode
                ? "bg-[#45CFFF] text-[#090D14] shadow-sm font-bold"
                : "bg-[#131C29] border border-[#273548] text-[#91A0B4] hover:text-[#ECF3FC]"
            )}
          >
            <Columns className="w-3.5 h-3.5" />
            CHECKPOINT MATRIX
          </button>
        </div>
      </div>

      {/* Main Grid: Controls Column (1 col) + Output & Prompt (2 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Controls Column */}
        <div className="space-y-3">
          {/* Checkpoint Selection */}
          <div className="glass-panel p-3.5 rounded-lg space-y-3 relative overflow-hidden">
            <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#45CFFF]/15 to-transparent pointer-events-none" />

            <div className="flex items-center justify-between pb-1.5 border-b border-[#273548]">
              <h4 className="text-xs font-bold text-[#ECF3FC] uppercase tracking-wider flex items-center gap-1.5">
                <Bookmark className="w-3.5 h-3.5 text-[#45CFFF]" /> MODEL CHECKPOINT
              </h4>
              <button
                type="button"
                onClick={fetchCheckpoints}
                disabled={isLoadingCheckpoints}
                className="text-[10px] text-[#91A0B4] hover:text-[#45CFFF] transition-colors flex items-center gap-1"
                title="Refresh checkpoints and saved models from disk"
              >
                <RefreshCw className={cn("w-3 h-3", isLoadingCheckpoints && "animate-spin")} />
                Refresh
              </button>
            </div>

            {!isCompareMode ? (
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-[10px] text-[#91A0B4]">
                  <span>WEIGHT SOURCE</span>
                  {selectedCheckpoint && (
                    <button
                      type="button"
                      onClick={() => setSelectedCheckpoint("")}
                      className="text-[#80DEFF] hover:underline"
                    >
                      Use In-Memory
                    </button>
                  )}
                </div>
                <select
                  value={selectedCheckpoint}
                  onChange={(e) => setSelectedCheckpoint(e.target.value)}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2.5 py-1.5 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF]/50 text-xs"
                >
                  <option value="">⚡ Current In-Memory Weights (Active Model)</option>
                  {checkpoints.map((c) => {
                    const valStr = c.val_loss !== null && c.val_loss !== undefined ? ` | Val: ${Number(c.val_loss).toFixed(4)}` : "";
                    const prefix = c.is_saved_model ? "📦 Saved Model:" : "💾 Checkpoint:";
                    const displayName = c.model_name || c.filename;
                    return (
                      <option key={c.id || c.path || c.filename} value={c.path || c.filename}>
                        {prefix} {displayName} (Step {c.step}{valStr})
                      </option>
                    );
                  })}
                </select>
              </div>
            ) : (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-[10px] text-[#91A0B4]">
                  <span>MODELS TO COMPARE ({compareCheckpoints.length} selected)</span>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setCompareCheckpoints(checkpoints.map((c) => c.path || c.filename))}
                      className="text-[#80DEFF] hover:underline"
                    >
                      All
                    </button>
                    <span>·</span>
                    <button
                      type="button"
                      onClick={() => setCompareCheckpoints([])}
                      className="text-[#80DEFF] hover:underline"
                    >
                      Clear
                    </button>
                  </div>
                </div>
                <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                  {checkpoints.length === 0 ? (
                    <div className="text-[11px] text-[#5E7188] italic p-2">
                      No saved models or checkpoints found.
                    </div>
                  ) : (
                    checkpoints.map((c) => {
                      const path = c.path || c.filename;
                      const isChecked = compareCheckpoints.includes(path);
                      return (
                        <label
                          key={c.id || path}
                          className={cn(
                            "flex items-center gap-2 p-1.5 rounded-md border text-[11px] cursor-pointer transition-all",
                            isChecked
                              ? "bg-[#131C29] border-[#45CFFF]/40 text-[#ECF3FC]"
                              : "bg-[#0E1420] border-[#273548] text-[#91A0B4] hover:text-[#ECF3FC]"
                          )}
                        >
                          <input
                            type="checkbox"
                            checked={isChecked}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setCompareCheckpoints([...compareCheckpoints, path]);
                              } else {
                                setCompareCheckpoints(compareCheckpoints.filter((p) => p !== path));
                              }
                            }}
                            className="accent-[#45CFFF] rounded"
                          />
                          <div className="flex items-center gap-1.5 truncate flex-1 min-w-0">
                            {c.is_saved_model ? (
                              <Package className="w-3 h-3 text-[#35D49A] shrink-0" />
                            ) : (
                              <Bookmark className="w-3 h-3 text-[#45CFFF] shrink-0" />
                            )}
                            <span className="truncate">{c.model_name || c.filename}</span>
                          </div>
                          <span className="text-[9px] text-[#5E7188] shrink-0 font-mono-num">
                            Step {c.step}
                          </span>
                        </label>
                      );
                    })
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Granular Sampling Hyperparameters (NO PRESETS - Fully Configurable) */}
          <div className="glass-panel p-3.5 rounded-lg space-y-3 relative overflow-hidden">
            <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#45CFFF]/15 to-transparent pointer-events-none" />

            <div className="flex items-center justify-between pb-1.5 border-b border-[#273548]">
              <h4 className="text-xs font-bold text-[#ECF3FC] uppercase tracking-wider flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-[#45CFFF]" /> SAMPLING CONFIGURATION
              </h4>
              <button
                type="button"
                onClick={handleResetDefaults}
                className="text-[10px] text-[#91A0B4] hover:text-[#45CFFF] transition-colors flex items-center gap-1"
                title="Reset all generation parameters to recommended defaults"
              >
                <RotateCcw className="w-3 h-3" />
                Defaults
              </button>
            </div>

            {/* Max New Tokens (Slider + Direct Input) */}
            <div>
              <div className="flex justify-between items-center text-[10px] text-[#91A0B4] mb-1">
                <span className="font-semibold uppercase">MAX NEW TOKENS</span>
                <input
                  type="number"
                  min="16"
                  max="2048"
                  step="16"
                  value={config.maxTokens}
                  onChange={(e) => updateParam("maxTokens", Math.max(16, parseInt(e.target.value) || 16))}
                  className="w-16 bg-[#0E1420] border border-[#273548] rounded px-1.5 py-0.5 text-right text-[#45CFFF] font-bold font-mono-num text-[11px] focus:outline-none focus:border-[#45CFFF]"
                />
              </div>
              <input
                type="range"
                min="16"
                max="1024"
                step="16"
                value={config.maxTokens}
                onChange={(e) => updateParam("maxTokens", parseInt(e.target.value))}
                className="w-full accent-[#45CFFF] bg-[#0E1420] h-1.5 rounded cursor-pointer"
              />
            </div>

            {/* Temperature (Slider + Direct Input, 0.0 = Greedy) */}
            <div>
              <div className="flex justify-between items-center text-[10px] text-[#91A0B4] mb-1">
                <span className="font-semibold uppercase">
                  TEMPERATURE <span className="text-[9px] text-[#5E7188]">{config.temperature === 0 ? "(GREEDY)" : ""}</span>
                </span>
                <input
                  type="number"
                  min="0"
                  max="2.0"
                  step="0.05"
                  value={config.temperature}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    updateParam("temperature", isNaN(v) ? 0 : Math.max(0, Math.min(2.0, v)));
                  }}
                  className="w-16 bg-[#0E1420] border border-[#273548] rounded px-1.5 py-0.5 text-right text-[#45CFFF] font-bold font-mono-num text-[11px] focus:outline-none focus:border-[#45CFFF]"
                />
              </div>
              <input
                type="range"
                min="0.0"
                max="2.0"
                step="0.05"
                value={config.temperature}
                onChange={(e) => updateParam("temperature", parseFloat(e.target.value))}
                className="w-full accent-[#45CFFF] bg-[#0E1420] h-1.5 rounded cursor-pointer"
              />
            </div>

            {/* Top-P (Nucleus) & Top-K Dual Numeric Inputs */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <div className="flex justify-between items-center text-[10px] text-[#91A0B4] mb-1">
                  <span className="font-semibold uppercase">TOP-P</span>
                  <span className="text-[#ECF3FC] font-mono-num">{config.topP.toFixed(2)}</span>
                </div>
                <input
                  type="number"
                  step="0.05"
                  min="0.05"
                  max="1.0"
                  value={config.topP}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    updateParam("topP", isNaN(v) ? 0.9 : Math.max(0.05, Math.min(1.0, v)));
                  }}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2 py-1 text-[#ECF3FC] text-xs font-mono-num focus:outline-none focus:border-[#45CFFF]"
                />
              </div>

              <div>
                <div className="flex justify-between items-center text-[10px] text-[#91A0B4] mb-1">
                  <span className="font-semibold uppercase">TOP-K</span>
                  <span className="text-[#ECF3FC] font-mono-num">{config.topK === 0 ? "Off" : config.topK}</span>
                </div>
                <input
                  type="number"
                  min="0"
                  max="200"
                  step="1"
                  value={config.topK}
                  onChange={(e) => {
                    const v = parseInt(e.target.value, 10);
                    updateParam("topK", isNaN(v) ? 0 : Math.max(0, v));
                  }}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2 py-1 text-[#ECF3FC] text-xs font-mono-num focus:outline-none focus:border-[#45CFFF]"
                />
              </div>
            </div>

            {/* Min-P Sampling & Repetition Penalty */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <div className="flex justify-between items-center text-[10px] text-[#91A0B4] mb-1">
                  <span className="font-semibold uppercase">MIN-P</span>
                  <span className="text-[#ECF3FC] font-mono-num">{config.minP === 0 ? "Off" : config.minP.toFixed(2)}</span>
                </div>
                <input
                  type="number"
                  step="0.01"
                  min="0.0"
                  max="1.0"
                  value={config.minP}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    updateParam("minP", isNaN(v) ? 0 : Math.max(0.0, Math.min(1.0, v)));
                  }}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2 py-1 text-[#ECF3FC] text-xs font-mono-num focus:outline-none focus:border-[#45CFFF]"
                />
              </div>

              <div>
                <div className="flex justify-between items-center text-[10px] text-[#91A0B4] mb-1">
                  <span className="font-semibold uppercase">REP PENALTY</span>
                  <span className="text-[#ECF3FC] font-mono-num">{config.repPenalty.toFixed(2)}</span>
                </div>
                <input
                  type="number"
                  step="0.05"
                  min="1.0"
                  max="3.0"
                  value={config.repPenalty}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    updateParam("repPenalty", isNaN(v) ? 1.0 : Math.max(1.0, v));
                  }}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2 py-1 text-[#ECF3FC] text-xs font-mono-num focus:outline-none focus:border-[#45CFFF]"
                />
              </div>
            </div>

            {/* No Repeat N-Gram (0 = disabled) & RNG Seed */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <div className="flex justify-between items-center text-[10px] text-[#91A0B4] mb-1">
                  <span className="font-semibold uppercase">NO REPEAT N-GRAM</span>
                  <span className="text-[#ECF3FC] font-mono-num">{config.noRepeatNgram === 0 ? "Off" : config.noRepeatNgram}</span>
                </div>
                <input
                  type="number"
                  min="0"
                  max="10"
                  step="1"
                  value={config.noRepeatNgram}
                  onChange={(e) => {
                    const raw = e.target.value;
                    if (raw === "") {
                      updateParam("noRepeatNgram", 0);
                      return;
                    }
                    const parsed = parseInt(raw, 10);
                    updateParam("noRepeatNgram", isNaN(parsed) ? 0 : Math.max(0, parsed));
                  }}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2 py-1 text-[#ECF3FC] text-xs font-mono-num focus:outline-none focus:border-[#45CFFF]"
                />
              </div>

              <div>
                <div className="flex justify-between items-center text-[10px] text-[#91A0B4] mb-1">
                  <span className="font-semibold uppercase">RNG SEED</span>
                  <span className="text-[#5E7188] text-[9px]">{config.seed ? `#${config.seed}` : "Random"}</span>
                </div>
                <input
                  type="number"
                  placeholder="e.g. 42"
                  value={config.seed}
                  onChange={(e) => updateParam("seed", e.target.value)}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2 py-1 text-[#ECF3FC] text-xs font-mono-num placeholder-[#5E7188] focus:outline-none focus:border-[#45CFFF]"
                />
              </div>
            </div>

            {/* Presence & Frequency Penalties */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                  PRESENCE PENALTY
                </label>
                <input
                  type="number"
                  step="0.05"
                  min="0.0"
                  max="2.0"
                  value={config.presencePenalty}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    updateParam("presencePenalty", isNaN(v) ? 0 : Math.max(0, v));
                  }}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2 py-1 text-[#ECF3FC] text-xs font-mono-num focus:outline-none focus:border-[#45CFFF]"
                />
              </div>

              <div>
                <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                  FREQUENCY PENALTY
                </label>
                <input
                  type="number"
                  step="0.05"
                  min="0.0"
                  max="2.0"
                  value={config.frequencyPenalty}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    updateParam("frequencyPenalty", isNaN(v) ? 0 : Math.max(0, v));
                  }}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2 py-1 text-[#ECF3FC] text-xs font-mono-num focus:outline-none focus:border-[#45CFFF]"
                />
              </div>
            </div>

            {/* Custom Stop Sequences */}
            <div>
              <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                CUSTOM STOP SEQUENCES
              </label>
              <input
                type="text"
                placeholder="<|end_turn|>, \n\n, </s>"
                value={config.stopSequences}
                onChange={(e) => updateParam("stopSequences", e.target.value)}
                className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2 py-1 text-[#ECF3FC] text-xs font-mono placeholder-[#5E7188] focus:outline-none focus:border-[#45CFFF]"
              />
            </div>
          </div>
        </div>

        {/* Right Output Area (2 cols) */}
        <div className="lg:col-span-2 space-y-3">
          {/* Prompt Editor */}
          <div className="glass-panel p-3.5 rounded-lg space-y-2.5 relative overflow-hidden">
            <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#45CFFF]/15 to-transparent pointer-events-none" />

            <div className="flex flex-wrap items-center justify-between gap-2 pb-1.5 border-b border-[#273548]">
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold text-[#ECF3FC] uppercase tracking-wider">
                  INPUT PROMPT & CONTEXT
                </span>
                <button
                  type="button"
                  onClick={() => setIsChatMode(!isChatMode)}
                  className={cn(
                    "px-2 py-0.5 rounded text-[10px] font-bold border transition-colors flex items-center gap-1",
                    isChatMode
                      ? "bg-[#45CFFF]/15 text-[#45CFFF] border-[#45CFFF]/30"
                      : "bg-[#0E1420] text-[#91A0B4] border-[#273548] hover:text-[#ECF3FC]"
                  )}
                  title="Wrap prompt with <|user|> ... <|assistant|> ChatML tags"
                >
                  <Sparkles className="w-2.5 h-2.5" />
                  {isChatMode ? "CHAT FORMAT" : "RAW COMPLETION"}
                </button>
              </div>

              <div className="flex flex-wrap gap-1 text-[10px]">
                <button
                  onClick={() => {
                    setPrompt("Salut, cine ești și ce poți face?");
                    setIsChatMode(true);
                  }}
                  className="px-2 py-0.5 rounded-md bg-[#0E1420] border border-[#273548] text-[#91A0B4] hover:text-[#45CFFF] hover:border-[#45CFFF]/40 transition-colors"
                >
                  Salut!
                </button>
                <button
                  onClick={() => {
                    setPrompt("Explică pe scurt ce este un model de limbaj neuronal și cum funcționează atenția.");
                    setIsChatMode(true);
                  }}
                  className="px-2 py-0.5 rounded-md bg-[#0E1420] border border-[#273548] text-[#91A0B4] hover:text-[#45CFFF] hover:border-[#45CFFF]/40 transition-colors"
                >
                  Instrucțiune
                </button>
                <button
                  onClick={() => {
                    setPrompt("Definiția rețelelor neuronale artificiale și a mecanismului de atenție:");
                    setIsChatMode(false);
                  }}
                  className="px-2 py-0.5 rounded-md bg-[#0E1420] border border-[#273548] text-[#91A0B4] hover:text-[#45CFFF] transition-colors"
                >
                  Tech
                </button>
                <button
                  onClick={() => {
                    setPrompt("Capitala României este București, iar istoria sa modernă începe cu");
                    setIsChatMode(false);
                  }}
                  className="px-2 py-0.5 rounded-md bg-[#0E1420] border border-[#273548] text-[#91A0B4] hover:text-[#45CFFF] transition-colors"
                >
                  Wiki
                </button>
              </div>
            </div>

            <textarea
              rows={3}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full bg-[#0E1420] border border-[#273548] rounded-md p-2.5 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF]/50"
              placeholder="Type your prompt here..."
            />

            <div className="flex justify-between items-center pt-1">
              <div className="text-[10px] text-[#5E7188]">
                {prompt.length} chars · ~{Math.round(prompt.length / 4)} tokens
              </div>
              <div>
                {!isCompareMode ? (
                  <button
                    onClick={handleGenerate}
                    disabled={isGenerating || !prompt.trim()}
                    className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-md bg-[#45CFFF] hover:bg-[#80DEFF] disabled:opacity-40 text-[#090D14] font-bold transition-all shadow-sm"
                  >
                    <Play className="w-3.5 h-3.5 fill-current" />
                    {isGenerating ? "GENERATING TOKENS..." : "EXECUTE INFERENCE"}
                  </button>
                ) : (
                  <button
                    onClick={handleRunComparison}
                    disabled={isComparing || compareCheckpoints.length === 0 || !prompt.trim()}
                    className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-md bg-[#45CFFF] hover:bg-[#80DEFF] disabled:opacity-40 text-[#090D14] font-bold transition-all shadow-sm"
                  >
                    <Columns className="w-3.5 h-3.5" />
                    {isComparing ? "COMPARING CHECKPOINTS..." : `COMPARE (${compareCheckpoints.length}) CHECKPOINTS`}
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Inline Error Alert */}
          {evalError && (
            <div className="p-3 rounded-lg bg-[#FF657A]/15 border border-[#FF657A]/40 text-[#FF657A] flex items-start justify-between gap-2 animate-in fade-in">
              <div className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                <div>
                  <strong className="block text-xs font-bold uppercase tracking-wider">Inference Error</strong>
                  <p className="text-[11px] text-[#FF657A]/90 mt-0.5 whitespace-pre-wrap">{evalError}</p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setEvalError(null)}
                className="text-[#FF657A] hover:opacity-80 p-1"
                title="Dismiss"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          )}

          {/* Single Generation Output */}
          {!isCompareMode && (
            <div className="glass-panel p-4 rounded-lg space-y-3 min-h-[220px] relative overflow-hidden">
              <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#45CFFF]/15 to-transparent pointer-events-none" />

              <div className="flex items-center justify-between pb-2 border-b border-[#273548]">
                <span className="text-xs font-bold text-[#ECF3FC] uppercase tracking-wider flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-[#45CFFF]" /> GENERATED COMPLETION
                </span>
                {generationResult && (
                  <div className="flex items-center gap-3 text-[10px] font-mono text-[#91A0B4]">
                    <span className="flex items-center gap-1">
                      <Gauge className="w-3 h-3 text-[#45CFFF]" />
                      <span className="text-[#ECF3FC] font-mono-num font-bold">
                        {Math.round(generationResult.tokens_per_sec || generationResult.tokens_per_second || 0)}
                      </span> tok/s
                    </span>
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3 text-[#9B8CFF]" />
                      <span className="text-[#ECF3FC] font-mono-num">
                        {generationResult.generation_time_sec?.toFixed(2) || "0.00"}s
                      </span>
                    </span>
                    <span className="flex items-center gap-1">
                      <Cpu className="w-3 h-3 text-[#5E7188]" />
                      {generationResult.device?.toUpperCase() || "ACCELERATOR"}
                    </span>
                  </div>
                )}
              </div>

              {generationResult ? (
                <div className="p-3 bg-[#090D14] border border-[#273548] rounded-md font-mono text-xs leading-relaxed text-[#ECF3FC] whitespace-pre-wrap select-text">
                  {isChatMode ? (
                    <div>
                      <div className="text-[10px] text-[#45CFFF] font-bold uppercase mb-1 flex items-center gap-1.5">
                        <CheckCircle2 className="w-3 h-3 text-[#35D49A]" />
                        <span>ASSISTANT OUTPUT:</span>
                      </div>
                      <div className="text-[#ECF3FC] pl-2 border-l-2 border-[#45CFFF]/50">
                        {(generationResult.completion || generationResult.generated_text) ? (
                          generationResult.completion || generationResult.generated_text
                        ) : (
                          <span className="text-[#5E7188] italic">
                            [Model emitted stop/EOS token immediately; no output tokens generated]
                          </span>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div>
                      <span className="text-[#45CFFF] font-bold">{prompt}</span>
                      <span> {generationResult.completion || generationResult.generated_text}</span>
                    </div>
                  )}
                </div>
              ) : (
                <div className="h-32 flex items-center justify-center text-[#5E7188] font-mono text-xs">
                  {isGenerating ? (
                    <div className="flex items-center gap-2 text-[#45CFFF]">
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      <span>Synthesizing tokens from transformer forward pass...</span>
                    </div>
                  ) : (
                    <span>Configure parameters and click Execute Inference to evaluate output quality.</span>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Side-by-Side Comparison Output */}
          {isCompareMode && (
            <div className="glass-panel p-4 rounded-lg space-y-3 min-h-[220px] relative overflow-hidden">
              <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#45CFFF]/15 to-transparent pointer-events-none" />

              <div className="flex items-center justify-between pb-2 border-b border-[#273548]">
                <span className="text-xs font-bold text-[#ECF3FC] uppercase tracking-wider flex items-center gap-1.5">
                  <Columns className="w-3.5 h-3.5 text-[#45CFFF]" />
                  CHECKPOINT COMPARISON MATRIX ({compareResults.length} / {compareCheckpoints.length} evaluated)
                </span>
                {compareResults.length > 0 && (
                  <button
                    type="button"
                    onClick={() => setCompareResults([])}
                    className="text-[10px] text-[#91A0B4] hover:text-[#ECF3FC]"
                  >
                    Clear Results
                  </button>
                )}
              </div>

              {compareResults.length === 0 ? (
                <div className="h-32 flex items-center justify-center text-[#5E7188] font-mono text-xs">
                  {isComparing ? (
                    <div className="flex items-center gap-2 text-[#45CFFF]">
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      <span>Running sequential inference passes across selected checkpoints...</span>
                    </div>
                  ) : (
                    <span>Select 2+ checkpoints on the left and click Compare Checkpoints.</span>
                  )}
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {compareResults.map((r, i) => {
                    const isErr = r.success === false || !!r.error;
                    const text = r.generated_text || r.completion || "";
                    const modelTitle = r.checkpoint || r.checkpoint_name || `Model #${i + 1}`;
                    return (
                      <div
                        key={i}
                        className={cn(
                          "p-3 rounded-lg border space-y-2 flex flex-col justify-between transition-all",
                          isErr
                            ? "bg-[#FF657A]/10 border-[#FF657A]/40"
                            : "bg-[#0E1420] border-[#273548]"
                        )}
                      >
                        <div className="flex justify-between items-center pb-1.5 border-b border-[#273548] text-[10px]">
                          <span className={cn("font-bold truncate max-w-[200px]", isErr ? "text-[#FF657A]" : "text-[#45CFFF]")}>
                            {modelTitle}
                          </span>
                          <span className="text-[#5E7188] font-mono-num">
                            Step {r.step || "—"} {r.val_loss ? `| Val ${Number(r.val_loss).toFixed(4)}` : ""}
                          </span>
                        </div>

                        <div className="text-xs text-[#ECF3FC] whitespace-pre-wrap leading-relaxed flex-1 py-1 font-mono select-text">
                          {isErr ? (
                            <span className="text-[#FF657A] flex items-center gap-1.5">
                              <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                              <span>{r.error || text || "Evaluation failed on this checkpoint."}</span>
                            </span>
                          ) : text.trim() ? (
                            text
                          ) : (
                            <span className="text-[#5E7188] italic">
                              [Model emitted stop/EOS token immediately; no output tokens generated]
                            </span>
                          )}
                        </div>

                        <div className="pt-1.5 border-t border-[#273548] flex justify-between text-[10px] text-[#91A0B4]">
                          <span>
                            Speed:{" "}
                            <strong className="text-[#ECF3FC] font-mono-num">
                              {Math.round(r.tokens_per_sec || r.tokens_per_second || 0)}
                            </strong>{" "}
                            tok/s
                          </span>
                          <span>
                            Time:{" "}
                            <strong className="text-[#ECF3FC] font-mono-num">
                              {r.generation_time_sec?.toFixed(2) || "0.00"}s
                            </strong>
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Automated Benchmark Suite Table */}
      <div className="space-y-2 pt-2">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-bold text-[#ECF3FC] uppercase tracking-wider flex items-center gap-1.5">
            <Award className="w-3.5 h-3.5 text-[#45CFFF]" /> AUTOMATED BENCHMARK SUITE
          </h3>
          <button
            onClick={handleRunBenchmarkSuite}
            disabled={isRunningBenchmarks}
            className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md bg-[#131C29] border border-[#273548] hover:border-[#45CFFF]/50 text-xs font-bold text-[#ECF3FC] transition-colors"
          >
            <Play className="w-3 h-3 text-[#45CFFF]" />
            {isRunningBenchmarks ? "RUNNING SUITE..." : "RUN BENCHMARK SUITE"}
          </button>
        </div>

        <DataTable
          data={benchmarks}
          columns={benchmarkColumns}
          keyExtractor={(b) => b.id}
          searchPlaceholder="Search benchmark results..."
          emptyText="No benchmarks recorded yet. Click Run Benchmark Suite to evaluate model quality."
        />
      </div>
    </div>
  );
};
