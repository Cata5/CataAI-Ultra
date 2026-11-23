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
  minTokens: number;
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
  minTokens: 1,
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

  const formatFinalPrompt = (rawPrompt: string) => {
    if (!isChatMode) return rawPrompt;
    return `<|user|>\n${rawPrompt}\n<|assistant|>\n`;
  };

  const handleGenerate = async () => {
    if (!prompt.trim() || isGenerating) return;
    setIsGenerating(true);
    setEvalError(null);
    setGenerationResult(null);

    const formatted = formatFinalPrompt(prompt);
    const stopList = config.stopSequences
      ? config.stopSequences.split(",").map((s) => s.trim()).filter(Boolean)
      : undefined;

    try {
      const res = await api.generateEvaluation({
        prompt: formatted,
        checkpoint: selectedCheckpoint || undefined,
        max_new_tokens: config.maxTokens,
        min_new_tokens: config.minTokens,
        temperature: config.temperature,
        top_p: config.topP,
        top_k: config.topK,
        min_p: config.minP,
        repetition_penalty: config.repPenalty,
        no_repeat_ngram_size: config.noRepeatNgram,
        presence_penalty: config.presencePenalty,
        frequency_penalty: config.frequencyPenalty,
        seed: config.seed ? parseInt(config.seed, 10) : undefined,
        stop_sequences: stopList,
      });

      if (res.success === false && res.error) {
        setEvalError(res.error);
      } else {
        setGenerationResult(res);
      }
    } catch (e: any) {
      setEvalError(e?.message || "Inference execution failed.");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleRunComparison = async () => {
    if (compareCheckpoints.length === 0 || !prompt.trim() || isComparing) return;
    setIsComparing(true);
    setEvalError(null);
    setCompareResults([]);

    const formatted = formatFinalPrompt(prompt);
    const stopList = config.stopSequences
      ? config.stopSequences.split(",").map((s) => s.trim()).filter(Boolean)
      : undefined;

    try {
      const res = await api.compareEvaluation({
        prompt: formatted,
        checkpoint_paths: compareCheckpoints,
        max_new_tokens: config.maxTokens,
        min_new_tokens: config.minTokens,
        temperature: config.temperature,
        top_p: config.topP,
        top_k: config.topK,
        min_p: config.minP,
        repetition_penalty: config.repPenalty,
        no_repeat_ngram_size: config.noRepeatNgram,
        presence_penalty: config.presencePenalty,
        frequency_penalty: config.frequencyPenalty,
        seed: config.seed ? parseInt(config.seed, 10) : undefined,
        stop_sequences: stopList,
      });

      const list = res.results || res || [];
      setCompareResults(list);
    } catch (e: any) {
      setEvalError(e?.message || "Comparison failed.");
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
        <span className="font-medium text-[#F2F2F2] font-mono">{b.task_name}</span>
      ),
    },
    {
      header: "CHECKPOINT",
      accessorKey: "checkpoint_id",
      cell: (b) => (
        <span className="text-[#A3A3A3] font-mono">{b.checkpoint_id || "Active Weights"}</span>
      ),
    },
    {
      header: "METRIC",
      accessorKey: "metric_name",
      cell: (b) => (
        <span className="text-[#737373] font-mono uppercase text-[10px]">{b.metric_name || "Score"}</span>
      ),
    },
    {
      header: "SCORE",
      accessorKey: "score",
      sortable: true,
      cell: (b) => (
        <span className="font-mono-num font-medium text-[#F2F2F2]">
          {typeof b.score === "number" ? b.score.toFixed(4) : "—"}
        </span>
      ),
    },
    {
      header: "TIMESTAMP",
      accessorKey: "created_at",
      cell: (b) => (
        <span className="text-[#525252] text-[10px] font-mono">
          {b.created_at ? new Date(b.created_at).toLocaleString() : "—"}
        </span>
      ),
    },
  ];

  return (
    <div className="space-y-3 font-mono text-xs max-w-[1600px] mx-auto animate-in fade-in duration-150">
      {/* Top Header Mode Toggle */}
      <div className="panel p-3 bg-[#202020] border-[#333333] flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded bg-[#262626] border border-[#333333] flex items-center justify-center shrink-0 text-[#F2F2F2]">
            <FlaskConical className="w-3.5 h-3.5" />
          </div>
          <div>
            <h2 className="text-xs font-semibold text-[#F2F2F2] uppercase tracking-wide">
              Evaluation & Inference
            </h2>
            <p className="text-[10px] text-[#737373]">
              Sampling parameter controls, multi-model comparison matrix, and benchmark suites.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1.5 bg-[#1B1B1B] p-0.5 rounded border border-[#262626]">
          <button
            onClick={() => setIsCompareMode(false)}
            className={cn(
              "px-3 py-1 rounded text-xs transition-colors",
              !isCompareMode
                ? "bg-[#D4D4D4] text-[#171717] font-semibold"
                : "text-[#737373] hover:text-[#F2F2F2]"
            )}
          >
            Single Inference
          </button>
          <button
            onClick={() => setIsCompareMode(true)}
            className={cn(
              "px-3 py-1 rounded text-xs transition-colors flex items-center gap-1",
              isCompareMode
                ? "bg-[#D4D4D4] text-[#171717] font-semibold"
                : "text-[#737373] hover:text-[#F2F2F2]"
            )}
          >
            <Columns className="w-3 h-3" />
            Comparison Matrix
          </button>
        </div>
      </div>

      {/* Main Grid: Controls Column + Output Area */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        {/* Controls Column */}
        <div className="space-y-3">
          {/* Checkpoint Selection */}
          <div className="panel p-3 bg-[#202020] border-[#333333] space-y-2.5">
            <div className="flex items-center justify-between pb-1.5 border-b border-[#262626]">
              <h4 className="panel-title">
                <Bookmark className="w-3.5 h-3.5 text-[#737373]" /> Model Source
              </h4>
              <button
                type="button"
                onClick={fetchCheckpoints}
                disabled={isLoadingCheckpoints}
                className="text-[10px] text-[#737373] hover:text-[#F2F2F2] transition-colors flex items-center gap-1"
                title="Refresh checkpoints"
              >
                <RefreshCw className={cn("w-3 h-3", isLoadingCheckpoints && "animate-spin")} />
                Refresh
              </button>
            </div>

            {!isCompareMode ? (
              <div className="space-y-1">
                <div className="flex items-center justify-between text-[10px] text-[#737373]">
                  <span>WEIGHT SOURCE</span>
                  {selectedCheckpoint && (
                    <button
                      type="button"
                      onClick={() => setSelectedCheckpoint("")}
                      className="text-[#A3A3A3] hover:underline"
                    >
                      In-Memory
                    </button>
                  )}
                </div>
                <select
                  value={selectedCheckpoint}
                  onChange={(e) => setSelectedCheckpoint(e.target.value)}
                  className="input"
                >
                  <option value="">Current In-Memory Weights (Active Model)</option>
                  {checkpoints.map((c) => {
                    const valStr = c.val_loss !== null && c.val_loss !== undefined ? ` | Val: ${Number(c.val_loss).toFixed(4)}` : "";
                    const prefix = c.is_saved_model ? "📦 Model:" : "💾 Checkpoint:";
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
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-[10px] text-[#737373]">
                  <span>MODELS ({compareCheckpoints.length} selected)</span>
                  <div className="flex items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => setCompareCheckpoints(checkpoints.map((c) => c.path || c.filename))}
                      className="text-[#A3A3A3] hover:underline"
                    >
                      All
                    </button>
                    <span>·</span>
                    <button
                      type="button"
                      onClick={() => setCompareCheckpoints([])}
                      className="text-[#A3A3A3] hover:underline"
                    >
                      Clear
                    </button>
                  </div>
                </div>
                <div className="space-y-1 max-h-44 overflow-y-auto pr-1">
                  {checkpoints.length === 0 ? (
                    <div className="text-[11px] text-[#525252] italic p-2">
                      No models or checkpoints found.
                    </div>
                  ) : (
                    checkpoints.map((c) => {
                      const path = c.path || c.filename;
                      const isChecked = compareCheckpoints.includes(path);
                      return (
                        <label
                          key={c.id || path}
                          className={cn(
                            "flex items-center gap-2 p-1.5 rounded border text-[11px] cursor-pointer transition-colors",
                            isChecked
                              ? "bg-[#262626] border-[#D4D4D4] text-[#F2F2F2]"
                              : "bg-[#1B1B1B] border-[#262626] text-[#737373] hover:text-[#F2F2F2]"
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
                            className="accent-[#D4D4D4]"
                          />
                          <div className="flex items-center gap-1.5 truncate flex-1 min-w-0">
                            {c.is_saved_model ? (
                              <Package className="w-3 h-3 text-[#737373] shrink-0" />
                            ) : (
                              <Bookmark className="w-3 h-3 text-[#737373] shrink-0" />
                            )}
                            <span className="truncate">{c.model_name || c.filename}</span>
                          </div>
                          <span className="text-[9px] text-[#525252] shrink-0 font-mono-num">
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

          {/* Granular Sampling Hyperparameters */}
          <div className="panel p-3 bg-[#202020] border-[#333333] space-y-2.5">
            <div className="flex items-center justify-between pb-1.5 border-b border-[#262626]">
              <h4 className="panel-title">
                <Sliders className="w-3.5 h-3.5 text-[#737373]" /> Sampling Parameters
              </h4>
              <button
                type="button"
                onClick={handleResetDefaults}
                className="text-[10px] text-[#737373] hover:text-[#F2F2F2] transition-colors flex items-center gap-1"
                title="Reset parameters to defaults"
              >
                <RotateCcw className="w-3 h-3" />
                Defaults
              </button>
            </div>

            {/* Max & Min New Tokens */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <div className="flex justify-between items-center text-[10px] text-[#A3A3A3] font-medium mb-1">
                  <span className="uppercase">Min Tokens</span>
                  <input
                    type="number"
                    min="1"
                    max="256"
                    value={config.minTokens}
                    onChange={(e) => updateParam("minTokens", Math.max(1, parseInt(e.target.value) || 1))}
                    className="input w-14 text-right h-5 px-1 font-mono-num text-[10px]"
                    title="Minimum tokens before EOS/stop tokens are allowed"
                  />
                </div>
                <input
                  type="range"
                  min="1"
                  max="64"
                  value={config.minTokens}
                  onChange={(e) => updateParam("minTokens", parseInt(e.target.value))}
                  className="range"
                />
              </div>

              <div>
                <div className="flex justify-between items-center text-[10px] text-[#A3A3A3] font-medium mb-1">
                  <span className="uppercase">Max Tokens</span>
                  <input
                    type="number"
                    min="16"
                    max="2048"
                    step="16"
                    value={config.maxTokens}
                    onChange={(e) => updateParam("maxTokens", Math.max(16, parseInt(e.target.value) || 16))}
                    className="input w-14 text-right h-5 px-1 font-mono-num text-[10px]"
                  />
                </div>
                <input
                  type="range"
                  min="16"
                  max="1024"
                  step="16"
                  value={config.maxTokens}
                  onChange={(e) => updateParam("maxTokens", parseInt(e.target.value))}
                  className="range"
                />
              </div>
            </div>

            {/* Temperature */}
            <div>
              <div className="flex justify-between items-center text-[10px] text-[#737373] mb-1">
                <span className="uppercase">
                  Temperature {config.temperature === 0 ? "(Greedy)" : ""}
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
                  className="input w-16 text-right h-5 px-1 font-mono-num text-[10px]"
                />
              </div>
              <input
                type="range"
                min="0.0"
                max="2.0"
                step="0.05"
                value={config.temperature}
                onChange={(e) => updateParam("temperature", parseFloat(e.target.value))}
                className="range"
              />
            </div>

            {/* Top-P & Top-K */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <div className="flex justify-between items-center text-[10px] text-[#737373] mb-1">
                  <span className="uppercase">Top-P</span>
                  <span className="text-[#F2F2F2] font-mono-num">{config.topP.toFixed(2)}</span>
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
                  className="input"
                />
              </div>

              <div>
                <div className="flex justify-between items-center text-[10px] text-[#737373] mb-1">
                  <span className="uppercase">Top-K</span>
                  <span className="text-[#F2F2F2] font-mono-num">{config.topK === 0 ? "Off" : config.topK}</span>
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
                  className="input"
                />
              </div>
            </div>

            {/* Min-P & Repetition Penalty */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <div className="flex justify-between items-center text-[10px] text-[#737373] mb-1">
                  <span className="uppercase">Min-P</span>
                  <span className="text-[#F2F2F2] font-mono-num">{config.minP === 0 ? "Off" : config.minP.toFixed(2)}</span>
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
                  className="input"
                />
              </div>

              <div>
                <div className="flex justify-between items-center text-[10px] text-[#737373] mb-1">
                  <span className="uppercase">Rep Penalty</span>
                  <span className="text-[#F2F2F2] font-mono-num">{config.repPenalty.toFixed(2)}</span>
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
                  className="input"
                />
              </div>
            </div>

            {/* No Repeat N-Gram & RNG Seed */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <div className="flex justify-between items-center text-[10px] text-[#737373] mb-1">
                  <span className="uppercase">No Repeat N-Gram</span>
                  <span className="text-[#F2F2F2] font-mono-num">{config.noRepeatNgram === 0 ? "Off" : config.noRepeatNgram}</span>
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
                  className="input"
                />
              </div>

              <div>
                <div className="flex justify-between items-center text-[10px] text-[#737373] mb-1">
                  <span className="uppercase">Seed</span>
                  <span className="text-[#525252] text-[9px]">{config.seed ? `#${config.seed}` : "Random"}</span>
                </div>
                <input
                  type="number"
                  placeholder="e.g. 42"
                  value={config.seed}
                  onChange={(e) => updateParam("seed", e.target.value)}
                  className="input"
                />
              </div>
            </div>

            {/* Custom Stop Sequences */}
            <div>
              <label className="text-[10px] text-[#737373] uppercase font-medium block mb-1">
                Custom Stop Sequences
              </label>
              <input
                type="text"
                placeholder="<|end_turn|>, \n\n, </s>"
                value={config.stopSequences}
                onChange={(e) => updateParam("stopSequences", e.target.value)}
                className="input"
              />
            </div>
          </div>
        </div>

        {/* Right Output Area */}
        <div className="lg:col-span-2 space-y-3">
          {/* Prompt Editor */}
          <div className="panel p-3 bg-[#202020] border-[#333333] space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-1.5 border-b border-[#262626]">
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold text-[#F2F2F2] uppercase tracking-wide">
                  Prompt
                </span>
                <button
                  type="button"
                  onClick={() => setIsChatMode(!isChatMode)}
                  className={cn(
                    "px-2 py-0.5 rounded text-[10px] border transition-colors flex items-center gap-1",
                    isChatMode
                      ? "bg-[#262626] text-[#F2F2F2] border-[#D4D4D4]"
                      : "bg-[#1B1B1B] text-[#737373] border-[#262626]"
                  )}
                >
                  <Sparkles className="w-2.5 h-2.5" />
                  {isChatMode ? "Chat Template" : "Raw Completion"}
                </button>
              </div>

              <div className="flex flex-wrap gap-1 text-[10px]">
                <button
                  onClick={() => {
                    setPrompt("Salut, cine ești și ce poți face?");
                    setIsChatMode(true);
                  }}
                  className="px-2 py-0.5 rounded bg-[#1B1B1B] border border-[#262626] text-[#737373] hover:text-[#F2F2F2] transition-colors"
                >
                  Salut
                </button>
                <button
                  onClick={() => {
                    setPrompt("Explică pe scurt ce este un model de limbaj neuronal.");
                    setIsChatMode(true);
                  }}
                  className="px-2 py-0.5 rounded bg-[#1B1B1B] border border-[#262626] text-[#737373] hover:text-[#F2F2F2] transition-colors"
                >
                  Instrucțiune
                </button>
                <button
                  onClick={() => {
                    setPrompt("Capitala României este București, iar istoria sa modernă începe cu");
                    setIsChatMode(false);
                  }}
                  className="px-2 py-0.5 rounded bg-[#1B1B1B] border border-[#262626] text-[#737373] hover:text-[#F2F2F2] transition-colors"
                >
                  Completare
                </button>
              </div>
            </div>

            <textarea
              rows={3}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full bg-[#1B1B1B] border border-[#333333] rounded p-2 text-[#F2F2F2] text-xs font-mono outline-none focus:border-[#737373]"
              placeholder="Enter prompt text here..."
            />

            <div className="flex justify-between items-center pt-0.5">
              <div className="text-[10px] text-[#525252]">
                {prompt.length} chars · ~{Math.round(prompt.length / 4)} tokens
              </div>
              <div>
                {!isCompareMode ? (
                  <button
                    onClick={handleGenerate}
                    disabled={isGenerating || !prompt.trim()}
                    className="btn btn-primary btn-sm flex items-center gap-1.5"
                  >
                    <Play className="w-3 h-3 fill-current" />
                    {isGenerating ? "Generating..." : "Generate"}
                  </button>
                ) : (
                  <button
                    onClick={handleRunComparison}
                    disabled={isComparing || compareCheckpoints.length === 0 || !prompt.trim()}
                    className="btn btn-primary btn-sm flex items-center gap-1.5"
                  >
                    <Columns className="w-3 h-3" />
                    {isComparing ? "Comparing..." : `Compare (${compareCheckpoints.length})`}
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Inline Error Alert */}
          {evalError && (
            <div className="p-2.5 rounded bg-[rgba(248,113,113,0.08)] border border-[rgba(248,113,113,0.25)] text-[#f87171] flex items-start justify-between gap-2">
              <div className="flex items-start gap-2">
                <AlertCircle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                <div>
                  <strong className="block text-xs uppercase">Inference Error</strong>
                  <p className="text-[11px] text-[#f87171]/90 mt-0.5 whitespace-pre-wrap">{evalError}</p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setEvalError(null)}
                className="text-[#f87171] p-1"
                title="Dismiss"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          )}

          {/* Single Generation Output */}
          {!isCompareMode && (
            <div className="panel p-3 bg-[#202020] border-[#333333] space-y-2 min-h-[200px]">
              <div className="flex items-center justify-between pb-1.5 border-b border-[#262626]">
                <span className="text-xs font-semibold text-[#F2F2F2] uppercase tracking-wide flex items-center gap-1.5">
                  <Sparkles className="w-3 h-3 text-[#737373]" /> Generated Completion
                </span>
                {generationResult && (
                  <div className="flex items-center gap-3 text-[10px] font-mono text-[#737373]">
                    <span className="flex items-center gap-1">
                      <Gauge className="w-3 h-3" />
                      <span className="text-[#F2F2F2] font-mono-num font-medium">
                        {Math.round(generationResult.tokens_per_sec || generationResult.tokens_per_second || 0)}
                      </span> tok/s
                    </span>
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      <span className="text-[#F2F2F2] font-mono-num">
                        {generationResult.generation_time_sec?.toFixed(2) || "0.00"}s
                      </span>
                    </span>
                    <span className="flex items-center gap-1">
                      <Cpu className="w-3 h-3" />
                      {generationResult.device?.toUpperCase() || "ACCELERATOR"}
                    </span>
                  </div>
                )}
              </div>

              {generationResult ? (
                <div className="p-2.5 bg-[#171717] border border-[#262626] rounded font-mono text-xs leading-relaxed text-[#F2F2F2] whitespace-pre-wrap select-text">
                  {isChatMode ? (
                    <div>
                      <div className="text-[10px] text-[#A3A3A3] font-medium uppercase mb-1 flex items-center gap-1">
                        <CheckCircle2 className="w-3 h-3 text-[#4ade80]" />
                        <span>Assistant Output:</span>
                      </div>
                      <div className="text-[#F2F2F2] pl-2 border-l border-[#333333]">
                        {(generationResult.completion || generationResult.generated_text) ? (
                          generationResult.completion || generationResult.generated_text
                        ) : (
                          <span className="text-[#525252] italic">
                            [Model emitted EOS immediately]
                          </span>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div>
                      <span className="text-[#A3A3A3]">{prompt}</span>
                      <span> {generationResult.completion || generationResult.generated_text}</span>
                    </div>
                  )}
                </div>
              ) : (
                <div className="h-28 flex items-center justify-center text-[#525252] font-mono text-xs">
                  {isGenerating ? (
                    <div className="flex items-center gap-2 text-[#A3A3A3]">
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      <span>Generating tokens...</span>
                    </div>
                  ) : (
                    <span>Configure parameters and click Generate to evaluate output.</span>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Comparison Matrix Output */}
          {isCompareMode && (
            <div className="panel p-3 bg-[#202020] border-[#333333] space-y-2 min-h-[200px]">
              <div className="flex items-center justify-between pb-1.5 border-b border-[#262626]">
                <span className="text-xs font-semibold text-[#F2F2F2] uppercase tracking-wide flex items-center gap-1.5">
                  <Columns className="w-3 h-3 text-[#737373]" />
                  Comparison Matrix ({compareResults.length} / {compareCheckpoints.length})
                </span>
                {compareResults.length > 0 && (
                  <button
                    type="button"
                    onClick={() => setCompareResults([])}
                    className="text-[10px] text-[#737373] hover:text-[#F2F2F2]"
                  >
                    Clear
                  </button>
                )}
              </div>

              {compareResults.length === 0 ? (
                <div className="h-28 flex items-center justify-center text-[#525252] font-mono text-xs">
                  {isComparing ? (
                    <div className="flex items-center gap-2 text-[#A3A3A3]">
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      <span>Running sequential inference passes...</span>
                    </div>
                  ) : (
                    <span>Select 2+ checkpoints on the left and click Compare.</span>
                  )}
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
                  {compareResults.map((r, i) => {
                    const isErr = r.success === false || !!r.error;
                    const text = r.generated_text || r.completion || "";
                    const modelTitle = r.checkpoint || r.checkpoint_name || `Model #${i + 1}`;
                    return (
                      <div
                        key={i}
                        className={cn(
                          "p-2.5 rounded border space-y-1.5 flex flex-col justify-between",
                          isErr
                            ? "bg-[rgba(248,113,113,0.06)] border-[rgba(248,113,113,0.25)]"
                            : "bg-[#171717] border-[#262626]"
                        )}
                      >
                        <div className="flex justify-between items-center pb-1 border-b border-[#262626] text-[10px]">
                          <span className={cn("font-medium truncate max-w-[180px]", isErr ? "text-[#f87171]" : "text-[#F2F2F2]")}>
                            {modelTitle}
                          </span>
                          <span className="text-[#525252] font-mono-num">
                            Step {r.step || "—"} {r.val_loss ? `| Val ${Number(r.val_loss).toFixed(4)}` : ""}
                          </span>
                        </div>

                        <div className="text-xs text-[#F2F2F2] whitespace-pre-wrap leading-relaxed flex-1 py-1 font-mono select-text">
                          {isErr ? (
                            <span className="text-[#f87171] flex items-center gap-1">
                              <AlertCircle className="w-3 h-3 shrink-0" />
                              <span>{r.error || text || "Evaluation failed."}</span>
                            </span>
                          ) : text.trim() ? (
                            text
                          ) : (
                            <span className="text-[#525252] italic">
                              [Model emitted stop token]
                            </span>
                          )}
                        </div>

                        <div className="pt-1 border-t border-[#262626] flex justify-between text-[10px] text-[#737373]">
                          <span>
                            Speed: <strong className="text-[#F2F2F2] font-mono-num">{Math.round(r.tokens_per_sec || r.tokens_per_second || 0)}</strong> tok/s
                          </span>
                          <span>
                            Time: <strong className="text-[#F2F2F2] font-mono-num">{r.generation_time_sec?.toFixed(2) || "0.00"}s</strong>
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
      <div className="space-y-2 pt-1">
        <div className="flex items-center justify-between">
          <h3 className="panel-title">
            <Award className="w-3.5 h-3.5 text-[#737373]" /> Automated Benchmark Suite
          </h3>
          <button
            onClick={handleRunBenchmarkSuite}
            disabled={isRunningBenchmarks}
            className="btn btn-secondary btn-sm flex items-center gap-1"
          >
            <Play className="w-3 h-3 text-[#A3A3A3]" />
            {isRunningBenchmarks ? "Running Suite..." : "Run Benchmarks"}
          </button>
        </div>

        <DataTable
          data={benchmarks}
          columns={benchmarkColumns}
          keyExtractor={(b) => b.id}
          searchPlaceholder="Search benchmark results..."
          emptyText="No benchmarks recorded yet. Click Run Benchmarks to evaluate model quality."
        />
      </div>
    </div>
  );
};
