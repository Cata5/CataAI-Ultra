"use client";

import React, { useState, useEffect } from "react";
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
} from "lucide-react";
import { cn } from "@/lib/utils";

export const EvaluationView: React.FC = () => {
  const [checkpoints, setCheckpoints] = useState<CheckpointItem[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>("");
  const [benchmarks, setBenchmarks] = useState<BenchmarkResult[]>([]);

  // Generation parameters
  const [prompt, setPrompt] = useState<string>(
    "Inteligența artificială modernă funcționează prin"
  );
  const [maxTokens, setMaxTokens] = useState<number>(128);
  const [temperature, setTemperature] = useState<number>(0.7);
  const [topP, setTopP] = useState<number>(0.9);
  const [topK, setTopK] = useState<number>(50);
  const [repPenalty, setRepPenalty] = useState<number>(1.2);
  const [noRepeatNgram, setNoRepeatNgram] = useState<number>(3);
  const [isChatMode, setIsChatMode] = useState<boolean>(true);

  // Single inference result
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [generationResult, setGenerationResult] = useState<any | null>(null);

  // Side-by-side comparison
  const [isCompareMode, setIsCompareMode] = useState<boolean>(false);
  const [compareCheckpoints, setCompareCheckpoints] = useState<string[]>([]);
  const [compareResults, setCompareResults] = useState<any[]>([]);
  const [isComparing, setIsComparing] = useState<boolean>(false);

  // Benchmark suite
  const [isRunningBenchmarks, setIsRunningBenchmarks] = useState<boolean>(false);

  useEffect(() => {
    api.listCheckpoints().then((list) => {
      setCheckpoints(list);
      if (list.length > 0) {
        setSelectedCheckpoint(list[0].path || list[0].filename);
        if (list.length >= 2) {
          setCompareCheckpoints([list[0].path || list[0].filename, list[1].path || list[1].filename]);
        }
      }
    });

    api.listBenchmarkResults().then(setBenchmarks).catch(console.error);
  }, []);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    setIsGenerating(true);
    setGenerationResult(null);
    try {
      const res = await api.generateEvaluation({
        prompt,
        max_new_tokens: maxTokens,
        temperature,
        top_p: topP,
        top_k: topK,
        repetition_penalty: repPenalty,
        no_repeat_ngram_size: noRepeatNgram,
        checkpoint: selectedCheckpoint || undefined,
        chat_mode: isChatMode,
      });
      setGenerationResult(res);
    } catch (e: any) {
      alert(`Inference failed: ${e.message}`);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleRunComparison = async () => {
    if (!prompt.trim() || compareCheckpoints.length === 0) return;
    setIsComparing(true);
    setCompareResults([]);
    try {
      const res = await api.compareEvaluation({
        prompt,
        checkpoint_paths: compareCheckpoints,
      });
      setCompareResults(res.results || []);
    } catch (e: any) {
      alert(`Comparison failed: ${e.message}`);
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
      alert("Benchmark suite completed successfully!");
    } catch (e: any) {
      alert(`Benchmark error: ${e.message}`);
    } finally {
      setIsRunningBenchmarks(false);
    }
  };

  const benchmarkColumns: Column<BenchmarkResult>[] = [
    {
      header: "TASK / BENCHMARK",
      accessorKey: "task_name",
      sortable: true,
      cell: (item) => (
        <span className="font-bold text-slate-200">{item.task_name}</span>
      ),
    },
    {
      header: "SCORE",
      accessorKey: "score",
      sortable: true,
      cell: (item) => (
        <span className="font-mono-num font-bold text-cyan-400">
          {(item.score * 100).toFixed(1)}%
        </span>
      ),
    },
    {
      header: "METRIC",
      accessorKey: "metric_name",
      cell: (item) => (
        <span className="text-slate-400 font-mono text-[10px] uppercase">
          {item.metric_name}
        </span>
      ),
    },
    {
      header: "TIMESTAMP",
      accessorKey: "created_at",
      cell: (item) => (
        <span className="text-slate-400 text-[10px]">
          {item.created_at ? new Date(item.created_at).toLocaleString() : "—"}
        </span>
      ),
    },
  ];

  return (
    <div className="space-y-4 font-mono text-xs">
      {/* Top Bar with Mode Toggle */}
      <div className="glass-panel p-3.5 rounded-lg flex flex-wrap items-center justify-between gap-3 border-l-4 border-l-cyan-400">
        <div className="flex items-center gap-3">
          <FlaskConical className="w-5 h-5 text-cyan-400" />
          <div className="flex flex-col">
            <span className="text-xs font-bold text-slate-100 uppercase tracking-wider">
              EVALUATION LAB & INFERENCE ENGINE
            </span>
            <span className="text-[10px] text-slate-400">
              GPU Single-Owner Arbitration: CPU fallback active during background training runs
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsCompareMode(false)}
            className={cn(
              "px-3 py-1.5 rounded text-xs font-bold transition-all",
              !isCompareMode
                ? "bg-cyan-600 text-white shadow-[0_0_10px_rgba(0,240,255,0.3)]"
                : "bg-slate-900 text-slate-400 border border-slate-800 hover:text-slate-200"
            )}
          >
            SINGLE GENERATION
          </button>
          <button
            onClick={() => setIsCompareMode(true)}
            className={cn(
              "px-3 py-1.5 rounded text-xs font-bold transition-all flex items-center gap-1.5",
              isCompareMode
                ? "bg-cyan-600 text-white shadow-[0_0_10px_rgba(0,240,255,0.3)]"
                : "bg-slate-900 text-slate-400 border border-slate-800 hover:text-slate-200"
            )}
          >
            <Columns className="w-3.5 h-3.5" /> SIDE-BY-SIDE COMPARISON
          </button>
        </div>
      </div>

      {/* Main Grid: Inputs & Controls (Left 1 col) + Output Console (Right 2 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Controls Column */}
        <div className="space-y-3">
          {/* Checkpoint Selection */}
          <div className="glass-panel p-3.5 rounded-lg space-y-3">
            <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider pb-1.5 border-b border-slate-800 flex items-center gap-1.5">
              <Bookmark className="w-3.5 h-3.5 text-cyan-400" /> MODEL CHECKPOINT
            </h4>

            {!isCompareMode ? (
              <div>
                <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
                  Selected Weights
                </label>
                <select
                  value={selectedCheckpoint}
                  onChange={(e) => setSelectedCheckpoint(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-cyan-400 text-xs"
                >
                  <option value="">Current In-Memory Weights</option>
                  {checkpoints.map((c) => (
                    <option key={c.id} value={c.path || c.filename}>
                      {c.filename} (Step {c.step}
                      {c.val_loss ? ` | Val: ${c.val_loss.toFixed(4)}` : ""})
                    </option>
                  ))}
                </select>
              </div>
            ) : (
              <div>
                <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
                  Checkpoints to Compare
                </label>
                <div className="space-y-1.5 max-h-36 overflow-y-auto">
                  {checkpoints.map((c) => {
                    const path = c.path || c.filename;
                    const isChecked = compareCheckpoints.includes(path);
                    return (
                      <label
                        key={c.id}
                        className="flex items-center gap-2 p-1.5 rounded bg-slate-900 border border-slate-800 text-[11px] cursor-pointer"
                      >
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setCompareCheckpoints([...compareCheckpoints, path]);
                            } else {
                              setCompareCheckpoints(
                                compareCheckpoints.filter((p) => p !== path)
                              );
                            }
                          }}
                          className="accent-cyan-400"
                        />
                        <span className="truncate text-slate-200">{c.filename}</span>
                        <span className="text-[9px] text-slate-400 ml-auto">
                          Step {c.step}
                        </span>
                      </label>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          {/* Sampling Hyperparameters */}
          <div className="glass-panel p-3.5 rounded-lg space-y-3">
            <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider pb-1.5 border-b border-slate-800 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-cyan-400" /> SAMPLING PARAMETERS
            </h4>

            <div>
              <div className="flex justify-between text-[10px] text-slate-400 mb-1">
                <span>MAX NEW TOKENS</span>
                <span className="text-cyan-400 font-bold">{maxTokens}</span>
              </div>
              <input
                type="range"
                min="16"
                max="512"
                step="16"
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                className="w-full accent-cyan-400 bg-slate-800 h-1.5 rounded"
              />
            </div>

            <div>
              <div className="flex justify-between text-[10px] text-slate-400 mb-1">
                <span>TEMPERATURE</span>
                <span className="text-cyan-400 font-bold">{temperature.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="2.0"
                step="0.05"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full accent-cyan-400 bg-slate-800 h-1.5 rounded"
              />
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
                  Top-P (Nucleus)
                </label>
                <input
                  type="number"
                  step="0.05"
                  min="0.1"
                  max="1.0"
                  value={topP}
                  onChange={(e) => setTopP(parseFloat(e.target.value) || 0.9)}
                  className="w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-slate-200"
                />
              </div>

              <div>
                <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
                  Top-K
                </label>
                <input
                  type="number"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value) || 50)}
                  className="w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-slate-200"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
                  Repetition Penalty
                </label>
                <input
                  type="number"
                  step="0.05"
                  min="1.0"
                  max="2.0"
                  value={repPenalty}
                  onChange={(e) => setRepPenalty(parseFloat(e.target.value) || 1.2)}
                  className="w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-slate-200"
                />
              </div>

              <div>
                <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
                  No-Repeat N-Gram
                </label>
                <input
                  type="number"
                  min="0"
                  max="10"
                  value={noRepeatNgram}
                  onChange={(e) => setNoRepeatNgram(parseInt(e.target.value) || 0)}
                  className="w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-slate-200"
                  title="Bans repeated n-grams to permanently eliminate phrase loops (0 to disable)"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Output & Generation Column (2 cols on lg) */}
        <div className="lg:col-span-2 space-y-3">
          {/* Prompt Area */}
          <div className="glass-panel p-3.5 rounded-lg space-y-2">
            <div className="flex flex-wrap justify-between items-center gap-2">
              <div className="flex items-center gap-2">
                <label className="text-[10px] text-slate-400 uppercase font-semibold">
                  INFERENCE PROMPT
                </label>
                <button
                  type="button"
                  onClick={() => setIsChatMode(!isChatMode)}
                  className={cn(
                    "px-2 py-0.5 rounded text-[10px] font-bold border transition-colors flex items-center gap-1",
                    isChatMode
                      ? "bg-cyan-950/70 text-cyan-300 border-cyan-500/40 shadow-[0_0_8px_rgba(0,240,255,0.2)]"
                      : "bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200"
                  )}
                  title="Wrap prompt with <|user|> ... <|assistant|> and stop on <|end_turn|>"
                >
                  <Sparkles className="w-2.5 h-2.5" />
                  {isChatMode ? "CHAT MODE (<|user|> ... <|assistant|>)" : "RAW COMPLETION"}
                </button>
              </div>

              <div className="flex flex-wrap gap-1 text-[10px]">
                <button
                  onClick={() => {
                    setPrompt("Salut, cine ești și ce poți face?");
                    setIsChatMode(true);
                  }}
                  className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300 hover:text-cyan-300 hover:border-cyan-500/40"
                >
                  Salut!
                </button>
                <button
                  onClick={() => {
                    setPrompt("Explică pe scurt ce este un model de limbaj neuronal.");
                    setIsChatMode(true);
                  }}
                  className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300 hover:text-cyan-300 hover:border-cyan-500/40"
                >
                  Instrucțiune
                </button>
                <button
                  onClick={() => {
                    setPrompt(
                      "Definiția rețelelor neuronale artificiale și a mecanismului de atenție:"
                    );
                    setIsChatMode(false);
                  }}
                  className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400 hover:text-cyan-300"
                >
                  Tech
                </button>
                <button
                  onClick={() => {
                    setPrompt("Capitala României este București, iar istoria sa");
                    setIsChatMode(false);
                  }}
                  className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400 hover:text-cyan-300"
                >
                  Wiki
                </button>
              </div>
            </div>

            <textarea
              rows={3}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full bg-slate-900/80 border border-slate-800 rounded p-2.5 text-slate-200 focus:outline-none focus:border-cyan-400"
            />

            <div className="flex justify-end pt-1">
              {!isCompareMode ? (
                <button
                  onClick={handleGenerate}
                  disabled={isGenerating}
                  className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white font-bold transition-all shadow-[0_0_12px_rgba(0,240,255,0.3)]"
                >
                  <Play className="w-3.5 h-3.5" />
                  {isGenerating ? "GENERATING TOKENS..." : "EXECUTE INFERENCE"}
                </button>
              ) : (
                <button
                  onClick={handleRunComparison}
                  disabled={isComparing || compareCheckpoints.length === 0}
                  className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white font-bold transition-all shadow-[0_0_12px_rgba(0,240,255,0.3)]"
                >
                  <Columns className="w-3.5 h-3.5" />
                  {isComparing ? "COMPARING CHECKPOINTS..." : "COMPARE CHECKPOINTS"}
                </button>
              )}
            </div>
          </div>

          {/* Single Generation Output */}
          {!isCompareMode && (
            <div className="glass-panel p-4 rounded-lg space-y-3 min-h-[220px]">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <span className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                  GENERATED COMPLETION
                </span>
                {generationResult && (
                  <div className="flex items-center gap-3 text-[10px] font-mono text-slate-400">
                    <span className="flex items-center gap-1">
                      <Gauge className="w-3 h-3 text-cyan-400" />
                      {Math.round(generationResult.tokens_per_second || 0)} tok/s
                    </span>
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3 text-indigo-400" />
                      {generationResult.generation_time_sec?.toFixed(2)}s
                    </span>
                    <span className="flex items-center gap-1">
                      <Cpu className="w-3 h-3 text-slate-400" />
                      {generationResult.device?.toUpperCase() || "CPU"}
                    </span>
                  </div>
                )}
              </div>

              {generationResult ? (
                <div className="p-3 bg-slate-950/70 border border-slate-800 rounded font-mono text-xs leading-relaxed text-slate-100 whitespace-pre-wrap">
                  {isChatMode ? (
                    <div>
                      <div className="text-[10px] text-cyan-400 font-bold uppercase mb-1 flex items-center gap-1.5">
                        <Sparkles className="w-3 h-3 text-cyan-400" />
                        <span>ASSISTANT GENERATION (Chat Mode):</span>
                      </div>
                      <div className="text-slate-100 pl-2 border-l-2 border-cyan-500/50">
                        {generationResult.completion || generationResult.generated_text}
                      </div>
                    </div>
                  ) : (
                    <div>
                      <span className="text-cyan-400 font-bold">{prompt}</span>
                      <span> {generationResult.completion || generationResult.generated_text}</span>
                    </div>
                  )}
                </div>
              ) : (
                <div className="h-32 flex items-center justify-center text-slate-400 font-mono text-xs">
                  Enter prompt and click Execute Inference to evaluate model outputs.
                </div>
              )}
            </div>
          )}

          {/* Side-by-Side Comparison Output */}
          {isCompareMode && (
            <div className="glass-panel p-4 rounded-lg space-y-3 min-h-[220px]">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <span className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                  COMPARISON MATRIX ({compareResults.length} checkpoints)
                </span>
              </div>

              {compareResults.length === 0 ? (
                <div className="h-32 flex items-center justify-center text-slate-400 font-mono text-xs">
                  Select 2+ checkpoints and click Compare Checkpoints to view side-by-side outputs.
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {compareResults.map((r, i) => (
                    <div
                      key={i}
                      className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 space-y-2 flex flex-col justify-between"
                    >
                      <div className="flex justify-between items-center pb-1.5 border-b border-slate-850 text-[10px]">
                        <span className="font-bold text-cyan-400 truncate max-w-[180px]">
                          {r.checkpoint}
                        </span>
                        <span className="text-slate-400">Step {r.step || "—"}</span>
                      </div>
                      <p className="text-xs text-slate-200 whitespace-pre-wrap leading-relaxed flex-1">
                        {r.generated_text}
                      </p>
                      <div className="pt-1.5 border-t border-slate-850 flex justify-between text-[10px] text-slate-400">
                        <span>Speed: {Math.round(r.tokens_per_second || 0)} tok/s</span>
                        <span>Time: {r.generation_time_sec?.toFixed(2)}s</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Automated Benchmark Suite Table */}
      <div className="space-y-2 pt-2">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider flex items-center gap-1.5">
            <Award className="w-3.5 h-3.5 text-cyan-400" /> AUTOMATED BENCHMARK SUITE
          </h3>
          <button
            onClick={handleRunBenchmarkSuite}
            disabled={isRunningBenchmarks}
            className="inline-flex items-center gap-1.5 px-3 py-1 rounded bg-slate-900 border border-slate-700 hover:border-cyan-400 text-xs font-bold text-slate-200 transition-colors"
          >
            <Play className="w-3 h-3 text-cyan-400" />
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
