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
        <span className="text-[#91A0B4] font-mono">{b.checkpoint_id || "Active Run"}</span>
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
              Interactive token generation, checkpoint cross-evaluations and quality benchmarking.
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

      {/* Main Grid: Inputs & Controls (Left 1 col) + Output Console (Right 2 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Controls Column */}
        <div className="space-y-3">
          {/* Checkpoint Selection */}
          <div className="glass-panel p-3.5 rounded-lg space-y-3 relative overflow-hidden">
            <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#45CFFF]/15 to-transparent pointer-events-none" />

            <h4 className="text-xs font-bold text-[#ECF3FC] uppercase tracking-wider pb-1.5 border-b border-[#273548] flex items-center gap-1.5">
              <Bookmark className="w-3.5 h-3.5 text-[#45CFFF]" /> MODEL CHECKPOINT
            </h4>

            {!isCompareMode ? (
              <div>
                <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                  Selected Weights
                </label>
                <select
                  value={selectedCheckpoint}
                  onChange={(e) => setSelectedCheckpoint(e.target.value)}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2.5 py-1.5 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF]/50 text-xs"
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
                <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                  Checkpoints to Compare
                </label>
                <div className="space-y-1.5 max-h-36 overflow-y-auto">
                  {checkpoints.map((c) => {
                    const path = c.path || c.filename;
                    const isChecked = compareCheckpoints.includes(path);
                    return (
                      <label
                        key={c.id}
                        className="flex items-center gap-2 p-1.5 rounded-md bg-[#0E1420] border border-[#273548] text-[11px] cursor-pointer"
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
                          className="accent-[#45CFFF]"
                        />
                        <span className="truncate text-[#ECF3FC]">{c.filename}</span>
                        <span className="text-[9px] text-[#5E7188] ml-auto">
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
          <div className="glass-panel p-3.5 rounded-lg space-y-3 relative overflow-hidden">
            <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#45CFFF]/15 to-transparent pointer-events-none" />

            <h4 className="text-xs font-bold text-[#ECF3FC] uppercase tracking-wider pb-1.5 border-b border-[#273548] flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-[#45CFFF]" /> SAMPLING PARAMETERS
            </h4>

            <div>
              <div className="flex justify-between text-[10px] text-[#91A0B4] mb-1">
                <span>MAX NEW TOKENS</span>
                <span className="text-[#45CFFF] font-bold font-mono-num">{maxTokens}</span>
              </div>
              <input
                type="range"
                min="16"
                max="512"
                step="16"
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                className="w-full accent-[#45CFFF] bg-[#0E1420] h-1.5 rounded cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between text-[10px] text-[#91A0B4] mb-1">
                <span>TEMPERATURE</span>
                <span className="text-[#45CFFF] font-bold font-mono-num">{temperature.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="2.0"
                step="0.05"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full accent-[#45CFFF] bg-[#0E1420] h-1.5 rounded cursor-pointer"
              />
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                  Top-P (Nucleus)
                </label>
                <input
                  type="number"
                  step="0.05"
                  min="0.1"
                  max="1.0"
                  value={topP}
                  onChange={(e) => setTopP(parseFloat(e.target.value) || 0.9)}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2 py-1 text-[#ECF3FC]"
                />
              </div>

              <div>
                <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                  Top-K
                </label>
                <input
                  type="number"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value) || 50)}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2 py-1 text-[#ECF3FC]"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                  Repetition Penalty
                </label>
                <input
                  type="number"
                  step="0.05"
                  min="1.0"
                  max="2.0"
                  value={repPenalty}
                  onChange={(e) => setRepPenalty(parseFloat(e.target.value) || 1.2)}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2 py-1 text-[#ECF3FC]"
                />
              </div>

              <div>
                <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                  No Repeat N-Gram (0 = Off)
                </label>
                <input
                  type="number"
                  min="0"
                  max="10"
                  value={noRepeatNgram}
                  onChange={(e) => {
                    const raw = e.target.value;
                    if (raw === "") {
                      setNoRepeatNgram(0);
                      return;
                    }
                    const parsed = parseInt(raw, 10);
                    setNoRepeatNgram(isNaN(parsed) ? 0 : Math.max(0, parsed));
                  }}
                  className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2 py-1 text-[#ECF3FC]"
                />
              </div>
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
                  title="Wrap prompt with <|user|> ... <|assistant|> and stop on <|end_turn|>"
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
                    setPrompt("Explică pe scurt ce este un model de limbaj neuronal.");
                    setIsChatMode(true);
                  }}
                  className="px-2 py-0.5 rounded-md bg-[#0E1420] border border-[#273548] text-[#91A0B4] hover:text-[#45CFFF] hover:border-[#45CFFF]/40 transition-colors"
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
                  className="px-2 py-0.5 rounded-md bg-[#0E1420] border border-[#273548] text-[#91A0B4] hover:text-[#45CFFF] transition-colors"
                >
                  Tech
                </button>
                <button
                  onClick={() => {
                    setPrompt("Capitala României este București, iar istoria sa");
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
            />

            <div className="flex justify-end pt-1">
              {!isCompareMode ? (
                <button
                  onClick={handleGenerate}
                  disabled={isGenerating}
                  className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-md bg-[#45CFFF] hover:bg-[#80DEFF] disabled:opacity-40 text-[#090D14] font-bold transition-all shadow-sm"
                >
                  <Play className="w-3.5 h-3.5 fill-current" />
                  {isGenerating ? "GENERATING TOKENS..." : "EXECUTE INFERENCE"}
                </button>
              ) : (
                <button
                  onClick={handleRunComparison}
                  disabled={isComparing || compareCheckpoints.length === 0}
                  className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-md bg-[#45CFFF] hover:bg-[#80DEFF] disabled:opacity-40 text-[#090D14] font-bold transition-all shadow-sm"
                >
                  <Columns className="w-3.5 h-3.5" />
                  {isComparing ? "COMPARING CHECKPOINTS..." : "COMPARE CHECKPOINTS"}
                </button>
              )}
            </div>
          </div>

          {/* Single Generation Output */}
          {!isCompareMode && (
            <div className="glass-panel p-4 rounded-lg space-y-3 min-h-[220px] relative overflow-hidden">
              <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#45CFFF]/15 to-transparent pointer-events-none" />

              <div className="flex items-center justify-between pb-2 border-b border-[#273548]">
                <span className="text-xs font-bold text-[#ECF3FC] uppercase tracking-wider">
                  GENERATED COMPLETION
                </span>
                {generationResult && (
                  <div className="flex items-center gap-3 text-[10px] font-mono text-[#91A0B4]">
                    <span className="flex items-center gap-1">
                      <Gauge className="w-3 h-3 text-[#45CFFF]" />
                      <span className="text-[#ECF3FC] font-mono-num font-bold">{Math.round(generationResult.tokens_per_second || 0)}</span> tok/s
                    </span>
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3 text-[#9B8CFF]" />
                      <span className="text-[#ECF3FC] font-mono-num">{generationResult.generation_time_sec?.toFixed(2)}s</span>
                    </span>
                    <span className="flex items-center gap-1">
                      <Cpu className="w-3 h-3 text-[#5E7188]" />
                      {generationResult.device?.toUpperCase() || "ACCELERATOR"}
                    </span>
                  </div>
                )}
              </div>

              {generationResult ? (
                <div className="p-3 bg-[#090D14] border border-[#273548] rounded-md font-mono text-xs leading-relaxed text-[#ECF3FC] whitespace-pre-wrap">
                  {isChatMode ? (
                    <div>
                      <div className="text-[10px] text-[#45CFFF] font-bold uppercase mb-1 flex items-center gap-1.5">
                        <Sparkles className="w-3 h-3 text-[#45CFFF]" />
                        <span>ASSISTANT GENERATION:</span>
                      </div>
                      <div className="text-[#ECF3FC] pl-2 border-l-2 border-[#45CFFF]/50">
                        {generationResult.completion || generationResult.generated_text}
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
                  Enter prompt and click Execute Inference to evaluate model outputs.
                </div>
              )}
            </div>
          )}

          {/* Side-by-Side Comparison Output */}
          {isCompareMode && (
            <div className="glass-panel p-4 rounded-lg space-y-3 min-h-[220px] relative overflow-hidden">
              <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#45CFFF]/15 to-transparent pointer-events-none" />

              <div className="flex items-center justify-between pb-2 border-b border-[#273548]">
                <span className="text-xs font-bold text-[#ECF3FC] uppercase tracking-wider">
                  COMPARISON MATRIX ({compareResults.length} checkpoints)
                </span>
              </div>

              {compareResults.length === 0 ? (
                <div className="h-32 flex items-center justify-center text-[#5E7188] font-mono text-xs">
                  Select 2+ checkpoints and click Compare Checkpoints to view side-by-side outputs.
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {compareResults.map((r, i) => (
                    <div
                      key={i}
                      className="p-3 rounded-lg bg-[#0E1420] border border-[#273548] space-y-2 flex flex-col justify-between"
                    >
                      <div className="flex justify-between items-center pb-1.5 border-b border-[#273548] text-[10px]">
                        <span className="font-bold text-[#45CFFF] truncate max-w-[180px]">
                          {r.checkpoint}
                        </span>
                        <span className="text-[#5E7188]">Step {r.step || "—"}</span>
                      </div>
                      <p className="text-xs text-[#ECF3FC] whitespace-pre-wrap leading-relaxed flex-1">
                        {r.generated_text}
                      </p>
                      <div className="pt-1.5 border-t border-[#273548] flex justify-between text-[10px] text-[#91A0B4]">
                        <span>Speed: <strong className="text-[#ECF3FC] font-mono-num">{Math.round(r.tokens_per_second || 0)}</strong> tok/s</span>
                        <span>Time: <strong className="text-[#ECF3FC] font-mono-num">{r.generation_time_sec?.toFixed(2)}s</strong></span>
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