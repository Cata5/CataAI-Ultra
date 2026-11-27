"use client";

import React, { useState, useEffect, useMemo, useRef } from "react";
import { api } from "@/services/api";
import { DatasetItem } from "@/types/api";
import { useDatasetOperations } from "@/features/datasets/DatasetOperations";
import { DataTable, Column } from "@/components/ui/DataTable";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { formatCompactNumber } from "@/lib/utils";
import {
  UploadCloud,
  FileText,
  Database,
  Sparkles,
  CheckCircle2,
  Eye,
  RefreshCw,
  Trash2,
  Wand2,
  AlertTriangle,
  ShieldCheck,
} from "lucide-react";
import { cn } from "@/lib/utils";

export const DatasetsView: React.FC = () => {
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const {
    operations,
    startOperation,
    dismissOperation,
    clearFinished,
    activeOperation,
    latestCompleted,
    latestQualityOperation,
  } = useDatasetOperations();
  const isUploading = operations.some(
    (operation) =>
      operation.kind === "upload" &&
      (operation.status === "running" || operation.status === "queued")
  );
  const [cleanOnImport, setCleanOnImport] = useState(true);
  const [uploadFeedback, setUploadFeedback] = useState<string | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<any | null>(null);
  const [datasetToDelete, setDatasetToDelete] = useState<DatasetItem | null>(null);
  const [sftSource, setSftSource] = useState<string>("");
  const [sftTemplate, setSftTemplate] = useState<string>("chatml");
  const [sftMaxRecords, setSftMaxRecords] = useState<string>("");
  const [sftResult, setSftResult] = useState<any | null>(null);
  const [sftBusy, setSftBusy] = useState(false);

  // Tokenizer Inspector State
  const [tokenizerText, setTokenizerText] = useState(
    "LLM Maker: A general-purpose local neural network training workstation. Proiectare și antrenare eficientă."
  );
  const [tokenizerResult, setTokenizerResult] = useState<any | null>(null);
  const [isTokenizing, setIsTokenizing] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchDatasets = async () => {
    try {
      const data = await api.listDatasets();
      setDatasets(data);
    } catch (e: any) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchDatasets();
  }, []);

  const completedOperations = useRef<Map<string, number>>(new Map());
  useEffect(() => {
    const finished = operations.filter(
      (operation) => operation.status === "completed" || operation.status === "failed"
    );
    const newlyFinished = finished.filter(
      (operation) =>
        completedOperations.current.get(operation.jobId) !== (operation.finishedAt ?? 0)
    );
    if (newlyFinished.length === 0) return;
    finished.forEach((operation) =>
      completedOperations.current.set(operation.jobId, operation.finishedAt ?? 0)
    );
    if (newlyFinished.some((operation) => operation.status === "completed")) {
      void fetchDatasets();
    }
  }, [operations]);

  const displayDatasets = useMemo(() => {
    return datasets.map((item) => {
      const operation = latestCompleted("tokenize", item.id);
      const result = operation?.result;
      if (!result) return item;
      const tokenCount = result.token_count || result.train_tokens || item.total_tokens || 0;
      return {
        ...item,
        status: "Tokenized",
        total_tokens: tokenCount,
        train_tokens: result.train_tokens || Math.round(tokenCount * 0.9),
        val_tokens: result.val_tokens || Math.round(tokenCount * 0.1),
      };
    });
  }, [datasets, latestCompleted]);

  const qualityOperation = latestQualityOperation;
  const qualityResult =
    qualityOperation?.status === "completed" ? qualityOperation.result : null;

  const handleInspectTokenizer = async () => {
    if (!tokenizerText.trim()) return;
    setIsTokenizing(true);
    try {
      const res = await api.inspectTokenizer(tokenizerText);
      setTokenizerResult(res);
    } catch (e: any) {
      console.error(e);
    } finally {
      setIsTokenizing(false);
    }
  };

  useEffect(() => {
    handleInspectTokenizer();
  }, []);

  const handleFilesUpload = async (files: FileList | File[]) => {
    const fileArray = Array.from(files);
    if (fileArray.length === 0 || isUploading) return;
    const batchName =
      fileArray.length === 1 ? fileArray[0].name : `${fileArray.length} files`;
    void startOperation({
      kind: "upload",
      datasetId: `upload-${Date.now()}`,
      datasetName: batchName,
      run: async (jobId, update) => {
        let successCount = 0;
        const cleanings: any[] = [];
        for (let i = 0; i < fileArray.length; i++) {
          const file = fileArray[i];
          update({
            progress: Math.round((i / fileArray.length) * 100),
            current: i,
            total: fileArray.length,
            stage: "uploading",
            message: `Uploading ${i + 1} of ${fileArray.length}: ${file.name}`,
          });
          const res = await api.uploadDataset(file, {
            clean: cleanOnImport,
            jobId: fileArray.length === 1 ? jobId : undefined,
          });
          successCount++;
          if (res?.cleaning && typeof res.cleaning === "object") {
            cleanings.push(res.cleaning);
          }
        }
        update({
          progress: 100,
          current: fileArray.length,
          total: fileArray.length,
          stage: "finalizing",
          message: `Uploaded ${successCount} dataset${successCount > 1 ? "s" : ""}`,
        });
        const failures = cleanings.filter((item) => item.success === false);
        const cleaning =
          cleanings.length === 0
            ? undefined
            : cleanings.length === 1
              ? cleanings[0]
              : {
                  success: failures.length === 0,
                  records_in: cleanings.reduce(
                    (sum, item) => sum + Number(item.records_in || 0),
                    0
                  ),
                  records_out: cleanings.reduce(
                    (sum, item) => sum + Number(item.records_out || 0),
                    0
                  ),
                  error: failures.length
                    ? `${failures.length} of ${cleanings.length} files failed cleaning: ${failures
                        .map((item) => item.error || "unknown error")
                        .join("; ")}`
                    : undefined,
                };
        return {
          success: true,
          uploaded: successCount,
          cleaning,
        };
      },
    });
  };

  const handleDeleteDataset = async () => {
    if (!datasetToDelete) return;
    const targetId = datasetToDelete.id;
    const targetName = datasetToDelete.name;
    // Optimistic removal: remove immediately from local state
    setDatasets((prev) => prev.filter((d) => d.id !== targetId));
    setDatasetToDelete(null);
    try {
      await api.deleteDataset(targetId);
      setUploadFeedback(`Dataset "${targetName}" deleted.`);
      await fetchDatasets();
      setTimeout(() => setUploadFeedback(null), 4000);
    } catch (e: any) {
      alert(`Deletion failed: ${e.message}`);
      await fetchDatasets();
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFilesUpload(e.dataTransfer.files);
    }
  };

  const columns: Column<DatasetItem>[] = [
    {
      header: "DATASET NAME",
      accessorKey: "name",
      sortable: true,
      cell: (item) => (
        <div className="flex items-center gap-2">
          <FileText className="w-3.5 h-3.5 text-[#737373] shrink-0" />
          <div className="flex flex-col">
            <span className="font-medium text-[#F2F2F2]">{item.name}</span>
            <span className="text-[10px] text-[#525252] font-mono">{item.id}</span>
          </div>
        </div>
      ),
    },
    {
      header: "SOURCE",
      accessorKey: "source",
      sortable: true,
      cell: (item) => (
        <span className="tech-badge">
          {item.is_uploaded ? "Uploaded" : item.source || "Built-in"}
        </span>
      ),
    },
    {
      header: "TOKENS",
      accessorKey: "total_tokens",
      sortable: true,
      cell: (item) => (
        <span className="font-mono-num font-medium text-[#F2F2F2]">
          {item.total_tokens ? formatCompactNumber(item.total_tokens) : "—"}
        </span>
      ),
    },
    {
      header: "SIZE",
      accessorKey: "size_mb",
      sortable: true,
      cell: (item) => (
        <span className="font-mono-num text-[#737373]">
          {item.size_mb ? `${item.size_mb.toFixed(1)} MB` : "—"}
        </span>
      ),
    },
    {
      header: "STATUS",
      accessorKey: "status",
      cell: (item) => (
        item.status === "Tokenized" ? (
          <span className="inline-flex items-center gap-1 text-[10px] text-[#4ade80] font-mono font-medium">
            <Sparkles className="w-3 h-3 text-[#4ade80]" /> Tokenized
          </span>
        ) : (
          <span className="inline-flex items-center gap-1 text-[10px] text-[#737373] font-mono">
            <CheckCircle2 className="w-3 h-3 text-[#737373]" /> Raw
          </span>
        )
      ),
    },
    {
      header: "ACTIONS",
      cell: (item) => (
        <div className="flex items-center gap-1">
          <button
            onClick={() => {
              if (activeOperation("tokenize", item.id)) return;
              void startOperation({
                kind: "tokenize",
                datasetId: item.id,
                datasetName: item.name,
                run: (jobId) => api.tokenizeDataset(item.id, jobId),
              });
            }}
            disabled={Boolean(activeOperation("tokenize", item.id))}
            className="btn btn-secondary btn-sm px-2 text-[10px]"
            title="Pre-tokenize dataset"
          >
            {activeOperation("tokenize", item.id) ? (
              <RefreshCw className="w-3 h-3 animate-spin" />
            ) : (
              <Sparkles className="w-3 h-3 text-[#737373]" />
            )}
            {activeOperation("tokenize", item.id)
              ? activeOperation("tokenize", item.id)?.progress !== null
                ? `${Math.round(activeOperation("tokenize", item.id)?.progress || 0)}%`
                : "…"
              : item.status === "Tokenized"
                ? "Re-tokenize"
                : "Tokenize"}
          </button>
          <button
            onClick={() => {
              if (activeOperation("clean", item.id)) return;
              void startOperation({
                kind: "clean",
                datasetId: item.id,
                datasetName: item.name,
                run: (jobId) =>
                  api.cleanDataset(
                    item.id,
                    { jaccard_threshold: 0.80, lid_threshold: 0.80 },
                    jobId
                  ),
              });
            }}
            disabled={Boolean(activeOperation("clean", item.id))}
            className="btn btn-secondary btn-sm px-2 text-[10px]"
            title="Clean quality, language, PII, and near-duplicates"
          >
            {activeOperation("clean", item.id) ? (
              <RefreshCw className="w-3 h-3 animate-spin" />
            ) : (
              <Wand2 className="w-3 h-3 text-[#737373]" />
            )}
            {activeOperation("clean", item.id)
              ? activeOperation("clean", item.id)?.progress !== null
                ? `${Math.round(activeOperation("clean", item.id)?.progress || 0)}%`
                : "Cleaning"
              : "Clean"}
          </button>
          <button
            onClick={() => {
              if (activeOperation("quality", item.id)) return;
              void startOperation({
                kind: "quality",
                datasetId: item.id,
                datasetName: item.name,
                run: (jobId) => api.auditDatasetQuality(item.id, { max_samples: 5000 }, jobId),
              });
            }}
            disabled={Boolean(activeOperation("quality", item.id))}
            className="btn btn-secondary btn-sm px-2 text-[10px]"
            title="Read-only audit: duplicates, PII, contamination, language, tokenizer fertility"
          >
            {activeOperation("quality", item.id) ? (
              <RefreshCw className="w-3 h-3 animate-spin" />
            ) : (
              <ShieldCheck className="w-3 h-3 text-[#737373]" />
            )}
            {activeOperation("quality", item.id)
              ? activeOperation("quality", item.id)?.progress !== null
                ? `${Math.round(activeOperation("quality", item.id)?.progress || 0)}%`
                : "Auditing"
              : "Quality"}
          </button>
          <button
            onClick={async () => {
              try {
                const stats = await api.getDatasetStats(item.id);
                setSelectedDataset(stats);
              } catch (e: any) {
                alert(`Inspection failed: ${e.message}`);
              }
            }}
            className="btn btn-ghost btn-sm px-2 text-[10px]"
          >
            <Eye className="w-3 h-3 text-[#737373]" /> Inspect
          </button>
          <button
            onClick={() => setDatasetToDelete(item)}
            className="btn btn-danger btn-sm p-1"
            title="Delete Dataset"
          >
            <Trash2 className="w-3 h-3 text-[#f87171]" />
          </button>
        </div>
      ),
    },
  ];

  return (
    <div className="space-y-3 font-mono text-xs max-w-[1600px] mx-auto animate-in fade-in duration-150">
      {/* Upload Drop Zone */}
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={cn(
          "panel border-dashed p-6 flex flex-col items-center justify-center text-center cursor-pointer transition-colors bg-[#202020] border-[#333333]",
          isUploading
            ? "border-[#D4D4D4] bg-[#262626]"
            : "hover:border-[#525252] hover:bg-[#262626]"
        )}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".txt,.json,.jsonl,.csv,.parquet,.arrow"
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              handleFilesUpload(e.target.files);
            }
          }}
        />
        <label
          onClick={(e) => e.stopPropagation()}
          className="mb-2 inline-flex items-center gap-1.5 text-[10px] text-[#A3A3A3] cursor-pointer select-none"
        >
          <input
            type="checkbox"
            checked={cleanOnImport}
            onChange={(e) => setCleanOnImport(e.target.checked)}
            className="accent-[#D4D4D4]"
          />
          Clean on import
        </label>
        <div className="p-2.5 bg-[#1B1B1B] rounded-full border border-[#333333] text-[#A3A3A3] mb-2">
          <UploadCloud className="w-5 h-5" />
        </div>
        <h4 className="text-xs font-semibold text-[#F2F2F2] uppercase tracking-wide mb-0.5">
          Upload Training Dataset
        </h4>
        <p className="text-[11px] text-[#737373] max-w-md mb-1.5 leading-relaxed">
          Supports .parquet, .arrow, .jsonl, .txt, .csv with block-aware quality and language filtering, PII masking, near-duplicate removal, and Jinja2 chat templates.
        </p>
        <span className="px-2 py-0.5 rounded bg-[#1B1B1B] border border-[#262626] text-[10px] text-[#525252] uppercase">
          Click or drop files (.parquet, .arrow, .jsonl, .txt, .csv)
        </span>
      </div>

      {uploadFeedback && (
        <div className="p-2 rounded bg-[#262626] border border-[#333333] text-[#F2F2F2] font-mono text-xs flex items-center gap-2">
          <CheckCircle2 className="w-3.5 h-3.5 text-[#4ade80]" /> {uploadFeedback}
        </div>
      )}

      {operations.length > 0 && (
        <div className="panel p-3 bg-[#202020] border-[#333333] space-y-2">
          <div className="flex items-center justify-between pb-1.5 border-b border-[#262626]">
            <h3 className="panel-title">
              <RefreshCw className="w-3.5 h-3.5 text-[#737373]" /> Dataset Operations
            </h3>
            <div className="flex items-center gap-3 text-[10px] text-[#737373]">
              <span>{operations.filter((operation) => operation.status === "running" || operation.status === "queued").length} active</span>
              {operations.some((operation) => operation.status !== "running" && operation.status !== "queued") && (
                <button onClick={clearFinished} className="hover:text-[#F2F2F2]">
                  clear finished
                </button>
              )}
            </div>
          </div>
          <div className="space-y-2">
            {operations.slice(0, 6).map((operation) => {
              const active = operation.status === "running" || operation.status === "queued";
              const percent = operation.progress;
              const cleaningResult =
                operation.result &&
                typeof operation.result === "object" &&
                operation.result.cleaning &&
                typeof operation.result.cleaning === "object"
                  ? operation.result.cleaning
                  : null;
              const statusTone =
                operation.status === "completed"
                  ? "text-[#4ade80]"
                  : operation.status === "failed"
                    ? "text-[#f87171]"
                    : "text-[#D4D4D4]";
              return (
                <div
                  key={operation.jobId}
                  className="p-2 rounded bg-[#1B1B1B] border border-[#262626] space-y-1.5"
                >
                  <div className="flex items-center justify-between gap-2 text-[10px]">
                    <div className="flex items-center gap-1.5 min-w-0">
                      {operation.kind === "tokenize" ? (
                        <Sparkles className="w-3 h-3 text-[#737373] shrink-0" />
                      ) : operation.kind === "clean" ? (
                        <Wand2 className="w-3 h-3 text-[#737373] shrink-0" />
                      ) : operation.kind === "quality" ? (
                        <ShieldCheck className="w-3 h-3 text-[#737373] shrink-0" />
                      ) : (
                        <UploadCloud className="w-3 h-3 text-[#737373] shrink-0" />
                      )}
                      <span className="uppercase tracking-wide text-[#A3A3A3]">
                        {operation.kind === "tokenize"
                          ? "Tokenize"
                          : operation.kind === "clean"
                            ? "Clean"
                            : operation.kind === "quality"
                              ? "Quality"
                              : "Upload"}
                      </span>
                      <span className="text-[#525252] truncate">· {operation.datasetName}</span>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <span className={cn("font-mono uppercase", statusTone)}>
                        {operation.status === "running" ? "running" : operation.status}
                      </span>
                      {!active && (
                        <button
                          onClick={() => dismissOperation(operation.jobId)}
                          className="text-[#525252] hover:text-[#F2F2F2]"
                          title="Dismiss"
                        >
                          ×
                        </button>
                      )}
                    </div>
                  </div>
                  <div className="h-1.5 rounded-full bg-[#262626] overflow-hidden">
                    {percent !== null ? (
                      <div
                        className={cn(
                          "h-full rounded-full transition-all duration-500",
                          operation.status === "completed"
                            ? "bg-[#4ade80]"
                            : operation.status === "failed"
                              ? "bg-[#f87171]"
                              : "bg-[#D4D4D4]"
                        )}
                        style={{ width: `${Math.max(0, Math.min(100, percent))}%` }}
                      />
                    ) : (
                      <div className="h-full w-1/3 rounded-full bg-[#D4D4D4] animate-pulse" />
                    )}
                  </div>
                  <div className="flex items-center justify-between gap-2 text-[10px] text-[#737373]">
                    <span className="truncate">{operation.message || operation.stage}</span>
                    <span className="font-mono shrink-0">
                      {percent !== null ? `${Math.round(percent)}%` : ""}
                      {operation.total &&
                      !(operation.kind === "tokenize" && operation.total === 100)
                        ? ` · ${formatCompactNumber(operation.current)} / ${formatCompactNumber(operation.total)}`
                        : operation.current > 0 &&
                            !(operation.kind === "tokenize" && operation.total === 100)
                          ? ` · ${formatCompactNumber(operation.current)} processed`
                          : ""}
                    </span>
                  </div>
                  {operation.error && (
                    <div className="flex items-center gap-1.5 text-[10px] text-[#f87171]">
                      <AlertTriangle className="w-3 h-3 shrink-0" /> {operation.error}
                    </div>
                  )}
                  {operation.kind === "upload" &&
                    operation.status === "completed" &&
                    cleaningResult && (
                      <div
                        className={cn(
                          "text-[10px]",
                          cleaningResult.success ? "text-[#737373]" : "text-[#fbbf24]"
                        )}
                      >
                        {cleaningResult.success
                          ? `Cleaning kept ${formatCompactNumber(cleaningResult.records_out)} of ${formatCompactNumber(cleaningResult.records_in)} records`
                          : `Uploaded raw; cleaning failed: ${cleaningResult.error || "unknown error"}`}
                      </div>
                    )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* SFT Formatter */}
      <div className="panel p-3 bg-[#202020] border-[#333333] space-y-2">
        <div className="flex items-center justify-between pb-1.5 border-b border-[#262626]">
          <h3 className="panel-title">
            <Wand2 className="w-3.5 h-3.5 text-[#737373]" /> SFT Formatter
          </h3>
          <span className="text-[10px] font-mono text-[#525252]">Alpaca / ShareGPT / ChatML → masked JSONL</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-[1fr_160px_120px_auto] gap-2 items-end">
          <label className="flex flex-col gap-1 text-[10px] font-mono text-[#737373]">
            SOURCE DATASET
            <select
              value={sftSource}
              onChange={(e) => setSftSource(e.target.value)}
              className="bg-[#1B1B1B] border border-[#333333] rounded px-2 py-1.5 text-xs font-mono text-[#F2F2F2] outline-none focus:border-[#737373]"
            >
              <option value="">Select a dataset…</option>
              {displayDatasets.filter((d) => d.is_uploaded).map((d) => (
                <option key={d.id} value={d.id}>{d.name}</option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-[10px] font-mono text-[#737373]">
            TEMPLATE
            <select
              value={sftTemplate}
              onChange={(e) => setSftTemplate(e.target.value)}
              className="bg-[#1B1B1B] border border-[#333333] rounded px-2 py-1.5 text-xs font-mono text-[#F2F2F2] outline-none focus:border-[#737373]"
            >
              <option value="chatml">ChatML</option>
              <option value="llama3">Llama 3</option>
              <option value="deepseek_r1">DeepSeek R1</option>
              <option value="alpaca">Alpaca</option>
            </select>
          </label>
          <label className="flex flex-col gap-1 text-[10px] font-mono text-[#737373]">
            MAX RECORDS
            <input
              value={sftMaxRecords}
              onChange={(e) => setSftMaxRecords(e.target.value.replace(/[^0-9]/g, ""))}
              placeholder="all"
              className="bg-[#1B1B1B] border border-[#333333] rounded px-2 py-1.5 text-xs font-mono text-[#F2F2F2] outline-none focus:border-[#737373]"
            />
          </label>
          <button
            disabled={!sftSource || sftBusy}
            onClick={async () => {
              setSftBusy(true);
              setSftResult(null);
              try {
                const res = await api.formatSftDataset(sftSource, {
                  template: sftTemplate,
                  max_records: sftMaxRecords ? Number(sftMaxRecords) : undefined,
                  register: true,
                });
                setSftResult(res);
                await fetchDatasets();
              } catch (e: any) {
                setSftResult({ error: e.message });
              } finally {
                setSftBusy(false);
              }
            }}
            className="btn btn-primary btn-sm h-[30px] flex items-center gap-1"
          >
            {sftBusy ? <RefreshCw className="w-3 h-3 animate-spin" /> : <Wand2 className="w-3 h-3" />}
            Format
          </button>
        </div>
        {sftResult && (
          <div className="p-2 rounded bg-[#1B1B1B] border border-[#262626] text-[11px] font-mono space-y-1">
            {sftResult.error ? (
              <div className="flex items-center gap-2 text-rose-400"><AlertTriangle className="w-3.5 h-3.5" /> {sftResult.error}</div>
            ) : (
              <>
                <div className="flex flex-wrap gap-x-4 gap-y-1 text-[#A3A3A3]">
                  <span>valid <strong className="text-[#4ade80]">{sftResult.valid_records}</strong></span>
                  <span>invalid <strong className={sftResult.invalid_records ? "text-amber-400" : "text-[#737373]"}>{sftResult.invalid_records}</strong></span>
                  <span>sha256 <strong className="text-[#737373]">{String(sftResult.sha256 || "").slice(0, 12)}…</strong></span>
                  <span className="truncate max-w-[420px] text-[#525252]">{sftResult.output_path}</span>
                </div>
                {sftResult.errors?.slice(0, 3).map((err: any, i: number) => (
                  <div key={i} className="text-amber-400 truncate">line {err.line}: {err.error}</div>
                ))}
              </>
            )}
          </div>
        )}
      </div>

      {qualityResult && (
        <div className="panel p-3 bg-[#202020] border-[#333333] space-y-2">
          <div className="flex items-center justify-between pb-1.5 border-b border-[#262626]">
            <h3 className="panel-title">
              <ShieldCheck className="w-3.5 h-3.5 text-[#737373]" /> Quality Audit · {qualityResult.name || qualityOperation?.datasetName}
            </h3>
            <button
              onClick={() => qualityOperation && dismissOperation(qualityOperation.jobId)}
              className="text-[10px] text-[#737373] hover:text-[#F2F2F2]"
            >
              close
            </button>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-2 text-[10px]">
            {[
              ["ANALYZED", qualityResult.documents_analyzed],
              ["TRAINABLE", qualityResult.accepted_for_training],
              ["QUALITY DROP", qualityResult.quality_rejected],
              ["DUPLICATES", qualityResult.duplicates],
              ["CONTAMINATED", qualityResult.contaminated],
              ["PII DOCS", qualityResult.pii_documents],
              ["TOKENS/WORD", qualityResult.tokens_per_word ?? "—"],
            ].map(([label, value]) => (
              <div key={String(label)} className="p-2 rounded bg-[#1B1B1B] border border-[#262626]">
                <div className="text-[#737373]">{label}</div>
                <div className="text-[#F2F2F2] text-sm">{String(value ?? "—")}</div>
              </div>
            ))}
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-[#A3A3A3]">
            <span>language rejected <strong>{qualityResult.language_rejected}</strong></span>
            <span>unknown tokens <strong>{qualityResult.unknown_token_rate_pct ?? "—"}%</strong></span>
            <span>eval docs indexed <strong>{qualityResult.eval_documents}</strong></span>
            <span className="text-amber-300/90">PII is masked in previews; audit never rewrites the source file.</span>
          </div>
          {qualityResult.flagged_preview?.length ? (
            <div className="space-y-1">
              <div className="text-[10px] text-[#737373]">FLAGGED PREVIEW ({qualityResult.flagged_preview.length}{qualityResult.flagged_preview_truncated ? "+" : ""})</div>
              {qualityResult.flagged_preview.slice(0, 8).map((row: any) => (
                <div key={row.index} className="p-2 rounded bg-[#1B1B1B] border border-[#262626] text-[10px]">
                  <div className="flex gap-2 text-amber-300"><strong>#{row.index}</strong> {row.reasons.join(", ")}</div>
                  <div className="text-[#737373] truncate">{row.snippet}</div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-[10px] text-[#4ade80]">No flagged documents in the analyzed sample.</div>
          )}
        </div>
      )}

      {/* Dataset Table */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <h3 className="panel-title">
            <Database className="w-3.5 h-3.5 text-[#737373]" /> Datasets ({displayDatasets.length})
          </h3>
          <button
            onClick={fetchDatasets}
            className="icon-btn w-6 h-6"
            title="Refresh Datasets"
          >
            <RefreshCw className="w-3 h-3" />
          </button>
        </div>

        <DataTable
          data={displayDatasets}
          columns={columns}
          keyExtractor={(d) => d.id}
          searchPlaceholder="Search datasets by name..."
        />
      </div>

      {/* Tokenizer Live Inspector */}
      <div className="panel p-3 bg-[#202020] border-[#333333] space-y-2">
        <div className="flex items-center justify-between pb-1.5 border-b border-[#262626]">
          <h3 className="panel-title">
            <Sparkles className="w-3.5 h-3.5 text-[#737373]" /> Tokenizer & Vocabulary Inspector
          </h3>
          <button
            onClick={handleInspectTokenizer}
            disabled={isTokenizing}
            className="btn btn-secondary btn-sm"
          >
            {isTokenizing ? "Tokenizing..." : "Inspect Tokens"}
          </button>
        </div>

        <textarea
          rows={2}
          value={tokenizerText}
          onChange={(e) => setTokenizerText(e.target.value)}
          placeholder="Enter text to inspect BPE tokenization..."
          className="w-full bg-[#1B1B1B] border border-[#333333] rounded p-2 text-xs font-mono text-[#F2F2F2] outline-none focus:border-[#737373]"
        />

        {tokenizerResult && (
          <div className="space-y-1.5 pt-0.5">
            <div className="flex flex-wrap items-center gap-3 text-[11px] font-mono text-[#737373]">
              <span>Tokens: <strong className="text-[#F2F2F2] font-mono-num">{tokenizerResult.token_count || 0}</strong></span>
              <span>Characters: <strong className="text-[#A3A3A3] font-mono-num">{tokenizerResult.character_count || 0}</strong></span>
              <span>Compression: <strong className="text-[#A3A3A3] font-mono-num">{tokenizerResult.compression_ratio ? tokenizerResult.compression_ratio.toFixed(2) : "—"}</strong></span>
            </div>

            {/* Token Badges */}
            <div className="p-2 bg-[#171717] border border-[#262626] rounded flex flex-wrap gap-1 max-h-36 overflow-y-auto">
              {(tokenizerResult.tokens || []).map((t: any, idx: number) => (
                <span
                  key={idx}
                  className="px-1.5 py-0.2 rounded text-[10px] font-mono border flex items-center gap-1 bg-[#202020] border-[#333333] text-[#F2F2F2]"
                  title={`Token ID: ${t.id || t}`}
                >
                  <span>{typeof t === "object" ? t.text : String(t)}</span>
                  {t.id !== undefined && (
                    <span className="text-[9px] text-[#525252]">#{t.id}</span>
                  )}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Dataset Details Modal */}
      {selectedDataset && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
          <div className="bg-[#202020] border border-[#333333] rounded-lg p-4 max-w-md w-full space-y-3 shadow-xl">
            <div className="flex justify-between items-center pb-2 border-b border-[#262626]">
              <h3 className="text-xs font-semibold text-[#F2F2F2] uppercase">
                {selectedDataset.name || "Dataset Details"}
              </h3>
              <button
                onClick={() => setSelectedDataset(null)}
                className="text-[#737373] hover:text-[#F2F2F2]"
              >
                ✕
              </button>
            </div>

            <div className="space-y-1.5 text-xs">
              <div className="flex justify-between">
                <span className="text-[#737373]">File Path:</span>
                <span className="text-[#F2F2F2] truncate max-w-xs">{selectedDataset.file_path || selectedDataset.path || "—"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#737373]">Total Tokens:</span>
                <span className="text-[#F2F2F2] font-mono-num font-medium">{(selectedDataset.total_tokens || selectedDataset.token_count)?.toLocaleString() || "—"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#737373]">Size:</span>
                <span className="text-[#A3A3A3] font-mono-num">{selectedDataset.size_mb ? `${selectedDataset.size_mb.toFixed(2)} MB` : "—"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#737373]">Cache:</span>
                <span className="text-[#4ade80]">{selectedDataset.cached ? "Cached (.pt ready)" : "Raw"}</span>
              </div>
            </div>

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setSelectedDataset(null)}
                className="btn btn-secondary btn-sm"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Confirm Deletion */}
      <ConfirmDialog
        isOpen={!!datasetToDelete}
        title="Delete Dataset"
        message={`Delete "${datasetToDelete?.name}" (${datasetToDelete?.id})? This removes uploaded source files and pre-tokenized cache.`}
        confirmLabel="Delete"
        cancelLabel="Cancel"
        variant="danger"
        onConfirm={handleDeleteDataset}
        onCancel={() => setDatasetToDelete(null)}
      />
    </div>
  );
};
