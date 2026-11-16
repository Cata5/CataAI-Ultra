"use client";

import React, { useState, useEffect, useRef } from "react";
import { api } from "@/services/api";
import { DatasetItem } from "@/types/api";
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
} from "lucide-react";
import { cn } from "@/lib/utils";

export const DatasetsView: React.FC = () => {
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [tokenizingDatasetId, setTokenizingDatasetId] = useState<string | null>(null);
  const [uploadFeedback, setUploadFeedback] = useState<string | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<any | null>(null);
  const [datasetToDelete, setDatasetToDelete] = useState<DatasetItem | null>(null);
  const [isDeleting, setIsDeleting] = useState<boolean>(false);

  // Tokenizer Inspector State
  const [tokenizerText, setTokenizerText] = useState(
    "RoCanon-48.5M este un model hibrid antrenat pe limba română cu GQA și SwiGLU."
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

  const handleFileUpload = async (file: File) => {
    setIsUploading(true);
    setUploadFeedback(`Uploading and validating ${file.name}...`);
    try {
      const res = await api.uploadDataset(file);
      setUploadFeedback(`Dataset ${file.name} successfully registered! Tokens: ${res.total_tokens || "Pre-tokenized"}`);
      fetchDatasets();
      setTimeout(() => setUploadFeedback(null), 5000);
    } catch (e: any) {
      alert(`Upload error: ${e.message}`);
      setUploadFeedback(null);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDeleteDataset = async () => {
    if (!datasetToDelete) return;
    setIsDeleting(true);
    try {
      await api.deleteDataset(datasetToDelete.id);
      setUploadFeedback(`Dataset "${datasetToDelete.name}" successfully deleted.`);
      setDatasetToDelete(null);
      await fetchDatasets();
      setTimeout(() => setUploadFeedback(null), 4000);
    } catch (e: any) {
      alert(`Deletion failed: ${e.message}`);
    } finally {
      setIsDeleting(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const columns: Column<DatasetItem>[] = [
    {
      header: "DATASET NAME",
      accessorKey: "name",
      sortable: true,
      cell: (item) => (
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4 text-cyan-400 shrink-0" />
          <div className="flex flex-col">
            <span className="font-bold text-slate-200">{item.name}</span>
            <span className="text-[10px] text-slate-400 font-mono">{item.id}</span>
          </div>
        </div>
      ),
    },
    {
      header: "TYPE / SOURCE",
      accessorKey: "source",
      sortable: true,
      cell: (item) => (
        <span
          className={cn(
            "px-2 py-0.5 rounded text-[10px] font-bold uppercase",
            item.is_uploaded
              ? "bg-purple-950/60 text-purple-300 border border-purple-500/30"
              : "bg-cyan-950/60 text-cyan-300 border border-cyan-500/30"
          )}
        >
          {item.is_uploaded ? "UPLOADED" : item.source || "BUILT-IN"}
        </span>
      ),
    },
    {
      header: "TOTAL TOKENS",
      accessorKey: "total_tokens",
      sortable: true,
      cell: (item) => (
        <span className="font-mono-num font-bold text-slate-100">
          {item.total_tokens ? formatCompactNumber(item.total_tokens) : "—"}
        </span>
      ),
    },
    {
      header: "SIZE (MB)",
      accessorKey: "size_mb",
      sortable: true,
      cell: (item) => (
        <span className="font-mono-num text-slate-300">
          {item.size_mb ? `${item.size_mb.toFixed(1)} MB` : "—"}
        </span>
      ),
    },
    {
      header: "STATUS",
      accessorKey: "status",
      cell: (item) => (
        <span className="inline-flex items-center gap-1 text-[10px] text-emerald-400 font-bold uppercase">
          <CheckCircle2 className="w-3 h-3" /> READY
        </span>
      ),
    },
    {
      header: "ACTIONS",
      cell: (item) => (
        <div className="flex items-center gap-1.5">
          <button
            onClick={async () => {
              setTokenizingDatasetId(item.id);
              try {
                const res = await api.tokenizeDataset(item.id);
                setUploadFeedback(
                  `Dataset "${item.name}" tokenized: ${(res.token_count || res.train_tokens || 0).toLocaleString()} tokens cached.`
                );
                await fetchDatasets();
                setTimeout(() => setUploadFeedback(null), 5000);
              } catch (e: any) {
                alert(`Tokenization error: ${e.message}`);
              } finally {
                setTokenizingDatasetId(null);
              }
            }}
            disabled={tokenizingDatasetId === item.id}
            className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-cyan-950/60 border border-cyan-500/40 hover:bg-cyan-900/60 text-xs font-mono text-cyan-300 transition-colors disabled:opacity-50"
            title="Pre-tokenize and cache to disk"
          >
            {tokenizingDatasetId === item.id ? (
              <RefreshCw className="w-3 h-3 animate-spin text-cyan-400" />
            ) : (
              <Sparkles className="w-3 h-3 text-cyan-400" />
            )}
            TOKENIZE
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
            className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-slate-900 border border-slate-700 hover:border-cyan-400 text-xs font-mono text-slate-200 transition-colors"
          >
            <Eye className="w-3 h-3 text-cyan-400" /> INSPECT
          </button>
          <button
            onClick={() => setDatasetToDelete(item)}
            className="inline-flex items-center gap-1 p-1 rounded bg-rose-950/40 border border-rose-800/80 hover:bg-rose-900/60 text-rose-300 transition-colors"
            title="Delete Dataset"
          >
            <Trash2 className="w-3 h-3 text-rose-400" />
          </button>
        </div>
      ),
    },
  ];

  return (
    <div className="space-y-4 font-mono text-xs">
      {/* Upload Drop Zone */}
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={cn(
          "glass-panel border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-300 group select-none",
          isUploading
            ? "border-cyan-400 bg-cyan-950/20"
            : "border-slate-700/80 hover:border-cyan-400/60 hover:bg-slate-900/40"
        )}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".txt,.json,.jsonl"
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              handleFileUpload(e.target.files[0]);
            }
          }}
        />
        <div className="p-3 bg-cyan-950/60 rounded-full border border-cyan-500/40 text-cyan-400 mb-3 shadow-[0_0_15px_rgba(0,240,255,0.2)] group-hover:scale-105 transition-transform">
          <UploadCloud className="w-6 h-6" />
        </div>
        <h4 className="text-sm font-bold text-slate-100 uppercase tracking-wider mb-1">
          DRAG & DROP TRAINING DATASET
        </h4>
        <p className="text-xs text-slate-400 max-w-md mb-2">
          Upload raw or formatted <span className="text-cyan-400">.txt</span>,{" "}
          <span className="text-cyan-400">.jsonl</span>, or{" "}
          <span className="text-cyan-400">.json</span> files. Supports up to 500MB with
          automatic pre-tokenization caching.
        </p>
        <span className="px-3 py-1 rounded bg-slate-900 border border-slate-800 text-[10px] text-slate-400">
          CLICK TO BROWSE LOCAL FILES
        </span>
      </div>

      {uploadFeedback && (
        <div className="p-3 rounded-lg bg-emerald-950/80 border border-emerald-500/40 text-emerald-300 font-bold flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4" /> {uploadFeedback}
        </div>
      )}

      {/* Dataset Table */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider flex items-center gap-1.5">
            <Database className="w-3.5 h-3.5 text-cyan-400" /> REGISTERED DATASETS ({datasets.length})
          </h3>
          <button
            onClick={fetchDatasets}
            className="p-1.5 rounded bg-slate-900 border border-slate-800 text-slate-400 hover:text-cyan-400"
            title="Refresh Datasets"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>

        <DataTable
          data={datasets}
          columns={columns}
          keyExtractor={(d) => d.id}
          searchPlaceholder="Search datasets by name or source..."
        />
      </div>

      {/* Tokenizer Live Inspector */}
      <div className="glass-panel p-4 rounded-lg space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800">
          <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-cyan-400" /> TOKENIZER & VOCABULARY INSPECTOR (VOCAB 12,288)
          </h3>
          <button
            onClick={handleInspectTokenizer}
            disabled={isTokenizing}
            className="px-3 py-1 rounded bg-cyan-600 hover:bg-cyan-500 text-white font-bold transition-all shadow-[0_0_10px_rgba(0,240,255,0.3)] text-xs"
          >
            {isTokenizing ? "TOKENIZING..." : "INSPECT TOKENS"}
          </button>
        </div>

        <textarea
          rows={3}
          value={tokenizerText}
          onChange={(e) => setTokenizerText(e.target.value)}
          placeholder="Enter text to test BPE tokenization..."
          className="w-full bg-slate-900/80 border border-slate-800 rounded p-2.5 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-400"
        />

        {tokenizerResult && (
          <div className="space-y-2 pt-1">
            <div className="flex flex-wrap items-center gap-4 text-xs font-mono text-slate-400">
              <span>Tokens: <strong className="text-cyan-400 font-mono-num">{tokenizerResult.token_count || 0}</strong></span>
              <span>Characters: <strong className="text-slate-200 font-mono-num">{tokenizerResult.character_count || 0}</strong></span>
              <span>Ratio: <strong className="text-indigo-400 font-mono-num">{tokenizerResult.compression_ratio ? tokenizerResult.compression_ratio.toFixed(2) : "—"}</strong></span>
            </div>

            {/* Token Badges */}
            <div className="p-3 bg-slate-950/70 border border-slate-800 rounded flex flex-wrap gap-1.5 max-h-48 overflow-y-auto">
              {(tokenizerResult.tokens || []).map((t: any, idx: number) => {
                const colors = [
                  "bg-cyan-950/60 text-cyan-300 border-cyan-500/30",
                  "bg-sky-950/60 text-sky-300 border-sky-500/30",
                  "bg-indigo-950/60 text-indigo-300 border-indigo-500/30",
                  "bg-purple-950/60 text-purple-300 border-purple-500/30",
                  "bg-emerald-950/60 text-emerald-300 border-emerald-500/30",
                ];
                const theme = colors[idx % colors.length];
                return (
                  <span
                    key={idx}
                    className={cn(
                      "px-2 py-0.5 rounded text-[11px] font-mono border flex items-center gap-1",
                      theme
                    )}
                    title={`Token ID: ${t.id || t}`}
                  >
                    <span>{typeof t === "object" ? t.text : String(t)}</span>
                    {t.id !== undefined && (
                      <span className="text-[9px] opacity-60">#{t.id}</span>
                    )}
                  </span>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Dataset Details Modal */}
      {selectedDataset && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm">
          <div className="glass-panel border border-slate-700 rounded-xl p-5 max-w-lg w-full space-y-4 shadow-2xl">
            <div className="flex justify-between items-center pb-2 border-b border-slate-800">
              <h3 className="text-sm font-bold text-slate-100 uppercase">
                {selectedDataset.name || "Dataset Details"}
              </h3>
              <button
                onClick={() => setSelectedDataset(null)}
                className="text-slate-400 hover:text-slate-200"
              >
                ✕
              </button>
            </div>

            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-slate-400">File Path:</span>
                <span className="text-slate-200 truncate max-w-xs">{selectedDataset.file_path || selectedDataset.path || "—"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Total Tokens:</span>
                <span className="text-cyan-400 font-bold">{(selectedDataset.total_tokens || selectedDataset.token_count)?.toLocaleString() || "—"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">File Size:</span>
                <span className="text-slate-200">{selectedDataset.size_mb ? `${selectedDataset.size_mb.toFixed(2)} MB` : (selectedDataset.file_size_bytes ? `${(selectedDataset.file_size_bytes / (1024 * 1024)).toFixed(2)} MB` : "—")}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Cached Tokens:</span>
                <span className="text-emerald-400 font-bold">{selectedDataset.cached ? "YES (.pt cache ready)" : "NO"}</span>
              </div>
            </div>

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setSelectedDataset(null)}
                className="px-4 py-1.5 rounded bg-slate-900 border border-slate-800 text-slate-300 hover:bg-slate-800 text-xs font-mono"
              >
                CLOSE
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Confirm Dataset Deletion Dialog */}
      <ConfirmDialog
        isOpen={!!datasetToDelete}
        title="DELETE DATASET"
        message={`Are you sure you want to delete "${datasetToDelete?.name}" (${datasetToDelete?.id})? This will permanently delete uploaded source files and pre-tokenized cache files from disk.`}
        confirmLabel="DELETE DATASET"
        cancelLabel="CANCEL"
        variant="danger"
        onConfirm={handleDeleteDataset}
        onCancel={() => setDatasetToDelete(null)}
      />
    </div>
  );
};
