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
  Filter,
} from "lucide-react";
import { cn } from "@/lib/utils";

export const DatasetsView: React.FC = () => {
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [tokenizingDatasetId, setTokenizingDatasetId] = useState<string | null>(null);
  const [curatingDatasetId, setCuratingDatasetId] = useState<string | null>(null);
  const [uploadFeedback, setUploadFeedback] = useState<string | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<any | null>(null);
  const [datasetToDelete, setDatasetToDelete] = useState<DatasetItem | null>(null);

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
    setUploadFeedback(`Uploading ${file.name}...`);
    try {
      const res = await api.uploadDataset(file);
      setUploadFeedback(`Dataset ${file.name} registered. Tokens: ${res.total_tokens || "Pre-tokenized"}`);
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
      cell: () => (
        <span className="inline-flex items-center gap-1 text-[10px] text-[#4ade80] font-mono">
          <CheckCircle2 className="w-3 h-3" /> Ready
        </span>
      ),
    },
    {
      header: "ACTIONS",
      cell: (item) => (
        <div className="flex items-center gap-1">
          <button
            onClick={async () => {
              setTokenizingDatasetId(item.id);
              try {
                const res = await api.tokenizeDataset(item.id);
                setUploadFeedback(
                  `Dataset "${item.name}" tokenized: ${(res.token_count || res.train_tokens || 0).toLocaleString()} tokens.`
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
            className="btn btn-secondary btn-sm px-2 text-[10px]"
            title="Pre-tokenize dataset"
          >
            {tokenizingDatasetId === item.id ? (
              <RefreshCw className="w-3 h-3 animate-spin" />
            ) : (
              <Sparkles className="w-3 h-3 text-[#737373]" />
            )}
            Tokenize
          </button>
          <button
            onClick={async () => {
              setCuratingDatasetId(item.id);
              try {
                const res = await api.curateDataset(item.id, { jaccard_threshold: 0.80, max_samples: 5000 });
                setUploadFeedback(
                  `MinHash LSH Dedup: ${res.duplicates_purged ?? 0} duplicates removed (${res.accepted_samples_count ?? res.accepted_count ?? "all"} unique kept).`
                );
                await fetchDatasets();
                setTimeout(() => setUploadFeedback(null), 6000);
              } catch (e: any) {
                alert(`MinHash Deduplication failed: ${e.message}`);
              } finally {
                setCuratingDatasetId(null);
              }
            }}
            disabled={curatingDatasetId === item.id}
            className="btn btn-secondary btn-sm px-2 text-[10px]"
            title="Run MinHash LSH deduplication (0.80 Jaccard threshold)"
          >
            {curatingDatasetId === item.id ? (
              <RefreshCw className="w-3 h-3 animate-spin" />
            ) : (
              <Filter className="w-3 h-3 text-[#737373]" />
            )}
            MinHash
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
          accept=".txt,.json,.jsonl,.csv,.parquet,.arrow"
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              handleFileUpload(e.target.files[0]);
            }
          }}
        />
        <div className="p-2.5 bg-[#1B1B1B] rounded-full border border-[#333333] text-[#A3A3A3] mb-2">
          <UploadCloud className="w-5 h-5" />
        </div>
        <h4 className="text-xs font-semibold text-[#F2F2F2] uppercase tracking-wide mb-0.5">
          Upload Training Dataset
        </h4>
        <p className="text-[11px] text-[#737373] max-w-md mb-1.5 leading-relaxed">
          Supports .parquet, .arrow, .jsonl, .txt, .csv with zero-copy memory mapping, MinHash deduplication, and Jinja2 chat templates.
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

      {/* Dataset Table */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <h3 className="panel-title">
            <Database className="w-3.5 h-3.5 text-[#737373]" /> Datasets ({datasets.length})
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
          data={datasets}
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
