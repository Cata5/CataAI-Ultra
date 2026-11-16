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
          <FileText className="w-4 h-4 text-[#45CFFF] shrink-0" />
          <div className="flex flex-col">
            <span className="font-semibold text-[#ECF3FC]">{item.name}</span>
            <span className="text-[10px] text-[#5E7188] font-mono">{item.id}</span>
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
            "tech-badge",
            item.is_uploaded
              ? "tech-badge-violet"
              : "tech-badge-cyan"
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
        <span className="font-mono-num font-bold text-[#ECF3FC]">
          {item.total_tokens ? formatCompactNumber(item.total_tokens) : "—"}
        </span>
      ),
    },
    {
      header: "SIZE (MB)",
      accessorKey: "size_mb",
      sortable: true,
      cell: (item) => (
        <span className="font-mono-num text-[#91A0B4]">
          {item.size_mb ? `${item.size_mb.toFixed(1)} MB` : "—"}
        </span>
      ),
    },
    {
      header: "STATUS",
      accessorKey: "status",
      cell: (item) => (
        <span className="inline-flex items-center gap-1 text-[10px] text-[#35D49A] font-bold uppercase font-mono">
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
            className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-[#131C29] border border-[#273548] hover:border-[#45CFFF]/50 hover:bg-[#182333] text-xs font-mono text-[#45CFFF] transition-colors disabled:opacity-50"
            title="Pre-tokenize and cache to disk"
          >
            {tokenizingDatasetId === item.id ? (
              <RefreshCw className="w-3 h-3 animate-spin text-[#45CFFF]" />
            ) : (
              <Sparkles className="w-3 h-3 text-[#45CFFF]" />
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
            className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-[#131C29] border border-[#273548] hover:border-[#384C66] text-xs font-mono text-[#ECF3FC] transition-colors"
          >
            <Eye className="w-3 h-3 text-[#45CFFF]" /> INSPECT
          </button>
          <button
            onClick={() => setDatasetToDelete(item)}
            className="inline-flex items-center gap-1 p-1 rounded-md bg-[#FF657A]/10 border border-[#FF657A]/30 hover:bg-[#FF657A]/20 text-[#FF657A] transition-colors"
            title="Delete Dataset"
          >
            <Trash2 className="w-3 h-3 text-[#FF657A]" />
          </button>
        </div>
      ),
    },
  ];

  return (
    <div className="space-y-4 font-mono text-xs max-w-[1600px] mx-auto animate-in fade-in duration-200">
      {/* Upload Drop Zone */}
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={cn(
          "glass-panel border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 group select-none relative overflow-hidden",
          isUploading
            ? "border-[#45CFFF] bg-[#45CFFF]/10"
            : "border-[#273548] hover:border-[#45CFFF]/50 hover:bg-[#182333]"
        )}
      >
        <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#45CFFF]/20 to-transparent pointer-events-none" />

        <input
          ref={fileInputRef}
          type="file"
          accept=".txt,.json,.jsonl,.csv"
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              handleFileUpload(e.target.files[0]);
            }
          }}
        />
        <div className="p-3 bg-[#131C29] rounded-full border border-[#45CFFF]/40 text-[#45CFFF] mb-3 shadow-[0_0_15px_rgba(69,207,255,0.2)] group-hover:scale-105 transition-transform">
          <UploadCloud className="w-6 h-6" />
        </div>
        <h4 className="text-sm font-bold text-[#ECF3FC] uppercase tracking-wider mb-1">
          DRAG & DROP TRAINING DATASET
        </h4>
        <p className="text-xs text-[#91A0B4] max-w-md mb-2">
          Upload raw or formatted <span className="text-[#45CFFF]">.txt</span>,{" "}
          <span className="text-[#45CFFF]">.jsonl</span>, or{" "}
          <span className="text-[#45CFFF]">.csv</span> files. Supports up to 500MB with
          automatic validation and pre-tokenization caching.
        </p>
        <span className="px-3 py-1 rounded-md bg-[#0E1420] border border-[#273548] text-[10px] text-[#5E7188] uppercase tracking-wider font-semibold">
          CLICK TO BROWSE LOCAL FILES
        </span>
      </div>

      {uploadFeedback && (
        <div className="p-3 rounded-md bg-[#35D49A]/15 border border-[#35D49A]/30 text-[#35D49A] font-bold flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4" /> {uploadFeedback}
        </div>
      )}

      {/* Dataset Table */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-bold text-[#ECF3FC] uppercase tracking-wider flex items-center gap-1.5">
            <Database className="w-3.5 h-3.5 text-[#45CFFF]" /> REGISTERED DATASETS ({datasets.length})
          </h3>
          <button
            onClick={fetchDatasets}
            className="p-1.5 rounded-md bg-[#131C29] border border-[#273548] text-[#91A0B4] hover:text-[#45CFFF] transition-colors"
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
      <div className="glass-panel p-4 rounded-lg space-y-3 relative overflow-hidden">
        <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#45CFFF]/15 to-transparent pointer-events-none" />

        <div className="flex items-center justify-between pb-2 border-b border-[#273548]">
          <h3 className="text-xs font-bold text-[#ECF3FC] uppercase tracking-wider flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-[#45CFFF]" /> TOKENIZER & VOCABULARY INSPECTOR (VOCAB 12,288)
          </h3>
          <button
            onClick={handleInspectTokenizer}
            disabled={isTokenizing}
            className="px-3 py-1 rounded-md bg-[#45CFFF] hover:bg-[#80DEFF] text-[#090D14] font-bold transition-all shadow-sm text-xs"
          >
            {isTokenizing ? "TOKENIZING..." : "INSPECT TOKENS"}
          </button>
        </div>

        <textarea
          rows={3}
          value={tokenizerText}
          onChange={(e) => setTokenizerText(e.target.value)}
          placeholder="Enter text to test BPE tokenization..."
          className="w-full bg-[#0E1420] border border-[#273548] rounded-md p-2.5 text-xs font-mono text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF]/50"
        />

        {tokenizerResult && (
          <div className="space-y-2 pt-1">
            <div className="flex flex-wrap items-center gap-4 text-xs font-mono text-[#91A0B4]">
              <span>Tokens: <strong className="text-[#45CFFF] font-mono-num">{tokenizerResult.token_count || 0}</strong></span>
              <span>Characters: <strong className="text-[#ECF3FC] font-mono-num">{tokenizerResult.character_count || 0}</strong></span>
              <span>Ratio: <strong className="text-[#9B8CFF] font-mono-num">{tokenizerResult.compression_ratio ? tokenizerResult.compression_ratio.toFixed(2) : "—"}</strong></span>
            </div>

            {/* Token Badges */}
            <div className="p-3 bg-[#090D14] border border-[#273548] rounded-md flex flex-wrap gap-1.5 max-h-48 overflow-y-auto">
              {(tokenizerResult.tokens || []).map((t: any, idx: number) => {
                return (
                  <span
                    key={idx}
                    className="px-2 py-0.5 rounded text-[11px] font-mono border flex items-center gap-1 bg-[#131C29] border-[#273548] text-[#ECF3FC]"
                    title={`Token ID: ${t.id || t}`}
                  >
                    <span>{typeof t === "object" ? t.text : String(t)}</span>
                    {t.id !== undefined && (
                      <span className="text-[9px] text-[#5E7188]">#{t.id}</span>
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
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
          <div className="bg-[#0E1420] border border-[#384C66] rounded-xl p-5 max-w-lg w-full space-y-4 shadow-2xl relative">
            <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#45CFFF]/30 to-transparent pointer-events-none" />

            <div className="flex justify-between items-center pb-2 border-b border-[#273548]">
              <h3 className="text-sm font-bold text-[#ECF3FC] uppercase tracking-wide">
                {selectedDataset.name || "Dataset Details"}
              </h3>
              <button
                onClick={() => setSelectedDataset(null)}
                className="text-[#91A0B4] hover:text-[#ECF3FC]"
              >
                ✕
              </button>
            </div>

            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-[#5E7188]">File Path:</span>
                <span className="text-[#ECF3FC] truncate max-w-xs">{selectedDataset.file_path || selectedDataset.path || "—"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#5E7188]">Total Tokens:</span>
                <span className="text-[#45CFFF] font-bold font-mono-num">{(selectedDataset.total_tokens || selectedDataset.token_count)?.toLocaleString() || "—"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#5E7188]">File Size:</span>
                <span className="text-[#ECF3FC] font-mono-num">{selectedDataset.size_mb ? `${selectedDataset.size_mb.toFixed(2)} MB` : (selectedDataset.file_size_bytes ? `${(selectedDataset.file_size_bytes / (1024 * 1024)).toFixed(2)} MB` : "—")}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#5E7188]">Cached Tokens:</span>
                <span className="text-[#35D49A] font-bold">{selectedDataset.cached ? "YES (.pt cache ready)" : "NO"}</span>
              </div>
            </div>

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setSelectedDataset(null)}
                className="px-4 py-1.5 rounded-md bg-[#131C29] border border-[#273548] text-[#ECF3FC] hover:bg-[#182333] text-xs font-mono"
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