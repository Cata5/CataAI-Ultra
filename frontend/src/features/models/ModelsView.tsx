"use client";

import React, { useCallback, useEffect, useState } from "react";
import { AlertTriangle, Boxes, Download, RefreshCw } from "lucide-react";
import { api } from "@/services/api";

interface ModelFile {
  name: string;
  bytes: number;
  has_sha256: boolean;
}

interface SavedModel {
  id: string;
  name: string;
  saved_at?: number;
  step?: number;
  val_loss?: number;
  total_parameters?: number;
  files: ModelFile[];
}

const mb = (bytes: number) => `${(bytes / (1024 * 1024)).toFixed(1)} MB`;

export const ModelsView: React.FC = () => {
  const [models, setModels] = useState<SavedModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [exporting, setExporting] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.listModels();
      setModels(response.models || []);
    } catch (loadError: any) {
      setError(String(loadError?.message || loadError));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const exportGguf = async (model: SavedModel, quant: "Q8_0" | "F16", arch: "auto" | "llama" | "rocanon" = "auto") => {
    setExporting(`${model.id}:${arch}:${quant}`);
    setError(null);
    setResult(null);
    try {
      const response = await api.exportGguf(model.id, quant, arch);
      setResult(response);
      await load();
    } catch (exportError: any) {
      setError(String(exportError?.message || exportError));
    } finally {
      setExporting(null);
    }
  };

  return (
    <div className="space-y-4 font-mono text-xs">
      <div className="panel p-3.5 bg-[#202020] border-[#333333] flex items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <Boxes className="w-4 h-4 text-[#A3A3A3]" />
          <div>
            <h3 className="font-semibold text-[#F2F2F2] uppercase tracking-wider text-xs">Saved Models</h3>
            <p className="text-[10px] text-[#737373]">Local packages with real weight checksums. Export lossless GGUF or resume from the full package.</p>
          </div>
        </div>
        <button
          onClick={load}
          disabled={loading}
          className="px-2.5 py-1.5 rounded bg-[#262626] border border-[#333333] hover:border-[#525252] text-[#A3A3A3] hover:text-[#F2F2F2] transition-colors flex items-center gap-1.5 text-[10px]"
        >
          <RefreshCw className={`w-3 h-3 ${loading ? "animate-spin" : ""}`} /> Refresh
        </button>
      </div>

      {error && (
        <div className="p-3 rounded bg-rose-950/40 border border-rose-500/40 text-rose-300 flex gap-2">
          <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" /> {error}
        </div>
      )}

      {result && (
        <div className="panel p-3 bg-[#202020] border-[#333333] space-y-1">
          <div className="text-[#4ade80] font-semibold">GGUF WRITTEN</div>
          <div className="text-[#F2F2F2] break-all">{result.path}</div>
          <div className="text-[#A3A3A3]">{mb(result.bytes)} · {result.tensor_count} tensors · {result.quantization} · {result.architecture}</div>
          <div className={result.runtime_compatible ? "text-[#4ade80]" : "text-amber-300"}>
            {result.runtime_compatible ? "Runtime-compatible layout (llama.cpp)" : "Custom container (not stock llama.cpp)"}
          </div>
          <div className="text-[#737373] break-all">sha256 {result.sha256}</div>
          <div className="text-amber-300/90 text-[10px] leading-relaxed">{result.note}</div>
        </div>
      )}

      {loading && models.length === 0 ? (
        <div className="panel p-6 text-center text-[#737373]">Loading saved models…</div>
      ) : models.length === 0 ? (
        <div className="panel p-6 text-center text-[#737373] space-y-1">
          <div>No saved model packages yet.</div>
          <div className="text-[10px]">Save a checkpoint as a local model package from the training view first.</div>
        </div>
      ) : (
        <div className="grid gap-3 lg:grid-cols-2">
          {models.map((model) => (
            <div key={model.id} className="panel p-3.5 bg-[#202020] border-[#333333] space-y-3">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="font-semibold text-[#F2F2F2] truncate">{model.name}</div>
                  <div className="text-[10px] text-[#737373] truncate">{model.id}</div>
                </div>
                <span className="text-[9px] px-1.5 py-0.5 rounded bg-[#262626] border border-[#333333] text-[#A3A3A3]">
                  Step {model.step?.toLocaleString() || 0}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-2 text-[10px]">
                <div><div className="text-[#737373]">PARAMS</div><div className="text-[#F2F2F2]">{model.total_parameters ? (model.total_parameters / 1e6).toFixed(2) + "M" : "—"}</div></div>
                <div><div className="text-[#737373]">VAL LOSS</div><div className="text-[#4ade80]">{model.val_loss != null ? Number(model.val_loss).toFixed(4) : "—"}</div></div>
                <div><div className="text-[#737373]">FILES</div><div className="text-[#F2F2F2]">{model.files.length}</div></div>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {model.files.map((file) => (
                  <span key={file.name} title={file.has_sha256 ? "SHA-256 sidecar present" : "No checksum sidecar"} className="text-[9px] px-1.5 py-0.5 rounded bg-[#1B1B1B] border border-[#333333] text-[#A3A3A3]">
                    {file.name} · {mb(file.bytes)}
                  </span>
                ))}
              </div>
              <div className="flex flex-wrap items-center gap-2">
                {(["Q8_0", "F16"] as const).map((quant) => (
                  <button
                    key={`llama-${quant}`}
                    onClick={() => exportGguf(model, quant, "llama")}
                    disabled={exporting !== null}
                    title="Requires the Llama-Compatible preset; Canon checkpoints refuse honestly"
                    className="px-2.5 py-1.5 rounded bg-[#262626] border border-[#333333] hover:border-[#525252] text-[#A3A3A3] hover:text-[#F2F2F2] transition-colors flex items-center gap-1.5 text-[10px] disabled:opacity-50"
                  >
                    <Download className="w-3 h-3" />
                    {exporting === `${model.id}:llama:${quant}` ? "Exporting…" : `llama.cpp ${quant}`}
                  </button>
                ))}
                {(["Q8_0", "F16"] as const).map((quant) => (
                  <button
                    key={`rocanon-${quant}`}
                    onClick={() => exportGguf(model, quant, "rocanon")}
                    disabled={exporting !== null}
                    title="Lossless custom RoCanon container (not runnable in stock llama.cpp)"
                    className="px-2.5 py-1.5 rounded bg-[#1B1B1B] border border-[#333333] hover:border-[#525252] text-[#737373] hover:text-[#F2F2F2] transition-colors flex items-center gap-1.5 text-[10px] disabled:opacity-50"
                  >
                    <Download className="w-3 h-3" />
                    {exporting === `${model.id}:rocanon:${quant}` ? "Exporting…" : `RoCanon ${quant}`}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
