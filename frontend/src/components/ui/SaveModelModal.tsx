"use client";

import React, { useState, useEffect } from "react";
import {
  Bookmark,
  CheckCircle2,
  FileCheck,
  BarChart3,
  Layers,
  Sparkles,
  X,
  AlertCircle,
  HardDrive,
} from "lucide-react";
import { Button } from "./Button";
import { api } from "@/services/api";

interface SaveModelModalProps {
  isOpen: boolean;
  onClose: () => void;
  defaultModelId?: string;
  onSaved?: (result: any) => void;
}

export const SaveModelModal: React.FC<SaveModelModalProps> = ({
  isOpen,
  onClose,
  defaultModelId = "rocanon_48_5m",
  onSaved,
}) => {
  const [modelName, setModelName] = useState(defaultModelId);
  const [saveMode, setSaveMode] = useState<"best" | "current">("best");
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedResult, setSavedResult] = useState<any | null>(null);

  useEffect(() => {
    if (isOpen) {
      setModelName(defaultModelId);
      setError(null);
      setSavedResult(null);
    }
  }, [isOpen, defaultModelId]);

  if (!isOpen) return null;

  const handleSave = async () => {
    setIsSaving(true);
    setError(null);
    try {
      const res = await api.saveLocalModel({
        model_id: defaultModelId,
        custom_name: modelName.trim() || defaultModelId,
        save_mode: saveMode,
      });
      if (res.success && res.package) {
        setSavedResult(res.package);
        if (onSaved) onSaved(res.package);
      } else {
        throw new Error("Failed to save local model package.");
      }
    } catch (e: any) {
      setError(e?.message || "Failed to save model locally.");
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-in fade-in duration-150 font-mono">
      <div className="glass-panel-elevated border border-[#384C66] rounded-xl w-full max-w-lg p-5 shadow-2xl relative">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1 rounded-md text-[#91A0B4] hover:text-[#ECF3FC] hover:bg-[#182333] transition-colors"
        >
          <X className="w-4 h-4" />
        </button>

        {/* Modal Header */}
        <div className="flex items-center gap-2.5 pb-3 border-b border-[#273548]">
          <div className="w-8 h-8 rounded-lg bg-[#131C29] border border-[#45CFFF]/40 flex items-center justify-center text-[#45CFFF]">
            <HardDrive className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-[#ECF3FC]">Save Model Locally</h2>
            <p className="text-[11px] text-[#91A0B4]">
              Persists keep-training .pt, stripped inference weights & dynamics charts
            </p>
          </div>
        </div>

        {/* Content */}
        {!savedResult ? (
          <div className="py-4 space-y-4 text-xs">
            {error && (
              <div className="p-2.5 rounded-lg bg-[#FF657A]/10 border border-[#FF657A]/30 text-[#FF657A] flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {/* Model Name Input */}
            <div>
              <label className="block text-[11px] uppercase font-semibold text-[#91A0B4] mb-1.5">
                Model Name / Alias
              </label>
              <input
                type="text"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                placeholder="e.g. rocanon_48_5m_best"
                className="w-full px-3 py-2 rounded-md bg-[#131C29] border border-[#273548] text-[#ECF3FC] focus:border-[#45CFFF] focus:outline-none transition-colors"
              />
              <span className="text-[10px] text-[#5E7188] mt-1 block">
                Output will be saved in: <code className="text-[#80DEFF]">models/{modelName.trim() || defaultModelId}/</code>
              </span>
            </div>

            {/* Checkpoint Source Selection */}
            <div>
              <label className="block text-[11px] uppercase font-semibold text-[#91A0B4] mb-1.5">
                Target Checkpoint
              </label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setSaveMode("best")}
                  className={`p-2.5 rounded-lg border text-left transition-all ${
                    saveMode === "best"
                      ? "bg-[#182333] border-[#35D49A] text-[#ECF3FC] shadow-[0_0_10px_rgba(53,212,154,0.15)]"
                      : "bg-[#131C29] border-[#273548] text-[#91A0B4] hover:border-[#384C66]"
                  }`}
                >
                  <div className="flex items-center gap-1.5 font-bold text-xs">
                    <Sparkles className="w-3.5 h-3.5 text-[#35D49A]" />
                    <span>Best Checkpoint</span>
                  </div>
                  <span className="text-[10px] text-[#5E7188] block mt-1">
                    Lowest validation loss weights
                  </span>
                </button>

                <button
                  type="button"
                  onClick={() => setSaveMode("current")}
                  className={`p-2.5 rounded-lg border text-left transition-all ${
                    saveMode === "current"
                      ? "bg-[#182333] border-[#45CFFF] text-[#ECF3FC] shadow-[0_0_10px_rgba(69,207,255,0.15)]"
                      : "bg-[#131C29] border-[#273548] text-[#91A0B4] hover:border-[#384C66]"
                  }`}
                >
                  <div className="flex items-center gap-1.5 font-bold text-xs">
                    <Layers className="w-3.5 h-3.5 text-[#45CFFF]" />
                    <span>Current Step State</span>
                  </div>
                  <span className="text-[10px] text-[#5E7188] block mt-1">
                    Most recent in-memory weights
                  </span>
                </button>
              </div>
            </div>

            {/* Package Contents Explanation */}
            <div className="p-3 rounded-lg bg-[#090D14]/80 border border-[#273548] space-y-2 text-[11px]">
              <div className="font-semibold text-[#ECF3FC] flex items-center gap-1.5">
                <FileCheck className="w-3.5 h-3.5 text-[#45CFFF]" /> What will be saved:
              </div>
              <ul className="space-y-1 text-[#91A0B4]">
                <li className="flex items-start gap-1.5">
                  <span className="text-[#45CFFF] font-bold">•</span>
                  <span>
                    <strong className="text-[#ECF3FC]">{modelName}_full.pt:</strong> Complete weights with Muon & AdamW optimizer moments to continue training anytime.
                  </span>
                </li>
                <li className="flex items-start gap-1.5">
                  <span className="text-[#35D49A] font-bold">•</span>
                  <span>
                    <strong className="text-[#ECF3FC]">{modelName}_stripped.pt:</strong> Lightweight inference weights with zero optimizer states (~50% smaller).
                  </span>
                </li>
                <li className="flex items-start gap-1.5">
                  <span className="text-[#F2B85B] font-bold">•</span>
                  <span>
                    <strong className="text-[#ECF3FC]">val_loss_curve.png & training_dynamics.png:</strong> Publication-grade convergence and performance charts.
                  </span>
                </li>
              </ul>
            </div>

            {/* Footer Buttons */}
            <div className="flex justify-end gap-2 pt-2 border-t border-[#273548]">
              <Button variant="ghost" size="sm" onClick={onClose} disabled={isSaving}>
                Cancel
              </Button>
              <Button
                variant="primary"
                size="sm"
                onClick={handleSave}
                loading={isSaving}
                icon={<Bookmark className="w-3.5 h-3.5 fill-current" />}
              >
                Save Model Locally
              </Button>
            </div>
          </div>
        ) : (
          /* Success Result View */
          <div className="py-4 space-y-3 text-xs">
            <div className="p-3 rounded-lg bg-[#35D49A]/10 border border-[#35D49A]/30 text-[#35D49A] flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 shrink-0" />
              <div>
                <span className="font-bold block">Model Package Successfully Saved!</span>
                <span className="text-[10px] text-[#ECF3FC]">
                  Saved in <code className="text-[#80DEFF]">models/{savedResult.name}/</code>
                </span>
              </div>
            </div>

            <div className="p-3 rounded-lg bg-[#131C29] border border-[#273548] space-y-2 font-mono text-[11px]">
              <div className="flex justify-between items-center text-[#91A0B4]">
                <span>Full Keep-Training Weights:</span>
                <span className="text-[#ECF3FC] font-bold">
                  {savedResult.files?.full_weights?.file_name} ({savedResult.files?.full_weights?.size_mb} MB)
                </span>
              </div>
              <div className="flex justify-between items-center text-[#91A0B4]">
                <span>Stripped Inference Weights:</span>
                <span className="text-[#35D49A] font-bold">
                  {savedResult.files?.stripped_weights?.file_name} ({savedResult.files?.stripped_weights?.size_mb} MB, -{savedResult.files?.stripped_weights?.reduction_pct}%)
                </span>
              </div>
              <div className="flex justify-between items-center text-[#91A0B4]">
                <span>Convergence Charts:</span>
                <span className="text-[#80DEFF]">
                  {Object.keys(savedResult.files?.charts || {}).length} figures generated
                </span>
              </div>
              {savedResult.val_loss !== null && (
                <div className="flex justify-between items-center text-[#91A0B4] pt-1 border-t border-[#273548]">
                  <span>Validation Loss at Best Step {savedResult.step}:</span>
                  <span className="text-[#35D49A] font-bold font-mono-num">
                    {savedResult.val_loss?.toFixed(4)}
                  </span>
                </div>
              )}
            </div>

            <div className="flex justify-end pt-2">
              <Button variant="primary" size="sm" onClick={onClose}>
                Done
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
