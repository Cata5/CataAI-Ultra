"use client";

import React, { useState, useEffect } from "react";
import {
  Bookmark,
  CheckCircle2,
  FileCheck,
  X,
  AlertCircle,
  HardDrive,
  Sparkles,
  Layers,
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
  defaultModelId = "rocanon_48m",
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
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm font-mono">
      <div className="bg-[#202020] border border-[#333333] rounded-lg w-full max-w-md p-4 shadow-xl relative">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-3 right-3 p-1 rounded text-[#737373] hover:text-[#F2F2F2] hover:bg-[#262626] transition-colors"
        >
          <X className="w-3.5 h-3.5" />
        </button>

        {/* Modal Header */}
        <div className="flex items-center gap-2 pb-2.5 border-b border-[#262626]">
          <div className="w-7 h-7 rounded bg-[#262626] border border-[#333333] flex items-center justify-center text-[#D4D4D4]">
            <HardDrive className="w-3.5 h-3.5" />
          </div>
          <div>
            <h2 className="text-xs font-semibold text-[#F2F2F2]">Save Model Locally</h2>
            <p className="text-[10px] text-[#737373]">
              Persists keep-training .pt, stripped weights & charts
            </p>
          </div>
        </div>

        {/* Content */}
        {!savedResult ? (
          <div className="py-3 space-y-3 text-xs">
            {error && (
              <div className="p-2 rounded bg-[rgba(248,113,113,0.08)] border border-[rgba(248,113,113,0.25)] text-[#f87171] flex items-center gap-1.5 text-[11px]">
                <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {/* Model Name Input */}
            <div>
              <label className="block text-[10px] uppercase font-medium text-[#737373] mb-1">
                Model Name / Alias
              </label>
              <input
                type="text"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                placeholder="e.g. rocanon_48m_best"
                className="input"
              />
              <span className="text-[10px] text-[#525252] mt-0.5 block">
                Saved in: <code className="text-[#A3A3A3]">models/{modelName.trim() || defaultModelId}/</code>
              </span>
            </div>

            {/* Checkpoint Source Selection */}
            <div>
              <label className="block text-[10px] uppercase font-medium text-[#737373] mb-1">
                Target Checkpoint
              </label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setSaveMode("best")}
                  className={`p-2 rounded border text-left transition-colors ${
                    saveMode === "best"
                      ? "bg-[#262626] border-[#D4D4D4] text-[#F2F2F2]"
                      : "bg-[#1B1B1B] border-[#262626] text-[#737373] hover:border-[#333333]"
                  }`}
                >
                  <div className="flex items-center gap-1.5 font-medium text-xs">
                    <Sparkles className="w-3 h-3 text-[#A3A3A3]" />
                    <span>Best Checkpoint</span>
                  </div>
                  <span className="text-[10px] text-[#525252] block mt-0.5">
                    Lowest val loss
                  </span>
                </button>

                <button
                  type="button"
                  onClick={() => setSaveMode("current")}
                  className={`p-2 rounded border text-left transition-colors ${
                    saveMode === "current"
                      ? "bg-[#262626] border-[#D4D4D4] text-[#F2F2F2]"
                      : "bg-[#1B1B1B] border-[#262626] text-[#737373] hover:border-[#333333]"
                  }`}
                >
                  <div className="flex items-center gap-1.5 font-medium text-xs">
                    <Layers className="w-3 h-3 text-[#A3A3A3]" />
                    <span>Current Step</span>
                  </div>
                  <span className="text-[10px] text-[#525252] block mt-0.5">
                    Active state
                  </span>
                </button>
              </div>
            </div>

            {/* Package Contents Explanation */}
            <div className="p-2.5 rounded bg-[#1B1B1B] border border-[#262626] space-y-1.5 text-[11px]">
              <div className="font-medium text-[#A3A3A3] flex items-center gap-1.5">
                <FileCheck className="w-3.5 h-3.5 text-[#737373]" /> Package contents:
              </div>
              <ul className="space-y-1 text-[#737373] text-[10px]">
                <li className="flex items-start gap-1.5">
                  <span className="text-[#A3A3A3]">•</span>
                  <span>
                    <strong className="text-[#F2F2F2]">{modelName}_full.pt:</strong> Complete weights with optimizer moments.
                  </span>
                </li>
                <li className="flex items-start gap-1.5">
                  <span className="text-[#A3A3A3]">•</span>
                  <span>
                    <strong className="text-[#F2F2F2]">{modelName}_stripped.pt:</strong> Lightweight inference weights.
                  </span>
                </li>
                <li className="flex items-start gap-1.5">
                  <span className="text-[#A3A3A3]">•</span>
                  <span>
                    <strong className="text-[#F2F2F2]">training_dynamics.png:</strong> Loss & learning rate curves.
                  </span>
                </li>
              </ul>
            </div>

            {/* Footer Buttons */}
            <div className="flex justify-end gap-2 pt-2 border-t border-[#262626]">
              <Button variant="ghost" size="sm" onClick={onClose} disabled={isSaving}>
                Cancel
              </Button>
              <Button
                variant="primary"
                size="sm"
                onClick={handleSave}
                loading={isSaving}
                icon={<Bookmark className="w-3 h-3 fill-current" />}
              >
                Save Model
              </Button>
            </div>
          </div>
        ) : (
          /* Success Result View */
          <div className="py-3 space-y-2.5 text-xs">
            <div className="p-2.5 rounded bg-[rgba(74,222,128,0.08)] border border-[rgba(74,222,128,0.25)] text-[#4ade80] flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 shrink-0" />
              <div>
                <span className="font-semibold block">Model Package Saved!</span>
                <span className="text-[10px] text-[#A3A3A3]">
                  Location: <code className="text-[#F2F2F2]">models/{savedResult.name}/</code>
                </span>
              </div>
            </div>

            <div className="p-2.5 rounded bg-[#1B1B1B] border border-[#262626] space-y-1 font-mono text-[10px]">
              <div className="flex justify-between items-center text-[#737373]">
                <span>Full Weights:</span>
                <span className="text-[#F2F2F2]">
                  {savedResult.files?.full_weights?.file_name} ({savedResult.files?.full_weights?.size_mb} MB)
                </span>
              </div>
              <div className="flex justify-between items-center text-[#737373]">
                <span>Stripped Weights:</span>
                <span className="text-[#4ade80]">
                  {savedResult.files?.stripped_weights?.file_name} ({savedResult.files?.stripped_weights?.size_mb} MB)
                </span>
              </div>
              {savedResult.val_loss !== null && (
                <div className="flex justify-between items-center text-[#737373] pt-1 border-t border-[#262626]">
                  <span>Val Loss at Step {savedResult.step}:</span>
                  <span className="text-[#4ade80] font-mono-num font-medium">
                    {savedResult.val_loss?.toFixed(4)}
                  </span>
                </div>
              )}
            </div>

            <div className="flex justify-end pt-1">
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
