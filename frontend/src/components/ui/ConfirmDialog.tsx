"use client";

import React from "react";
import { AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface ConfirmDialogProps {
  isOpen: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: "danger" | "warning" | "info";
  onConfirm: () => void;
  onCancel: () => void;
}

export const ConfirmDialog: React.FC<ConfirmDialogProps> = ({
  isOpen,
  title,
  message,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  variant = "danger",
  onConfirm,
  onCancel,
}) => {
  if (!isOpen) return null;

  const confirmBtnClass =
    variant === "danger"
      ? "bg-rose-600 hover:bg-rose-500 text-white shadow-[0_0_12px_rgba(239,68,68,0.4)]"
      : variant === "warning"
      ? "bg-amber-600 hover:bg-amber-500 text-white shadow-[0_0_12px_rgba(245,158,11,0.4)]"
      : "bg-cyan-600 hover:bg-cyan-500 text-white shadow-[0_0_12px_rgba(0,240,255,0.4)]";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="glass-panel border border-slate-700/80 rounded-xl p-5 max-w-md w-full shadow-2xl relative space-y-4">
        <div className="flex items-start gap-3">
          <div className="p-2 rounded-lg bg-slate-900 border border-slate-800 text-rose-400">
            <AlertCircle className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-mono text-sm font-bold text-slate-100 uppercase tracking-wide">
              {title}
            </h3>
            <p className="mt-1 text-xs text-slate-300 leading-relaxed font-mono">
              {message}
            </p>
          </div>
        </div>

        <div className="flex items-center justify-end gap-2.5 pt-2 border-t border-slate-800">
          <button
            onClick={onCancel}
            className="px-3.5 py-1.5 rounded text-xs font-mono text-slate-300 hover:bg-slate-800/80 transition-colors"
          >
            {cancelLabel}
          </button>
          <button
            onClick={onConfirm}
            className={cn("px-4 py-1.5 rounded text-xs font-mono font-bold transition-all", confirmBtnClass)}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
};
