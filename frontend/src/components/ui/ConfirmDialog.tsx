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
      ? "bg-[#FF657A] hover:bg-[#FF657A]/90 text-white shadow-[0_0_12px_rgba(255,101,122,0.35)]"
      : variant === "warning"
      ? "bg-[#F2B85B] hover:bg-[#F2B85B]/90 text-[#090D14] shadow-[0_0_12px_rgba(242,184,91,0.35)]"
      : "bg-[#45CFFF] hover:bg-[#80DEFF] text-[#090D14] shadow-[0_0_12px_rgba(69,207,255,0.35)]";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-in fade-in duration-150">
      <div className="bg-[#0E1420] border border-[#384C66] rounded-xl p-5 max-w-md w-full shadow-2xl relative space-y-4">
        <div className="flex items-start gap-3">
          <div className="p-2 rounded-lg bg-[#131C29] border border-[#273548] text-[#FF657A]">
            <AlertCircle className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-mono text-sm font-bold text-[#ECF3FC] uppercase tracking-wide">
              {title}
            </h3>
            <p className="mt-1 text-xs text-[#91A0B4] leading-relaxed font-mono">
              {message}
            </p>
          </div>
        </div>

        <div className="flex items-center justify-end gap-2.5 pt-2 border-t border-[#273548]">
          <button
            onClick={onCancel}
            className="px-3.5 py-1.5 rounded text-xs font-mono text-[#91A0B4] hover:bg-[#182333] hover:text-[#ECF3FC] transition-colors"
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
