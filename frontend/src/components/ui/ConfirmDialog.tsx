"use client";

import React, { useEffect, useRef } from "react";
import { AlertCircle, AlertTriangle, Info } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "./Button";

interface ConfirmDialogProps {
  isOpen: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: "danger" | "warning" | "info";
  loading?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

const VARIANT_META = {
  danger: {
    icon: <AlertCircle className="w-4 h-4 text-[#f87171]" />,
    tone: "text-[#f87171]",
    ring: "border-[rgba(248,113,113,0.25)] bg-[rgba(248,113,113,0.06)]",
    button: "danger" as const,
  },
  warning: {
    icon: <AlertTriangle className="w-4 h-4 text-[#facc15]" />,
    tone: "text-[#facc15]",
    ring: "border-[rgba(250,204,21,0.25)] bg-[rgba(250,204,21,0.06)]",
    button: "warning" as const,
  },
  info: {
    icon: <Info className="w-4 h-4 text-[#D4D4D4]" />,
    tone: "text-[#D4D4D4]",
    ring: "border-[#333333] bg-[#262626]",
    button: "primary" as const,
  },
};

export const ConfirmDialog: React.FC<ConfirmDialogProps> = ({
  isOpen,
  title,
  message,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  variant = "danger",
  loading = false,
  onConfirm,
  onCancel,
}) => {
  const cancelRef = useRef<HTMLButtonElement>(null);
  const meta = VARIANT_META[variant];

  useEffect(() => {
    if (!isOpen) return;
    const previouslyFocused = document.activeElement as HTMLElement | null;
    const focusTimer = window.setTimeout(() => cancelRef.current?.focus(), 0);
    return () => {
      window.clearTimeout(focusTimer);
      previouslyFocused?.focus?.();
    };
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !loading) onCancel();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [isOpen, loading, onCancel]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget && !loading) onCancel();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
        aria-describedby="confirm-dialog-message"
        className="w-full max-w-md space-y-3 rounded-lg border border-[#333333] bg-[#202020] p-4 shadow-xl"
      >
        <div className="flex items-start gap-3">
          <div
            className={cn(
              "flex items-center justify-center w-8 h-8 rounded border shrink-0",
              meta.ring,
              meta.tone
            )}
            aria-hidden="true"
          >
            {meta.icon}
          </div>
          <div className="min-w-0 space-y-1">
            <h3
              id="confirm-dialog-title"
              className="font-mono text-xs font-semibold text-[#F2F2F2] uppercase tracking-wide"
            >
              {title}
            </h3>
            <p
              id="confirm-dialog-message"
              className="text-xs leading-relaxed text-[#A3A3A3]"
            >
              {message}
            </p>
          </div>
        </div>

        <div className="flex items-center justify-end gap-2 pt-2.5 border-t border-[#262626]">
          <button ref={cancelRef} type="button" className="btn btn-ghost btn-sm" onClick={onCancel} disabled={loading}>
            {cancelLabel}
          </button>
          <Button variant={meta.button} size="sm" onClick={onConfirm} loading={loading}>
            {confirmLabel}
          </Button>
        </div>
      </div>
    </div>
  );
};
