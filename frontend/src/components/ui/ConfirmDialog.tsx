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
  /** Disables both actions and shows a spinner on the confirm button. */
  loading?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

const VARIANT_META = {
  danger: {
    icon: <AlertCircle className="w-5 h-5" />,
    tone: "text-[#FF657A]",
    ring: "border-[rgba(255,101,122,0.3)] bg-[rgba(255,101,122,0.08)]",
    button: "danger" as const,
  },
  warning: {
    icon: <AlertTriangle className="w-5 h-5" />,
    tone: "text-[#F2B85B]",
    ring: "border-[rgba(242,184,91,0.3)] bg-[rgba(242,184,91,0.08)]",
    button: "warning" as const,
  },
  info: {
    icon: <Info className="w-5 h-5" />,
    tone: "text-[#45CFFF]",
    ring: "border-[rgba(69,207,255,0.3)] bg-[rgba(69,207,255,0.08)]",
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
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-[rgba(5,8,13,0.8)] backdrop-blur-sm"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget && !loading) onCancel();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
        aria-describedby="confirm-dialog-message"
        className="w-full max-w-md space-y-4 rounded-[12px] border border-[var(--border-strong)] bg-[var(--bg-surface)] p-5 shadow-[0_24px_80px_-20px_rgba(0,0,0,0.95)] animate-in fade-in zoom-in-95 duration-150"
      >
        <div className="flex items-start gap-3">
          <div
            className={cn(
              "flex items-center justify-center w-9 h-9 rounded-lg border shrink-0",
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
              className="font-mono text-[12px] font-bold text-[#ECF3FC] uppercase tracking-wide"
            >
              {title}
            </h3>
            <p
              id="confirm-dialog-message"
              className="text-[12px] leading-relaxed text-[#91A0B4]"
            >
              {message}
            </p>
          </div>
        </div>

        <div className="flex items-center justify-end gap-2 pt-3 border-t border-[var(--border-subtle)]">
          <button ref={cancelRef} type="button" className="btn btn-ghost" onClick={onCancel} disabled={loading}>
            {cancelLabel}
          </button>
          <Button variant={meta.button} onClick={onConfirm} loading={loading}>
            {confirmLabel}
          </Button>
        </div>
      </div>
    </div>
  );
};
