import React from "react";
import { AlertTriangle, CheckCircle2, Info, X, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";

export type AlertTone = "info" | "success" | "warning" | "error";

interface InlineAlertProps {
  tone?: AlertTone;
  title?: string;
  children?: React.ReactNode;
  message?: string;
  action?: React.ReactNode;
  onDismiss?: () => void;
  className?: string;
}

const TONE_ICON: Record<AlertTone, React.ReactNode> = {
  info: <Info className="w-3.5 h-3.5 text-[#A3A3A3]" />,
  success: <CheckCircle2 className="w-3.5 h-3.5 text-[#4ade80]" />,
  warning: <AlertTriangle className="w-3.5 h-3.5 text-[#facc15]" />,
  error: <XCircle className="w-3.5 h-3.5 text-[#f87171]" />,
};

export const InlineAlert: React.FC<InlineAlertProps> = ({
  tone = "info",
  title,
  children,
  message,
  action,
  onDismiss,
  className,
}) => {
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      aria-live={tone === "error" ? "assertive" : "polite"}
      className={cn("alert", `alert-${tone}`, className)}
    >
      <span className="mt-0.5 shrink-0">{TONE_ICON[tone]}</span>

      <div className="min-w-0 flex-1 space-y-0.5">
        {title && (
          <p className="font-mono text-[10px] font-medium uppercase tracking-wider text-[#F2F2F2]">
            {title}
          </p>
        )}
        {message && <p className="text-[11px] leading-relaxed text-[#A3A3A3] break-words">{message}</p>}
        {children}
      </div>

      {action && <div className="shrink-0 self-center">{action}</div>}

      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          className="icon-btn shrink-0 -mr-1 -my-1 h-6 w-6 text-[#737373] hover:text-[#F2F2F2]"
          aria-label="Dismiss message"
        >
          <X className="w-3 h-3" />
        </button>
      )}
    </div>
  );
};
