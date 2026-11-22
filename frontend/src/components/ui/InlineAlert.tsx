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
  info: <Info className="w-4 h-4 text-[#45CFFF]" />,
  success: <CheckCircle2 className="w-4 h-4 text-[#35D49A]" />,
  warning: <AlertTriangle className="w-4 h-4 text-[#F2B85B]" />,
  error: <XCircle className="w-4 h-4 text-[#FF657A]" />,
};

/**
 * Inline feedback banner — use instead of alert(): it stays in context,
 * is dismissible, and is announced to assistive tech.
 */
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
          <p className="font-mono text-[11px] font-semibold uppercase tracking-wider text-[#ECF3FC]">
            {title}
          </p>
        )}
        {message && <p className="text-[12px] leading-relaxed text-[#C3CEE0] break-words">{message}</p>}
        {children}
      </div>

      {action && <div className="shrink-0 self-center">{action}</div>}

      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          className="icon-btn shrink-0 -mr-1 -my-1 h-7 w-7"
          aria-label="Dismiss message"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      )}
    </div>
  );
};
