import React from "react";
import { cn } from "@/lib/utils";
import { AlertTriangle, RefreshCw } from "lucide-react";
import { Button } from "./Button";

interface ErrorStateProps {
  title?: string;
  message: string;
  hint?: string;
  onRetry?: () => void;
  retrying?: boolean;
  className?: string;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = "Something went wrong",
  message,
  hint,
  onRetry,
  retrying = false,
  className,
}) => {
  return (
    <div
      role="alert"
      className={cn(
        "panel flex flex-col items-center justify-center text-center gap-3 p-8 border-[rgba(255,101,122,0.28)]",
        className
      )}
    >
      <div className="flex items-center justify-center w-11 h-11 rounded-full border border-[rgba(255,101,122,0.3)] bg-[rgba(255,101,122,0.08)] text-[#FF657A]">
        <AlertTriangle className="w-5 h-5" aria-hidden="true" />
      </div>
      <h4 className="font-mono text-[12px] font-semibold text-[#FF657A] uppercase tracking-wider">
        {title}
      </h4>
      <p className="max-w-md text-[12px] leading-relaxed text-[#C3CEE0] font-mono break-words">
        {message}
      </p>
      {hint && <p className="max-w-md text-[11px] leading-relaxed text-[#5E7188]">{hint}</p>}
      {onRetry && (
        <Button
          variant="secondary"
          size="sm"
          onClick={onRetry}
          loading={retrying}
          icon={<RefreshCw className="w-3.5 h-3.5 text-[#45CFFF]" />}
          className="mt-1"
        >
          Retry
        </Button>
      )}
    </div>
  );
};
