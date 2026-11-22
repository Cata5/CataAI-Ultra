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
        "panel bg-[#202020] border-[#f87171]/30 flex flex-col items-center justify-center text-center gap-3 p-8",
        className
      )}
    >
      <div className="flex items-center justify-center w-10 h-10 rounded-full border border-[#f87171]/30 bg-[#f87171]/10 text-[#f87171]">
        <AlertTriangle className="w-4 h-4" aria-hidden="true" />
      </div>
      <h4 className="font-mono text-[12px] font-semibold text-[#f87171] uppercase tracking-wider">
        {title}
      </h4>
      <p className="max-w-md text-[12px] leading-relaxed text-[#F2F2F2] font-mono break-words">
        {message}
      </p>
      {hint && <p className="max-w-md text-[11px] leading-relaxed text-[#737373]">{hint}</p>}
      {onRetry && (
        <Button
          variant="secondary"
          size="sm"
          onClick={onRetry}
          loading={retrying}
          icon={<RefreshCw className="w-3.5 h-3.5 text-[#A3A3A3]" />}
          className="mt-1"
        >
          Retry
        </Button>
      )}
    </div>
  );
};
