import React from "react";
import { cn } from "@/lib/utils";
import { Loader2 } from "lucide-react";

interface LoadingStateProps {
  message?: string;
  hint?: string;
  className?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({
  message = "Loading telemetry…",
  hint,
  className,
}) => {
  return (
    <div
      className={cn(
        "panel flex flex-col items-center justify-center text-center gap-3 p-10",
        className
      )}
      role="status"
      aria-live="polite"
      aria-busy="true"
    >
      <Loader2 className="w-6 h-6 text-[#45CFFF] animate-spin" aria-hidden="true" />
      <span className="font-mono text-[11px] text-[#91A0B4] tracking-wider uppercase">
        {message}
      </span>
      {hint && <span className="max-w-sm text-[11px] text-[#5E7188]">{hint}</span>}
    </div>
  );
};
