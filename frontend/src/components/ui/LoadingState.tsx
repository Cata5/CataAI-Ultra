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
        "panel bg-[#202020] border-[#333333] flex flex-col items-center justify-center text-center gap-3 p-10",
        className
      )}
      role="status"
      aria-live="polite"
      aria-busy="true"
    >
      <Loader2 className="w-5 h-5 text-[#D4D4D4] animate-spin" aria-hidden="true" />
      <span className="font-mono text-[11px] text-[#A3A3A3] tracking-wider uppercase font-medium">
        {message}
      </span>
      {hint && <span className="max-w-sm text-[11px] text-[#737373]">{hint}</span>}
    </div>
  );
};
