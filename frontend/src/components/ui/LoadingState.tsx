import React from "react";
import { cn } from "@/lib/utils";
import { Loader2 } from "lucide-react";

interface LoadingStateProps {
  message?: string;
  className?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({
  message = "Loading telemetry & neural state...",
  className,
}) => {
  return (
    <div
      className={cn(
        "glass-panel rounded-lg p-8 flex flex-col items-center justify-center text-center space-y-3 border border-[#273548]",
        className
      )}
    >
      <Loader2 className="w-8 h-8 text-[#45CFFF] animate-spin" />
      <span className="font-mono text-xs text-[#91A0B4] tracking-wider uppercase">
        {message}
      </span>
    </div>
  );
};
