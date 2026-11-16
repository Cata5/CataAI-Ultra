import React from "react";
import { cn } from "@/lib/utils";
import { AlertTriangle, RefreshCw } from "lucide-react";

interface ErrorStateProps {
  title?: string;
  message: string;
  onRetry?: () => void;
  className?: string;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = "Telemetry Error",
  message,
  onRetry,
  className,
}) => {
  return (
    <div
      className={cn(
        "glass-panel border-[#FF657A]/30 rounded-lg p-6 flex flex-col items-center justify-center text-center space-y-3",
        className
      )}
    >
      <div className="p-2.5 bg-[#FF657A]/10 rounded-full border border-[#FF657A]/30 text-[#FF657A]">
        <AlertTriangle className="w-6 h-6" />
      </div>
      <h4 className="font-mono text-sm font-semibold text-[#FF657A] uppercase tracking-wider">
        {title}
      </h4>
      <p className="text-xs font-mono text-[#91A0B4] max-w-md">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-2 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-[#182333] border border-[#273548] hover:border-[#45CFFF] text-xs font-mono text-[#ECF3FC] transition-colors"
        >
          <RefreshCw className="w-3.5 h-3.5 text-[#45CFFF]" />
          RETRY
        </button>
      )}
    </div>
  );
};
