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
        "glass-panel border-rose-500/30 rounded-lg p-6 flex flex-col items-center justify-center text-center space-y-3",
        className
      )}
    >
      <div className="p-2.5 bg-rose-500/10 rounded-full border border-rose-500/30 text-rose-400">
        <AlertTriangle className="w-6 h-6" />
      </div>
      <h4 className="font-mono text-sm font-semibold text-rose-300 uppercase tracking-wider">
        {title}
      </h4>
      <p className="text-xs font-mono text-slate-400 max-w-md">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-2 inline-flex items-center gap-1.5 px-3 py-1.5 rounded bg-slate-900 border border-slate-700 hover:border-cyan-400 text-xs font-mono text-slate-200 transition-colors"
        >
          <RefreshCw className="w-3.5 h-3.5 text-cyan-400" />
          RETRY
        </button>
      )}
    </div>
  );
};
