import React from "react";
import { cn } from "@/lib/utils";
import { RunState } from "@/types/api";

interface StatusIndicatorProps {
  state?: RunState | string;
  size?: "sm" | "md" | "lg";
  showLabel?: boolean;
  className?: string;
}

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  state = "IDLE",
  size = "md",
  showLabel = true,
  className,
}) => {
  const normalized = state.toUpperCase();

  let colorBg = "bg-slate-500";
  let colorGlow = "rgba(100, 116, 139, 0.4)";
  let textColor = "text-slate-400";
  let isPulsing = false;

  switch (normalized) {
    case "RUNNING":
      colorBg = "bg-cyan-400";
      colorGlow = "rgba(0, 240, 255, 0.6)";
      textColor = "text-cyan-300";
      isPulsing = true;
      break;
    case "STARTING":
      colorBg = "bg-sky-400";
      colorGlow = "rgba(56, 189, 248, 0.5)";
      textColor = "text-sky-300";
      isPulsing = true;
      break;
    case "PAUSED":
      colorBg = "bg-amber-400";
      colorGlow = "rgba(245, 158, 11, 0.5)";
      textColor = "text-amber-300";
      break;
    case "COMPLETED":
      colorBg = "bg-emerald-400";
      colorGlow = "rgba(16, 185, 129, 0.5)";
      textColor = "text-emerald-300";
      break;
    case "FAILED":
    case "CRITICAL":
    case "ERROR":
      colorBg = "bg-rose-500";
      colorGlow = "rgba(239, 68, 68, 0.6)";
      textColor = "text-rose-400";
      isPulsing = true;
      break;
    case "STOPPING":
      colorBg = "bg-orange-400";
      colorGlow = "rgba(251, 146, 60, 0.5)";
      textColor = "text-orange-300";
      break;
    default:
      colorBg = "bg-slate-500";
      colorGlow = "rgba(100, 116, 139, 0.2)";
      textColor = "text-slate-400";
      break;
  }

  const dotSize =
    size === "sm" ? "w-2 h-2" : size === "lg" ? "w-3.5 h-3.5" : "w-2.5 h-2.5";

  return (
    <div className={cn("inline-flex items-center gap-2", className)}>
      <span className="relative flex items-center justify-center">
        {isPulsing && (
          <span
            className={cn("absolute rounded-full animate-ping opacity-75", dotSize, colorBg)}
          />
        )}
        <span
          className={cn("relative rounded-full shadow-sm", dotSize, colorBg)}
          style={{ boxShadow: `0 0 8px ${colorGlow}` }}
        />
      </span>
      {showLabel && (
        <span
          className={cn(
            "font-mono text-xs font-semibold tracking-wider uppercase",
            textColor
          )}
        >
          {normalized}
        </span>
      )}
    </div>
  );
};
