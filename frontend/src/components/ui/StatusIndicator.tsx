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

  let colorBg = "bg-[#5E7188]";
  let colorGlow = "rgba(94, 113, 136, 0.4)";
  let textColor = "text-[#91A0B4]";
  let isPulsing = false;

  switch (normalized) {
    case "RUNNING":
      colorBg = "bg-[#45CFFF]";
      colorGlow = "rgba(69, 207, 255, 0.6)";
      textColor = "text-[#45CFFF]";
      isPulsing = true;
      break;
    case "STARTING":
      colorBg = "bg-[#80DEFF]";
      colorGlow = "rgba(128, 222, 255, 0.5)";
      textColor = "text-[#80DEFF]";
      isPulsing = true;
      break;
    case "PAUSED":
      colorBg = "bg-[#F2B85B]";
      colorGlow = "rgba(242, 184, 91, 0.5)";
      textColor = "text-[#F2B85B]";
      break;
    case "COMPLETED":
      colorBg = "bg-[#35D49A]";
      colorGlow = "rgba(53, 212, 154, 0.5)";
      textColor = "text-[#35D49A]";
      break;
    case "FAILED":
    case "CRITICAL":
    case "ERROR":
      colorBg = "bg-[#FF657A]";
      colorGlow = "rgba(255, 101, 122, 0.6)";
      textColor = "text-[#FF657A]";
      isPulsing = true;
      break;
    case "STOPPING":
      colorBg = "bg-[#F2B85B]";
      colorGlow = "rgba(242, 184, 91, 0.5)";
      textColor = "text-[#F2B85B]";
      break;
    default:
      colorBg = "bg-[#5E7188]";
      colorGlow = "rgba(94, 113, 136, 0.2)";
      textColor = "text-[#91A0B4]";
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
