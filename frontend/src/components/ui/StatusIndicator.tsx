import React from "react";
import { cn } from "@/lib/utils";
import { RunState } from "@/types/api";

interface StatusIndicatorProps {
  state?: RunState | string;
  size?: "sm" | "md" | "lg";
  showLabel?: boolean;
  className?: string;
}

type StateStyle = {
  dot: string;
  text: string;
};

const STATE_STYLES: Record<string, StateStyle> = {
  RUNNING: { dot: "bg-[#4ade80]", text: "text-[#4ade80]" },
  STARTING: { dot: "bg-[#D4D4D4]", text: "text-[#D4D4D4]" },
  PAUSED: { dot: "bg-[#facc15]", text: "text-[#facc15]" },
  STOPPING: { dot: "bg-[#facc15]", text: "text-[#facc15]" },
  STOPPED: { dot: "bg-[#737373]", text: "text-[#737373]" },
  COMPLETED: { dot: "bg-[#4ade80]", text: "text-[#4ade80]" },
  FAILED: { dot: "bg-[#f87171]", text: "text-[#f87171]" },
  CRITICAL: { dot: "bg-[#f87171]", text: "text-[#f87171]" },
  ERROR: { dot: "bg-[#f87171]", text: "text-[#f87171]" },
  IDLE: { dot: "bg-[#525252]", text: "text-[#737373]" },
};

const FALLBACK: StateStyle = STATE_STYLES.IDLE;

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  state = "IDLE",
  size = "md",
  showLabel = true,
  className,
}) => {
  const normalized = state.toUpperCase();
  const style = STATE_STYLES[normalized] ?? FALLBACK;

  const dotSize =
    size === "sm" ? "w-1.5 h-1.5" : size === "lg" ? "w-2.5 h-2.5" : "w-2 h-2";

  return (
    <div
      className={cn("inline-flex items-center gap-1.5", className)}
      title={normalized}
      role="status"
      aria-label={`Run state: ${normalized}`}
    >
      <span className={cn("rounded-full shrink-0", dotSize, style.dot)} />
      {showLabel && (
        <span
          className={cn(
            "font-mono text-[10px] font-medium tracking-wide uppercase",
            style.text
          )}
        >
          {normalized}
        </span>
      )}
    </div>
  );
};
