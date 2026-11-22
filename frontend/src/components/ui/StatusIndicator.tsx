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
  ring: string;
  pulse?: boolean;
};

const STATE_STYLES: Record<string, StateStyle> = {
  RUNNING: {
    dot: "bg-[#45CFFF]",
    text: "text-[#45CFFF]",
    ring: "bg-[rgba(69,207,255,0.35)]",
    pulse: true,
  },
  STARTING: {
    dot: "bg-[#80DEFF]",
    text: "text-[#80DEFF]",
    ring: "bg-[rgba(128,222,255,0.3)]",
    pulse: true,
  },
  PAUSED: { dot: "bg-[#F2B85B]", text: "text-[#F2B85B]", ring: "bg-[rgba(242,184,91,0.3)]" },
  STOPPING: { dot: "bg-[#F2B85B]", text: "text-[#F2B85B]", ring: "bg-[rgba(242,184,91,0.3)]" },
  STOPPED: { dot: "bg-[#8291A5]", text: "text-[#91A0B4]", ring: "bg-[rgba(130,145,165,0.25)]" },
  COMPLETED: { dot: "bg-[#35D49A]", text: "text-[#35D49A]", ring: "bg-[rgba(53,212,154,0.3)]" },
  FAILED: { dot: "bg-[#FF657A]", text: "text-[#FF657A]", ring: "bg-[rgba(255,101,122,0.35)]", pulse: true },
  CRITICAL: { dot: "bg-[#FF657A]", text: "text-[#FF657A]", ring: "bg-[rgba(255,101,122,0.35)]", pulse: true },
  ERROR: { dot: "bg-[#FF657A]", text: "text-[#FF657A]", ring: "bg-[rgba(255,101,122,0.35)]", pulse: true },
  IDLE: { dot: "bg-[#5E7188]", text: "text-[#91A0B4]", ring: "bg-[rgba(94,113,136,0.2)]" },
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
    size === "sm" ? "w-2 h-2" : size === "lg" ? "w-3 h-3" : "w-2.5 h-2.5";

  return (
    <div
      className={cn("inline-flex items-center gap-2", className)}
      title={normalized}
      role="status"
      aria-label={`Run state: ${normalized}`}
    >
      <span className="relative flex items-center justify-center" aria-hidden="true">
        {style.pulse && (
          <span className={cn("absolute rounded-full animate-ping opacity-60", dotSize, style.ring)} />
        )}
        <span className={cn("relative rounded-full", dotSize, style.dot)} />
      </span>
      {showLabel && (
        <span
          className={cn(
            "font-mono text-[11px] font-semibold tracking-[0.08em] uppercase",
            style.text
          )}
        >
          {normalized}
        </span>
      )}
    </div>
  );
};
