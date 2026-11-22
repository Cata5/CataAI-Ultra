import React from "react";
import { ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";
import { cn } from "@/lib/utils";

type Accent = "cyan" | "violet" | "emerald" | "amber" | "rose";

interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;
  subtitle?: string;
  icon?: React.ReactNode;
  trend?: "up" | "down" | "neutral";
  trendValue?: string;
  /** Accent applied to the icon and trend chip. No glow, no colored borders. */
  glowColor?: Accent;
  /** Shows placeholder skeleton lines while the first payload is loading. */
  loading?: boolean;
  /** Native tooltip for the metric label (what the number means). */
  hint?: string;
  className?: string;
}

const ACCENT_ICON: Record<Accent, string> = {
  cyan: "text-[#45CFFF]",
  violet: "text-[#9B8CFF]",
  emerald: "text-[#35D49A]",
  amber: "text-[#F2B85B]",
  rose: "text-[#FF657A]",
};

const TREND_STYLE: Record<NonNullable<MetricCardProps["trend"]>, string> = {
  up: "text-[#35D49A] bg-[rgba(53,212,154,0.08)] border-[rgba(53,212,154,0.24)]",
  down: "text-[#FF657A] bg-[rgba(255,101,122,0.08)] border-[rgba(255,101,122,0.24)]",
  neutral: "text-[#91A0B4] bg-[var(--bg-subtle)] border-[var(--border-default)]",
};

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  unit,
  subtitle,
  icon,
  trend = "neutral",
  trendValue,
  glowColor = "cyan",
  loading = false,
  hint,
  className,
}) => {
  const TrendIcon =
    trend === "up" ? ArrowUpRight : trend === "down" ? ArrowDownRight : Minus;

  return (
    <div
      className={cn(
        "panel group p-3.5 flex flex-col justify-between transition-colors duration-150",
        "hover:border-[var(--border-default)]",
        className
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span
          className="font-mono text-[10px] font-semibold uppercase tracking-[0.08em] text-[#91A0B4] truncate"
          title={hint || title}
        >
          {title}
        </span>
        {icon && (
          <span
            className={cn(
              "shrink-0 opacity-70 transition-opacity group-hover:opacity-100",
              ACCENT_ICON[glowColor]
            )}
            aria-hidden="true"
          >
            {icon}
          </span>
        )}
      </div>

      <div className="mt-2 min-h-[30px] flex items-baseline gap-1.5">
        {loading ? (
          <span className="skeleton block h-6 w-24" aria-label="Loading value" />
        ) : (
          <>
            <span className="font-mono-num text-[22px] leading-none font-semibold tracking-tight text-[#ECF3FC] truncate">
              {value}
            </span>
            {unit && (
              <span className="font-mono text-[10px] font-medium text-[#5E7188] shrink-0">
                {unit}
              </span>
            )}
          </>
        )}
      </div>

      {(subtitle || trendValue) && (
        <div className="mt-2.5 pt-2 border-t border-[var(--border-subtle)] flex items-center justify-between gap-2 min-h-[22px]">
          {subtitle && (
            <span className="truncate font-mono text-[10px] text-[#5E7188]" title={subtitle}>
              {subtitle}
            </span>
          )}
          {trendValue && (
            <span
              className={cn(
                "inline-flex shrink-0 items-center gap-0.5 rounded border px-1.5 py-0.5 font-mono text-[10px] font-semibold",
                TREND_STYLE[trend]
              )}
            >
              <TrendIcon className="w-3 h-3" aria-hidden="true" />
              {trendValue}
            </span>
          )}
        </div>
      )}
    </div>
  );
};
