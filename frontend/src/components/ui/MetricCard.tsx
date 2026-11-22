import React from "react";
import { ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";
import { cn } from "@/lib/utils";

type Accent = "cyan" | "violet" | "emerald" | "amber" | "rose" | "gray";

interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;
  subtitle?: string;
  icon?: React.ReactNode;
  trend?: "up" | "down" | "neutral";
  trendValue?: string;
  glowColor?: Accent;
  loading?: boolean;
  hint?: string;
  className?: string;
}

const TREND_STYLE: Record<NonNullable<MetricCardProps["trend"]>, string> = {
  up: "text-[#4ade80] bg-[rgba(74,222,128,0.08)] border-[rgba(74,222,128,0.2)]",
  down: "text-[#f87171] bg-[rgba(248,113,113,0.08)] border-[rgba(248,113,113,0.2)]",
  neutral: "text-[#737373] bg-[#262626] border-[#333333]",
};

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  unit,
  subtitle,
  icon,
  trend = "neutral",
  trendValue,
  loading = false,
  hint,
  className,
}) => {
  const TrendIcon =
    trend === "up" ? ArrowUpRight : trend === "down" ? ArrowDownRight : Minus;

  return (
    <div
      className={cn(
        "panel group p-3 flex flex-col justify-between transition-colors bg-[#202020] border-[#333333] hover:border-[#404040]",
        className
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span
          className="font-mono text-[10px] font-medium uppercase tracking-[0.06em] text-[#737373] truncate"
          title={hint || title}
        >
          {title}
        </span>
        {icon && (
          <span
            className="shrink-0 text-[#737373] transition-colors group-hover:text-[#A3A3A3]"
            aria-hidden="true"
          >
            {icon}
          </span>
        )}
      </div>

      <div className="mt-1.5 min-h-[26px] flex items-baseline gap-1.5">
        {loading ? (
          <span className="skeleton block h-5 w-20" aria-label="Loading value" />
        ) : (
          <>
            <span className="font-mono-num text-[20px] leading-none font-medium tracking-tight text-[#F2F2F2] truncate">
              {value}
            </span>
            {unit && (
              <span className="font-mono text-[10px] text-[#737373] shrink-0">
                {unit}
              </span>
            )}
          </>
        )}
      </div>

      {(subtitle || trendValue) && (
        <div className="mt-2 pt-1.5 border-t border-[#262626] flex items-center justify-between gap-2 min-h-[18px]">
          {subtitle && (
            <span className="truncate font-mono text-[10px] text-[#737373]" title={subtitle}>
              {subtitle}
            </span>
          )}
          {trendValue && (
            <span
              className={cn(
                "inline-flex shrink-0 items-center gap-0.5 rounded border px-1.5 py-0.2 font-mono text-[9px] font-medium",
                TREND_STYLE[trend]
              )}
            >
              <TrendIcon className="w-2.5 h-2.5" aria-hidden="true" />
              {trendValue}
            </span>
          )}
        </div>
      )}
    </div>
  );
};
