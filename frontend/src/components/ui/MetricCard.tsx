import React from "react";
import { cn } from "@/lib/utils";

interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;
  subtitle?: string;
  icon?: React.ReactNode;
  trend?: "up" | "down" | "neutral";
  trendValue?: string;
  glowColor?: "cyan" | "violet" | "emerald" | "amber" | "rose";
  className?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  unit,
  subtitle,
  icon,
  trend,
  trendValue,
  glowColor = "cyan",
  className,
}) => {
  const glowBorderClass =
    glowColor === "cyan"
      ? "hover:border-cyan-500/50 hover:shadow-[0_0_15px_rgba(0,240,255,0.15)]"
      : glowColor === "violet"
      ? "hover:border-indigo-500/50 hover:shadow-[0_0_15px_rgba(129,140,248,0.15)]"
      : glowColor === "emerald"
      ? "hover:border-emerald-500/50 hover:shadow-[0_0_15px_rgba(16,185,129,0.15)]"
      : glowColor === "amber"
      ? "hover:border-amber-500/50 hover:shadow-[0_0_15px_rgba(245,158,11,0.15)]"
      : "hover:border-rose-500/50 hover:shadow-[0_0_15px_rgba(239,68,68,0.15)]";

  return (
    <div
      className={cn(
        "glass-panel relative overflow-hidden rounded-lg p-3.5 transition-all duration-300",
        glowBorderClass,
        className
      )}
    >
      <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-1.5 uppercase tracking-wider">
        <span>{title}</span>
        {icon && <span className="text-cyan-400/80">{icon}</span>}
      </div>

      <div className="flex items-baseline gap-1.5">
        <span className="font-mono-num text-2xl font-bold tracking-tight text-slate-100">
          {value}
        </span>
        {unit && (
          <span className="font-mono text-xs font-medium text-slate-400">{unit}</span>
        )}
      </div>

      {(subtitle || trendValue) && (
        <div className="mt-2 flex items-center justify-between text-xs text-slate-400">
          {subtitle && <span className="truncate">{subtitle}</span>}
          {trendValue && (
            <span
              className={cn(
                "font-mono font-medium",
                trend === "up"
                  ? "text-emerald-400"
                  : trend === "down"
                  ? "text-rose-400"
                  : "text-slate-400"
              )}
            >
              {trendValue}
            </span>
          )}
        </div>
      )}

      {/* Subtle corner tech accent */}
      <div className="absolute top-0 right-0 w-2 h-2 border-t border-r border-cyan-400/30" />
    </div>
  );
};
