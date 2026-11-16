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
  const accentBorderHover =
    glowColor === "cyan"
      ? "hover:border-[#45CFFF]/40 hover:shadow-[0_4px_20px_-4px_rgba(69,207,255,0.12)]"
      : glowColor === "violet"
      ? "hover:border-[#9B8CFF]/40 hover:shadow-[0_4px_20px_-4px_rgba(155,140,255,0.12)]"
      : glowColor === "emerald"
      ? "hover:border-[#35D49A]/40 hover:shadow-[0_4px_20px_-4px_rgba(53,212,154,0.12)]"
      : glowColor === "amber"
      ? "hover:border-[#F2B85B]/40 hover:shadow-[0_4px_20px_-4px_rgba(242,184,91,0.12)]"
      : "hover:border-[#FF657A]/40 hover:shadow-[0_4px_20px_-4px_rgba(255,101,122,0.12)]";

  return (
    <div
      className={cn(
        "glass-panel relative overflow-hidden rounded-lg p-3.5 transition-all duration-200 group",
        accentBorderHover,
        className
      )}
    >
      <div className="flex items-center justify-between text-[11px] font-mono text-[#91A0B4] mb-1.5 uppercase tracking-wider">
        <span>{title}</span>
        {icon && <span className="text-[#91A0B4] group-hover:text-[#45CFFF] transition-colors">{icon}</span>}
      </div>

      <div className="flex items-baseline gap-1.5">
        <span className="font-mono-num text-2xl font-bold tracking-tight text-[#ECF3FC]">
          {value}
        </span>
        {unit && (
          <span className="font-mono text-xs font-medium text-[#91A0B4]">{unit}</span>
        )}
      </div>

      {(subtitle || trendValue) && (
        <div className="mt-2 flex items-center justify-between text-xs text-[#91A0B4] pt-1.5 border-t border-[#273548]/60">
          {subtitle && <span className="truncate text-[11px]">{subtitle}</span>}
          {trendValue && (
            <span
              className={cn(
                "font-mono font-medium text-[11px]",
                trend === "up"
                  ? "text-[#35D49A]"
                  : trend === "down"
                  ? "text-[#FF657A]"
                  : "text-[#91A0B4]"
              )}
            >
              {trendValue}
            </span>
          )}
        </div>
      )}

      {/* Subtle hairline inset highlight */}
      <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#45CFFF]/15 to-transparent pointer-events-none" />
    </div>
  );
};
