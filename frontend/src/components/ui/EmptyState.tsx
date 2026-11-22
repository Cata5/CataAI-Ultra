import React from "react";
import { cn } from "@/lib/utils";
import { Inbox } from "lucide-react";

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
  /** Compact variant for use inside panels/tables. */
  compact?: boolean;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon = <Inbox className="w-6 h-6 text-[#5E7188]" />,
  title,
  description,
  action,
  className,
  compact = false,
}) => {
  return (
    <div
      className={cn(
        "panel flex flex-col items-center justify-center text-center",
        compact ? "p-6 gap-2" : "p-10 gap-3",
        className
      )}
    >
      <div
        className="flex items-center justify-center rounded-full border border-[var(--border-default)] bg-[var(--bg-subtle)] text-[#91A0B4]"
        style={{ width: compact ? 40 : 48, height: compact ? 40 : 48 }}
        aria-hidden="true"
      >
        {icon}
      </div>
      <h4 className="font-mono text-[12px] font-semibold text-[#ECF3FC] uppercase tracking-wider">
        {title}
      </h4>
      {description && (
        <p className="max-w-sm text-[12px] leading-relaxed text-[#91A0B4]">{description}</p>
      )}
      {action && <div className="pt-1.5">{action}</div>}
    </div>
  );
};
