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
  icon = <Inbox className="w-5 h-5 text-[#737373]" />,
  title,
  description,
  action,
  className,
  compact = false,
}) => {
  return (
    <div
      className={cn(
        "panel bg-[#202020] border-[#333333] flex flex-col items-center justify-center text-center",
        compact ? "p-6 gap-2" : "p-10 gap-3",
        className
      )}
    >
      <div
        className="flex items-center justify-center rounded-full border border-[#333333] bg-[#1B1B1B] text-[#A3A3A3]"
        style={{ width: compact ? 36 : 44, height: compact ? 36 : 44 }}
        aria-hidden="true"
      >
        {icon}
      </div>
      <h4 className="font-mono text-[12px] font-semibold text-[#F2F2F2] uppercase tracking-wider">
        {title}
      </h4>
      {description && (
        <p className="max-w-sm text-[12px] leading-relaxed text-[#A3A3A3]">{description}</p>
      )}
      {action && <div className="pt-1.5">{action}</div>}
    </div>
  );
};
