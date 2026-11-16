import React from "react";
import { cn } from "@/lib/utils";
import { Inbox } from "lucide-react";

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon = <Inbox className="w-8 h-8 text-slate-500" />,
  title,
  description,
  action,
  className,
}) => {
  return (
    <div
      className={cn(
        "glass-panel rounded-lg p-8 flex flex-col items-center justify-center text-center space-y-3",
        className
      )}
    >
      <div className="p-3 bg-slate-900/80 rounded-full border border-slate-800 text-slate-400">
        {icon}
      </div>
      <h4 className="font-mono text-sm font-semibold text-slate-200 uppercase tracking-wider">
        {title}
      </h4>
      {description && (
        <p className="text-xs text-slate-400 max-w-sm">{description}</p>
      )}
      {action && <div className="pt-2">{action}</div>}
    </div>
  );
};
