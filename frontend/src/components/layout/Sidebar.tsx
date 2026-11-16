"use client";

import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Network,
  Play,
  Database,
  FlaskConical,
  History,
  Settings,
  ChevronLeft,
  ChevronRight,
  Zap,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useHardwareTelemetry } from "@/hooks/useHardwareTelemetry";

interface NavItem {
  name: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  tag?: string;
}

const NAV_ITEMS: NavItem[] = [
  { name: "Overview", href: "/", icon: LayoutDashboard },
  { name: "Network Studio", href: "/studio", icon: Network, tag: "STUDIO" },
  { name: "Training", href: "/training", icon: Play },
  { name: "Datasets", href: "/datasets", icon: Database },
  { name: "Evaluation Lab", href: "/evaluation", icon: FlaskConical },
  { name: "Runs & History", href: "/runs", icon: History },
  { name: "Settings", href: "/settings", icon: Settings },
];

export const Sidebar: React.FC = () => {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const { telemetry } = useHardwareTelemetry(5000);

  const deviceShort = telemetry?.device_name
    ? telemetry.device_name.replace("NVIDIA GeForce ", "").replace(" Laptop GPU", "")
    : "Detected Hardware";

  const ceilingStr = telemetry?.hardware_profile?.max_memory_allocated_gb
    ? `< ${telemetry.hardware_profile.max_memory_allocated_gb} GB`
    : "< Safe Limit";

  return (
    <aside
      className={cn(
        "h-screen flex flex-col glass-panel border-r border-slate-800/80 bg-[#060a14]/95 transition-all duration-300 z-30 select-none",
        collapsed ? "w-16" : "w-60"
      )}
    >
      {/* Brand Header */}
      <div className="h-14 flex items-center justify-between px-3.5 border-b border-slate-800/80">
        <Link href="/" className="flex items-center gap-2.5 overflow-hidden">
          <div className="w-8 h-8 rounded-lg bg-cyan-950/80 border border-cyan-500/40 flex items-center justify-center shrink-0 shadow-[0_0_12px_rgba(0,240,255,0.25)]">
            <Zap className="w-4 h-4 text-cyan-400" />
          </div>
          {!collapsed && (
            <div className="flex flex-col">
              <span className="font-mono text-xs font-bold tracking-wider text-slate-100 flex items-center gap-1">
                LLM Maker <span className="text-cyan-400 font-normal">v2</span>
              </span>
              <span className="text-[10px] font-mono text-cyan-400/80 truncate">
                Universal ML Platform
              </span>
            </div>
          )}
        </Link>

        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-1 rounded text-slate-400 hover:text-cyan-400 hover:bg-slate-800/60 transition-colors"
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>

      {/* Nav List */}
      <nav className="flex-1 py-4 px-2 space-y-1 overflow-y-auto">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive =
            item.href === "/"
              ? pathname === "/"
              : pathname.startsWith(item.href);

          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-mono transition-all duration-200 group relative",
                isActive
                  ? "bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 shadow-[0_0_10px_rgba(0,240,255,0.15)]"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 border border-transparent"
              )}
              title={collapsed ? item.name : undefined}
            >
              <Icon
                className={cn(
                  "w-4 h-4 shrink-0 transition-colors",
                  isActive ? "text-cyan-400" : "text-slate-400 group-hover:text-slate-300"
                )}
              />
              {!collapsed && (
                <span className="truncate tracking-wide font-medium">
                  {item.name}
                </span>
              )}
              {!collapsed && item.tag && (
                <span className="ml-auto text-[9px] px-1.5 py-0.5 rounded bg-cyan-950/60 border border-cyan-500/30 text-cyan-400 font-mono">
                  {item.tag}
                </span>
              )}
              {collapsed && isActive && (
                <div className="absolute right-0 top-1/2 -translate-y-1/2 w-1 h-5 bg-cyan-400 rounded-l" />
              )}
            </Link>
          );
        })}
      </nav>

      {/* Footer Info */}
      <div className="p-3 border-t border-slate-800/80">
        {!collapsed ? (
          <div className="flex flex-col gap-1 text-[10px] font-mono text-slate-400">
            <div className="flex justify-between items-center">
              <span>TARGET</span>
              <span className="text-slate-300 truncate max-w-[110px]" title={telemetry?.device_name}>
                {deviceShort}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span>CEILING</span>
              <span className="text-cyan-400 font-bold">{ceilingStr}</span>
            </div>
          </div>
        ) : (
          <div className="flex justify-center text-slate-400">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          </div>
        )}
      </div>
    </aside>
  );
};
