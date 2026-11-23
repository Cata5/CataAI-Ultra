"use client";

import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  FolderKanban,
  Boxes,
  Network,
  Database,
  Play,
  FlaskConical,
  History,
  Bookmark,
  BarChart2,
  Cpu,
  Settings,
  PanelLeftClose,
  PanelLeftOpen,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useHardwareTelemetry } from "@/hooks/useHardwareTelemetry";

interface NavItem {
  name: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
}

interface NavGroup {
  label: string;
  items: NavItem[];
}

const NAV_GROUPS: NavGroup[] = [
  {
    label: "Workspace",
    items: [
      { name: "Overview", href: "/", icon: LayoutDashboard },
      { name: "Projects", href: "/projects", icon: FolderKanban },
      { name: "Models", href: "/models", icon: Boxes },
    ],
  },
  {
    label: "Development",
    items: [
      { name: "Network Studio", href: "/studio", icon: Network },
      { name: "Datasets", href: "/datasets", icon: Database },
      { name: "Training", href: "/training", icon: Play },
      { name: "Evaluation", href: "/evaluation", icon: FlaskConical },
    ],
  },
  {
    label: "Management",
    items: [
      { name: "Experiments", href: "/runs", icon: History },
      { name: "Checkpoints", href: "/checkpoints", icon: Bookmark },
      { name: "Analytics", href: "/analytics", icon: BarChart2 },
    ],
  },
  {
    label: "System",
    items: [
      { name: "Hardware", href: "/hardware", icon: Cpu },
      { name: "Settings", href: "/settings", icon: Settings },
    ],
  },
];

export const Sidebar: React.FC = () => {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const { telemetry } = useHardwareTelemetry(5000);

  const deviceShort = telemetry?.device_name
    ? telemetry.device_name.replace("NVIDIA GeForce ", "").replace(" Laptop GPU", "")
    : "GPU";

  const vramUsedMb = Math.round(telemetry?.memory_used_mb ?? telemetry?.used_vram_mb ?? 0);
  const vramTotalMb = Math.round(telemetry?.memory_total_mb ?? telemetry?.total_vram_mb ?? 6144);

  return (
    <aside
      className={cn(
        "h-screen flex flex-col bg-[#141414] border-r border-[#333333] transition-all duration-200 z-30 select-none",
        collapsed ? "w-14" : "w-56"
      )}
    >
      {/* Brand Header */}
      <div
        className={cn(
          "h-12 flex items-center border-b border-[#262626] px-3",
          collapsed ? "justify-center" : "justify-between"
        )}
      >
        <Link href="/" className="flex items-center gap-2.5 overflow-hidden group">
          <img
            src="/logo.png"
            alt="LLM Maker"
            className="w-6 h-6 rounded object-contain shrink-0"
          />
          {!collapsed && (
            <div className="flex flex-col min-w-0">
              <span className="font-sans text-xs font-semibold text-[#F2F2F2] tracking-tight truncate">
                LLM Maker
              </span>
            </div>
          )}
        </Link>

        {!collapsed && (
          <button
            onClick={() => setCollapsed(true)}
            className="p-1 rounded text-[#737373] hover:text-[#F2F2F2] hover:bg-[#202020] transition-colors"
            title="Collapse sidebar"
            aria-label="Collapse sidebar"
          >
            <PanelLeftClose className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Nav List */}
      <nav className="flex-1 py-3 px-2 space-y-4 overflow-y-auto">
        {NAV_GROUPS.map((group) => (
          <div key={group.label} className="space-y-0.5">
            {!collapsed && (
              <div className="px-2.5 py-1 text-[10px] font-mono font-medium uppercase tracking-wider text-[#737373]">
                {group.label}
              </div>
            )}
            {group.items.map((item) => {
              const Icon = item.icon;
              const isActive =
                item.href === "/" ? pathname === "/" : pathname === item.href;

              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-2.5 px-2.5 py-1.5 rounded text-xs transition-colors group relative",
                    isActive
                      ? "bg-[#262626] text-[#F2F2F2] font-medium"
                      : "text-[#A3A3A3] hover:text-[#F2F2F2] hover:bg-[#202020]",
                    collapsed && "justify-center px-0"
                  )}
                  title={collapsed ? item.name : undefined}
                >
                  {/* Subtle active left bar indicator */}
                  {isActive && !collapsed && (
                    <span className="absolute left-0 top-1.5 bottom-1.5 w-0.5 rounded-full bg-[#D4D4D4]" />
                  )}
                  <Icon
                    className={cn(
                      "w-4 h-4 shrink-0 transition-colors",
                      isActive ? "text-[#F2F2F2]" : "text-[#737373] group-hover:text-[#D4D4D4]"
                    )}
                  />
                  {!collapsed && <span className="truncate">{item.name}</span>}
                </Link>
              );
            })}
          </div>
        ))}
      </nav>

      {/* Footer: Expand button (when collapsed) + Hardware summary */}
      <div className="p-2 border-t border-[#262626] bg-[#141414]">
        {collapsed ? (
          <div className="flex flex-col items-center gap-2">
            <button
              onClick={() => setCollapsed(false)}
              className="p-1.5 rounded text-[#737373] hover:text-[#F2F2F2] hover:bg-[#202020] transition-colors"
              title="Expand sidebar"
              aria-label="Expand sidebar"
            >
              <PanelLeftOpen className="w-4 h-4" />
            </button>
            <div
              className="w-2 h-2 rounded-full bg-[#737373]"
              title={`${deviceShort} · ${vramUsedMb}/${vramTotalMb} MB`}
            />
          </div>
        ) : (
          <div className="bg-[#1B1B1B] p-2 rounded border border-[#262626] font-mono text-[10px] space-y-1">
            <div className="flex items-center justify-between text-[#737373]">
              <span className="truncate">{deviceShort}</span>
              <span className="text-[#A3A3A3] font-mono-num">{telemetry?.gpu_utilization_pct ?? telemetry?.gpu_util_percent ?? 0}%</span>
            </div>
            <div className="flex items-center justify-between text-[#737373]">
              <span>VRAM</span>
              <span className="text-[#A3A3A3] font-mono-num">
                {vramUsedMb} / {vramTotalMb} MB
              </span>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
};
