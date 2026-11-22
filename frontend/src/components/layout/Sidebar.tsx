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
  Cpu,
  Layers,
  FolderGit2,
  Bookmark,
  BarChart3,
  ShieldCheck,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useHardwareTelemetry } from "@/hooks/useHardwareTelemetry";

interface NavItem {
  name: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  tag?: string;
  badge?: "cyan" | "violet" | "emerald";
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
      { name: "Architecture Studio", href: "/studio", icon: Network, tag: "STUDIO", badge: "cyan" },
    ],
  },
  {
    label: "Development",
    items: [
      { name: "Training Console", href: "/training", icon: Play, tag: "LIVE", badge: "emerald" },
      { name: "Dataset Explorer", href: "/datasets", icon: Database },
      { name: "Evaluation Lab", href: "/evaluation", icon: FlaskConical },
    ],
  },
  {
    label: "Management",
    items: [
      { name: "Runs & Experiments", href: "/runs", icon: History },
    ],
  },
  {
    label: "System",
    items: [
      { name: "Hardware & Settings", href: "/settings", icon: Settings },
    ],
  },
];

export const Sidebar: React.FC = () => {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const { telemetry } = useHardwareTelemetry(5000);

  const deviceShort = telemetry?.device_name
    ? telemetry.device_name.replace("NVIDIA GeForce ", "").replace(" Laptop GPU", "")
    : "Detected Device";

  const ceilingStr = telemetry?.hardware_profile?.max_memory_allocated_gb
    ? `${telemetry.hardware_profile.max_memory_allocated_gb} GB`
    : "Auto";

  const profileMode = telemetry?.hardware_profile?.mode || "Balanced";

  return (
    <aside
      className={cn(
        "h-screen flex flex-col bg-[#0E1420] border-r border-[#273548] transition-all duration-300 z-30 select-none shadow-[4px_0_24px_rgba(0,0,0,0.3)]",
        collapsed ? "w-16" : "w-60"
      )}
    >
      {/* Brand & Identity Header */}
      <div
        className={cn(
          "h-14 flex items-center border-b border-[#273548] bg-[#090D14]/90 relative",
          collapsed ? "justify-center px-2" : "justify-between px-3.5"
        )}
      >
        <Link href="/" className="flex items-center gap-2.5 overflow-hidden group">
          <div className="w-8 h-8 rounded-lg bg-[#131C29] border border-[#45CFFF]/40 flex items-center justify-center shrink-0 shadow-[0_0_12px_rgba(69,207,255,0.2)] group-hover:border-[#45CFFF] transition-colors">
            <Zap className="w-4 h-4 text-[#45CFFF]" />
          </div>
          {!collapsed && (
            <div className="flex flex-col min-w-0">
              <span className="font-mono text-xs font-bold tracking-wider text-[#ECF3FC] flex items-center gap-1.5 truncate">
                LLM Maker
                <span className="text-[10px] px-1 py-0.2 rounded bg-[#45CFFF]/10 text-[#45CFFF] font-normal border border-[#45CFFF]/30">
                  NRI
                </span>
              </span>
              <span className="text-[9px] font-mono text-[#91A0B4] tracking-wide truncate">
                Neural Research Interface
              </span>
            </div>
          )}
        </Link>

        <button
          onClick={() => setCollapsed(!collapsed)}
          className={cn(
            "rounded transition-all text-[#91A0B4] hover:text-[#45CFFF]",
            collapsed
              ? "absolute -right-3 top-4 w-6 h-6 bg-[#131C29] border border-[#384C66] hover:border-[#45CFFF] rounded-full flex items-center justify-center shadow-lg z-40"
              : "p-1 hover:bg-[#182333]"
          )}
          title={collapsed ? "Expand navigation" : "Collapse navigation"}
        >
          {collapsed ? <ChevronRight className="w-3.5 h-3.5" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>

      {/* Nav List with Sections */}
      <nav className="flex-1 py-3 px-2 space-y-4 overflow-y-auto">
        {NAV_GROUPS.map((group) => (
          <div key={group.label} className="space-y-1">
            {!collapsed && (
              <div className="px-3 py-1 text-[10px] font-mono font-semibold uppercase tracking-wider text-[#5E7188]">
                {group.label}
              </div>
            )}
            {group.items.map((item) => {
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
                    "flex items-center gap-2.5 px-3 py-2 rounded-md text-xs font-mono transition-all duration-150 group relative",
                    isActive
                      ? "bg-[#131C29] text-[#ECF3FC] border border-[#384C66] shadow-[0_2px_8px_rgba(0,0,0,0.3)] font-semibold"
                      : "text-[#91A0B4] hover:text-[#ECF3FC] hover:bg-[#131C29]/60 border border-transparent"
                  )}
                  title={collapsed ? item.name : undefined}
                >
                  <Icon
                    className={cn(
                      "w-4 h-4 shrink-0 transition-colors",
                      isActive
                        ? "text-[#45CFFF]"
                        : "text-[#91A0B4] group-hover:text-[#ECF3FC]"
                    )}
                  />
                  {!collapsed && (
                    <span className="truncate tracking-wide font-sans text-xs">
                      {item.name}
                    </span>
                  )}
                  {!collapsed && item.tag && (
                    <span
                      className={cn(
                        "ml-auto text-[9px] px-1.5 py-0.2 rounded font-mono font-bold uppercase",
                        item.badge === "emerald"
                          ? "bg-[#35D49A]/10 text-[#35D49A] border border-[#35D49A]/30"
                          : item.badge === "violet"
                          ? "bg-[#9B8CFF]/10 text-[#9B8CFF] border border-[#9B8CFF]/30"
                          : "bg-[#45CFFF]/10 text-[#45CFFF] border border-[#45CFFF]/30"
                      )}
                    >
                      {item.tag}
                    </span>
                  )}
                  {collapsed && isActive && (
                    <div className="absolute right-0 top-1/2 -translate-y-1/2 w-1 h-5 bg-[#45CFFF] rounded-l shadow-[0_0_8px_rgba(69,207,255,0.6)]" />
                  )}
                </Link>
              );
            })}
          </div>
        ))}
      </nav>

      {/* Footer Hardware & Safety Widget */}
      <div className="p-3 border-t border-[#273548] bg-[#090D14]/80">
        {!collapsed ? (
          <div className="bg-[#131C29] p-2.5 rounded-lg border border-[#273548] space-y-1.5 font-mono text-[10px]">
            <div className="flex items-center justify-between text-[#91A0B4]">
              <span className="flex items-center gap-1.5">
                <Cpu className="w-3 h-3 text-[#45CFFF]" /> DEVICE
              </span>
              <span className="text-[#ECF3FC] font-semibold truncate max-w-[100px]" title={telemetry?.device_name}>
                {deviceShort}
              </span>
            </div>
            <div className="flex items-center justify-between text-[#91A0B4] pt-0.5 border-t border-[#273548]/80">
              <span className="flex items-center gap-1.5">
                <ShieldCheck className="w-3 h-3 text-[#35D49A]" /> CEILING
              </span>
              <span className="text-[#45CFFF] font-bold">
                {ceilingStr} ({profileMode})
              </span>
            </div>
          </div>
        ) : (
          <div className="flex justify-center text-[#91A0B4]" title={`GPU: ${deviceShort}`}>
            <span className="w-2 h-2 rounded-full bg-[#45CFFF] shadow-[0_0_6px_#45CFFF] animate-pulse" />
          </div>
        )}
      </div>
    </aside>
  );
};
