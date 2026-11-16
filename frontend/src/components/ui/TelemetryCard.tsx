import React from "react";
import { cn } from "@/lib/utils";
import { HardwareTelemetry } from "@/types/api";
import { Cpu, Zap, Thermometer, HardDrive, Activity, ShieldCheck } from "lucide-react";

interface TelemetryCardProps {
  telemetry: HardwareTelemetry | null;
  className?: string;
}

export const TelemetryCard: React.FC<TelemetryCardProps> = ({
  telemetry,
  className,
}) => {
  if (!telemetry) {
    return (
      <div className={cn("glass-panel rounded-lg p-4 animate-pulse", className)}>
        <div className="h-4 bg-[#1E2B3E] rounded w-1/3 mb-4" />
        <div className="h-8 bg-[#1E2B3E] rounded w-full mb-2" />
        <div className="h-8 bg-[#1E2B3E] rounded w-full" />
      </div>
    );
  }

  const usedMb = telemetry.used_vram_mb ?? (telemetry as any).memory_used_mb ?? telemetry.pytorch_allocated_mb ?? 0;
  const totalMb = telemetry.total_vram_mb ?? (telemetry as any).memory_total_mb ?? 6144;
  const vramPct = Math.min(100, Math.round(telemetry.vram_percent ?? (totalMb > 0 ? (usedMb / totalMb) * 100 : 0)));
  const temp = Math.round(telemetry.temperature_c ?? 0);
  const tempColor =
    temp >= 80 ? "text-[#FF657A]" : temp >= 72 ? "text-[#F2B85B]" : "text-[#35D49A]";
  const vramBarColor =
    vramPct >= 90 ? "bg-[#FF657A]" : vramPct >= 80 ? "bg-[#F2B85B]" : "bg-[#45CFFF]";
  const gpuName = telemetry.device_name || (telemetry as any).gpu_name || "Detected Accelerator";
  const gpuUtil = telemetry.gpu_util_percent ?? (telemetry as any).gpu_utilization_pct ?? 0;
  const powerW = telemetry.power_draw_watts ?? (telemetry as any).power_draw_w ?? 0;
  const coreClock = telemetry.sm_clock_mhz ?? (telemetry as any).core_clock_mhz ?? 0;
  const ceiling = telemetry.hardware_profile?.max_memory_allocated_gb
    ? `${telemetry.hardware_profile.max_memory_allocated_gb} GB`
    : "Auto";

  return (
    <div className={cn("glass-panel rounded-lg p-4 space-y-4 relative overflow-hidden", className)}>
      {/* Top Hairline Highlight */}
      <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#45CFFF]/20 to-transparent" />

      {/* Header */}
      <div className="flex items-center justify-between border-b border-[#273548] pb-2.5">
        <div className="flex items-center gap-2">
          <Cpu className="w-4 h-4 text-[#45CFFF]" />
          <span className="font-mono text-xs font-semibold tracking-wide text-[#ECF3FC] truncate">
            {gpuName}
          </span>
        </div>
        <div className="flex items-center gap-1.5 font-mono text-xs text-[#91A0B4]">
          <Activity className="w-3.5 h-3.5 text-[#45CFFF]" />
          <span className="font-bold text-[#ECF3FC] font-mono-num">{gpuUtil}%</span>
          <span className="text-[10px] text-[#5E7188]">UTIL</span>
        </div>
      </div>

      {/* VRAM Bar */}
      <div>
        <div className="flex justify-between items-center text-xs font-mono mb-1.5">
          <span className="text-[#91A0B4] flex items-center gap-1.5 text-[11px]">
            <HardDrive className="w-3.5 h-3.5 text-[#5E7188]" /> VRAM ALLOCATION
          </span>
          <span className="text-[#ECF3FC] font-semibold font-mono-num">
            {usedMb.toFixed(0)} MB <span className="text-[#5E7188]">/</span> {totalMb.toFixed(0)} MB ({vramPct}%)
          </span>
        </div>
        <div className="w-full bg-[#0E1420] rounded-full h-2 overflow-hidden border border-[#273548]">
          <div
            className={cn("h-full transition-all duration-500 rounded-full", vramBarColor)}
            style={{ width: `${vramPct}%`, boxShadow: "0 0 8px rgba(69,207,255,0.3)" }}
          />
        </div>
      </div>

      {/* Multi-Telemetry Grid */}
      <div className="grid grid-cols-4 gap-2 pt-0.5 font-mono text-xs">
        <div className="bg-[#0E1420] p-2 rounded-md border border-[#273548]">
          <div className="text-[#5E7188] text-[10px] flex items-center gap-1 mb-1 font-semibold uppercase">
            <Thermometer className="w-3 h-3" /> TEMP
          </div>
          <div className={cn("font-bold text-sm font-mono-num", tempColor)}>{temp > 0 ? `${temp}°C` : "—"}</div>
        </div>

        <div className="bg-[#0E1420] p-2 rounded-md border border-[#273548]">
          <div className="text-[#5E7188] text-[10px] flex items-center gap-1 mb-1 font-semibold uppercase">
            <Zap className="w-3 h-3" /> POWER
          </div>
          <div className="font-bold text-sm text-[#ECF3FC] font-mono-num">
            {powerW > 0 ? `${powerW.toFixed(0)}W` : "—"}
          </div>
        </div>

        <div className="bg-[#0E1420] p-2 rounded-md border border-[#273548]">
          <div className="text-[#5E7188] text-[10px] mb-1 font-semibold uppercase">CORE CLK</div>
          <div className="font-bold text-sm text-[#45CFFF] font-mono-num">
            {coreClock > 0 ? `${coreClock} MHz` : "—"}
          </div>
        </div>

        <div className="bg-[#0E1420] p-2 rounded-md border border-[#273548]">
          <div className="text-[#5E7188] text-[10px] mb-1 font-semibold uppercase">CEILING</div>
          <div className="font-bold text-sm text-[#35D49A] font-mono-num">
            {ceiling}
          </div>
        </div>
      </div>
    </div>
  );
};