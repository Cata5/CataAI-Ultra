import React from "react";
import { cn } from "@/lib/utils";
import { HardwareTelemetry } from "@/types/api";
import { Cpu, Zap, Thermometer, HardDrive, Activity } from "lucide-react";

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
        <div className="h-4 bg-slate-800 rounded w-1/3 mb-4" />
        <div className="h-8 bg-slate-800 rounded w-full mb-2" />
        <div className="h-8 bg-slate-800 rounded w-full" />
      </div>
    );
  }

  const usedMb = telemetry.used_vram_mb ?? (telemetry as any).memory_used_mb ?? telemetry.pytorch_allocated_mb ?? 0;
  const totalMb = telemetry.total_vram_mb ?? (telemetry as any).memory_total_mb ?? 6144;
  const vramPct = Math.min(100, Math.round(telemetry.vram_percent ?? (totalMb > 0 ? (usedMb / totalMb) * 100 : 0)));
  const temp = Math.round(telemetry.temperature_c ?? 0);
  const tempColor =
    temp >= 80 ? "text-rose-400" : temp >= 72 ? "text-amber-400" : "text-emerald-400";
  const vramBarColor =
    vramPct >= 90 ? "bg-rose-500" : vramPct >= 80 ? "bg-amber-400" : "bg-cyan-400";
  const gpuName = telemetry.device_name || (telemetry as any).gpu_name || "NVIDIA RTX 3060 Laptop (6GB)";
  const gpuUtil = telemetry.gpu_util_percent ?? (telemetry as any).gpu_utilization_pct ?? 0;
  const powerW = telemetry.power_draw_watts ?? (telemetry as any).power_draw_w ?? 0;
  const coreClock = telemetry.sm_clock_mhz ?? (telemetry as any).core_clock_mhz ?? 0;

  return (
    <div className={cn("glass-panel rounded-lg p-4 space-y-4 relative", className)}>
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
        <div className="flex items-center gap-2">
          <Cpu className="w-4 h-4 text-cyan-400" />
          <span className="font-mono text-sm font-semibold tracking-wide text-slate-200 truncate">
            {gpuName}
          </span>
        </div>
        <div className="flex items-center gap-1.5 font-mono text-xs text-slate-400">
          <Activity className="w-3.5 h-3.5 text-cyan-400" />
          <span>{gpuUtil}% UTIL</span>
        </div>
      </div>

      {/* VRAM Bar */}
      <div>
        <div className="flex justify-between items-center text-xs font-mono mb-1.5">
          <span className="text-slate-400 flex items-center gap-1">
            <HardDrive className="w-3.5 h-3.5 text-slate-400" /> VRAM USAGE
          </span>
          <span className="text-slate-200 font-bold">
            {usedMb.toFixed(0)} MB / {totalMb.toFixed(0)} MB ({vramPct}%)
          </span>
        </div>
        <div className="w-full bg-slate-900/90 rounded-full h-2 overflow-hidden border border-slate-800">
          <div
            className={cn("h-full transition-all duration-500", vramBarColor)}
            style={{ width: `${vramPct}%`, boxShadow: "0 0 8px rgba(0,240,255,0.4)" }}
          />
        </div>
      </div>

      {/* Multi-Telemetry Grid */}
      <div className="grid grid-cols-4 gap-2 pt-1 font-mono text-xs">
        <div className="bg-slate-900/60 p-2 rounded border border-slate-800/80">
          <div className="text-slate-400 flex items-center gap-1 mb-1">
            <Thermometer className="w-3 h-3" /> TEMP
          </div>
          <div className={cn("font-bold text-sm", tempColor)}>{temp > 0 ? `${temp}°C` : "—"}</div>
        </div>

        <div className="bg-slate-900/60 p-2 rounded border border-slate-800/80">
          <div className="text-slate-400 flex items-center gap-1 mb-1">
            <Zap className="w-3 h-3" /> POWER
          </div>
          <div className="font-bold text-sm text-slate-200">
            {powerW > 0 ? `${powerW.toFixed(0)}W` : "—"}
          </div>
        </div>

        <div className="bg-slate-900/60 p-2 rounded border border-slate-800/80">
          <div className="text-slate-400 mb-1">CORE CLK</div>
          <div className="font-bold text-sm text-cyan-400">
            {coreClock > 0 ? `${coreClock} MHz` : "—"}
          </div>
        </div>

        <div className="bg-slate-900/60 p-2 rounded border border-slate-800/80">
          <div className="text-slate-400 mb-1">MEM CLK</div>
          <div className="font-bold text-sm text-indigo-400">
            {(telemetry.memory_clock_mhz ?? 0) > 0 ? `${telemetry.memory_clock_mhz} MHz` : "—"}
          </div>
        </div>
      </div>

      <div className="absolute top-0 right-0 w-3 h-3 border-t-2 border-r-2 border-cyan-400/30" />
    </div>
  );
};
