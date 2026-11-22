import React from "react";
import { cn } from "@/lib/utils";
import { HardwareTelemetry } from "@/types/api";
import { Cpu, Zap, Thermometer, HardDrive, Activity } from "lucide-react";
import { InlineAlert } from "./InlineAlert";

interface TelemetryCardProps {
  telemetry: HardwareTelemetry | null;
  error?: string | null;
  isLoading?: boolean;
  className?: string;
}

export const TelemetryCard: React.FC<TelemetryCardProps> = ({
  telemetry,
  error,
  isLoading = false,
  className,
}) => {
  if (!telemetry) {
    if (error && !isLoading) {
      return (
        <div className={cn("panel p-3.5 bg-[#202020] border-[#333333]", className)}>
          <div className="panel-header">
            <h3 className="panel-title panel-title--muted">Hardware telemetry</h3>
          </div>
          <InlineAlert tone="warning" title="Telemetry unavailable" message={error}>
            <p className="mt-1 font-mono text-[10px] text-[#737373]">
              Retrying automatically every few seconds.
            </p>
          </InlineAlert>
        </div>
      );
    }

    return (
      <div
        className={cn("panel p-3.5 bg-[#202020] border-[#333333]", className)}
        aria-busy="true"
        aria-label="Loading hardware telemetry"
      >
        <div className="panel-header">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-[#737373]" />
            <span className="skeleton block h-3 w-32" />
          </div>
          <span className="skeleton block h-3 w-10" />
        </div>
        <div className="space-y-3">
          <span className="skeleton block h-1.5 w-full" />
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {[0, 1, 2, 3].map((i) => (
              <span key={i} className="skeleton block h-12 w-full" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  const usedMb = telemetry.used_vram_mb ?? (telemetry as any).memory_used_mb ?? telemetry.pytorch_allocated_mb ?? 0;
  const totalMb = telemetry.total_vram_mb ?? (telemetry as any).memory_total_mb ?? 6144;
  const vramPct = Math.min(100, Math.round(telemetry.vram_percent ?? (totalMb > 0 ? (usedMb / totalMb) * 100 : 0)));
  const temp = Math.round(telemetry.temperature_c ?? 0);
  const tempColor =
    temp >= 80 ? "text-[#f87171]" : temp >= 72 ? "text-[#facc15]" : "text-[#A3A3A3]";
  const vramBarColor =
    vramPct >= 90 ? "bg-[#f87171]" : vramPct >= 80 ? "bg-[#facc15]" : "bg-[#D4D4D4]";
  const gpuName = telemetry.device_name || (telemetry as any).gpu_name || "Detected Accelerator";
  const gpuUtil = telemetry.gpu_util_percent ?? (telemetry as any).gpu_utilization_pct ?? 0;
  const powerW = telemetry.power_draw_watts ?? (telemetry as any).power_draw_w ?? 0;
  const coreClock = telemetry.sm_clock_mhz ?? (telemetry as any).core_clock_mhz ?? 0;
  const ceiling = telemetry.hardware_profile?.max_memory_allocated_gb
    ? `${telemetry.hardware_profile.max_memory_allocated_gb} GB`
    : "Auto";

  return (
    <section
      className={cn("panel p-3.5 flex flex-col gap-3 bg-[#202020] border-[#333333]", className)}
      aria-label="Hardware telemetry"
    >
      <div className="panel-header">
        <div className="flex items-center gap-2 min-w-0">
          <Cpu className="w-4 h-4 text-[#A3A3A3] shrink-0" aria-hidden="true" />
          <h3
            className="font-mono text-[11px] font-medium tracking-wide text-[#F2F2F2] truncate"
            title={gpuName}
          >
            {gpuName}
          </h3>
        </div>
        <div className="flex items-center gap-1.5 font-mono text-xs text-[#737373] shrink-0">
          <Activity className="w-3.5 h-3.5 text-[#A3A3A3]" aria-hidden="true" />
          <span className="font-medium text-[#F2F2F2] font-mono-num">{gpuUtil}%</span>
          <span className="text-[10px] text-[#525252]">UTIL</span>
        </div>
      </div>

      {error && <InlineAlert tone="warning" message={`Polling issue: ${error}. Retrying…`} />}

      {/* VRAM Progress */}
      <div>
        <div className="flex justify-between items-center gap-3 font-mono mb-1">
          <span className="text-[#737373] flex items-center gap-1.5 text-[10px] uppercase tracking-wide shrink-0">
            <HardDrive className="w-3 h-3 text-[#737373]" aria-hidden="true" /> VRAM
          </span>
          <span className="text-[#F2F2F2] text-[11px] font-medium font-mono-num text-right">
            {usedMb.toFixed(0)} <span className="text-[#525252]">/</span> {totalMb.toFixed(0)} MB
            <span className="text-[#737373]"> ({vramPct}%)</span>
          </span>
        </div>
        <div
          className="w-full bg-[#1B1B1B] rounded-full h-1 overflow-hidden border border-[#262626]"
          role="progressbar"
          aria-valuenow={vramPct}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="VRAM allocation"
        >
          <div
            className={cn("h-full transition-all duration-300 rounded-full", vramBarColor)}
            style={{ width: `${vramPct}%` }}
          />
        </div>
      </div>

      {/* Secondary metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono">
        <div className="bg-[#1B1B1B] p-2 rounded border border-[#262626]">
          <div className="text-[#737373] text-[10px] flex items-center gap-1 mb-0.5 uppercase">
            <Thermometer className="w-3 h-3" aria-hidden="true" /> Temp
          </div>
          <div className={cn("font-medium text-xs font-mono-num", tempColor)}>
            {temp > 0 ? `${temp}°C` : "—"}
          </div>
        </div>

        <div className="bg-[#1B1B1B] p-2 rounded border border-[#262626]">
          <div className="text-[#737373] text-[10px] flex items-center gap-1 mb-0.5 uppercase">
            <Zap className="w-3 h-3" aria-hidden="true" /> Power
          </div>
          <div className="font-medium text-xs text-[#F2F2F2] font-mono-num">
            {powerW > 0 ? `${powerW.toFixed(0)}W` : "—"}
          </div>
        </div>

        <div className="bg-[#1B1B1B] p-2 rounded border border-[#262626]">
          <div className="text-[#737373] text-[10px] mb-0.5 uppercase">
            Core clk
          </div>
          <div className="font-medium text-xs text-[#F2F2F2] font-mono-num">
            {coreClock > 0 ? `${coreClock} MHz` : "—"}
          </div>
        </div>

        <div className="bg-[#1B1B1B] p-2 rounded border border-[#262626]">
          <div className="text-[#737373] text-[10px] mb-0.5 uppercase">
            Ceiling
          </div>
          <div className="font-medium text-xs text-[#A3A3A3] font-mono-num">{ceiling}</div>
        </div>
      </div>
    </section>
  );
};
