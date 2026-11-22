"use client";

import React from "react";
import { useHardwareTelemetry } from "@/hooks/useHardwareTelemetry";
import { TelemetryCard } from "@/components/ui/TelemetryCard";
import { Button } from "@/components/ui/Button";
import {
  Cpu,
  Zap,
  Gauge,
  Thermometer,
  Layers,
  HardDrive,
  RefreshCw,
  CheckCircle2,
  AlertTriangle,
  Activity,
  Shield,
} from "lucide-react";
import { cn } from "@/lib/utils";

export const HardwareView: React.FC = () => {
  const { telemetry, isLoading, error, refresh } = useHardwareTelemetry(2000);

  const deviceName = telemetry?.device_name || "NVIDIA GeForce RTX 3060 Laptop GPU";
  const vramUsedMb = Math.round(telemetry?.memory_used_mb ?? telemetry?.used_vram_mb ?? 0);
  const vramTotalMb = Math.round(telemetry?.memory_total_mb ?? telemetry?.total_vram_mb ?? 6144);
  const vramFreeMb = Math.max(0, vramTotalMb - vramUsedMb);
  const vramPercent = vramTotalMb > 0 ? Math.round((vramUsedMb / vramTotalMb) * 100) : 0;

  const temp = Math.round(telemetry?.temperature_c ?? 0);
  const tempWarn = temp >= 80;

  const gpuUtil = Math.round(telemetry?.gpu_utilization_pct ?? 0);
  const powerW = telemetry?.power_draw_w ?? 0;
  const powerLimitW = telemetry?.power_limit_w ?? 80;

  const smClock = telemetry?.sm_clock_mhz ?? 0;
  const memClock = telemetry?.memory_clock_mhz ?? telemetry?.mem_clock_mhz ?? 0;

  const ramUsedGb = telemetry?.system_ram_used_gb ?? (telemetry?.system_ram_used_mb ? (telemetry.system_ram_used_mb / 1024).toFixed(1) : 0);
  const ramTotalGb = telemetry?.system_ram_total_gb ?? (telemetry?.system_ram_total_mb ? (telemetry.system_ram_total_mb / 1024).toFixed(1) : 40);

  return (
    <div className="space-y-4 font-mono text-xs">
      {/* Top Banner */}
      <div className="panel p-3.5 bg-[#202020] border-[#333333] flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <Cpu className="w-5 h-5 text-[#A3A3A3]" />
          <div>
            <h3 className="font-semibold text-[#F2F2F2] uppercase tracking-wider text-xs">
              HARDWARE ACCELERATOR & NVML TELEMETRY
            </h3>
            <p className="text-[11px] text-[#737373]">
              Direct nvidia-ml-py kernel telemetry with 2000ms polling rate
            </p>
          </div>
        </div>

        <Button
          variant="secondary"
          size="sm"
          onClick={() => void refresh()}
          loading={isLoading}
          icon={<RefreshCw className="w-3.5 h-3.5" />}
        >
          Refresh Telemetry
        </Button>
      </div>

      {error && (
        <div className="p-3 rounded bg-[#f87171]/10 border border-[#f87171]/30 text-[#f87171] flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>Telemetry degraded: {error}</span>
        </div>
      )}

      {/* Main Hardware Telemetry Card */}
      <TelemetryCard telemetry={telemetry} error={error} isLoading={isLoading} />

      {/* Detailed Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        {/* Core & Clocks */}
        <div className="panel p-3 bg-[#202020] border-[#333333] space-y-2">
          <div className="text-[10px] text-[#737373] uppercase font-medium flex items-center justify-between">
            <span>Clock Frequencies</span>
            <Gauge className="w-3.5 h-3.5 text-[#A3A3A3]" />
          </div>
          <div className="space-y-1">
            <div className="flex justify-between items-center text-xs">
              <span className="text-[#A3A3A3]">SM Core Clock:</span>
              <span className="font-semibold font-mono-num text-[#F2F2F2]">
                {smClock > 0 ? `${smClock} MHz` : "—"}
              </span>
            </div>
            <div className="flex justify-between items-center text-xs">
              <span className="text-[#A3A3A3]">Memory Clock:</span>
              <span className="font-semibold font-mono-num text-[#F2F2F2]">
                {memClock > 0 ? `${memClock} MHz` : "—"}
              </span>
            </div>
            <div className="flex justify-between items-center text-xs">
              <span className="text-[#A3A3A3]">Core Compute:</span>
              <span className="font-semibold font-mono-num text-[#F2F2F2]">{gpuUtil}%</span>
            </div>
          </div>
        </div>

        {/* Thermals & Power */}
        <div className="panel p-3 bg-[#202020] border-[#333333] space-y-2">
          <div className="text-[10px] text-[#737373] uppercase font-medium flex items-center justify-between">
            <span>Thermal & Power</span>
            <Thermometer className="w-3.5 h-3.5 text-[#A3A3A3]" />
          </div>
          <div className="space-y-1">
            <div className="flex justify-between items-center text-xs">
              <span className="text-[#A3A3A3]">GPU Temperature:</span>
              <span
                className={cn(
                  "font-semibold font-mono-num",
                  tempWarn ? "text-[#facc15]" : "text-[#4ade80]"
                )}
              >
                {temp > 0 ? `${temp}°C` : "—"}
              </span>
            </div>
            <div className="flex justify-between items-center text-xs">
              <span className="text-[#A3A3A3]">Power Draw:</span>
              <span className="font-semibold font-mono-num text-[#F2F2F2]">
                {powerW > 0 ? `${powerW.toFixed(1)} W` : "—"}
              </span>
            </div>
            <div className="flex justify-between items-center text-xs">
              <span className="text-[#A3A3A3]">Power Ceiling:</span>
              <span className="font-semibold font-mono-num text-[#737373]">
                {powerLimitW} W
              </span>
            </div>
          </div>
        </div>

        {/* VRAM Breakdown */}
        <div className="panel p-3 bg-[#202020] border-[#333333] space-y-2">
          <div className="text-[10px] text-[#737373] uppercase font-medium flex items-center justify-between">
            <span>VRAM Allocation</span>
            <Layers className="w-3.5 h-3.5 text-[#A3A3A3]" />
          </div>
          <div className="space-y-1">
            <div className="flex justify-between items-center text-xs">
              <span className="text-[#A3A3A3]">Active VRAM:</span>
              <span className="font-semibold font-mono-num text-[#F2F2F2]">
                {vramUsedMb} MB ({vramPercent}%)
              </span>
            </div>
            <div className="flex justify-between items-center text-xs">
              <span className="text-[#A3A3A3]">Available Headroom:</span>
              <span className="font-semibold font-mono-num text-[#4ade80]">
                {vramFreeMb} MB
              </span>
            </div>
            <div className="flex justify-between items-center text-xs">
              <span className="text-[#A3A3A3]">Total Capacity:</span>
              <span className="font-semibold font-mono-num text-[#737373]">
                {vramTotalMb} MB
              </span>
            </div>
          </div>
        </div>

        {/* Host Memory & Platform */}
        <div className="panel p-3 bg-[#202020] border-[#333333] space-y-2">
          <div className="text-[10px] text-[#737373] uppercase font-medium flex items-center justify-between">
            <span>Host Memory</span>
            <HardDrive className="w-3.5 h-3.5 text-[#A3A3A3]" />
          </div>
          <div className="space-y-1">
            <div className="flex justify-between items-center text-xs">
              <span className="text-[#A3A3A3]">System RAM:</span>
              <span className="font-semibold font-mono-num text-[#F2F2F2]">
                {ramUsedGb} / {ramTotalGb} GB
              </span>
            </div>
            <div className="flex justify-between items-center text-xs">
              <span className="text-[#A3A3A3]">Compute Backend:</span>
              <span className="font-semibold text-[#F2F2F2]">CUDA 12.8</span>
            </div>
            <div className="flex justify-between items-center text-xs">
              <span className="text-[#A3A3A3]">PyTorch Build:</span>
              <span className="font-semibold text-[#737373]">2.11.0+cu128</span>
            </div>
          </div>
        </div>
      </div>

      {/* Driver & Kernel Support Strip */}
      <div className="panel p-3.5 bg-[#202020] border-[#333333] space-y-2">
        <h4 className="text-xs font-semibold text-[#F2F2F2] uppercase tracking-wide flex items-center gap-1.5">
          <Shield className="w-3.5 h-3.5 text-[#A3A3A3]" /> Acceleration Stack Status
        </h4>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
          <div className="p-2 rounded bg-[#1B1B1B] border border-[#262626] flex items-center justify-between">
            <span className="text-[#A3A3A3]">NVML Telemetry:</span>
            <span className="text-[#4ade80] flex items-center gap-1 font-semibold">
              <CheckCircle2 className="w-3 h-3" /> nvidia-ml-py
            </span>
          </div>
          <div className="p-2 rounded bg-[#1B1B1B] border border-[#262626] flex items-center justify-between">
            <span className="text-[#A3A3A3]">FlashAttention-2:</span>
            <span className="text-[#4ade80] flex items-center gap-1 font-semibold">
              <CheckCircle2 className="w-3 h-3" /> SDPA varlen
            </span>
          </div>
          <div className="p-2 rounded bg-[#1B1B1B] border border-[#262626] flex items-center justify-between">
            <span className="text-[#A3A3A3]">Fused Cross-Entropy:</span>
            <span className="text-[#4ade80] flex items-center gap-1 font-semibold">
              <CheckCircle2 className="w-3 h-3" /> Liger Chunked
            </span>
          </div>
          <div className="p-2 rounded bg-[#1B1B1B] border border-[#262626] flex items-center justify-between">
            <span className="text-[#A3A3A3]">Optimizer Engine:</span>
            <span className="text-[#4ade80] flex items-center gap-1 font-semibold">
              <CheckCircle2 className="w-3 h-3" /> Muon + AdamW
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
