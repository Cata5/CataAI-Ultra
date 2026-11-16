"use client";

import React from "react";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useHardwareTelemetry } from "@/hooks/useHardwareTelemetry";
import { useSystemStatus } from "@/hooks/useSystemStatus";
import { formatCompactNumber } from "@/lib/utils";
import { Activity, Flame, HardDrive, Radio } from "lucide-react";

export const StatusBar: React.FC = () => {
  const { isConnected, latestMetric } = useWebSocket();
  const { status } = useSystemStatus(4000);
  const { telemetry } = useHardwareTelemetry(2500);

  const runId = status?.run_id || "None";
  const step = latestMetric?.step ?? status?.latest_metrics?.step ?? 0;
  const tokens = latestMetric?.tokens_seen ?? status?.latest_metrics?.tokens_seen ?? 0;
  const tokPerSec = latestMetric?.tokens_per_second ?? status?.latest_metrics?.tokens_per_second ?? 0;
  const loss = latestMetric?.loss ?? status?.latest_metrics?.loss ?? null;
  const phase = latestMetric?.phase ?? status?.latest_metrics?.phase ?? "IDLE";

  return (
    <footer className="h-7 border-t border-[#273548] bg-[#090D14] px-4 flex items-center justify-between text-[11px] font-mono text-[#91A0B4] select-none shrink-0 z-20">
      {/* Left: Run Identity & Phase */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-1.5">
          <span className="text-[#5E7188]">RUN:</span>
          <span className="text-[#ECF3FC] font-bold max-w-[140px] truncate" title={runId}>
            {runId}
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          <span className="text-[#5E7188]">PHASE:</span>
          <span className="text-[#45CFFF] font-bold uppercase">{phase}</span>
        </div>

        <div className="flex items-center gap-1.5">
          <span className="text-[#5E7188]">STEP:</span>
          <span className="text-[#ECF3FC] font-mono-num font-bold">
            {step.toLocaleString()}
          </span>
        </div>

        {loss !== null && (
          <div className="hidden sm:flex items-center gap-1.5">
            <span className="text-[#5E7188]">LOSS:</span>
            <span className="text-[#35D49A] font-mono-num font-bold">
              {Number(loss).toFixed(4)}
            </span>
          </div>
        )}
      </div>

      {/* Right: Tokens, GPU, Telemetry */}
      <div className="flex items-center gap-4">
        <div className="hidden md:flex items-center gap-1.5">
          <Activity className="w-3 h-3 text-[#45CFFF]" />
          <span>TOKENS:</span>
          <span className="text-[#ECF3FC] font-bold font-mono-num">
            {formatCompactNumber(tokens)}
          </span>
          {tokPerSec > 0 && (
            <span className="text-[#5E7188]">({Math.round(tokPerSec)} tok/s)</span>
          )}
        </div>

        {telemetry && (
          <>
            <div className="flex items-center gap-1.5">
              <HardDrive className="w-3 h-3 text-[#45CFFF]" />
              <span>VRAM:</span>
              <span className="text-[#ECF3FC] font-bold font-mono-num">
                {Math.round(telemetry.memory_used_mb ?? telemetry.used_vram_mb ?? 0)} / {Math.round(telemetry.memory_total_mb ?? telemetry.total_vram_mb ?? 6144)} MB
              </span>
            </div>

            <div className="hidden sm:flex items-center gap-1.5">
              <Flame className="w-3 h-3 text-[#F2B85B]" />
              <span className="text-[#ECF3FC] font-bold font-mono-num">
                {Math.round(telemetry.temperature_c ?? 0)}°C
              </span>
            </div>
          </>
        )}

        <div className="flex items-center gap-1.5">
          <Radio
            className={`w-3 h-3 ${isConnected ? "text-[#35D49A] animate-pulse" : "text-[#FF657A]"}`}
          />
          <span className={isConnected ? "text-[#35D49A]" : "text-[#FF657A]"}>
            {isConnected ? "WS 8081" : "DISCONNECTED"}
          </span>
        </div>
      </div>
    </footer>
  );
};
