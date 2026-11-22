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
    <footer className="h-6 border-t border-[#333333] bg-[#141414] px-3 flex items-center justify-between text-[11px] font-mono text-[#737373] select-none shrink-0 z-20">
      {/* Left: Run Identity & Phase */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5">
          <span>RUN:</span>
          <span className="text-[#A3A3A3] font-medium max-w-[120px] truncate" title={runId}>
            {runId}
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          <span>PHASE:</span>
          <span className="text-[#A3A3A3] uppercase">{phase}</span>
        </div>

        <div className="flex items-center gap-1.5">
          <span>STEP:</span>
          <span className="text-[#F2F2F2] font-mono-num font-medium">
            {step.toLocaleString()}
          </span>
        </div>

        {loss !== null && (
          <div className="hidden sm:flex items-center gap-1.5">
            <span>LOSS:</span>
            <span className="text-[#F2F2F2] font-mono-num font-medium">
              {Number(loss).toFixed(4)}
            </span>
          </div>
        )}
      </div>

      {/* Right: Tokens, GPU, Telemetry */}
      <div className="flex items-center gap-3">
        <div className="hidden md:flex items-center gap-1.5">
          <Activity className="w-3 h-3 text-[#737373]" />
          <span>TOKENS:</span>
          <span className="text-[#A3A3A3] font-mono-num">
            {formatCompactNumber(tokens)}
          </span>
          {tokPerSec > 0 && (
            <span className="text-[#525252]">({Math.round(tokPerSec)} tok/s)</span>
          )}
        </div>

        {telemetry && (
          <>
            <div className="flex items-center gap-1.5">
              <HardDrive className="w-3 h-3 text-[#737373]" />
              <span>VRAM:</span>
              <span className="text-[#A3A3A3] font-mono-num">
                {Math.round(telemetry.memory_used_mb ?? telemetry.used_vram_mb ?? 0)} / {Math.round(telemetry.memory_total_mb ?? telemetry.total_vram_mb ?? 6144)} MB
              </span>
            </div>

            <div className="hidden sm:flex items-center gap-1.5">
              <Flame className="w-3 h-3 text-[#737373]" />
              <span className="text-[#A3A3A3] font-mono-num">
                {Math.round(telemetry.temperature_c ?? 0)}°C
              </span>
            </div>
          </>
        )}

        <div className="flex items-center gap-1.5">
          <Radio
            className={`w-3 h-3 ${isConnected ? "text-[#4ade80]" : "text-[#f87171]"}`}
          />
          <span className={isConnected ? "text-[#4ade80]" : "text-[#f87171]"}>
            {isConnected ? "CONNECTED" : "OFFLINE"}
          </span>
        </div>
      </div>
    </footer>
  );
};
