"use client";

import React, { useState } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  LogarithmicScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";
import { Line } from "react-chartjs-2";
import { cn } from "@/lib/utils";
import { Maximize2, Minimize2 } from "lucide-react";

ChartJS.register(
  CategoryScale,
  LinearScale,
  LogarithmicScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface DatasetConfig {
  label: string;
  data: (number | null)[];
  borderColor: string;
  backgroundColor?: string;
  fill?: boolean;
  borderDash?: number[];
  borderWidth?: number;
  pointRadius?: number;
  spanGaps?: boolean;
}

interface ChartPanelProps {
  title: string;
  labels: (string | number)[];
  datasets: DatasetConfig[];
  yAxisLabel?: string;
  xAxisLabel?: string;
  allowLogScale?: boolean;
  className?: string;
  height?: number;
}

export const ChartPanel: React.FC<ChartPanelProps> = ({
  title,
  labels,
  datasets,
  yAxisLabel,
  xAxisLabel = "Step",
  allowLogScale = true,
  className,
  height = 240,
}) => {
  const [isLogScale, setIsLogScale] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const chartData = {
    labels: labels.map(String),
    datasets: datasets.map((ds) => ({
      label: ds.label,
      data: ds.data,
      borderColor: ds.borderColor,
      backgroundColor: ds.backgroundColor || "transparent",
      fill: ds.fill ?? false,
      borderDash: ds.borderDash || [],
      borderWidth: ds.borderWidth ?? 2,
      pointRadius: ds.pointRadius ?? 0,
      tension: 0.2,
      spanGaps: ds.spanGaps ?? true,
    })),
  };

  const chartOptions: any = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 0, // Zero animation duration to keep high-frequency telemetry non-blocking
    },
    scales: {
      x: {
        grid: {
          color: "rgba(56, 189, 248, 0.06)",
        },
        ticks: {
          color: "#94a3b8",
          font: { family: "monospace", size: 10 },
          maxTicksLimit: 10,
        },
        title: {
          display: !!xAxisLabel,
          text: xAxisLabel,
          color: "#64748b",
          font: { size: 10, family: "monospace" },
        },
      },
      y: {
        type: isLogScale ? "logarithmic" : "linear",
        grid: {
          color: "rgba(56, 189, 248, 0.08)",
        },
        ticks: {
          color: "#94a3b8",
          font: { family: "monospace", size: 10 },
        },
        title: {
          display: !!yAxisLabel,
          text: yAxisLabel,
          color: "#64748b",
          font: { size: 10, family: "monospace" },
        },
      },
    },
    plugins: {
      legend: {
        display: true,
        position: "top" as const,
        align: "end" as const,
        labels: {
          color: "#cbd5e1",
          font: { family: "monospace", size: 11 },
          boxWidth: 12,
          boxHeight: 2,
        },
      },
      tooltip: {
        backgroundColor: "rgba(10, 16, 31, 0.95)",
        titleColor: "#00f0ff",
        bodyColor: "#f1f5f9",
        borderColor: "rgba(56, 189, 248, 0.3)",
        borderWidth: 1,
        bodyFont: { family: "monospace", size: 11 },
        titleFont: { family: "monospace", size: 11 },
        padding: 8,
        displayColors: true,
      },
    },
  };

  return (
    <div
      className={cn(
        "glass-panel rounded-lg p-3.5 flex flex-col relative",
        isFullscreen ? "fixed inset-4 z-50 p-6" : "",
        className
      )}
    >
      <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-800/80">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-3 bg-cyan-400 rounded-sm shadow-[0_0_6px_#00f0ff]" />
          <h3 className="font-mono text-xs font-semibold text-slate-200 tracking-wider uppercase">
            {title}
          </h3>
        </div>

        <div className="flex items-center gap-1.5">
          {allowLogScale && (
            <button
              onClick={() => setIsLogScale(!isLogScale)}
              className={cn(
                "px-2 py-0.5 text-[10px] font-mono rounded border transition-colors",
                isLogScale
                  ? "bg-cyan-500/20 text-cyan-300 border-cyan-400/50"
                  : "bg-slate-900/60 text-slate-400 border-slate-800 hover:text-slate-300"
              )}
              title="Toggle Logarithmic Scale"
            >
              LOG
            </button>
          )}

          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-1 rounded text-slate-400 hover:text-cyan-400 hover:bg-slate-800/60 transition-colors"
            title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
          >
            {isFullscreen ? (
              <Minimize2 className="w-3.5 h-3.5" />
            ) : (
              <Maximize2 className="w-3.5 h-3.5" />
            )}
          </button>
        </div>
      </div>

      <div style={{ height: isFullscreen ? "calc(100% - 40px)" : `${height}px` }} className="w-full">
        {labels.length === 0 ? (
          <div className="h-full flex items-center justify-center font-mono text-xs text-slate-400">
            Awaiting telemetry data...
          </div>
        ) : (
          <Line data={chartData} options={chartOptions} />
        )}
      </div>
    </div>
  );
};
