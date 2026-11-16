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
import { Maximize2, Minimize2, LineChart as ChartIcon } from "lucide-react";

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
      borderWidth: ds.borderWidth ?? 1.8,
      pointRadius: ds.pointRadius ?? 0,
      tension: 0.15,
      spanGaps: ds.spanGaps ?? true,
    })),
  };

  const chartOptions: any = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 0,
    },
    scales: {
      x: {
        grid: {
          color: "rgba(39, 53, 72, 0.45)",
          lineWidth: 1,
        },
        ticks: {
          color: "#91A0B4",
          font: { family: "monospace", size: 10 },
          maxTicksLimit: 10,
        },
        title: {
          display: !!xAxisLabel,
          text: xAxisLabel,
          color: "#5E7188",
          font: { size: 10, family: "monospace" },
        },
      },
      y: {
        type: isLogScale ? "logarithmic" : "linear",
        grid: {
          color: "rgba(39, 53, 72, 0.45)",
          lineWidth: 1,
        },
        ticks: {
          color: "#91A0B4",
          font: { family: "monospace", size: 10 },
        },
        title: {
          display: !!yAxisLabel,
          text: yAxisLabel,
          color: "#5E7188",
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
          color: "#91A0B4",
          font: { family: "monospace", size: 11 },
          boxWidth: 10,
          boxHeight: 2,
        },
      },
      tooltip: {
        backgroundColor: "#182333",
        titleColor: "#45CFFF",
        bodyColor: "#ECF3FC",
        borderColor: "#384C66",
        borderWidth: 1,
        bodyFont: { family: "monospace", size: 11 },
        titleFont: { family: "monospace", size: 11 },
        padding: 8,
        displayColors: true,
        cornerRadius: 6,
      },
    },
  };

  return (
    <div
      className={cn(
        "glass-panel rounded-lg p-3.5 flex flex-col relative overflow-hidden",
        isFullscreen ? "fixed inset-4 z-50 p-6 bg-[#0E1420]/95 backdrop-blur-md" : "",
        className
      )}
    >
      {/* Top Hairline Highlight */}
      <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#45CFFF]/15 to-transparent pointer-events-none" />

      <div className="flex items-center justify-between pb-2 mb-2 border-b border-[#273548]">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-3 bg-[#45CFFF] rounded-sm shadow-[0_0_6px_rgba(69,207,255,0.4)]" />
          <h3 className="font-mono text-xs font-semibold text-[#ECF3FC] tracking-wider uppercase">
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
                  ? "bg-[#45CFFF]/15 text-[#45CFFF] border-[#45CFFF]/50"
                  : "bg-[#0E1420] text-[#91A0B4] border-[#273548] hover:text-[#ECF3FC] hover:border-[#384C66]"
              )}
              title="Toggle Logarithmic Scale"
            >
              LOG
            </button>
          )}

          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-1 rounded text-[#91A0B4] hover:text-[#45CFFF] hover:bg-[#182333] transition-colors"
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
          <div className="h-full flex flex-col items-center justify-center font-mono text-xs text-[#5E7188] gap-1.5">
            <ChartIcon className="w-5 h-5 text-[#384C66]" />
            <span>Awaiting telemetry stream...</span>
          </div>
        ) : (
          <Line data={chartData} options={chartOptions} />
        )}
      </div>
    </div>
  );
};