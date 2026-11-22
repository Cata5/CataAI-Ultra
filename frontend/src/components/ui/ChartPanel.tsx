"use client";

import React, { useEffect, useState } from "react";
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
import type { ChartOptions } from "chart.js";
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
  loading?: boolean;
  emptyTitle?: string;
  emptyHint?: string;
}

// Convert neon / bright colors to restrained grayscale series
function sanitizeGrayscaleColor(color: string, index: number): string {
  if (color.includes("45CFFF") || color.includes("cyan")) {
    return "#F2F2F2"; // Primary data: clean crisp white/light gray
  }
  if (color.includes("9B8CFF") || color.includes("violet")) {
    return "#A3A3A3"; // Secondary data: neutral mid gray
  }
  if (color.includes("35D49A") || color.includes("emerald") || color.includes("green")) {
    return "#D4D4D4"; // Validation / success: soft light gray
  }
  if (color.includes("F2B85B") || color.includes("amber")) {
    return "#8C8C8C"; // Auxiliary
  }
  // Fallbacks by index
  const palette = ["#F2F2F2", "#A3A3A3", "#737373", "#D4D4D4"];
  return color.startsWith("#") ? color : palette[index % palette.length];
}

export const ChartPanel: React.FC<ChartPanelProps> = ({
  title,
  labels,
  datasets,
  yAxisLabel,
  xAxisLabel = "Step",
  allowLogScale = true,
  className,
  height = 220,
  loading = false,
  emptyTitle = "No telemetry yet",
  emptyHint = "Data points appear here once the run emits metrics.",
}) => {
  const [isLogScale, setIsLogScale] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    if (!isFullscreen) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setIsFullscreen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isFullscreen]);

  const isEmpty = labels.length === 0 || datasets.length === 0;

  const chartData = {
    labels: labels.map(String),
    datasets: datasets.map((ds, idx) => ({
      label: ds.label,
      data: ds.data,
      borderColor: sanitizeGrayscaleColor(ds.borderColor, idx),
      backgroundColor: "transparent",
      fill: false,
      borderDash: ds.borderDash || (idx === 1 ? [4, 4] : idx === 2 ? [2, 2] : []),
      borderWidth: ds.borderWidth ?? (idx === 0 ? 1.6 : 1.2),
      pointRadius: ds.pointRadius ?? 0,
      tension: 0.1,
      spanGaps: ds.spanGaps ?? true,
    })),
  };

  const chartOptions: ChartOptions<"line"> = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 0,
    },
    interaction: {
      mode: "index",
      intersect: false,
    },
    scales: {
      x: {
        grid: {
          color: "rgba(255, 255, 255, 0.04)",
          lineWidth: 1,
        },
        ticks: {
          color: "#737373",
          font: { family: "monospace", size: 10 },
          maxTicksLimit: 8,
        },
        title: {
          display: !!xAxisLabel,
          text: xAxisLabel,
          color: "#525252",
          font: { size: 10, family: "monospace" },
        },
      },
      y: {
        type: isLogScale ? "logarithmic" : "linear",
        grid: {
          color: "rgba(255, 255, 255, 0.04)",
          lineWidth: 1,
        },
        ticks: {
          color: "#737373",
          font: { family: "monospace", size: 10 },
        },
        title: {
          display: !!yAxisLabel,
          text: yAxisLabel,
          color: "#525252",
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
          color: "#A3A3A3",
          font: { family: "monospace", size: 10 },
          boxWidth: 8,
          boxHeight: 2,
        },
      },
      tooltip: {
        backgroundColor: "#202020",
        titleColor: "#F2F2F2",
        bodyColor: "#A3A3A3",
        borderColor: "#333333",
        borderWidth: 1,
        bodyFont: { family: "monospace", size: 11 },
        titleFont: { family: "monospace", size: 11 },
        padding: 6,
        displayColors: true,
        cornerRadius: 4,
      },
    },
  };

  return (
    <section
      className={cn(
        "panel flex flex-col p-3 bg-[#202020] border-[#333333]",
        isFullscreen
          ? "fixed inset-3 z-50 p-4 bg-[#171717] md:inset-6"
          : "",
        className
      )}
      aria-label={title}
    >
      <div className="panel-header">
        <h3 className="panel-title">
          <span className="truncate">{title}</span>
        </h3>

        <div className="flex items-center gap-1">
          {allowLogScale && (
            <button
              type="button"
              onClick={() => setIsLogScale(!isLogScale)}
              aria-pressed={isLogScale}
              className="btn btn-ghost btn-sm px-1.5 text-[10px]"
              title="Toggle logarithmic scale"
            >
              LOG
            </button>
          )}

          <button
            type="button"
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="icon-btn w-6 h-6"
            aria-label={isFullscreen ? "Exit fullscreen" : "Expand chart fullscreen"}
            title={isFullscreen ? "Exit fullscreen (Esc)" : "Fullscreen"}
          >
            {isFullscreen ? (
              <Minimize2 className="w-3 h-3" />
            ) : (
              <Maximize2 className="w-3 h-3" />
            )}
          </button>
        </div>
      </div>

      <div
        style={{ height: isFullscreen ? "calc(100% - 40px)" : `${height}px` }}
        className="w-full min-h-0"
      >
        {loading ? (
          <div className="h-full flex flex-col justify-end gap-2 pb-2" aria-label="Loading chart">
            <span className="skeleton block h-full w-full opacity-40" />
          </div>
        ) : isEmpty ? (
          <div className="h-full flex flex-col items-center justify-center text-center gap-1.5 px-4">
            <ChartIcon className="w-4 h-4 text-[#404040]" aria-hidden="true" />
            <p className="font-mono text-[10px] font-medium uppercase tracking-wider text-[#737373]">
              {emptyTitle}
            </p>
            <p className="max-w-[220px] text-[10px] text-[#525252]">{emptyHint}</p>
          </div>
        ) : (
          <Line data={chartData} options={chartOptions} aria-label={`${title} chart`} role="img" />
        )}
      </div>
    </section>
  );
};
