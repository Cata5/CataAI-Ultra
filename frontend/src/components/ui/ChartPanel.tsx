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
  /** Renders a skeleton plot while the first payload is in flight. */
  loading?: boolean;
  emptyTitle?: string;
  emptyHint?: string;
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
          color: "rgba(39, 53, 72, 0.35)",
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
          color: "rgba(39, 53, 72, 0.35)",
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
    <section
      className={cn(
        "panel flex flex-col p-3.5",
        isFullscreen
          ? "fixed inset-3 z-50 p-5 bg-[var(--bg-base)] md:inset-6"
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
              className="btn btn-ghost btn-sm px-2"
              title="Toggle logarithmic scale"
            >
              LOG
            </button>
          )}

          <button
            type="button"
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="icon-btn"
            aria-label={isFullscreen ? "Exit fullscreen" : "Expand chart fullscreen"}
            title={isFullscreen ? "Exit fullscreen (Esc)" : "Fullscreen"}
          >
            {isFullscreen ? (
              <Minimize2 className="w-3.5 h-3.5" />
            ) : (
              <Maximize2 className="w-3.5 h-3.5" />
            )}
          </button>
        </div>
      </div>

      <div
        style={{ height: isFullscreen ? "calc(100% - 44px)" : `${height}px` }}
        className="w-full min-h-0"
      >
        {loading ? (
          <div className="h-full flex flex-col justify-end gap-2 pb-2" aria-label="Loading chart">
            <span className="skeleton block h-full w-full opacity-60" />
          </div>
        ) : isEmpty ? (
          <div className="h-full flex flex-col items-center justify-center text-center gap-2 px-4">
            <ChartIcon className="w-5 h-5 text-[#384C66]" aria-hidden="true" />
            <p className="font-mono text-[11px] font-semibold uppercase tracking-wider text-[#91A0B4]">
              {emptyTitle}
            </p>
            <p className="max-w-[260px] text-[11px] leading-relaxed text-[#5E7188]">{emptyHint}</p>
          </div>
        ) : (
          <Line data={chartData} options={chartOptions} aria-label={`${title} chart`} role="img" />
        )}
      </div>
    </section>
  );
};
