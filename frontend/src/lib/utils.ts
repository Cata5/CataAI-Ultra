import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNumber(val: number | undefined | null, decimals: number = 2): string {
  if (val === undefined || val === null || isNaN(val)) return "—";
  return val.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export function formatCompactNumber(val: number | undefined | null): string {
  if (val === undefined || val === null || isNaN(val)) return "—";
  if (val >= 1_000_000_000) return `${(val / 1_000_000_000).toFixed(2)}B`;
  if (val >= 1_000_000) return `${(val / 1_000_000).toFixed(2)}M`;
  if (val >= 1_000) return `${(val / 1_000).toFixed(1)}k`;
  return val.toLocaleString();
}

export function formatBytes(bytes: number | undefined | null): string {
  if (bytes === undefined || bytes === null || isNaN(bytes)) return "—";
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

/**
 * Adaptive duration formatter:
 * Under 1 hour: MM:SS (e.g. 08:42)
 * 1 hour or more: HH:MM:SS (e.g. 02:18:43, 26:14:10)
 * Does NOT wrap at 24 hours.
 */
export function formatDuration(seconds: number | undefined | null): string {
  if (seconds === undefined || seconds === null || isNaN(seconds) || seconds < 0) return "--:--";
  const totalSecs = Math.floor(seconds);
  const hrs = Math.floor(totalSecs / 3600);
  const mins = Math.floor((totalSecs % 3600) / 60);
  const secs = totalSecs % 60;
  const pad = (n: number) => n.toString().padStart(2, "0");
  if (hrs > 0) {
    return `${pad(hrs)}:${pad(mins)}:${pad(secs)}`;
  }
  return `${pad(mins)}:${pad(secs)}`;
}

/**
 * Milliseconds to step latency display
 */
export function formatStepLatency(ms: number | undefined | null): string {
  if (ms === undefined || ms === null || isNaN(ms) || ms < 0) return "—";
  if (ms < 1000) return `${ms.toFixed(0)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

/**
 * Adaptive parameter count formatting:
 * 48.53M, 3.84M, 488.7K, 12.4K
 */
export function formatParamCount(params: number | undefined | null): string {
  if (params === undefined || params === null || isNaN(params)) return "—";
  if (params >= 1_000_000_000) return `${(params / 1_000_000_000).toFixed(2)}B`;
  if (params >= 1_000_000) return `${(params / 1_000_000).toFixed(2)}M`;
  if (params >= 1_000) return `${(params / 1_000).toFixed(1)}K`;
  return params.toLocaleString();
}
