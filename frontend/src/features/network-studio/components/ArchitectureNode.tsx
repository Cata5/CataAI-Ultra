"use client";

import React, { memo } from "react";
import { Handle, Position } from "@xyflow/react";
import { cn } from "@/lib/utils";
import {
  Layers,
  Sparkles,
  Cpu,
  Binary,
  CheckCircle2,
  AlertCircle,
  Info,
} from "lucide-react";

export interface ArchitectureNodeData {
  label: string;
  category: "embedding" | "attention" | "feedforward" | "norm" | "structural";
  layerType: string;
  config: Record<string, any>;
  params?: number;
  inputShape?: string;
  outputShape?: string;
  isValid?: boolean;
  onShowInfo?: (layerType: string) => void;
}

const CATEGORY_COLORS: Record<string, { border: string; glow: string; text: string; bg: string }> = {
  embedding: {
    border: "border-sky-500/60",
    glow: "rgba(56, 189, 248, 0.25)",
    text: "text-sky-400",
    bg: "bg-sky-950/40",
  },
  attention: {
    border: "border-cyan-400",
    glow: "rgba(0, 240, 255, 0.35)",
    text: "text-cyan-400",
    bg: "bg-cyan-950/40",
  },
  feedforward: {
    border: "border-indigo-400",
    glow: "rgba(129, 140, 248, 0.3)",
    text: "text-indigo-400",
    bg: "bg-indigo-950/40",
  },
  norm: {
    border: "border-emerald-400",
    glow: "rgba(16, 185, 129, 0.25)",
    text: "text-emerald-400",
    bg: "bg-emerald-950/40",
  },
  structural: {
    border: "border-purple-400",
    glow: "rgba(192, 132, 252, 0.25)",
    text: "text-purple-400",
    bg: "bg-purple-950/40",
  },
};

export const ArchitectureNode: React.FC<any> = memo(({ data, selected }: any) => {
  const nodeData = (data || {}) as ArchitectureNodeData;
  const theme = CATEGORY_COLORS[nodeData.category] || CATEGORY_COLORS.structural;

  return (
    <div
      className={cn(
        "rounded-lg p-2.5 min-w-[190px] glass-panel border transition-all duration-200 shadow-lg text-xs font-mono relative",
        theme.border,
        selected
          ? "ring-2 ring-cyan-400 shadow-[0_0_15px_rgba(0,240,255,0.4)]"
          : "hover:shadow-[0_0_10px_rgba(0,240,255,0.2)]"
      )}
      style={{
        boxShadow: selected ? `0 0 15px ${theme.glow}` : undefined,
      }}
    >
      {/* Target input port */}
      <Handle
        type="target"
        position={Position.Top}
        className="!bg-cyan-400 !border-white !w-2.5 !h-2.5"
      />

      {/* Node Header */}
      <div className="flex items-center justify-between pb-1.5 mb-1.5 border-b border-slate-800">
        <div className="flex items-center gap-1.5">
          <span className={cn("p-1 rounded", theme.bg, theme.text)}>
            {nodeData.category === "attention" ? (
              <Sparkles className="w-3 h-3" />
            ) : nodeData.category === "embedding" ? (
              <Binary className="w-3 h-3" />
            ) : nodeData.category === "feedforward" ? (
              <Cpu className="w-3 h-3" />
            ) : (
              <Layers className="w-3 h-3" />
            )}
          </span>
          <span className="font-bold text-slate-100 tracking-wide">
            {nodeData.label}
          </span>
        </div>

        <div className="flex items-center gap-1">
          {nodeData.onShowInfo && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                nodeData.onShowInfo?.(nodeData.layerType);
              }}
              className="p-0.5 rounded text-slate-400 hover:text-cyan-400 hover:bg-slate-800 transition-colors"
              title="Layer Architecture Info"
            >
              <Info className="w-3 h-3" />
            </button>
          )}
          {nodeData.isValid !== false ? (
            <CheckCircle2 className="w-3 h-3 text-emerald-400" />
          ) : (
            <AlertCircle className="w-3 h-3 text-rose-400" />
          )}
        </div>
      </div>

      {/* Layer metadata */}
      <div className="space-y-1 text-[10px] text-slate-400">
        <div className="flex justify-between">
          <span>Type:</span>
          <span className="text-slate-300 font-semibold">{nodeData.layerType}</span>
        </div>

        {nodeData.inputShape && (
          <div className="flex justify-between">
            <span>Shape:</span>
            <span className="text-cyan-400 font-bold">{nodeData.inputShape}</span>
          </div>
        )}

        {nodeData.params !== undefined && (
          <div className="flex justify-between">
            <span>Params:</span>
            <span className="text-indigo-300 font-semibold">
              {nodeData.params.toLocaleString()}
            </span>
          </div>
        )}
      </div>

      {/* Source output port */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="!bg-cyan-400 !border-white !w-2.5 !h-2.5"
      />
    </div>
  );
});

ArchitectureNode.displayName = "ArchitectureNode";
