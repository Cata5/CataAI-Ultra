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
  category: "embedding" | "attention" | "feedforward" | "norm" | "structural" | "convolution";
  layerType: string;
  config: Record<string, any>;
  params?: number;
  inputShape?: string;
  outputShape?: string;
  isValid?: boolean;
  onShowInfo?: (layerType: string) => void;
}

const CATEGORY_STYLES: Record<string, { border: string; accent: string; text: string; bg: string }> = {
  embedding: {
    border: "border-l-[#45CFFF]",
    accent: "#45CFFF",
    text: "text-[#45CFFF]",
    bg: "bg-[#45CFFF]/10",
  },
  attention: {
    border: "border-l-[#9B8CFF]",
    accent: "#9B8CFF",
    text: "text-[#9B8CFF]",
    bg: "bg-[#9B8CFF]/10",
  },
  feedforward: {
    border: "border-l-[#35D49A]",
    accent: "#35D49A",
    text: "text-[#35D49A]",
    bg: "bg-[#35D49A]/10",
  },
  norm: {
    border: "border-l-[#F2B85B]",
    accent: "#F2B85B",
    text: "text-[#F2B85B]",
    bg: "bg-[#F2B85B]/10",
  },
  convolution: {
    border: "border-l-[#80DEFF]",
    accent: "#80DEFF",
    text: "text-[#80DEFF]",
    bg: "bg-[#80DEFF]/10",
  },
  structural: {
    border: "border-l-[#8291A5]",
    accent: "#8291A5",
    text: "text-[#8291A5]",
    bg: "bg-[#8291A5]/10",
  },
};

export const ArchitectureNode: React.FC<any> = memo(({ data, selected }: any) => {
  const nodeData = (data || {}) as ArchitectureNodeData;
  const style = CATEGORY_STYLES[nodeData.category] || CATEGORY_STYLES.structural;

  return (
    <div
      className={cn(
        "rounded-lg p-2.5 min-w-[200px] bg-[#131C29] border border-[#273548] border-l-[3px] transition-all duration-150 shadow-md text-xs font-mono relative overflow-hidden",
        style.border,
        selected
          ? "ring-1 ring-[#45CFFF] border-[#45CFFF]/50 shadow-[0_0_16px_rgba(69,207,255,0.25)]"
          : "hover:border-[#384C66] hover:shadow-[0_4px_12px_rgba(0,0,0,0.3)]"
      )}
    >
      {/* Target input port */}
      <Handle
        type="target"
        position={Position.Top}
        className="!bg-[#45CFFF] !border-[#090D14] !w-2.5 !h-2.5 shadow-[0_0_6px_rgba(69,207,255,0.6)]"
      />

      {/* Node Header */}
      <div className="flex items-center justify-between pb-1.5 mb-1.5 border-b border-[#273548]">
        <div className="flex items-center gap-1.5 min-w-0">
          <span className={cn("p-1 rounded shrink-0", style.bg, style.text)}>
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
          <span className="font-semibold text-[#ECF3FC] tracking-wide truncate">
            {nodeData.label}
          </span>
        </div>

        <div className="flex items-center gap-1 shrink-0 ml-1">
          {nodeData.onShowInfo && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                nodeData.onShowInfo?.(nodeData.layerType);
              }}
              className="p-0.5 rounded text-[#91A0B4] hover:text-[#45CFFF] hover:bg-[#182333] transition-colors"
              title="View architectural docs"
            >
              <Info className="w-3 h-3" />
            </button>
          )}
          {nodeData.isValid !== false ? (
            <CheckCircle2 className="w-3 h-3 text-[#35D49A]" />
          ) : (
            <AlertCircle className="w-3 h-3 text-[#FF657A]" />
          )}
        </div>
      </div>

      {/* Layer metadata */}
      <div className="space-y-1 text-[10px] text-[#91A0B4]">
        <div className="flex justify-between items-center">
          <span className="text-[#5E7188]">TYPE:</span>
          <span className="text-[#ECF3FC] font-semibold">{nodeData.layerType}</span>
        </div>

        {nodeData.inputShape && (
          <div className="flex justify-between items-center">
            <span className="text-[#5E7188]">SHAPE:</span>
            <span className="text-[#45CFFF] font-bold font-mono-num">{nodeData.inputShape}</span>
          </div>
        )}

        {nodeData.params !== undefined && (
          <div className="flex justify-between items-center">
            <span className="text-[#5E7188]">PARAMS:</span>
            <span className="text-[#9B8CFF] font-semibold font-mono-num">
              {nodeData.params.toLocaleString()}
            </span>
          </div>
        )}
      </div>

      {/* Source output port */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="!bg-[#45CFFF] !border-[#090D14] !w-2.5 !h-2.5 shadow-[0_0_6px_rgba(69,207,255,0.6)]"
      />
    </div>
  );
});

ArchitectureNode.displayName = "ArchitectureNode";