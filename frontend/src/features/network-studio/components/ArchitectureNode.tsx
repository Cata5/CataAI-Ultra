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

export const ArchitectureNode: React.FC<any> = memo(({ data, selected }: any) => {
  const nodeData = (data || {}) as ArchitectureNodeData;

  return (
    <div
      className={cn(
        "rounded p-2.5 min-w-[190px] border transition-colors text-xs font-mono select-none",
        selected
          ? "bg-[#262626] border-[#A3A3A3] shadow-md"
          : "bg-[#202020] border-[#333333] hover:border-[#404040]"
      )}
    >
      {/* Target input port */}
      <Handle
        type="target"
        position={Position.Top}
        className="!bg-[#737373] !border-[#171717] !w-2 !h-2"
      />

      {/* Node Header */}
      <div className="flex items-center justify-between pb-1.5 mb-1.5 border-b border-[#262626]">
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="p-0.5 rounded bg-[#1B1B1B] text-[#A3A3A3] shrink-0">
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
          <span className="font-medium text-[#F2F2F2] tracking-tight truncate">
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
              className="p-0.5 rounded text-[#737373] hover:text-[#F2F2F2] hover:bg-[#262626] transition-colors"
              title="Documentation"
            >
              <Info className="w-3 h-3" />
            </button>
          )}
          {nodeData.isValid !== false ? (
            <CheckCircle2 className="w-3 h-3 text-[#4ade80]" />
          ) : (
            <AlertCircle className="w-3 h-3 text-[#f87171]" />
          )}
        </div>
      </div>

      {/* Layer metadata */}
      <div className="space-y-0.5 text-[10px] text-[#737373]">
        <div className="flex justify-between items-center">
          <span>TYPE:</span>
          <span className="text-[#A3A3A3] font-medium">{nodeData.layerType}</span>
        </div>

        {nodeData.inputShape && (
          <div className="flex justify-between items-center">
            <span>SHAPE:</span>
            <span className="text-[#F2F2F2] font-mono-num">{nodeData.inputShape}</span>
          </div>
        )}

        {nodeData.params !== undefined && (
          <div className="flex justify-between items-center">
            <span>PARAMS:</span>
            <span className="text-[#A3A3A3] font-mono-num">
              {nodeData.params.toLocaleString()}
            </span>
          </div>
        )}
      </div>

      {/* Source output port */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="!bg-[#737373] !border-[#171717] !w-2 !h-2"
      />
    </div>
  );
});

ArchitectureNode.displayName = "ArchitectureNode";
