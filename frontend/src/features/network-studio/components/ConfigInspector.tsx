"use client";

import React from "react";
import { Sliders, Settings2, Trash2, Save, Copy, Info, Plus } from "lucide-react";
import { cn, formatParamCount } from "@/lib/utils";

interface ConfigInspectorProps {
  selectedNode: any | null;
  onUpdateNodeConfig: (nodeId: string, newConfig: Record<string, any>) => void;
  onDeleteNode: (nodeId: string) => void;
  globalConfig: {
    id: string;
    name: string;
    task_type: string;
    architecture_type: string;
    vocab_size: number;
    d_model: number;
    n_layers: number;
    n_q: number;
    n_kv: number;
    d_ff: number;
    kernel_size: number;
    tied_embeddings: boolean;
  };
  onUpdateGlobalConfig: (updates: Partial<any>) => void;
  onSaveNetwork: () => void;
  onCloneNetwork: () => void;
  onLoadTemplate: (templateId: string) => void;
  onNewScratch?: () => void;
  onDeleteNetwork?: () => void;
  onShowInfo?: (layerType: string) => void;
  templates: any[];
}

export const ConfigInspector: React.FC<ConfigInspectorProps> = ({
  selectedNode,
  onUpdateNodeConfig,
  onDeleteNode,
  globalConfig,
  onUpdateGlobalConfig,
  onSaveNetwork,
  onCloneNetwork,
  onLoadTemplate,
  onNewScratch,
  onDeleteNetwork,
  onShowInfo,
  templates,
}) => {
  if (selectedNode) {
    const { id, data } = selectedNode;
    const config = data.config || {};

    const handleChange = (key: string, value: any) => {
      onUpdateNodeConfig(id, { ...config, [key]: value });
    };

    return (
      <div className="w-72 glass-panel rounded-lg p-3.5 flex flex-col h-full overflow-hidden select-none font-mono text-xs">
        <div className="flex items-center justify-between pb-2 mb-3 border-b border-slate-800">
          <div className="flex items-center gap-1.5">
            <Sliders className="w-3.5 h-3.5 text-cyan-400" />
            <span className="font-bold text-slate-100 uppercase tracking-wider">
              NODE INSPECTOR
            </span>
          </div>
          <button
            onClick={() => onDeleteNode(id)}
            className="p-1 rounded text-rose-400 hover:bg-rose-950/40 hover:text-rose-300 transition-colors"
            title="Delete Node"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto space-y-3 pr-1">
          <div>
            <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
              Layer Label
            </label>
            <input
              type="text"
              value={data.label || ""}
              onChange={(e) =>
                onUpdateNodeConfig(id, { ...config, label: e.target.value })
              }
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1 text-slate-200 focus:outline-none focus:border-cyan-400"
            />
          </div>

          <div>
            <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
              Layer Type
            </label>
            <div className="flex items-center justify-between px-2.5 py-1 bg-slate-950/60 rounded border border-slate-800 text-cyan-400 font-bold">
              <span>{data.layerType}</span>
              {onShowInfo && (
                <button
                  type="button"
                  onClick={() => onShowInfo(data.layerType)}
                  className="p-0.5 rounded text-slate-400 hover:text-cyan-400 hover:bg-slate-800 transition-colors"
                  title="View Architectural Documentation"
                >
                  <Info className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          </div>

          {/* Dynamic Configuration fields */}
          {Object.entries(config).map(([k, v]) => {
            if (k === "label") return null;
            const isNumber = typeof v === "number";
            const isBool = typeof v === "boolean";

            return (
              <div key={k}>
                <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
                  {k.replace(/_/g, " ")}
                </label>
                {isBool ? (
                  <button
                    type="button"
                    onClick={() => handleChange(k, !v)}
                    className={cn(
                      "px-3 py-1 rounded text-xs border w-full text-left font-bold transition-colors",
                      v
                        ? "bg-cyan-950/50 text-cyan-300 border-cyan-500/40"
                        : "bg-slate-900 text-slate-400 border-slate-800"
                    )}
                  >
                    {v ? "ENABLED (TRUE)" : "DISABLED (FALSE)"}
                  </button>
                ) : (
                  <input
                    type={isNumber ? "number" : "text"}
                    value={String(v)}
                    onChange={(e) =>
                      handleChange(
                        k,
                        isNumber ? parseFloat(e.target.value) || 0 : e.target.value
                      )
                    }
                    className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1 text-slate-200 focus:outline-none focus:border-cyan-400"
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  // Global Architecture Inspector
  return (
    <div className="w-72 glass-panel rounded-lg p-3.5 flex flex-col h-full overflow-hidden select-none font-mono text-xs">
      <div className="flex items-center justify-between pb-2 mb-3 border-b border-slate-800">
        <div className="flex items-center gap-1.5">
          <Settings2 className="w-3.5 h-3.5 text-cyan-400" />
          <span className="font-bold text-slate-100 uppercase tracking-wider">
            ARCHITECTURE CONFIG
          </span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto space-y-3 pr-1">
        {/* Template Presets or New Scratch */}
        <div>
          {onNewScratch && (
            <button
              type="button"
              onClick={onNewScratch}
              className="w-full mb-2.5 py-1.5 px-3 rounded bg-cyan-950/60 border border-cyan-500/40 hover:bg-cyan-900/60 text-cyan-300 font-bold text-xs flex items-center justify-center gap-1.5 transition-all shadow-[0_0_8px_rgba(0,240,255,0.15)]"
            >
              <Plus className="w-3.5 h-3.5" /> Start New From Scratch
            </button>
          )}
          <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
            Load Template Preset
          </label>
          <select
            onChange={(e) => e.target.value && onLoadTemplate(e.target.value)}
            defaultValue=""
            className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1 text-slate-200 focus:outline-none focus:border-cyan-400 text-xs"
          >
            <option value="" disabled>
              Select Architecture Preset...
            </option>
            {templates.map((t) => {
              const pCount = t.total_params || t.total_parameters || t.params || 0;
              return (
                <option key={t.id} value={t.id}>
                  {t.name} {pCount > 0 ? `(${formatParamCount(pCount)})` : ""}
                </option>
              );
            })}
          </select>
        </div>

        {/* Network ID & Name */}
        <div>
          <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
            Network Identifier
          </label>
          <input
            type="text"
            value={globalConfig.id}
            onChange={(e) => onUpdateGlobalConfig({ id: e.target.value })}
            className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1 text-slate-200 focus:outline-none focus:border-cyan-400"
          />
        </div>

        <div>
          <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
            Display Name
          </label>
          <input
            type="text"
            value={globalConfig.name}
            onChange={(e) => onUpdateGlobalConfig({ name: e.target.value })}
            className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1 text-slate-200 focus:outline-none focus:border-cyan-400"
          />
        </div>

        {/* Task Type */}
        <div>
          <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
            Training Task
          </label>
          <select
            value={globalConfig.task_type}
            onChange={(e) => onUpdateGlobalConfig({ task_type: e.target.value })}
            className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1 text-slate-200 focus:outline-none focus:border-cyan-400 text-xs"
          >
            <option value="causal_lm">Causal LM (Autoregressive)</option>
            <option value="masked_lm">Masked LM (BERT-style)</option>
            <option value="classification">Sequence Classification</option>
            <option value="regression">Feature Regression</option>
          </select>
        </div>

        {/* Hyperparameter Inputs */}
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
              d_model
            </label>
            <input
              type="number"
              value={globalConfig.d_model}
              onChange={(e) =>
                onUpdateGlobalConfig({ d_model: parseInt(e.target.value) || 512 })
              }
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1 text-slate-200"
            />
          </div>
          <div>
            <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
              n_layers
            </label>
            <input
              type="number"
              value={globalConfig.n_layers}
              onChange={(e) =>
                onUpdateGlobalConfig({ n_layers: parseInt(e.target.value) || 16 })
              }
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1 text-slate-200"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
              n_q heads
            </label>
            <input
              type="number"
              value={globalConfig.n_q}
              onChange={(e) =>
                onUpdateGlobalConfig({ n_q: parseInt(e.target.value) || 8 })
              }
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1 text-slate-200"
            />
          </div>
          <div>
            <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
              n_kv heads
            </label>
            <input
              type="number"
              value={globalConfig.n_kv}
              onChange={(e) =>
                onUpdateGlobalConfig({ n_kv: parseInt(e.target.value) || 2 })
              }
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1 text-slate-200"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
              d_ff
            </label>
            <input
              type="number"
              value={globalConfig.d_ff}
              onChange={(e) =>
                onUpdateGlobalConfig({ d_ff: parseInt(e.target.value) || 1280 })
              }
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1 text-slate-200"
            />
          </div>
          <div>
            <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
              kernel_size
            </label>
            <input
              type="number"
              value={globalConfig.kernel_size}
              onChange={(e) =>
                onUpdateGlobalConfig({ kernel_size: parseInt(e.target.value) || 4 })
              }
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1 text-slate-200"
            />
          </div>
        </div>

        <div>
          <label className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
            vocab_size
          </label>
          <input
            type="number"
            value={globalConfig.vocab_size}
            onChange={(e) =>
              onUpdateGlobalConfig({ vocab_size: parseInt(e.target.value) || 12288 })
            }
            className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1 text-slate-200"
          />
        </div>

        <div>
          <button
            type="button"
            onClick={() =>
              onUpdateGlobalConfig({ tied_embeddings: !globalConfig.tied_embeddings })
            }
            className={cn(
              "px-3 py-1.5 rounded text-xs border w-full text-left font-bold transition-colors",
              globalConfig.tied_embeddings
                ? "bg-cyan-950/50 text-cyan-300 border-cyan-500/40"
                : "bg-slate-900 text-slate-400 border-slate-800"
            )}
          >
            Tied Embeddings: {globalConfig.tied_embeddings ? "ENABLED (RoCanon default)" : "DISABLED"}
          </button>
        </div>
      </div>

      {/* Save, Clone & Delete Actions */}
      <div className="pt-3 border-t border-slate-800 flex items-center gap-2">
        <button
          onClick={onSaveNetwork}
          className="flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded bg-cyan-600 hover:bg-cyan-500 text-white font-bold transition-all shadow-[0_0_10px_rgba(0,240,255,0.3)]"
        >
          <Save className="w-3.5 h-3.5" /> SAVE
        </button>
        <button
          onClick={onCloneNetwork}
          className="inline-flex items-center justify-center p-1.5 rounded bg-slate-900 border border-slate-700 hover:border-cyan-400 text-slate-200 transition-colors"
          title="Clone Network"
        >
          <Copy className="w-3.5 h-3.5" />
        </button>
        {onDeleteNetwork && (
          <button
            onClick={onDeleteNetwork}
            className="inline-flex items-center justify-center p-1.5 rounded bg-rose-950/40 border border-rose-800/80 hover:bg-rose-900/60 text-rose-300 transition-colors"
            title="Delete Architecture from Registry"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
    </div>
  );
};
