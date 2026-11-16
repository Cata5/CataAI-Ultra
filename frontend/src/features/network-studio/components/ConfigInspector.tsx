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
  registeredNetworks?: any[];
  isSaving?: boolean;
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
  registeredNetworks = [],
  isSaving = false,
}) => {
  // Determine if currently selected architecture is custom (user-created) or built-in template
  const defaultTemplateIds = [
    "rocanon_48m",
    "rocanon_48_5m",
    "mini_gpt_15m",
    "bert_encoder_25m",
    "conv1d_network_4m",
    "deep_mlp_5m",
    "hybrid_cnn_transformer_20m",
  ];
  const isCustom = !defaultTemplateIds.includes(globalConfig.id);

  // Group registered networks vs templates
  const customNetworks = registeredNetworks.filter((n) => !defaultTemplateIds.includes(n.id));
  const baseTemplates = templates.length > 0 ? templates : registeredNetworks.filter((n) => defaultTemplateIds.includes(n.id));

  if (selectedNode) {
    const { id, data } = selectedNode;
    const config = data.config || {};

    const handleChange = (key: string, value: any) => {
      onUpdateNodeConfig(id, { ...config, [key]: value });
    };

    return (
      <div className="w-72 glass-panel rounded-lg p-3.5 flex flex-col h-full overflow-hidden select-none font-mono text-xs relative">
        <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#45CFFF]/15 to-transparent pointer-events-none" />

        <div className="flex items-center justify-between pb-2 mb-3 border-b border-[#273548]">
          <div className="flex items-center gap-1.5">
            <Sliders className="w-3.5 h-3.5 text-[#45CFFF]" />
            <span className="font-semibold text-[#ECF3FC] uppercase tracking-wider">
              NODE INSPECTOR
            </span>
          </div>
          <button
            onClick={() => onDeleteNode(id)}
            className="p-1 rounded text-[#FF657A] hover:bg-[#FF657A]/15 transition-colors"
            title="Delete Node"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto space-y-3 pr-1">
          <div>
            <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
              Layer Label
            </label>
            <input
              type="text"
              value={data.label || ""}
              onChange={(e) =>
                onUpdateNodeConfig(id, { ...config, label: e.target.value })
              }
              className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2.5 py-1 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF]/50"
            />
          </div>

          <div>
            <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
              Layer Type
            </label>
            <div className="flex items-center justify-between px-2.5 py-1 bg-[#0E1420] rounded-md border border-[#273548] text-[#45CFFF] font-bold">
              <span>{data.layerType}</span>
              {onShowInfo && (
                <button
                  type="button"
                  onClick={() => onShowInfo(data.layerType)}
                  className="p-0.5 rounded text-[#91A0B4] hover:text-[#45CFFF] hover:bg-[#182333] transition-colors"
                  title="View Architectural Documentation"
                >
                  <Info className="w-3 h-3" />
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
                <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
                  {k.replace(/_/g, " ")}
                </label>
                {isBool ? (
                  <button
                    type="button"
                    onClick={() => handleChange(k, !v)}
                    className={cn(
                      "px-3 py-1 rounded-md text-xs border w-full text-left font-bold transition-colors",
                      v
                        ? "bg-[#45CFFF]/15 text-[#45CFFF] border-[#45CFFF]/40"
                        : "bg-[#0E1420] text-[#91A0B4] border-[#273548]"
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
                    className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2.5 py-1 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF]/50"
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
    <div className="w-72 glass-panel rounded-lg p-3.5 flex flex-col h-full overflow-hidden select-none font-mono text-xs relative">
      <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#45CFFF]/15 to-transparent pointer-events-none" />

      <div className="flex items-center justify-between pb-2 mb-3 border-b border-[#273548]">
        <div className="flex items-center gap-1.5">
          <Settings2 className="w-3.5 h-3.5 text-[#45CFFF]" />
          <span className="font-semibold text-[#ECF3FC] uppercase tracking-wider">
            ARCHITECTURE CONFIG
          </span>
        </div>
        {isCustom ? (
          <span className="text-[9px] px-2 py-0.5 rounded bg-[#45CFFF]/15 border border-[#45CFFF]/30 text-[#45CFFF] font-bold">
            CUSTOM
          </span>
        ) : (
          <span className="text-[9px] px-2 py-0.5 rounded bg-[#91A0B4]/15 border border-[#91A0B4]/30 text-[#91A0B4] font-bold">
            TEMPLATE
          </span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto space-y-3 pr-1">
        {/* Template Presets or New Scratch */}
        <div>
          {onNewScratch && (
            <button
              type="button"
              onClick={onNewScratch}
              className="w-full mb-2.5 py-1.5 px-3 rounded-md bg-[#45CFFF]/10 border border-[#45CFFF]/40 hover:bg-[#45CFFF]/20 text-[#45CFFF] font-bold text-xs flex items-center justify-center gap-1.5 transition-all shadow-sm"
            >
              <Plus className="w-3.5 h-3.5" /> Start New From Scratch
            </button>
          )}

          <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
            Select Architecture to Edit
          </label>
          <select
            value={globalConfig.id}
            onChange={(e) => e.target.value && onLoadTemplate(e.target.value)}
            className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2.5 py-1 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF] text-xs font-mono"
          >
            {customNetworks.length > 0 && (
              <optgroup label="MY REGISTERED ARCHITECTURES">
                {customNetworks.map((t) => {
                  const pCount = t.total_params || t.total_parameters || t.params || 0;
                  return (
                    <option key={t.id} value={t.id}>
                      {t.name} {pCount > 0 ? `(${formatParamCount(pCount)})` : ""}
                    </option>
                  );
                })}
              </optgroup>
            )}
            <optgroup label="BASE TEMPLATES">
              {baseTemplates.map((t) => {
                const pCount = t.total_params || t.total_parameters || t.params || 0;
                return (
                  <option key={t.id} value={t.id}>
                    {t.name} {pCount > 0 ? `(${formatParamCount(pCount)})` : ""}
                  </option>
                );
              })}
            </optgroup>
          </select>
        </div>

        {/* Network ID & Name */}
        <div>
          <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
            Network Identifier (ID)
          </label>
          <input
            type="text"
            value={globalConfig.id}
            onChange={(e) => onUpdateGlobalConfig({ id: e.target.value })}
            className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2.5 py-1 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF]/50"
          />
        </div>

        <div>
          <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
            Display Name
          </label>
          <input
            type="text"
            value={globalConfig.name}
            onChange={(e) => onUpdateGlobalConfig({ name: e.target.value })}
            className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2.5 py-1 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF]/50"
          />
        </div>

        {/* Task Type */}
        <div>
          <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
            Training Task
          </label>
          <select
            value={globalConfig.task_type}
            onChange={(e) => onUpdateGlobalConfig({ task_type: e.target.value })}
            className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2.5 py-1 text-[#ECF3FC] focus:outline-none focus:border-[#45CFFF]/50 text-xs"
          >
            <option value="causal_lm">Causal LM (Autoregressive Next-Token)</option>
            <option value="masked_lm">Masked LM (BERT-style)</option>
            <option value="classification">Sequence Classification</option>
            <option value="regression">Feature Regression</option>
          </select>
        </div>

        {/* Hyperparameter Inputs */}
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
              d_model
            </label>
            <input
              type="number"
              value={globalConfig.d_model}
              onChange={(e) =>
                onUpdateGlobalConfig({ d_model: parseInt(e.target.value) || 512 })
              }
              className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2.5 py-1 text-[#ECF3FC]"
            />
          </div>
          <div>
            <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
              n_layers
            </label>
            <input
              type="number"
              value={globalConfig.n_layers}
              onChange={(e) =>
                onUpdateGlobalConfig({ n_layers: parseInt(e.target.value) || 16 })
              }
              className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2.5 py-1 text-[#ECF3FC]"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
              n_q (Query Heads)
            </label>
            <input
              type="number"
              value={globalConfig.n_q}
              onChange={(e) =>
                onUpdateGlobalConfig({ n_q: parseInt(e.target.value) || 8 })
              }
              className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2.5 py-1 text-[#ECF3FC]"
            />
          </div>
          <div>
            <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
              n_kv (KV Heads)
            </label>
            <input
              type="number"
              value={globalConfig.n_kv}
              onChange={(e) =>
                onUpdateGlobalConfig({ n_kv: parseInt(e.target.value) || 2 })
              }
              className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2.5 py-1 text-[#ECF3FC]"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
              d_ff (FFN Dim)
            </label>
            <input
              type="number"
              value={globalConfig.d_ff}
              onChange={(e) =>
                onUpdateGlobalConfig({ d_ff: parseInt(e.target.value) || 1280 })
              }
              className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2.5 py-1 text-[#ECF3FC]"
            />
          </div>
          <div>
            <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
              kernel_size
            </label>
            <input
              type="number"
              value={globalConfig.kernel_size}
              onChange={(e) =>
                onUpdateGlobalConfig({ kernel_size: parseInt(e.target.value) || 4 })
              }
              className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2.5 py-1 text-[#ECF3FC]"
            />
          </div>
        </div>

        <div>
          <label className="text-[10px] text-[#91A0B4] uppercase font-semibold block mb-1">
            vocab_size
          </label>
          <input
            type="number"
            value={globalConfig.vocab_size}
            onChange={(e) =>
              onUpdateGlobalConfig({ vocab_size: parseInt(e.target.value) || 12288 })
            }
            className="w-full bg-[#0E1420] border border-[#273548] rounded-md px-2.5 py-1 text-[#ECF3FC]"
          />
        </div>

        <div>
          <button
            type="button"
            onClick={() =>
              onUpdateGlobalConfig({ tied_embeddings: !globalConfig.tied_embeddings })
            }
            className={cn(
              "px-3 py-1.5 rounded-md text-xs border w-full text-left font-bold transition-colors",
              globalConfig.tied_embeddings
                ? "bg-[#45CFFF]/10 text-[#45CFFF] border-[#45CFFF]/30"
                : "bg-[#0E1420] text-[#91A0B4] border-[#273548]"
            )}
          >
            Tied Embeddings: {globalConfig.tied_embeddings ? "ENABLED" : "DISABLED"}
          </button>
        </div>
      </div>

      {/* Save, Clone & Delete Actions */}
      <div className="pt-3 border-t border-[#273548] flex items-center gap-2">
        <button
          onClick={onSaveNetwork}
          disabled={isSaving}
          className="flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md bg-[#45CFFF] hover:bg-[#80DEFF] text-[#090D14] font-bold transition-all shadow-sm disabled:opacity-50"
        >
          <Save className="w-3.5 h-3.5" />
          <span>{isSaving ? "SAVING..." : isCustom ? "SAVE CHANGES" : "SAVE TO REGISTRY"}</span>
        </button>
        <button
          onClick={onCloneNetwork}
          className="inline-flex items-center justify-center p-1.5 rounded-md bg-[#0E1420] border border-[#273548] hover:border-[#384C66] text-[#ECF3FC] transition-colors"
          title="Clone as New Architecture"
        >
          <Copy className="w-3.5 h-3.5" />
        </button>
        {onDeleteNetwork && isCustom && (
          <button
            onClick={onDeleteNetwork}
            className="inline-flex items-center justify-center p-1.5 rounded-md bg-[#FF657A]/15 border border-[#FF657A]/30 hover:bg-[#FF657A]/25 text-[#FF657A] transition-colors"
            title="Delete Architecture from Registry"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
    </div>
  );
};
