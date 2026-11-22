"use client";

import React from "react";
import { Sliders, Trash2, Save, Copy, Info, Plus } from "lucide-react";
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
  const defaultTemplateIds = [
    "rocanon_48m",
    "mini_gpt_15m",
    "bert_encoder_25m",
    "conv1d_network_4m",
    "deep_mlp_5m",
    "hybrid_cnn_transformer_20m",
  ];
  const isCustom = !defaultTemplateIds.includes(globalConfig.id);

  const customNetworks = registeredNetworks.filter((n) => !defaultTemplateIds.includes(n.id));
  const baseTemplates = templates.length > 0 ? templates : registeredNetworks.filter((n) => defaultTemplateIds.includes(n.id));

  if (selectedNode) {
    const { id, data } = selectedNode;
    const config = data.config || {};

    const handleChange = (key: string, value: any) => {
      onUpdateNodeConfig(id, { ...config, [key]: value });
    };

    return (
      <div className="w-72 panel bg-[#202020] border-[#333333] rounded-lg p-3 flex flex-col h-full overflow-hidden select-none font-mono text-xs">
        <div className="flex items-center justify-between pb-2 mb-2.5 border-b border-[#262626]">
          <div className="flex items-center gap-1.5">
            <Sliders className="w-3.5 h-3.5 text-[#737373]" />
            <span className="font-semibold text-[#F2F2F2] uppercase tracking-wide text-xs">
              Node Inspector
            </span>
          </div>
          <button
            onClick={() => onDeleteNode(id)}
            className="p-1 rounded text-[#f87171] hover:bg-[rgba(248,113,113,0.1)] transition-colors"
            title="Delete Node"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto space-y-2.5 pr-0.5">
          <div>
            <label className="text-[10px] text-[#737373] uppercase font-medium block mb-1">
              Layer Label
            </label>
            <input
              type="text"
              value={data.label || ""}
              onChange={(e) => onUpdateNodeConfig(id, { ...config, label: e.target.value })}
              className="input"
            />
          </div>

          <div>
            <label className="text-[10px] text-[#737373] uppercase font-medium block mb-1">
              Layer Type
            </label>
            <div className="flex items-center justify-between p-2 rounded bg-[#1B1B1B] border border-[#262626]">
              <span className="text-[#F2F2F2] font-semibold">{data.layerType}</span>
              {onShowInfo && (
                <button
                  type="button"
                  onClick={() => onShowInfo(data.layerType)}
                  className="text-[#737373] hover:text-[#F2F2F2] transition-colors"
                  title="View layer documentation"
                >
                  <Info className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          </div>

          {/* Dynamic Layer Config Fields */}
          <div className="pt-2 border-t border-[#262626] space-y-2">
            <span className="text-[10px] text-[#737373] uppercase font-medium block">
              Parameters
            </span>

            {Object.keys(config).map((key) => {
              if (key === "label") return null;
              const val = config[key];
              const isNum = typeof val === "number";
              const isBool = typeof val === "boolean";

              return (
                <div key={key}>
                  <label className="text-[10px] text-[#A3A3A3] block mb-0.5">{key}</label>
                  {isBool ? (
                    <input
                      type="checkbox"
                      checked={val}
                      onChange={(e) => handleChange(key, e.target.checked)}
                      className="accent-[#D4D4D4]"
                    />
                  ) : (
                    <input
                      type={isNum ? "number" : "text"}
                      value={val ?? ""}
                      onChange={(e) => {
                        if (isNum) {
                          const v = e.target.value;
                          if (v === "") handleChange(key, 0);
                          else {
                            const p = parseFloat(v);
                            handleChange(key, isNaN(p) ? 0 : Math.min(65536, p));
                          }
                        } else {
                          handleChange(key, e.target.value);
                        }
                      }}
                      className="input"
                    />
                  )}
                </div>
              );
            })}
          </div>

          {data.params !== undefined && (
            <div className="p-2 rounded bg-[#1B1B1B] border border-[#262626] flex justify-between items-center text-[10px]">
              <span className="text-[#737373]">EST. WEIGHTS:</span>
              <span className="text-[#F2F2F2] font-mono-num font-medium">
                {formatParamCount(data.params)}
              </span>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="w-72 panel bg-[#202020] border-[#333333] rounded-lg p-3 flex flex-col h-full overflow-hidden select-none font-mono text-xs">
      <div className="flex items-center justify-between pb-2 mb-2.5 border-b border-[#262626]">
        <div className="flex items-center gap-1.5">
          <Sliders className="w-3.5 h-3.5 text-[#737373]" />
          <span className="font-semibold text-[#F2F2F2] uppercase tracking-wide text-xs">
            Model Inspector
          </span>
        </div>
        <div className="flex items-center gap-1">
          {onNewScratch && (
            <button
              onClick={onNewScratch}
              className="p-1 rounded text-[#737373] hover:text-[#F2F2F2] hover:bg-[#262626] transition-colors"
              title="Design from scratch"
            >
              <Plus className="w-3.5 h-3.5" />
            </button>
          )}
          <button
            onClick={onCloneNetwork}
            className="p-1 rounded text-[#737373] hover:text-[#F2F2F2] hover:bg-[#262626] transition-colors"
            title="Clone model"
          >
            <Copy className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto space-y-2.5 pr-0.5">
        {/* Template Selector */}
        <div>
          <label className="text-[10px] text-[#737373] uppercase font-medium block mb-1">
            Active Architecture
          </label>
          <select
            value={globalConfig.id}
            onChange={(e) => onLoadTemplate(e.target.value)}
            className="input"
          >
            {customNetworks.length > 0 && (
              <optgroup label="Custom Models">
                {customNetworks.map((n) => (
                  <option key={n.id} value={n.id}>
                    {n.name}
                  </option>
                ))}
              </optgroup>
            )}
            <optgroup label="Built-in Templates">
              {baseTemplates.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name}
                </option>
              ))}
            </optgroup>
          </select>
        </div>

        {/* Global Architecture Hyperparameters */}
        <div>
          <label className="text-[10px] text-[#737373] uppercase font-medium block mb-1">
            Model Identifier
          </label>
          <input
            type="text"
            value={globalConfig.id}
            onChange={(e) => onUpdateGlobalConfig({ id: e.target.value })}
            className="input"
          />
        </div>

        <div>
          <label className="text-[10px] text-[#737373] uppercase font-medium block mb-1">
            Display Name
          </label>
          <input
            type="text"
            value={globalConfig.name}
            onChange={(e) => onUpdateGlobalConfig({ name: e.target.value })}
            className="input"
          />
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-[10px] text-[#737373] uppercase font-medium block mb-0.5">
              Hidden Dim (d)
            </label>
            <input
              type="text"
              inputMode="numeric"
              value={globalConfig.d_model ?? ""}
              onChange={(e) => {
                const v = e.target.value;
                if (v === "") onUpdateGlobalConfig({ d_model: "" as any });
                else {
                  const p = parseInt(v, 10);
                  if (!isNaN(p)) onUpdateGlobalConfig({ d_model: Math.min(16384, Math.max(16, p)) });
                }
              }}
              onBlur={() => {
                if (!globalConfig.d_model || isNaN(Number(globalConfig.d_model))) {
                  onUpdateGlobalConfig({ d_model: 512 });
                }
              }}
              className="input"
            />
          </div>

          <div>
            <label className="text-[10px] text-[#737373] uppercase font-medium block mb-0.5">
              Layers (N)
            </label>
            <input
              type="text"
              inputMode="numeric"
              value={globalConfig.n_layers ?? ""}
              onChange={(e) => {
                const v = e.target.value;
                if (v === "") onUpdateGlobalConfig({ n_layers: "" as any });
                else {
                  const p = parseInt(v, 10);
                  if (!isNaN(p)) onUpdateGlobalConfig({ n_layers: Math.min(128, Math.max(1, p)) });
                }
              }}
              onBlur={() => {
                if (!globalConfig.n_layers || isNaN(Number(globalConfig.n_layers))) {
                  onUpdateGlobalConfig({ n_layers: 12 });
                }
              }}
              className="input"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-[10px] text-[#737373] uppercase font-medium block mb-0.5">
              Query Heads (n_q)
            </label>
            <input
              type="text"
              inputMode="numeric"
              value={globalConfig.n_q ?? ""}
              onChange={(e) => {
                const v = e.target.value;
                if (v === "") onUpdateGlobalConfig({ n_q: "" as any });
                else {
                  const p = parseInt(v, 10);
                  if (!isNaN(p)) onUpdateGlobalConfig({ n_q: Math.min(128, Math.max(1, p)) });
                }
              }}
              onBlur={() => {
                if (!globalConfig.n_q || isNaN(Number(globalConfig.n_q))) {
                  onUpdateGlobalConfig({ n_q: 8 });
                }
              }}
              className="input"
            />
          </div>

          <div>
            <label className="text-[10px] text-[#737373] uppercase font-medium block mb-0.5">
              KV Heads (n_kv)
            </label>
            <input
              type="text"
              inputMode="numeric"
              value={globalConfig.n_kv ?? ""}
              onChange={(e) => {
                const v = e.target.value;
                if (v === "") onUpdateGlobalConfig({ n_kv: "" as any });
                else {
                  const p = parseInt(v, 10);
                  if (!isNaN(p)) onUpdateGlobalConfig({ n_kv: Math.min(128, Math.max(1, p)) });
                }
              }}
              onBlur={() => {
                if (!globalConfig.n_kv || isNaN(Number(globalConfig.n_kv))) {
                  onUpdateGlobalConfig({ n_kv: 2 });
                }
              }}
              className="input"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-[10px] text-[#737373] uppercase font-medium block mb-0.5">
              FFN Dim (d_ff)
            </label>
            <input
              type="text"
              inputMode="numeric"
              value={globalConfig.d_ff ?? ""}
              onChange={(e) => {
                const v = e.target.value;
                if (v === "") onUpdateGlobalConfig({ d_ff: "" as any });
                else {
                  const p = parseInt(v, 10);
                  if (!isNaN(p)) onUpdateGlobalConfig({ d_ff: Math.min(65536, Math.max(16, p)) });
                }
              }}
              onBlur={() => {
                if (!globalConfig.d_ff || isNaN(Number(globalConfig.d_ff))) {
                  onUpdateGlobalConfig({ d_ff: 1280 });
                }
              }}
              className="input"
            />
          </div>

          <div>
            <label className="text-[10px] text-[#737373] uppercase font-medium block mb-0.5">
              Vocab Size
            </label>
            <input
              type="text"
              inputMode="numeric"
              value={globalConfig.vocab_size ?? ""}
              onChange={(e) => {
                const v = e.target.value;
                if (v === "") onUpdateGlobalConfig({ vocab_size: "" as any });
                else {
                  const p = parseInt(v, 10);
                  if (!isNaN(p)) onUpdateGlobalConfig({ vocab_size: Math.min(300000, Math.max(128, p)) });
                }
              }}
              onBlur={() => {
                if (!globalConfig.vocab_size || isNaN(Number(globalConfig.vocab_size))) {
                  onUpdateGlobalConfig({ vocab_size: 12288 });
                }
              }}
              className="input"
            />
          </div>
        </div>

        <div>
          <label className="flex items-center gap-2 cursor-pointer pt-1">
            <input
              type="checkbox"
              checked={globalConfig.tied_embeddings}
              onChange={(e) => onUpdateGlobalConfig({ tied_embeddings: e.target.checked })}
              className="accent-[#D4D4D4]"
            />
            <span className="text-[11px] text-[#A3A3A3]">Tied Word Embeddings</span>
          </label>
        </div>
      </div>

      <div className="pt-2.5 border-t border-[#262626] flex items-center justify-between gap-1.5">
        {isCustom && onDeleteNetwork && (
          <button
            onClick={onDeleteNetwork}
            className="p-1.5 rounded text-[#f87171] hover:bg-[rgba(248,113,113,0.1)] transition-colors"
            title="Delete this custom model"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        )}
        <button
          onClick={onSaveNetwork}
          disabled={isSaving}
          className="btn btn-primary btn-sm flex-1 flex items-center justify-center gap-1.5"
        >
          <Save className="w-3 h-3" />
          {isSaving ? "Saving..." : "Save Model"}
        </button>
      </div>
    </div>
  );
};
