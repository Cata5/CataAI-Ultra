"use client";

import React, { useState } from "react";
import { Search, Plus, Layers, Info } from "lucide-react";
import { cn } from "@/lib/utils";

export interface PaletteItem {
  id: string;
  name: string;
  category: "embedding" | "attention" | "feedforward" | "norm" | "structural";
  layerType: string;
  defaultConfig: Record<string, any>;
  description: string;
}

interface ComponentPaletteProps {
  onAddNode: (item: PaletteItem) => void;
  onShowInfo?: (layerType: string) => void;
}

export const PALETTE_ITEMS: PaletteItem[] = [
  // Embedding
  {
    id: "token_emb",
    name: "Token Embedding",
    category: "embedding",
    layerType: "Embedding",
    defaultConfig: { vocab_size: 12288, d_model: 512 },
    description: "Discrete token index projection to vector space",
  },
  {
    id: "pos_emb",
    name: "Positional Embedding",
    category: "embedding",
    layerType: "PositionalEncoding",
    defaultConfig: { max_seq_len: 2048, d_model: 512 },
    description: "Learnable or sinusoidal sequence position vectors",
  },
  {
    id: "linear_proj",
    name: "Linear Projection",
    category: "embedding",
    layerType: "Linear",
    defaultConfig: { in_features: 512, out_features: 512, bias: false },
    description: "Affine feature transform",
  },

  // Attention
  {
    id: "mha",
    name: "Multi-Head Attention",
    category: "attention",
    layerType: "MultiheadAttention",
    defaultConfig: { d_model: 512, n_heads: 8, dropout: 0.0 },
    description: "Standard multi-head scaled dot-product attention",
  },
  {
    id: "gqa",
    name: "Grouped Query Attention (GQA)",
    category: "attention",
    layerType: "GQAAttention",
    defaultConfig: { d_model: 512, n_q: 8, n_kv: 2 },
    description: "RoCanon baseline grouped KV heads (4:1 ratio)",
  },
  {
    id: "mqa",
    name: "Multi-Query Attention (MQA)",
    category: "attention",
    layerType: "MQAAttention",
    defaultConfig: { d_model: 512, n_q: 8, n_kv: 1 },
    description: "Single shared KV head for memory-constrained decoding",
  },
  {
    id: "causal_attn",
    name: "Causal Attention",
    category: "attention",
    layerType: "CausalSelfAttention",
    defaultConfig: { d_model: 512, n_heads: 8, is_causal: true },
    description: "Autoregressive triangular attention mask",
  },
  {
    id: "cross_attn",
    name: "Cross Attention",
    category: "attention",
    layerType: "CrossAttention",
    defaultConfig: { d_model: 512, n_heads: 8 },
    description: "Encoder-decoder query/key cross mapping",
  },

  // Feed Forward
  {
    id: "swiglu",
    name: "SwiGLU FFN",
    category: "feedforward",
    layerType: "SwiGLU",
    defaultConfig: { d_model: 512, d_ff: 1280 },
    description: "Swish-Gated Linear Unit feedforward block",
  },
  {
    id: "mlp",
    name: "Dense MLP",
    category: "feedforward",
    layerType: "MLP",
    defaultConfig: { d_model: 512, d_ff: 2048, activation: "gelu" },
    description: "Two-layer projection with intermediate non-linearity",
  },
  {
    id: "gelu_act",
    name: "GELU Activation",
    category: "feedforward",
    layerType: "GELU",
    defaultConfig: { approximate: "none" },
    description: "Gaussian Error Linear Unit",
  },
  {
    id: "silu_act",
    name: "SiLU / Swish",
    category: "feedforward",
    layerType: "SiLU",
    defaultConfig: {},
    description: "Sigmoid Linear Unit non-linearity",
  },

  // Normalization
  {
    id: "rmsnorm",
    name: "RMSNorm",
    category: "norm",
    layerType: "RMSNorm",
    defaultConfig: { d_model: 512, eps: 1e-6 },
    description: "Root Mean Square Layer Normalization",
  },
  {
    id: "layernorm",
    name: "LayerNorm",
    category: "norm",
    layerType: "LayerNorm",
    defaultConfig: { d_model: 512, eps: 1e-5 },
    description: "Mean-centered variance normalized layer",
  },
  {
    id: "batchnorm",
    name: "Sequence BatchNorm1d",
    category: "norm",
    layerType: "SequenceBatchNorm1d",
    defaultConfig: { num_features: 512 },
    description: "3D sequence-aware batch normalization",
  },

  // Structural
  {
    id: "residual",
    name: "Residual Block",
    category: "structural",
    layerType: "Residual",
    defaultConfig: { use_pre_norm: true },
    description: "Skip-connection addition block x + f(x)",
  },
  {
    id: "conv1d",
    name: "Conv1D / Depthwise",
    category: "structural",
    layerType: "Conv1d",
    defaultConfig: { in_channels: 512, out_channels: 512, kernel_size: 4 },
    description: "1D causal or depthwise temporal convolution",
  },
  {
    id: "dropout",
    name: "Dropout",
    category: "structural",
    layerType: "Dropout",
    defaultConfig: { p: 0.1 },
    description: "Stochastic node deactivation for regularization",
  },
];

export const ComponentPalette: React.FC<ComponentPaletteProps> = ({ onAddNode, onShowInfo }) => {
  const [search, setSearch] = useState("");
  const [selectedCat, setSelectedCat] = useState<string>("all");

  const categories = [
    { id: "all", name: "All" },
    { id: "embedding", name: "Embed" },
    { id: "attention", name: "Attention" },
    { id: "feedforward", name: "FFN" },
    { id: "norm", name: "Norm" },
    { id: "structural", name: "Structure" },
  ];

  const filtered = PALETTE_ITEMS.filter((item) => {
    const matchesCat = selectedCat === "all" || item.category === selectedCat;
    const matchesSearch =
      item.name.toLowerCase().includes(search.toLowerCase()) ||
      item.layerType.toLowerCase().includes(search.toLowerCase());
    return matchesCat && matchesSearch;
  });

  const onDragStart = (event: React.DragEvent, item: PaletteItem) => {
    event.dataTransfer.setData("application/reactflow-item", JSON.stringify(item));
    event.dataTransfer.effectAllowed = "move";
  };

  return (
    <div className="w-64 glass-panel rounded-lg p-3 flex flex-col h-full overflow-hidden select-none">
      <div className="pb-2 mb-2 border-b border-slate-800">
        <h3 className="font-mono text-xs font-bold text-slate-200 tracking-wider uppercase flex items-center gap-1.5">
          <Layers className="w-3.5 h-3.5 text-cyan-400" /> COMPONENT PALETTE
        </h3>
        <p className="text-[10px] font-mono text-slate-400">
          Drag components onto the studio canvas
        </p>
      </div>

      {/* Search Input */}
      <div className="relative mb-2">
        <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-1/2 -translate-y-1/2" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Filter modules..."
          className="w-full bg-slate-900/80 border border-slate-800 rounded pl-8 pr-2.5 py-1 text-xs font-mono text-slate-200 placeholder-slate-400 focus:outline-none focus:border-cyan-400/50"
        />
      </div>

      {/* Category Pills */}
      <div className="flex flex-wrap gap-1 mb-2.5">
        {categories.map((cat) => (
          <button
            key={cat.id}
            onClick={() => setSelectedCat(cat.id)}
            className={cn(
              "px-2 py-0.5 rounded text-[10px] font-mono transition-colors",
              selectedCat === cat.id
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                : "bg-slate-900/60 text-slate-400 border border-slate-800 hover:text-slate-300"
            )}
          >
            {cat.name}
          </button>
        ))}
      </div>

      {/* Module List */}
      <div className="flex-1 overflow-y-auto space-y-1.5 pr-1">
        {filtered.map((item) => (
          <div
            key={item.id}
            draggable
            onDragStart={(e) => onDragStart(e, item)}
            onClick={() => onAddNode(item)}
            className="p-2 rounded-lg bg-slate-900/70 border border-slate-800/80 hover:border-cyan-400/40 hover:bg-slate-850 cursor-grab active:cursor-grabbing transition-all group relative text-xs font-mono"
          >
            <div className="flex items-center justify-between">
              <span className="font-semibold text-slate-200 group-hover:text-cyan-300">
                {item.name}
              </span>
              <div className="flex items-center gap-1">
                {onShowInfo && (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onShowInfo(item.layerType);
                    }}
                    className="opacity-0 group-hover:opacity-100 p-0.5 rounded text-slate-400 hover:text-cyan-300 hover:bg-slate-800 transition-all"
                    title="Educational Info & Docs"
                  >
                    <Info className="w-3 h-3" />
                  </button>
                )}
                <button
                  type="button"
                  className="opacity-0 group-hover:opacity-100 p-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-500/30"
                  title="Add to Canvas"
                >
                  <Plus className="w-3 h-3" />
                </button>
              </div>
            </div>
            <p className="text-[10px] text-slate-400 line-clamp-1 mt-0.5">
              {item.description}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
};
