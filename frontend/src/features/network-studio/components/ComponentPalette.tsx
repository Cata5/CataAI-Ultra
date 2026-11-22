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
  // Embedding & Heads
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
    id: "lm_head",
    name: "LM Head (Output Logits)",
    category: "embedding",
    layerType: "Linear",
    defaultConfig: { in_features: 512, out_features: 12288, bias: false },
    description: "Linear projection head mapping d_model to vocabulary logits",
  },
  {
    id: "linear_proj",
    name: "Linear / Dense Projection",
    category: "embedding",
    layerType: "Linear",
    defaultConfig: { in_features: 512, out_features: 512, bias: false },
    description: "General affine linear transform (d_in -> d_out)",
  },
  {
    id: "cls_head",
    name: "Classification Head",
    category: "embedding",
    layerType: "Linear",
    defaultConfig: { in_features: 512, out_features: 2, bias: true },
    description: "Linear classification head projecting hidden states to class logits",
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
    description: "Grouped KV heads (4:1 ratio)",
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
    { id: "embedding", name: "Embed / Head" },
    { id: "attention", name: "Attention" },
    { id: "feedforward", name: "Feed-Forward" },
    { id: "norm", name: "Normalization" },
    { id: "structural", name: "Structural" },
  ];

  const filteredItems = PALETTE_ITEMS.filter((item) => {
    const matchesSearch =
      item.name.toLowerCase().includes(search.toLowerCase()) ||
      item.description.toLowerCase().includes(search.toLowerCase());
    const matchesCat = selectedCat === "all" || item.category === selectedCat;
    return matchesSearch && matchesCat;
  });

  return (
    <div className="w-64 panel bg-[#202020] border-[#333333] rounded-lg p-3 flex flex-col h-full overflow-hidden select-none font-mono text-xs">
      <div className="flex items-center justify-between pb-2 mb-2.5 border-b border-[#262626]">
        <div className="flex items-center gap-1.5">
          <Layers className="w-3.5 h-3.5 text-[#737373]" />
          <span className="font-semibold text-[#F2F2F2] uppercase tracking-wide text-xs">
            Layer Palette
          </span>
        </div>
        <span className="text-[10px] text-[#525252]">{filteredItems.length} items</span>
      </div>

      {/* Search Input */}
      <div className="relative mb-2.5">
        <Search className="w-3.5 h-3.5 text-[#737373] absolute left-2.5 top-1/2 -translate-y-1/2" />
        <input
          type="text"
          placeholder="Filter layers..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="input pl-8 text-xs"
        />
      </div>

      {/* Category Pills */}
      <div className="flex gap-1 overflow-x-auto pb-1.5 mb-2 border-b border-[#262626]">
        {categories.map((c) => (
          <button
            key={c.id}
            onClick={() => setSelectedCat(c.id)}
            className={cn(
              "px-2 py-0.5 rounded text-[10px] whitespace-nowrap transition-colors",
              selectedCat === c.id
                ? "bg-[#D4D4D4] text-[#171717] font-semibold"
                : "bg-[#1B1B1B] text-[#737373] hover:text-[#F2F2F2]"
            )}
          >
            {c.name}
          </button>
        ))}
      </div>

      {/* Item List */}
      <div className="flex-1 overflow-y-auto space-y-1.5 pr-0.5">
        {filteredItems.map((item) => (
          <div
            key={item.id}
            className="p-2 rounded bg-[#1B1B1B] border border-[#262626] hover:border-[#404040] hover:bg-[#262626] transition-colors flex flex-col gap-1 group"
          >
            <div className="flex items-center justify-between">
              <span className="font-medium text-[#F2F2F2] truncate">{item.name}</span>
              <div className="flex items-center gap-1">
                {onShowInfo && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onShowInfo(item.layerType);
                    }}
                    className="p-0.5 rounded text-[#737373] hover:text-[#F2F2F2] transition-colors"
                    title="Documentation"
                  >
                    <Info className="w-3 h-3" />
                  </button>
                )}
                <button
                  onClick={() => onAddNode(item)}
                  className="p-1 rounded bg-[#202020] border border-[#333333] hover:bg-[#D4D4D4] hover:text-[#171717] text-[#A3A3A3] transition-colors"
                  title="Add to canvas"
                >
                  <Plus className="w-3 h-3" />
                </button>
              </div>
            </div>
            <p className="text-[10px] text-[#737373] line-clamp-2 leading-relaxed">
              {item.description}
            </p>
          </div>
        ))}
        {filteredItems.length === 0 && (
          <div className="text-center py-6 text-[11px] text-[#525252]">
            No layers found matching query.
          </div>
        )}
      </div>
    </div>
  );
};
