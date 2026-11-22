"use client";

import React from "react";
import { X, BookOpen, Cpu, ShieldCheck, Zap } from "lucide-react";

export interface LayerDocumentation {
  title: string;
  category: string;
  math: string;
  description: string;
  parameterFormula: string;
  memoryAndVram: string;
  rocanonContext: string;
  bestPractices: string;
}

export const LAYER_DOCS: Record<string, LayerDocumentation> = {
  Embedding: {
    title: "Token Embedding (Discrete Vector Lookup)",
    category: "embedding",
    math: "y = W_e[x], \\quad W_e \\in \\mathbb{R}^{V \\times d}",
    description: "Maps discrete vocabulary token IDs into a continuous d-dimensional latent representation vector space.",
    parameterFormula: "Params = V \\times d",
    memoryAndVram: "Weights: ~12.58 MB in BF16. Tied embedding shares overhead with 0 additional parameters.",
    rocanonContext: "Uses tied word embeddings where input embedding is reused as output projection head.",
    bestPractices: "Scale initial weights by 1/sqrt(d_model). Use tied embeddings for sub-100M models.",
  },
  PositionalEncoding: {
    title: "Rotary Embeddings (RoPE)",
    category: "embedding",
    math: "R_{\\Theta, m}^d = \\text{diag}(R_{\\theta_1, m}, \\dots, R_{\\theta_{d/2}, m})",
    description: "Encodes spatial token sequence order applied directly to Query and Key tensors.",
    parameterFormula: "0 learnable parameters (uses fixed frequency buffers).",
    memoryAndVram: "Precomputed cos/sin cache occupies negligible VRAM (< 1 MB).",
    rocanonContext: "Implements RoPE with base frequency 10,000 for sequence extrapolation.",
    bestPractices: "Apply RoPE only to Query and Key projections before attention dot products.",
  },
  GQAAttention: {
    title: "Grouped Query Attention (GQA)",
    category: "attention",
    math: "\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}} + M\\right) V",
    description: "Groups multiple query attention heads to share a smaller number of Key/Value heads.",
    parameterFormula: "W_q (d \\times d) + W_k (d \\times (n_{kv} \\cdot d_h)) + W_v (d \\times (n_{kv} \\cdot d_h)) + W_o (d \\times d)",
    memoryAndVram: "Significantly reduces KV cache memory footprint compared to full multi-head attention.",
    rocanonContext: "Uses 8 Q-heads and 2 KV-heads for rapid generation throughput.",
    bestPractices: "For consumer accelerators, GQA 8:2 or 8:4 provides optimal throughput-to-VRAM balance.",
  },
  MultiheadAttention: {
    title: "Standard Multi-Head Attention (MHA)",
    category: "attention",
    math: "\\text{MHA}(Q, K, V) = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_h) W^O",
    description: "Full multi-head scaled dot product where every attention head maintains an independent Key and Value projection.",
    parameterFormula: "4 \\times d^2 \\text{ params/layer}",
    memoryAndVram: "Higher KV-cache consumption during generation.",
    rocanonContext: "Standard reference baseline attention mechanism.",
    bestPractices: "Ensure d_model is evenly divisible by n_heads. Apply casual masking for autoregressive models.",
  },
  MQAAttention: {
    title: "Multi-Query Attention (MQA)",
    category: "attention",
    math: "K, V \\in \\mathbb{R}^{B \\times S \\times 1 \\times d_h}, \\quad Q \\in \\mathbb{R}^{B \\times S \\times H \\times d_h}",
    description: "Extreme variant of GQA where all query heads share a single Key and Value head.",
    parameterFormula: "W_q (d \\times d) + W_k (d \\times d_h) + W_v (d \\times d_h) + W_o (d \\times d)",
    memoryAndVram: "Extremely lightweight KV cache, optimal for mobile or edge deployment.",
    rocanonContext: "Available in Network Studio for ultra-compact models.",
    bestPractices: "May cause minor degradation in complex syntax reasoning compared to GQA 8:2.",
  },
  SwiGLU: {
    title: "SwiGLU Feed-Forward Network",
    category: "feedforward",
    math: "\\text{SwiGLU}(x) = (x W_{gate} \\cdot \\text{silu}(x W_{gate})) W_{down}",
    description: "Gated linear unit with Swish/SiLU activation offering superior gradient flow.",
    parameterFormula: "3 \\times d \\times d_{ff} \\text{ params/layer}",
    memoryAndVram: "Allocates 8/3 d_model hidden dimension for optimal FLOPS-to-parameter ratio.",
    rocanonContext: "Core MLP component for expressive feature representation.",
    bestPractices: "Set d_ff to multiple of 64 or 256 for tensor core alignment.",
  },
  RMSNorm: {
    title: "Root Mean Square Normalization (RMSNorm)",
    category: "normalization",
    math: "\\bar{a}_i = \\frac{a_i}{\\text{RMS}(a)} g_i, \\quad \\text{RMS}(a) = \\sqrt{\\frac{1}{d} \\sum_{i=1}^d a_i^2 + \\epsilon}",
    description: "Computationally efficient alternative to LayerNorm that scales input without calculating mean offset.",
    parameterFormula: "d \\text{ learnable gain parameters per layer}",
    memoryAndVram: "Minimal VRAM footprint, eliminates one reduction pass compared to LayerNorm.",
    rocanonContext: "Used for pre-attention and pre-FFN normalization layers.",
    bestPractices: "Use epsilon = 1e-5 or 1e-6 to maintain numerical stability in BF16.",
  },
};

export interface LayerInfoModalProps {
  isOpen?: boolean;
  layerType: string | null;
  onClose: () => void;
}

export const LayerInfoModal: React.FC<LayerInfoModalProps> = ({
  isOpen = true,
  layerType,
  onClose,
}) => {
  if (!isOpen || !layerType) return null;

  const doc = LAYER_DOCS[layerType] || {
    title: `${layerType} Documentation`,
    category: "general",
    math: "f(x) = y",
    description: "Standard architectural component for deep neural network topologies.",
    parameterFormula: "Configured dynamically based on input and output tensor shapes.",
    memoryAndVram: "VRAM footprint depends on active precision and batch sequence dimension.",
    rocanonContext: "Supported in custom model architectures.",
    bestPractices: "Maintain consistent hidden dimension throughout residual blocks.",
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm font-mono text-xs">
      <div className="bg-[#202020] border border-[#333333] rounded-lg w-full max-w-xl p-4 shadow-xl relative max-h-[85vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between pb-3 border-b border-[#262626]">
          <div className="flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-[#A3A3A3]" />
            <div>
              <h2 className="text-sm font-semibold text-[#F2F2F2]">{doc.title}</h2>
              <span className="text-[10px] text-[#737373] uppercase">
                {doc.category} · {layerType}
              </span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded text-[#737373] hover:text-[#F2F2F2] transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content Body */}
        <div className="py-3 space-y-3">
          {/* Math Block */}
          <div className="p-2.5 rounded bg-[#171717] border border-[#262626]">
            <span className="text-[10px] text-[#737373] uppercase block mb-1 flex items-center gap-1">
              <Zap className="w-3 h-3 text-[#A3A3A3]" /> Mathematical Formulation
            </span>
            <code className="text-xs text-[#D4D4D4] font-mono block">{doc.math}</code>
          </div>

          {/* Description */}
          <div>
            <span className="text-[10px] text-[#737373] uppercase block mb-1">
              Operational Dynamics
            </span>
            <p className="text-[#A3A3A3] text-xs leading-relaxed">{doc.description}</p>
          </div>

          {/* Complexity & VRAM */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            <div className="p-2 rounded bg-[#1B1B1B] border border-[#262626]">
              <span className="text-[10px] text-[#737373] uppercase block mb-0.5 flex items-center gap-1">
                <ShieldCheck className="w-3 h-3 text-[#A3A3A3]" /> Parameter Complexity
              </span>
              <p className="text-[11px] text-[#D4D4D4] font-mono">{doc.parameterFormula}</p>
            </div>
            <div className="p-2 rounded bg-[#1B1B1B] border border-[#262626]">
              <span className="text-[10px] text-[#737373] uppercase block mb-0.5 flex items-center gap-1">
                <Cpu className="w-3 h-3 text-[#A3A3A3]" /> Memory Footprint
              </span>
              <p className="text-[11px] text-[#A3A3A3]">{doc.memoryAndVram}</p>
            </div>
          </div>

          {/* Architecture Context */}
          <div className="p-2.5 rounded bg-[#1B1B1B] border border-[#262626]">
            <span className="text-[10px] text-[#737373] uppercase block mb-1">
              Architecture Context
            </span>
            <p className="text-[#A3A3A3] text-xs leading-relaxed">{doc.rocanonContext}</p>
          </div>

          {/* Best Practices */}
          <div className="p-2.5 rounded bg-[#1B1B1B] border border-[#262626]">
            <span className="text-[10px] text-[#737373] uppercase block mb-1">
              Best Practices
            </span>
            <p className="text-[#A3A3A3] text-xs leading-relaxed">{doc.bestPractices}</p>
          </div>
        </div>

        <div className="flex justify-end pt-2 border-t border-[#262626]">
          <button
            onClick={onClose}
            className="btn btn-secondary btn-sm"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
