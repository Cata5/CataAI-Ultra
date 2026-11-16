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
    parameterFormula: "Params = V \\times d = 12,288 \\times 512 = 6,291,456",
    memoryAndVram: "Weights: ~12.58 MB in BF16. When tied with the LM Head, memory overhead is shared with 0 additional parameters.",
    rocanonContext: "RoCanon-48.5M uses tied word embeddings (input embedding weight matrix is transposed and reused as output unembedding head).",
    bestPractices: "Always scale initial embedding weights by 1/sqrt(d_model) or standard normal std=0.02. Use tied embeddings for sub-100M models to prevent parameter bloat.",
  },
  PositionalEncoding: {
    title: "Positional Encoding & Rotary Embeddings (RoPE)",
    category: "embedding",
    math: "R_{\\Theta, m}^d = \\text{diag}(R_{\\theta_1, m}, \\dots, R_{\\theta_{d/2}, m})",
    description: "Encodes spatial token sequence order. Modern architectures favor Rotary Position Embeddings (RoPE) applied directly to Query and Key tensors.",
    parameterFormula: "0 learnable parameters (RoPE uses fixed frequency buffers).",
    memoryAndVram: "Precomputed cos/sin cache tensors occupy negligible VRAM (< 1 MB).",
    rocanonContext: "RoCanon implements RoPE with base frequency 10,000 for relative distance invariance and sequence length extrapolation up to 2048 tokens.",
    bestPractices: "Apply RoPE only to Query and Key projections before attention dot products. Never apply to Value tensors.",
  },
  GQAAttention: {
    title: "Grouped Query Attention (GQA)",
    category: "attention",
    math: "\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}} + M\\right) V",
    description: "Groups multiple query attention heads to share a smaller number of Key/Value heads (e.g. 8 Q-heads sharing 2 KV-heads, 4:1 ratio). Reduces KV-cache memory during autoregressive inference while preserving representational capacity.",
    parameterFormula: "W_q (d \\times d) + W_k (d \\times (n_{kv} \\cdot d_h)) + W_v (d \\times (n_{kv} \\cdot d_h)) + W_o (d \\times d)",
    memoryAndVram: "In RoCanon: W_q (512x512) + W_k (512x128) + W_v (512x128) + W_o (512x512) = 655,360 params/layer. Total across 16 layers = 10.48M params.",
    rocanonContext: "RoCanon uses 8 Q-heads and 2 KV-heads with head dimension d_h=64. This enables 4x faster KV-cache generation during interactive Romanian text generation.",
    bestPractices: "For consumer accelerators (6GB to 16GB), GQA 8:2 or 8:4 provides optimal throughput-to-VRAM balance. Use QK-Norm to eliminate loss spikes.",
  },
  MultiheadAttention: {
    title: "Standard Multi-Head Attention (MHA)",
    category: "attention",
    math: "\\text{MHA}(Q, K, V) = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_h) W^O",
    description: "Full multi-head scaled dot product where every attention head maintains an independent Key and Value projection.",
    parameterFormula: "4 \\times d^2 = 4 \\times 512^2 = 1,048,576 \\text{ params/layer}",
    memoryAndVram: "Higher KV-cache consumption than GQA during generation. 16 layers require ~16.7M parameters.",
    rocanonContext: "Used in Mini-GPT 15M template. Replaced by GQA in RoCanon-48.5M for inference efficiency.",
    bestPractices: "Ensure d_model is evenly divisible by n_heads. Apply casual masking for autoregressive models.",
  },
  MQAAttention: {
    title: "Multi-Query Attention (MQA)",
    category: "attention",
    math: "K, V \\in \\mathbb{R}^{B \\times S \\times 1 \\times d_h}, \\quad Q \\in \\mathbb{R}^{B \\times S \\times H \\times d_h}",
    description: "Extreme variant of GQA where all query heads share a single Key and Value head (n_kv = 1). Minimizes KV memory to the absolute minimum.",
    parameterFormula: "W_q (d \\times d) + W_k (d \\times d_h) + W_v (d \\times d_h) + W_o (d \\times d)",
    memoryAndVram: "Extremely lightweight KV cache, optimal for mobile or edge deployment.",
    rocanonContext: "Available in Network Studio for ultra-compact or edge-device custom models.",
    bestPractices: "May cause minor degradation in complex syntax reasoning compared to GQA 8:2.",
  },
  SwiGLU: {
    title: "SwiGLU Feed-Forward Network",
    category: "feedforward",
    math: "\\text{SwiGLU}(x) = \\left(\\text{swish}(x W_{\\text{gate}}) \\otimes (x W_{\\text{up}})\\right) W_{\\text{down}}",
    description: "Swish-Gated Linear Unit feedforward layer. Replaces traditional two-matrix ReLU/GELU MLP with three weight matrices: gate, up-projection, and down-projection. Significantly outperforms standard MLPs at the same parameter budget.",
    parameterFormula: "3 \\times (d_{\\text{model}} \\times d_{ff}) = 3 \\times (512 \\times 1280) = 1,966,080 \\text{ params/layer}",
    memoryAndVram: "16 layers = 31.45M parameters (~65% of RoCanon's total weight budget). Consumes ~62.9 MB in BF16.",
    rocanonContext: "RoCanon uses d_ff = 1280 (approx. 2.5x d_model) to maintain high representational capacity while staying within consumer VRAM budgets.",
    bestPractices: "Maintain d_ff as a multiple of 64 or 128 for tensor core alignment. Do not include bias terms in linear projections.",
  },
  MLP: {
    title: "Dense Multi-Layer Perceptron (Standard MLP)",
    category: "feedforward",
    math: "\\text{MLP}(x) = \\text{Act}(x W_1 + b_1) W_2 + b_2",
    description: "Traditional two-matrix feedforward network with GELU or ReLU non-linearity.",
    parameterFormula: "2 \\times (d_{\\text{model}} \\times d_{ff}) = 2 \\times (512 \\times 2048) = 2,097,152 \\text{ params/layer}",
    memoryAndVram: "Standard memory profile. Requires 2 matrices instead of SwiGLU's 3, but requires larger d_ff (usually 4x d_model) for comparable loss.",
    rocanonContext: "Available for BERT and MLP templates.",
    bestPractices: "Prefer SwiGLU for modern LLM architectures.",
  },
  RMSNorm: {
    title: "Root Mean Square Layer Normalization (RMSNorm)",
    category: "norm",
    math: "\\text{RMSNorm}(x) = \\frac{x}{\\sqrt{\\frac{1}{d}\\sum_{i=1}^d x_i^2 + \\epsilon}} \\odot \\gamma",
    description: "Faster and more numerically stable alternative to LayerNorm. Enforces scale invariance without subtracting the activation mean, reducing computation time by 10-20% per layer.",
    parameterFormula: "d_{\\text{model}} \\text{ learnable scale parameters } \\gamma = 512 \\text{ params/layer} (0 \\text{ bias})",
    memoryAndVram: "Negligible memory (< 1 KB per layer). Very high compute efficiency on modern CUDA and Tensor cores.",
    rocanonContext: "RoCanon applies Pre-RMSNorm before attention and SwiGLU blocks, plus a final RMSNorm before the LM Head.",
    bestPractices: "Set eps=1e-6 for BF16 precision. Avoid adding bias parameters.",
  },
  LayerNorm: {
    title: "Layer Normalization (LayerNorm)",
    category: "norm",
    math: "\\text{LN}(x) = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\odot \\gamma + \\beta",
    description: "Standard normalization normalizing mean and variance across the hidden feature dimension.",
    parameterFormula: "2 \\times d_{\\text{model}} = 1,024 \\text{ params/layer (gain + bias)}",
    memoryAndVram: "Requires mean subtraction and variance calculation.",
    rocanonContext: "Available in BERT Encoder and legacy templates.",
    bestPractices: "Use RMSNorm instead for modern autoregressive transformers.",
  },
  Residual: {
    title: "Residual Skip Connection (x + f(x))",
    category: "structural",
    math: "y = x + f(\\text{Norm}(x))",
    description: "Identity skip-connection that provides an uninterrupted gradient highway back to early layers, preventing vanishing/exploding gradients in deep networks.",
    parameterFormula: "0 learnable parameters.",
    memoryAndVram: "Requires caching input tensor in memory for the addition step.",
    rocanonContext: "Pre-norm residual connections wrap each GQA attention block and SwiGLU FFN block in all 16 layers.",
    bestPractices: "Always use Pre-LN / Pre-RMSNorm structure (normalizing before the sub-layer) rather than Post-LN for stable training without warm-up instability.",
  },
  Conv1d: {
    title: "Causal Depthwise 1D Convolution",
    category: "structural",
    math: "y_t = \\sum_{k=0}^{K-1} W_k \\odot x_{t-k}",
    description: "Temporal depthwise 1D convolution with causal left-padding. Injects inductive local spatial bias into the transformer without increasing quadratic attention complexity.",
    parameterFormula: "d_{\\text{model}} \\times K = 512 \\times 4 = 2,048 \\text{ params per conv block}",
    memoryAndVram: "Extremely low memory footprint. Adds less than 0.1M parameters total across the model.",
    rocanonContext: "RoCanon interleaves 4 causal depthwise Conv1D blocks (kernel_size=4) at layers 0, 4, 8, and 12, creating the hybrid CNN-Transformer architecture.",
    bestPractices: "Must strictly use causal padding (pad left with K-1 zeros) to prevent future token information leakage.",
  },
  Linear: {
    title: "Linear Affine Projection",
    category: "embedding",
    math: "y = x W^T + b",
    description: "Standard matrix multiplication projecting from in_features to out_features.",
    parameterFormula: "\\text{in\\_features} \\times \\text{out\\_features} (+ \\text{out\\_features if bias})",
    memoryAndVram: "Depends on input/output dimensions.",
    rocanonContext: "Used for input projections and the LM Head.",
    bestPractices: "Disable bias when followed immediately by RMSNorm or when using Muon optimizer.",
  },
};

interface LayerInfoModalProps {
  isOpen: boolean;
  layerType: string | null;
  onClose: () => void;
}

export const LayerInfoModal: React.FC<LayerInfoModalProps> = ({
  isOpen,
  layerType,
  onClose,
}) => {
  if (!isOpen || !layerType) return null;

  const doc = LAYER_DOCS[layerType] || {
    title: `${layerType} Layer`,
    category: "generic",
    math: "y = f(x; \\theta)",
    description: `Configurable neural network layer of type ${layerType}.`,
    parameterFormula: "Configured by layer dimensions and kernel sizes.",
    memoryAndVram: "VRAM footprint scales with batch size, sequence length, and hidden dimension.",
    rocanonContext: "Integrated into the modular RoCanon computation graph.",
    bestPractices: "Validate input/output tensor shapes and numerical precision.",
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in select-none font-mono text-xs">
      <div className="w-full max-w-2xl glass-panel rounded-xl border border-cyan-500/40 shadow-[0_0_30px_rgba(0,240,255,0.2)] overflow-hidden flex flex-col max-h-[85vh]">
        {/* Header */}
        <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/70">
          <div className="flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-cyan-400" />
            <div>
              <h2 className="text-sm font-bold text-slate-100 uppercase tracking-wide">
                {doc.title}
              </h2>
              <span className="text-[10px] text-cyan-400 uppercase font-semibold">
                Category: {doc.category} | Layer Identifier: {layerType}
              </span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-md text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-5 space-y-4 overflow-y-auto text-slate-300">
          {/* Mathematical Formulation */}
          <div className="p-3 rounded-lg bg-slate-900/90 border border-slate-800">
            <div className="text-[10px] uppercase font-bold text-cyan-400 mb-1 flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5" /> Mathematical Formulation
            </div>
            <code className="text-xs text-amber-300 font-mono block bg-black/40 p-2 rounded border border-slate-800/80">
              {doc.math}
            </code>
          </div>

          {/* Description */}
          <div>
            <div className="text-[10px] uppercase font-bold text-slate-400 mb-1">
              Description & Operational Dynamics
            </div>
            <p className="text-xs leading-relaxed text-slate-200">
              {doc.description}
            </p>
          </div>

          {/* Parameter Complexity & Memory */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="p-3 rounded-lg bg-slate-950/50 border border-slate-800">
              <div className="text-[10px] uppercase font-bold text-emerald-400 mb-1 flex items-center gap-1.5">
                <ShieldCheck className="w-3.5 h-3.5" /> Parameter Complexity
              </div>
              <p className="text-[11px] text-slate-300 font-mono">
                {doc.parameterFormula}
              </p>
            </div>

            <div className="p-3 rounded-lg bg-slate-950/50 border border-slate-800">
              <div className="text-[10px] uppercase font-bold text-purple-400 mb-1 flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5" /> Memory & VRAM Dynamics (6GB GPU)
              </div>
              <p className="text-[11px] text-slate-300">
                {doc.memoryAndVram}
              </p>
            </div>
          </div>

          {/* RoCanon-48.5M Context */}
          <div className="p-3 rounded-lg bg-cyan-950/30 border border-cyan-500/20">
            <div className="text-[10px] uppercase font-bold text-cyan-300 mb-1">
              RoCanon-48.5M Architectural Integration
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              {doc.rocanonContext}
            </p>
          </div>

          {/* Engineering Best Practices */}
          <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
            <div className="text-[10px] uppercase font-bold text-amber-400 mb-1">
              Engineering Best Practices & Convergence Tips
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              {doc.bestPractices}
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-slate-800 bg-slate-950/70 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-bold transition-all shadow-[0_0_10px_rgba(0,240,255,0.3)]"
          >
            DISMISS
          </button>
        </div>
      </div>
    </div>
  );
};
