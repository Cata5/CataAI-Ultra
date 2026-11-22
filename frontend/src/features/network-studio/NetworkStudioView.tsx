"use client";

import React, { useState, useCallback, useEffect, useMemo } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Edge,
  Node,
  BackgroundVariant,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { ArchitectureNode } from "./components/ArchitectureNode";
import { ComponentPalette, PaletteItem } from "./components/ComponentPalette";
import { ConfigInspector } from "./components/ConfigInspector";
import { LayerInfoModal } from "./components/LayerInfoModal";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { api } from "@/services/api";
import { formatParamCount, cn } from "@/lib/utils";
import {
  HardDrive,
  CheckCircle2,
  AlertTriangle,
  Play,
  Plus,
  Trash2,
  Copy,
  Layers,
  Save,
  ChevronDown,
  Lightbulb,
  ArrowDown,
  Sparkles,
  X,
  Check,
} from "lucide-react";

// Canonical 5-stage autoregressive LLM blueprint flow
const LLM_BLUEPRINT_STAGES = [
  {
    step: 1,
    id: "step-embed",
    name: "Token & Position Embedding",
    layerType: "Embedding",
    category: "embedding" as const,
    shape: (d: number, _v: number) => `[B, S] → [B, S, ${d}]`,
    role: "Input Discrete-to-Dense Mapping",
    description: "Transforms discrete vocabulary token IDs into dense semantic vector representations of dimension d_model.",
  },
  {
    step: 2,
    id: "step-norm",
    name: "Input RMSNorm",
    layerType: "RMSNorm",
    category: "norm" as const,
    shape: (d: number) => `[B, S, ${d}] → [B, S, ${d}]`,
    role: "Root Mean Square Variance Stabilizer",
    description: "Pre-normalizes hidden representations before attention blocks to eliminate internal covariate shift.",
  },
  {
    step: 3,
    id: "step-attn",
    name: "Causal Attention (GQA / MHA)",
    layerType: "GQAAttention",
    category: "attention" as const,
    shape: (d: number) => `[B, S, ${d}] → [B, S, ${d}]`,
    role: "Contextual Token Interaction",
    description: "Attends to preceding context using causal masking, rotary positional embeddings (RoPE), and grouped KV cache.",
  },
  {
    step: 4,
    id: "step-ffn",
    name: "SwiGLU Feed-Forward Network",
    layerType: "SwiGLU",
    category: "feedforward" as const,
    shape: (d: number) => `[B, S, ${d}] → [B, S, ${d}]`,
    role: "Non-Linear Feature Expansion",
    description: "Expands token vectors to intermediate dimension (d_ff ≈ 2.5x d_model) with SiLU activation gating.",
  },
  {
    step: 5,
    id: "step-head",
    name: "Language Model Head (Linear)",
    layerType: "Linear",
    category: "embedding" as const,
    shape: (d: number, v: number) => `[B, S, ${d}] → [B, S, ${v}]`,
    role: "Vocabulary Logits Projection",
    description: "Projects final hidden states back to vocabulary logits for next-token probability distribution sampling.",
  },
];

// Canonical layer type to category mapping
const LAYER_TYPE_CATEGORY_MAP: Record<string, "embedding" | "attention" | "feedforward" | "norm" | "convolution" | "structural"> = {
  Embedding: "embedding",
  PositionalEncoding: "embedding",
  Linear: "embedding",
  MultiheadAttention: "attention",
  MultiHeadAttention: "attention",
  GQAAttention: "attention",
  MQAAttention: "attention",
  CausalSelfAttention: "attention",
  CrossAttention: "attention",
  SwiGLU: "feedforward",
  MLP: "feedforward",
  GELU: "feedforward",
  RMSNorm: "norm",
  LayerNorm: "norm",
  BatchNorm1d: "norm",
  Conv1D: "convolution",
  Conv1d: "convolution",
  Conv2D: "convolution",
  ResidualBlock: "structural",
  Dropout: "structural",
};

// Initial default nodes representing RoCanon-48.5M architectural pipeline
const INITIAL_NODES: Node[] = [
  {
    id: "node-embed",
    type: "archNode",
    position: { x: 250, y: 30 },
    data: {
      label: "Token & Position Embed",
      category: "embedding",
      layerType: "Embedding",
      config: { vocab_size: 12288, d_model: 512 },
      inputShape: "[B, S]",
      outputShape: "[B, S, 512]",
      params: 6291456,
      isValid: true,
    },
  },
  {
    id: "node-norm1",
    type: "archNode",
    position: { x: 250, y: 150 },
    data: {
      label: "Input RMSNorm",
      category: "norm",
      layerType: "RMSNorm",
      config: { d_model: 512, eps: 1e-6 },
      inputShape: "[B, S, 512]",
      outputShape: "[B, S, 512]",
      params: 512,
      isValid: true,
    },
  },
  {
    id: "node-attn",
    type: "archNode",
    position: { x: 250, y: 270 },
    data: {
      label: "RoCanon GQA Blocks (16x)",
      category: "attention",
      layerType: "GQAAttention",
      config: { d_model: 512, n_q: 8, n_kv: 2 },
      inputShape: "[B, S, 512]",
      outputShape: "[B, S, 512]",
      params: 8388608,
      isValid: true,
    },
  },
  {
    id: "node-ffn",
    type: "archNode",
    position: { x: 250, y: 390 },
    data: {
      label: "SwiGLU FFN Blocks (16x)",
      category: "feedforward",
      layerType: "SwiGLU",
      config: { d_model: 512, d_ff: 1280 },
      inputShape: "[B, S, 512]",
      outputShape: "[B, S, 512]",
      params: 31457280,
      isValid: true,
    },
  },
  {
    id: "node-head",
    type: "archNode",
    position: { x: 250, y: 510 },
    data: {
      label: "LM Head (Tied)",
      category: "embedding",
      layerType: "Linear",
      config: { in_features: 512, out_features: 12288, bias: false },
      inputShape: "[B, S, 512]",
      outputShape: "[B, S, 12288]",
      params: 0,
      isValid: true,
    },
  },
];

const INITIAL_EDGES: Edge[] = [
  { id: "e1-2", source: "node-embed", target: "node-norm1", animated: true },
  { id: "e2-3", source: "node-norm1", target: "node-attn", animated: true },
  { id: "e3-4", source: "node-attn", target: "node-ffn", animated: true },
  { id: "e4-5", source: "node-ffn", target: "node-head", animated: true },
];

const PRESET_GRAPHS: Record<string, { config: any; nodes: Node[]; edges: Edge[] }> = {
  rocanon_48m: {
    config: {
      id: "rocanon_48m",
      name: "RoCanon-48.5M Baseline",
      task_type: "causal_lm",
      architecture_type: "hybrid_cnn_transformer",
      vocab_size: 12288,
      d_model: 512,
      n_layers: 16,
      n_q: 8,
      n_kv: 2,
      d_ff: 1280,
      kernel_size: 4,
      tied_embeddings: true,
    },
    nodes: INITIAL_NODES,
    edges: INITIAL_EDGES,
  },
  mini_gpt_15m: {
    config: {
      id: "mini_gpt_15m",
      name: "Llama / GPT Causal Transformer 15M",
      task_type: "causal_lm",
      architecture_type: "decoder_only",
      vocab_size: 12288,
      d_model: 384,
      n_layers: 8,
      n_q: 6,
      n_kv: 6,
      d_ff: 1152,
      kernel_size: 3,
      tied_embeddings: true,
    },
    nodes: [
      {
        id: "node-embed",
        type: "archNode",
        position: { x: 250, y: 30 },
        data: {
          label: "Token & Position Embed",
          category: "embedding",
          layerType: "Embedding",
          config: { vocab_size: 12288, d_model: 384 },
          inputShape: "[B, S]",
          outputShape: "[B, S, 384]",
          params: 4718592,
          isValid: true,
        },
      },
      {
        id: "node-norm1",
        type: "archNode",
        position: { x: 250, y: 150 },
        data: {
          label: "Input RMSNorm",
          category: "norm",
          layerType: "RMSNorm",
          config: { d_model: 384, eps: 1e-6 },
          inputShape: "[B, S, 384]",
          outputShape: "[B, S, 384]",
          params: 384,
          isValid: true,
        },
      },
      {
        id: "node-attn",
        type: "archNode",
        position: { x: 250, y: 270 },
        data: {
          label: "Causal MHA Blocks (8x)",
          category: "attention",
          layerType: "MultiheadAttention",
          config: { d_model: 384, n_heads: 6 },
          inputShape: "[B, S, 384]",
          outputShape: "[B, S, 384]",
          params: 4718592,
          isValid: true,
        },
      },
      {
        id: "node-ffn",
        type: "archNode",
        position: { x: 250, y: 390 },
        data: {
          label: "SwiGLU FFN Blocks (8x)",
          category: "feedforward",
          layerType: "SwiGLU",
          config: { d_model: 384, d_ff: 1152 },
          inputShape: "[B, S, 384]",
          outputShape: "[B, S, 384]",
          params: 5308416,
          isValid: true,
        },
      },
      {
        id: "node-head",
        type: "archNode",
        position: { x: 250, y: 510 },
        data: {
          label: "LM Head (Tied)",
          category: "embedding",
          layerType: "Linear",
          config: { in_features: 384, out_features: 12288, bias: false },
          inputShape: "[B, S, 384]",
          outputShape: "[B, S, 12288]",
          params: 0,
          isValid: true,
        },
      },
    ],
    edges: [
      { id: "e1-2", source: "node-embed", target: "node-norm1", animated: true },
      { id: "e2-3", source: "node-norm1", target: "node-attn", animated: true },
      { id: "e3-4", source: "node-attn", target: "node-ffn", animated: true },
      { id: "e4-5", source: "node-ffn", target: "node-head", animated: true },
    ],
  },
  bert_encoder_25m: {
    config: {
      id: "bert_encoder_25m",
      name: "BERT / RoBERTa Encoder 25M",
      task_type: "classification",
      architecture_type: "encoder_only",
      vocab_size: 12288,
      d_model: 448,
      n_layers: 10,
      n_q: 7,
      n_kv: 7,
      d_ff: 1344,
      kernel_size: 3,
      tied_embeddings: false,
    },
    nodes: [
      {
        id: "node-embed",
        type: "archNode",
        position: { x: 250, y: 30 },
        data: {
          label: "Token & Position Embed",
          category: "embedding",
          layerType: "Embedding",
          config: { vocab_size: 12288, d_model: 448 },
          inputShape: "[B, S]",
          outputShape: "[B, S, 448]",
          params: 5505024,
          isValid: true,
        },
      },
      {
        id: "node-norm",
        type: "archNode",
        position: { x: 250, y: 150 },
        data: {
          label: "LayerNorm Pre-Attn",
          category: "norm",
          layerType: "LayerNorm",
          config: { d_model: 448, eps: 1e-5 },
          inputShape: "[B, S, 448]",
          outputShape: "[B, S, 448]",
          params: 896,
          isValid: true,
        },
      },
      {
        id: "node-attn",
        type: "archNode",
        position: { x: 250, y: 270 },
        data: {
          label: "Bidirectional Self-Attn (10x)",
          category: "attention",
          layerType: "MultiheadAttention",
          config: { d_model: 448, n_heads: 7 },
          inputShape: "[B, S, 448]",
          outputShape: "[B, S, 448]",
          params: 8028160,
          isValid: true,
        },
      },
      {
        id: "node-ffn",
        type: "archNode",
        position: { x: 250, y: 390 },
        data: {
          label: "GELU FFN Blocks (10x)",
          category: "feedforward",
          layerType: "MLP",
          config: { in_features: 448, out_features: 1344 },
          inputShape: "[B, S, 448]",
          outputShape: "[B, S, 448]",
          params: 12042240,
          isValid: true,
        },
      },
      {
        id: "node-head",
        type: "archNode",
        position: { x: 250, y: 510 },
        data: {
          label: "Sequence Classifier Head",
          category: "embedding",
          layerType: "Linear",
          config: { in_features: 448, out_features: 2, bias: true },
          inputShape: "[B, S, 448]",
          outputShape: "[B, 2]",
          params: 898,
          isValid: true,
        },
      },
    ],
    edges: [
      { id: "e1-2", source: "node-embed", target: "node-norm", animated: true },
      { id: "e2-3", source: "node-norm", target: "node-attn", animated: true },
      { id: "e3-4", source: "node-attn", target: "node-ffn", animated: true },
      { id: "e4-5", source: "node-ffn", target: "node-head", animated: true },
    ],
  },
  conv1d_network_4m: {
    config: {
      id: "conv1d_network_4m",
      name: "1D Causal CNN Network 4M",
      task_type: "classification",
      architecture_type: "cnn",
      vocab_size: 12288,
      d_model: 384,
      n_layers: 8,
      n_q: 6,
      n_kv: 6,
      d_ff: 768,
      kernel_size: 5,
      tied_embeddings: false,
    },
    nodes: [
      {
        id: "node-conv1",
        type: "archNode",
        position: { x: 250, y: 30 },
        data: {
          label: "Conv1D Input Stem",
          category: "convolution",
          layerType: "Conv1D",
          config: { in_channels: 128, out_channels: 384, kernel_size: 5 },
          inputShape: "[B, 128, S]",
          outputShape: "[B, 384, S]",
          params: 245760,
          isValid: true,
        },
      },
      {
        id: "node-conv2",
        type: "archNode",
        position: { x: 250, y: 150 },
        data: {
          label: "Stacked Conv1D Blocks (8x)",
          category: "convolution",
          layerType: "Conv1D",
          config: { in_channels: 384, out_channels: 384, kernel_size: 5 },
          inputShape: "[B, 384, S]",
          outputShape: "[B, 384, S]",
          params: 3932160,
          isValid: true,
        },
      },
      {
        id: "node-norm",
        type: "archNode",
        position: { x: 250, y: 270 },
        data: {
          label: "RMSNorm & Activation",
          category: "norm",
          layerType: "RMSNorm",
          config: { d_model: 384, eps: 1e-6 },
          inputShape: "[B, 384, S]",
          outputShape: "[B, 384, S]",
          params: 384,
          isValid: true,
        },
      },
      {
        id: "node-head",
        type: "archNode",
        position: { x: 250, y: 390 },
        data: {
          label: "Classification Head",
          category: "embedding",
          layerType: "Linear",
          config: { in_features: 384, out_features: 2, bias: true },
          inputShape: "[B, 384]",
          outputShape: "[B, 2]",
          params: 770,
          isValid: true,
        },
      },
    ],
    edges: [
      { id: "e1-2", source: "node-conv1", target: "node-conv2", animated: true },
      { id: "e2-3", source: "node-conv2", target: "node-norm", animated: true },
      { id: "e3-4", source: "node-norm", target: "node-head", animated: true },
    ],
  },
};

export const NetworkStudioView: React.FC = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState(INITIAL_NODES);
  const [edges, setEdges, onEdgesChange] = useEdgesState(INITIAL_EDGES);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  // Global Config state
  const [globalConfig, setGlobalConfig] = useState({
    id: "rocanon_48m",
    name: "RoCanon-48.5M Baseline",
    task_type: "causal_lm",
    architecture_type: "hybrid_cnn_transformer",
    vocab_size: 12288,
    d_model: 512,
    n_layers: 16,
    n_q: 8,
    n_kv: 2,
    d_ff: 1280,
    kernel_size: 4,
    tied_embeddings: true,
  });

  const [templates, setTemplates] = useState<any[]>([]);
  const [registeredNetworks, setRegisteredNetworks] = useState<any[]>([]);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [estimation, setEstimation] = useState<{
    parameters: any;
    memory: any;
  } | null>(null);

  const [validation, setValidation] = useState<{
    valid: boolean;
    issues: string[];
    warnings: string[];
  }>({ valid: true, issues: [], warnings: [] });

  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [infoModalLayer, setInfoModalLayer] = useState<string | null>(null);
  const [confirmDeleteNet, setConfirmDeleteNet] = useState<boolean>(false);
  const [showBlueprintGuide, setShowBlueprintGuide] = useState<boolean>(true);

  // Computes which layers exist in the current canvas graph to provide shadow guidance
  const blueprintStatus = useMemo(() => {
    const hasEmbed = nodes.some((n) => {
      const d = n.data as any;
      return d?.category === "embedding" && d?.layerType !== "Linear";
    });
    const hasNorm = nodes.some((n) => (n.data as any)?.category === "norm");
    const hasAttn = nodes.some((n) => (n.data as any)?.category === "attention");
    const hasFFN = nodes.some((n) => (n.data as any)?.category === "feedforward");
    const hasHead = nodes.some((n) => {
      const d = n.data as any;
      return d?.layerType === "Linear" || String(d?.label || "").toLowerCase().includes("head");
    });

    const placedMap: Record<number, boolean> = {
      1: hasEmbed,
      2: hasNorm,
      3: hasAttn,
      4: hasFFN,
      5: hasHead,
    };

    return LLM_BLUEPRINT_STAGES.map((s) => ({
      ...s,
      placed: Boolean(placedMap[s.step]),
    }));
  }, [nodes]);

  const placedStagesCount = blueprintStatus.filter((s) => s.placed).length;

  // Custom node types definition for ReactFlow
  const nodeTypes = useMemo(() => ({ archNode: ArchitectureNode as any }), []);

  // Fetch networks & templates
  const fetchAllNetworks = useCallback(async () => {
    try {
      const [nets, tmpls] = await Promise.all([
        api.listNetworks(),
        api.getNetworkTemplates(),
      ]);
      setRegisteredNetworks(Array.isArray(nets) ? nets : []);
      setTemplates(Array.isArray(tmpls) ? tmpls : []);
    } catch (err) {
      console.error("Failed to load networks:", err);
    }
  }, []);

  useEffect(() => {
    fetchAllNetworks();
  }, [fetchAllNetworks]);

  // Recalculate metrics whenever globalConfig changes
  useEffect(() => {
    let active = true;
    api
      .estimateNetwork({
        config: globalConfig,
        micro_batch_size: 4,
        seq_len: 256,
        dtype: "bfloat16",
        optimizer: "muon_adamw",
      })
      .then((res) => {
        if (active && res) setEstimation(res);
      })
      .catch(console.error);

    api
      .validateNetwork(globalConfig)
      .then((res: any) => {
        if (active && res) {
          setValidation({
            valid: Boolean(res.valid),
            issues: Array.isArray(res.issues)
              ? res.issues
              : Array.isArray(res.errors)
              ? res.errors
              : [],
            warnings: Array.isArray(res.warnings) ? res.warnings : [],
          });
        }
      })
      .catch(console.error);

    return () => {
      active = false;
    };
  }, [globalConfig]);

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge({ ...params, animated: true }, eds)),
    [setEdges]
  );

  const onNodeClick = useCallback((_: any, node: Node) => {
    setSelectedNodeId(node.id);
  }, []);

  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null);
  }, []);

  const handleAddPaletteNode = (item: PaletteItem) => {
    const id = `node-${Date.now()}`;
    const newNode: Node = {
      id,
      type: "archNode",
      position: { x: 250, y: (nodes.length + 1) * 110 },
      data: {
        label: item.name,
        category: item.category,
        layerType: item.layerType,
        config: { ...item.defaultConfig },
        inputShape: "[B, S, D]",
        outputShape: "[B, S, D]",
        params: 0,
        isValid: true,
      },
    };
    setNodes((nds) => [...nds, newNode]);
    setSelectedNodeId(id);
  };

  const handleUpdateGlobalConfig = useCallback(
    (updates: Partial<any>) => {
      setGlobalConfig((prev) => {
        const next = { ...prev, ...updates };

        // Automatically propagate d_model to all existing nodes in canvas and their shapes
        if (updates.d_model !== undefined && updates.d_model !== prev.d_model) {
          const newD = Number(updates.d_model);
          const oldD = Number(prev.d_model);
          setNodes((nds) =>
            nds.map((n) => {
              const d = n.data as any;
              const cfg = { ...(d?.config || {}) };
              let changed = false;
              if ("d_model" in cfg) {
                cfg.d_model = newD;
                changed = true;
              }
              if ("in_features" in cfg && (cfg.in_features === oldD || d?.layerType === "Linear")) {
                cfg.in_features = newD;
                changed = true;
              }
              if (changed) {
                const inShape = typeof d?.inputShape === "string"
                  ? d.inputShape.replace(String(oldD), String(newD))
                  : d?.inputShape;
                const outShape = typeof d?.outputShape === "string"
                  ? d.outputShape.replace(String(oldD), String(newD))
                  : d?.outputShape;
                return {
                  ...n,
                  data: {
                    ...d,
                    config: cfg,
                    inputShape: inShape,
                    outputShape: outShape,
                  },
                };
              }
              return n;
            })
          );
          setStatusMessage(`d_model synchronized to ${newD} across all layers`);
          setTimeout(() => setStatusMessage(null), 3000);
        }

        // Automatically propagate vocab_size to Embedding and Linear LM Head
        if (updates.vocab_size !== undefined && updates.vocab_size !== prev.vocab_size) {
          const newV = Number(updates.vocab_size);
          const oldV = Number(prev.vocab_size);
          setNodes((nds) =>
            nds.map((n) => {
              const d = n.data as any;
              const cfg = { ...(d?.config || {}) };
              let changed = false;
              if ("vocab_size" in cfg) {
                cfg.vocab_size = newV;
                changed = true;
              }
              if ("out_features" in cfg && (cfg.out_features === oldV || d?.layerType === "Linear")) {
                cfg.out_features = newV;
                changed = true;
              }
              if (changed) {
                const outShape = typeof d?.outputShape === "string"
                  ? d.outputShape.replace(String(oldV), String(newV))
                  : d?.outputShape;
                return {
                  ...n,
                  data: {
                    ...d,
                    config: cfg,
                    outputShape: outShape,
                  },
                };
              }
              return n;
            })
          );
        }

        return next;
      });
    },
    [setNodes]
  );

  const handleUpdateNodeConfig = useCallback(
    (nodeId: string, newConfig: Record<string, any>) => {
      const targetNode = nodes.find((n) => n.id === nodeId);
      const oldD = (targetNode?.data as any)?.config?.d_model;
      const newD = newConfig.d_model;

      // If user changed d_model on this node, auto-sync across all layers and globalConfig!
      if (newD !== undefined && oldD !== undefined && newD !== oldD) {
        handleUpdateGlobalConfig({ d_model: Number(newD) });
        return;
      }

      setNodes((nds) =>
        nds.map((n) => {
          if (n.id === nodeId) {
            return {
              ...n,
              data: {
                ...n.data,
                config: newConfig,
                label: newConfig.label || n.data.label,
              },
            };
          }
          return n;
        })
      );
    },
    [nodes, setNodes, handleUpdateGlobalConfig]
  );

  // Auto-places the complete recommended 5-stage architecture pipeline
  const handleAutoPlaceRecommendedPipeline = useCallback(() => {
    const d_m = globalConfig.d_model || 512;
    const v_s = globalConfig.vocab_size || 12288;
    const n_q = globalConfig.n_q || 8;
    const n_kv = globalConfig.n_kv || 2;
    const d_ff = globalConfig.d_ff || Math.round(d_m * 2.5);

    const recommendedNodes: Node[] = [
      {
        id: "node-embed",
        type: "archNode",
        position: { x: 250, y: 30 },
        data: {
          label: "Token & Position Embed",
          category: "embedding",
          layerType: "Embedding",
          config: { vocab_size: v_s, d_model: d_m },
          inputShape: "[B, S]",
          outputShape: `[B, S, ${d_m}]`,
          params: v_s * d_m,
          isValid: true,
        },
      },
      {
        id: "node-norm1",
        type: "archNode",
        position: { x: 250, y: 150 },
        data: {
          label: "Input RMSNorm",
          category: "norm",
          layerType: "RMSNorm",
          config: { d_model: d_m, eps: 1e-6 },
          inputShape: `[B, S, ${d_m}]`,
          outputShape: `[B, S, ${d_m}]`,
          params: d_m,
          isValid: true,
        },
      },
      {
        id: "node-attn",
        type: "archNode",
        position: { x: 250, y: 270 },
        data: {
          label: `GQA Attention Blocks (${globalConfig.n_layers || 16}x)`,
          category: "attention",
          layerType: "GQAAttention",
          config: { d_model: d_m, n_q, n_kv },
          inputShape: `[B, S, ${d_m}]`,
          outputShape: `[B, S, ${d_m}]`,
          params: Math.round(d_m * d_m * 2),
          isValid: true,
        },
      },
      {
        id: "node-ffn",
        type: "archNode",
        position: { x: 250, y: 390 },
        data: {
          label: `SwiGLU FFN Blocks (${globalConfig.n_layers || 16}x)`,
          category: "feedforward",
          layerType: "SwiGLU",
          config: { d_model: d_m, d_ff },
          inputShape: `[B, S, ${d_m}]`,
          outputShape: `[B, S, ${d_m}]`,
          params: Math.round(d_m * d_ff * 3),
          isValid: true,
        },
      },
      {
        id: "node-head",
        type: "archNode",
        position: { x: 250, y: 510 },
        data: {
          label: "LM Head (Tied)",
          category: "embedding",
          layerType: "Linear",
          config: { in_features: d_m, out_features: v_s, bias: false },
          inputShape: `[B, S, ${d_m}]`,
          outputShape: `[B, S, ${v_s}]`,
          params: globalConfig.tied_embeddings ? 0 : d_m * v_s,
          isValid: true,
        },
      },
    ];

    const recommendedEdges: Edge[] = [
      { id: "e1-2", source: "node-embed", target: "node-norm1", animated: true },
      { id: "e2-3", source: "node-norm1", target: "node-attn", animated: true },
      { id: "e3-4", source: "node-attn", target: "node-ffn", animated: true },
      { id: "e4-5", source: "node-ffn", target: "node-head", animated: true },
    ];

    setNodes(recommendedNodes);
    setEdges(recommendedEdges);
    setSelectedNodeId("node-embed");
    setStatusMessage("Recommended 5-stage autoregressive LLM pipeline placed!");
    setTimeout(() => setStatusMessage(null), 3500);
  }, [globalConfig, setNodes, setEdges]);

  // Places an individual step from the shadow blueprint guide
  const handlePlaceBlueprintStage = useCallback(
    (stepNum: number) => {
      const d_m = globalConfig.d_model || 512;
      const v_s = globalConfig.vocab_size || 12288;
      const n_q = globalConfig.n_q || 8;
      const n_kv = globalConfig.n_kv || 2;
      const d_ff = globalConfig.d_ff || Math.round(d_m * 2.5);

      const stepConfigs: Record<
        number,
        {
          id: string;
          label: string;
          layerType: string;
          category: any;
          config: any;
          inShape: string;
          outShape: string;
          y: number;
        }
      > = {
        1: {
          id: `node_embed_${Date.now()}`,
          label: "Token Embedding",
          layerType: "Embedding",
          category: "embedding",
          config: { vocab_size: v_s, d_model: d_m },
          inShape: "[B, S]",
          outShape: `[B, S, ${d_m}]`,
          y: 30,
        },
        2: {
          id: `node_norm_${Date.now()}`,
          label: "Input RMSNorm",
          layerType: "RMSNorm",
          category: "norm",
          config: { d_model: d_m, eps: 1e-6 },
          inShape: `[B, S, ${d_m}]`,
          outShape: `[B, S, ${d_m}]`,
          y: 150,
        },
        3: {
          id: `node_attn_${Date.now()}`,
          label: "GQA Attention",
          layerType: "GQAAttention",
          category: "attention",
          config: { d_model: d_m, n_q, n_kv },
          inShape: `[B, S, ${d_m}]`,
          outShape: `[B, S, ${d_m}]`,
          y: 270,
        },
        4: {
          id: `node_ffn_${Date.now()}`,
          label: "SwiGLU FFN",
          layerType: "SwiGLU",
          category: "feedforward",
          config: { d_model: d_m, d_ff },
          inShape: `[B, S, ${d_m}]`,
          outShape: `[B, S, ${d_m}]`,
          y: 390,
        },
        5: {
          id: `node_head_${Date.now()}`,
          label: "LM Head",
          layerType: "Linear",
          category: "embedding",
          config: { in_features: d_m, out_features: v_s, bias: false },
          inShape: `[B, S, ${d_m}]`,
          outShape: `[B, S, ${v_s}]`,
          y: 510,
        },
      };

      const target = stepConfigs[stepNum];
      if (!target) return;

      const newNode: Node = {
        id: target.id,
        type: "archNode",
        position: { x: 250, y: target.y },
        data: {
          label: target.label,
          category: target.category,
          layerType: target.layerType,
          config: target.config,
          inputShape: target.inShape,
          outputShape: target.outShape,
          params: 0,
          isValid: true,
        },
      };

      setNodes((nds) => [...nds, newNode]);
      setSelectedNodeId(target.id);

      // Auto-wire to last existing node
      if (nodes.length > 0) {
        const lastNode = nodes[nodes.length - 1];
        setEdges((eds) => [
          ...eds,
          {
            id: `e_${lastNode.id}_${target.id}`,
            source: lastNode.id,
            target: target.id,
            animated: true,
          },
        ]);
      }

      setStatusMessage(`Step ${stepNum} (${target.label}) placed on canvas`);
      setTimeout(() => setStatusMessage(null), 3000);
    },
    [globalConfig, nodes, setNodes, setEdges]
  );

  const handleDeleteNode = useCallback(
    (nodeId: string) => {
      setNodes((nds) => nds.filter((n) => n.id !== nodeId));
      setEdges((eds) => eds.filter((e) => e.source !== nodeId && e.target !== nodeId));
      setSelectedNodeId(null);
    },
    [setNodes, setEdges]
  );

  const handleDuplicateSelectedNode = useCallback(() => {
    if (!selectedNodeId) return;
    const nodeToDup = nodes.find((n) => n.id === selectedNodeId);
    if (!nodeToDup) return;
    const newId = `node_${Date.now().toString().slice(-6)}`;
    const newNode: Node = {
      ...nodeToDup,
      id: newId,
      position: {
        x: nodeToDup.position.x + 30,
        y: nodeToDup.position.y + 30,
      },
      data: {
        ...nodeToDup.data,
        label: `${nodeToDup.data.label} (Copy)`,
      },
    };
    setNodes((nds) => [...nds, newNode]);
    setSelectedNodeId(newId);
    setStatusMessage(`Duplicated '${nodeToDup.data.label}'`);
    setTimeout(() => setStatusMessage(null), 2500);
  }, [selectedNodeId, nodes, setNodes]);

  // Canvas building zone keyboard shortcuts: Del/Backspace to delete selected, Ctrl+D to duplicate
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable)
      ) {
        return;
      }

      if ((e.key === "Delete" || e.key === "Backspace") && selectedNodeId) {
        e.preventDefault();
        handleDeleteNode(selectedNodeId);
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "d" && selectedNodeId) {
        e.preventDefault();
        handleDuplicateSelectedNode();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedNodeId, handleDeleteNode, handleDuplicateSelectedNode]);

  const handleCreateScratch = useCallback(() => {
    setNodes([]);
    setEdges([]);
    setSelectedNodeId(null);
    setGlobalConfig({
      id: `custom_net_${Date.now().toString().slice(-6)}`,
      name: "Custom Architecture (Scratch)",
      task_type: "causal_lm",
      architecture_type: "custom",
      vocab_size: 12288,
      d_model: 512,
      n_layers: 4,
      n_q: 8,
      n_kv: 2,
      d_ff: 1024,
      kernel_size: 3,
      tied_embeddings: false,
    });
    setStatusMessage("Blank canvas ready. Build your architecture from scratch!");
    setTimeout(() => setStatusMessage(null), 3500);
  }, [setNodes, setEdges]);

  // Master Network Loader: handles templates, custom user models, and URL parameters
  const handleLoadNetwork = useCallback(
    async (networkId: string) => {
      if (!networkId) return;

      if (networkId === "custom_scratch" || networkId === "scratch") {
        handleCreateScratch();
        return;
      }

      try {
        let fullNet: any = null;
        try {
          fullNet = await api.getNetwork(networkId);
        } catch {
          // Fallback to presets or templates
        }

        if (!fullNet && PRESET_GRAPHS[networkId]) {
          const preset = PRESET_GRAPHS[networkId];
          setGlobalConfig(preset.config);
          setNodes(preset.nodes);
          setEdges(preset.edges);
          setSelectedNodeId(null);
          setStatusMessage(`Loaded architecture preset: ${preset.config.name}`);
          setTimeout(() => setStatusMessage(null), 3000);
          return;
        }

        if (!fullNet) {
          const t = templates.find((item) => item.id === networkId);
          if (t) fullNet = t;
        }

        if (!fullNet) {
          const r = registeredNetworks.find((item) => item.id === networkId);
          if (r) fullNet = r;
        }

        if (!fullNet) {
          setStatusMessage(`Network '${networkId}' not found.`);
          setTimeout(() => setStatusMessage(null), 3500);
          return;
        }

        // 1. Update Global Config
        setGlobalConfig({
          id: fullNet.id || networkId,
          name: fullNet.name || networkId,
          task_type: fullNet.task_type || fullNet.supported_tasks?.[0] || "causal_lm",
          architecture_type: fullNet.architecture_type || fullNet.architecture_family || "decoder_only",
          vocab_size: fullNet.vocab_size || 12288,
          d_model: fullNet.d_model || 512,
          n_layers: fullNet.n_layers || 8,
          n_q: fullNet.n_q || 8,
          n_kv: fullNet.n_kv || 2,
          d_ff: fullNet.d_ff || 1280,
          kernel_size: fullNet.kernel_size || 4,
          tied_embeddings: fullNet.tied_embeddings ?? fullNet.tie_word_embeddings ?? true,
        });

        // 2. Restore Visual Graph (nodes & edges)
        if (Array.isArray(fullNet.nodes) && fullNet.nodes.length > 0) {
          setNodes(fullNet.nodes);
          setEdges(Array.isArray(fullNet.edges) ? fullNet.edges : []);
        } else if (Array.isArray(fullNet.layers) && fullNet.layers.length > 0) {
          // Reconstruct graph from layers list
          const reconstructedNodes: Node[] = fullNet.layers.map((l: any, i: number) => {
            const layerType = l.type || l.layerType || "Linear";
            const category = LAYER_TYPE_CATEGORY_MAP[layerType] || "structural";
            return {
              id: l.id || `node-${i + 1}`,
              type: "archNode",
              position: l.position || { x: 250, y: 30 + i * 120 },
              data: {
                label: l.label || `${layerType} Layer`,
                category,
                layerType,
                config: l.config || {},
                inputShape: l.inputShape || "[B, S, D]",
                outputShape: l.outputShape || "[B, S, D]",
                params: l.params || 0,
                isValid: true,
              },
            };
          });

          const reconstructedEdges: Edge[] = reconstructedNodes.slice(0, -1).map((n, i) => ({
            id: `e${i}-${i + 1}`,
            source: n.id,
            target: reconstructedNodes[i + 1].id,
            animated: true,
          }));

          setNodes(reconstructedNodes);
          setEdges(reconstructedEdges);
        } else if (PRESET_GRAPHS[networkId]) {
          setNodes(PRESET_GRAPHS[networkId].nodes);
          setEdges(PRESET_GRAPHS[networkId].edges);
        } else {
          // Synthesize canonical LLM pipeline
          const d_m = fullNet.d_model || 512;
          const v_s = fullNet.vocab_size || 12288;
          const n_l = fullNet.n_layers || 8;
          const n_q = fullNet.n_q || 8;
          const n_kv = fullNet.n_kv || 2;
          const d_ff = fullNet.d_ff || d_m * 4;

          const syntheticNodes: Node[] = [
            {
              id: "node-embed",
              type: "archNode",
              position: { x: 250, y: 30 },
              data: {
                label: "Token & Position Embed",
                category: "embedding",
                layerType: "Embedding",
                config: { vocab_size: v_s, d_model: d_m },
                inputShape: "[B, S]",
                outputShape: `[B, S, ${d_m}]`,
                params: v_s * d_m,
                isValid: true,
              },
            },
            {
              id: "node-norm1",
              type: "archNode",
              position: { x: 250, y: 150 },
              data: {
                label: "Input RMSNorm",
                category: "norm",
                layerType: "RMSNorm",
                config: { d_model: d_m, eps: 1e-6 },
                inputShape: `[B, S, ${d_m}]`,
                outputShape: `[B, S, ${d_m}]`,
                params: d_m,
                isValid: true,
              },
            },
            {
              id: "node-attn",
              type: "archNode",
              position: { x: 250, y: 270 },
              data: {
                label: `Attention Blocks (${n_l}x)`,
                category: "attention",
                layerType: n_kv === n_q ? "MultiheadAttention" : "GQAAttention",
                config: { d_model: d_m, n_q, n_kv },
                inputShape: `[B, S, ${d_m}]`,
                outputShape: `[B, S, ${d_m}]`,
                params: 4 * d_m * d_m * n_l,
                isValid: true,
              },
            },
            {
              id: "node-ffn",
              type: "archNode",
              position: { x: 250, y: 390 },
              data: {
                label: `SwiGLU FFN Blocks (${n_l}x)`,
                category: "feedforward",
                layerType: "SwiGLU",
                config: { d_model: d_m, d_ff },
                inputShape: `[B, S, ${d_m}]`,
                outputShape: `[B, S, ${d_m}]`,
                params: 3 * d_m * d_ff * n_l,
                isValid: true,
              },
            },
            {
              id: "node-head",
              type: "archNode",
              position: { x: 250, y: 510 },
              data: {
                label: "LM Head (Projection)",
                category: "embedding",
                layerType: "Linear",
                config: { in_features: d_m, out_features: v_s, bias: false },
                inputShape: `[B, S, ${d_m}]`,
                outputShape: `[B, S, ${v_s}]`,
                params: 0,
                isValid: true,
              },
            },
          ];

          const syntheticEdges: Edge[] = [
            { id: "e1-2", source: "node-embed", target: "node-norm1", animated: true },
            { id: "e2-3", source: "node-norm1", target: "node-attn", animated: true },
            { id: "e3-4", source: "node-attn", target: "node-ffn", animated: true },
            { id: "e4-5", source: "node-ffn", target: "node-head", animated: true },
          ];

          setNodes(syntheticNodes);
          setEdges(syntheticEdges);
        }

        setSelectedNodeId(null);
        setStatusMessage(`Loaded architecture: ${fullNet.name || networkId}`);
        setTimeout(() => setStatusMessage(null), 3500);
      } catch (e: any) {
        console.error("Failed to load network:", e);
        setStatusMessage(`Failed to load network: ${e.message}`);
        setTimeout(() => setStatusMessage(null), 4000);
      }
    },
    [handleCreateScratch, templates, registeredNetworks, setNodes, setEdges]
  );

  // URL Query parameter navigation: ?network=<id> or ?id=<id> or ?template=<id> or ?scratch=true
  useEffect(() => {
    if (typeof window !== "undefined") {
      const search = window.location.search;
      if (search) {
        const params = new URLSearchParams(search);
        if (params.get("scratch") === "true") {
          handleCreateScratch();
        } else {
          const target =
            params.get("network") ||
            params.get("id") ||
            params.get("model") ||
            params.get("template");
          if (target) {
            handleLoadNetwork(target);
          }
        }
      }
    }
  }, [handleCreateScratch, handleLoadNetwork]);

  const handleSaveNetwork = async () => {
    setIsSaving(true);
    try {
      const payload = {
        ...globalConfig,
        layers: nodes.map((n) => ({
          id: n.id,
          type: n.data.layerType,
          label: n.data.label,
          config: n.data.config,
          position: n.position,
        })),
        nodes: nodes,
        edges: edges,
        parameters: estimation?.parameters,
        memory: estimation?.memory,
      };
      await api.saveNetwork(payload);
      setStatusMessage(`Architecture '${globalConfig.name}' saved to Registry!`);
      await fetchAllNetworks();
      setTimeout(() => setStatusMessage(null), 4000);
    } catch (e: any) {
      alert(`Save error: ${e.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  const handleCloneNetwork = async () => {
    const newName = prompt(`Enter a new name for the cloned model:`, `Clone of ${globalConfig.name}`);
    if (!newName) return;
    const cleanId = newName.toLowerCase().replace(/[^a-z0-9_-]/g, "_").slice(0, 48);
    const newId = `${cleanId}_${Date.now().toString().slice(-4)}`;
    try {
      await api.cloneNetwork(globalConfig.id, newId, newName);
      setGlobalConfig((prev) => ({ ...prev, id: newId, name: newName }));
      await fetchAllNetworks();
      setStatusMessage(`Architecture cloned as '${newName}'!`);
      setTimeout(() => setStatusMessage(null), 4000);
    } catch (e: any) {
      alert(`Clone error: ${e.message}`);
    }
  };

  const handleDeleteNetwork = async () => {
    try {
      await api.deleteNetwork(globalConfig.id);
      setConfirmDeleteNet(false);
      setStatusMessage(`Deleted '${globalConfig.name}' from Registry`);
      await fetchAllNetworks();
      handleLoadNetwork("rocanon_48m");
    } catch (e: any) {
      alert(`Delete error: ${e.message}`);
    }
  };

  const selectedNode = nodes.find((n) => n.id === selectedNodeId) || null;

  const defaultTemplateIds = [
    "rocanon_48m",
    "mini_gpt_15m",
    "bert_encoder_25m",
    "conv1d_network_4m",
    "deep_mlp_5m",
    "hybrid_cnn_transformer_20m",
  ];
  const isCustomModel = !defaultTemplateIds.includes(globalConfig.id);

  // Grouped list of registered models
  const userCustomNetworks = registeredNetworks.filter((n) => !defaultTemplateIds.includes(n.id));
  const baseTemplates = templates.length > 0 ? templates : registeredNetworks.filter((n) => defaultTemplateIds.includes(n.id));

  return (
    <div className="h-[calc(100vh-7.5rem)] flex flex-col space-y-3">
      {/* Workspace Area: Palette + Graph Canvas + Inspector */}
      <div className="flex-1 flex gap-3 min-h-0">
        {/* Left Palette */}
        <ComponentPalette
          onAddNode={handleAddPaletteNode}
          onShowInfo={setInfoModalLayer}
        />

        {/* Center Canvas */}
        <div className="flex-1 h-full min-h-[450px] bg-[#171717] rounded-lg overflow-hidden relative border border-[#333333]">
          <ReactFlowProvider>
            <ReactFlow
              nodes={nodes.map((n) => ({
                ...n,
                data: {
                  ...n.data,
                  onShowInfo: setInfoModalLayer,
                },
              }))}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onNodeClick={onNodeClick}
              onPaneClick={onPaneClick}
              onNodesDelete={(deleted) => {
                deleted.forEach((n) => handleDeleteNode(n.id));
              }}
              deleteKeyCode={["Delete", "Backspace"]}
              nodeTypes={nodeTypes}
              fitView
            >
              <Background
                variant={BackgroundVariant.Dots}
                gap={24}
                size={1}
                color="rgba(255, 255, 255, 0.05)"
              />
              <Controls className="!bg-[#202020] !border-[#333333] !rounded !text-[#A3A3A3]" />
              <MiniMap
                nodeColor="#737373"
                maskColor="rgba(20, 20, 20, 0.85)"
                className="!bg-[#202020] !border !border-[#333333] !rounded"
              />
            </ReactFlow>
          </ReactFlowProvider>

          {/* Canvas Floating Top Info & Model Switcher Toolbar */}
          <div className="absolute top-2.5 left-2.5 right-2.5 flex flex-wrap items-center justify-between gap-2 pointer-events-none z-10">
            {/* Left: Active Model Switcher */}
            <div className="flex items-center gap-2 pointer-events-auto">
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#202020]/95 border border-[#333333] shadow-sm">
                <span className="text-[10px] text-[#737373] uppercase font-mono">MODEL:</span>
                <div className="relative flex items-center">
                  <select
                    value={globalConfig.id}
                    onChange={(e) => handleLoadNetwork(e.target.value)}
                    className="bg-transparent text-[#F2F2F2] font-medium text-xs font-mono pr-4 focus:outline-none cursor-pointer appearance-none"
                  >
                    {userCustomNetworks.length > 0 && (
                      <optgroup label="Custom Models">
                        {userCustomNetworks.map((m) => (
                          <option key={m.id} value={m.id} className="bg-[#202020] text-[#F2F2F2]">
                            {m.name} ({formatParamCount(m.total_parameters || 0)})
                          </option>
                        ))}
                      </optgroup>
                    )}
                    <optgroup label="Templates">
                      {baseTemplates.map((t) => (
                        <option key={t.id} value={t.id} className="bg-[#202020] text-[#F2F2F2]">
                          {t.name}
                        </option>
                      ))}
                    </optgroup>
                  </select>
                  <ChevronDown className="w-3 h-3 text-[#737373] absolute right-0 pointer-events-none" />
                </div>
              </div>

              {isCustomModel ? (
                <span className="px-1.5 py-0.5 rounded bg-[#262626] border border-[#333333] text-[#A3A3A3] text-[10px] font-mono">
                  Custom
                </span>
              ) : (
                <span className="px-1.5 py-0.5 rounded bg-[#1B1B1B] border border-[#262626] text-[#737373] text-[10px] font-mono">
                  Template
                </span>
              )}

              {statusMessage && (
                <span className="px-2 py-0.5 rounded bg-[#262626] border border-[#333333] font-mono text-xs text-[#F2F2F2]">
                  {statusMessage}
                </span>
              )}
            </div>

            {/* Right: Quick Action Buttons */}
            <div className="flex items-center gap-1.5 pointer-events-auto">
              {selectedNodeId && (
                <>
                  <button
                    type="button"
                    onClick={handleDuplicateSelectedNode}
                    className="btn btn-secondary btn-sm flex items-center gap-1"
                    title="Duplicate selected block (Ctrl+D)"
                  >
                    <Copy className="w-3 h-3" />
                    <span className="hidden sm:inline">Duplicate</span>
                    <span className="text-[9px] px-1 py-0.2 rounded bg-[#303030] text-[#A3A3A3] font-mono">^D</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => handleDeleteNode(selectedNodeId)}
                    className="btn btn-sm bg-[#262626] border border-[#333333] hover:border-[#f87171]/50 text-[#f87171] flex items-center gap-1"
                    title="Delete selected block (Del / Backspace)"
                  >
                    <Trash2 className="w-3 h-3" />
                    <span className="hidden sm:inline">Delete</span>
                    <span className="text-[9px] px-1 py-0.2 rounded bg-[#303030] text-[#f87171] font-mono">Del</span>
                  </button>
                </>
              )}

              <button
                type="button"
                onClick={() => setShowBlueprintGuide((prev) => !prev)}
                className={cn(
                  "btn btn-sm",
                  showBlueprintGuide ? "btn-primary" : "btn-secondary"
                )}
                title="Toggle Architecture Blueprint & Layer Order Guide"
              >
                <Lightbulb className="w-3 h-3" />
                <span className="hidden sm:inline">Guide</span>
                <span className="text-[10px] opacity-75">
                  {placedStagesCount}/5
                </span>
              </button>

              <button
                type="button"
                onClick={handleSaveNetwork}
                disabled={isSaving}
                className="btn btn-primary btn-sm flex items-center gap-1"
                title="Save current model configuration and layers"
              >
                <Save className="w-3 h-3" />
                <span>{isSaving ? "Saving..." : isCustomModel ? "Save Changes" : "Save Model"}</span>
              </button>

              <button
                type="button"
                onClick={handleCreateScratch}
                className="btn btn-secondary btn-sm flex items-center gap-1"
                title="Start a new architecture on a blank canvas"
              >
                <Plus className="w-3 h-3" />
                <span className="hidden sm:inline">New</span>
              </button>

              {nodes.length > 0 && (
                <button
                  type="button"
                  onClick={() => {
                    setNodes([]);
                    setEdges([]);
                    setSelectedNodeId(null);
                  }}
                  className="btn btn-ghost btn-sm text-[#737373] hover:text-[#f87171]"
                  title="Clear canvas"
                >
                  <Trash2 className="w-3 h-3" />
                  <span>Clear</span>
                </button>
              )}
            </div>
          </div>

          {/* Floating Shadow Architecture Blueprint Guide (When canvas has nodes) */}
          {showBlueprintGuide && nodes.length > 0 && (
            <div className="absolute top-12 right-3 z-20 w-80 max-h-[calc(100%-60px)] overflow-y-auto bg-[#202020] p-3 rounded-lg border border-[#333333] shadow-xl space-y-2.5 font-mono text-xs">
              <div className="flex items-center justify-between pb-1.5 border-b border-[#262626]">
                <div className="flex items-center gap-1.5">
                  <Lightbulb className="w-3.5 h-3.5 text-[#737373]" />
                  <span className="font-semibold text-[#F2F2F2] uppercase text-[11px]">
                    Blueprint Flow
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="text-[10px] px-1.5 py-0.2 rounded bg-[#1B1B1B] text-[#A3A3A3]">
                    {placedStagesCount}/5 Stages
                  </span>
                  <button
                    onClick={() => setShowBlueprintGuide(false)}
                    className="p-1 rounded text-[#737373] hover:text-[#F2F2F2]"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              </div>

              <p className="text-[10px] text-[#737373] leading-relaxed">
                Standard autoregressive LLM layer execution flow:
              </p>

              <div className="space-y-1">
                {blueprintStatus.map((stage, idx) => (
                  <React.Fragment key={stage.step}>
                    <div
                      className={cn(
                        "p-2 rounded border text-[11px] transition-colors",
                        stage.placed
                          ? "bg-[#1B1B1B] border-[#262626]"
                          : "bg-[#202020] border-dashed border-[#404040]"
                      )}
                    >
                      <div className="flex items-center justify-between gap-1 mb-0.5">
                        <div className="flex items-center gap-1.5 font-medium">
                          <span className="text-[#737373] text-[10px]">#{stage.step}</span>
                          <span className="text-[#F2F2F2]">{stage.name}</span>
                        </div>
                        {stage.placed ? (
                          <span className="text-[9px] px-1 py-0.2 rounded bg-[#262626] text-[#4ade80] flex items-center gap-0.5">
                            <Check className="w-2.5 h-2.5" /> Placed
                          </span>
                        ) : (
                          <button
                            type="button"
                            onClick={() => handlePlaceBlueprintStage(stage.step)}
                            className="text-[9px] px-1.5 py-0.2 rounded bg-[#262626] hover:bg-[#333333] text-[#D4D4D4] border border-[#333333]"
                          >
                            + Insert
                          </button>
                        )}
                      </div>
                      <div className="text-[9px] text-[#525252] mb-0.5 font-mono">
                        {stage.shape(globalConfig.d_model || 512, globalConfig.vocab_size || 12288)}
                      </div>
                      <div className="text-[10px] text-[#737373] leading-tight">
                        {stage.description}
                      </div>
                    </div>

                    {idx < blueprintStatus.length - 1 && (
                      <div className="flex justify-center -my-0.5 text-[#404040]">
                        <ArrowDown className="w-2.5 h-2.5" />
                      </div>
                    )}
                  </React.Fragment>
                ))}
              </div>

              <div className="pt-1.5 border-t border-[#262626]">
                <button
                  type="button"
                  onClick={handleAutoPlaceRecommendedPipeline}
                  className="btn btn-secondary btn-sm w-full"
                >
                  <Sparkles className="w-3 h-3 text-[#A3A3A3]" /> Align Recommended Flow
                </button>
              </div>
            </div>
          )}

          {/* Empty Canvas Overlay */}
          {nodes.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none p-6 z-10 overflow-y-auto">
              <div className="max-w-md w-full bg-[#202020] border border-[#333333] rounded-lg p-5 pointer-events-auto shadow-xl space-y-3 font-mono">
                <div className="flex items-center justify-between pb-2 border-b border-[#262626]">
                  <div className="flex items-center gap-2">
                    <Layers className="w-4 h-4 text-[#737373]" />
                    <h3 className="text-xs font-semibold text-[#F2F2F2] uppercase">
                      Architecture Blueprint
                    </h3>
                  </div>
                  <button
                    type="button"
                    onClick={handleAutoPlaceRecommendedPipeline}
                    className="btn btn-primary btn-sm flex items-center gap-1"
                  >
                    <Sparkles className="w-3 h-3" /> Place Full Flow
                  </button>
                </div>

                {/* 5-Stage Shadow Flow in Order */}
                <div className="space-y-1.5">
                  {blueprintStatus.map((stage, idx) => (
                    <React.Fragment key={stage.step}>
                      <div className="p-2 rounded bg-[#1B1B1B] border border-[#262626] flex items-center justify-between gap-2 text-xs">
                        <div className="flex items-center gap-2 min-w-0">
                          <span className="w-4 h-4 rounded-full bg-[#262626] text-[#A3A3A3] text-[9px] flex items-center justify-center shrink-0">
                            {stage.step}
                          </span>
                          <div className="min-w-0">
                            <span className="font-medium text-[#F2F2F2] truncate block">{stage.name}</span>
                            <span className="text-[10px] text-[#737373] truncate block">{stage.description}</span>
                          </div>
                        </div>

                        <button
                          type="button"
                          onClick={() => handlePlaceBlueprintStage(stage.step)}
                          className="btn btn-secondary btn-sm px-2 text-[10px] shrink-0"
                        >
                          + Place
                        </button>
                      </div>

                      {idx < blueprintStatus.length - 1 && (
                        <div className="flex justify-center -my-0.5 text-[#333333]">
                          <ArrowDown className="w-2.5 h-2.5" />
                        </div>
                      )}
                    </React.Fragment>
                  ))}
                </div>

                <div className="p-2 rounded bg-[#1B1B1B] border border-[#262626] flex items-center justify-between text-[10px] text-[#737373]">
                  <span>d_model: <strong className="text-[#F2F2F2]">{globalConfig.d_model || 512}</strong> | vocab: <strong className="text-[#F2F2F2]">{globalConfig.vocab_size || 12288}</strong></span>
                  <span>Syncs automatically across layers</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Inspector */}
        <ConfigInspector
          selectedNode={selectedNode}
          onUpdateNodeConfig={handleUpdateNodeConfig}
          onDeleteNode={handleDeleteNode}
          globalConfig={globalConfig}
          onUpdateGlobalConfig={handleUpdateGlobalConfig}
          onSaveNetwork={handleSaveNetwork}
          onCloneNetwork={handleCloneNetwork}
          onLoadTemplate={handleLoadNetwork}
          onNewScratch={handleCreateScratch}
          onDeleteNetwork={() => setConfirmDeleteNet(true)}
          onShowInfo={setInfoModalLayer}
          templates={templates}
          registeredNetworks={registeredNetworks}
          isSaving={isSaving}
        />
      </div>

      {/* Bottom Console: Architecture Metrics & Validation */}
      <div className="h-20 panel bg-[#202020] border-[#333333] rounded-lg p-2.5 flex items-center justify-between gap-6 shrink-0 font-mono text-xs select-none">
        {/* Left: Parameter Calculations */}
        <div className="flex items-center gap-5">
          <div className="space-y-0.5">
            <span className="text-[10px] text-[#737373] uppercase font-medium">
              Parameters
            </span>
            <div className="text-base font-semibold font-mono-num text-[#F2F2F2]">
              {formatParamCount(estimation?.parameters?.total_params || 48531968)}{" "}
              <span className="text-[10px] text-[#737373] font-normal">
                ({(estimation?.parameters?.total_params || 48531968).toLocaleString()})
              </span>
            </div>
            <span className="text-[10px] text-[#525252]">
              Trainable: {formatParamCount(estimation?.parameters?.trainable_params || 48531968)}
            </span>
          </div>

          <div className="h-8 w-px bg-[#262626]" />

          {/* Memory Estimations */}
          <div className="space-y-0.5">
            <span className="text-[10px] text-[#737373] uppercase font-medium flex items-center gap-1">
              <HardDrive className="w-3 h-3 text-[#737373]" /> Training VRAM
            </span>
            <div className="text-base font-semibold font-mono-num text-[#F2F2F2]">
              {estimation?.memory?.total_training_vram_mb
                ? `${Math.round(estimation.memory.total_training_vram_mb)} MB`
                : "2,630 MB"}
            </div>
            <span className="text-[10px] text-[#525252]">
              Weights: {estimation?.memory?.model_weights_mb || 97} MB | Opt: {estimation?.memory?.optimizer_states_mb || 388} MB
            </span>
          </div>

          <div className="h-8 w-px bg-[#262626]" />

          {/* Hardware Compatibility */}
          <div className="space-y-0.5">
            <span className="text-[10px] text-[#737373] uppercase font-medium">
              Hardware
            </span>
            <div>
              {estimation?.memory?.safety_status === "SAFE" || estimation?.memory?.fits_6gb_gpu !== false ? (
                <span className="px-1.5 py-0.2 rounded bg-[rgba(74,222,128,0.08)] border border-[rgba(74,222,128,0.25)] text-[#4ade80] text-[10px] font-medium">
                  Fits Budget ({estimation?.memory?.hardware_target || "SAFE"})
                </span>
              ) : estimation?.memory?.safety_status === "TIGHT" ? (
                <span className="px-1.5 py-0.2 rounded bg-[rgba(250,204,21,0.08)] border border-[rgba(250,204,21,0.25)] text-[#facc15] text-[10px] font-medium">
                  Tight Margin
                </span>
              ) : (
                <span className="px-1.5 py-0.2 rounded bg-[rgba(248,113,113,0.08)] border border-[rgba(248,113,113,0.25)] text-[#f87171] text-[10px] font-medium">
                  Exceeds Budget
                </span>
              )}
            </div>
            <span className="text-[10px] text-[#525252]">
              Batch 4 | Seq 256 | BF16
            </span>
          </div>
        </div>

        {/* Right: Validation status & deploy */}
        <div className="flex items-center gap-3">
          <div className="text-right">
            <div className="flex items-center justify-end gap-1 font-medium">
              {validation.valid ? (
                <>
                  <CheckCircle2 className="w-3.5 h-3.5 text-[#4ade80]" />
                  <span className="text-[#4ade80] text-xs">Validated</span>
                </>
              ) : (
                <>
                  <AlertTriangle className="w-3.5 h-3.5 text-[#f87171]" />
                  <span className="text-[#f87171] text-xs">Issues Found</span>
                </>
              )}
            </div>
            <p className="text-[10px] text-[#737373] truncate max-w-xs">
              {(validation?.issues && validation.issues.length > 0)
                ? validation.issues[0]
                : "Ready for model instantiation."}
            </p>
          </div>

          <a
            href={`/training?network=${encodeURIComponent(globalConfig.id)}`}
            className="btn btn-primary btn-sm flex items-center gap-1.5 shrink-0"
          >
            <Play className="w-3 h-3 fill-current" /> Deploy to Training
          </a>
        </div>
      </div>

      {/* Educational Layer Documentation Modal */}
      <LayerInfoModal
        isOpen={!!infoModalLayer}
        layerType={infoModalLayer}
        onClose={() => setInfoModalLayer(null)}
      />

      {/* Confirm Network Deletion Dialog */}
      <ConfirmDialog
        isOpen={confirmDeleteNet}
        title="DELETE ARCHITECTURE"
        message={`Are you sure you want to delete '${globalConfig.name}' (${globalConfig.id}) from the registry? Default templates cannot be deleted.`}
        confirmLabel="DELETE ARCHITECTURE"
        cancelLabel="CANCEL"
        variant="danger"
        onConfirm={handleDeleteNetwork}
        onCancel={() => setConfirmDeleteNet(false)}
      />
    </div>
  );
};
