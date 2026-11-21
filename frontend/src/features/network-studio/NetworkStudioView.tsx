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
  rocanon_48_5m: {
    config: {
      id: "rocanon_48_5m",
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
  rocanon_48m: {
    config: {
      id: "rocanon_48_5m",
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
    id: "rocanon_48_5m",
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

  const handleDeleteNode = (nodeId: string) => {
    setNodes((nds) => nds.filter((n) => n.id !== nodeId));
    setEdges((eds) => eds.filter((e) => e.source !== nodeId && e.target !== nodeId));
    setSelectedNodeId(null);
  };

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
      handleLoadNetwork("rocanon_48_5m");
    } catch (e: any) {
      alert(`Delete error: ${e.message}`);
    }
  };

  const selectedNode = nodes.find((n) => n.id === selectedNodeId) || null;

  const defaultTemplateIds = [
    "rocanon_48m",
    "rocanon_48_5m",
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
        <div className="flex-1 h-full min-h-[450px] glass-panel rounded-lg overflow-hidden relative border border-[#273548]">
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
              nodeTypes={nodeTypes}
              fitView
              className="sci-fi-grid"
            >
              <Background
                variant={BackgroundVariant.Dots}
                gap={24}
                size={1}
                color="rgba(69, 207, 255, 0.15)"
              />
              <Controls className="!bg-[#0E1420] !border-[#273548] !rounded-lg !text-[#45CFFF]" />
              <MiniMap
                nodeColor="#45CFFF"
                maskColor="rgba(9, 13, 20, 0.8)"
                className="!bg-[#0E1420]/90 !border !border-[#273548] !rounded-lg"
              />
            </ReactFlow>
          </ReactFlowProvider>

          {/* Canvas Floating Top Info & Model Switcher Toolbar */}
          <div className="absolute top-3 left-3 right-3 flex flex-wrap items-center justify-between gap-2 pointer-events-none z-10">
            {/* Left: Active Model Switcher */}
            <div className="flex items-center gap-2 pointer-events-auto">
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-[#0E1420]/95 border border-[#273548] backdrop-blur-md shadow-md">
                <span className="text-[10px] text-[#91A0B4] font-bold uppercase font-mono">MODEL:</span>
                <div className="relative flex items-center">
                  <select
                    value={globalConfig.id}
                    onChange={(e) => handleLoadNetwork(e.target.value)}
                    className="bg-transparent text-[#45CFFF] font-bold text-xs font-mono pr-5 focus:outline-none cursor-pointer appearance-none"
                  >
                    {userCustomNetworks.length > 0 && (
                      <optgroup label="MY REGISTERED ARCHITECTURES">
                        {userCustomNetworks.map((m) => (
                          <option key={m.id} value={m.id} className="bg-[#0E1420] text-[#ECF3FC]">
                            {m.name} ({formatParamCount(m.total_parameters || 0)})
                          </option>
                        ))}
                      </optgroup>
                    )}
                    <optgroup label="BASE ARCHITECTURE TEMPLATES">
                      {baseTemplates.map((t) => (
                        <option key={t.id} value={t.id} className="bg-[#0E1420] text-[#ECF3FC]">
                          {t.name}
                        </option>
                      ))}
                    </optgroup>
                  </select>
                  <ChevronDown className="w-3 h-3 text-[#45CFFF] absolute right-0 pointer-events-none" />
                </div>
              </div>

              {isCustomModel ? (
                <span className="px-2 py-0.5 rounded-md bg-[#45CFFF]/15 border border-[#45CFFF]/30 text-[#45CFFF] text-[10px] font-bold font-mono">
                  CUSTOM (EDITABLE)
                </span>
              ) : (
                <span className="px-2 py-0.5 rounded-md bg-[#91A0B4]/15 border border-[#91A0B4]/30 text-[#91A0B4] text-[10px] font-bold font-mono">
                  BASE TEMPLATE
                </span>
              )}

              {statusMessage && (
                <span className="px-2.5 py-1 rounded-md bg-[#35D49A]/15 border border-[#35D49A]/40 font-mono text-xs text-[#35D49A] font-bold animate-in fade-in">
                  {statusMessage}
                </span>
              )}
            </div>

            {/* Right: Quick Action Buttons */}
            <div className="flex items-center gap-2 pointer-events-auto">
              <button
                type="button"
                onClick={() => setShowBlueprintGuide((prev) => !prev)}
                className={cn(
                  "flex items-center gap-1.5 px-3 py-1.5 rounded-md font-mono text-xs font-semibold transition-all shadow-sm border",
                  showBlueprintGuide
                    ? "bg-[#45CFFF]/15 border-[#45CFFF]/50 text-[#45CFFF]"
                    : "bg-[#131C29] border-[#273548] text-[#91A0B4] hover:text-[#ECF3FC]"
                )}
                title="Toggle Architecture Blueprint & Layer Order Guide"
              >
                <Lightbulb className="w-3.5 h-3.5 text-[#45CFFF]" />
                <span className="hidden sm:inline">Layer Order Guide</span>
                <span
                  className={cn(
                    "text-[10px] px-1.5 py-0.2 rounded font-bold font-mono",
                    placedStagesCount === 5
                      ? "bg-[#35D49A]/20 text-[#35D49A]"
                      : "bg-[#F59E0B]/20 text-[#F59E0B]"
                  )}
                >
                  {placedStagesCount}/5
                </span>
              </button>

              <button
                type="button"
                onClick={handleSaveNetwork}
                disabled={isSaving}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-[#45CFFF] hover:bg-[#80DEFF] text-[#090D14] font-mono text-xs font-bold transition-all shadow-md disabled:opacity-50"
                title="Save current model configuration and layers"
              >
                <Save className="w-3.5 h-3.5" />
                <span>{isSaving ? "SAVING..." : isCustomModel ? "SAVE CHANGES" : "SAVE TO REGISTRY"}</span>
              </button>

              <button
                type="button"
                onClick={handleCreateScratch}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-[#131C29] border border-[#273548] hover:border-[#45CFFF] text-[#ECF3FC] font-mono text-xs font-semibold transition-all shadow-sm"
                title="Start a new architecture on a blank canvas"
              >
                <Plus className="w-3.5 h-3.5 text-[#45CFFF]" />
                <span className="hidden sm:inline">+ New Architecture</span>
              </button>

              {nodes.length > 0 && (
                <button
                  type="button"
                  onClick={() => {
                    setNodes([]);
                    setEdges([]);
                    setSelectedNodeId(null);
                  }}
                  className="flex items-center gap-1 px-2.5 py-1.5 rounded-md bg-[#0E1420] border border-[#273548] hover:border-[#FF657A]/50 hover:bg-[#FF657A]/15 text-[#91A0B4] hover:text-[#FF657A] font-mono text-xs transition-all"
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
            <div className="absolute top-14 right-4 z-20 w-84 max-h-[calc(100%-80px)] overflow-y-auto glass-panel p-3.5 rounded-xl border border-[#45CFFF]/30 shadow-2xl backdrop-blur-xl animate-in fade-in zoom-in-95 duration-150 space-y-3 font-mono text-xs">
              <div className="flex items-center justify-between pb-2 border-b border-[#273548]">
                <div className="flex items-center gap-2">
                  <Lightbulb className="w-4 h-4 text-[#45CFFF]" />
                  <span className="font-bold text-[#ECF3FC] uppercase tracking-wider text-[11px]">
                    SHADOW BLUEPRINT ORDER
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span
                    className={cn(
                      "text-[10px] px-2 py-0.5 rounded font-bold font-mono",
                      placedStagesCount === 5
                        ? "bg-[#35D49A]/20 text-[#35D49A] border border-[#35D49A]/30"
                        : "bg-[#F59E0B]/20 text-[#F59E0B] border border-[#F59E0B]/30"
                    )}
                  >
                    {placedStagesCount}/5 STAGES
                  </span>
                  <button
                    onClick={() => setShowBlueprintGuide(false)}
                    className="p-1 rounded text-[#91A0B4] hover:text-[#ECF3FC] hover:bg-[#182333]"
                    title="Close Guide"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              <p className="text-[10px] text-[#91A0B4] leading-relaxed">
                Standard causal autoregressive LLM layer execution flow. Follow this order for proper forward/backward dynamics:
              </p>

              <div className="space-y-1.5">
                {blueprintStatus.map((stage, idx) => (
                  <React.Fragment key={stage.step}>
                    <div
                      className={cn(
                        "p-2.5 rounded-lg border text-[11px] transition-all relative",
                        stage.placed
                          ? "bg-[#0E1420]/80 border-[#35D49A]/30"
                          : "bg-[#131C29]/70 border-dashed border-[#F59E0B]/40 hover:border-[#45CFFF]/60"
                      )}
                    >
                      <div className="flex items-center justify-between gap-1.5 mb-1">
                        <div className="flex items-center gap-1.5 font-bold">
                          <span className="text-[#45CFFF] text-[10px]">#{stage.step}</span>
                          <span className="text-[#ECF3FC]">{stage.name}</span>
                        </div>
                        {stage.placed ? (
                          <span className="text-[9px] px-1.5 py-0.5 rounded bg-[#35D49A]/15 text-[#35D49A] font-bold flex items-center gap-1">
                            <Check className="w-2.5 h-2.5" /> PLACED
                          </span>
                        ) : (
                          <button
                            type="button"
                            onClick={() => handlePlaceBlueprintStage(stage.step)}
                            className="text-[9px] px-2 py-0.5 rounded bg-[#45CFFF]/15 hover:bg-[#45CFFF]/25 text-[#45CFFF] border border-[#45CFFF]/40 font-bold transition-colors"
                          >
                            + INSERT
                          </button>
                        )}
                      </div>
                      <div className="text-[9px] text-[#5E7188] mb-0.5 font-mono">
                        {stage.shape(globalConfig.d_model || 512, globalConfig.vocab_size || 12288)}
                      </div>
                      <div className="text-[10px] text-[#91A0B4] leading-tight">
                        {stage.description}
                      </div>
                    </div>

                    {idx < blueprintStatus.length - 1 && (
                      <div className="flex justify-center -my-0.5 text-[#45CFFF]/40">
                        <ArrowDown className="w-3 h-3" />
                      </div>
                    )}
                  </React.Fragment>
                ))}
              </div>

              <div className="pt-2 border-t border-[#273548]">
                <button
                  type="button"
                  onClick={handleAutoPlaceRecommendedPipeline}
                  className="w-full py-1.5 px-3 rounded-md bg-[#45CFFF]/15 hover:bg-[#45CFFF]/25 border border-[#45CFFF]/40 text-[#45CFFF] font-bold text-xs flex items-center justify-center gap-1.5 transition-colors shadow-sm"
                >
                  <Sparkles className="w-3.5 h-3.5" /> Reset & Align Recommended Pipeline
                </button>
              </div>
            </div>
          )}

          {/* Empty Canvas Overlay with Architectural Shadow Section */}
          {nodes.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none p-6 z-10 overflow-y-auto">
              <div className="max-w-xl w-full bg-[#0E1420]/95 border border-[#384C66] rounded-xl p-6 backdrop-blur-xl pointer-events-auto shadow-2xl relative my-auto">
                <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#45CFFF]/40 to-transparent pointer-events-none" />

                <div className="flex items-center justify-between pb-3 mb-3 border-b border-[#273548]">
                  <div className="flex items-center gap-2.5">
                    <div className="w-9 h-9 rounded-lg bg-[#131C29] border border-[#45CFFF]/40 flex items-center justify-center shadow-[0_0_12px_rgba(69,207,255,0.2)]">
                      <Layers className="w-4 h-4 text-[#45CFFF]" />
                    </div>
                    <div>
                      <h3 className="font-mono font-bold text-xs text-[#ECF3FC] uppercase tracking-wider">
                        LLM ARCHITECTURE BLUEPRINT & SHADOW ORDER
                      </h3>
                      <p className="text-[10px] text-[#91A0B4] mt-0.5">
                        Follow the canonical 5-stage pipeline for causal autoregressive models:
                      </p>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={handleAutoPlaceRecommendedPipeline}
                    className="py-1.5 px-3 rounded-md bg-[#45CFFF] hover:bg-[#80DEFF] text-[#090D14] font-bold text-xs flex items-center gap-1.5 transition-all shadow-md shrink-0"
                  >
                    <Sparkles className="w-3.5 h-3.5" /> Place Full Flow
                  </button>
                </div>

                {/* 5-Stage Shadow Flow in Order */}
                <div className="space-y-2 mb-4">
                  {blueprintStatus.map((stage, idx) => (
                    <React.Fragment key={stage.step}>
                      <div className="p-2.5 rounded-lg bg-[#131C29]/80 border border-[#273548] hover:border-[#45CFFF]/50 flex items-center justify-between gap-3 text-xs transition-colors">
                        <div className="flex items-center gap-2.5 min-w-0">
                          <span className="w-5 h-5 rounded-full bg-[#45CFFF]/15 border border-[#45CFFF]/40 text-[#45CFFF] text-[10px] font-bold flex items-center justify-center shrink-0">
                            {stage.step}
                          </span>
                          <div className="min-w-0">
                            <div className="flex items-center gap-2">
                              <span className="font-bold text-[#ECF3FC] truncate">{stage.name}</span>
                              <span className="text-[9px] px-1.5 py-0.2 rounded bg-[#0E1420] border border-[#273548] text-[#5E7188] font-mono shrink-0">
                                {stage.layerType}
                              </span>
                            </div>
                            <p className="text-[10px] text-[#91A0B4] truncate mt-0.5">
                              {stage.description}
                            </p>
                          </div>
                        </div>

                        <div className="flex items-center gap-2 shrink-0">
                          <span className="text-[9px] text-[#5E7188] font-mono hidden md:inline">
                            {stage.shape(globalConfig.d_model || 512, globalConfig.vocab_size || 12288)}
                          </span>
                          <button
                            type="button"
                            onClick={() => handlePlaceBlueprintStage(stage.step)}
                            className="px-2.5 py-1 rounded bg-[#45CFFF]/10 hover:bg-[#45CFFF]/20 border border-[#45CFFF]/40 text-[#45CFFF] text-[10px] font-bold transition-all"
                          >
                            + Place
                          </button>
                        </div>
                      </div>

                      {idx < blueprintStatus.length - 1 && (
                        <div className="flex justify-center -my-1 text-[#45CFFF]/40">
                          <ArrowDown className="w-3.5 h-3.5" />
                        </div>
                      )}
                    </React.Fragment>
                  ))}
                </div>

                <div className="p-2.5 rounded-lg bg-[#0E1420] border border-[#273548] flex items-center justify-between text-[11px] text-[#91A0B4]">
                  <span>Active Dimensions: <strong className="text-[#ECF3FC]">d_model={globalConfig.d_model || 512}</strong> | <strong className="text-[#ECF3FC]">vocab={globalConfig.vocab_size || 12288}</strong></span>
                  <span className="text-[#45CFFF] text-[10px]">d_model auto-syncs across all sections</span>
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
      <div className="h-26 glass-panel rounded-lg p-3 flex items-center justify-between gap-6 shrink-0 font-mono text-xs select-none relative overflow-hidden">
        <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#45CFFF]/15 to-transparent pointer-events-none" />

        {/* Left: Parameter Calculations */}
        <div className="flex items-center gap-6">
          <div className="space-y-0.5">
            <span className="text-[10px] text-[#5E7188] uppercase font-semibold">
              TOTAL PARAMETERS
            </span>
            <div className="text-xl font-bold font-mono-num text-[#45CFFF]">
              {formatParamCount(estimation?.parameters?.total_params || 48531968)}{" "}
              <span className="text-xs text-[#91A0B4] font-normal">
                ({(estimation?.parameters?.total_params || 48531968).toLocaleString()})
              </span>
            </div>
            <span className="text-[10px] text-[#91A0B4]">
              Trainable:{" "}
              {formatParamCount(estimation?.parameters?.trainable_params || 48531968)}
            </span>
          </div>

          <div className="h-10 w-px bg-[#273548]" />

          {/* Memory Estimations */}
          <div className="space-y-0.5">
            <span className="text-[10px] text-[#5E7188] uppercase font-semibold flex items-center gap-1">
              <HardDrive className="w-3 h-3 text-[#45CFFF]" /> TRAINING VRAM
            </span>
            <div className="text-xl font-bold font-mono-num text-[#ECF3FC]">
              {estimation?.memory?.total_training_vram_mb
                ? `${Math.round(estimation.memory.total_training_vram_mb)} MB`
                : "2,630 MB"}
            </div>
            <span className="text-[10px] text-[#91A0B4]">
              Weights: {estimation?.memory?.model_weights_mb || 97} MB | Opt:{" "}
              {estimation?.memory?.optimizer_states_mb || 388} MB
            </span>
          </div>

          <div className="h-10 w-px bg-[#273548]" />

          {/* Hardware Compatibility */}
          <div className="space-y-1">
            <span className="text-[10px] text-[#5E7188] uppercase font-semibold">
              HARDWARE COMPLIANCE
            </span>
            <div>
              {estimation?.memory?.safety_status === "SAFE" || estimation?.memory?.fits_6gb_gpu !== false ? (
                <span className="px-2 py-0.5 rounded-md bg-[#35D49A]/15 border border-[#35D49A]/30 text-[#35D49A] font-bold text-[10px]">
                  FITS HARDWARE BUDGET ({estimation?.memory?.hardware_target || "SAFE"})
                </span>
              ) : estimation?.memory?.safety_status === "TIGHT" ? (
                <span className="px-2 py-0.5 rounded-md bg-[#F2B85B]/15 border border-[#F2B85B]/30 text-[#F2B85B] font-bold text-[10px]">
                  TIGHT BUDGET MARGIN ({estimation?.memory?.total_training_vram_gb || 0} GB)
                </span>
              ) : (
                <span className="px-2 py-0.5 rounded-md bg-[#FF657A]/15 border border-[#FF657A]/30 text-[#FF657A] font-bold text-[10px]">
                  EXCEEDS ALLOCATION CEILING
                </span>
              )}
            </div>
            <span className="text-[10px] text-[#91A0B4]">
              Micro-batch 4 | Seq 256 | BF16
            </span>
          </div>
        </div>

        {/* Right: Pre-Flight Validation status */}
        <div className="flex items-center gap-4">
          <div className="text-right">
            <div className="flex items-center justify-end gap-1.5 font-bold">
              {validation.valid ? (
                <>
                  <CheckCircle2 className="w-4 h-4 text-[#35D49A]" />
                  <span className="text-[#35D49A]">ARCHITECTURE VALIDATED</span>
                </>
              ) : (
                <>
                  <AlertTriangle className="w-4 h-4 text-[#FF657A]" />
                  <span className="text-[#FF657A]">VALIDATION ISSUES</span>
                </>
              )}
            </div>
            <p className="text-[10px] text-[#91A0B4] mt-0.5">
              {(validation?.issues && validation.issues.length > 0)
                ? validation.issues[0]
                : "Ready for ModelFactory instantiation & execution pipeline."}
            </p>
          </div>

          <a
            href={`/training?network=${encodeURIComponent(globalConfig.id)}`}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-md bg-[#45CFFF] hover:bg-[#80DEFF] text-[#090D14] font-bold transition-all shadow-sm shrink-0"
          >
            <Play className="w-3.5 h-3.5 fill-current" /> DEPLOY TO TRAINING
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
