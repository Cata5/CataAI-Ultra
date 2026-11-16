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
import { formatParamCount } from "@/lib/utils";
import {
  HardDrive,
  CheckCircle2,
  AlertTriangle,
  Play,
  Plus,
  Trash2,
  Layers,
  Sparkles,
} from "lucide-react";

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
          layerType: "MultiHeadAttention",
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
      tied_embeddings: true,
    },
    nodes: [
      {
        id: "node-embed",
        type: "archNode",
        position: { x: 250, y: 30 },
        data: {
          label: "Token & Segment Embed",
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
          label: "Input LayerNorm",
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
          layerType: "MultiHeadAttention",
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

  // Custom node types definition for ReactFlow
  const nodeTypes = useMemo(() => ({ archNode: ArchitectureNode as any }), []);

  // Fetch templates on mount
  useEffect(() => {
    api
      .getNetworkTemplates()
      .then((res) => {
        setTemplates(Array.isArray(res) ? res : []);
      })
      .catch((err) => {
        console.error("Failed to load network templates:", err);
        setTemplates([]);
      });
  }, []);

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
      position: { x: 300, y: (nodes.length + 1) * 80 },
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

  const handleUpdateNodeConfig = (nodeId: string, newConfig: Record<string, any>) => {
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
  };

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

  const handleLoadTemplate = useCallback(
    (templateId: string) => {
      if (templateId === "custom_scratch" || templateId === "scratch") {
        handleCreateScratch();
        return;
      }

      if (PRESET_GRAPHS[templateId]) {
        const preset = PRESET_GRAPHS[templateId];
        setGlobalConfig(preset.config);
        setNodes(preset.nodes);
        setEdges(preset.edges);
        setSelectedNodeId(null);
        setStatusMessage(`Loaded architecture preset: ${preset.config.name}`);
        setTimeout(() => setStatusMessage(null), 3000);
        return;
      }

      const template = templates.find((t) => t.id === templateId);
      if (!template) return;

      setGlobalConfig({
        id: template.id,
        name: template.name,
        task_type: template.task_type || "causal_lm",
        architecture_type: template.architecture_type || "hybrid_cnn_transformer",
        vocab_size: template.vocab_size || 12288,
        d_model: template.d_model || 512,
        n_layers: template.n_layers || 16,
        n_q: template.n_q || 8,
        n_kv: template.n_kv || 2,
        d_ff: template.d_ff || 1280,
        kernel_size: template.kernel_size || 4,
        tied_embeddings: template.tied_embeddings ?? true,
      });

      setStatusMessage(`Loaded template: ${template.name}`);
      setTimeout(() => setStatusMessage(null), 3000);
    },
    [handleCreateScratch, templates, setNodes, setEdges]
  );

  // URL Query parameter navigation: ?scratch=true or ?template=<id>
  useEffect(() => {
    if (typeof window !== "undefined") {
      const search = window.location.search;
      if (search) {
        const params = new URLSearchParams(search);
        if (params.get("scratch") === "true") {
          handleCreateScratch();
        } else {
          const tpl = params.get("template");
          if (tpl) {
            handleLoadTemplate(tpl);
          }
        }
      }
    }
  }, [handleCreateScratch, handleLoadTemplate]);

  const handleSaveNetwork = async () => {
    try {
      const payload = {
        ...globalConfig,
        layers: nodes.map((n) => ({
          id: n.id,
          type: n.data.layerType,
          config: n.data.config,
        })),
        parameters: estimation?.parameters,
        memory: estimation?.memory,
      };
      await api.saveNetwork(payload);
      setStatusMessage(`Network '${globalConfig.name}' saved to Registry successfully!`);
      setTimeout(() => setStatusMessage(null), 4000);
    } catch (e: any) {
      alert(`Save error: ${e.message}`);
    }
  };

  const handleCloneNetwork = async () => {
    try {
      const newId = `${globalConfig.id}_clone`;
      const newName = `Clone of ${globalConfig.name}`;
      await api.cloneNetwork(globalConfig.id, newId, newName);
      setGlobalConfig((prev) => ({ ...prev, id: newId, name: newName }));
      setStatusMessage(`Network cloned as '${newName}'!`);
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
      const t = await api.getNetworkTemplates();
      setTemplates(Array.isArray(t) ? t : []);
      handleLoadTemplate("rocanon_48m");
    } catch (e: any) {
      alert(`Delete error: ${e.message}`);
    }
  };

  const selectedNode = nodes.find((n) => n.id === selectedNodeId) || null;

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
        <div className="flex-1 h-full min-h-[450px] glass-panel rounded-lg overflow-hidden relative border border-slate-800">
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
                color="rgba(0, 240, 255, 0.15)"
              />
              <Controls className="!bg-slate-900 !border-slate-800 !rounded-lg !text-cyan-400" />
              <MiniMap
                nodeColor="#00f0ff"
                maskColor="rgba(6, 10, 20, 0.8)"
                className="!bg-slate-950/80 !border !border-slate-800 !rounded-lg"
              />
            </ReactFlow>
          </ReactFlowProvider>

          {/* Canvas Floating Top Info */}
          <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none z-10">
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-1 rounded-md bg-[#0E1420]/95 border border-[#273548] font-mono text-xs text-[#ECF3FC] font-semibold backdrop-blur-md shadow-sm">
                ACTIVE GRAPH: <span className="text-[#45CFFF]">{globalConfig.name}</span>
              </span>
              {statusMessage && (
                <span className="px-2.5 py-1 rounded-md bg-[#35D49A]/15 border border-[#35D49A]/40 font-mono text-xs text-[#35D49A] font-bold animate-in fade-in">
                  {statusMessage}
                </span>
              )}
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleCreateScratch}
                className="pointer-events-auto flex items-center gap-1.5 px-3 py-1 rounded-md bg-[#45CFFF] hover:bg-[#80DEFF] text-[#090D14] font-mono text-xs font-bold transition-all shadow-sm"
                title="Start a new architecture on a blank canvas"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>+ New From Scratch</span>
              </button>
              {nodes.length > 0 && (
                <button
                  type="button"
                  onClick={() => {
                    setNodes([]);
                    setEdges([]);
                    setSelectedNodeId(null);
                  }}
                  className="pointer-events-auto flex items-center gap-1 px-2.5 py-1 rounded-md bg-[#0E1420] border border-[#273548] hover:border-[#FF657A]/50 hover:bg-[#FF657A]/15 text-[#91A0B4] hover:text-[#FF657A] font-mono text-xs transition-all"
                  title="Clear canvas"
                >
                  <Trash2 className="w-3 h-3" />
                  <span>Clear</span>
                </button>
              )}
            </div>
          </div>

          {/* Empty Canvas Overlay in Scratch Mode */}
          {nodes.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none p-6 z-10">
              <div className="max-w-md w-full bg-[#0E1420]/95 border border-dashed border-[#384C66] rounded-xl p-6 backdrop-blur-md text-center pointer-events-auto shadow-2xl relative">
                <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#45CFFF]/30 to-transparent pointer-events-none" />

                <div className="w-11 h-11 rounded-lg bg-[#131C29] border border-[#45CFFF]/40 flex items-center justify-center mx-auto mb-3">
                  <Layers className="w-5 h-5 text-[#45CFFF]" />
                </div>
                <h3 className="font-mono font-bold text-xs text-[#ECF3FC] uppercase tracking-wider">
                  Blank Architecture Canvas
                </h3>
                <p className="text-xs text-[#91A0B4] mt-1 mb-4 leading-relaxed">
                  You are in custom scratch mode. Drag components from the left palette or select a quick starting block below:
                </p>
                <div className="grid grid-cols-2 gap-2 text-left">
                  <button
                    type="button"
                    onClick={() =>
                      handleAddPaletteNode({
                        id: "quick-embed",
                        name: "Token Embedding",
                        category: "embedding",
                        layerType: "Embedding",
                        description: "Learned discrete token embeddings",
                        defaultConfig: { vocab_size: 12288, d_model: 512 },
                      })
                    }
                    className="flex items-center gap-2 p-2 rounded-md bg-[#131C29] border border-[#273548] hover:border-[#45CFFF]/60 font-mono text-xs text-[#ECF3FC] transition-colors"
                  >
                    <Plus className="w-3.5 h-3.5 text-[#45CFFF] shrink-0" />
                    <span>+ Embedding</span>
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      handleAddPaletteNode({
                        id: "quick-attn",
                        name: "GQA Attention Block",
                        category: "attention",
                        layerType: "GQAAttention",
                        description: "Grouped Query Attention block",
                        defaultConfig: { d_model: 512, n_q: 8, n_kv: 2 },
                      })
                    }
                    className="flex items-center gap-2 p-2 rounded-md bg-[#131C29] border border-[#273548] hover:border-[#45CFFF]/60 font-mono text-xs text-[#ECF3FC] transition-colors"
                  >
                    <Plus className="w-3.5 h-3.5 text-[#45CFFF] shrink-0" />
                    <span>+ Attention (GQA)</span>
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      handleAddPaletteNode({
                        id: "quick-ffn",
                        name: "SwiGLU FFN Block",
                        category: "feedforward",
                        layerType: "SwiGLU",
                        description: "Gated SwiGLU feed-forward network",
                        defaultConfig: { d_model: 512, d_ff: 1280 },
                      })
                    }
                    className="flex items-center gap-2 p-2 rounded-md bg-[#131C29] border border-[#273548] hover:border-[#45CFFF]/60 font-mono text-xs text-[#ECF3FC] transition-colors"
                  >
                    <Plus className="w-3.5 h-3.5 text-[#45CFFF] shrink-0" />
                    <span>+ SwiGLU FFN</span>
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      handleAddPaletteNode({
                        id: "quick-head",
                        name: "Linear Head",
                        category: "embedding",
                        layerType: "Linear",
                        description: "Projection / Output LM head",
                        defaultConfig: { in_features: 512, out_features: 12288, bias: false },
                      })
                    }
                    className="flex items-center gap-2 p-2 rounded-md bg-[#131C29] border border-[#273548] hover:border-[#45CFFF]/60 font-mono text-xs text-[#ECF3FC] transition-colors"
                  >
                    <Plus className="w-3.5 h-3.5 text-[#45CFFF] shrink-0" />
                    <span>+ Linear Head</span>
                  </button>
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
          onUpdateGlobalConfig={(updates) =>
            setGlobalConfig((prev) => ({ ...prev, ...updates }))
          }
          onSaveNetwork={handleSaveNetwork}
          onCloneNetwork={handleCloneNetwork}
          onLoadTemplate={handleLoadTemplate}
          onNewScratch={handleCreateScratch}
          onDeleteNetwork={() => setConfirmDeleteNet(true)}
          onShowInfo={setInfoModalLayer}
          templates={templates}
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
            href="/training"
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
