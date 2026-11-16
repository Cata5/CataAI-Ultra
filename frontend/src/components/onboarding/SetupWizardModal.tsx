"use client";

import React, { useState, useEffect } from "react";
import {
  Zap,
  Cpu,
  Shield,
  Layers,
  CheckCircle2,
  ArrowRight,
  ArrowLeft,
  X,
  HardDrive,
  Sparkles,
  Info,
} from "lucide-react";

interface SetupWizardModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const SetupWizardModal: React.FC<SetupWizardModalProps> = ({ isOpen, onClose }) => {
  const [step, setStep] = useState(1);
  const [hardware, setHardware] = useState<any>(null);
  const [selectedProfile, setSelectedProfile] = useState<string>("balanced");
  const [workspaceName, setWorkspaceName] = useState<string>("My First Workspace");
  const [selectedTemplate, setSelectedTemplate] = useState<string>("rocanon_48m");
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    if (isOpen) {
      fetch("/api/hardware")
        .then((res) => res.json())
        .then((data) => setHardware(data))
        .catch(() => {});
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleFinish = async () => {
    setLoading(true);
    try {
      // 1. Set Hardware Profile
      await fetch("/api/hardware/profile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: selectedProfile }),
      });

      // 2. Create or activate workspace if not default
      if (workspaceName.trim() && workspaceName !== "Default Workspace") {
        await fetch("/api/workspaces", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: workspaceName }),
        });
      }

      localStorage.setItem("llm_maker_onboarding_completed", "true");
    } catch {}
    setLoading(false);
    onClose();
  };

  const deviceName = hardware?.device_name || "Detected Accelerator / Host CPU";
  const backend = (hardware?.backend_type || "cuda").toUpperCase();
  const vramGb = hardware?.total_vram_mb ? (hardware.total_vram_mb / 1024).toFixed(1) : "N/A";
  const ramGb = hardware?.system_ram_total_mb ? (hardware.system_ram_total_mb / 1024).toFixed(1) : "32.0";
  const computeCap = hardware?.compute_capability || "N/A";
  const precisions = hardware?.supported_precisions?.join(", ") || "FP32, FP16, BF16";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4">
      <div className="w-full max-w-2xl bg-[#090e1a] border border-cyan-500/40 rounded-xl shadow-[0_0_30px_rgba(0,240,255,0.2)] flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/50">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-cyan-950/80 border border-cyan-500/50 flex items-center justify-center">
              <Zap className="w-4 h-4 text-cyan-400" />
            </div>
            <div>
              <h2 className="text-sm font-mono font-bold text-slate-100 uppercase tracking-wider">
                LLM Maker Onboarding Wizard
              </h2>
              <p className="text-[11px] font-mono text-cyan-400/80">Step {step} of 5</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body Content */}
        <div className="p-6 overflow-y-auto max-h-[65vh]">
          {/* STEP 1: Welcome */}
          {step === 1 && (
            <div className="space-y-4">
              <div className="p-4 rounded-lg bg-cyan-950/30 border border-cyan-500/30 flex gap-3">
                <Sparkles className="w-6 h-6 text-cyan-400 shrink-0 mt-0.5" />
                <div>
                  <h3 className="text-sm font-mono font-bold text-slate-100">
                    Welcome to LLM Maker
                  </h3>
                  <p className="text-xs text-slate-300 mt-1 leading-relaxed">
                    LLM Maker is a general-purpose, local-first artificial intelligence and neural
                    network experimentation platform. Train, architect, evaluate, and benchmark
                    models on your own hardware without cloud dependencies or telemetry leaks.
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 pt-2">
                <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
                  <div className="flex items-center gap-2 text-cyan-400 font-mono text-xs font-bold mb-1">
                    <Layers className="w-3.5 h-3.5" />
                    <span>Visual Network Studio</span>
                  </div>
                  <p className="text-[11px] text-slate-400">
                    Design Decoder-Only Transformers, MLPs, CNNs, and Hybrid architectures with live
                    VRAM estimation.
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
                  <div className="flex items-center gap-2 text-emerald-400 font-mono text-xs font-bold mb-1">
                    <Cpu className="w-3.5 h-3.5" />
                    <span>Hardware-Aware Training</span>
                  </div>
                  <p className="text-[11px] text-slate-400">
                    Auto-configured safety thresholds, zero-OOM protection, and task-specific loss
                    adapters.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* STEP 2: Hardware Detection */}
          {step === 2 && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-mono font-bold text-slate-100">
                  Detected Hardware Environment
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Real physical device metrics queried via NVML & PyTorch runtime.
                </p>
              </div>

              <div className="p-4 rounded-lg bg-slate-900/80 border border-slate-800 space-y-3 font-mono text-xs">
                <div className="flex justify-between items-center border-b border-slate-800/80 pb-2">
                  <span className="text-slate-400">PRIMARY DEVICE:</span>
                  <span className="text-slate-100 font-bold">{deviceName}</span>
                </div>
                <div className="flex justify-between items-center border-b border-slate-800/80 pb-2">
                  <span className="text-slate-400">EXECUTION BACKEND:</span>
                  <span className="text-cyan-400 font-bold">{backend}</span>
                </div>
                <div className="flex justify-between items-center border-b border-slate-800/80 pb-2">
                  <span className="text-slate-400">ACCELERATOR VRAM:</span>
                  <span className="text-emerald-400 font-bold">{vramGb} GB</span>
                </div>
                <div className="flex justify-between items-center border-b border-slate-800/80 pb-2">
                  <span className="text-slate-400">SYSTEM RAM:</span>
                  <span className="text-slate-200">{ramGb} GB</span>
                </div>
                <div className="flex justify-between items-center border-b border-slate-800/80 pb-2">
                  <span className="text-slate-400">COMPUTE CAPABILITY:</span>
                  <span className="text-slate-200">{computeCap}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">PRECISION MODES:</span>
                  <span className="text-purple-400 font-bold">{precisions}</span>
                </div>
              </div>

              <div className="flex items-center gap-2 p-2.5 rounded bg-emerald-950/20 border border-emerald-500/30 text-emerald-400 text-xs font-mono">
                <CheckCircle2 className="w-4 h-4 shrink-0" />
                <span>Compatible hardware profile dynamically constructed.</span>
              </div>
            </div>
          )}

          {/* STEP 3: Resource Policy Profile */}
          {step === 3 && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-mono font-bold text-slate-100">
                  Select Memory Safety Profile
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Governs memory allocation ceilings, gradient checkpointing defaults, and batch size suggestions.
                </p>
              </div>

              <div className="space-y-2.5">
                {[
                  {
                    id: "conservative",
                    name: "Conservative (65% Limit)",
                    desc: "Leaves ample memory for OS, browser, and background tasks. Minimal risk of Out-Of-Memory.",
                  },
                  {
                    id: "balanced",
                    name: "Balanced (80% Limit) - Recommended",
                    desc: "Optimal balance of training throughput and stability. Standard safe default for local ML.",
                  },
                  {
                    id: "performance",
                    name: "Performance (90% Limit)",
                    desc: "Maximizes batch sizes and parameter allocation. Recommended when no other GPU apps are running.",
                  },
                ].map((p) => (
                  <label
                    key={p.id}
                    onClick={() => setSelectedProfile(p.id)}
                    className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
                      selectedProfile === p.id
                        ? "bg-cyan-950/40 border-cyan-500 text-slate-100 shadow-[0_0_10px_rgba(0,240,255,0.15)]"
                        : "bg-slate-900/50 border-slate-800 text-slate-400 hover:border-slate-700"
                    }`}
                  >
                    <input
                      type="radio"
                      name="profile"
                      checked={selectedProfile === p.id}
                      onChange={() => setSelectedProfile(p.id)}
                      className="mt-1 text-cyan-500 focus:ring-0"
                    />
                    <div>
                      <div className="text-xs font-mono font-bold text-slate-200">{p.name}</div>
                      <div className="text-[11px] text-slate-400 mt-0.5">{p.desc}</div>
                    </div>
                  </label>
                ))}
              </div>
            </div>
          )}

          {/* STEP 4: Workspace Setup */}
          {step === 4 && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-mono font-bold text-slate-100">
                  Workspace Configuration
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Workspaces isolate checkpoints, datasets, custom models, and experiments.
                </p>
              </div>

              <div className="p-4 rounded-lg bg-slate-900/80 border border-slate-800 space-y-3 font-mono text-xs">
                <div>
                  <label className="text-slate-400 block mb-1">INITIAL WORKSPACE NAME</label>
                  <input
                    type="text"
                    value={workspaceName}
                    onChange={(e) => setWorkspaceName(e.target.value)}
                    className="w-full px-3 py-2 rounded bg-slate-950 border border-slate-700 text-slate-100 focus:border-cyan-500 focus:outline-none"
                    placeholder="e.g., Default Workspace or Research Project"
                  />
                </div>
                <div className="text-[11px] text-slate-400 flex items-center gap-1.5 pt-1">
                  <HardDrive className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Datasets & Checkpoints will be stored in isolated directories.</span>
                </div>
              </div>
            </div>
          )}

          {/* STEP 5: First Experiment Guided Choice */}
          {step === 5 && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-mono font-bold text-slate-100">
                  Choose Starting Architecture
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Select a baseline template or begin with a custom visual graph.
                </p>
              </div>

              <div className="grid grid-cols-2 gap-3">
                {[
                  {
                    id: "rocanon_48m",
                    name: "RoCanon-48.5M Baseline",
                    family: "Hybrid Conv1D + GQA",
                    desc: "Registered SLM baseline with tied embeddings and SwiGLU.",
                  },
                  {
                    id: "mini_gpt",
                    name: "Mini-GPT (15M)",
                    family: "Decoder-Only Transformer",
                    desc: "Lightweight causal model for rapid experimentation.",
                  },
                  {
                    id: "scratch",
                    name: "Network Studio (Scratch)",
                    family: "Custom Architecture",
                    desc: "Build your own MLP, CNN, or Transformer from scratch.",
                  },
                  {
                    id: "bert_encoder",
                    name: "Encoder Classifier (10M)",
                    family: "Encoder-Only",
                    desc: "BERT-style sequence classification model.",
                  },
                ].map((tpl) => (
                  <button
                    key={tpl.id}
                    onClick={() => setSelectedTemplate(tpl.id)}
                    className={`p-3 rounded-lg border text-left transition-all font-mono ${
                      selectedTemplate === tpl.id
                        ? "bg-cyan-950/40 border-cyan-500 shadow-[0_0_10px_rgba(0,240,255,0.15)]"
                        : "bg-slate-900/50 border-slate-800 hover:border-slate-700"
                    }`}
                  >
                    <div className="text-xs font-bold text-slate-200">{tpl.name}</div>
                    <div className="text-[10px] text-cyan-400 mt-0.5">{tpl.family}</div>
                    <div className="text-[11px] text-slate-400 mt-1 font-sans">{tpl.desc}</div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer Navigation */}
        <div className="px-6 py-3.5 border-t border-slate-800 flex items-center justify-between bg-slate-900/50">
          {step > 1 ? (
            <button
              onClick={() => setStep(step - 1)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-700 text-slate-300 hover:bg-slate-800 font-mono text-xs transition-colors"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Previous</span>
            </button>
          ) : (
            <div />
          )}

          {step < 5 ? (
            <button
              onClick={() => setStep(step + 1)}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-mono text-xs font-bold transition-all shadow-[0_0_12px_rgba(0,240,255,0.3)]"
            >
              <span>Next Step</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          ) : (
            <button
              disabled={loading}
              onClick={handleFinish}
              className="flex items-center gap-1.5 px-5 py-1.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-mono text-xs font-bold transition-all shadow-[0_0_15px_rgba(16,185,129,0.3)] disabled:opacity-50"
            >
              <span>{loading ? "Launching..." : "Complete Setup & Launch"}</span>
              <CheckCircle2 className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
