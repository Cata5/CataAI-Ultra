"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  Zap,
  Cpu,
  Layers,
  CheckCircle2,
  ArrowRight,
  ArrowLeft,
  X,
  HardDrive,
  Sparkles,
} from "lucide-react";

interface SetupWizardModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const SetupWizardModal: React.FC<SetupWizardModalProps> = ({ isOpen, onClose }) => {
  const router = useRouter();
  const [step, setStep] = useState(1);
  const [hardware, setHardware] = useState<any>(null);
  const [selectedProfile, setSelectedProfile] = useState<string>("balanced");
  const [workspaceName, setWorkspaceName] = useState<string>("My First Workspace");
  const [selectedTemplate, setSelectedTemplate] = useState<string>("mini_gpt_15m");
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

    // Redirect to Network Studio (Builder page)
    if (selectedTemplate === "custom_scratch") {
      router.push("/studio?scratch=true");
    } else {
      router.push(`/studio?template=${selectedTemplate}`);
    }
  };

  const deviceName = hardware?.device_name || "Detected Accelerator / Host CPU";
  const backend = (hardware?.backend_type || "cuda").toUpperCase();
  const vramGb = hardware?.total_vram_mb ? (hardware.total_vram_mb / 1024).toFixed(1) : "N/A";
  const ramGb = hardware?.system_ram_total_mb ? (hardware.system_ram_total_mb / 1024).toFixed(1) : "32.0";
  const computeCap = hardware?.compute_capability || "N/A";
  const precisions = hardware?.supported_precisions?.join(", ") || "FP32, FP16, BF16";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4">
      <div className="w-full max-w-2xl bg-[#0E1420] border border-[#384C66] rounded-xl shadow-2xl flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        {/* Header */}
        <div className="px-6 py-4 border-b border-[#273548] flex items-center justify-between bg-[#090D14]/70">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-[#45CFFF]/10 border border-[#45CFFF]/30 flex items-center justify-center">
              <Zap className="w-4 h-4 text-[#45CFFF]" />
            </div>
            <div>
              <h2 className="text-sm font-mono font-bold text-[#ECF3FC] uppercase tracking-wider">
                LLM Maker Onboarding Wizard
              </h2>
              <p className="text-[11px] font-mono text-[#45CFFF]">Step {step} of 5</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded text-[#91A0B4] hover:text-[#ECF3FC] hover:bg-[#182333] transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body Content */}
        <div className="p-6 overflow-y-auto max-h-[65vh]">
          {/* STEP 1: Welcome */}
          {step === 1 && (
            <div className="space-y-4">
              <div className="p-4 rounded-lg bg-[#45CFFF]/10 border border-[#45CFFF]/30 flex gap-3">
                <Sparkles className="w-6 h-6 text-[#45CFFF] shrink-0 mt-0.5" />
                <div>
                  <h3 className="text-sm font-mono font-bold text-[#ECF3FC]">
                    Welcome to LLM Maker
                  </h3>
                  <p className="text-xs text-[#91A0B4] mt-1 leading-relaxed">
                    LLM Maker is a general-purpose, local-first artificial intelligence and neural
                    network experimentation platform. Train, architect, evaluate, and benchmark
                    models on your own hardware without cloud dependencies or telemetry leaks.
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 pt-2">
                <div className="p-3 rounded-lg bg-[#131C29] border border-[#273548]">
                  <div className="flex items-center gap-2 text-[#45CFFF] font-mono text-xs font-bold mb-1">
                    <Layers className="w-3.5 h-3.5" />
                    <span>Visual Network Studio</span>
                  </div>
                  <p className="text-[11px] text-[#91A0B4]">
                    Design Decoder-Only Transformers, MLPs, CNNs, and Hybrid architectures with live
                    VRAM estimation.
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-[#131C29] border border-[#273548]">
                  <div className="flex items-center gap-2 text-[#35D49A] font-mono text-xs font-bold mb-1">
                    <Cpu className="w-3.5 h-3.5" />
                    <span>Hardware-Aware Training</span>
                  </div>
                  <p className="text-[11px] text-[#91A0B4]">
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
                <h3 className="text-sm font-mono font-bold text-[#ECF3FC]">
                  Detected Hardware Environment
                </h3>
                <p className="text-xs text-[#91A0B4] mt-0.5">
                  Real physical device metrics queried via NVML & PyTorch runtime.
                </p>
              </div>

              <div className="p-4 rounded-lg bg-[#131C29] border border-[#273548] space-y-3 font-mono text-xs">
                <div className="flex justify-between items-center border-b border-[#273548]/70 pb-2">
                  <span className="text-[#91A0B4]">PRIMARY DEVICE:</span>
                  <span className="text-[#ECF3FC] font-bold">{deviceName}</span>
                </div>
                <div className="flex justify-between items-center border-b border-[#273548]/70 pb-2">
                  <span className="text-[#91A0B4]">EXECUTION BACKEND:</span>
                  <span className="text-[#45CFFF] font-bold">{backend}</span>
                </div>
                <div className="flex justify-between items-center border-b border-[#273548]/70 pb-2">
                  <span className="text-[#91A0B4]">ACCELERATOR VRAM:</span>
                  <span className="text-[#35D49A] font-bold">{vramGb} GB</span>
                </div>
                <div className="flex justify-between items-center border-b border-[#273548]/70 pb-2">
                  <span className="text-[#91A0B4]">SYSTEM RAM:</span>
                  <span className="text-[#ECF3FC]">{ramGb} GB</span>
                </div>
                <div className="flex justify-between items-center border-b border-[#273548]/70 pb-2">
                  <span className="text-[#91A0B4]">COMPUTE CAPABILITY:</span>
                  <span className="text-[#ECF3FC]">{computeCap}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[#91A0B4]">PRECISION MODES:</span>
                  <span className="text-[#9B8CFF] font-bold">{precisions}</span>
                </div>
              </div>

              <div className="flex items-center gap-2 p-2.5 rounded bg-[#35D49A]/10 border border-[#35D49A]/30 text-[#35D49A] text-xs font-mono">
                <CheckCircle2 className="w-4 h-4 shrink-0" />
                <span>Compatible hardware profile dynamically constructed.</span>
              </div>
            </div>
          )}

          {/* STEP 3: Resource Policy Profile */}
          {step === 3 && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-mono font-bold text-[#ECF3FC]">
                  Select Memory Safety Profile
                </h3>
                <p className="text-xs text-[#91A0B4] mt-0.5">
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
                        ? "bg-[#45CFFF]/10 border-[#45CFFF] text-[#ECF3FC] shadow-[0_0_12px_rgba(69,207,255,0.2)]"
                        : "bg-[#131C29] border-[#273548] text-[#91A0B4] hover:border-[#384C66]"
                    }`}
                  >
                    <input
                      type="radio"
                      name="profile"
                      checked={selectedProfile === p.id}
                      onChange={() => setSelectedProfile(p.id)}
                      className="mt-1 text-[#45CFFF] focus:ring-0"
                    />
                    <div>
                      <div className="text-xs font-mono font-bold text-[#ECF3FC]">{p.name}</div>
                      <div className="text-[11px] text-[#91A0B4] mt-0.5">{p.desc}</div>
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
                <h3 className="text-sm font-mono font-bold text-[#ECF3FC]">
                  Workspace Configuration
                </h3>
                <p className="text-xs text-[#91A0B4] mt-0.5">
                  Workspaces isolate checkpoints, datasets, custom models, and experiments.
                </p>
              </div>

              <div className="p-4 rounded-lg bg-[#131C29] border border-[#273548] space-y-3 font-mono text-xs">
                <div>
                  <label className="text-[#91A0B4] block mb-1">INITIAL WORKSPACE NAME</label>
                  <input
                    type="text"
                    value={workspaceName}
                    onChange={(e) => setWorkspaceName(e.target.value)}
                    className="w-full px-3 py-2 rounded bg-[#0E1420] border border-[#273548] text-[#ECF3FC] focus:border-[#45CFFF] focus:outline-none"
                    placeholder="e.g., Default Workspace or Research Project"
                  />
                </div>
                <div className="text-[11px] text-[#91A0B4] flex items-center gap-1.5 pt-1">
                  <HardDrive className="w-3.5 h-3.5 text-[#45CFFF]" />
                  <span>Datasets & Checkpoints will be stored in isolated directories.</span>
                </div>
              </div>
            </div>
          )}

          {/* STEP 5: First Experiment Guided Choice */}
          {step === 5 && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-mono font-bold text-[#ECF3FC]">
                  Choose Starting Architecture
                </h3>
                <p className="text-xs text-[#91A0B4] mt-0.5">
                  Select one of the 3 most recognized neural network architectures, or design a custom model from scratch. You will be redirected straight to the visual builder page.
                </p>
              </div>

              <div className="grid grid-cols-2 gap-3">
                {[
                  {
                    id: "mini_gpt_15m",
                    name: "Llama / GPT Transformer",
                    family: "Decoder-Only Causal LM",
                    badge: "Most Popular LLM",
                    desc: "Autoregressive generation with causal attention, RoPE, and SwiGLU / MLP blocks. The foundation of modern LLMs.",
                  },
                  {
                    id: "bert_encoder_25m",
                    name: "BERT / RoBERTa Encoder",
                    family: "Encoder-Only Transformer",
                    badge: "Embeddings & NLU",
                    desc: "Bidirectional self-attention network for embeddings, text classification, and masked language understanding.",
                  },
                  {
                    id: "conv1d_network_4m",
                    name: "ResNet / 1D-2D CNN",
                    family: "Convolutional Neural Network",
                    badge: "Classic Vision & Audio",
                    desc: "Feature extraction via convolutional kernels and pooling for computer vision, audio, or sequence modeling.",
                  },
                  {
                    id: "custom_scratch",
                    name: "Custom (Build From Scratch)",
                    family: "Blank Visual Canvas",
                    badge: "Freeform Design",
                    desc: "Start with an empty workspace canvas in Network Studio to freely design your custom layers, shapes, and connections.",
                  },
                ].map((tpl) => (
                  <button
                    key={tpl.id}
                    onClick={() => setSelectedTemplate(tpl.id)}
                    className={`p-3.5 rounded-lg border text-left transition-all font-mono relative flex flex-col justify-between ${
                      selectedTemplate === tpl.id
                        ? "bg-[#45CFFF]/10 border-[#45CFFF] shadow-[0_0_12px_rgba(69,207,255,0.2)] ring-1 ring-[#45CFFF]/50"
                        : "bg-[#131C29] border-[#273548] hover:border-[#384C66]"
                    }`}
                  >
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#45CFFF]/15 border border-[#45CFFF]/30 text-[#45CFFF] font-bold">
                          {tpl.badge}
                        </span>
                        {selectedTemplate === tpl.id && (
                          <CheckCircle2 className="w-3.5 h-3.5 text-[#45CFFF]" />
                        )}
                      </div>
                      <div className="text-xs font-bold text-[#ECF3FC]">{tpl.name}</div>
                      <div className="text-[10px] text-[#45CFFF] mt-0.5">{tpl.family}</div>
                    </div>
                    <div className="text-[11px] text-[#91A0B4] mt-2 font-sans leading-relaxed">
                      {tpl.desc}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer Navigation */}
        <div className="px-6 py-3.5 border-t border-[#273548] flex items-center justify-between bg-[#090D14]/70">
          {step > 1 ? (
            <button
              onClick={() => setStep(step - 1)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-[#273548] text-[#91A0B4] hover:bg-[#182333] hover:text-[#ECF3FC] font-mono text-xs transition-colors"
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
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-md bg-[#45CFFF] hover:bg-[#80DEFF] text-[#090D14] font-mono text-xs font-bold transition-all shadow-[0_0_12px_rgba(69,207,255,0.3)]"
            >
              <span>Next Step</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          ) : (
            <button
              disabled={loading}
              onClick={handleFinish}
              className="flex items-center gap-1.5 px-5 py-1.5 rounded-md bg-[#35D49A] hover:bg-[#35D49A]/90 text-[#090D14] font-mono text-xs font-bold transition-all shadow-[0_0_15px_rgba(53,212,154,0.3)] disabled:opacity-50"
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
