"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  Cpu,
  Layers,
  CheckCircle2,
  ArrowRight,
  ArrowLeft,
  X,
  HardDrive,
  Sparkles,
} from "lucide-react";
import { api } from "@/services/api";

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
      api.getHardwareTelemetry()
        .then((data) => setHardware(data))
        .catch(() => {});
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleFinish = async () => {
    setLoading(true);
    try {
      if (workspaceName.trim() && workspaceName !== "Default Workspace") {
        await api.createWorkspace(workspaceName);
      }

      localStorage.setItem("llm_maker_onboarding_completed", "true");
    } catch {}
    setLoading(false);
    onClose();

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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="w-full max-w-xl bg-[#202020] border border-[#333333] rounded-lg shadow-xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="px-5 py-3 border-b border-[#262626] flex items-center justify-between bg-[#1B1B1B]">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded bg-[#262626] border border-[#333333] flex items-center justify-center text-[#F2F2F2] text-xs font-mono font-medium">
              W
            </div>
            <div>
              <h2 className="text-xs font-mono font-semibold text-[#F2F2F2] uppercase tracking-wide">
                Onboarding Wizard
              </h2>
              <p className="text-[10px] font-mono text-[#737373]">Step {step} of 5</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded text-[#737373] hover:text-[#F2F2F2] hover:bg-[#262626] transition-colors"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Body Content */}
        <div className="p-5 overflow-y-auto max-h-[60vh]">
          {/* STEP 1: Welcome */}
          {step === 1 && (
            <div className="space-y-3">
              <div className="p-3 rounded bg-[#1B1B1B] border border-[#262626] flex gap-2.5">
                <Sparkles className="w-4 h-4 text-[#A3A3A3] shrink-0 mt-0.5" />
                <div>
                  <h3 className="text-xs font-mono font-semibold text-[#F2F2F2]">
                    Welcome to LLM Maker
                  </h3>
                  <p className="text-xs text-[#A3A3A3] mt-1 leading-relaxed">
                    A local-first neural network research and training workstation. Design, train,
                    evaluate, and benchmark models on your own hardware without cloud dependencies.
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2.5 pt-1">
                <div className="p-2.5 rounded bg-[#1B1B1B] border border-[#262626]">
                  <div className="flex items-center gap-1.5 text-[#F2F2F2] font-mono text-xs font-medium mb-1">
                    <Layers className="w-3.5 h-3.5 text-[#737373]" />
                    <span>Visual Network Studio</span>
                  </div>
                  <p className="text-[11px] text-[#737373]">
                    Design Decoder Transformers, MLPs, CNNs with live parameter estimation.
                  </p>
                </div>
                <div className="p-2.5 rounded bg-[#1B1B1B] border border-[#262626]">
                  <div className="flex items-center gap-1.5 text-[#F2F2F2] font-mono text-xs font-medium mb-1">
                    <Cpu className="w-3.5 h-3.5 text-[#737373]" />
                    <span>Hardware Telemetry</span>
                  </div>
                  <p className="text-[11px] text-[#737373]">
                    VRAM safety thresholds, zero-OOM protection, and task loss adapters.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* STEP 2: Hardware Detection */}
          {step === 2 && (
            <div className="space-y-3">
              <div>
                <h3 className="text-xs font-mono font-semibold text-[#F2F2F2]">
                  Detected Hardware Environment
                </h3>
                <p className="text-[11px] text-[#737373]">
                  Physical device metrics queried via NVML & PyTorch runtime.
                </p>
              </div>

              <div className="p-3 rounded bg-[#1B1B1B] border border-[#262626] space-y-2 font-mono text-xs">
                <div className="flex justify-between items-center border-b border-[#262626] pb-1.5">
                  <span className="text-[#737373]">PRIMARY DEVICE:</span>
                  <span className="text-[#F2F2F2] font-medium">{deviceName}</span>
                </div>
                <div className="flex justify-between items-center border-b border-[#262626] pb-1.5">
                  <span className="text-[#737373]">EXECUTION BACKEND:</span>
                  <span className="text-[#F2F2F2]">{backend}</span>
                </div>
                <div className="flex justify-between items-center border-b border-[#262626] pb-1.5">
                  <span className="text-[#737373]">ACCELERATOR VRAM:</span>
                  <span className="text-[#4ade80] font-medium">{vramGb} GB</span>
                </div>
                <div className="flex justify-between items-center border-b border-[#262626] pb-1.5">
                  <span className="text-[#737373]">SYSTEM RAM:</span>
                  <span className="text-[#A3A3A3]">{ramGb} GB</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[#737373]">COMPUTE CAPABILITY:</span>
                  <span className="text-[#A3A3A3]">{computeCap}</span>
                </div>
              </div>

              <div className="flex items-center gap-2 p-2 rounded bg-[rgba(74,222,128,0.06)] border border-[rgba(74,222,128,0.2)] text-[#4ade80] text-xs font-mono">
                <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
                <span>Hardware profile verified and compatible.</span>
              </div>
            </div>
          )}

          {/* STEP 3: Resource Policy Profile */}
          {step === 3 && (
            <div className="space-y-3">
              <div>
                <h3 className="text-xs font-mono font-semibold text-[#F2F2F2]">
                  Select Memory Safety Profile
                </h3>
                <p className="text-[11px] text-[#737373]">
                  Governs memory ceilings and batch size suggestions.
                </p>
              </div>

              <div className="space-y-2">
                {[
                  {
                    id: "conservative",
                    name: "Conservative (65% Limit)",
                    desc: "Leaves ample memory for OS, browser, and background tasks. Minimal risk of OOM.",
                  },
                  {
                    id: "balanced",
                    name: "Balanced (80% Limit) - Recommended",
                    desc: "Optimal balance of training throughput and stability. Standard safe default.",
                  },
                  {
                    id: "performance",
                    name: "Performance (90% Limit)",
                    desc: "Maximizes batch sizes and parameter allocation for dedicated GPU workloads.",
                  },
                ].map((p) => (
                  <label
                    key={p.id}
                    onClick={() => setSelectedProfile(p.id)}
                    className={`flex items-start gap-2.5 p-2.5 rounded border cursor-pointer transition-colors ${
                      selectedProfile === p.id
                        ? "bg-[#262626] border-[#D4D4D4] text-[#F2F2F2]"
                        : "bg-[#1B1B1B] border-[#262626] text-[#737373] hover:border-[#333333]"
                    }`}
                  >
                    <input
                      type="radio"
                      name="profile"
                      checked={selectedProfile === p.id}
                      onChange={() => setSelectedProfile(p.id)}
                      className="mt-0.5 text-[#D4D4D4]"
                    />
                    <div>
                      <div className="text-xs font-mono font-medium text-[#F2F2F2]">{p.name}</div>
                      <div className="text-[10px] text-[#737373] mt-0.5">{p.desc}</div>
                    </div>
                  </label>
                ))}
              </div>
            </div>
          )}

          {/* STEP 4: Workspace Setup */}
          {step === 4 && (
            <div className="space-y-3">
              <div>
                <h3 className="text-xs font-mono font-semibold text-[#F2F2F2]">
                  Workspace Configuration
                </h3>
                <p className="text-[11px] text-[#737373]">
                  Workspaces isolate checkpoints, datasets, models, and runs.
                </p>
              </div>

              <div className="p-3 rounded bg-[#1B1B1B] border border-[#262626] space-y-2 font-mono text-xs">
                <div>
                  <label className="text-[#737373] text-[10px] block mb-1">WORKSPACE NAME</label>
                  <input
                    type="text"
                    value={workspaceName}
                    onChange={(e) => setWorkspaceName(e.target.value)}
                    className="input"
                    placeholder="e.g., Default Workspace or Research Project"
                  />
                </div>
                <div className="text-[10px] text-[#525252] flex items-center gap-1 pt-0.5">
                  <HardDrive className="w-3 h-3 text-[#737373]" />
                  <span>Datasets & Checkpoints will be stored in isolated directories.</span>
                </div>
              </div>
            </div>
          )}

          {/* STEP 5: First Experiment Guided Choice */}
          {step === 5 && (
            <div className="space-y-3">
              <div>
                <h3 className="text-xs font-mono font-semibold text-[#F2F2F2]">
                  Choose Starting Architecture
                </h3>
                <p className="text-[11px] text-[#737373]">
                  Select a template or design from scratch in Network Studio.
                </p>
              </div>

              <div className="grid grid-cols-2 gap-2">
                {[
                  {
                    id: "mini_gpt_15m",
                    name: "Llama / GPT Transformer",
                    family: "Decoder-Only Causal LM",
                    desc: "Autoregressive generation with causal attention, RoPE, and SwiGLU.",
                  },
                  {
                    id: "bert_encoder_25m",
                    name: "BERT / RoBERTa Encoder",
                    family: "Encoder-Only Transformer",
                    desc: "Bidirectional self-attention for embeddings and text classification.",
                  },
                  {
                    id: "conv1d_network_4m",
                    name: "ResNet / 1D-2D CNN",
                    family: "Convolutional Neural Network",
                    desc: "Kernel feature extraction for computer vision, audio, or sequences.",
                  },
                  {
                    id: "custom_scratch",
                    name: "Custom (Blank Canvas)",
                    family: "Freeform Design",
                    desc: "Empty canvas to freely configure layers, shapes, and connections.",
                  },
                ].map((tpl) => (
                  <button
                    key={tpl.id}
                    onClick={() => setSelectedTemplate(tpl.id)}
                    className={`p-2.5 rounded border text-left transition-colors font-mono relative flex flex-col justify-between ${
                      selectedTemplate === tpl.id
                        ? "bg-[#262626] border-[#D4D4D4] text-[#F2F2F2]"
                        : "bg-[#1B1B1B] border-[#262626] text-[#737373] hover:border-[#333333]"
                    }`}
                  >
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-[9px] px-1 py-0.2 rounded bg-[#202020] border border-[#333333] text-[#A3A3A3]">
                          {tpl.family}
                        </span>
                        {selectedTemplate === tpl.id && (
                          <CheckCircle2 className="w-3 h-3 text-[#F2F2F2]" />
                        )}
                      </div>
                      <div className="text-xs font-medium text-[#F2F2F2]">{tpl.name}</div>
                    </div>
                    <div className="text-[10px] text-[#737373] mt-1.5 font-sans leading-relaxed">
                      {tpl.desc}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer Navigation */}
        <div className="px-5 py-2.5 border-t border-[#262626] flex items-center justify-between bg-[#1B1B1B]">
          {step > 1 ? (
            <button
              onClick={() => setStep(step - 1)}
              className="flex items-center gap-1 px-2.5 py-1 rounded border border-[#333333] text-[#737373] hover:text-[#F2F2F2] hover:bg-[#262626] font-mono text-xs transition-colors"
            >
              <ArrowLeft className="w-3 h-3" />
              <span>Back</span>
            </button>
          ) : (
            <div />
          )}

          {step < 5 ? (
            <button
              onClick={() => setStep(step + 1)}
              className="btn btn-primary btn-sm flex items-center gap-1"
            >
              <span>Next</span>
              <ArrowRight className="w-3 h-3" />
            </button>
          ) : (
            <button
              disabled={loading}
              onClick={handleFinish}
              className="btn btn-primary btn-sm flex items-center gap-1"
            >
              <span>{loading ? "Launching..." : "Complete Setup"}</span>
              <CheckCircle2 className="w-3 h-3" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
