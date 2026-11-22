import {
  Alert,
  BenchmarkResult,
  CheckpointItem,
  DatasetItem,
  HardwareTelemetry,
  MetricPoint,
  NetworkConfig,
  PipelineItem,
  RunItem,
  SystemStatus,
} from "@/types/api";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "";

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${BASE_URL}${endpoint}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers || {}),
    },
  });

  if (!response.ok) {
    let errorMsg = `API Error: ${response.status} ${response.statusText}`;
    try {
      const errJson = await response.json();
      if (errJson && errJson.error) {
        errorMsg = errJson.error;
      }
    } catch {
      // ignore
    }
    throw new Error(errorMsg);
  }

  return response.json();
}

export const api = {
  // System & Telemetry
  getSystemStatus: () => request<SystemStatus>("/api/system/status"),
  getHardwareTelemetry: () => request<HardwareTelemetry>("/api/hardware"),
  getAlerts: (runId?: string, limit: number = 50) => {
    const params = new URLSearchParams();
    if (runId) params.append("run_id", runId);
    params.append("limit", limit.toString());
    return request<Alert[]>(`/api/alerts?${params.toString()}`);
  },

  // Runs & Controls
  listRuns: (limit: number = 50) => request<RunItem[]>(`/api/runs?limit=${limit}`),
  startRun: (payload: Record<string, any>) =>
    request<{ status: string; payload: any }>("/api/runs", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  pauseRun: (runId: string = "current") =>
    request<{ success: boolean }>(`/api/runs/${runId}/pause`, { method: "POST" }),
  resumeRun: (runId: string = "current") =>
    request<{ success: boolean }>(`/api/runs/${runId}/resume`, { method: "POST" }),
  stopRun: (runId: string = "current") =>
    request<{ success: boolean }>(`/api/runs/${runId}/stop`, { method: "POST" }),
  checkpointRun: (runId: string = "current") =>
    request<{ success: boolean }>(`/api/runs/${runId}/checkpoint`, { method: "POST" }),
  validateRun: (runId: string = "current") =>
    request<{ success: boolean }>(`/api/runs/${runId}/validate`, { method: "POST" }),
  updateLr: (data: { multiplier?: number; peak_lr?: number; adam_lr?: number; muon_lr?: number }) =>
    request<{ success: boolean; updated: any }>("/api/runs/lr", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  getRunMetrics: (runId: string, startStep: number = 0, limit: number = 2000) =>
    request<MetricPoint[]>(`/api/runs/${runId}/metrics?start_step=${startStep}&limit=${limit}`),

  // Checkpoints & Local Model Packages
  listCheckpoints: () => request<CheckpointItem[]>("/api/checkpoints"),
  deleteCheckpoint: (id: string) =>
    request<{ success: boolean; id: string }>(`/api/checkpoints/${encodeURIComponent(id)}`, {
      method: "DELETE",
    }),
  saveLocalModel: (data: { model_id?: string; custom_name?: string; save_mode?: "best" | "current" }) =>
    request<{ success: boolean; package: any }>("/api/models/save-local", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  // Pipelines
  listPipelines: () => request<PipelineItem[]>("/api/pipelines"),
  createPipeline: (name: string, phases: any[]) =>
    request<{ pipeline_id: string }>("/api/pipelines", {
      method: "POST",
      body: JSON.stringify({ name, phases }),
    }),
  getPipeline: (id: string) => request<PipelineItem>(`/api/pipelines/${id}`),
  startPipeline: (id: string, phaseIndex?: number) =>
    request<any>(`/api/pipelines/${id}/start`, {
      method: "POST",
      body: JSON.stringify({ phase_index: phaseIndex }),
    }),
  advancePipeline: (id: string, completedStep: number, checkpointPath: string) =>
    request<any>(`/api/pipelines/${id}/advance`, {
      method: "POST",
      body: JSON.stringify({ completed_step: completedStep, checkpoint_path: checkpointPath }),
    }),

  // Schedulers
  getSchedulerCurve: (params: {
    schedule_type?: string;
    total_steps?: number;
    peak_lr?: number;
    min_lr?: number;
    warmup_steps?: number;
    stable_steps?: number;
    decay_style?: string;
  }) =>
    request<any>("/api/scheduler/curve", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  // Datasets
  listDatasets: () => request<DatasetItem[]>("/api/datasets"),
  getDatasetStats: (id: string) => request<any>(`/api/datasets/${encodeURIComponent(id)}/stats`),
  uploadDataset: async (file: File): Promise<any> => {
    const formData = new FormData();
    formData.append("file", file);
    // Direct connection to Python backend port 8081 to support large uploads (>10MB, up to 500MB) without proxy timeouts
    const backendDirect = typeof window !== "undefined"
      ? `${window.location.protocol}//${window.location.hostname}:8081/api/datasets/upload`
      : "http://127.0.0.1:8081/api/datasets/upload";
    const response = await fetch(backendDirect, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error || `Upload failed: ${response.statusText}`);
    }
    return response.json();
  },
  tokenizeDataset: (id: string) =>
    request<{
      success: boolean;
      dataset_id: string;
      cached: boolean;
      token_count: number;
      train_tokens: number;
      val_tokens: number;
    }>(`/api/datasets/${encodeURIComponent(id)}/tokenize`, {
      method: "POST",
    }),
  deleteDataset: (id: string) =>
    request<{ success: boolean; id: string; deleted_files?: string[] }>(
      `/api/datasets/${encodeURIComponent(id)}`,
      { method: "DELETE" }
    ),
  inspectTokenizer: (text: string) =>
    request<any>("/api/tokenizer/inspect", {
      method: "POST",
      body: JSON.stringify({ text }),
    }),

  // Evaluation
  generateEvaluation: (params: {
    prompt: string;
    max_new_tokens?: number;
    temperature?: number;
    top_p?: number;
    top_k?: number;
    repetition_penalty?: number;
    no_repeat_ngram_size?: number;
    min_p?: number;
    presence_penalty?: number;
    frequency_penalty?: number;
    seed?: number;
    stop_sequences?: string[];
    checkpoint?: string;
    chat_mode?: boolean;
  }) =>
    request<{
      generated_text: string;
      completion?: string;
      tokens_generated: number;
      generation_time_sec: number;
      tokens_per_second: number;
      device: string;
      gpu_arbitrated: boolean;
      checkpoint_loaded?: string | null;
      checkpoint_step?: number;
      checkpoint_val_loss?: number;
    }>("/api/evaluation/generate", {
      method: "POST",
      body: JSON.stringify(params),
    }),
  compareEvaluation: (params: {
    prompt: string;
    checkpoint_paths: string[];
    max_new_tokens?: number;
    temperature?: number;
    top_p?: number;
    top_k?: number;
    repetition_penalty?: number;
    no_repeat_ngram_size?: number;
    min_p?: number;
    presence_penalty?: number;
    frequency_penalty?: number;
    seed?: number;
    stop_sequences?: string[];
    chat_mode?: boolean;
  }) =>
    request<{ prompt: string; results: any[] }>("/api/evaluation/compare", {
      method: "POST",
      body: JSON.stringify(params),
    }),
  runBenchmarkSuite: () => request<any>("/api/evaluation/benchmark", { method: "POST" }),
  listBenchmarkResults: (limit: number = 100) =>
    request<BenchmarkResult[]>(`/api/evaluation/results?limit=${limit}`),

  // Network Studio
  listNetworks: () => request<NetworkConfig[]>("/api/networks"),
  getNetworkTemplates: () => request<NetworkConfig[]>("/api/networks/templates"),
  getCustomModules: () => request<any[]>("/api/networks/custom_modules"),
  getNetwork: (id: string) => request<NetworkConfig>(`/api/networks/${encodeURIComponent(id)}`),
  saveNetwork: (network: Partial<NetworkConfig>) =>
    request<NetworkConfig>("/api/networks", {
      method: "POST",
      body: JSON.stringify(network),
    }),
  cloneNetwork: (origId: string, newId: string, newName: string) =>
    request<NetworkConfig>(`/api/networks/${encodeURIComponent(origId)}/clone`, {
      method: "POST",
      body: JSON.stringify({ new_id: newId, new_name: newName }),
    }),
  deleteNetwork: (id: string) =>
    request<{ success: boolean; id: string }>(`/api/networks/${encodeURIComponent(id)}`, {
      method: "DELETE",
    }),
  validateNetwork: (config: any) =>
    request<{
      valid: boolean;
      issues: string[];
      warnings: string[];
      details?: Record<string, any>;
    }>("/api/networks/validate", {
      method: "POST",
      body: JSON.stringify(config),
    }),
  estimateNetwork: (params: {
    config: any;
    micro_batch_size?: number;
    seq_len?: number;
    dtype?: string;
    optimizer?: string;
    gradient_checkpointing?: boolean;
  }) =>
    request<{
      parameters: {
        total_params: number;
        trainable_params: number;
        non_trainable_params: number;
        breakdown?: Record<string, number>;
      };
      memory: {
        model_weights_mb: number;
        optimizer_states_mb: number;
        gradients_mb: number;
        activations_mb: number;
        total_training_vram_mb: number;
        fits_6gb_gpu: boolean;
      };
    }>("/api/networks/estimate", {
      method: "POST",
      body: JSON.stringify(params),
    }),
  recommendBudget: (targetParams: number) =>
    request<any>("/api/networks/budget_recommend", {
      method: "POST",
      body: JSON.stringify({ target_params: targetParams }),
    }),

  // Settings
  getSettings: () => request<Record<string, any>>("/api/settings"),
  saveSettings: (settings: Record<string, any>) =>
    request<{ status: string; settings: any }>("/api/settings", {
      method: "POST",
      body: JSON.stringify(settings),
    }),
};
