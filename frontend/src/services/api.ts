import {
  Alert,
  SystemLogEntry,
  BenchmarkResult,
  CheckpointItem,
  DatasetItem,
  HardwareTelemetry,
  MetricPoint,
  NetworkConfig,
  PipelineItem,
  RunItem,
  RunState,
  SystemStatus,
} from "@/types/api";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "";

export const isTauri = (): boolean =>
  typeof window !== "undefined" && Boolean((window as any).__TAURI_INTERNALS__);

let tauriInvoke: any = null;
async function getTauriInvoke() {
  if (tauriInvoke) return tauriInvoke;
  if (isTauri()) {
    try {
      const module = await import("@tauri-apps/api/core");
      tauriInvoke = module.invoke;
      return tauriInvoke;
    } catch {
      return null;
    }
  }
  return null;
}

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
  getSystemStatus: async (): Promise<SystemStatus> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        const ws: any = await inv("get_worker_status");
        return {
          state: (ws.state || "IDLE") as RunState,
          run_id: ws.run_id || null,
          latest_metrics: ws.current_step > 0 ? {
            step: ws.current_step,
            loss: ws.current_loss || 0,
            learning_rate: 0,
          } : null,
          alerts_count: 0,
          timestamp: Date.now() / 1000,
        };
      } catch (e) {
        console.warn("Tauri get_worker_status fallback:", e);
      }
    }
    return request<SystemStatus>("/api/system/status");
  },
  getHardwareTelemetry: async (): Promise<HardwareTelemetry> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("get_hardware_telemetry");
      } catch (e) {
        console.warn("Tauri get_hardware_telemetry fallback:", e);
      }
    }
    return request<HardwareTelemetry>("/api/hardware");
  },
  getAlerts: async (runId?: string, limit: number = 50): Promise<Alert[]> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("get_alerts", { runId, limit });
      } catch (e) {
        console.warn("Tauri get_alerts fallback:", e);
      }
    }
    return request<Alert[]>(`/api/alerts?limit=${limit}${runId ? `&run_id=${runId}` : ""}`);
  },
  getSystemLogs: async (limit: number = 2000): Promise<SystemLogEntry[]> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("get_system_logs", { limit });
      } catch (e) {
        console.warn("Tauri get_system_logs fallback:", e);
      }
    }
    return request<SystemLogEntry[]>(`/api/logs?limit=${limit}`);
  },
  clearSystemLogs: async (): Promise<void> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        await inv("clear_system_logs");
        return;
      } catch (e) {
        console.warn("Tauri clear_system_logs fallback:", e);
      }
    }
    await request<any>("/api/logs/clear", { method: "POST" });
  },

  // Runs & Controls
  listRuns: async (limit: number = 50): Promise<RunItem[]> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        const raw: any[] = await inv("list_runs");
        return raw.slice(0, limit).map((r) => ({
          id: r.run_id,
          name: r.name,
          status: (r.status || "IDLE") as RunState,
          config: {},
          started_at: new Date(r.created_at * 1000).toISOString(),
          total_steps: r.total_steps || 0,
          total_tokens: 0,
          best_val_loss: null,
          final_loss: r.final_loss,
        }));
      } catch (e) {
        console.warn("Tauri list_runs fallback:", e);
      }
    }
    return request<RunItem[]>(`/api/runs?limit=${limit}`);
  },
  startRun: async (payload: Record<string, any>): Promise<any> => {
    const inv = await getTauriInvoke();
    if (inv) {
      const runId = await inv("start_training", { payloadJson: JSON.stringify(payload) });
      return { status: "STARTING", run_id: runId };
    }
    return request<{ status: string; payload: any }>("/api/runs", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },
  pauseRun: async (runId: string = "current"): Promise<{ success: boolean }> => {
    const inv = await getTauriInvoke();
    if (inv) {
      await inv("pause_run", { runId });
      return { success: true };
    }
    return request<{ success: boolean }>(`/api/runs/${runId}/pause`, { method: "POST" });
  },
  resumeRun: async (runId: string = "current"): Promise<{ success: boolean }> => {
    const inv = await getTauriInvoke();
    if (inv) {
      await inv("resume_run", { runId });
      return { success: true };
    }
    return request<{ success: boolean }>(`/api/runs/${runId}/resume`, { method: "POST" });
  },
  stopRun: async (runId: string = "current"): Promise<{ success: boolean }> => {
    const inv = await getTauriInvoke();
    if (inv) {
      await inv("stop_training");
      return { success: true };
    }
    return request<{ success: boolean }>(`/api/runs/${runId}/stop`, { method: "POST" });
  },
  checkpointRun: async (runId: string = "current"): Promise<{ success: boolean }> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        await inv("checkpoint_run", { runId });
        return { success: true };
      } catch (e) {
        console.warn("Tauri checkpoint_run fallback:", e);
      }
    }
    return request<{ success: boolean }>(`/api/runs/${runId}/checkpoint`, { method: "POST" });
  },
  validateRun: async (runId: string = "current"): Promise<{ success: boolean }> => {
    const inv = await getTauriInvoke();
    if (inv) {
      await inv("validate_run");
      return { success: true };
    }
    const res = await request<{ success: boolean }>(`/api/runs/${runId}/validate`, { method: "POST" });
    if (!res.success) {
      throw new Error("No active training run is available for validation.");
    }
    return res;
  },
  updateLr: async (data: { multiplier?: number; peak_lr?: number; adam_lr?: number; muon_lr?: number }): Promise<any> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        await inv("update_lr", { data });
        return { success: true, updated: data };
      } catch (e) {
        console.warn("Tauri update_lr fallback:", e);
      }
    }
    return request<{ success: boolean; updated: any }>("/api/runs/lr", {
      method: "POST",
      body: JSON.stringify(data),
    });
  },
  getRunMetrics: async (runId: string, startStep: number = 0, limit: number = 2000): Promise<MetricPoint[]> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("get_run_metrics", { runId, startStep, limit });
      } catch (e) {
        console.warn("Tauri get_run_metrics fallback:", e);
      }
    }
    return request<MetricPoint[]>(`/api/runs/${runId}/metrics?start_step=${startStep}&limit=${limit}`);
  },

  // Checkpoints & Local Model Packages
  listCheckpoints: async (): Promise<CheckpointItem[]> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("list_checkpoints");
      } catch (e) {
        console.warn("Tauri list_checkpoints fallback:", e);
      }
    }
    return request<CheckpointItem[]>("/api/checkpoints");
  },
  deleteCheckpoint: async (id: string): Promise<{ success: boolean; id: string }> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        await inv("delete_checkpoint", { id });
        return { success: true, id };
      } catch (e) {
        console.warn("Tauri delete_checkpoint fallback:", e);
      }
    }
    return request<{ success: boolean; id: string }>(`/api/checkpoints/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
  },
  saveLocalModel: async (data: { model_id?: string; custom_name?: string; save_mode?: "best" | "current" }): Promise<any> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("save_local_model", { data });
      } catch (e) {
        console.warn("Tauri save_local_model fallback:", e);
      }
    }
    return request<{ success: boolean; package: any }>("/api/models/save-local", {
      method: "POST",
      body: JSON.stringify(data),
    });
  },

  // Pipelines
  listPipelines: async (): Promise<PipelineItem[]> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("list_pipelines");
      } catch (e) {
        console.warn("Tauri list_pipelines fallback:", e);
      }
    }
    return request<PipelineItem[]>("/api/pipelines");
  },
  createPipeline: async (name: string, phases: any[]): Promise<{ pipeline_id: string }> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("create_pipeline", { name, phases });
      } catch (e) {
        console.warn("Tauri create_pipeline fallback:", e);
      }
    }
    return request<{ pipeline_id: string }>("/api/pipelines", {
      method: "POST",
      body: JSON.stringify({ name, phases }),
    });
  },
  getPipeline: async (id: string): Promise<PipelineItem> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("get_pipeline", { id });
      } catch (e) {
        console.warn("Tauri get_pipeline fallback:", e);
      }
    }
    return request<PipelineItem>(`/api/pipelines/${id}`);
  },
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
  getSchedulerCurve: async (params: {
    schedule_type?: string;
    total_steps?: number;
    peak_lr?: number;
    min_lr?: number;
    warmup_steps?: number;
    stable_steps?: number;
    decay_style?: string;
  }): Promise<any> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("get_scheduler_curve", { params });
      } catch (e) {
        console.warn("Tauri get_scheduler_curve fallback:", e);
      }
    }
    return request<any>("/api/scheduler/curve", {
      method: "POST",
      body: JSON.stringify(params),
    });
  },

  // Datasets
  listDatasets: async (): Promise<DatasetItem[]> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("list_datasets");
      } catch (e) {
        console.warn("Tauri list_datasets fallback:", e);
      }
    }
    return request<DatasetItem[]>("/api/datasets");
  },
  getDatasetStats: async (id: string): Promise<any> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("get_dataset_stats", { id });
      } catch (e) {
        console.warn("Tauri get_dataset_stats fallback:", e);
      }
    }
    return request<any>(`/api/datasets/${encodeURIComponent(id)}/stats`);
  },
  uploadDataset: async (file: File): Promise<any> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        const filePath = (file as any).path || (file as any).webkitRelativePath || null;
        if (filePath && typeof filePath === "string" && filePath.length > 0) {
          return await inv("upload_dataset", {
            filename: file.name,
            filePath,
          });
        }
        // Read as base64 string without huge array conversion
        const base64 = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => {
            const res = reader.result as string;
            const commaIdx = res.indexOf(",");
            resolve(commaIdx !== -1 ? res.substring(commaIdx + 1) : res);
          };
          reader.onerror = reject;
          reader.readAsDataURL(file);
        });
        return await inv("upload_dataset", {
          filename: file.name,
          base64Data: base64,
        });
      } catch (e: any) {
        console.warn("Tauri upload_dataset IPC failed, trying fallback:", e);
        throw new Error(typeof e === "string" ? e : e?.message || "Failed to upload file");
      }
    }
    const formData = new FormData();
    formData.append("file", file);
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
  tokenizeDataset: async (id: string): Promise<any> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("tokenize_dataset", { id });
      } catch (e: any) {
        console.warn("Tauri tokenize_dataset error:", e);
        throw new Error(typeof e === "string" ? e : e?.message || "Tokenization failed");
      }
    }
    return request<{
      success: boolean;
      dataset_id: string;
      cached: boolean;
      token_count: number;
      train_tokens: number;
      val_tokens: number;
    }>(`/api/datasets/${encodeURIComponent(id)}/tokenize`, {
      method: "POST",
    });
  },
  curateDataset: async (
    id: string,
    options?: { jaccard_threshold?: number; lid_threshold?: number; max_samples?: number }
  ): Promise<any> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("curate_dataset", { id, options: options || {} });
      } catch (e: any) {
        console.warn("Tauri curate_dataset error:", e);
        throw new Error(typeof e === "string" ? e : e?.message || "MinHash Deduplication failed");
      }
    }
    return request<any>(`/api/datasets/${encodeURIComponent(id)}/curate`, {
      method: "POST",
      body: JSON.stringify(options || {}),
    });
  },
  deleteDataset: async (id: string): Promise<{ success: boolean; id: string; deleted_files?: string[] }> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        const res: any = await inv("delete_dataset", { id });
        return { success: true, id: res.id || id, deleted_files: res.deleted_files };
      } catch (e) {
        console.warn("Tauri delete_dataset fallback:", e);
      }
    }
    return request<{ success: boolean; id: string; deleted_files?: string[] }>(
      `/api/datasets/${encodeURIComponent(id)}`,
      { method: "DELETE" }
    );
  },
  inspectTokenizer: async (text: string): Promise<any> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("inspect_tokenizer", { text });
      } catch (e) {
        console.warn("Tauri inspect_tokenizer fallback:", e);
      }
    }
    return request<any>("/api/tokenizer/inspect", {
      method: "POST",
      body: JSON.stringify({ text }),
    });
  },

  // Evaluation
  generateEvaluation: async (params: {
    prompt: string;
    max_new_tokens?: number;
    min_new_tokens?: number;
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
  }): Promise<any> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("generate_evaluation", { params });
      } catch (e) {
        console.warn("Tauri generate_evaluation fallback:", e);
      }
    }
    return request<any>("/api/evaluation/generate", {
      method: "POST",
      body: JSON.stringify(params),
    });
  },
  compareEvaluation: async (params: {
    prompt: string;
    checkpoint_paths: string[];
    max_new_tokens?: number;
    min_new_tokens?: number;
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
  }): Promise<any> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("compare_evaluation", { params });
      } catch (e) {
        console.warn("Tauri compare_evaluation fallback:", e);
      }
    }
    return request<{ prompt: string; results: any[] }>("/api/evaluation/compare", {
      method: "POST",
      body: JSON.stringify(params),
    });
  },
  runBenchmarkSuite: async (): Promise<any> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("run_benchmark_suite");
      } catch (e) {
        console.warn("Tauri run_benchmark_suite fallback:", e);
      }
    }
    return request<any>("/api/evaluation/benchmark", { method: "POST" });
  },
  listBenchmarkResults: async (limit: number = 100): Promise<BenchmarkResult[]> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("list_benchmarks", { limit });
      } catch (e) {
        console.warn("Tauri list_benchmarks fallback:", e);
      }
    }
    return request<BenchmarkResult[]>(`/api/evaluation/results?limit=${limit}`);
  },

  // Network Studio
  listNetworks: async (): Promise<NetworkConfig[]> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("list_networks");
      } catch (e) {
        console.warn("Tauri list_networks fallback:", e);
      }
    }
    return request<NetworkConfig[]>("/api/networks");
  },
  getNetworkTemplates: async (): Promise<NetworkConfig[]> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("get_network_templates");
      } catch (e) {
        console.warn("Tauri get_network_templates fallback:", e);
      }
    }
    return request<NetworkConfig[]>("/api/networks/templates");
  },
  getCustomModules: async (): Promise<any[]> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("get_custom_modules");
      } catch (e) {
        console.warn("Tauri get_custom_modules fallback:", e);
      }
    }
    return request<any[]>("/api/networks/custom_modules");
  },
  getNetwork: async (id: string): Promise<NetworkConfig> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("get_network", { id });
      } catch (e) {
        console.warn("Tauri get_network fallback:", e);
      }
    }
    return request<NetworkConfig>(`/api/networks/${encodeURIComponent(id)}`);
  },
  saveNetwork: async (network: Partial<NetworkConfig>): Promise<any> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        const name = network.name || network.id || "custom_network";
        return await inv("save_network", { name, config: network });
      } catch (e) {
        console.warn("Tauri save_network fallback:", e);
      }
    }
    return request<NetworkConfig>("/api/networks", {
      method: "POST",
      body: JSON.stringify(network),
    });
  },
  cloneNetwork: async (origId: string, newId: string, newName: string): Promise<NetworkConfig> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("clone_network", { origId, newId, newName });
      } catch (e) {
        console.warn("Tauri clone_network fallback:", e);
      }
    }
    return request<NetworkConfig>(`/api/networks/${encodeURIComponent(origId)}/clone`, {
      method: "POST",
      body: JSON.stringify({ new_id: newId, new_name: newName }),
    });
  },
  deleteNetwork: async (id: string): Promise<{ success: boolean; id: string }> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        await inv("delete_network", { id });
        return { success: true, id };
      } catch (e) {
        console.warn("Tauri delete_network fallback:", e);
      }
    }
    return request<{ success: boolean; id: string }>(`/api/networks/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
  },
  validateNetwork: async (config: any): Promise<any> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("validate_network", { config });
      } catch (e) {
        console.warn("Tauri validate_network fallback:", e);
      }
    }
    return request<{
      valid: boolean;
      issues: string[];
      warnings: string[];
      details?: Record<string, any>;
    }>("/api/networks/validate", {
      method: "POST",
      body: JSON.stringify(config),
    });
  },
  estimateNetwork: async (params: {
    config: any;
    micro_batch_size?: number;
    seq_len?: number;
    dtype?: string;
    optimizer?: string;
    gradient_checkpointing?: boolean;
  }): Promise<any> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        const est: any = await inv("estimate_network", { config: params.config, params });
        if (est && est.parameters) return est;
        if (est) {
          return {
            parameters: {
              total_params: est.total_params || est.parameter_count,
              trainable_params: est.trainable_params || est.parameter_count,
              non_trainable_params: 0,
              breakdown: est.parameter_breakdown,
            },
            memory: est.memory,
          };
        }
      } catch (e) {
        console.warn("Tauri estimate_network fallback:", e);
      }
    }
    return request<{
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
    });
  },
  recommendBudget: async (targetParams: number): Promise<any> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("recommend_budget", { targetParams });
      } catch (e) {
        console.warn("Tauri recommend_budget fallback:", e);
      }
    }
    return request<any>("/api/networks/budget_recommend", {
      method: "POST",
      body: JSON.stringify({ target_params: targetParams }),
    });
  },

  // Settings
  getSettings: async (): Promise<Record<string, any>> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("get_settings");
      } catch (e) {
        // A key/decryption failure must not silently fall back to another
        // backend that might show an empty credential.
        if (/encryption key|cannot decrypt|stored secret/i.test(String(e))) throw e;
        console.warn("Tauri get_settings fallback:", e);
      }
    }
    return request<Record<string, any>>("/api/settings");
  },
  saveSettings: async (settings: Record<string, any>): Promise<{ status: string; settings: any }> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        await inv("save_settings", { settings });
        return { status: "ok", settings };
      } catch (e) {
        if (/encryption key|cannot encrypt|stored secret/i.test(String(e))) throw e;
        console.warn("Tauri save_settings fallback:", e);
      }
    }
    return request<{ status: string; settings: any }>("/api/settings", {
      method: "POST",
      body: JSON.stringify(settings),
    });
  },

  testNotification: async (data: Record<string, any>): Promise<{ discord: boolean; telegram: boolean; errors: string[] }> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("test_notification", { data });
      } catch (e) {
        console.warn("Tauri test_notification fallback:", e);
      }
    }
    return request<{ discord: boolean; telegram: boolean; errors: string[] }>("/api/notifications/test", {
      method: "POST",
      body: JSON.stringify(data),
    });
  },

  // Foundation Models & Compute Topology
  getFoundationPresets: async (): Promise<any[]> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("get_foundation_presets");
      } catch (e) {
        console.warn("Tauri get_foundation_presets fallback:", e);
      }
    }
    return request<any[]>("/api/models/foundation_presets");
  },
  importFoundationModel: async (
    hfRepoId: string,
    modelName?: string,
    defaultTask?: string,
    allowRemoteCode = false,
  ): Promise<any> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("import_foundation_model", { hfRepoId, modelName, defaultTask });
      } catch (e) {
        console.warn("Tauri import_foundation_model fallback:", e);
      }
    }
    return request<any>("/api/models/import_hf", {
      method: "POST",
      body: JSON.stringify({
        hf_repo_id: hfRepoId,
        model_name: modelName,
        default_task: defaultTask,
        allow_remote_code: allowRemoteCode,
      }),
    });
  },
  getComputeTopology: async (): Promise<any> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("get_compute_topology");
      } catch (e) {
        console.warn("Tauri get_compute_topology fallback:", e);
      }
    }
    return request<any>("/api/compute/topology");
  },
  exportOllama: async (modelId: string, options?: any): Promise<any> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("export_ollama", { modelId, options });
      } catch (e) {
        console.warn("Tauri export_ollama fallback:", e);
      }
    }
    return request<any>(`/api/models/${encodeURIComponent(modelId)}/export_ollama`, {
      method: "POST",
      body: JSON.stringify(options || {}),
    });
  },

  // Workspaces
  listWorkspaces: async (): Promise<any[]> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("list_workspaces");
      } catch (e) {
        console.warn("Tauri list_workspaces fallback:", e);
      }
    }
    return request<any[]>("/api/workspaces");
  },
  getActiveWorkspace: async (): Promise<any> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("get_active_workspace");
      } catch (e) {
        console.warn("Tauri get_active_workspace fallback:", e);
      }
    }
    return request<any>("/api/workspaces/current");
  },
  setActiveWorkspace: async (workspaceId: string): Promise<{ status: string; workspace: any }> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        await inv("set_active_workspace", { workspaceId });
        return { status: "ok", workspace: { id: workspaceId } };
      } catch (e) {
        console.warn("Tauri set_active_workspace fallback:", e);
      }
    }
    return request<{ status: string; workspace: any }>("/api/workspaces/current", {
      method: "POST",
      body: JSON.stringify({ workspace_id: workspaceId }),
    });
  },
  createWorkspace: async (name: string, description?: string): Promise<any> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        return await inv("create_workspace", { name, description });
      } catch (e: any) {
        console.warn("Tauri create_workspace IPC failed:", e);
        throw new Error(typeof e === "string" ? e : e?.message || "Failed to create workspace");
      }
    }
    return request<any>("/api/workspaces", {
      method: "POST",
      body: JSON.stringify({ name, description }),
    });
  },
  exportWorkspaceArchive: async (id: string): Promise<Blob> => {
    const response = await fetch(
      `${BASE_URL}/api/workspaces/${encodeURIComponent(id)}/export`,
      { method: "POST" },
    );
    if (!response.ok) {
      let message = `Export failed: ${response.status}`;
      try {
        const body = await response.json();
        if (body?.error) message = body.error;
      } catch {
        // binary error body
      }
      throw new Error(message);
    }
    return response.blob();
  },
  importWorkspaceArchive: async (file: File): Promise<any> => {
    const form = new FormData();
    form.append("file", file, file.name);
    const response = await fetch(`${BASE_URL}/api/workspaces/import`, {
      method: "POST",
      body: form,
    });
    let body: any = null;
    try {
      body = await response.json();
    } catch {
      // ignore
    }
    if (!response.ok) {
      throw new Error(body?.error || `Import failed: ${response.status}`);
    }
    return body;
  },
  deleteWorkspace: async (id: string): Promise<{ success: boolean }> => {
    const inv = await getTauriInvoke();
    if (inv) {
      try {
        await inv("delete_workspace", { id });
        return { success: true };
      } catch (e: any) {
        console.warn("Tauri delete_workspace fallback:", e);
        throw new Error(typeof e === "string" ? e : e?.message || "Failed to delete workspace");
      }
    }
    return request<{ success: boolean }>(`/api/workspaces/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
  },
};
