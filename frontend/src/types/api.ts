/**
 * Type definitions for RoCanon-48.5M WebUI v2 Sci-Fi Workstation.
 * Matches backend contracts in src/dashboard/server.py and worker protocols.
 */

export type RunState =
  | "IDLE"
  | "STARTING"
  | "RUNNING"
  | "PAUSED"
  | "STOPPING"
  | "STOPPED"
  | "COMPLETED"
  | "FAILED";

export interface MetricPoint {
  step: number;
  loss: number;
  train_loss?: number;
  val_loss?: number | null;
  latest_val_loss?: number | null;
  latest_val_step?: number | null;
  latest_val_timestamp?: number | null;
  val_status?: "completed" | "awaiting" | "evaluating";
  eval_interval?: number;
  last_val_step?: number | null;
  next_val_step?: number | null;
  learning_rate: number;
  muon_lr?: number;
  adamw_lr?: number;
  base_muon_lr?: number;
  base_adamw_lr?: number;
  sched_phase?: "warmup" | "stable" | "decay";
  sched_factor?: number;
  lr_multiplier?: number;
  adaptive_lr_factor?: number;
  tokens_seen?: number;
  tokens?: number;
  tokens_per_second?: number;
  tokens_per_sec?: number;
  tokens_in_epoch?: number;
  tokens_per_epoch?: number;
  current_epoch?: number;
  target_epochs?: number | null;
  duration_mode?: "steps" | "epochs";
  epoch_pct?: number;
  step_time_ms?: number;
  active_training_time_s?: number;
  grad_norm?: number | null;
  memory_allocated_mb?: number | null;
  vram_used_gb?: number;
  phase?: string | null;
  timestamp?: number;
  created_at?: string;
  [key: string]: any;
}

export interface SystemStatus {
  state: RunState;
  run_id: string | null;
  active_network?: string;
  active_workspace?: string;
  latest_metrics: MetricPoint | null;
  alerts_count: number;
  timestamp: number;
}

export interface HardwareTelemetry {
  gpu_name?: string;
  device_name?: string;
  gpu_count?: number;
  cuda_available: boolean;
  backend_type?: string;
  compute_capability?: string;
  system_ram_total_mb?: number;
  system_ram_free_mb?: number;
  supported_precisions?: string[];
  hardware_profile?: {
    mode: string;
    device_type: string;
    total_memory_mb: number;
    total_memory_gb: number;
    max_memory_allocated_mb: number;
    max_memory_allocated_gb: number;
    usable_ratio: number;
    warning_threshold_mb: number;
    critical_threshold_mb: number;
    emergency_threshold_mb: number;
    recommended_micro_batch: number;
    enable_grad_checkpointing: boolean;
    supported_precisions: string[];
    active_precision: string;
  };
  resource_profiles?: Record<string, any>;
  gpu_utilization_pct?: number;
  gpu_util_percent?: number;
  memory_used_mb?: number;
  used_vram_mb?: number;
  memory_total_mb?: number;
  total_vram_mb?: number;
  memory_free_mb?: number;
  free_vram_mb?: number;
  memory_utilization_pct?: number;
  vram_percent?: number;
  temperature_c: number;
  power_draw_w?: number;
  power_draw_watts?: number;
  power_limit_w?: number;
  power_limit_watts?: number;
  core_clock_mhz?: number;
  sm_clock_mhz?: number;
  memory_clock_mhz?: number;
  mem_clock_mhz?: number;
  system_ram_used_mb?: number;
  system_ram_used_gb?: number;
  system_ram_total_gb?: number;
  pytorch_allocated_mb?: number;
  pytorch_reserved_mb?: number;
  throttle_reasons?: string[];
  timestamp: number;
  is_available?: boolean;
}

export interface WorkspaceItem {
  id: string;
  name: string;
  description: string;
  root_path: string;
  is_active: boolean;
  config: Record<string, any>;
  run_count?: number;
  checkpoint_count?: number;
  dataset_count?: number;
  created_at: number;
  updated_at: number;
}

export interface PluginItem {
  id: string;
  name: string;
  version: string;
  description: string;
  category: string;
  author: string;
  enabled: boolean;
  is_valid: boolean;
  validation_error?: string | null;
  supported_features?: string[];
  config_schema?: Record<string, any>;
}

export interface Alert {
  id: number;
  run_id?: string;
  step?: number;
  level: "INFO" | "WARNING" | "CRITICAL" | "ERROR";
  category: string;
  message: string;
  created_at?: string;
  timestamp?: number;
}

export interface RunItem {
  id: string;
  name: string;
  status: RunState;
  config: Record<string, any>;
  started_at: string;
  completed_at?: string | null;
  total_steps: number;
  total_tokens: number;
  best_val_loss?: number | null;
  final_loss?: number | null;
}

export interface CheckpointItem {
  id: string;
  filename: string;
  run_id?: string;
  step: number;
  val_loss?: number | null;
  size_mb: number;
  created_at?: string;
  path: string;
  is_best?: boolean;
  is_saved_model?: boolean;
  model_name?: string;
}

export interface DatasetItem {
  id: string;
  name: string;
  source: string;
  status: string;
  size_mb: number;
  total_tokens?: number;
  train_tokens?: number;
  val_tokens?: number;
  stats?: Record<string, any>;
  created_at?: string;
  is_uploaded?: boolean;
}

export interface DatasetOperationStatus {
  job_id: string;
  kind: "tokenize" | "clean" | "quality" | string;
  dataset_id?: string | null;
  dataset_name?: string | null;
  status: "queued" | "running" | "completed" | "failed" | "unknown" | string;
  phase?: string;
  message?: string;
  current?: number;
  total?: number | null;
  percent?: number | null;
  result?: any;
  error?: string;
  updated_at?: string;
}

export interface RuntimeStatus {
  ready: boolean;
  python: string;
  manifest: string | null;
  data_dir: string;
  message: string;
}

export interface SFTFormatResult {
  success: boolean;
  input_path: string;
  output_path: string;
  format: string;
  valid_records: number;
  invalid_records: number;
  total_records: number;
  token_count?: number | null;
  sha256: string;
  errors: { line: number; error: string }[];
  dataset?: any;
}

export interface NetworkConfig {
  id: string;
  name: string;
  description?: string;
  task_type?: string;
  architecture_type?: string;
  layers?: any[];
  vocab_size?: number;
  d_model?: number;
  n_layers?: number;
  n_q?: number;
  n_kv?: number;
  d_ff?: number;
  kernel_size?: number;
  total_params?: number;
  total_parameters?: number;
  parameters?: {
    total_params: number;
    trainable_params: number;
    non_trainable_params?: number;
    breakdown?: Record<string, number>;
  };
  memory?: {
    model_weights_mb: number;
    optimizer_states_mb: number;
    gradients_mb: number;
    activations_mb: number;
    total_training_vram_mb: number;
    fits_6gb_gpu: boolean;
  };
}

export interface PipelinePhase {
  name: string;
  steps: number;
  lr: number;
  dataset: string;
  task: string;
}

export interface PipelineItem {
  id: string;
  name: string;
  status: string;
  current_phase_index: number;
  phases: PipelinePhase[];
}

export interface BenchmarkResult {
  id: number;
  run_id?: string;
  checkpoint_id?: string;
  task_name: string;
  score: number;
  metric_name: string;
  details?: Record<string, any>;
  created_at: string | number;
}

export interface SystemLogEntry {
  id: number;
  timestamp: string;
  level: "ERROR" | "WARN" | "INFO" | "DEBUG";
  source: "WORKER" | "TAURI" | "SYSTEM" | "DATABASE";
  message: string;
  raw: string;
}

export interface WebSocketEvent {
  event_type:
    | "STEP_METRICS"
    | "STATE_CHANGED"
    | "VALIDATION_RESULT"
    | "CHECKPOINT_SAVED"
    | "ALERT"
    | "LOG"
    | "TELEMETRY";
  payload: any;
  timestamp?: number;
}
