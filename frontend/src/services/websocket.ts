import { Alert, MetricPoint, RunState, WebSocketEvent } from "@/types/api";

type Listener<T> = (data: T) => void;

class WebSocketClient {
  private socket: WebSocket | null = null;
  private reconnectTimeout: any = null;
  private reconnectDelay = 1000;
  private maxReconnectDelay = 10000;
  private isExplicitlyClosed = false;
  private isConnected = false;

  private metricsBuffer: MetricPoint[] = [];
  private metricsRunId: string | null = null;
  private maxBufferSize = 20000;
  private alertsBuffer: Alert[] = [];
  private eventsBuffer: WebSocketEvent[] = [];
  private currentRunState: RunState = "IDLE";
  private latestValLoss: number | null = null;
  private latestValStep: number | null = null;
  private latestValTimestamp: number | null = null;
  private valStatus: "awaiting" | "completed" | "evaluating" = "awaiting";
  private cumulativeActiveTime: number = 0;
  private latestMetric: MetricPoint | null = null;

  private statusListeners: Set<Listener<boolean>> = new Set();
  private eventListeners: Set<Listener<WebSocketEvent>> = new Set();
  private metricListeners: Set<Listener<MetricPoint>> = new Set();
  private stateListeners: Set<Listener<RunState>> = new Set();
  private alertListeners: Set<Listener<Alert>> = new Set();

  public get connected(): boolean {
    return this.isConnected;
  }

  public getMetricsHistory(): MetricPoint[] {
    return [...this.metricsBuffer];
  }

  public getMetricsRunId(): string | null {
    return this.metricsRunId;
  }

  public getLatestMetric(): MetricPoint | null {
    if (this.latestMetric) return this.latestMetric;
    return this.metricsBuffer.length > 0 ? this.metricsBuffer[this.metricsBuffer.length - 1] : null;
  }

  public getValidationState(): {
    val_loss: number | null;
    step: number | null;
    timestamp: number | null;
    status: "awaiting" | "completed" | "evaluating";
  } {
    return {
      val_loss: this.latestValLoss,
      step: this.latestValStep,
      timestamp: this.latestValTimestamp,
      status: this.valStatus,
    };
  }

  public getActiveTrainingTime(): number {
    return this.cumulativeActiveTime;
  }

  public setMetricsHistory(history: MetricPoint[], runId?: string): void {
    if (runId) {
      this.metricsRunId = runId;
    }
    this.metricsBuffer = history.slice(-this.maxBufferSize);
    const latest = this.metricsBuffer.length > 0
      ? this.metricsBuffer[this.metricsBuffer.length - 1]
      : null;
    this.latestMetric = latest;
    this.latestValLoss = null;
    this.latestValStep = null;
    this.latestValTimestamp = null;
    this.valStatus = "awaiting";
    this.cumulativeActiveTime = 0;
    if (latest) {
      // Scan backwards for last non-null val_loss
      for (let i = this.metricsBuffer.length - 1; i >= 0; i--) {
        const pt = this.metricsBuffer[i];
        if (pt.val_loss !== undefined && pt.val_loss !== null && !isNaN(pt.val_loss)) {
          this.latestValLoss = Number(pt.val_loss);
          this.latestValStep = pt.step;
          this.latestValTimestamp = pt.timestamp || null;
          this.valStatus = "completed";
          break;
        }
      }
      if (latest.active_training_time_s !== undefined) {
        this.cumulativeActiveTime = Number(latest.active_training_time_s);
      }
    }
  }

  public clearMetricsHistory(runId: string | null = null): void {
    this.metricsBuffer = [];
    this.metricsRunId = runId;
    this.latestMetric = null;
    this.latestValLoss = null;
    this.latestValStep = null;
    this.latestValTimestamp = null;
    this.valStatus = "awaiting";
    this.cumulativeActiveTime = 0;
  }

  public setValidationState(valLoss: number | null, step: number | null, timestamp?: number | null): void {
    if (valLoss !== null && !isNaN(valLoss)) {
      this.latestValLoss = valLoss;
      this.latestValStep = step;
      this.latestValTimestamp = timestamp ?? Date.now();
      this.valStatus = "completed";
    }
  }

  public getRecentAlerts(): Alert[] {
    return [...this.alertsBuffer];
  }

  public getRecentEvents(): WebSocketEvent[] {
    return [...this.eventsBuffer];
  }

  public getCurrentState(): RunState {
    return this.currentRunState;
  }

  public clearTelemetry(): void {
    this.metricsBuffer = [];
    this.metricsRunId = null;
    this.alertsBuffer = [];
    this.eventsBuffer = [];
    this.latestValLoss = null;
    this.latestValStep = null;
    this.latestValTimestamp = null;
    this.valStatus = "awaiting";
    this.cumulativeActiveTime = 0;
    this.latestMetric = null;
  }

  public connect(urlOverride?: string): void {
    if (typeof window === "undefined") return;

    // In Tauri desktop environment, listen to native Tauri event bus
    if (typeof (window as any).__TAURI_INTERNALS__ !== "undefined" || typeof (window as any).__TAURI__ !== "undefined") {
      this.isConnected = true;
      this.notifyStatus(true);
      import("@tauri-apps/api/event").then(({ listen }) => {
        listen<any>("tauri://training-metrics", (event) => {
          this.handleIncoming(event.payload);
        }).catch(() => {});

        listen<any>("tauri://run-stopped", () => {
          this.handleIncoming({
            event_type: "STATE_CHANGED",
            payload: { state: "IDLE" },
            state: "IDLE",
          });
        }).catch(() => {});
      }).catch(() => {});
      return;
    }

    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this.isExplicitlyClosed = false;

    // Default to port 8081 if port 8080 is Next.js, or use NEXT_PUBLIC_WS_URL
    let wsUrl = process.env.NEXT_PUBLIC_WS_URL;
    if (!wsUrl) {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const host = window.location.hostname;
      // The Python backend runs on port 8081
      wsUrl = `${protocol}//${host}:8081/ws`;
    }

    if (urlOverride) {
      wsUrl = urlOverride;
    }

    try {
      this.socket = new WebSocket(wsUrl);

      this.socket.onopen = () => {
        this.isConnected = true;
        this.reconnectDelay = 1000;
        this.notifyStatus(true);
      };

      this.socket.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data);
          this.handleIncoming(parsed);
        } catch {
          // non-JSON message
        }
      };

      this.socket.onerror = () => {
        // WebSocket error
      };

      this.socket.onclose = () => {
        this.isConnected = false;
        this.notifyStatus(false);
        if (!this.isExplicitlyClosed) {
          this.scheduleReconnect();
        }
      };
    } catch {
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimeout) clearTimeout(this.reconnectTimeout);
    this.reconnectTimeout = setTimeout(() => {
      this.reconnectDelay = Math.min(this.reconnectDelay * 1.5, this.maxReconnectDelay);
      this.connect();
    }, this.reconnectDelay);
  }

  public disconnect(): void {
    this.isExplicitlyClosed = true;
    if (this.reconnectTimeout) clearTimeout(this.reconnectTimeout);
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
    this.isConnected = false;
    this.notifyStatus(false);
  }

  private handleIncoming(data: any): void {
    // Normalization for WorkerEvent vs raw dicts
    const eventType = data.event_type || data.type;
    const payload = data.payload || data;

    const wsEvent: WebSocketEvent = {
      event_type: eventType,
      payload: payload,
      timestamp: data.timestamp || Date.now(),
    };

    // Keep events in singleton buffer
    this.eventsBuffer.unshift(wsEvent);
    if (this.eventsBuffer.length > 50) this.eventsBuffer.pop();

    // Notify generic event listeners
    this.eventListeners.forEach((fn) => fn(wsEvent));

    const isStepEvent =
      eventType === "STEP_COMPLETED" ||
      eventType === "STEP_METRICS" ||
      eventType === "METRICS_UPDATED" ||
      data.step !== undefined ||
      payload.step !== undefined;

    if (eventType === "VALIDATION_COMPLETED") {
      if (payload.val_loss !== undefined && payload.val_loss !== null && !isNaN(payload.val_loss)) {
        this.latestValLoss = Number(payload.val_loss);
        this.latestValStep = Number(payload.step ?? payload.best_step);
        this.latestValTimestamp = payload.timestamp || Date.now();
        this.valStatus = "completed";
      }
    }

    if (isStepEvent) {
      const eventRunId = data.run_id ? String(data.run_id) : null;
      if (eventRunId && this.metricsRunId && eventRunId !== this.metricsRunId) {
        // A new/resumed run must not inherit the previous run's chart points.
        this.clearMetricsHistory(eventRunId);
      } else if (eventRunId && !this.metricsRunId) {
        this.metricsRunId = eventRunId;
      }
      const rawStep = Number(payload.step ?? data.step ?? 0);
      const rawLoss = Number(payload.loss ?? payload.train_loss ?? data.loss ?? data.train_loss ?? 0);
      const rawValLoss =
        payload.val_loss !== undefined && payload.val_loss !== null
          ? Number(payload.val_loss)
          : data.val_loss !== undefined && data.val_loss !== null
          ? Number(data.val_loss)
          : null;

      if (rawValLoss !== null && !isNaN(rawValLoss)) {
        this.latestValLoss = rawValLoss;
        this.latestValStep = rawStep;
        this.latestValTimestamp = payload.timestamp || Date.now();
        this.valStatus = "completed";
      } else if (payload.latest_val_loss !== undefined && payload.latest_val_loss !== null && !isNaN(payload.latest_val_loss)) {
        this.latestValLoss = Number(payload.latest_val_loss);
        this.latestValStep = payload.latest_val_step ?? this.latestValStep;
        this.latestValTimestamp = payload.latest_val_timestamp ?? this.latestValTimestamp;
        this.valStatus = "completed";
      }

      if (payload.active_training_time_s !== undefined && !isNaN(payload.active_training_time_s)) {
        this.cumulativeActiveTime = Number(payload.active_training_time_s);
      }

      const rawLr = Number(
        payload.learning_rate ??
        payload.lr ??
        payload.muon_lr ??
        payload.adamw_lr ??
        data.learning_rate ??
        data.lr ??
        0
      );
      const rawTokens = Number(payload.tokens_seen ?? payload.tokens ?? data.tokens_seen ?? data.tokens ?? 0);
      const rawTps = Number(
        payload.tokens_per_second ??
        payload.tok_per_sec ??
        data.tokens_per_second ??
        data.tok_per_sec ??
        0
      );
      const rawStepTime = Number(
        payload.step_time_ms ??
        (payload.dt !== undefined ? payload.dt * 1000 : data.step_time_ms ?? 0)
      );
      const rawGradNorm =
        payload.grad_norm !== undefined && payload.grad_norm !== null
          ? Number(payload.grad_norm)
          : data.grad_norm !== undefined && data.grad_norm !== null
          ? Number(data.grad_norm)
          : null;
      const rawMem = Number(
        payload.memory_allocated_mb ??
        (payload.vram_used_gb !== undefined ? payload.vram_used_gb * 1024 : data.memory_allocated_mb ?? 0)
      );
      const rawPhase = String(payload.phase ?? data.phase ?? "pretrain");

      const metric: MetricPoint = {
        step: rawStep,
        loss: rawLoss,
        train_loss: rawLoss,
        val_loss: rawValLoss,
        latest_val_loss: this.latestValLoss,
        latest_val_step: this.latestValStep,
        latest_val_timestamp: this.latestValTimestamp,
        val_status: this.valStatus,
        learning_rate: rawLr,
        muon_lr: payload.muon_lr !== undefined ? Number(payload.muon_lr) : rawLr,
        adamw_lr: payload.adamw_lr !== undefined ? Number(payload.adamw_lr) : undefined,
        base_muon_lr: payload.base_muon_lr !== undefined ? Number(payload.base_muon_lr) : undefined,
        base_adamw_lr: payload.base_adamw_lr !== undefined ? Number(payload.base_adamw_lr) : undefined,
        sched_phase: payload.sched_phase,
        sched_factor: payload.sched_factor !== undefined ? Number(payload.sched_factor) : undefined,
        lr_multiplier: payload.lr_multiplier !== undefined ? Number(payload.lr_multiplier) : undefined,
        adaptive_lr_factor: payload.adaptive_lr_factor !== undefined ? Number(payload.adaptive_lr_factor) : undefined,
        tokens_seen: rawTokens,
        tokens_per_second: rawTps,
        tokens_in_epoch: payload.tokens_in_epoch !== undefined ? Number(payload.tokens_in_epoch) : undefined,
        tokens_per_epoch: payload.tokens_per_epoch !== undefined ? Number(payload.tokens_per_epoch) : undefined,
        current_epoch: payload.current_epoch !== undefined ? Number(payload.current_epoch) : (payload.epoch !== undefined ? Number(payload.epoch) : undefined),
        target_epochs: payload.target_epochs !== undefined ? Number(payload.target_epochs) : undefined,
        duration_mode: payload.duration_mode,
        epoch_pct: payload.epoch_pct !== undefined ? Number(payload.epoch_pct) : undefined,
        step_time_ms: rawStepTime,
        active_training_time_s: this.cumulativeActiveTime,
        eval_interval: payload.eval_interval !== undefined ? Number(payload.eval_interval) : undefined,
        last_val_step: this.latestValStep,
        next_val_step: payload.next_val_step !== undefined ? Number(payload.next_val_step) : undefined,
        grad_norm: rawGradNorm !== null ? rawGradNorm : undefined,
        memory_allocated_mb: rawMem > 0 ? rawMem : undefined,
        vram_used_gb: payload.vram_used_gb !== undefined ? Number(payload.vram_used_gb) : undefined,
        phase: rawPhase,
        timestamp: payload.timestamp ?? data.timestamp ?? Date.now(),
      };
      this.latestMetric = metric;

      // Append to metricsBuffer (replacing same step if duplicates arrive)
      const last = this.metricsBuffer[this.metricsBuffer.length - 1];
      if (last && last.step === metric.step) {
        // If the new one has val_loss and the old one didn't, merge it
        if (metric.val_loss !== null && last.val_loss === null) {
          last.val_loss = metric.val_loss;
        }
        this.metricsBuffer[this.metricsBuffer.length - 1] = { ...last, ...metric };
      } else {
        this.metricsBuffer.push(metric);
        if (this.metricsBuffer.length > this.maxBufferSize) {
          this.metricsBuffer.shift();
        }
      }

      this.metricListeners.forEach((fn) => fn(metric));
    } else if (eventType === "STATE_CHANGED" || eventType === "RUN_STARTED" || eventType === "RUN_COMPLETED" || eventType === "RUN_STOPPED" || eventType === "RUN_FAILED") {
      const newState: RunState =
        eventType === "RUN_STARTED" ? "RUNNING" :
        eventType === "RUN_COMPLETED" ? "COMPLETED" :
        eventType === "RUN_STOPPED" ? "STOPPED" :
        eventType === "RUN_FAILED" ? "FAILED" :
        (payload.new_state || payload.state || "IDLE");
      this.currentRunState = newState;
      if (eventType === "RUN_STARTED") {
        // Keep history up to the resume checkpoint and drop any stale future
        // points from the previous branch; a fresh run starts at step 0.
        const startStep = Number(payload.start_step ?? data.start_step ?? 0);
        const eventRunId = data.run_id ? String(data.run_id) : undefined;
        this.setMetricsHistory(
          this.metricsBuffer.filter((point) => point.step <= startStep),
          eventRunId
        );
      }
      this.stateListeners.forEach((fn) => fn(newState));
    } else if (eventType === "ALERT" || eventType === "LOSS_SPIKE" || eventType === "GRADIENT_WARNING") {
      const alert: Alert = {
        id: payload.id || Date.now(),
        run_id: payload.run_id,
        step: payload.step,
        level: eventType === "LOSS_SPIKE" ? "WARNING" : eventType === "GRADIENT_WARNING" ? "WARNING" : payload.level || "INFO",
        category: eventType === "LOSS_SPIKE" ? "LOSS_SPIKE" : eventType === "GRADIENT_WARNING" ? "GRAD_WARNING" : payload.category || "GENERAL",
        message: payload.message || payload.alert || (payload.loss ? `Loss Spike: ${payload.loss}` : "Telemetry Alert"),
        created_at: payload.created_at || new Date().toISOString(),
      };
      this.alertsBuffer.unshift(alert);
      if (this.alertsBuffer.length > 50) this.alertsBuffer.pop();
      this.alertListeners.forEach((fn) => fn(alert));
    }
  }

  private notifyStatus(connected: boolean): void {
    this.statusListeners.forEach((fn) => fn(connected));
  }

  // Subscriptions
  public onStatus(listener: Listener<boolean>): () => void {
    this.statusListeners.add(listener);
    listener(this.isConnected);
    return () => this.statusListeners.delete(listener);
  }

  public onEvent(listener: Listener<WebSocketEvent>): () => void {
    this.eventListeners.add(listener);
    return () => this.eventListeners.delete(listener);
  }

  public onMetric(listener: Listener<MetricPoint>): () => void {
    this.metricListeners.add(listener);
    return () => this.metricListeners.delete(listener);
  }

  public onState(listener: Listener<RunState>): () => void {
    this.stateListeners.add(listener);
    return () => this.stateListeners.delete(listener);
  }

  public onAlert(listener: Listener<Alert>): () => void {
    this.alertListeners.add(listener);
    return () => this.alertListeners.delete(listener);
  }
}

export const wsService = new WebSocketClient();
