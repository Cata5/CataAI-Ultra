"use client";

import { useEffect, useState, useCallback } from "react";
import { Alert, MetricPoint, RunState, WebSocketEvent } from "@/types/api";
import { wsService } from "@/services/websocket";
import { api } from "@/services/api";

export function useWebSocket() {
  const [isConnected, setIsConnected] = useState<boolean>(() => wsService.connected);
  const [currentState, setCurrentState] = useState<RunState>(() => wsService.getCurrentState());
  const [metricsHistory, setMetricsHistory] = useState<MetricPoint[]>(() => wsService.getMetricsHistory());
  const [latestMetric, setLatestMetric] = useState<MetricPoint | null>(() => wsService.getLatestMetric());
  const [valState, setValState] = useState(() => wsService.getValidationState());
  const [recentAlerts, setRecentAlerts] = useState<Alert[]>(() => wsService.getRecentAlerts());
  const [recentEvents, setRecentEvents] = useState<WebSocketEvent[]>(() => wsService.getRecentEvents());

  const loadRunHistory = useCallback(async (runId: string) => {
    if (!runId) return;
    try {
      const history = await api.getRunMetrics(runId, 0, 2000);
      if (Array.isArray(history) && history.length > 0) {
        wsService.setMetricsHistory(history);
        setMetricsHistory(history);
        setLatestMetric(history[history.length - 1]);
        setValState(wsService.getValidationState());
      }
    } catch {
      // silently ignore if run not yet written
    }
  }, []);

  useEffect(() => {
    wsService.connect();

    // Sync latest buffer state immediately on mount
    const initialHistory = wsService.getMetricsHistory();
    if (initialHistory.length > 0) {
      setMetricsHistory(initialHistory);
      setLatestMetric(wsService.getLatestMetric());
      setValState(wsService.getValidationState());
    }
    const initialAlerts = wsService.getRecentAlerts();
    if (initialAlerts.length > 0) {
      setRecentAlerts(initialAlerts);
    }
    const initialEvents = wsService.getRecentEvents();
    if (initialEvents.length > 0) {
      setRecentEvents(initialEvents);
    }

    const unsubStatus = wsService.onStatus((status) => {
      setIsConnected(status);
    });

    const unsubMetric = (metric: MetricPoint) => {
      setLatestMetric(metric);
      setMetricsHistory(wsService.getMetricsHistory());
      setValState(wsService.getValidationState());
    };
    const unsubMetricDispose = wsService.onMetric(unsubMetric);

    const unsubState = wsService.onState((state) => {
      setCurrentState(state);
    });

    const unsubAlert = (alert: Alert) => {
      setRecentAlerts(wsService.getRecentAlerts());
    };
    const unsubAlertDispose = wsService.onAlert(unsubAlert);

    const unsubEvent = (event: WebSocketEvent) => {
      setRecentEvents(wsService.getRecentEvents());
      if (event.event_type === "VALIDATION_RESULT" || event.event_type === "STATE_CHANGED") {
        setValState(wsService.getValidationState());
      }
    };
    const unsubEventDispose = wsService.onEvent(unsubEvent);

    return () => {
      unsubStatus();
      unsubMetricDispose();
      unsubState();
      unsubAlertDispose();
      unsubEventDispose();
    };
  }, []);

  return {
    isConnected,
    currentState,
    setCurrentState,
    latestMetric,
    latestValLoss: valState.val_loss,
    latestValStep: valState.step,
    latestValTimestamp: valState.timestamp,
    valStatus: valState.status,
    activeTrainingTime: wsService.getActiveTrainingTime(),
    metricsHistory,
    setMetricsHistory,
    recentAlerts,
    recentEvents,
    loadRunHistory,
  };
}
