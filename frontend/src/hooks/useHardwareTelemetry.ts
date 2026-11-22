"use client";

import { useEffect, useState, useCallback } from "react";
import { HardwareTelemetry } from "@/types/api";
import { api } from "@/services/api";

export function useHardwareTelemetry(intervalMs: number = 2000) {
  const [telemetry, setTelemetry] = useState<HardwareTelemetry | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const fetchTelem = useCallback(async () => {
    try {
      const data = await api.getHardwareTelemetry();
      setTelemetry(data);
      setError(null);
      setIsLoading(false);
    } catch (err: any) {
      setError(err.message || "Failed to fetch hardware telemetry");
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchTelem();
    const interval = setInterval(fetchTelem, intervalMs);

    return () => {
      clearInterval(interval);
    };
  }, [fetchTelem, intervalMs]);

  return { telemetry, error, isLoading, refetch: fetchTelem, refresh: fetchTelem };
}
