"use client";

import { useEffect, useState } from "react";
import { HardwareTelemetry } from "@/types/api";
import { api } from "@/services/api";

export function useHardwareTelemetry(intervalMs: number = 2000) {
  const [telemetry, setTelemetry] = useState<HardwareTelemetry | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    let mounted = true;

    const fetchTelem = async () => {
      try {
        const data = await api.getHardwareTelemetry();
        if (mounted) {
          setTelemetry(data);
          setError(null);
          setIsLoading(false);
        }
      } catch (err: any) {
        if (mounted) {
          setError(err.message || "Failed to fetch hardware telemetry");
          setIsLoading(false);
        }
      }
    };

    fetchTelem();
    const interval = setInterval(fetchTelem, intervalMs);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [intervalMs]);

  return { telemetry, error, isLoading };
}
