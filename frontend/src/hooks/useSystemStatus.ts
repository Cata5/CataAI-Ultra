"use client";

import { useEffect, useState, useCallback } from "react";
import { SystemStatus } from "@/types/api";
import { api } from "@/services/api";

export function useSystemStatus(pollIntervalMs: number = 3000) {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const refresh = useCallback(async () => {
    try {
      const data = await api.getSystemStatus();
      setStatus(data);
      setError(null);
      setIsLoading(false);
    } catch (err: any) {
      setError(err.message || "Failed to fetch system status");
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    let mounted = true;
    const fetchStatus = async () => {
      try {
        const data = await api.getSystemStatus();
        if (mounted) {
          setStatus(data);
          setError(null);
          setIsLoading(false);
        }
      } catch (err: any) {
        if (mounted) {
          setError(err.message || "Failed to fetch system status");
          setIsLoading(false);
        }
      }
    };
    void fetchStatus();
    const interval = setInterval(() => {
      void fetchStatus();
    }, pollIntervalMs);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [pollIntervalMs]);

  return { status, error, isLoading, refresh };
}
