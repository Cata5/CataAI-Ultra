"use client";

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { api } from "@/services/api";

export type DatasetOperationKind = "tokenize" | "clean" | "quality" | "upload";
export type DatasetOperationStatus = "queued" | "running" | "completed" | "failed";

export interface DatasetOperation {
  jobId: string;
  kind: DatasetOperationKind;
  datasetId: string;
  datasetName: string;
  status: DatasetOperationStatus;
  progress: number | null;
  stage: string;
  message: string;
  current: number;
  total: number | null;
  startedAt: number;
  finishedAt?: number;
  result?: any;
  error?: string;
}

interface StartOperationArgs {
  kind: DatasetOperationKind;
  datasetId: string;
  datasetName: string;
  run: (
    jobId: string,
    update: (patch: Partial<DatasetOperation>) => void
  ) => Promise<any>;
}

interface DatasetOperationsContextValue {
  operations: DatasetOperation[];
  startOperation: (args: StartOperationArgs) => Promise<any | null>;
  dismissOperation: (jobId: string) => void;
  clearFinished: () => void;
  activeOperation: (kind: DatasetOperationKind, datasetId: string) => DatasetOperation | undefined;
  latestCompleted: (kind: DatasetOperationKind, datasetId: string) => DatasetOperation | undefined;
  latestQualityOperation: DatasetOperation | undefined;
}

const STORAGE_KEY = "llm-maker:dataset-operations:v1";
const MAX_OPERATIONS = 10;
const POLL_INTERVAL_MS = 800;

const DatasetOperationsContext = createContext<DatasetOperationsContextValue | null>(null);

function isActive(operation: DatasetOperation): boolean {
  return operation.status === "queued" || operation.status === "running";
}

function newJobId(kind: DatasetOperationKind): string {
  const suffix =
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
  return `${kind}-${suffix}`;
}

function readStoredOperations(): DatasetOperation[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    const now = Date.now();
    return parsed
      .filter((item): item is DatasetOperation =>
        Boolean(item && typeof item.jobId === "string" && typeof item.kind === "string")
      )
      .map((item) =>
        isActive(item) && item.kind === "upload"
          ? {
              ...item,
              status: "failed" as const,
              stage: "interrupted",
              message:
                "Upload was interrupted by a page reload. Check the dataset list before retrying.",
              error: "Operation interrupted",
              finishedAt: now,
            }
          : item
      )
      .slice(0, MAX_OPERATIONS);
  } catch {
    return [];
  }
}

export const DatasetOperationsProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [operations, setOperations] = useState<DatasetOperation[]>([]);
  const [hydrated, setHydrated] = useState(false);
  const timers = useRef<Map<string, ReturnType<typeof setInterval>>>(new Map());
  const operationsRef = useRef<DatasetOperation[]>([]);

  const stopPolling = useCallback((jobId: string) => {
    const timer = timers.current.get(jobId);
    if (timer) {
      clearInterval(timer);
      timers.current.delete(jobId);
    }
  }, []);

  const finishOperation = useCallback(
    (jobId: string, patch: Partial<DatasetOperation>) => {
      stopPolling(jobId);
      setOperations((prev) =>
        prev.map((operation) =>
          operation.jobId === jobId ? { ...operation, ...patch } : operation
        )
      );
    },
    [stopPolling]
  );

  const pollOperation = useCallback(
    async (jobId: string) => {
      const status = await api.getDatasetOperation(jobId).catch(() => null);
      if (!status || status.status === "unknown") {
        const current = operationsRef.current.find((operation) => operation.jobId === jobId);
        if (current && Date.now() - current.startedAt > 2 * 60 * 60 * 1000) {
          finishOperation(jobId, {
            status: "failed",
            stage: "interrupted",
            message: "The operation stopped reporting progress.",
            error: "Operation interrupted",
            finishedAt: Date.now(),
          });
        }
        return;
      }
      if (status.status === "completed") {
        const completed: Partial<DatasetOperation> = {
          status: "completed",
          progress: 100,
          stage: "completed",
          message: "Completed",
          result: status.result,
          finishedAt: Date.now(),
        };
        if (typeof status.current === "number") completed.current = status.current;
        if (typeof status.total === "number") completed.total = status.total;
        finishOperation(jobId, completed);
        return;
      }
      if (status.status === "failed") {
        finishOperation(jobId, {
          status: "failed",
          stage: "failed",
          message: status.error || status.message || "Operation failed",
          error: status.error || status.message || "Operation failed",
          finishedAt: Date.now(),
        });
        return;
      }
      const updatedAt = Date.parse(status.updated_at || "");
      if (
        status.status === "running" &&
        Number.isFinite(updatedAt) &&
        Date.now() - updatedAt > 2 * 60 * 60 * 1000
      ) {
        finishOperation(jobId, {
          status: "failed",
          stage: "interrupted",
          message: "The operation stopped reporting progress.",
          error: "Operation interrupted",
          finishedAt: Date.now(),
        });
        return;
      }
      setOperations((prev) =>
        prev.map((operation) => {
          if (operation.jobId !== jobId || !isActive(operation)) return operation;
          return {
            ...operation,
            status: "running",
            progress:
              typeof status.percent === "number" ? status.percent : operation.progress,
            stage: status.phase || operation.stage,
            message: status.message || operation.message,
            current: typeof status.current === "number" ? status.current : operation.current,
            total: typeof status.total === "number" ? status.total : operation.total,
          };
        })
      );
    },
    [finishOperation]
  );

  const ensurePolling = useCallback(
    (jobId: string) => {
      if (timers.current.has(jobId)) return;
      void pollOperation(jobId);
      timers.current.set(
        jobId,
        setInterval(() => {
          void pollOperation(jobId);
        }, POLL_INTERVAL_MS)
      );
    },
    [pollOperation]
  );

  const startOperation = useCallback(
    async ({ kind, datasetId, datasetName, run }: StartOperationArgs) => {
      const jobId = newJobId(kind);
      const operation: DatasetOperation = {
        jobId,
        kind,
        datasetId,
        datasetName,
        status: "running",
        progress: null,
        stage: "queued",
        message: "Starting…",
        current: 0,
        total: null,
        startedAt: Date.now(),
      };
      setOperations((prev) => [operation, ...prev].slice(0, MAX_OPERATIONS));
      ensurePolling(jobId);
      const updateOperation = (patch: Partial<DatasetOperation>) => {
        setOperations((prev) =>
          prev.map((item) =>
            item.jobId === jobId && isActive(item) ? { ...item, ...patch } : item
          )
        );
      };
      try {
        const result = await run(jobId, updateOperation);
        if (result && typeof result === "object" && result.error) {
          throw new Error(String(result.error));
        }
        finishOperation(jobId, {
          status: "completed",
          progress: 100,
          stage: "completed",
          message: "Completed",
          result,
          finishedAt: Date.now(),
        });
        return result;
      } catch (error: any) {
        const message = typeof error === "string" ? error : error?.message || "Operation failed";
        finishOperation(jobId, {
          status: "failed",
          stage: "failed",
          message,
          error: message,
          finishedAt: Date.now(),
        });
        return null;
      }
    },
    [ensurePolling, finishOperation]
  );

  useEffect(() => {
    setOperations(readStoredOperations());
    setHydrated(true);
  }, []);

  useEffect(() => {
    operationsRef.current = operations;
  }, [operations]);

  useEffect(() => {
    if (!hydrated) return;
    operations
      .filter(isActive)
      .forEach((operation) => ensurePolling(operation.jobId));
  }, [hydrated, operations, ensurePolling]);

  useEffect(() => {
    if (!hydrated) return;
    const timer = setTimeout(() => {
      try {
        window.sessionStorage.setItem(
          STORAGE_KEY,
          JSON.stringify(operations.slice(0, MAX_OPERATIONS))
        );
      } catch {
        // Session storage can be unavailable in private/incognito contexts.
      }
    }, 200);
    return () => clearTimeout(timer);
  }, [hydrated, operations]);

  useEffect(() => {
    const activeTimers = timers.current;
    return () => {
      activeTimers.forEach((timer) => clearInterval(timer));
      activeTimers.clear();
    };
  }, []);

  const dismissOperation = useCallback((jobId: string) => {
    stopPolling(jobId);
    setOperations((prev) => prev.filter((operation) => operation.jobId !== jobId));
  }, [stopPolling]);

  const clearFinished = useCallback(() => {
    setOperations((prev) => prev.filter(isActive));
  }, []);

  const activeOperation = useCallback(
    (kind: DatasetOperationKind, datasetId: string) =>
      operations.find(
        (operation) =>
          operation.kind === kind && operation.datasetId === datasetId && isActive(operation)
      ),
    [operations]
  );

  const latestCompleted = useCallback(
    (kind: DatasetOperationKind, datasetId: string) =>
      operations.find(
        (operation) =>
          operation.kind === kind &&
          operation.datasetId === datasetId &&
          operation.status === "completed"
      ),
    [operations]
  );

  const latestQualityOperation = useMemo(
    () => operations.find((operation) => operation.kind === "quality"),
    [operations]
  );

  const value = useMemo(
    () => ({
      operations,
      startOperation,
      dismissOperation,
      clearFinished,
      activeOperation,
      latestCompleted,
      latestQualityOperation,
    }),
    [
      operations,
      startOperation,
      dismissOperation,
      clearFinished,
      activeOperation,
      latestCompleted,
      latestQualityOperation,
    ]
  );

  return (
    <DatasetOperationsContext.Provider value={value}>
      {children}
    </DatasetOperationsContext.Provider>
  );
};

export function useDatasetOperations(): DatasetOperationsContextValue {
  const context = useContext(DatasetOperationsContext);
  if (!context) {
    throw new Error("useDatasetOperations must be used inside DatasetOperationsProvider");
  }
  return context;
}