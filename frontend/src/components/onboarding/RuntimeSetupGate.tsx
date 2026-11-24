"use client";

import React, { useEffect, useRef, useState } from "react";
import { api, isTauri } from "@/services/api";
import { Cpu, Download, AlertTriangle, RefreshCw, X } from "lucide-react";

const CHANNELS = [
  { id: "auto", label: "Auto-detect (recommended)" },
  { id: "cu128", label: "CUDA 12.8" },
  { id: "cu126", label: "CUDA 12.6" },
  { id: "cu124", label: "CUDA 12.4" },
  { id: "cu121", label: "CUDA 12.1" },
  { id: "cu118", label: "CUDA 11.8" },
  { id: "cpu", label: "CPU only" },
];

export const RuntimeSetupGate: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [checking, setChecking] = useState(true);
  const [ready, setReady] = useState(false);
  const [status, setStatus] = useState<any>(null);
  const [channel, setChannel] = useState("auto");
  const [installing, setInstalling] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  const check = async () => {
    if (!isTauri()) {
      setReady(true);
      setChecking(false);
      return;
    }
    try {
      const res = await api.getRuntimeStatus();
      setStatus(res);
      setReady(Boolean(res.ready));
    } catch {
      // A missing command means an older/dev shell; do not lock the user out.
      setReady(true);
    } finally {
      setChecking(false);
    }
  };

  useEffect(() => {
    check();
  }, []);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [logs]);

  useEffect(() => {
    if (!isTauri() || !installing) return;
    let unlistenLog: (() => void) | undefined;
    let unlistenDone: (() => void) | undefined;
    (async () => {
      const { listen } = await import("@tauri-apps/api/event");
      unlistenLog = await listen<any>("setup-log", (event) => {
        const line = String(event.payload?.line || "");
        if (line) setLogs((prev) => [...prev.slice(-399), line]);
      });
      unlistenDone = await listen<any>("setup-done", async (event) => {
        setInstalling(false);
        const code = Number(event.payload?.code ?? -1);
        if (code === 0) {
          try {
            const verification = await api.verifyRuntime();
            setLogs((prev) => [...prev, verification]);
            await check();
          } catch (verifyError: any) {
            setError(String(verifyError?.message || verifyError));
          }
        } else {
          setError(`Runtime setup exited with code ${code}. See the log above.`);
        }
      });
    })();
    return () => {
      unlistenLog?.();
      unlistenDone?.();
    };
  }, [installing]);

  const start = async () => {
    setError(null);
    setLogs([]);
    setInstalling(true);
    try {
      await api.runSetup({
        cpuOnly: channel === "cpu",
        cuda: channel !== "cpu" && channel !== "auto" ? channel : undefined,
      });
    } catch (startError: any) {
      setInstalling(false);
      setError(String(startError?.message || startError));
    }
  };

  if (checking) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-[var(--bg-base)] text-[var(--text-primary)] font-mono text-xs">
        <RefreshCw className="w-4 h-4 animate-spin mr-2" /> Checking Python runtime…
      </div>
    );
  }

  if (ready) return <>{children}</>;

  return (
    <div className="h-screen w-screen flex items-center justify-center bg-[var(--bg-base)] p-4">
      <div className="w-full max-w-2xl bg-[#202020] border border-[#333333] rounded-lg shadow-2xl overflow-hidden">
        <div className="px-5 py-3 border-b border-[#262626] bg-[#1B1B1B] flex items-center gap-2">
          <Cpu className="w-4 h-4 text-[#A3A3A3]" />
          <div>
            <h1 className="text-xs font-mono font-semibold text-[#F2F2F2] uppercase tracking-wide">First-run runtime setup</h1>
            <p className="text-[10px] font-mono text-[#737373]">A private Python environment is installed under your user data folder.</p>
          </div>
        </div>
        <div className="p-5 space-y-4">
          {!installing && (
            <>
              <div className="p-3 rounded bg-[#1B1B1B] border border-[#262626] text-[11px] font-mono text-[#A3A3A3] space-y-1">
                <div>Runtime root: <span className="text-[#F2F2F2]">{status?.data_dir || "—"}</span></div>
                <div>Python: <span className="text-[#F2F2F2] break-all">{status?.python || "not installed"}</span></div>
                <div className="text-[#737373]">The installer downloads PyTorch for the selected channel. Expect several GB.</div>
              </div>
              <label className="block text-[10px] font-mono text-[#737373]">
                ACCELERATOR CHANNEL
                <select
                  value={channel}
                  onChange={(e) => setChannel(e.target.value)}
                  className="mt-1 w-full bg-[#1B1B1B] border border-[#333333] rounded px-3 py-2 text-xs font-mono text-[#F2F2F2] outline-none focus:border-[#737373]"
                >
                  {CHANNELS.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
                </select>
              </label>
              {error && (
                <div className="p-2 rounded bg-rose-950/40 border border-rose-500/40 text-rose-300 text-[11px] font-mono flex gap-2">
                  <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" /> {error}
                </div>
              )}
              <button onClick={start} className="btn btn-primary btn-sm w-full flex items-center justify-center gap-2">
                <Download className="w-3.5 h-3.5" /> Install Python runtime
              </button>
            </>
          )}

          {installing && (
            <>
              <div className="flex items-center justify-between text-[11px] font-mono text-[#A3A3A3]">
                <span className="flex items-center gap-2"><RefreshCw className="w-3.5 h-3.5 animate-spin" /> Installing PyTorch and dependencies…</span>
                <button
                  onClick={async () => { setInstalling(false); await api.cancelSetup(); }}
                  className="btn btn-secondary btn-sm flex items-center gap-1"
                >
                  <X className="w-3 h-3" /> Cancel
                </button>
              </div>
              <div ref={logRef} className="h-64 overflow-y-auto bg-[#111111] border border-[#333333] rounded p-2 text-[10px] font-mono text-[#A3A3A3] whitespace-pre-wrap">
                {logs.length ? logs.join("\n") : "Starting installer…"}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
